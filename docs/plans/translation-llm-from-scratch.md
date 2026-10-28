# 在双 H100 上从零训练翻译 LLM

**状态：计划，尚未执行。核查日期：2026-10-27。** 本文给出西班牙语与低资源语言翻译的随机初始化训练方案。本轮没有连接 GPU 服务器、下载数据、启动训练或测量吞吐。主线建议约 **2.92B 参数、60B processed tokens、2048 上下文**，预算备选为 **1.15B 参数、24B tokens**。保守按 **2×80GB 可用显存**规划，不假定 NVLink；历史型号记录不能替代开跑前的型号、互联、磁盘与独占状态检查。3B 主线的假设周期为约 **45–146 天**，而单次训练硬上限设为 **120 天**，因此必须先校准，必要时在长跑前缩小模型或 token 预算。全部配置与验收线都是项目建议，不是已验证结果或质量保证。

| 一页决策项 | 主线计划 | 降级或停止门槛 |
|---|---|---|
| 模型与初始化 | 2.92B，28×3072，32k自训词表，所有权重随机 | 校准超预算则1.15B/24B；不自动扩大到7B |
| 数据与方向 | EN/ES单语为主；en↔es、es↔gn、es↔许可通过的quy | 第二低资源语言少于5k独立句对或变体不清则暂停该语言 |
| 主训练量 | A/B/C=56B/3.5B/0.5B processed tokens | 60B或120天先到即停；pilot/基线/消融另计 |
| 两卡预算 | 假设100–300TF/s合计有效吞吐，约45–146天 | 100–500步实测预测≤120天才允许完整60B路线 |
| 工程验收 | 32对target-token CE≤0.1且贪心exact-match≥90%；恢复50步不跳步；峰值目标≤65GB/卡 | 非finite、泄漏、状态损坏立即暂停 |
| 质量目标，未实测 | 每个低资源方向test chrF++较同数据随机NMT至少+2.0；母语adequacy≥4/5的比例≥80%，严重语义错误≤5% | 数字是拟议项目目标；未达标交付失败结论，不承诺超过NLLB |
| 当前状态 | 方案与来源核查完成；吞吐、许可后量、质量均未知 | 本文命令为未实现模板，本轮不启动训练 |

## 1. 将“从零”落实为可审计的模型与预算

主路线复用成熟 Transformer 实现，但 tokenizer 只在本项目 train split 上训练，模型各层权重全部随机初始化。`LlamaForCausalLM(config)` 可以创建随机权重结构；`from_pretrained` 加载已有权重，两者不能混称。主路线禁止加载 NLLB、Qwen、mBART、Llama 等预训练权重，也不先用教师模型生成翻译。若以后加入合成数据，实验应改标“随机初始化 + 合成数据/蒸馏”，单独报告教师、生成量与许可。([Transformers 官方初始化说明](https://huggingface.co/docs/transformers/main/en/model_doc/llama))

研究问题是：在固定两卡与数据许可约束下，西语中心的单语预训练和真实平行监督，能否形成 en↔es、es↔Guaraní、es↔指定 Quechua 变体的翻译能力。English 是迁移锚点；Guaraní 是第一低资源语言；第二语言先审查 Ayacucho/Southern Quechua 的来源和变体。不能只增加语言标签便声称支持低资源语言，也不能用法语、德语、葡语替代真正低资源实验。

| 角色 | 建议结构 | 近似参数 | processed token 上限 | 用途与边界 |
|---|---|---:|---:|---|
| 流水线 pilot | 12 层，d=768，FFN=2048，12 Q/4 KV heads | 0.10B | 0.2B | 检查监督、分布式、生成与恢复，不能代表成品质量 |
| 预算备选 | 24 层，d=2048，FFN=5632，16 Q/4 KV heads | 1.15B | 24B | 在较短预算内完成一条完整随机初始化路线 |
| **主线** | **28 层，d=3072，FFN=8192，24 Q/8 KV heads** | **2.92B** | **60B** | 约 3B 的翻译语言模型，按里程碑决定继续 |
| 扩展比较 | 约 7B 稠密 decoder-only | 7B | 140B 量级 | 本次不默认启动，显存和周期均明显更紧 |

以上采用 32k 词表、RoPE、SwiGLU、RMSNorm、无 attention/MLP bias、输入输出 embedding tying。参数是按矩阵形状计算的近似值，最终必须输出去重后的实际 `numel`。3B 的估算为 28×(attention 25.17M + FFN 75.50M)+embedding 98.30M，再加 norm，约 2.917B。GQA 是工程选择，尚无本项目实验证明它是翻译质量最优设计。OLMo 公开配置提供了 2048 上下文、bf16、AdamW 等成熟参考，但不证明本方案的吞吐或最佳超参。([LlamaConfig](https://huggingface.co/docs/transformers/main/en/model_doc/llama)、[OLMo-1B 官方配置](https://github.com/allenai/OLMo/blob/main/configs/official-0724/OLMo-1B.yaml))

建议自训练 SentencePiece BPE：vocab=32,000、byte fallback、Unicode NFC，保留重音、鼻化符号、撇号及大小写。tokenizer 采样池按 es/en/gn/quy 的 40/30/15/15 作为目标比例，小语种不足时使用真实可用量，不能反复复制凑足字节。保留源/目标语言标签与分隔符，逐语言报告 tokens/word、bytes/token、byte-fallback 率、2048 截断率。pilot 可比较 BPE 与 unigram；正式训练后固定 token ID，不通过查看 test 重选词表。([SentencePiece 官方仓库](https://github.com/google/sentencepiece))

下面是**尚未运行的模型构造草图**，不是仓库现有训练入口；special token ID 必须从自训练 tokenizer 读取。

```python
import torch
from transformers import LlamaConfig, LlamaForCausalLM

torch.manual_seed(3407)
config = LlamaConfig(
    vocab_size=32000, hidden_size=3072, intermediate_size=8192,
    num_hidden_layers=28, num_attention_heads=24,
    num_key_value_heads=8, max_position_embeddings=2048,
    hidden_act="silu", rms_norm_eps=1e-6, rope_theta=10000.0,
    tie_word_embeddings=True, attention_bias=False, mlp_bias=False,
    attention_dropout=0.0, initializer_range=0.02, use_cache=False,
)
model = LlamaForCausalLM(config)  # 全部模型权重随机初始化
model.gradient_checkpointing_enable()
print(sum(p.numel() for p in model.parameters()))
```

实现时再配置与锁定版本兼容的 tokenizer 包装器、bf16 autocast 和 FP32 optimizer 状态。不能将全部 Adam 状态强转 bf16 来换取表面显存节省，也不能因构造成功就宣称分布式训练可用。

## 2. 用许可通过的数据组成真实低资源实验

数据表中的量是**拟定配额或上限，未下载、未清洗、未确认最终可用量**。每条记录保留 revision、source URL、license URL、取得日期、语言/方言、split、人工/挖掘/合成类别与校验值。数据库包装许可和原始文本许可需要分别记录；公开仓库优先发布索引、统计和处理脚本，不默认重发原始文本。

| 数据部分 | 首选来源 | 计划配额与重复约束 | 必须先解决的问题 |
|---|---|---|---|
| 西语单语 | FineWeb2 `spa_Latn`，可补指定 Wikipedia 快照 | 阶段 A 约 36B processed tokens，尽量一遍；另报 unique | 锁定 ODC-BY 数据库版本，保留原网页出处与处理记录 |
| 英语单语 | FineWeb 合适切片 | 阶段 A 约 18B，不能把 10B 切片说成 18B unique | 按可用量选切片，完成来源审查与去污染 |
| en–es 平行 | ParaCrawl，经许可审查的补充来源 | 阶段 A 约 2B；全流程真实句对目标 5M–20M | ParaCrawl 的 CC0 仅覆盖包装，不能替原网页重新授权 |
| Guaraní–es | Jojajovai official train，AmericasNLP train 待审扩展 | Jojajovai 原始 train 20,207 对；主模型全部阶段合计最多20次方向暴露；先10次检查 | 跨来源去重；MIT 仓库许可不能替代每个原来源权利核查 |
| Quechua–es | AmericasNLP 中已核实授权、已标变体的 train 子集 | 上游基础规模 125,008 对；本项目可用量可能为 0–125k | 检查 quy/quz/Southern 映射及 JW300、教材、词典来源许可 |
| 低资源单语 | 明确许可与变体的 Wikipedia/FineWeb2 子集 | 每语最多 3 遍；两语合计上限 0.2B，缺额由 es 回填 | 逐文档核查 LID，不将少量文本重复数百遍形成 B 级配额 |
| 自建补充 | 明确授权的社区生活、公共服务文本 | 每语先 1k–5k train，另建 300–500 句 test | 人员、授权与费用尚未落实，不能计为已有数据 |

FineWeb2 发布多语数据并包含西语，FineWeb 面向英语；其 ODC-BY 标签不能单独解决每个原网页的权利问题。ParaCrawl 明确不拥有原文本，CC0 指向平行数据包装。Wikipedia 应按具体快照和页面适用的 CC-BY-SA/GFDL 条款保留来源。([FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)、[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb/blob/main/README.md)、[ParaCrawl 许可](https://paracrawl.eu/)、[Wikimedia 内容复用说明](https://www.mediawiki.org/wiki/Wikimedia_APIs/Content_reuse))

Jojajovai 总规模 30,855 对，官方 train/dev/test 为 20,207/5,314/5,334；这些 split 必须保留。AmericasNLP 2024 的基础 Guaraní train 为 26,032 对、Quechua 为 125,008 对，额外数据和合成数据另列，因此不能把两个 Guaraní 来源简单相加。Quechua 数据包含变体和多种来源，下载得到不等于许可已经通过。([Jojajovai 官方仓库](https://github.com/pln-fing-udelar/jojajovai)、[仓库 LICENSE](https://raw.githubusercontent.com/pln-fing-udelar/jojajovai/master/LICENSE)、[AmericasNLP 2024 findings](https://aclanthology.org/2024.americasnlp-1.28.pdf)、[Quechua 来源说明](https://raw.githubusercontent.com/AmericasNLP/americasnlp2023/main/data/quechua-spanish/README.md))

**第二低资源语言的启动门槛：** 许可通过后至少 5k 高质量独立句对、有可靠变体标记、有独立 dev/test、有匹配变体的评审人员。未满足就保留 Guaraní 主实验，先补授权数据；Aymara 只能作为重新审查后的替代。第一版排除来源或许可尚不清晰的 Quechua 扩展，以及含教师生成、后编辑和多种来源混合的 IWSLT2026 文本，避免把它们称为统一授权的全人工训练集。([IWSLT2026 Quechua 数据说明](https://raw.githubusercontent.com/johneortega/IWSLT2026_Quechua_data/main/README.md))

按以下顺序完成数据工程，每一步都输出输入量、删除量、原因分布和剩余 unique 统计：

1. 先冻结全部 dev/test 的源句、目标句、文档与来源清单；保留官方 split，对新增数据按文章、域名、书籍分组后划分，再训练 tokenizer。
2. 规范化 NFC、空白和控制字符，保留正字法。en–es 可以用较严语言识别；gn/quy 每来源先抽 200 对做人工审查，不能套用高资源语言阈值自动大量删除。
3. 以 3–250 词、长度比 0.25–4 作为异常候选起点，检查错位、乱码、模板与重复短句。低资源形态复杂，阈值不是自动丢弃规则；抽审对齐完整率低于 90% 的来源暂停纳入。
4. 做句对、单边、反向句对的 exact hash，再做字符 5-gram 的近重复候选；Jaccard≥0.8 先人工校验误删率。同一源句跨语言转译、文章改版也要检查。
5. train 任一侧命中 benchmark 源句/目标句即删除；网页单语删去同源文章和高重叠片段。test 不参与分词、BT、过滤阈值、超参和早停选择。

DataTrove 提供文档去重工具；挖掘语料本身可能含 LID 错误、机器翻译和其他数据集测试样本。若使用 NLLB bitext，应当作为待审数据源，不能因名称相同而等同于加载 NLLB 权重，也不能视为无噪声金标准。([DataTrove](https://github.com/huggingface/datatrove)、[NLLB 数据卡](https://huggingface.co/datasets/allenai/nllb))

建议平行样本为 `BOS <src_es><tgt_gn> source <sep> target EOS`。普通语言建模对全部文本 token 监督；翻译阶段只对 target+EOS 算 loss，源句 label=-100。双向复用是两个方向视图，仍只有一条独立句对。pilot 先关闭跨例 packing；后续若启用，要测试 attention 边界和 label 边界。日志同时记录 processed、nonpadding、loss-bearing tokens，不能只报 target token 而漏掉源句的计算成本。

## 3. 先校准两卡，再进入有停止点的训练阶段

3B 先试 ZeRO-2，再与 FSDP FULL_SHARD 做同样本、同 global token batch 的短校准。DeepSpeed 官方的无 offload 估算为 ZeRO-2 约 `4N+16N/world_size` bytes，ZeRO-3 约 `18N/world_size+largest_layer_memory`；都未包含全部 activations 和临时 buffer。2.92B 两卡的 ZeRO-2 基础状态约 **35.0GB/卡**，ZeRO-3 约 **26.3GB/卡加最大层临时量**。7B 的 ZeRO-3 已约 63GB/卡加其他开销，80GB 余量紧张。([DeepSpeed 显存公式](https://deepspeed.readthedocs.io/en/stable/memory.html))

启用 bf16、activation checkpointing，microbatch 从每卡 2 条开始，峰值目标≤65GB/卡，留出保存和通信余量。PCIe/跨 NUMA 条件下，FULL_SHARD 的参数 all-gather 可能降低吞吐；FSDP 的 `no_sync` 积累行为也可能增加梯度显存，不能把不同框架的配置机械混用。主线不默认 CPU/NVMe offload。H100 存在不同显存/互联规格，厂商带稀疏性标注的峰值不能当作本项目 dense 有效吞吐。([PyTorch FSDP](https://docs.pytorch.org/docs/stable/fsdp.html)、[Accelerate FSDP](https://huggingface.co/docs/accelerate/main/en/usage_guides/fsdp)、[NVIDIA H100 规格](https://www.nvidia.com/en-us/data-center/h100/))

下面是**待实现配置模板**，不是已验证 YAML schema。若每卡 microbatch=2，则 accumulation=32，以保持同一 global batch。可变长和 target-only loss 还需按实际目标 token 归一化，避免不同 rank 长度不同造成权重偏差。

```yaml
initialization: random
seed: 3407
sequence_length: 2048
precision: bf16
optimizer: adamw
adam_betas: [0.9, 0.95]
adam_eps: 1.0e-8
weight_decay: 0.1
max_grad_norm: 1.0
world_size: 2
micro_batch_sequences_per_gpu: 4
gradient_accumulation_steps: 16
processed_tokens_per_update: 262144
activation_checkpointing: true
attention: sdpa
parallelism: zero2
cpu_offload: false
nvme_offload: false
```

AdamW 的 norm/bias 衰减排除规则、embedding 衰减策略和 FlashAttention/SDPA 选择要在 pilot 固定。`parallelism: zero2` 本身不会启动 DeepSpeed，工程实现必须生成匹配的 plugin/配置并核对实际启动日志；FSDP1/FSDP2 字段也要随锁定版本核实。([OLMo 配置参考](https://github.com/allenai/OLMo/blob/main/configs/official-0724/OLMo-1B.yaml)、[Accelerate FSDP 配置](https://huggingface.co/docs/accelerate/main/en/usage_guides/fsdp))

| 编号与阶段 | 预算 | 数据与 loss | 学习率建议 | 继续条件 |
|---|---:|---|---|---|
| 0. tiny overfit / pilot | 0.10B 模型；≤0.2B tokens；≤12小时 | 先 32 对，再覆盖全部已授权语言；验证目标 mask、EOS、方向与恢复 | peak 6e-4，100 更新 warmup，cosine 至 6e-5 | 32对target-token CE≤0.1且贪心整句exact-match≥90%；仅作工程门。输出错语种/监督错位即停 |
| 1. 两卡校准 | 100–500 更新；≤6小时 | ZeRO-2/FULL_SHARD 同 batch 对照；包含保存和恢复 | 固定短程策略，不拿 test 调参 | 稳态显存、吞吐、resume、预算全通过 |
| 2. A 基础预训练 | 56B；约213,624更新 | 36B es+18B en 单语+2B en↔es；低资源单语替换部分 es；全序列 LM loss | peak 3e-4；1000 更新 warmup；cosine 至3e-5 | 累计0.5B、2B、10B、30B审查后决定继续 |
| 3. B 翻译强化 | 3.5B；约13,352更新 | 高资源平行为主；低资源温度采样但重复封顶；target+EOS loss | peak5e-5；200更新warmup；cosine至5e-6 | 每500更新或低资源每1等效epoch评估，先到者执行 |
| 4. C 高质量收尾 | ≤0.5B；约1,908更新 | 最佳真实句对，方向均衡；低资源再最多5次暴露 | peak2e-5；50更新warmup；cosine至2e-6 | dev恶化即恢复best，不强行用满配额 |

更新数按 262,144 tokens 近似计算，最后一步按实际 `tokens_seen` 截止。A/B/C 合计 60B，不含 pilot、基线和消融。1.15B 备选分配为 22B/1.75B/0.25B，A 的 peak LR 从4e-4开始校准，其余门槛相同。阶段切换保留 optimizer moments，记录 scheduler 变更；不得无记录重置。主模型每条低资源句对在A/B/C合计最多20次方向暴露：同一句对正向、反向各用一次算2次；先累计10次检查，B最多用到15次，C最多用剩余5次，不新增第21–25次。pilot与独立基线分别记录自己的暴露量。

**校准记录与恢复验收。** 前30步预热不计稳态，31–100、101–200等至少50步窗口分别记录 nonpadding tokens/s、loss tokens/s、step p50/p90、卡间不均衡、max allocated/reserved、梯度范数、主机 RAM、I/O wait 与保存耗时。各框架先跑100–150步，再保存、退出、恢复50步，与不间断对照核对 batch ID、LR、token计数和loss。非确定性kernel不要求逐bit相同，但不能跳步、重复大段数据或重启LR。用最慢稳定窗口估算剩余周期，并加实测评测/保存耗时。([Accelerate checkpointing](https://huggingface.co/docs/accelerate/main/en/usage_guides/checkpoint))

checkpoint 应包含模型、optimizer、scheduler、Python/NumPy/torch CPU/CUDA RNG、sampler/shard/offset、packing残余、方向采样状态、tokens_seen、tokenizer/manifest/config/代码hash与版本锁；尽量在 optimizer step 边界存。每60分钟或500更新先到者保存，临时目录写完并校验后原子rename，保留最近2份、best1份和最近阶段边界1份，合计最多4份独立完整状态，同一步多种角色只存一次。单份3B全状态预估40–60GB，4份约160–240GB，另加写入中临时一份40–60GB；还要给数据、缓存留空间。60B uint16 token ID 仅数组就约120GB。开跑前建议规划500GB–1TB独立工作区、至少300GB checkpoint/临时余量，实际容量尚未检查。([PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html))

下列均为**尚未实现的接口模板**；`train_translation.py`、配置和数据脚本不是本文新增的可执行程序。正式实现并通过 schema、loss mask、采样、token计数、恢复检查后才具备运行条件，本轮没有执行这些命令。

```bash
TASK_ROOT=/path/to/future/translation-project

# 未来开跑前，在目标服务器现场检查
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
nvidia-smi topo -m
df -h "$TASK_ROOT"

# 待实现训练入口：先校准
accelerate launch --num_processes 2 --mixed_precision bf16 \
  "$TASK_ROOT/train_translation.py" \
  --config "$TASK_ROOT/configs/translation_2p92b.yaml" \
  --mode calibrate --max_steps 300

# 门槛通过后才允许阶段 A；示意参数不代表现有 CLI
accelerate launch --num_processes 2 --mixed_precision bf16 \
  "$TASK_ROOT/train_translation.py" \
  --config "$TASK_ROOT/configs/translation_2p92b.yaml" \
  --stage A --max_processed_tokens 56000000000
```

## 4. 以封存测试和母语评审判断翻译质量

计划基线包括 copy/空输出 sanity check、在同一授权平行集上从零训练的约60–150M encoder-decoder NMT、固定 NLLB-600M 的零额外训练结果，以及预算许可时的同数据 NLLB SFT。NMT基线与3B的参数量、单语预训练量不同，只作任务参照；不能凭两者差异给出架构或数据的因果结论。NLLB 是独立的预训练基线，模型卡为 CC-BY-NC-4.0；不能进入本主线初始化，也不能把输出当母语真值。基线只做离线研究，单独计时。([NLLB-600M 模型卡](https://huggingface.co/facebook/nllb-200-distilled-600M))

固定 FLORES-200 或 FLORES+ 的一个 revision，不能混合新旧版本分数。FLORES-200 包含 `spa_Latn`、`grn_Latn`、`quy_Latn`；FLORES+ 的发布说明将 dev 用于开发、devtest 用于测试，并明确不能用于训练。另保留 Jojajovai、AmericasNLP 和社区跨域测试，训练来源全部去污染。([FLORES-200](https://raw.githubusercontent.com/facebookresearch/flores/main/flores200/README.md)、[FLORES+ 数据卡](https://huggingface.co/datasets/openlanguagedata/flores_plus)、[AmericasNLP split规则](https://raw.githubusercontent.com/AmericasNLP/americasnlp2023/main/README.md))

每个方向报告 BLEU、chrF++、SacreBLEU signature、配对 bootstrap 区间，并按来源、变体、长度给切片。主自动指标用 macro chrF++，同时展示全部方向，防止 en↔es 掩盖低资源退化。COMET 只在 en↔es 作补充；其官方覆盖限制意味着不能将 Guaraní/Quechua 的得分作为可靠主排序。字符指标也无法充分识别否定反转、数字错误和幻觉。([SacreBLEU](https://github.com/mjpost/sacrebleu)、[COMET 语言覆盖说明](https://github.com/Unbabel/COMET))

若要声明四个低资源方向全部通过，每个方向固定至少300个未进训练或合成流程的新句并做盲评，合计至少1200句。资源不足可先评es→低资源，但人工质量结论仅覆盖已评方向，逆向明确标未验收。至少两名熟悉指定变体的双语评审独立盲评，模型顺序随机、同输入配对，分歧仲裁。记录 adequacy、fluency、漏译、增译、否定、姓名/数字、术语、错语言/方言与毒性。LLM回译评分只能补充，不能替代母语评审。300句的统计功效有限，须报告区间；没有评审资源时可以报告研究过程，不能宣布低资源翻译已可用。

以下是**预先拟定的项目阈值**，不代表普适质量标准或预期结果：

| 门槛 | 触发条件 | 动作 |
|---|---|---|
| 开跑前 | 许可/变体/切分/恢复/独占/容量任一不通过，或预计超过硬预算 | 先补证据或缩为1.15B，不启动长跑 |
| 立即暂停 | nonfinite loss/grad、test泄漏、损坏checkpoint、磁盘不足两份完整状态 | 保留现场，只诊断，不自动重试长跑 |
| 性能异常 | step time持续30分钟超过稳定基线2倍 | 检查IO/共享/通信，暂停计入预算 |
| 训练健康 | 0.5B、2B heldout loss未下降；或连续3次train降而dev显著升 | 检查监督与数据，停止该语种加采样 |
| 小语种过拟合 | 10次暴露后连续3次dev chrF++改善<0.5，或best回退>1.0且区间支持恶化 | 停止追加暴露并恢复best |
| 结论 | 只有高资源提升，或母语语义错误增加 | 报告迁移失败/数据不足，不以总体均值掩盖 |

正式质量目标为每个低资源方向较同数据随机NMT的test chrF++至少+2.0、母语adequacy≥4/5的比例≥80%、严重语义错误≤5%；这些数值须在看test前冻结，属于项目验收目标而非论文预测。

公平消融在同一中间checkpoint处分支，比较自然比例与 `p_i∝n_i^0.5`、target-only 与全序列loss，每次只改一个因素并保持额外tokens相同。每个0.2B分支另计算力；不默认追加BT、多seed或7B训练。“低资源改善”的结论需要清洁封存test优于公平对照，且母语盲评支持主要语义错误没有增加；不预先要求或承诺超过NLLB。CPU测试和tiny overfit只能证明局部工程路径，不能证明翻译质量。

## 5. 用硬预算决定 3B 主线是否启动

采用 dense Transformer 的量级近似 `C≈6NT`，其中T是前向处理token；attention、重计算、通信、I/O等还会影响真实周期。Chinchilla 支持在预算下联合考虑参数与token，不能推导本项目的低资源质量。下面假设**两卡合计有效模型计算吞吐100–300TF/s**，尚未测量，再留10–20%非训练余量。([Chinchilla 论文](https://arxiv.org/abs/2203.15556))

| 配置 | 6NT近似 | 纯计算天数 | 加非训练余量后的计划范围 |
|---|---:|---:|---:|
| 1.15B /24B tokens | 1.66e20 | 6.4–19.2 | 约7–23天，规划1–4周 |
| 2.92B /30B tokens | 5.26e20 | 20.3–60.8 | 约22–73天，明确标记探索/可能欠训练 |
| **2.92B /60B tokens** | **1.05e21** | **40.5–121.7** | **约45–146天** |
| 7B /140B tokens | 5.88e21 | 227–681 | 约250–817天，不列为默认 |

**单次主线最多60B tokens或120个wall-days，先到者停止；1.15B备选最多24B或30天。** 100–500步校准若按最慢稳定吞吐加实际保存/评估开销预测超120天，必须在正式启动前改为30B探索或1.15B/24B。pilot、校准、NMT/NLLB基线和消融独立登记，额外预算合计不超过主线GPU-hours的15%；到线便停止，不能将失败重跑排除在成本外。两卡120天为5760GPU-hours，若计费则用真实小时费率计算，不在此虚构租价。

本方案与ASR、TTS共用两卡，不能并行相加吞吐。建议先完成三条路线的许可审计与短pilot，再按ASR→TTS→翻译长跑串行；语音长跑的额外周期见[联合日程](./asr-tts-from-scratch.md#5-把语音与翻译放进同一份两卡总日程)。若选择3B翻译，三项完整实验应按数月规划；若总窗口只有数周，优先1.15B翻译加缩小语音路线，并明确范围。当前未确认数据、存储、硬件吞吐与人工评审资源，因而还不具备承诺最终日期或翻译分数的证据。

**下一份可执行工程交付**应包含版本锁、许可manifest、清洗与去污染统计、tokenizer审计、随机初始化参数记录、loss/packing测试、两卡校准报告、恢复验收、冻结的评测协议和模型卡空模板。只有这些门槛通过后，才讨论启动哪个预算；本文件本身没有授权或执行训练。

阅读版：[翻译方案 PDF](./pdf/translation-llm-from-scratch.pdf)。
