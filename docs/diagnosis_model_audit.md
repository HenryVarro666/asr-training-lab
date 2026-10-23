# 首轮 Qwen 退化：模型与训练链条只读审阅

审阅日期：2026-10-22。范围：固定代码、官方 SFT/模型实现、保存配置、实际 checkpoint、训练日志和已完成评测；本审阅没有加载 GPU 模型、启动训练或操作 PersonaPlex。新增执行仅为禁用 CUDA 的 processor 对照。结论是**尚未发现可确证的标签、保存/加载或解码实现错误；不能因此证明实现绝对无误，也不能将退化直接归因为学习率或过拟合。**

当前只推进一个验证：checkpoint-800 的同条件完整开发集生成评测。完成后再决策；本文件不是追加训练或自动执行后续全部实验的指令。

## 1. 已确认的结果与边界

源实验：`reports/qwen-campaign-20261022-a/`；远端：`/mnt/raid/asr-lab/runs/qwen-campaign-20261022-a/main-epoch1/`。代码 commit 为 `cf30f45e0beb005f43520c33e2ddad73ad67a01e`，记录 `git_dirty=false`。实际加载参数数目为 **2,038,052,480**；此前 2,349,217,408 是早期估计，不作为本次运行事实，也不臆测差异原因。

训练为 2 GPU、每卡 batch 8、累积 2、有效 batch 32、全参数 lr=1e-5、warmup=0.05、weight decay=0.01、clip=1、seed=42、1 epoch/892 步。FP32 参数/梯度/Adam 状态，BF16 autocast，SDPA。实际训练时间 1537.121 秒。

| 开发集 | 样本/参考词数 | 原始模型 WER | step892 WER | 增加百分点 | 错误数变化 |
|---|---:|---:|---:|---:|---:|
| dev-clean 合格样本 | 2694 / 54027 | 1.6122% | 2.3729% | +0.7607 | 871 → 1282 |
| dev-other 合格样本 | 2857 / 51053 | 3.0850% | 4.9850% | +1.9000 | 1575 → 2545 |

这里是**既定 ≤30 秒 manifest 的全部合格样本**，不是无条件宣称官方完整 benchmark：dev-clean 原有 2703 条中排除 9 条超长样本。两模型使用相同样本、batch16、normalizer、greedy 解码和 512 新 token 上限。

clean 的 S/D/I 从 643/157/71 变为 1038/109/135；other 从 1251/200/124 变为 2096/199/250。退化以替换错误为主，并有插入增加。`reports/diagnosis/text-audit.json` 对全部 5551 条重新归一化，与保存结果完全一致；未发现空结果、解析后前缀泄漏、所检查的重复 trigram 或截断标记。示例中的专名、近音词和词边界错误是观察，不能单独证明机制。

固定 128 的 teacher-forced eval loss 在 step100/200/300/400/500/600/700/800 为 0.080505/0.077961/0.074304/0.071282/0.064535/0.062818/0.062618/0.060103。缺少同条件 step0 loss 与早期生成 WER。**目前没有验证 loss 上升的证据**，不能描述成已证实的经典过拟合曲线。

远端实际仅保留 `checkpoint-800`、`checkpoint-892`。二者均含模型分片、processor/tokenizer、optimizer、scheduler、trainer state、两 rank RNG 和 checkpoint 元数据；本次没有重新逐字节加载所有训练状态。800 已在约 89.7% 的训练步数处，不能当作充分早期检查点。此前 step2→4 恢复测试中 707 个张量有限、最大绝对差 8.90e-6，是恢复路径证据，不是本轮精度或位级一致性保证。

## 2. 标签、掩码与 loss 审阅

审阅文件：`src/asr_lab/qwen_data.py`、`qwen_train.py`、`qwen_evaluate.py`；官方固定版本 `7c6daf77a2421100f5fb066495372c00129d39ff` 的 `finetuning/qwen3_asr_sft.py` 和 `qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`。

- prompt 是空 system、user audio、assistant generation prompt。训练串为 `prompt + language English<asr_text> + 原始转录 + <|im_end|>`，与官方 SFT 的前缀/目标/EOS 组织方式一致。
- 训练强制右 padding，逐样本验证 full input 开头等于 tokenized prefix，屏蔽 prompt/audio 和 attention-mask padding。语言标记、转录与 EOS 受到监督；EOS=151645，PAD=151643。没有按 token ID 误删 EOS。
- labels 保持与 input 对齐，由 thinker 的标准 causal loss 内部 shift；本地没有再手动 shift。outer forward 显式传递 labels，并禁用训练 cache。
- 官方原 collator 的 prefix-length masking 与默认左 padding 存在风险；本项目的显式右 padding 和 token 前缀断言避免该问题。不是因偏离官方左 padding 而认定本实现错误。
- `model_accepts_loss_kwargs=False` 与固定 transformers4.57.6 的累积逻辑配合，未发现重复除以 accumulation 的证据。当前目标是“每个 microbatch 有效 token CE 的均值，再对 microbatch/rank 求平均”，不是所有 rank token 的一次全局加权均值；run metadata 已披露。不同长度 microbatch 会获得不同的每-token 权重，这是目标选择与效率比较的限制，尚无因果证据表明它造成此次退化。
- teacher-forced CE 同时奖励语言标记、大小写、标点/无标点风格和参考词的 token 概率；它不等价于自由生成、归一化后的 WER。验证 loss 降低而 WER 升高并不矛盾。

## 3. 保存、加载与 generation config

基础模型固定 revision：`7278e1e70fe206f11671096ffdd38061171dd6e5`。评测双方都由官方 wrapper 加载，并显式 `dtype=bfloat16`、`attn_implementation=sdpa`、`eval()`、左 padding；生成显式 `do_sample=False,num_beams=1,use_cache=True,max_new_tokens=512`。切除的长度是完整 padded input width，随后官方 parser 固定 English；reference 没有加入 decoder prefix。

保存配置有可解释但应记录的变化：dtype 从 BF16 描述变成 FP32；模型类型被当前类序列化为 `qwen3_asr_thinker` / `qwen3_asr_text`；新增 `use_cache=false`、user token 等默认字段。评测显式 BF16 和 use_cache=true；这些差异本身不证明权重加载等价，尚未做无更新权重 roundtrip。

`sanitize_generation_configs` 重建配置，保留 BOS/EOS/PAD/decoder-start ID，去除 greedy 模式下不合法的 temperature 设置。实际加载配置比较如下：

| 字段 | 基础模型 | checkpoint-892 | 对当前调用的解释 |
|---|---|---|---|
| do_sample | false | false | 相同且调用显式指定 |
| num_beams | 1 | 1 | 相同且调用显式指定 |
| temperature | 1e-6 | 1.0 | greedy 不使用温度采样 |
| eos_token_id | [151643,151645] | [151645,151643,151645] | 同一停止集合；官方 outer generate 还显式传入 EOS 集合 |
| pad_token_id | 151643 | 151643 | 相同 |
| repetition_penalty / length_penalty | 1.0 / 1.0 | 1.0 / 1.0 | 相同 |
| forced_bos_token_id / forced_eos_token_id | null / null | null / null | 相同 |
| suppress_tokens / begin_suppress_tokens | null / null | null / null | 相同 |

基础 `chat_template.json` 内的模板字符串与 checkpoint 的 `chat_template.jinja` 相等；vocab/merges 字节相同。checkpoint tokenizer 增加 `fix_mistral_regex=true`，但 wrapper 原本也显式传入该值。checkpoint preprocessor 增加 `sampling_rate=16000`。以下 CPU 实测进一步检查其实际行为，而非只比较 JSON。

### CPU processor 对照：已执行

环境禁止 CUDA、禁止网络模型下载、禁止 Python bytecode 写入；只加载 processor，无模型权重。`torch.cuda.device_count()` 返回 **0**。在固定 128 manifest 中按 duration 排序取最短与最长，构成不等长 batch：

| utterance ID | 秒数 | 有监督 token 数 | 最末 label |
|---|---:|---:|---:|
| 2035-147960-0015 | 1.56 | 11 | 151645 |
| 8842-304647-0006 | 22.96 | 105 | 151645 |

短句 label 解码为 `language English<asr_text>OTTO WINKED AT ME<|im_end|>`。两个 processor 的 prompt 相同，以下全部逐项 `torch.equal=True`：

| 模式 | tensor | shape |
|---|---|---|
| train，右 padding | input_ids / attention_mask / labels | [2,418] |
| train | feature_attention_mask | [2,2296] |
| train | input_features | [2,128,2296] |
| train | _audio_seconds / _examples | [] / [] |
| eval，左 padding、English prefix | input_ids / attention_mask | [2,316] |
| eval | feature_attention_mask | [2,2296] |
| eval | input_features | [2,128,2296] |

这支持**这两个输入上的预处理和标签等价**，不外推到所有文本、不代替权重零更新保存/加载控制。

复现命令（只输出结果；不保存模型、不运行 GPU）：

```bash
ssh -o ConnectTimeout=10 -o ControlPath=/tmp/codex-h100-train-%C exx@100.70.230.73 \
 'cd /mnt/raid/asr-lab/repo && CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 /mnt/raid/asr-lab/venv/bin/python -' <<'PY'
import json, torch, qwen_asr
from transformers import AutoProcessor, GenerationConfig
from asr_lab.qwen_data import QwenCollator, prompt_text, load_audio
base='/mnt/raid/asr-lab/cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5'
saved='/mnt/raid/asr-lab/runs/qwen-campaign-20261022-a/main-epoch1/checkpoint-892'
rows=[json.loads(x) for x in open('/mnt/raid/asr-lab/data/manifests/dev-clean-fixed128.jsonl')]
rows=sorted(rows,key=lambda r:r['duration']); rows=[rows[0],rows[-1]]
a=AutoProcessor.from_pretrained(base,local_files_only=True,fix_mistral_regex=True)
b=AutoProcessor.from_pretrained(saved,local_files_only=True,fix_mistral_regex=True)
ca=QwenCollator(a)(rows); cb=QwenCollator(b)(rows)
print('CUDA devices',torch.cuda.device_count(),'IDs',[r['id'] for r in rows])
print('prompt_equal',prompt_text(a)==prompt_text(b))
print('train',{k:(list(ca[k].shape),torch.equal(ca[k],cb[k])) for k in ca})
for i in range(2):
    target=ca['labels'][i]; target=target[target!=-100]
    print('target',rows[i]['id'],target.numel(),int(target[-1]),a.tokenizer.decode(target))
for p in (a,b): p.tokenizer.padding_side='left'
kw=dict(audio=[load_audio(r) for r in rows],return_tensors='pt',padding=True,padding_side='left',truncation=False)
ea=a(text=[prompt_text(a,True)]*2,**kw); eb=b(text=[prompt_text(b,True)]*2,**kw)
print('eval',{k:(list(ea[k].shape),torch.equal(ea[k],eb[k])) for k in ea})
fields=['do_sample','temperature','num_beams','eos_token_id','pad_token_id','repetition_penalty','length_penalty','forced_bos_token_id','forced_eos_token_id','suppress_tokens','begin_suppress_tokens']
for path in (base,saved):
    g=GenerationConfig.from_pretrained(path,local_files_only=True)
    print(path,{k:getattr(g,k) for k in fields})
PY
```

## 4. 假设及证据强弱

| 假设 | 支持观察 | 尚缺证据 / 可证伪检查 |
|---|---|---|
| 全参数适配了转录风格，并伴随词汇能力漂移 | baseline 无全大写句；final clean 2688/2694、other 2853/2857 全大写。训练目标直接使用原始大写转录 | normalizer 已消除大小写；风格变化不是退化的充分原因。将来需要只改变 target canonicalization 的受控实验，不能用当前结果断言 |
| 最后阶段训练进一步伤害 WER | 最终 WER 退化，固定128 CE却下降 | 先比较 step800；若800≈892且均差，最后92步不是主要来源。800更好也不证明最优早停时刻 |
| 全参数更新幅度/学习率不合适 | 强基础模型在窄域100h训练后词替换错误增加 | 未做匹配更新数、数据顺序、globalbatch的LR对照；不能据此立即选5e-6或宣布LoRA必胜 |
| 保存或 greedy 配置改变导致退化 | 配置 JSON 有序列化差异 | 静态有效配置与CPU输入等价降低此怀疑；零更新原始权重 save/reload 的生成对照尚未做 |
| 计分、prefix或EOS错误 | 无正面支持 | 全量normalizer审计、无截断/前缀泄漏、两样本EOS/掩码对照均通过；仍不是穷尽所有边界测试 |

## 5. 唯一下一项实验与后续决策

**优先 A：checkpoint-800，0 次新训练更新，GPU1 单卡，顺序生成 dev-clean/dev-other 合格全集。** 复用与 base/892 相同的 manifest、source hash、batch16、固定 normalizer、SDPA/BF16、English prefix、greedy 和512上限。预计约8分钟仅是既往评测耗时估计；使用独立有界 runner/监控，记录实际耗时。不得占用或停止 GPU0 的 PersonaPlex，不并发开启另一实验，不修改现有 checkpoint。

问题预先固定为“最后92步是否贡献了观察到的主要退化”，不是从许多检查点中挑最小 WER。保存逐句预测与指标，比较800对base、800对892的 S/D/I、总错误、speaker paired-bootstrap 差值区间；两个dev分别报告，不以未预设加权平均掩盖other退化。bootstrap使用相同speaker重采样索引成对比较，每次重算总错误/总参考词；将其视为诊断性不确定性，不因查看多个比较就声称确认性显著。保留test-clean/test-other不参与选择。

- 若800与892相近且都差：大部分损失已经发生在前800步；不应继续892，也不能假称早停已解决。再决定是否做零更新控制或设计单因素训练试验。
- 若800明显优于892：支持最后92步有额外损害；仍要看800是否优于base及other保护线。不自动部署800，也不据此外推更早的最佳步数。
- 若800结果与格式/输入证据矛盾，或出现新异常：先做 B，暂停训练方案选择。

**B（暂不执行）**：原始权重经同一FP32加载、训练patch、sanitize、token ID 对齐、保存和重新加载，但optimizer更新数为0；以固定128、batch16、同BF16评测与原始直接加载对照，保存原始token IDs及首次分歧。用训练路径实际的FP32保存约8.15GB权重，先检查空间。这个对照比仅改generation config更强；若不一致，再逐项隔离dtype/token配置/加载逻辑。若完全一致，只能在受测128上排除这类路径差异。

**C（不优先）**：只对base应用sanitize后生成。当前greedy有效配置已比较等价，单独C的信息增益低；仅在B出现差异且怀疑配置分支时使用。

本阶段不启动 lr5e-6、LoRA、目标大小写转换或更大数据重训。A的结果与本审阅共同构成下一次规划依据；没有“必须微调胜过基础模型”的预设。

## 6. 远端来源指纹

下列 SHA256 为本审阅直接读取远端文件所得；训练源文件与保存的 `run.json` 指纹一致。

| 来源 | SHA256 |
|---|---|
| dev-clean-fixed128.jsonl | `1c657340c27cdc1f4b025ba5731da07fc7fd2f505e9dddbfc8fffc9cc59df999` |
| src/asr_lab/qwen_data.py | `769aed8478da4a6e9dcd33ed3bf87aaa6b846eede3d677c9ae4dc88469dc89b0` |
| src/asr_lab/qwen_train.py | `97951994468a14b10ce57b7c5c864dcef6ffb6af4ee9c75e621b48c01e9de961` |
| src/asr_lab/qwen_evaluate.py | `0a2012eb016cddadee7b3c0f0ca5ad9f3f94b502faad780392788a8ccf9e30d6` |
| vendor 官方 finetuning/qwen3_asr_sft.py | `6f3e5cc530d9da9ff405144fc94a5c4790018eaf956dbc623634818160418976` |
| vendor 官方 modeling_qwen3_asr.py | `2fb5d98da1933748f5117ee05ce4e7150c9ead8154fb8e25f7af3968b853adc7` |
| base config.json | `2e74a751548b8ad7d7526d29365ad8144c345d8b412b1152d25dc6698452712f` |
| base generation_config.json | `1da527824d81e07118facff437e03f2e24a23311e3bdeb2368973fe77e5f275c` |
| base tokenizer_config.json | `4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c` |
| base preprocessor_config.json | `45e120a4eda2c20c5d7f2ea9354e63536bf35e27aa573fb7cdf78017b378770d` |
| base chat_template.json | `75a8cfca24f00de72d796fbfed6858fc9614ef3dabd8696684cc3bc03a9c58ff` |
| checkpoint-892 config.json | `90907b1e2b2d95ee310a38daefda306f95dfcf2df3c797e252afdc161127eb6a` |
| checkpoint-892 generation_config.json | `a275709a685826eaf1c763af663dc9ee3653c9353251f6eb3f859c3526652204` |
| checkpoint-892 tokenizer_config.json | `7f335cbd3b6f1036808e23613c1c1919f8d5570c11cb55420e5094e036f45669` |
| checkpoint-892 preprocessor_config.json | `0443a40d411d3caad769e7b59971fa6eae75926ed4b7e337e60da909eb38c1f7` |
| checkpoint-892 chat_template.jinja | `a103579066642f0e035873f12de6046a1bbc909a7a2833732fda9fd6bc7c528d` |

`base`、`checkpoint-892` 完整根路径见复现命令；vendor 根为 `/mnt/raid/asr-lab/vendor/Qwen3-ASR`。JSON包裹模板与jinja文件的文件hash不同，不表示模板字符串不同。
