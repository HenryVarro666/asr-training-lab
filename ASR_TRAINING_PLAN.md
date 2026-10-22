# 双 H100 ASR 训练与 Speech-LLM 演进计划

版本：2026-10-22，Astra 6 策略审阅第二版。已纳入用户新增“现代架构、充分利用两张 H100、比较数据/模型/张量并行”的要求。实施与真实训练由 GPT-6.1 Sol 接续。

本文件是待执行和持续验收的方案，不代表模型已经训练完成、取得指标或上线。运行状态、实际命令及指标分别以仓库 README、运行目录和报告为准。原始输入为用户提供的 `deepinfer_speech_llm_asr_plan.md`；外部事实经过下述官方来源核查。

## 1. 现在要做的决定

**主实验升级为已预训练的 `Qwen/Qwen3-ASR-1.7B` 原生语音条件 LLM，全参数微调 LibriSpeech `train-clean-100`，首选两卡 DDP。** 官方现有多 GPU 微调脚本与模型权重均已核实，具体版本、必需补丁和效率实验见第11节。相比自研 Whisper→Qwen projector，这条路线直接保留已训练好的音频—语言对齐，可把精力用于数据与训练质量。[Qwen 官方微调目录](https://github.com/QwenLM/Qwen3-ASR/tree/7c6daf77a2421100f5fb066495372c00129d39ff/finetuning)

**原架构 `openai/whisper-small` 保留为工程基线和性能对照**，已有的数据准备、数据隔离、monitor 与评测代码继续复用。Whisper-small 是 244M 参数的 encoder-decoder ASR 模型；它不是响应“当前先进 Speech-LLM 主架构”的最终选择。[官方模型卡](https://huggingface.co/openai/whisper-small)

第一轮范围是完整训练划分的一次遍历，默认 1 epoch；不是只在几十条样本上跑几步。`whisper-tiny` 只用于必要的流程 smoke test。两个模型都保留各自预训练结构，不换 RoPE、不改因果卷积、不添加未对齐的新模块，不引入 DeepInfer 运行依赖。已有 Whisper 对照先完成必要的短验证与基线；不要因旧计划默认一整轮而阻塞当前主实验的 Qwen 路线。

Whisper 产出准确称为 **预训练 ASR 模型的领域微调及可复现训练基线**。Qwen 产出称为 **预训练语音条件 LLM 的 ASR SFT**；它也不是从零训练或通用语音对话系统。官方模型支持离线/流式推理不等于本次离线 SFT 已验证流式能力，二者必须分别交付结果。

## 2. 已核实边界与资源

### 2.1 DeepInfer 的角色

官方仓库当前存在。本次核查固定到 `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`，GitHub API 返回提交时间 `2026-10-21T04:55:43Z`。

| 核查项 | 证据与本项目决定 |
|---|---|
| 框架定位 | README 明确为大规模推理框架；不作为本项目训练器。[固定 README](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/README.md) |
| 官方支持硬件 | README 列出 NVIDIA SM100 和 Huawei Ascend 950；H100 不是其当前文档支持目标。此结论是支持矩阵结论，不声称我们已经在 H100 上复现编译失败。 |
| 模型注册 | 当前注册表只有 `DeepseekV41ForCausalLM`；没有注册本项目的 Whisper 或 Qwen 语音模型。[注册表](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/__init__.py) |
| 依赖 | `torch==2.13.0+cu132`、`flash-attn-4==4.0.0b34`。独立于本项目训练虚拟环境。[依赖文件](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/requirements-nvidia.txt) |
| 音频支持 | 完整 Git tree 中未发现路径名包含 audio/speech/asr 的文件，且没有找到官方音频模型入口。这支持“当前不把它作为音频训练/部署依赖”，不等同于已证明全仓库所有代码均无音频相关字符串。 |
| GPU 数量 | 部署 Q&A 区分可部署的 1P+1D 与推荐的高效 64P+64D；不可把后者误写成一切场景的绝对最小 GPU 数。[部署 Q&A](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md) |

仅借鉴可解释的模块边界、独立编码器 benchmark 和回归思路。复制类名、特殊 token 偏移或 projector 形状不能证明兼容；兼容性需要真实 adapter、权重转换、数值校验与目标后端端到端运行。第一阶段无需为这些接口付出成本。

### 2.2 真实主机约束

以下是主代理在本次会话中的只读 SSH 审计结果，启动前和扩容前需再次采样：

| 项目 | 实测快照与操作约束 |
|---|---|
| GPU | 2 × NVIDIA H100 NVL，各 `95,830 MiB`，不是预设的 80GB 型号 |
| 驱动 | `580.82`，`nvidia-smi` 显示 CUDA 13 能力；该值不是训练环境实际 PyTorch CUDA runtime 版本 |
| 互连 | 实测拓扑为 `NODE`，未显示 NVLink 路径；型号含 NVL 不等于本机已建立 NVLink，DDP 通信吞吐需实测 |
| 现有进程 | GPU0 已有 Python 进程占约 22.7GB；保留。一次 0% GPU 利用率不等于空闲或可终止 |
| 初次磁盘审计 | `/data` 约 610GiB 可用；该路径已被用户后续指定的 `/mnt/raid` 替代，仅保留审计历史 |
| 当前存储位置 | 用户指定 `/mnt/raid`；本次复核 `/dev/md127` 约44T总量、39T已用、3.3T可用、93%使用率。统一项目根为 `/mnt/raid/asr-lab` |
| 根分区 | `/` 仅约 572MiB 空闲；本项目 venv、tmp、HF/pip 缓存、TensorBoard 和全部产物均放 `/mnt/raid/asr-lab`，禁止依赖根分区默认缓存 |
| 当前实验预算 | 模型、数据、HF缓存、环境、两条受控训练run、checkpoints和临时写入合计限制在 **300GiB**；整个挂载盘保留至少 **500GiB** 空间。不做批量20k小时下载；此预算替代迁移前120GB/200GB的初始预算 |

先在当前空闲显存内做两卡小 batch 的真实 smoke。DDP 每卡有模型与优化器副本，应按较少空闲显存的 GPU0 定 batch。不要假定两张卡的显存可直接合并。遇到资源争用时只停止本项目进程，记录原因，保留检查点，不影响现有服务。

## 3. 数据：先把 100 小时做扎实

### 3.1 首轮来源和划分

直接使用 OpenSLR 官方归档，避免把第三方转换版本、登录门槛或易变数据加载脚本放在首轮关键路径。LibriSpeech 为 16kHz 英语朗读语料，官方资源页标注 CC BY 4.0；提供归档校验和。[官方数据页与下载](https://www.openslr.org/12)

| split | 用途 | 获取与处理 |
|---|---|---|
| `train-clean-100` | 唯一首轮训练数据 | 官方归档约 6.3G；实际样本数、总时长和保留量由 manifest 统计 |
| `dev-clean` | 基线、训练中检查、checkpoint 选择 | 固定 seed，跨说话人抽样 128 条作快速监控；保存 ID 清单与采样规则，不能取文件排序前128条；阶段结束完整生成评测 |
| `dev-other` | 后续稳健性诊断 | pilot 结束时补齐或先下载；与 clean 分开报 |
| `test-clean`, `test-other` | 最终一次独立报告 | 不参与学习率、过滤规则、early stop 或 checkpoint 选择 |

下载采用 `.part` 临时文件、重试与续传，完成后对照官方校验和，再写本地 SHA-256。解压前检查路径安全和剩余空间。只在本项目目录操作归档，不扫描或清理别人的数据。源文件保持只读；不把训练数据、HF 缓存和大权重提交 Git。

统一资源根目录为 `/mnt/raid/asr-lab`，下设 `downloads/`、`datasets/LibriSpeech/`、`manifests/`、`cache/`、`runs/`、`tmp/` 与独立 venv。设置 `TMPDIR`、`HF_HOME`、`PIP_CACHE_DIR`、`XDG_CACHE_HOME` 到该目录；所有 CLI 从配置读取。实际目录名以实施配置为准；迁移先前 `/data` 资产时校验已有下载与模型缓存，不能因路径变化无谓重复下载。

### 3.2 最小可审计 manifest

每条记录保存 `id, split, speaker_id, chapter_id, audio_path, sample_rate, channels, num_samples, duration_s, transcript_raw, source_url, archive_sha256, license`。音频内容校验结果与文本标准化版本另存。输出 `data_report.json`：原始/保留/拒绝条数、小时数、拒绝理由、最大时长、文字长度分布、split 间 speaker/ID/音频重复检查。

必须处理：损坏文件、非有限采样值、空标签、零时长、采样率异常、超过模型最大文本长度，以及音频超过 30 秒。首轮可排除超长条目，但逐项计数；不得截取前 30 秒音频后仍用全段文本监督。后续切段必须有可信对齐或使用支持长音频训练的独立方案。

保留原始 transcript，并维护单独标准化字段。首轮不过早进行强清洗，不按任意字符/秒速率或“非目标字符超过 5%”删除方言、数字、姓名和 code-switch。重复短句可由不同说话人产生，**文本相同不能自动作为删除理由**。跨 split 相同 ID/原始音频内容是硬冲突；文本 n-gram 和音频指纹命中作为候选报告，区分同书、常见短语与真实录音重叠。

公开预训练模型的完整训练来源不可由本项目审计完全还原，因此结果应表述为“本次微调 train/dev/test 隔离”，不宣称 pretrained 模型从未见过任何评测内容。

### 3.3 首轮存储方式

保留原始 FLAC，在线解码并用官方 processor 计算特征。首轮避免为所有样本保存固定 30s 的 float32 mel 副本，以免扩大 IO 与磁盘占用。每 rank 先 2 个 loader workers，按 CPU/IO 实测调整；`pin_memory`、有限预取、worker 种子固定。写清转采样方法；LibriSpeech 已符合 16kHz 时无需重复转换。

## 4. Whisper 对照实验的具体合同

### 4.1 默认配置

这些是起始超参，实施若变更，应写入 resolved config 和运行记录；不能静默沿用本表名称。

| 项 | 首轮默认 |
|---|---|
| 模型 | `openai/whisper-small`；模型与 processor 固定同一 HF revision |
| 数据 | 全量可用 `train-clean-100`，完整 1 epoch；实际覆盖率报告 |
| 并行 | `torchrun --standalone --nproc_per_node=2`，每进程绑定一张 GPU，NCCL DDP |
| 精度 / attention | bf16 autocast；PyTorch SDPA 优先；首轮不编译自定义 FlashAttention，不启用 `torch.compile` |
| Batch | 每卡 4 条，梯度累积 4，world size 2，标称全局 32 条；稳定后可调并记录 |
| Optimizer | AdamW，lr `1e-5`，betas `(0.9, 0.999)`，weight decay `0.01` |
| Schedule | warmup 为总 optimizer updates 的 5%，随后线性衰减；gradient clip `1.0` |
| 随机性 | seed `42`；记录 rank/worker 派生种子；每 epoch `sampler.set_epoch` |
| 内存 | 需要时启用 gradient checkpointing，训练 `use_cache=False`；不同时引入量化/LoRA/FSDP |
| 评测 | 同一固定 dev 子集基线、途中与末尾 greedy generation；完整 dev 至少在 pilot 结束完成 |
| 保存 | 每 100 optimizer steps 和 epoch 结束原子保存；保留 latest、best 和一个前序恢复点，避免无限堆积 |
| 总量约束 | 默认 1 epoch；另设 12 小时本项目训练墙钟上限，触发时标记 `budget_stopped`，不标记 1 epoch 完成 |

microbatch 增大只作为吞吐优化，不以填满 GPU 为目标。首个可用配置优先确认两卡都参与、显存仍有余量、现有进程未被影响。环境采用独立 venv，优先选主机已验证兼容的 PyTorch CUDA wheel；不把 `nvidia-smi` 的 CUDA 13 当作必须安装 cu13 的理由。

实施仓库目前采用 `asr_lab.train` / `asr_lab.evaluate` 模块。安装项目与配置路径验证后，正式 pilot 的入口如下；这段记录启动合同，不代表命令已经完成：

```bash
torchrun --standalone --nproc_per_node=2 -m asr_lab.train \
  --config configs/whisper_small_librispeech100.yaml \
  --run-dir /mnt/raid/asr-lab/runs/whisper-small-ls100-pilot

# 完整dev评测：不传--limit，使用实际已保存checkpoint。
python -m asr_lab.evaluate \
  --model /mnt/raid/asr-lab/runs/whisper-small-ls100-pilot/checkpoints/best \
  --manifest /mnt/raid/asr-lab/data/manifests/dev-clean.jsonl \
  --output /mnt/raid/asr-lab/runs/whisper-small-ls100-pilot/full-dev/predictions.jsonl
```

完整 dev 基线需用同样 evaluator 加载原始模型及配置中的固定 revision。恢复用相同训练命令附加 `--resume <checkpoint路径>`；改变总步数/学习率日程的续训属于新实验，不能冒充完全一致恢复。

### 4.2 Whisper 标签与生成必须正确

保留原始 convolution、位置编码、完整 attention 行为与 `WhisperProcessor`。固定短音频使用 processor 的 30s 特征合同；不手写近似 fbank 替代预训练输入分布。[Whisper 官方实现](https://github.com/openai/whisper)

1. Tokenizer 显式设置 `language=english`, `task=transcribe`, no timestamps。训练和 generation 的前缀策略一致，并保存 generation config。
2. 使用 tokenizer 返回的有效位置 mask 把 **padding** 改成 `-100`，保留真实 EOS 的监督。不能简单把等于 pad ID 的所有 token 都设为 `-100`，因为部分 tokenizer 的 pad 与 EOS 共享 ID。
3. `decoder_start_token_id` 只移除一次；之后由模型执行标准 shift-right。用一个真实 batch 导出 token ID、解码文本和 loss mask 作人工可读检查。
4. `attention_mask` 与实际输入同批传递，特别是 batched generation。HF 当前文档明确要求这一点。[Whisper API](https://huggingface.co/docs/transformers/model_doc/whisper)
5. 生成评测只使用音频和固定任务前缀；不传参考文字作为 decoder 输入。teacher-forced loss 下降不等于真实 WER 下降。
6. 对空输出、重复输出、达到 `max_new_tokens` 上限分别计数；生成上限根据预先统计的标签长度设置在模型允许范围内，记录截断比例。

### 4.3 DDP 正确性与恢复

训练与评测分别设计 sampler。训练可能为 rank 对齐重复少量样本，需报告 sampler padding；评测必须按 ID 去重且完整覆盖，不把 padding 重复条目重复计入 WER。所有 rank 在一致的 step 执行 collective，不能让 rank0 随意评测而其他 rank 继续训练造成死锁。可在同步边界由 rank0 使用 unwrapped model 评测，其他 rank 等待，并在显式有限超时下恢复训练。

梯度累积的最后不足窗口不能直接丢弃；scheduler/`global_step` 只按 optimizer update 递增。loss 聚合注明按 token、样本还是 batch，避免不同文本长度和 rank 不均导致指标失真。DDP 默认同步梯度而不自动给出全局 token 加权 loss；实现至少报告真实全局 token loss，必要时按有效 token 数修正梯度权重。[PyTorch DDP 文档](https://docs.pytorch.org/docs/stable/notes/ddp.html)

恢复 checkpoint 包含模型、optimizer、scheduler、epoch、optimizer step、sampler 进度/已消费 batch、各 rank CPU/CUDA/Python/NumPy RNG 和 resolved config。恢复前核对模型 revision、world size、数据 manifest hash、batch/累积方式；冲突需新建 continuation run，并说明不能逐步复现。临时目录写完再 rename，rank0 写权重，必要的 rank 私有状态单独收集。只保存 `save_pretrained` 权重称为导出，不能冒充完整恢复点。

在短验证运行中实际演示“保存 → 终止本项目进程 → 恢复 → step 连续增长且 optimizer/scheduler 连续”；无需故障注入外部服务。把恢复演练记录写入运行报告。

## 5. 启动顺序与验收门槛

| 阶段 | 实际动作 | 通过标准 | 未通过时的处理 |
|---|---|---|---|
| A 数据和环境 | 下载、校验、manifest、依赖锁、两卡可用性 | 归档校验通过、split 无硬冲突、磁盘满足预算 | 修正相应问题，不能先训练再补数据证据 |
| B 正确性 smoke | small 原模型 2 卡 5–20 updates；必要时 tiny 更早检查环境 | loss/grad 有限，两卡有本项目进程和更新，所有 label/shape 检查通过，checkpoint 可恢复 | 定位实现；smoke 不计正式训练结果 |
| C 零步基线 | 相同 small checkpoint，固定跨说话人的 128 条 dev 音频生成 | WER、S/D/I、参考词数、预测文本、时间与 config 已保存 | 无基线时不宣传后续提升 |
| D 正式 pilot | 全量 train-clean-100，1 epoch | 足量样本已消费；定期日志/checkpoint；结束 full dev 生成评测 | 只跑一部分则准确报告 coverage、steps 和剩余时间 |
| E 续训判断 | 对比基线、dev 预测、性能与资源 | 无技术错误，结果解释清楚，有可恢复产物 | 先修正；不以“WER 没达 3%”直接判为代码错误 |

阶段 B 与 C 可按资源安排顺序，C 必须在模型更新前或单独加载原始权重完成。阶段 D 的 checkpoint 选择只能用 dev。首轮最好补跑 baseline 的 full dev，使最终对比在完整相同 ID 集合上成立；若未完成，报告固定子集对比，不能将其和完整 dev 的数字直接相减。

自动停止条件：NaN/Inf、DDP rank 失败、持续 OOM、无有效标签、磁盘保留线突破、项目预算突破。技术异常标记 `failed`；预算用完标记 `budget_stopped`；无改进但完整运行仍可标记 `completed`，并据实报告结果。

扩至最多 3 epochs 的条件是：恢复与数据验证通过、完整 dev/子集结果可比较、预算内预计可完成、学习曲线仍有改进迹象。可预登记保守规则：若同一固定 dev 集相对基线恶化超过 1 个绝对 WER 百分点且连续两次评测无改善，先审计 transcript/前缀/学习率，不自动扩大训练量。这里的 1 点是实验运维规则，不是统计显著性阈值。

## 6. 实测时间和监控

原方案“3–5 天跑完 20k 小时、多阶段和消融”删除。训练时间由实际 batch、30s padding、输入解码、反向传播、DDP 通信和自回归评测共同决定，不能用理论 BF16 峰值和单一 `6NT` 公式承诺。

前 20 updates 作 warmup，随后至少 50 updates 记录稳态窗口；较短 epoch 就报告实际可用窗口。计入两卡实际消费音频秒数和 optimizer updates，保存：

```text
train_audio_hours_per_wall_hour = sum(actual_duration_seconds) / elapsed_wall_seconds
ETA_train_seconds = remaining_audio_seconds / observed_audio_seconds_per_wall_second
ETA_total = ETA_train + planned_eval_seconds + checkpoint_seconds + remaining_preparation_seconds
```

第一行数值等同于 audio-seconds/s；时长累计必须对两卡求和，不能误乘 world size 两次。另报 examples/s、p50/p95 update 秒数、实际音频/固定 pad 秒数比率、data wait、每卡显存峰值。ETA 给区间并随真实进度更新；发生 batch 改动或资源争用后重新估算。

每个 run 目录至少包括：

```text
resolved_config.json       # 所有默认值已展开
environment.json          # Python / torch / CUDA runtime / NCCL / GPU / driver
requirements-lock.txt     # 实际安装版本
provenance.json           # Git commit+dirty diff hash、model revision、manifest hashes
metrics.jsonl             # UTC时间、step、epoch、loss、lr、grad_norm、吞吐、coverage
events/                   # TensorBoard同口径曲线
gpu.csv                   # 每10秒GPU utilization、memory、power、temperature
stdout.log
status.json               # 状态、PID、最后进展、checkpoint、error、ETA
checkpoints/
eval/                     # 每个sample的ref/hyp、S/D/I与汇总
```

只由 rank0 写公共 JSONL/TensorBoard；rank 私有日志明确后缀。`nvidia-smi` 监控包含 UUID 与进程 PID，保留现有进程但不误算成本项目显存。训练用可持久运行的 session，断开 SSH 后不丢失；TensorBoard 绑定 loopback，通过 SSH tunnel 查看，不默认公开端口。

“正在训练”的证据是 step/样本数/权重或 optimizer 状态持续更新，不能只凭 GPU 内存有占用。“完成训练”的证据是终态、完整 coverage、可加载 checkpoint 和结束评测，不能只凭启动命令返回成功。

## 7. 评测与 GitHub 交付

英文主指标采用固定版本 normalizer 下的 corpus WER：先累加替换/删除/插入，再除以参考词数，不能平均每句 WER。保存 raw 与 normalized transcript；逐项记录 normalizer、解码 beam、generation 参数和空输出处理。报告 `WER%` 和 `S/D/I` 原始整数，并明确比例/百分比单位。中文后续用 CER，code-switch 用显式分词规则的 MER。

第一轮报告表：原始 Whisper-small 与 fine-tuned small 在相同 dev 集上的 WER、绝对差、样本数/小时数；训练 updates/epochs、两卡实际耗时与显存；最常见 10 类错误和对应 ID。先 greedy，beam sweep 后置。同一基线不能引用不同论文、不同 normalizer 的数字替代本机重跑。

最终 test 只在选型冻结后生成一次主要报告。需要声称提升时，在同一 test 样本做配对 bootstrap，并优先以 speaker/chapter 为重采样单元；若区间跨零，写“未得到清晰提升证据”。读书朗读集结果不外推至会议、电话、噪声、方言或生产语音。

GitHub 仓库应交付：README 快速开始；本计划和审阅表；配置；数据下载与 manifest CLI；训练/恢复/评测 CLI；离线 CPU 测试；GPU smoke 入口；监控说明；数据许可和模型来源；实验报告模板。测试针对真实风险：label padding/EOS、shift、超长拒绝、WER 累加、split 隔离、checkpoint 恢复、两 rank 评测不重复。不要为每个可逆细节堆砌测试。

公开仓库忽略 `.env`、凭证、SSH 配置、私有主机信息、大数据和权重。真实运行产物以小型汇总和可复现配置公开；模型发布须另附来源、适用范围与限制。仓库创建、推送或开 PR 的事实以实际工具回执为准。

## 8. 通过 pilot 后的路线图

### 8.1 先做 960 小时与中文，而非直接 20k 小时

1. **LibriSpeech 960h**：逐步加入 train-clean-360、train-other-500，每个新归档重新校验、统计和审计；dev/test 不变。先确认总预算仍小于300GiB且全盘仍余500GiB，并把已有训练恢复点和临时保存计入占用。归档与解压双份可能越限，只有在已验证解压内容完整后才清理本项目可再下载的归档；不清理别人文件。
2. **AISHELL-1 中文**：官方页列 Apache 2.0，同时正文有学术使用描述；入库时保留原许可文本并检查下载包，公开代码与模型说明准确保留来源和使用边界。[官方页](https://www.openslr.org/33/)
3. 中文先保持官方 train/dev/test speaker 划分，不从 test 抽示例训练。使用中英独立 dev 和加权汇总，避免英语单域微调掩盖中文遗忘。
4. 中英混合按实际训练音频时长或 tokens 记录采样分布，不能只写“50/50”而无定义。先各自小验证，再比较纯英、纯中、混合三种受控配置。
5. GigaSpeech/WenetSpeech/其他数据放到来源和许可核查后的待办列表，不再写“全部无登录直接下载”。授权等待、存储不足与带宽成本都可能改变路线；20k 小时不是当前机器的默认下载任务。

### 8.2 数据质量研究只做一个清楚的变量

首选可解释消融：原始数据 vs 基础审计 vs 质量排序采样。过滤器只在训练集与 dev 设计；test 不参与阈值。教师 ASR 与人工标签不一致是质量线索，不自动证明标签错误，也可能是口音、姓名、噪声或教师偏差。人工检查一小批高/中/低分样本，并记录理由。

比较时同时报告保留小时数、说话人/领域分布和训练曝光量。为分离“更好数据”与“更少数据/不同训练步数”，至少提供等小时或等计算预算的随机抽样对照。多系统多数票是伪标签置信代理，不是独立真值。首轮不为整库运行昂贵的多教师转写。

### 8.3 自研 Speech-LLM 拼接：可选研究分支

用户新增要求后，主线已采用第11节的预训练 Qwen3-ASR，本节从“必经阶段”降为可选架构研究。只有想研究 projector/编码器设计本身时才实施，不能为了展示自研而重复支付已预训练语音对齐的成本。

先从小配置开始：原始 Whisper-small encoder → 每 4 帧拼接/投影 → 2 层 GELU projector → `Qwen/Qwen3-0.6B-Base`。官方配置确认其 hidden size 1024、28 层、vocab 151936；实际总参数与 trainable 参数由程序计数，不手工估算后当成实测。[官方配置](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/main/config.json)

第一步仅做非流式转写。冻结 encoder 与 LLM，只训练 projector 和必要边界 embeddings；但 **冻结 LLM 权重仍需保留对输入 embedding 的反向传播**，不能对整个 LLM 使用 `torch.no_grad()`。只有 encoder 全冻结且无训练增强时才适合提前缓存其输出。第二步在明确收益后解冻顶部 encoder 或给 LLM 加 LoRA；全量 LLM SFT 后置。

Base 模型不是经过对话对齐的 instruct 模型。采用明确的纯文本任务 prefix，或验证 tokenizer/chat template 的实际行为再启用对应格式；不能因为模板字符串可以编码就声称模型已理解对话协议。新 token 必须正确注册并扩展 embedding，或以受控 `inputs_embeds` 边界实现。不得把 `vocab_size + offset` 未注册 ID 直接交给普通 HF embedding 查表。

音频有效长度必须从 waveform 经 frontend/conv 传播到下采样后的 token 数。音频 padding、文本 padding、prefix 和目标 loss mask 分开定义。causal LLM 的首个目标 token 由最后一个 prefix 位置预测；CE 仅覆盖目标文本与 EOS，并以标准 shift 验证。首轮不做样本 packing；后续 packing 必须隔离不同样本 attention、位置与标签，不能只用 EOS 分隔。

不设置“projector 一轮后 WER 必须 <6%”作为代码正确性门槛。先以小训练集过拟合、打乱音频对照、无音频对照确认模型真正使用声学信息，再以相同 dev 检验泛化。只完成 ASR 指令的模型称为“面向转写的 speech-conditioned LLM”，要宣称语音理解/对话能力还需要相应数据与任务评测。

### 8.4 流式独立推进

近期可以实现原 Whisper 的滑窗重解码、稳定前缀提交和 VAD endpoint，并称为 **在线分块/增量展示包装**。记录窗口、重叠、重算开销、partial 修改率与 endpoint 延迟；不要称其为原生因果编码器。可参考原作者的 Whisper streaming 实现思想，但本项目必须重跑指标。[Whisper-Streaming](https://github.com/ufal/whisper_streaming)

真正 chunk-causal encoder 是下一研究阶段。先只改变 attention 可见范围，保留已加载的位置编码和卷积，明确其右侧依赖；再逐项研究 causal conv、位置编码迁移和 cache。每次只改一个因素，对比全上下文性能下降与延迟收益。把双向 Whisper 一次性改为 causal conv + RoPE + chunk attention 会改变预训练语义，不能视为无代价迁移。

必须用未来扰动测试：保持已观察 prefix 不变，只改变未来 waveform，已提交的 encoder 输出/文本在宣称的 lookahead 范围外不应改变。检查 STFT center padding、log-mel 的整段归一化、卷积、VAD 和 resampler 是否泄漏未来；attention mask 正确并不自动保证整个前端因果。

交错音频/文本 LLM 再后置：训练时将对齐文字分配到真正可用的音频块，`wait`/`end` 监督与推理状态机一致；控制 token 必须在训练中实际出现。训练不能读取未来块构造当前条件。追加 KV cache 只在过去表示不再改变、位置和 masks 一致时成立；先验证 full recompute 与 cache 分块结果，再报告加速。CTC forced alignment 的 tokenizer、blank、词边界与失败样本必须可审计。

流式指标包括最终 WER/CER、首个稳定字符延迟、音频尾至最终文本延迟、partial 修订率、计算 RTF、各 chunk p50/p95/p99 和排队延迟。按实时音频到达重放，不能预先把全部音频送入再声称低延迟。chunk 长度、lookahead、VAD endpoint 等待与 compute 分开报告。

## 9. 完成条件与诚实表述

当前任务可交付的最低有效结果是：可运行仓库 + 已核实数据 + Whisper 对照 + 主路线 Qwen3-ASR 两卡训练有真实 optimizer updates + 可恢复 checkpoint + 可查看监控 + 明确运行状态；**启动不等于训练完成**。如果 Qwen 尚未启动，只能如实标为准备中，不能用 Whisper 训练替代用户新增的主架构要求。完整第一阶段结论还需完成指定 epoch/coverage 和相同划分的基线与结束评测。

研究成功与工程成功分开：流程稳定但 WER 未改善仍是有效实验，应解释领域适配/过拟合/normalization 等证据，保留原模型作为更好的 checkpoint。未做人工听辨、未跑真实流式、未部署时必须保持相应状态为 pending。面试和公开报告仅使用本项目已经验证的训练、数据、评测和部署事实。

## 10. 主要一手资料

- [DeepInfer 官方仓库与固定提交](https://github.com/deepseek-ai/DeepInfer/tree/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9)
- [OpenAI Whisper 官方仓库](https://github.com/openai/whisper)
- [Whisper-small 官方发布模型](https://huggingface.co/openai/whisper-small)
- [Transformers Whisper API](https://huggingface.co/docs/transformers/model_doc/whisper)
- [Hugging Face Whisper 微调示例与 collator 解释](https://huggingface.co/blog/fine-tune-whisper)
- [LibriSpeech 官方资源与校验和入口](https://www.openslr.org/12)
- [AISHELL-1 官方资源](https://www.openslr.org/33/)
- [PyTorch DDP](https://docs.pytorch.org/docs/stable/notes/ddp.html)
- [Qwen3-0.6B-Base 官方模型与配置](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
- [Whisper-Streaming 作者仓库](https://github.com/ufal/whisper_streaming)

## 11. 更新后的主实验：Qwen3-ASR 与双 H100 效率验证

### 11.1 选型与固定版本

核查时点为2026-10-22。以下是本次已核实、可立即开展对照的候选，不声称穷尽全球最新发布。架构较新、参数较大或公开排行榜较强，都不保证在本项目数据上微调后一定更好；验收仍以同一 evaluator 的结果为准。

| 候选 | 固定模型 revision | 官方 Hub safetensors 总参数 | 选择 |
|---|---|---:|---|
| `Qwen/Qwen3-ASR-1.7B` | `7278e1e70fe206f11671096ffdd38061171dd6e5` | 2,038,052,480（加载后唯一参数实测） | 主实验；命名中的1.7B不能误当完整语音模型总参数 |
| `Qwen/Qwen3-ASR-0.6B` | `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` | 938,008,576 | 显存/吞吐备选及效率消融；不是默认悄悄降级 |
| `openai/whisper-large-v3-turbo` | `41f01f3fe87f28c78e2fbf8b568835947dd65ed9` | 808,878,080 | 更强encoder-decoder的零样本精度/速度对照，暂不增加一条完整训练线 |
| `openai/whisper-small` | 以现有配置的固定revision为准 | 约244M | 工程与历史基线 |

参数量来自本次官方 HF API 的 safetensors 元数据，运行时再用 `sum(p.numel())` 验证并分别报告总参数与 trainable 参数。[Qwen1.7B模型](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/tree/7278e1e70fe206f11671096ffdd38061171dd6e5)、[Qwen0.6B模型](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/tree/5eb144179a02acc5e5ba31e748d22b0cf3e303b0)、[Whisper turbo](https://huggingface.co/openai/whisper-large-v3-turbo/tree/41f01f3fe87f28c78e2fbf8b568835947dd65ed9)

Qwen3-ASR是已有音频编码器与Qwen文本解码器联合预训练的语音条件模型，官方提供离线和流式推理接口。1.7B配置包括128维mel、24层音频编码器及28层文本解码器；使用其官方 processor 与特殊 token，不复用Whisper-small的80维输入或前缀。[官方配置](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/blob/7278e1e70fe206f11671096ffdd38061171dd6e5/config.json)

Whisper-large-v3-turbo将large-v3的decoder层数从32减到4，适合检验“更大声学编码器+浅decoder”的速度与精度；它仍是Whisper encoder-decoder架构。不能把不同模型族的参数规模直接当成速度排序。[官方说明](https://huggingface.co/openai/whisper-large-v3-turbo)

Qwen模型与代码均标Apache-2.0；turbo模型卡标MIT。保留原许可、来源和修改记录。Qwen代码固定 `QwenLM/Qwen3-ASR@7c6daf77a2421100f5fb066495372c00129d39ff`，其中包版本 `qwen-asr==0.0.6`，依赖锁有 `transformers==4.57.6`、`accelerate==1.12.0`；训练另需 `datasets`。完整resolved依赖仍需本机安装后锁定，不能把最低依赖列表当可复现环境。[官方pyproject](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/pyproject.toml)

### 11.2 官方训练入口与必须修复的边界

官方入口 `finetuning/qwen3_asr_sft.py` 使用HF Trainer，支持 `torchrun` 多GPU全参数SFT、`--resume_from`/`--resume`、定期checkpoint。它不是已集成LoRA/FSDP2/TP的完整训练平台，不能把官方推理的tensor-parallel参数当训练支持。[官方训练源码](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/finetuning/qwen3_asr_sft.py)

保留官方模型和processor代码，在本仓库添加范围清晰的训练adapter/补丁，避免重写模型。实际运行前至少完成以下审计：

1. **修复左padding的prefix label mask。** 官方processor默认 `padding_side="left"`，而示例collator使用 `labels[i, :prefix_len] = -100`。不同长度样本左侧pad后，这会让prompt尾部进入目标loss。训练显式传右padding并断言，或按attention mask计算有效起点再屏蔽prefix；推理维持适当的左padding。测试用不同音频长度、不同文本长度的两条真实样本，确认非`-100`部分解码只含目标语言prefix、转写与EOS。[processor默认值](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/qwen_asr/core/transformers_backend/processing_qwen3_asr.py)
2. **固定本地模型snapshot。** 先按revision下载到`/mnt/raid/asr-lab/models/`，再传本地`--model_path`。官方checkpoint callback从本地路径复制processor/tokenizer文件；传远程HF ID不等同于已完成这些复制。保存后实测用官方`Qwen3ASRModel.from_pretrained(local_checkpoint)`加载并转写。
3. **保留训练数值精度。** 官方示例直接加载bf16参数。本项目优先FP32可训练参数、FP32 Adam状态、bf16 autocast；不得默认以bf16参数+bf16 optimizer状态替代稳定mixed precision。若采用低精度optimizer，需明确有FP32 master/state机制，并测低学习率的真实参数更新。
4. **加入生成式WER与监控。** 官方脚本`report_to="none"`且eval主要是teacher-forced loss；必须接TensorBoard/JSONL callback，并在独立音频推理上计算WER。官方`eval_strategy="steps"`与无eval_file的默认组合也需修正或始终提供固定dev文件。
5. **补齐工程参数。** adapter显式暴露seed、attention backend、fused AdamW、gradient checkpointing、max_steps/max_runtime、评测和保存频率。不能向原始脚本传未定义CLI参数再声称已启用。
6. **保存与恢复核查。** 验证Trainer optimizer/scheduler/RNG恢复，保存processor/config，追加本项目数据hash与模型revision校验，防止换数据后误用resume。纯模型导出和恢复checkpoint分开。

训练JSONL沿用已校验的100h manifest，转换为官方格式，保留ID映射：

```json
{"audio":"/mnt/raid/asr-lab/data/LibriSpeech/train-clean-100/.../utt.flac","text":"language English<asr_text>THE TRANSCRIPT","prompt":""}
```

官方文档示例使用WAV，但源码通过`librosa.load`解码；FLAC可在本机真实两条音频smoke确认后直接使用，无需为100h复制整份WAV。原始LibriSpeech大写标签保留；若将训练target改小写，须把转换作为独立配置记录，不能只在一条模型线静默改。格式前缀与EOS独立于正常化WER；评分仅取模型识别文本部分。

官方入口对应的最小参考命令如下，`vendor/Qwen3-ASR`须为上述固定commit且已应用已测试补丁；最终真实入口以README和运行配置为准：

```bash
torchrun --standalone --nproc_per_node=2 vendor/Qwen3-ASR/finetuning/qwen3_asr_sft.py \
  --model_path /mnt/raid/asr-lab/models/Qwen3-ASR-1.7B \
  --train_file /mnt/raid/asr-lab/data/manifests/qwen-train-clean-100.jsonl \
  --eval_file /mnt/raid/asr-lab/data/manifests/qwen-dev-clean-monitor.jsonl \
  --output_dir /mnt/raid/asr-lab/runs/qwen3-asr-1.7b-ls100-pilot \
  --batch_size 2 --grad_acc 8 --lr 1e-5 --epochs 1 \
  --warmup_ratio 0.05 --log_steps 10 --save_steps 100 --save_total_limit 2
```

默认全局batch仍为32；1完整epoch；无新增复杂数据增强；保留完整train/dev隔离和超长拒绝审计。此处选择`1e-5`而非官方演示`2e-5`，用于100h单域初始适配的保守起点；最终选择依赖dev，不保证优于原checkpoint。

### 11.3 为什么先用DDP，不默认TP

本机两卡各约94GiB，GPU0已有服务占约23GB；拓扑显示NODE且没有已验证NVLink。模型并行要为模型容量或实测性能服务，不以“同时启用所有并行类型”作为完成标准。

| 策略 | 本机用途 | 选择条件 |
|---|---|---|
| 两卡DDP数据并行 | 每卡完整模型，处理不同样本；累积末尾一次同步梯度 | **默认主线**，容量可行、官方Trainer支持、通信可被较大有效batch摊薄 |
| FSDP2全分片 | 分摊参数/梯度/optimizer状态，前后向产生层级通信 | DDP最低可用batch仍OOM，或更大模型有明确必要时；先证明自定义Qwen模块、checkpoint、梯度累计和恢复正确 |
| ZeRO-2 | 主要分摊optimizer/梯度，可作为内存折衷 | 仅在已有兼容集成且DDP内存不够时评估；不是当前首跑依赖 |
| TP张量并行 | 每层tensor分片，频繁collective | 官方训练脚本没有现成TP合同；本机跨NODE通信可能抵消收益，除非实测优于DDP且数值正确，否则不选 |
| Pipeline并行 | 层分到不同GPU，需调度microbatch消除bubble | 两卡容量已够且模型不大时没有首轮必要 |

以run.json加载后实测唯一参数2.038B估算，FP32参数、梯度、Adam一二阶矩合计约`16P=32.61GB≈30.37GiB/卡`（修正早期HF元数据2.349B估算）。这还**不包括**DDP bucket、临时转换、attention/logits、激活和CUDA workspace；启动前按GPU0剩余空间实测。先microbatch2，必要时activation checkpointing；最低batch仍不适配时明确记录后采用0.6B或经验证分片，不能动现有服务。

恢复checkpoint一般不保存梯度，因此FP32模型+Adam状态约`12P=28.19GB≈26.3GiB/份`，还要加元数据。保留2份加一次原子写临时副本，单run峰值约84.6GB；两条run可超过原120GB预算。必须另计预训练snapshot、最终export重复权重、环境和数据。当前campaign采用300GiB上限、挂载盘500GiB保留线；每次保存前按实际文件大小预留临时空间，不能先删唯一可恢复点再保存新文件。可复用同一checkpoint作为最终模型，避免无意义复制整份export。

充分利用GPU的指标是有效音频吞吐、有效tokens吞吐、内存余量、训练质量和费用；不是让显存越满越好。若小batch通信占比高，优先增大每卡batch/梯度累积并保持实验有效batch口径，再考虑复杂并行。

### 11.4 固定有效batch的单卡/双卡性能实验

性能实验固定模型、数据、有效batch与学习率，尽可能只改执行方式。固定包含短/中/长音频的2048条训练子集与seed；先各做10个warmup updates，再30个测量updates，每个配置最多40updates。每个变体从同一模型初始化，不把前一个benchmark训练后的权重作为后一个起点。当前HF adapter采用“各microbatch的token平均loss再等权平均”，不同microbatch/rank分组并不严格等同于同一全局token加权目标，因此这些试验用于**吞吐选型**，不能据此声称梯度完全等价或训练质量优劣；目标reduction写入metadata。要研究质量/收敛等价性，应另实现窗口级global-token归一化并测试。检查点和生成评测耗时单独记录，并补充真实端到端耗时；不得把排除数据加载的纯kernel时间当用户可见训练速度。

| 试验 | GPU / microbatch / accumulation | 有效batch | 目的 |
|---|---|---:|---|
| E1 | 单GPU1 / 2 / 16 | 32 | 空闲单卡基准 |
| E2 | 双GPU / 2 / 8 | 32 | 双卡DDP强扩展 |
| E3 | 双GPU / 4 / 4 | 32 | 显存允许时减少累积/调度成本 |
| E4 | 双GPU / 8 / 2 | 32 | GPU0仍有安全余量时尝试更高microbatch；OOM则记录不可行，不反复撞内存 |
| E5 | 最佳可行batch + gradient checkpointing | 32 | 量化用重算换显存的成本，仅需要时启用 |
| E6 | 最佳可行batch，SDPA vs FA2 | 32 | 安装与mask检查通过后最多一个额外变体 |

报告`speedup=单卡时间/双卡时间`与`parallel_efficiency=speedup/2`；同时报告真实总音频秒数、padding比、各卡峰值显存、data wait和双卡通信占比。GPU0已有负载使E1/E2不完全对称，因此注明干扰并保存同时间窗的现有进程资源快照。目标是选择实测最好的稳定两卡配置，不预承诺接近2倍加速。

优化顺序：

1. bf16 autocast与可用的SDPA；PyTorch fused AdamW（`fused=True`/Trainer对应`optim`）经一次真实update验证后启用。
2. 同一DDP rank内按音频/总token长度分桶，保留epoch级随机与rank样本分配；避免所有长句集中某rank形成straggler。公开sampler规则与实际coverage，不在首轮把多样本拼到同一causal序列。
3. 按实测调整CPU workers、预取和本地FLAC读取。官方collator对full与prefix各调用一次processor，可能重复算音频特征；正确性稳定后才能复用已算特征/音频token长度，并测token与labels完全相同。
4. FA2只在目标环境可安装、当前mask受支持且前后向有效时比较。模型源码声明支持SDPA/Flash，不代表任意自定义mask都能直接使用。安装失败则继续SDPA，不让kernel构建阻塞训练。
5. Activation checkpointing用于满足显存需求，默认不把重算开销说成加速。`torch.compile`在动态音频长度下可能重编译，只有固定shape/分桶后经独立收益验证才考虑；不强制启用。

任何优化先比较相同真实batch的loss、非零有效梯度、目标mask和一步参数变化，再做吞吐试验。允许bf16引起的小数值差异，但不能以速度为由忽略音频/文本错位或生成质量退化。

### 11.5 主实验阶段与当前边界

顺序为：固定snapshot和依赖 → 两条真实音频collator/生成检查 → 两卡短更新与恢复 → 上述有界效率实验 → 相同128条dev的零步WER → Qwen1.7B全量100h的一轮SFT → 完整dev基线/结束评测 → 判断是否扩数据或继续训练。Whisper已有验证和数据资产可复用；新的模型loader/processor/labels必须独立核验。

首次训练不包含RL、TTS、对话微调、模型压缩或从零重建流式编码器。后续评估官方streaming接口时单独记录音频到达、chunk/revision/最终文本延迟与离线SFT后的回退。当前“质量与效率”是以测量和选择保证过程可靠，不能保证尚未观察到的精度提升、吞吐数值或某个排行榜位置。

### 11.6 自主执行的有限实验矩阵与第二轮证据门槛

用户已授权自动开展实验。执行代理据此连续推进下表，保存每次判断；不因一个阶段完成就停在建议层，也不将token额度解释成无限GPU费用或无边界超参搜索。训练进程可在持久session中继续，代理达到token/运行预算时必须留下真实状态、恢复点和后续队列，不伪称结果已完成。

| 队列 | 工作量上限 | 进入/退出条件 |
|---|---|---|
| Q0 正确性与恢复 | 两卡2–10 updates + 一次恢复验证 | 真实多长度batch标签正确，loss/grad有限，更新非零，checkpoint可加载；失败先修实现 |
| Q1 效率矩阵 | E1–E4各40updates；E5/E6仅有需要且可行时各40updates | 固定global32；选稳定且有效音频吞吐最高的两卡配置；记录失败配置和原因 |
| Q2 零步模型对照 | Whisper-small、Qwen1.7B；turbo和Qwen0.6B各最多一次固定128条dev评测 | 同一ID集合、同一normalizer；后两者不阻塞主训练，不引用官方排名替代实测 |
| Q3 主实验 | Qwen1.7B、lr1e-5、warmup5%、global32、全量合格100h样本1epoch | 起点是固定原模型snapshot；每100updates保存、固定128条dev监控；结束完整合格dev评测 |
| Q4 质量复核 | 原模型与Q3最佳checkpoint各一次完整合格dev，误差分析 | 区分数据/前缀/截断/输出重复/过拟合；只有此阶段证据才能决定第二条训练run |
| Q5 可选单因素第二run | 最多再1epoch、同一原模型起点、同数据和seed，仅lr改5e-6 | 仅当Q3 train loss下降但完整dev较原模型恶化≥0.3个绝对WER百分点，且样本检查排除实现错误时运行；这是保守LR验证，不承诺一定修复 |
| Q6 最终汇总 | 选定checkpoint后独立test及报告 | test结果不返回指导本campaign再调参；保存最佳基线，即使它是未微调模型 |

Q5的0.3百分点为预先设定的行动阈值，不是显著性声明。若Q3无明确改善、差值小于该阈值，不自动开更多learning-rate或seed搜索；保留结果并做误差分类、吞吐分析与可复现交付。若Q3已经改善，不为“用完token”再过拟合相同100h；可优先补充dev-other/中文零步诊断和文档，960h扩展遵循第8节的容量与数据门槛，作为新实验记录。

本次已收到主代理的真实阶段结果：Whisper-small最初固定seed42的随机128条dev子集，83次编辑/2659参考词，WER **3.1215%**，生成耗时约8.47秒，未触发截断。此结果是初始工程基线；**后续建立了独立的speaker-balanced 128条共同manifest，必须在新ID集合上重跑Whisper后才能与Qwen对比，不能沿用3.1215%相减。** 此结果也不是完整官方dev成绩。dev-clean原2703条中2694条符合当前≤30秒合同，9条超长被显式排除；“完整dev”在当前短音频pipeline中准确指**全部2694条合格样本**，必须报告这项覆盖限制。最终若使用原生长音频接口补齐9条，应单列全2703条结果并同步所有比较模型，避免混用分母。

### 11.7 Qwen adapter 训练前只读审阅

本次审阅覆盖 `qwen_data.py`、`qwen_train.py`、`qwen_evaluate.py` 与固定版本HF/Qwen相关源码，未修改实施代码。结论允许进入真实两卡smoke，不能替代该验证。

- 已核对：right-padding断言；full/prefix精确token匹配；prompt/audio loss屏蔽；EOS保留；labels由causal LM内部移位；FP32参数+BF16 autocast；`model_accepts_loss_kwargs=False`让HF执行一次且仅一次梯度累积缩放。
- 已核对：Qwen outer.generate返回含`sequences`的对象，当前按完整padded输入宽度切出continuation正确；已指定English的生成prefix不包含参考转写。
- 已核对：HF4.57.6 `DefaultFlowCallback`在达到`max_steps`且`save_strategy=steps`时强制保存。因此40步benchmark即使常规save_steps为200，末步仍会生成checkpoint；仍需本机确认optimizer/RNG文件与恢复可用。
- 长run前需补齐：rank0磁盘/IO错误广播；各rank finite-loss状态统一后再退出；实际配置与计划学习率/保存频率一致；代码/本地snapshot身份记录。
- 当前Qwen evaluator默认只取128条；完整合格dev显式用`--limit 2694`（或实施后新增的all选项），不能沿用Whisper evaluator“不传limit即完整”的假设。
