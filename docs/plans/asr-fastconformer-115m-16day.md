# 双 H100 十六天 FastConformer ASR 从零训练计划

**状态：执行方案，尚未执行。日期：2026-10-27。** [PDF 阅读版](./pdf/asr-fastconformer-115m-16day.pdf) · [简短 TTS 备选](./tts-vits-brief.md) · [方案索引](./README.md)。推荐用两张 H100 做一条 **约 115M-class、英文、离线 FastConformer-CTC**，在 LibriSpeech 三个训练划分上从随机权重开始。默认主训练最多 50 次自然数据暴露，Day 3–14 训练，Day 15–16 完整评测与交付；模型质量由结果判定，十六天不是准确率承诺。总时钟从未来获准执行的 `T0` 起连续计算 **384 小时**，包含下载、工程、等待、失败与恢复，不因故障重置。本轮只交付方案，没有连接 GPU、下载数据、安装环境或启动训练，也没有新的吞吐、参数量或 WER 实测。本文配置是待合并的差异片段，涉及的预算/恢复包装器尚未实现。

| 决策项 | 本轮固定范围 | 执行时仍须验证 |
|---|---|---|
| 模型 | 17 层、宽度 512、8 heads、CTC；所有参数随机初始化 | 精确参数量、固定版兼容性、CTC 数值与显存 |
| 数据 | LibriSpeech train-clean-100/360、train-other-500；train-only BPE256 | 校验、解码、20 秒训练过滤后的真实小时与条数 |
| 资源 | 最多 2×H100；保守按每卡 80GB，不预设 NVLink | 实际显存、驱动、互联、独占、磁盘与 I/O |
| 主训练 | 默认 50 次暴露；global batch 128；一个种子 | Day 2 实测能否在剩余窗口完成；否则预先缩量 |
| 截止 | Day 14 末停止训练，Day 16 末交付；不自动延长 | 统一时钟、可恢复保存和两卡一致停止适配 |
| 评价 | 完整双 dev 选模；完整双 test，greedy，无外部 LM | 全部句子、分子分母、长句与失败记录完整 |

## 1. 固定十七层模型，先证明确实从零开始

主模型沿用 NeMo Speech 正式 `v3.0.0`，固定完整提交 **`fd6a877539710e2b98f28c43272ff81312f83417`**，不在训练途中追踪 main。官方离线 CTC 配置的注释表把 Large 写为 17×512、8 heads、约 120M，但 YAML 实际设为 18 层；本方案必须显式覆盖 `n_layers=17`。因此“约 115M-class / 约 1.15 亿”是规模描述，精确值须在 Day 1 实例化后用 `sum(p.numel() for p in model.parameters())` 记录，不能把本文数字当作已测参数量。([固定提交](https://github.com/NVIDIA-NeMo/Speech/commit/fd6a877539710e2b98f28c43272ff81312f83417)、[正式版本](https://github.com/NVIDIA-NeMo/Speech/releases/tag/v3.0.0)、[固定配置](https://github.com/NVIDIA-NeMo/Speech/blob/fd6a877539710e2b98f28c43272ff81312f83417/examples/asr/conf/fastconformer/fast-conformer_ctc_bpe.yaml#L103-L133))

使用 80-bin log-mel、25ms 窗、10ms 步长、8 倍下采样、256 subsampling channels，继承固定版完整配置的其余模型字段。`att_context_size=[-1,-1]`、非因果下采样及卷积意味着模型使用双向上下文；本计划验收整句离线识别。原生流式需要另一套因果前端、有限右上下文与缓存训练，不能把离线模型切块调用称为已完成实时流式。朗读语料成绩也不证明会议、医疗语音或所有口音的能力。([官方离线配置](https://github.com/NVIDIA-NeMo/Speech/blob/fd6a877539710e2b98f28c43272ff81312f83417/examples/asr/conf/fastconformer/fast-conformer_ctc_bpe.yaml)、[cache-aware 路线](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#cache-aware-streaming-conformer))

**从零覆盖整个推理链**：声学前端的可训练层、编码器、CTC 输出层都随机初始化，不载入预训练 encoder、SSL 模型、teacher、伪标签或外部 LM。训练文本自产 tokenizer 合法，确定性信号处理和公开实现可以复用。固定版训练入口构建模型后仍会调用 `maybe_init_from_pretrained_checkpoint`；其检查按初始化键是否存在计数，三个键都设为 null 仍可能触发冲突。必须让 `init_from_nemo_model`、`init_from_pretrained_model`、`init_from_ptl_ckpt` **全部不存在**，并在包装器启动时断言，不靠“scratch”目录名证明初始化。([训练入口](https://github.com/NVIDIA-NeMo/Speech/blob/fd6a877539710e2b98f28c43272ff81312f83417/examples/asr/asr_ctc/speech_to_text_ctc_bpe.py)、[初始化判断](https://github.com/NVIDIA-NeMo/Speech/blob/fd6a877539710e2b98f28c43272ff81312f83417/nemo/core/classes/modelPT.py#L1176-L1189))

Day 1 先建立独立源码环境。固定 README 主动测试栈包括 Python 3.13、PyTorch 2.12、CUDA 12.6/13.2；cu12 extra 固定对应 Torch，不能直接拿旧环境可导入当作本版本已验收。若现场驱动支持，候选路线是 `uv sync --locked --python 3.13 --extra asr --extra cu12`，随后核对 lock、导入、`uv pip check`、单步与两 rank DDP；这是未来环境方案，尚未现场通过。保存源码 SHA、lock SHA256、软件版本、CUDA/cuDNN/NCCL 与 GPU 信息。若改用容器，先确认镜像实际存在，再固定 immutable digest，本文不编造 NGC 标签。([README](https://github.com/NVIDIA-NeMo/Speech/blob/fd6a877539710e2b98f28c43272ff81312f83417/README.md#L50)、[依赖约束](https://github.com/NVIDIA-NeMo/Speech/blob/fd6a877539710e2b98f28c43272ff81312f83417/pyproject.toml))

需新增小型入口包装器，在 Trainer/模型构建前固定 `pl.seed_everything(20261027, workers=True)`，输出 resolved config、实际参数量、初始参数指纹和数据/tokenizer hash，保存 step 0 记录。只在 YAML 增加 `seed:` 不会自动完成这些动作。tiny、校准、主训练用不同目录与标签；**Day 3 主模型重新随机初始化，不从已训练的 tiny 或校准权重接续**。主 run 启动使用空输出目录与 `resume_if_exists=false`，后续只允许恢复同一 run 自产检查点，并验证架构、词表、数据与状态对应。

## 2. 用真实保留小时约束数据与全部评估分母

数据只采用 LibriSpeech 官方三份训练 archive，名义约 960 小时；dev-clean/dev-other 用于开发，test-clean/test-other 最后封存评价。数据为 16kHz 英语朗读、CC BY 4.0，保留许可、署名及预处理改动说明。三个训练包标称约 6.3G、23G、30G，四个 dev/test 包合计约 1.3G；这些是压缩体积，不是解压空间。优先直接读原生 FLAC，避免旧脚本批量额外转 WAV。项目暂预留 180–250GB 用于原包、解压、环境和有限检查点，这是待现场核对的容量预算；若再落 16-bit PCM WAV，仅 960h 音频便额外约 110.6GB。([数据发布页](https://www.openslr.org/12)、[官方校验和](https://www.openslr.org/resources/12/md5sum.txt)、[FLAC 支持](https://github.com/NVIDIA-NeMo/Speech/blob/fd6a877539710e2b98f28c43272ff81312f83417/nemo/collections/asr/data/audio_to_text.py#L49))

先校验 archive，再逐文件检查可解码、16kHz、单声道、实际时长、缺失和重复 ID/音频 hash。训练保留 `0.1–20.0s`，排除空文本或损坏条目；20 秒仅是训练吞吐策略，超长样本被排除，不能切音频后继续使用整句文本。保存每种排除原因的条数、小时与 ID。最终名称应为“LibriSpeech 960 训练划分的保留子集，实际 X 小时”，不能仍宣称全部 960h 都参与。`D_clean_h` 表示经检查的完整训练池，包含 train-other-500，不是只用两个 clean 分片。

| 数据池 | 未过滤参考条数 | 本方案用途与要求 |
|---|---:|---|
| train-clean-100 | 28,539 | 与另两份合并审计，按训练规则过滤 |
| train-clean-360 | 104,014 | 训练；不按名称直接当成实际解码小时 |
| train-other-500 | 148,688 | 训练；保留真实困难条件，不为好分数随意删除 |
| dev-clean / dev-other | 2,703 / 2,864 | 全量保留，完整 dev 用于选模 |
| test-clean / test-other | 2,620 / 2,939 | 全量保留，包括超过 20 秒的样本；最后各自报告 |

以上条数用于完整性预检，不是过滤后实测，也不是 WER 分母。LibriSpeech 原始划分说话人互斥，但额外 LM 语料的书籍去重规则不能直接推成所有声学 split 的书籍文本绝对互斥。审计 ID、音频 hash、speaker 交集，并列出可核实的 chapter/book 与规范化文本重叠；常见短语重复不自动删除，不擅改官方 dev/test。([官方数据卡](https://huggingface.co/datasets/openslr/librispeech_asr/blob/main/README.md#data-splits)、[原始论文](https://www.danielpovey.com/files/2015_icassp_librispeech.pdf))

manifest 核心字段为 `audio_filepath/duration/text`；审计表另存 utterance、speaker、chapter、split、source/hash。训练、验证、测试统一小写、空白规则并保留 apostrophe，保存原文和规范化版本，执行前固定标点处理。只用**保留训练文本**训练 SentencePiece BPE256，corpus、tokenizer.model、vocab、manifest 全部归档 SHA256；不传入 dev/test、外部全书、SLR11 或网页。若估计数据级 CMVN，只能从 train 学习；固定的逐样本特征归一化同样在全部阶段一致。官方 tokenizer 会复用已存在的 `document.txt`，因此必须使用新的内容哈希目录，防止旧 corpus 混入。([tokenizer 工具](https://github.com/NVIDIA-NeMo/Speech/blob/fd6a877539710e2b98f28c43272ff81312f83417/scripts/tokenizers/process_asr_text_tokenizer.py#L178)、[词表大小建议](https://github.com/NVIDIA-NeMo/Speech/blob/fd6a877539710e2b98f28c43272ff81312f83417/examples/asr/conf/fastconformer/fast-conformer_ctc_bpe.yaml#L77-L81))

首轮单步要核对音频与文本配对、padding mask、CTC blank ID、tokenizer 往返，以及下采样后的有效帧数是否覆盖 target token 和相邻重复 token 所需的额外 blank。对随机抽取的短/中/长样本人工听读检查；异常条目进入修复账本，不用 loss 有限替代配对正确。全量 dev/test 不设 `max_duration=20`，不截断。遇长句 OOM 就降 evaluation batch，必要时 batch 1；读取/推理失败仍保留失败项，不能通过删句“改善”分数。

## 3. 把全局 batch、学习率和恢复写成一次冻结的配方

正式起点为两卡 DDP、每卡 microbatch 64、累积 1，**global batch=128**。OOM 时先改为每卡 32、累积 2，仍为 global 128，再重新校准峰值显存与完整更新耗时。普通 collate 是 padding 后 stack，不是 LLM packing；每次更新的真实音频时长必须求和，不能用 `128×20s` 当平均值。初版不加速度扰动、不启用拼接音频或新 loader，避免同时改变时长、采样和优化口径。([有效 batch](https://lightning.ai/docs/pytorch/stable/common/gradient_accumulation.html)、[NeMo collate](https://github.com/NVIDIA-NeMo/Speech/blob/fd6a877539710e2b98f28c43272ff81312f83417/nemo/collections/asr/data/audio_to_text.py#L49))

采用 AdamW、`betas=(0.9,0.98)`、weight decay `1e-3`、gradient clip `1.0`、BF16 mixed。**峰值 LR `5e-4`、warmup 为计划 optimizer updates 的 3% 且上限约 5,000 步、cosine floor `5e-6` 均是本项目起点**，不是双 H100 上已验证的最优参数。Day 2 工程门内可锁定一次；正式主 run 不盲扫 LR。若实际计划更新数 `U=110000`，warmup 约 3300；若 `U=60000`，约 1800，不强行设 15k。先计算绝对 `W=min(floor(0.03U),5000)`，只设置 `warmup_steps=W`，不再同时开比例字段。原官方 `lr=1e-3/warmup=15000/min_lr=1e-4/1000 epochs` 面向 global 2K，不能照搬到本短计划，更不能把 Noam 系数当作 cosine 峰值。([官方优化配置](https://github.com/NVIDIA-NeMo/Speech/blob/fd6a877539710e2b98f28c43272ff81312f83417/examples/asr/conf/fastconformer/fast-conformer_ctc_bpe.yaml#L157-L172)、[scheduler 实现](https://github.com/NVIDIA-NeMo/Speech/blob/fd6a877539710e2b98f28c43272ff81312f83417/nemo/core/optim/lr_scheduler.py#L467))

下面是**未来配置覆盖片段，不是独立可运行文件**。须复制固定提交的完整 YAML，再覆盖这些字段；`U/W` 由 Day 2 真实 loader 长度、采样器补齐、累积与暴露上限计算后写成整数。计划以 50 次音频暴露为主约束，`max_steps` 是附加保险，不以未过滤条数近似替代实测。

```yaml
# 待实施：先合并完整官方 YAML；这里仅列项目差异。
model:
  train_ds:
    batch_size: 64           # 每 rank；32 + accumulation=2 是退路
    min_duration: 0.1
    max_duration: 20.0
    num_workers: 8           # 按 Day 2 的 CPU/I/O 实测冻结
    pin_memory: true
  validation_ds:
    batch_size: 8           # 必须使用完整 dev，无 20s 过滤
  test_ds:
    manifest_filepath: null # 训练阶段不自动读取 test
  encoder:
    n_layers: 17
    d_model: 512
    n_heads: 8
    att_context_size: [-1, -1]
    causal_downsampling: false
    conv_context_size: null
  optim:
    name: adamw
    lr: 0.0005
    betas: [0.9, 0.98]
    weight_decay: 0.001
    sched:
      name: CosineAnnealing
      warmup_steps: W       # 待替换为 min(floor(.03*U),5000)
      warmup_ratio: null
      min_lr: 0.000005
trainer:
  devices: 2
  num_nodes: 1
  accelerator: gpu
  strategy: ddp
  precision: bf16-mixed
  accumulate_grad_batches: 1
  gradient_clip_val: 1.0
  max_steps: U              # 待换成锁定的计划 optimizer updates
  max_epochs: -1
  log_every_n_steps: 10
exp_manager:
  resume_if_exists: false
  resume_ignore_no_checkpoint: false
  resume_from_checkpoint: null
# 三个 init_from_* 键必须全部不存在；不是全设为 null。
```

正式沿用固定版 dropout 0.1 与 SpecAugment 的 2 个频率 mask、10 个时间 mask；tiny 诊断关闭增强及 dropout，不把其成绩当泛化结果。BF16 不免除检查：记录 loss、梯度、参数更新、NaN/Inf、输出长度、blank 帧率和空句率。CTC 训练初期 blank 高很常见，首几百步高 blank 本身不判永久失败；持续全空输出、loss/梯度异常或多个固定 dev 无改善才进入诊断。

未来启动接口如下，**本轮不创建或执行 `train_scratch_budgeted.py`**。该待实现包装器应调用固定官方模型/训练逻辑，并额外实现种子、初始化断言、自然音频计数、截止、全状态恢复和各 rank 一致退出。不能把本文命令说成当前仓库已有脚本。普通 Python 入口由 Lightning DDP 建两进程，避免再套两进程 torchrun 造成重复启动。示例 50 须替换成 Day 2 锁定的 cap；实际因时间截止提前停止时，不重设 cosine horizon 伪装完整收敛。

```bash
# 未来接口草案，当前不可直接执行。
python train_scratch_budgeted.py \
  --config-path configs --config-name fastconformer_ctc_115m_ls960 \
  --mode fresh --seed 20261027 \
  --max-natural-exposures 50 --main-deadline-utc '<T0+336h>'
# 恢复仅允许本次主 run 自产 ckpt；参数名属于待实现适配器。
```

检查点保存模型、optimizer、scheduler、precision/scaler（适用时）、global step、累计自然音频、各 rank Python/NumPy/CPU/CUDA RNG、sampler epoch/位置及配置 hash。Lightning 默认 checkpoint 保存很多训练状态，但不证明任意 worker 预取与数据游标逐位恢复；需明确能恢复到什么范围。Day 2 比较“连续 20 更新”与“10 更新→退出→新进程恢复 10 更新”，核对下一批 ID、LR、动量、loss/参数差异与暴露账本，不能只检查 load 成功。若允许重放，记录范围并计入资源；仅加载权重不是完整恢复。([checkpoint 内容及接口](https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html)、[固定 exp_manager](https://github.com/NVIDIA-NeMo/Speech/blob/fd6a877539710e2b98f28c43272ff81312f83417/nemo/utils/exp_manager.py#L871))

## 4. 四天固定开销决定五十次暴露是否可行

`T0` 在未来首次获准执行时登记；日界线是其后每 24 小时，而非午夜。主训练硬截止为 **`T0+336h`**，总截止为 **`T0+384h`**。最多两卡全程占用对应 **768 GPU-hours** 的物理上限，主训练十二天最多 576 GPU-hours；这不是预计实际用量。CPU/下载不一定占卡，仍消耗连续墙钟。GPU 占用小时、有效训练小时、CPU/等待小时分别记账，训练失败和恢复也不能删掉。截止由统一 UTC 时钟与外部 watchdog 监督，在截止前预留完整更新和保存时间；不能只依赖可能被 `min_steps/min_epochs` 优先级覆盖的 Trainer `max_time`，也不能让挂起的 DDP 无限等待。([Trainer 时间限制](https://lightning.ai/docs/pytorch/latest/core-api/trainer))

以真实保留自然音频小时 `D` 定义暴露；每次采样、DDP 补齐重复、恢复重放及失败尝试处理过的音频都计入资源总账。另记“已提交到当前 checkpoint”的暴露，不能用它掩盖已烧掉的计算。tiny/校准暴露记 `E_pilot`，主模型记 `E_main`；默认 50 上限仅约束主模型，**不宣称全部试验只有 50 次**。验证/测试读取另列，不算训练暴露。为严格执行 `E_main≤cap`，预算适配器须在下一次完整累积组进入计算前，汇总两 rank 的原始 manifest 时长并检查剩余额度；不足就不接纳该组并统一保存退出，允许少于整数 epoch。不能只在 optimizer step 后才发现越界，也不能把 padding 或两个重复 logger 算两次。该适配尚未实现，必须纳入 Day 2 验收。

```text
D = sum(保留训练条目的原始音频秒数) / 3600
E_main = 主 run 实际处理自然音频秒数 / (3600 × D)
E_project = E_main + E_pilot
S = 两 GPU 同一窗口实际处理的未 padding 自然音频秒数之和 / 墙钟秒数
T_main_days = D × E / (24 × S) × 1.3
T_total_days = T_main_days + 4
```

`S` 是**双卡合计**，分母是一段共同墙钟，不是两卡时间相加；包含读取、前反向、累积和同步。padding 不进入分子。若测的是稳定训练、不含定期 dev/存盘，就乘 1.3 预留 30% 工程开销；它是人为预算系数。若已用包含这些活动的端到端吞吐，则说明是否另留余量，不能不解释地重复扣 30%。校准预热约 30–50 更新，再用真实长短分布完成合计 200–500 optimizer updates，报告多个稳定窗口、低位吞吐、两卡显存、数据等待、full-dev 和保存耗时，不拿最快一批放行。

| 主暴露上限 | 假设 S（秒音频/秒墙钟，双卡） | 纯训练 | 加 30% 的主窗口 | 加四天后的总时长 |
|---|---:|---:|---:|---:|
| 50 | 250 | 192h / 8.0天 | 约 250h / 10.4天 | **约 14.4天** |
| 50 | 300 | 160h / 6.7天 | 208h / 8.7天 | **约 12.7天** |
| 60 | 250 | 230h / 9.6天 | 约 300h / 12.5天 | **约 16.5天，超预算** |
| 60 | 300 | 192h / 8.0天 | 约 250h / 10.4天 | **约 14.4天** |

以上仅用 `D=960h` 作情景计算，250/300 不是已测 H100 性能。十二天主窗口需要 `S_50≥960×50×1.3/288≈216.7`，60 次则 `S_60≥260`；达到边界几乎没有额外弹性。可先用约 230/280 作更保守的工程参考线，但最终以真实 `D`、低位稳定吞吐和剩余时钟为准。**默认 50；60 只可在 Day 2 主模型启动前选定**，同时冻结 LR horizon，不能 Day 12 因 WER 不够好临时加十次。

`281241/128×50≈109860` 是未过滤、固定 global128 的更新数近似，60 次约 131832；过滤、sampler 补齐和累积会改变它。不能把条数、epochs、自然音频暴露与 optimizer updates 混为一谈。若两天后 `S=180`，剩余 288h 只能预算约 41.5 次；`S=200` 约 46.2 次。执行前锁定的缩量公式为 `E_cap=min(50,H_remaining×S_conservative/(1.3×D))`，`H_remaining` 到 Day14 截止；实际不足就报告 partial。Day 2 数据/正确性门仍未通过则先记录 preflight 失败并重拟范围，不自行顺延十六天或让未就绪主训练开跑。

预算只回答“这些暴露能否塞进时钟”，不证明 50/60 次足够获得某个 WER。主训练中保持一个种子、一条固定配方；架构、词表、数据池或优化策略变更须保留原失败 run，另列新实验及其成本。性能下降时更新预计完成量并按截止保存，不靠改变模型规模或隐瞒等待维持“完成 50 次”的说法。

## 5. 每天产物和最后双测试共同决定交付状态

Day 2 冻结一个 tiny 子集（32–64 条训练句，关闭增强）、各 dev 约 200 条的固定 `dev_screen`，以及完整 `dev_joint`。screen 按说话人、长度、子集覆盖抽样，只作健康检查；**完整 dev-clean/dev-other 的总编辑错误除以总参考词数**决定 checkpoint，分数相同时选更早者。每约 5 次暴露做 full-dev、每天一次 screen，可作为初始频率；实际须由 Day 2 的 dev/I/O 耗时锁定并计入 30% 余量。每约 1–2h 保存 last，保留完整 dev 最佳与有限里程碑，临时写完校验再发布，不能删唯一可恢复文件。([NeMo WER 口径](https://github.com/NVIDIA-NeMo/Speech/blob/fd6a877539710e2b98f28c43272ff81312f83417/nemo/collections/asr/metrics/wer.py#L35)、[Experiment Manager](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/core/exp_manager.html))

| 日 / 连续时钟 | 当日输入与工作 | 必须保存的产物 | 继续、停止或调整门 |
|---|---|---|---|
| Day 1 / 0–24h | 官方源码、archive 与现场资源；建环境、校验下载、开始全量解码与 split 审计 | commit/lock、环境与资源清单、校验和、数据审计进度 | 容量/下载阻塞记账；不以仅 100h 小跑冒充全数据就绪 |
| Day 2 / 24–48h | 冻结数据/tokenizer；tiny、双卡 200–500 更新校准、真实恢复、dev 测时 | D/N/过滤表、tiny 预测、S/显存、恢复对照、唯一主配置及 U/W/cap | tiny WER≤5%、CER≤2%为诊断目标；配对、CTC、计分或恢复失败先修；未准备好则重拟范围，不顺延 |
| Day 3 / 48–72h | 通过全部门后，全新随机主模型启动 | step0 指纹、无载权重记录、首个完整 ckpt、暴露/LR/loss 台账 | 非有限梯度、非法 token、长度/计数错误立即停；不从 pilot 权重继续 |
| Day 4 / 72–96h | 唯一主 run；第一次固定 screen、低位 S 与剩余量预测 | screen 逐句预测、empty/blank、恢复文件索引 | 早期 blank 高结合 loss/非空输出趋势判断；吞吐下降修正预计暴露 |
| Day 5 / 96–120h | 按锁定里程碑执行第一轮 full-dev | 两个 dev 的 S/D/I/N、早期稳定 checkpoint 基准 | 数据泄漏或错误配对使 run 无效；不删难句修分数 |
| Day 6 / 120–144h | 持续同一配方；核对梯度、音频计数和恢复日志 | 每日 ckpt、曲线、E_main/E_project、GPU-hours | 只修基础设施/正确性缺陷并记影响，不做每日 LR 搜索 |
| Day 7 / 144–168h | 中期审计；比较已耗时与 Day14 可完成量 | 预算预测、替换/删除/插入样例、故障表 | 多次 dev 无改善先有时限诊断；持续故障侵占尾部窗口则提前停 |
| Day 8 / 168–192h | screen 与既定 full-dev；比较早期稳定 checkpoint | 全 dev 最佳指纹、相对 WER 改善、输出长度变化 | 单次 dev 波动不换 seed；查 loss 改善但解码不动的原因 |
| Day 9 / 192–216h | 样本覆盖、重复、1–3h 稳定吞吐与磁盘审计 | 自然音频总账、padding/I/O/显存、ckpt 空间清单 | 数据读取/磁盘风险优先处理；保留唯一可恢复状态 |
| Day 10 / 216–240h | 核对剩余窗口；开始汇总报告 | 当前 full-dev best/last、实际暴露和 partial 风险 | 余量不足只下调预计完成量，不隐藏失败或改变模型规模 |
| Day 11 / 240–264h | 固定最终解码/计分接口，错误分析只读 dev | 标准化版本、greedy 配置、报告 schema | test 继续封存；已达 cap 就停止，不利用空余自动加练 |
| Day 12 / 264–288h | 最后阶段候选 full-dev、核对停止条件 | 候选及哈希、cap/deadline 状态 | 50 不得临时改 60；预选 60 也受同一 Day14 截止 |
| Day 13 / 288–312h | 估算下次保存/full-dev 耗时，准备停止与导出 | 终止清单、恢复检查、候选索引 | 不开新种子、LM、TTS；恢复不跨 Day14 |
| Day 14 / 312–336h | 达暴露 cap 或本日末硬停，二者先到者 | last、完整资源/失败账本、最多三个 full-dev 候选 | 不等整数 epoch；实际 37.6 次就报告 37.6 次，不读 test 选模型 |
| Day 15 / 336–360h | 按预定 full-dev 规则确定唯一模型；完整双 test 最终计分 | 模型/tokenizer/解码哈希、两份逐句预测与计分表 | 失败句修读取/计分并留重跑原因；不因 WER 差调参再测 |
| Day 16 / 360–384h | 重新加载导出模型，用 dev 音频做单条推理 smoke、报告与公开材料审查 | 权重/配置清单、完整双测试、数据/时钟/暴露表、复现与局限 | 384h 到时交真实状态；缺测试/恢复证据即标未完成 |

tiny 最多约 2–6h 诊断，达到目标即停；它只检验实现能否学习训练句。主训练以 warmup 后第一个稳定 full-dev checkpoint 为早期参照，后续记录 `relative_improvement=(WER_early−WER_now)/WER_early`，并同时报两个子集与合并分数。预期后续多个固定 dev 显示可辨文字、非空率提高、相对 WER 下降；若 warmup 后连续两次 full-dev 均无改善且输出持续全空，先查 blank/token 索引、CTC 长度、音频配对与 LR，再决定提前停止。不能靠等待更多天代替诊断，也不把高 blank 帧率单独当故障。

最终模型、tokenizer、文本规则和 greedy 解码冻结后，**完整原始 test-clean/test-other 各进行一次最终评价**，无外部 LM、无 test 提示。每条保存原始/规范化 ref 与 hyp、S/D/I/N 和字符编辑分子分母；汇总 `WER=(S+D+I)/N_words`、CER、空句率、失败数、时长及长度分桶，不能平均逐句 WER。预期预测数必须分别为 2620/2939；长句降 batch 重试也不丢弃。缺失、无法解码或推理失败的条目保留空 hyp，按完整参考计删除错误并单独报告失败；若参考与计分本身无法完成，则标 `INCOMPLETE_16D`，不得删句后报完整分数。任何读取/计分正确性修复导致的重跑都保留旧版本、原因与新 hash，测试好坏不能重新触发训练或挑 checkpoint。Day 16 重载 smoke 只用 dev 音频，不通过再次测试挑模型。

下面是固定版官方评价入口的未来模板，使用导出的本次自训模型；other 用另一份完整 manifest 与输出文件。先在 dev 验证接口与文本规则；test 阶段只执行冻结协议。CER 从同一批已保存预测计算，避免再次解码产生不同候选。([官方 eval 接口](https://github.com/NVIDIA-NeMo/Speech/blob/fd6a877539710e2b98f28c43272ff81312f83417/examples/asr/speech_to_text_eval.py)、[BF16 推理字段](https://github.com/NVIDIA-NeMo/Speech/blob/fd6a877539710e2b98f28c43272ff81312f83417/examples/asr/transcribe_speech.py))

```bash
# 未来评价模板；变量、完整 manifest 与 best.nemo 目前尚未生成。
python "$NEMO_ROOT/examples/asr/speech_to_text_eval.py" \
  model_path="$RUN/best.nemo" pretrained_name=null \
  dataset_manifest="$DATA/test_clean_full.jsonl" \
  output_filename="$RUN/eval/test_clean.jsonl" \
  batch_size=8 amp=true amp_dtype=bfloat16 \
  ctc_decoding.strategy=greedy_batch use_cer=false \
  text_processing.do_lowercase=true text_processing.rm_punctuation=false \
  scores_per_sample=true
```

预先设置两个计分 sentinel：参考拷贝应为 0% WER，非空参考配空预测应为 100%；它们验证计分器，不是识别模型 baseline。随机初始化模型可在固定 screen 留初始 loss/WER，WER 超过 100% 时原样报告。可选预训练 inference comparator 必须独立命名，仅作测量对照，不能把其权重、特征、转写或伪标签带入 scratch 训练，也不能拿网上分数冒充本地结果。

建议首轮项目质量目标为 **完整 test-clean WER≤10%、test-other WER≤25%**，同时完整报告 CER 与错误分解；这两条阈值是负责人选定的项目验收线，**不是官方结果、不是 50 次暴露或十六天保证**。训练集 WER、screen 最佳句、clean 单一分数或加 LM 结果不能替代它。单种子和朗读语域的局限写入报告，不把两个测试集通过扩大成普遍可用。

| 判定维度 | 最终应填写的状态 | 所需证据 |
|---|---|---|
| 执行 | PLANNED / PREFLIGHT_PASSED / MAIN_RUNNING | 方案，或数据/tiny/吞吐/恢复门真实通过，或唯一主 run 在运行 |
| 主训练 | MAIN_STOPPED_FULL / MAIN_STOPPED_PARTIAL | 达锁定 cap，或因截止/故障停止；列真实 E_main 与 E_project |
| 交付 | DELIVERED_16D / INCOMPLETE_16D | 384h 内模型可重载、双测试与完整账本齐备，或明确列出缺项 |
| 质量 | NOT_EVALUATED / TARGET_MET / TARGET_MISSED / INVALID | 双测试未完、两项均达标、至少一项超标，或泄漏/初始化/计分错误使结论无效 |

本轮真实状态仍是 **PLANNED / NOT_EVALUATED**。未来最有价值的结论可以是“按时交付，但只完成 X 次且未达质量目标”，前提是保留正确分母和全部证据。提交代码、说明、配置与脱敏日志时审查公开范围；不把私有主机/IP、凭证、会话标识、个人路径、原始大音频或未经审核的权重推到公开仓库。十六天结束后，下一步是否增加训练、数据或转向流式，应由这次失败与误差证据重新立项。
