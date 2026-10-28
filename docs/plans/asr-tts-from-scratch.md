# 在双 H100 上从零训练 ASR 与 TTS

**状态：计划，尚未执行。核查日期：2026-10-27。** 建议主 ASR 为约 **0.61B FastConformer-CTC**，主 TTS 为 **335.8M F5 类声学生成器 + 随机训练 BigVGAN**。两条推理链均不加载预训练权重；ASR 先交付一条原生流式主模型，完整离线模型是额外预算。保守按 **2×80GB**、不保证 NVLink、两卡独占规划。以下时长全部来自明示的吞吐假设，没有本机测速证据；本轮没有连接 GPU、下载语料或启动训练。公开语音的上游分段/转录可能用过模型，因此“本实验全部参数随机”不等于“语料从未经过预训练工具”。

| 一页决策项 | 主线计划 | 限制与退路 |
|---|---|---|
| ASR 模型 | 24层、d=1024、16heads、FF=4的FastConformer-CTC，约0.61B | 115M/0.34B做工程门；1B另立预算，不用小模型成绩代替主模型 |
| ASR 数据/训练量 | MLS EN固定5000h+ES917.68h；最多20次自然暴露；过滤前118,353.6h是绝对上限 | B_eff=640s时约666K更新；真实音频计数控制停止，不把近似步数当精确epoch |
| ASR 两卡预算 | 一条主run假设S=100/250/500 audio-s/s，约64/26/13天 | 完整同预算offline对照另加约64/26/13天 |
| TTS 模型/数据 | F5 Base335.8M；EN clean245.07h+ES443.23h，合计688.30h | 审核EN other后最多998.38h；自训vocoder是全链从零必需部分 |
| TTS 更新/预算 | F5≤600K累计更新，约7–23天；BigVGAN≤500K，约2–11天 | tiny/工程run另计；达到上限仍失败不自动延长 |
| ASR 质量目标，未实测 | EN MLS test WER≤12%，ES≤18%，两语CER≤8%；流式RTF<0.5、p95稳定词延迟≤800ms | 对完整offline模型的WER增幅≤20%仅在额外等预算对照完成后验收 |
| TTS 质量目标，未实测 | CER≤5%、WER≤10%；MOS均值≥3.5、95%CI下界≥3.2；失败率<2%；暖RTF<0.5 | 数字是预注册项目目标，不是来源成绩或两卡质量保证 |
| 共同工程门 | tiny过拟合、200步稳态校准、保存后恢复、独立dev/test | 许可、初始化、监督、恢复或预算未过门便停止；CPU测试不能证明音质/识别质量 |

## 1. 固定随机初始化边界与有许可的数据配额

ASR 的编码器、CTC头、embedding全部随机；不接预训练wav2vec、Whisper、Qwen、外部LM或冻结语音前端。复用开源结构和确定性STFT/mel计算不违反这一边界。NeMo官方CTC入口构造模型后仍会调用 `maybe_init_from_pretrained_checkpoint`，所以必须在 resolved config 中关闭全部 `init_from_*` 来源，打印可训练模块、seed、参数量与step0 hash；看到“new model”不能替代初始化审计。([NeMo CTC入口](https://raw.githubusercontent.com/NVIDIA-NeMo/Speech/main/examples/asr/asr_ctc/speech_to_text_ctc_bpe.py))

TTS 的文本embedding、ConvNeXt、DiT和条件模块随机；参考声音直接转mel，参考文本手工提供。推理禁止自动外接ASR获取参考文本，也不用预训练speaker encoder、codec、文本LLM或神经G2P。全链从零还要求generator/discriminator均随机训练的vocoder；“随机F5+预训练Vocos/BigVGAN”只能作为额外诊断路线，不能冒充全链从零。F5代码MIT与公开checkpoint的CC-BY-NC许可不同，主线不下载或加载公开F5权重。([F5代码](https://github.com/SWivid/F5-TTS)、[公开权重卡](https://huggingface.co/SWivid/F5-TTS))

ASR主模型按官方relative-position attention估算，每层主要矩阵约24d²（两FFN16d²、Q/K/V/O4d²、位置投影1d²、卷积pointwise3d²），24×1024²×24约604M，加前端/CTC等约0.61B。1B选项可用32×1152、18heads，主要层约1.019B，含前端/输出约1.02–1.03B；缩小24×768配置约0.34B量级另加前端，所有实际参数均须实例化计数。该参数修正不改变下文独立假设的S吞吐情景，实际速度仍未测量。([NeMo attention源码](https://raw.githubusercontent.com/NVIDIA-NeMo/Speech/main/nemo/collections/asr/parts/submodules/multi_head_attention.py)、[Conformer模块](https://raw.githubusercontent.com/NVIDIA-NeMo/Speech/main/nemo/collections/asr/parts/submodules/conformer_modules.py))

F5论文给出的Small为158M，Base为335.8M；论文Base使用约95kh语音、8×A100和1.2M更新，不能将本方案约0.7–1kh数据训练称为复现其通用基础模型。0.5–1B TTS只作为容量扩展，不保证优于336M。([F5论文](https://arxiv.org/html/2410.06885v1)、[官方FastConformer流式结构](https://raw.githubusercontent.com/NVIDIA-NeMo/Speech/main/examples/asr/conf/fastconformer/cache_aware_streaming/fastconformer_ctc_bpe_streaming.yaml))

| 阶段/用途 | ASR train配额 | TTS train配额 | 数据来源与边界 |
|---|---|---|---|
| 排错 | 每语16–32条，约5–10min | 每语16–32条；另1–2个声音30–60min | 只取train，不用于泛化结论 |
| 工程小跑 | EN100h+ES100h | EN20h+ES20h | 固定ID，单独dev；与正式大run独立随机初始化 |
| 主线 | MLS EN5000h+ES917.68h=5917.68h | LibriTTS clean train245.07h+CML ES443.23h=688.30h | 官方split、CC-BY4.0，实际过滤后量待确认 |
| 扩展 | 1B ASR用EN10000h+ES917.68h | 通过审核再加EN other310.08h，最多998.38h | 重新冻结预算与清单，不直接扩大正式run |
| 独立test | MLS EN15.55h、ES10h及另建真实场景集 | 官方unseen-speaker test；另建seen-speaker文本holdout | 不训练、不调参、不学归一化或词表 |

MLS英语train共44,659.74h、西语917.68h，官方split说话人不重叠。英语5000h是本项目配额，不是现成独立下载包；LibriSpeech 960h可以作为较小ASR基线，不和MLS同源音频盲目相加。([MLS论文表2](https://arxiv.org/html/2012.03411v2)、[MLS发布页](https://www.openslr.org/94/)、[LibriSpeech](https://www.openslr.org/12))

**LibriTTS的585.80h包含dev/test，真正train是555.15h。** 其clean训练部分53.78+191.29=245.07h，other为310.08h，不能把子集名称100/360/500直接当LibriTTS小时。CML西语train是279.15+164.08=443.23h，dev5.74h、test4.83h；它保留MLS划分，并用预训练XLSR过滤转录。LibriTTS-R用了Miipher恢复，第一版优先原LibriTTS，保留上游处理说明。([LibriTTS论文表2](https://arxiv.org/html/1904.02882v1)、[OpenSLR60](https://www.openslr.org/60/)、[CML论文](https://arxiv.org/html/2306.10097v1)、[OpenSLR146](https://www.openslr.org/146/)、[LibriTTS-R](https://www.openslr.org/141))

按以下顺序准备数据，所有小时均以manifest实际解码时长重新统计：

1. 锁release、URL、许可、校验和；保留原始、可解码、过滤后hours/speaker/book/utterance数及删除原因。CC-BY数据保留署名、许可和改动说明，公开发布音频/权重另查适用条款。
2. ASR统一16kHz mono，TTS统一24kHz mono；保留可追踪原音频与派生记录。选择MLS Opus须注明有损压缩，同一对照内使用相同版本。
3. 先限制1–20s，检查静音、削波、空文本、截断与错位；长句只按已有对齐边界或人工核实切分，不能切音频后仍保留完整长文本。
4. 同时保存原文和规范化文本。ASR用NFC并冻结大小写/标点策略；TTS保留ñ、áéíóúü、¿¡、标点和数字原串，规则归一化另存版本。
5. 按speaker/book/utterance ID+seed的hash稳定抽样，保存精确清单。ASR自训约2048子词或train字符表；TTS只用train字符表，embedding随机，不沿用中文pinyin默认。
6. MLS、LibriSpeech、LibriTTS、CML有LibriVox亲缘，跨集核对speaker/book/wave hash。若自训ASR给TTS评分，先做跨任务共享test排除名单，去除对应speaker/book/音频；更推荐冻结的独立外部ASR作辅助测量并人工核验。

TTS采样先用自然语种比例，单speaker目标不超过该语训练暴露5%，超限是否调整由真实分布决定；50:50语种采样是独立消融，不能同时更换模型再归因。seen speaker的heldout文本和unseen speaker条件测试分开；参考3–10s录音与目标音频不重叠。真实麦克风场景计划另录EN/ES各100–200条、约1–2h，覆盖未见speaker、噪声、数字、专名与长短句，当前尚不存在。

存储必须先实测可用量。ASR16kHz PCM16每小时约115.2MB，5917.68h约682GB；TTS24kHz每小时172.8MB，998.38h约173GB。MLS官方英语Opus约651GB、西语14GB；英语FLAC约2.4TB，分卷需要拼接校验，不能把每个100GB卷当独立子集。原包、转码、TTS、临时缓存及checkpoint同时保留时，语音主线建议独占约1.8–2.2TB工作空间，1B路线至少2.5TB；不预缓存全量float32 mel，也不凭历史容量推断今天够用。([官方包与校验和](https://www.openslr.org/94/))

## 2. 以一条原生流式 ASR 为主线，先过拟合再扩大

默认训练 **一条0.61B原生流式FastConformer-CTC**，没有额外LM或标点恢复。CTC作为起步路线便于定位blank collapse；后续RNNT/AED需独立预算与lineage。官方cache-aware训练支持因果前端、因果卷积和有限attention；普通全上下文模型直接切块，在小chunk下不能保证同样精度。([NeMo模型说明](https://docs.nvidia.com/nemo-framework/user-guide/25.02/nemotoolkit/asr/models.html)、[Stateful Conformer论文](https://arxiv.org/abs/2312.17279))

主流式配置用10ms mel stride、8×下采样，编码步80ms；`att_context_size=[64,3]`表示4步chunk约320ms、名义最大右上下文240ms，仍须加取窗、缓冲、计算与解码延迟。`chunked_limited`要求left可被right+1整除，不能用不合法的`[70,3]`。因果下采样、`conv_context_size=causal`、layer norm、合适前处理normalize和cache必须一同核实；仅设attention参数不足以保证因果性。([官方流式配置](https://raw.githubusercontent.com/NVIDIA-NeMo/Speech/main/examples/asr/conf/fastconformer/cache_aware_streaming/fastconformer_ctc_bpe_streaming.yaml))

200h工程预算内，额外做一组同seed、同数据、同层宽深度的offline/streaming短程配对，检查流式退化和配置。**它不能替代完整等预算offline模型。** 若要最终声称“流式WER相对离线增加≤20%”，须再从随机起点训练完整offline对照，增加一整条主run成本。同一流式checkpoint切换context推理，只能给该checkpoint的推理曲线；从offline继续训练streaming须标流式续训，不能冒充独立随机对照。

| ASR阶段 | 预算上限，提前通过便停 | 拟议验收/动作 |
|---|---|---|
| G0数据/单步 | 每语32条逐条听；先2-step smoke | 0空文本/错位；CTC target及相邻重复token所需blank均可被时间步覆盖；finite loss、关键层梯度和参数变化非零 |
| G1 tiny | 115M/0.34B；32–64条；单卡batch8；≤5K更新或6h | 训练WER≤5%、CER≤2%；无持续全blank；关dropout/增强。失败就定位，不烧满上限等待奇迹 |
| G2工程 | EN100h+ES100h；每个配对run≤20K更新或48h | B_eff目标256s，约7.1次平均暴露；warmup约1000；dev持续下降、优于空串/常见词基线、恢复通过 |
| G3主线 | 0.61B；累计50K/200K/约666K审查 | 最多min(118,353.6h,20×过滤后冻结小时)真实音频暴露；EN test WER≤12%，ES≤18%，两语CER≤8%为项目目标 |
| G4流式 | 固定test、单流真实按墙钟送音频 | RTF<0.5，p95稳定词延迟≤800ms；完整offline对照若未做，相对退化项标未验收 |

正式run重新随机初始化，工程小跑不作为未披露的预训练。AdamW建议β=(0.9,0.98)、weight decay=1e-3、clip=1.0、bf16-mixed；先每卡microbatch8、累积4、名义global64条，按8–12s真实均长可能是512–768s/update，**不能直接等同640s**。按duration bucketing统计B_eff，OOM先降microbatch/最大时长再调积累。tiny通过后才加入弱SpecAugment/速度扰动，每次只改一项。

学习率可沿锁定官方NoamAnnealing：`lr=2.0`是scheduler系数，warmup10000，必须记实际参数更新LR；另一选择是整体替换为cosine、peak约2e-4。不得把Noam系数2.0当直接Adam LR，也不能同时混入两种scheduler。0.61B的典型FP32权重/梯度/Adam矩状态约9.8GB，1B约16GB，尚未含activation、CTC workspace和通信buffer；目标每卡peak reserved<70GB，长句、验证、保存均要测。先DDP，只有实测显存受限才比较FSDP，不假定分片更快。([PyTorch FSDP](https://docs.pytorch.org/docs/stable/fsdp.html))

**待实施、未验证的启动模板**如下。上游脚本路径来自官方仓库；本项目仍需新manifest、独立配置、实际音频停止counter与随机初始化审计。命令并不意味着当前仓库已存在可直接运行的配置；首次只做2步检查resolved config。

```bash
TASK_ROOT=/path/to/future/speech-project
SPEECH_SRC=/path/to/pinned/NVIDIA-NeMo-Speech

python "$SPEECH_SRC/examples/asr/asr_ctc/speech_to_text_ctc_bpe.py" \
  --config-path="$TASK_ROOT/configs" --config-name=asr_061b_stream_scratch \
  trainer.devices=2 trainer.accelerator=gpu trainer.precision=bf16-mixed \
  trainer.max_steps=2 trainer.accumulate_grad_batches=4 \
  model.encoder.n_layers=24 model.encoder.d_model=1024 \
  model.encoder.n_heads=16 'model.encoder.att_context_size=[64,3]' \
  model.train_ds.manifest_filepath="$TASK_ROOT/manifests/asr_train.jsonl" \
  model.validation_ds.manifest_filepath="$TASK_ROOT/manifests/asr_dev.jsonl"
# 正式max_steps须由pilot实际B_eff换算；另按真实音频累计量停止。
# resolved config所有预训练init入口为空，首次输出目录为空。
```

主manifest核心字段为`audio_filepath/duration/text`；另保留lang/speaker/book/utterance、split和来源hash。额外键是否被dataset接受须smoke-test。只锁定一个Speech commit、兼容CUDA/PyTorch/Lightning与依赖，不训练时跟main更新。

## 3. 将随机声码器与随机 TTS 分开验收再组合

选择F5类mel-flow主线可以避免预训练codec和speaker encoder。先训练自有vocoder，使GT mel可重建；再训练随机F5；最后比较“原声→自训vocoder重建→F5预测mel+自训vocoder”三组。若第一组重建已含嗡声、削波或不清晰，应先修vocoder；直接加大F5无法分离问题来源。

| TTS部分 | 固定起点 | 累计里程碑与门槛 |
|---|---|---|
| BigVGAN tiny | 16–32条，generator/discriminator全随机，空目录 | ≤10K更新或6h；GT mel重建大多数短句可听清；提前通过便停 |
| BigVGAN主run | 新随机run；24kHz/100mel/hop256；segment65536≈2.73s | 50K/200K/500K累计审查；达到上限仍失败即交付失败，不自动续到5M |
| F5 tiny | Small158M，32–64条，单卡batch8，关预训练vocoder日志 | ≤10K更新或6h；大多数训练短句读对、无两语系统性丢字；失败定位文本/mel/mask/时长 |
| F5工程 | EN20h+ES20h；固定每语20普通句+5数字专名句+5长句 | ≤30K更新或24h；B_eff目标102.4s，约21.3次平均暴露；warmup约1000 |
| F5主run | Base335.8M；新随机run；688.30h起步 | 100K/300K/600K累计审查，600K为上限，先通过或dev早停便结束 |

BigVGAN官方24kHz/100mel/hop256配置可作起点，先总batch4再校准到8–16；官方脚本会按GPU数分配总batch。最终总batch必须与500K步预算一起冻结；65536样本片段在24kHz约2.7307s，500K更新、总batch4/8/16分别约1517/3034/6068h裁剪音频暴露，改变batch必须重记成本和公平对照。建议先FP32确认GAN稳定，再单独检验bf16，不默认D/G全部loss都适合低精度。起点lr1e-4、β=(0.8,0.99)，其余取锁定配置。官方v2终版训练5M步，本项目500K是首轮预算，不能称复现终版。([BigVGAN训练源码](https://raw.githubusercontent.com/NVIDIA/BigVGAN/main/train.py)、[24kHz配置](https://raw.githubusercontent.com/NVIDIA/BigVGAN/main/configs/bigvgan_v2_24khz_100band_256x.json)、[官方README](https://raw.githubusercontent.com/NVIDIA/BigVGAN/main/README.md))

F5的mel建议24kHz、100通道、hop256、FFT/window1024，字符表包含英西NFC字符。9600frames/卡microbatch约102.4s，两卡×积累2约409.6s/update，这是frame预算，**不是9600条音频**。按实际长度和padding记录真实暴露量；600K约68,267h暴露，对688.30h约99次、998.38h约68次，不能用大量重复掩盖unique数据有限。([F5 Base配置](https://raw.githubusercontent.com/SWivid/F5-TTS/main/src/f5_tts/configs/F5TTS_Base.yaml))

以下为**新配置的差异片段模板，不是完整可运行YAML**；需合并锁定官方完整配置，并实现按optimizer update停止、独立dev generation/evaluation和分层sampler。F5的内部scheduler含world-size处理，必须打印实际warmup步数，不能机械乘除两卡。

```yaml
datasets:
  name: ENES_Scratch
  batch_size_per_gpu: 9600
  batch_size_type: frame
  max_samples: 32
optim:
  learning_rate: 0.000075
  num_warmup_updates: 10000
  grad_accumulation_steps: 2
  max_grad_norm: 1.0
  bnb_optimizer: false
model:
  name: F5TTS_ENES_Scratch
  tokenizer: custom
  tokenizer_path: /path/to/train_only/vocab.txt
  backbone: DiT
  arch:
    dim: 1024
    depth: 22
    heads: 16
    ff_mult: 2
    text_dim: 512
    conv_layers: 4
    attn_backend: torch
    checkpoint_activations: true
  mel_spec:
    target_sample_rate: 24000
    n_mel_channels: 100
    hop_length: 256
    win_length: 1024
    n_fft: 1024
    mel_spec_type: bigvgan
ckpts:
  log_samples: false
  save_dir: /path/to/new_empty/scratch_run
```

F5通用`train.py`创建新backbone后，Trainer仍会在输出目录自动选择`model_last`、`model_*`或`pretrained_*`恢复；必须使用全新空目录，构造前设置Python/NumPy/torch/CUDA seed，并保存step0 hash。`resumable_with_seed`只管部分数据shuffle，不能替代模型初始化seed。`log_samples=false`避免默认音频日志路径偷偷加载预训练vocoder，试听改用显式自训vocoder适配器。([F5 train.py](https://raw.githubusercontent.com/SWivid/F5-TTS/main/src/f5_tts/train/train.py)、[Trainer源码](https://raw.githubusercontent.com/SWivid/F5-TTS/main/src/f5_tts/model/trainer.py))

F5自定义CSV为`audio_file|text`，用绝对音频路径；speaker/lang/split放外部manifest，新分层sampler另行实现。使用通用训练入口，不能将finetune入口改个名字就称scratch。默认loop没有本方案完整独立dev验收，需要新增hook或离线脚本；不能用train flow loss最小替代生成质量。随机t/noise使train loss不严格单调，固定noise/t/mask的dev loss与实际发音都要看。([官方训练说明](https://raw.githubusercontent.com/SWivid/F5-TTS/main/src/f5_tts/train/README.md))

自训BigVGAN装入F5推理必须本地构造+本地state dict，不调用`from_pretrained`。两端mel的采样率、频带、fmin/fmax、window、center/padding、log clamp必须相同，用同一wav做数值一致性检查；仅同为100维不够。原HiFi-GAN默认22.05kHz/80mel，不能不经改配便接本项目24kHz/100mel。BigVGAN当前CLI没有`max_steps`，后续需增加step-budget保存退出逻辑，不能编一个现成参数或把epochs当updates。([HiFi-GAN原始配置](https://raw.githubusercontent.com/jik876/hifi-gan/master/config_v1.json)、[BigVGAN训练接口](https://raw.githubusercontent.com/NVIDIA/BigVGAN/main/train.py))

## 4. 让恢复、独立测试和人工评审成为交付的一部分

三种训练器都先在独占两卡上做50步预热+200步测量；至少100–500步覆盖稳态、长句峰值与checkpoint。整个optimizer update计时包含全部累积microstep、D/G优化、通信和数据等待，显式CUDA同步。记录p50/p90、每卡allocated/reserved、实际音频秒、数据等待、checkpoint耗时和RAM/I/O；不能用单microstep或GPU利用率代替吞吐。H100产品规格不证明现场NVLink或有效吞吐。([NVIDIA H100规格](https://www.nvidia.com/en-us/data-center/h100/))

恢复状态需包括模型、optimizer、scheduler、EMA、global step、Python/NumPy/CPU/CUDA RNG、sampler seed、epoch、batch cursor和数据hash。F5现有checkpoint并不保存全部RNG；BigVGAN保存epoch/steps但恢复loop可能重跑保存所在epoch。必须补RNG与sampler/cursor适配，并做“连续跑→保存退出→干净进程恢复20–50步”对照，报告样本ID、LR、loss/输出差异；不能声称上游默认支持精确恢复。([F5 Trainer](https://raw.githubusercontent.com/SWivid/F5-TTS/main/src/f5_tts/model/trainer.py)、[BigVGAN源码](https://raw.githubusercontent.com/NVIDIA/BigVGAN/main/train.py))

每50步记loss、实际LR、grad norm、参数更新范数、blank/输出长度、音频秒与耗时；每500步固定train解码；每2K–5K更新跑固定分层dev，阶段尾全dev。每30–60min或2K–5K更新先到者完整保存，临时写完校验再原子发布，按固定轮换上限保留latest/best/里程碑，不删唯一可恢复状态。记录代码commit、依赖lock、容器digest、完整resolved config、词表/manifest/hash、seed和评价版本。单seed实验直接说明限制，不能当多seed统计。

ASR每条保存ref/hyp、S/D/I/N、CER分子分母、空输出率、blank率，逐语言/speaker/长度报告区间。dev选checkpoint和解码；test只在模型、归一化、beam冻结后使用。流式稳定词延迟为`t_stable−t_audio_end`，报告p50/p95；没有词级对齐就人工标100段或仅报segment finalization delay。first partial、final delay、endpoint超时、revision次数和compute-only RTF分开；batch1真实墙钟送入和离线批量吞吐分开，至少100条、每点重复3次，冷/暖启动分别记录。

TTS冻结每语言100普通句、50难句、20长段落，覆盖数字、专名、缩写、短句、长句及seen/unseen speaker。参考3–10秒且不含目标文本，正常推理不能使用真实目标duration；oracle duration另组报告。预训练ASR/speaker encoder可以作明确标注的外部测量仪器，不能成为训练损失或隐藏推理依赖；原声也跑相同测量，建立其误差下限，并保留人工判断。

| TTS验收项 | 拟议目标，全部未实测 | 设计与限制 |
|---|---|---|
| 内容 | CER≤5%、WER≤10%；相对原声ASR的CER增加≤3个百分点 | 每语至少100条人工核对，尤其数字/名字；外部ASR错误不能当TTS真值 |
| GT mel重建 | 每语100条dev重建中≥95%人工听清、无系统嗡声/削波；相对原声ASR CER增量≤1个百分点；另报STFT/mel误差 | 先验收vocoder，再判断F5；不凭单一GAN loss放行 |
| 自然度 | MOS均值≥3.5、95%CI下界≥3.2；较冻结简单基线落后不超过0.3 | 每语至少10位母语听者、每系统50平衡句、每句≥5人；原声/重建作盲评锚点 |
| 稳定性 | ≥1000个冻结prompt的静音/重复/截断/漏句/拒绝总失败率<2% | 各类分别报率；每语另人工检查10长段落 |
| 性能 | H100 batch1暖RTF<0.5 | 端到端含文本、声学、vocoder、同步；NFE8/16/32分别报质量与速度 |
| 声音提示泛化 | 独立同reference盲配对相似度 | seen/unseen分开；不设跨encoder通用cosine阈值，不把固定speaker系统称零样本成功 |

简单基线预先固定为158M/40h工程run配合同一自训vocoder；其数据和更新数与主线不同，只作任务参照，不据分差归因参数规模。

MOS按speaker/utterance与rater分层或聚类bootstrap，不能把同一句多人评分当完全独立样本扩大精度。记录招募、耳机/音量条件、样本量和区间算法。人类听评尚未发生，不能以UTMOS或LLM分数替代并称人工MOS。上述阈值须在看test前冻结，未达标便报告“训练完成，质量未通过”，不修改测试集使其过线。

**流式TTS另立工作包。** F5普通推理看完整文本和mel序列，按句合成再分包最多叫分段生成/传输。主线报告“收到完整当前句→第一可播放真实音频”的TTFA；若整句生成完才能播，TTFA就是整句生成耗时。原生流式需要有限右上下文/因果或chunkwise训练、增量vocoder、同训练推理mask/cache、文本lookahead以及chunk边界和underrun验收，未计入本预算，不能在for-loop加chunk_size便宣布完成。

出现NaN/Inf、非法CTC target、泄漏、错误初始化来源、损坏checkpoint或磁盘低于两份完整状态+临时写入余量时立即停止。tiny达到时间/更新上限未通过便交付诊断；主线dev连续3次明显恶化则恢复best并检查数据/采样，不自动无限重启或追加epoch。程序exit=0、loss下降、CPU单元测试通过和存出checkpoint都只是工程证据，不等于识别或合成质量成功。

## 5. 把语音与翻译放进同一份两卡总日程

ASR令`B_eff`为一次optimizer update全部rank/microstep的真实音频秒，`S=真实音频秒/墙钟秒`。约更新数为`U=3600×D_hours×E/B_eff`，日历为`D_hours×E/(24×S)×overhead`；先取overhead=1.30，校准后用实测保存/评测开销替换。主线5917.68h×20=118,353.6h；B_eff640s时约665,739更新，50K/200K检查约8,889/35,556h暴露。过滤后冻结量为D_clean时，正式暴露上限取min(118,353.6h,20×D_clean)，不能为凑过滤前小时数增加真实epoch。按实际B_eff换算max_steps并以真实累计音频停止，速度扰动与过采样暴露也单列记账。

| 一次双卡run | 快情景假设 | 中情景假设 | 慢情景假设 | 含30%开销的日历 |
|---|---|---|---|---|
| 主ASR0.61B，5917.68h×20 | S=500 | S=250 | S=100 | 约13/26/64天 |
| 可选ASR1B，10917.68h×20 | S=250 | S=125 | S=50 | 约47/95/237天 |
| F5 Base，600K更新 | 0.8s/update | 1.5s/update | 2.5s/update | 约7.2/13.5/22.6天 |
| 随机BigVGAN，500K更新 | 0.3s/update | 0.8s/update | 1.5s/update | 约2.3/6.0/11.3天 |
| **默认语音主训练合计** | **13+7.2+2.3** | **26+13.5+6.0** | **64+22.6+11.3** | **约22.5/45.5/97.9天** |

**这些速度是算术情景，未测量，也不是H100理论峰值换算。** 若B_eff改变，update耗时和暴露预算同时重算。GPU-hours=双卡wall-hours×2；每天只开放12h时，日历约翻倍，GPU-hours不减。1B约1.228M更新，仅中情景ASR就约95天；0.5–1B TTS容量模型若仍600K、假设2/3.5/5s，则另需约18/32/45天，不是免费替换。

语音主线硬上限建议：单次0.61B ASR **min(118,353.6h,20×D_clean)暴露或70双卡wall-days**，F5 **600K或25天**，BigVGAN **500K或12天**，先到者停。校准若预测超限，启动前缩为0.34B ASR或158M TTS、缩曝光并重新标明探索实验；不得启动后自动延长。tiny、工程配对、三项校准合计另留 **最多7个双卡wall-days**：ASR tiny≤6h、两个工程run各≤48h，F5 tiny≤6h/工程≤24h，vocoder tiny≤6h，校准每项≤6h，提前过门便停止；失败预算也记入。数据审核、人工评审和报告另排 **3–7日历天**，招募/下载若更慢须公开顺延，不能压进主训练时长冒充已包含。

三项任务共享同两卡，建议顺序为：数据/许可与所有tiny门→ASR主线→随机vocoder→F5主线→翻译主线→封存测试。CPU数据处理可以穿插，但未经测量不抵扣日历。以下把[翻译方案](./translation-llm-from-scratch.md)的预算一起相加，先导与评测另列：

| 总方案 | 语音主训练 | 翻译主训练 | 额外预算 | 合计示例/范围 |
|---|---:|---:|---|---|
| 较短路线：1.15B翻译 | 22.5–97.9天 | 约7–23天 | 翻译pilot/基线/消融≤主线15%；语音pilot≤7天；人工/整理3–7天 | 约40–139天情景；中档45.5+11+1.65+7+3–7≈68–72天 |
| 3B主路线 | 22.5–97.9天 | 原估45–146天，只有预测≤120天才启动完整60B | 同上；翻译额外按45–120天的15%预留 | 获准情景约84–250天；中档45.5+70+10.5+7+3–7≈136–140天 |
| 加完整offline ASR | 在上述方案上再加一整条0.61B同预算run | 不变 | 对应恢复/测试也记账 | 额外约13/26/64天；中档3B总计约162–166天 |

范围用于排程，不能解释为最晚交付承诺。3B慢情景146天触发其120天开跑前门槛，须改30B探索或1.15B，不能照抄146天后仍说满足硬预算。各短程run未必用满时间上限；若质量门失败，停止会缩短实际成本，但不会得到一个成功模型。若只留一个月，应先选一条主任务或缩小所有范围，不承诺三项完整大模型同时交付。

语音工作区1.8–2.2TB与翻译约0.5–1TB若同时保留，合计约2.3–3.2TB，另看checkpoint轮换临时余量；串行训练不自动释放数据。需先做公开许可和本地容量清单，计划可恢复的归档，不擅自删除现有数据。下一步工程交付是锁版本、manifest与过滤统计、随机初始化证据、mel兼容检查、seed/RNG/cursor适配、budget-stop hook、两卡校准/恢复报告及冻结评测协议。本文件只交付计划，训练和结果仍未发生。

阅读版：[ASR 与 TTS 方案 PDF](./pdf/asr-tts-from-scratch.pdf)。
