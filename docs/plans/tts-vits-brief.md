# 用十六天验证从零 VITS 基线

**状态：方案，未执行；日期：2026-10-27。** [PDF 阅读版](./pdf/tts-vits-brief.pdf) · [推荐 ASR 主方案](./asr-fastconformer-115m-16day.md)。TTS 是另一项独立选择：两张 H100 的同一段十六天不能同时完整提供给 ASR 和 TTS。本方案只做英语、单音色、离线合成，目标是得到可审计、能试听的基线，不承诺人声级质量。所有兼容性改动、训练与评价适配器均待实现，本轮没有安装、下载或训练。

## 约二十四小时单音色数据足以限定问题

选用 **LJ Speech 1.1 + VITS**。数据包含 13,100 段、约 24 小时、一个说话人，采样率 22,050 Hz，来源是七本非虚构书籍；官方提供原文与规范化文本，并说明原始录音经过 MP3 编码。首选锁定 VITS 官方 filelists 的 **12,500 train / 100 validation / 500 test**，核验 ID/音频 hash 不交叉，列出规范化文本重复，不擅改官方 split；报告 loader 过滤后的真实条数和时长。保留原采样率，审计缺失、静音与幅度；测试只说明同一个声音的保留文本表现，不声称跨书籍、未见说话人或克隆能力。按整书分组是可选后续实验，需先重建可信来源映射，不能凭编号猜测。([数据官方说明](https://keithito.com/LJ-Speech-Dataset/)、[官方训练清单](https://github.com/jaywalnut310/vits/blob/main/filelists/ljs_audio_text_train_filelist.txt)、[验证清单](https://github.com/jaywalnut310/vits/blob/main/filelists/ljs_audio_text_val_filelist.txt)、[测试清单](https://github.com/jaywalnut310/vits/blob/main/filelists/ljs_audio_text_test_filelist.txt))

**VITS 所有可训练模块从随机权重开始**，包括文本/后验编码器、flow、时长预测器、波形生成器和判别器，不加载预训练声码器、codec、teacher 或神经 G2P。使用固定版本的 eSpeak 规则与词典生成音素，归档前端、符号表和文本哈希；推理只输入文本。VITS 的波形生成器采用 HiFi-GAN 结构并不要求载入其权重。参数量先称数千万量级，执行时分别计数 G、D 和实际推理路径。([VITS 论文](https://arxiv.org/html/2106.06103v1#S2.SS5)、[模型实现](https://github.com/jaywalnut310/vits/blob/main/models.py)、[前端](https://github.com/jaywalnut310/vits/blob/main/text/cleaners.py)、[eSpeak 机制](https://github.com/espeak-ng/espeak-ng/blob/master/docs/dictionary.md#translation-files))

## 先验证旧代码，再决定完整更新数

锁定官方代码 commit 和现代 H100 环境，先检查 Cython MAS、STFT、旧依赖、AMP 与双卡 DDP。官方训练器内部使用 `mp.spawn`，原端口 `80000` 无效，拟改为未占用的 `29521`；保留内部 spawn 时只启动一次 Python，不套 `torchrun`。必须新增显式 fresh/resume、`max_steps`、墙钟截止和完整状态保存，避免输出目录残留 G/D 导致自动续训。原检查点不足以证明精确恢复，需保存 scheduler、scaler、global step、各 rank RNG 和 sampler 状态，再做中断对照。([训练代码](https://github.com/jaywalnut310/vits/blob/main/train.py)、[依赖](https://github.com/jaywalnut310/vits/blob/main/requirements.txt)、[检查点](https://github.com/jaywalnut310/vits/blob/main/utils.py))

建议 **每卡 batch 32、两卡 global 64**，显存不足则每卡 16，重新核算预算。以 AdamW `lr=2e-4, betas=(0.8,0.99)`、8192-sample 片段和官方联合损失起步；FP16+scaler 先验收，敏感运算保留 FP32。官方按 epoch 衰减，batch 改变会改变同一步数的累计衰减，必须记录。一次更新包含完整 D 和 G 更新。原论文用四张 V100、每卡 64，训练 800k 步；本项目 **300k 只是待校准上限**，不能当作论文收敛点。([官方配置](https://github.com/jaywalnut310/vits/blob/main/configs/ljs_base.json)、[论文训练设置](https://arxiv.org/html/2106.06103v1#S3.SS3))

预热 50 个完整更新，再测 200–500 个更新，计入数据等待、通信、验证和保存。`可做步数=floor(可用训练秒数/实测秒每完整更新)`。十天训练窗口做 300k 步要求平均不超过 2.88 秒；另留 20% 机动则要求不超过 2.304 秒。实测不满足就降低 step cap，预算不足不自动续期。

| 时间 | 输入与产物 | 进入下一阶段的门槛 |
|---|---|---|
| Day 1–2 | 数据划分、前端、环境；32–64 条 tiny 过拟合；双卡校准与恢复记录 | 配对正确、更新有限且有效、可生成可理解的训练短句；恢复与 fresh 检查通过 |
| Day 3–12 | 新随机主 run；固定验证文本、对齐图、音频与 G/D 全状态 | 预先锁定 cap/deadline；无持续崩溃、严重漏读或静音；按验证集保存最佳模型 |
| Day 13–14 | 停止训练、冻结模型与合成参数；生成保留测试句 | 测试文本未用于选模，短/长句、数字和缩写均覆盖 |
| Day 15–16 | 盲听、自动评价、失败列表、音频画廊与预算报告 | 交付配置、哈希、实际 steps/GPU 小时及 WAV；不足如实记未收敛 |

## 用完整读句与盲听判断基线是否成立

外部固定 ASR 可辅助报告合成语音及对应自然录音的 WER/CER、漏词、重复和失败率，只用于测量，不进入训练；它的误差不能代表音质。随机排列自然录音和模型音频，统一播放响度，让英语熟练听者评价自然度与可懂度，报告实际人数、样本量和区间。人数不足就称内部试听，不宣称可靠 MOS。建议预先固定 100 条未见测试文本，目标为 **≥90% 可理解且完整读完，静音/循环/截断合计≤5%**；外部 ASR CER≤10% 仅作辅助指标，另检查爆音与漏词。这些是项目目标而非保证，未达标如实报告。([论文主观评价方法](https://arxiv.org/html/2106.06103v1#S4.SS1))

十六天硬上限包括下载、工程、等待和失败，最多占用 768 GPU 小时，实际占用另记；第十六天交付已有证据。若只给十天，压缩为两天准备、六天训练、两天评测，重新计算 cap，不能沿用十六天训练量。
