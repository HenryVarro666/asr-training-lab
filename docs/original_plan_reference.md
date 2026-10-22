# 基于 DeepInfer 多模态接口的小型流式 Speech-LLM（ASR/STT）训练计划书

> 目标硬件：2 × H100 80GB（单机）。日期：2026-10-22。
> 对标岗位：DeepSeek「Research Scientist - Speech LLMs」与「Applied Scientist: Speech & LLM Training Data」。

---

## 0. 一页结论（先看这个）

**DeepInfer 不能用来训练，也跑不在 H100 上。** 这不是推测，是读源码得出的结论（证据见第 1 节）：

1. DeepInfer 是纯推理引擎（vLLM 衍生），没有任何优化器、loss 或反向传播代码。
2. 硬件只支持 NVIDIA SM100（Blackwell：B200/GB200）和昇腾 950。核心路径直接 import FlashAttention-4 的 `flash_attn.cute`、DeepGEMM，并用 `is_sm_major_10()` 判断架构；依赖锁死在 `torch==2.13.0+cu132` + CUDA 13。H100 是 SM90，直接编译不过或者运行时断言失败。
3. 唯一注册的模型是 `DeepseekV41ForCausalLM`，也就是 DeepSeek-V4.1-Flash：552B MoE（prefill 激活 8B / decode 激活 16B）外加 196B Engram 条件记忆。官方文档写明最小高效部署单元是 64 卡 prefill + 64 卡 decode。两张 H100 共 160GB 显存，连权重都装不下。
4. 它的多模态只有图像（ViT + MLP projector），没有任何音频代码。

**所以本计划的定位是：**

- **训练**用自研 PyTorch 代码栈（这正是 JD 要求的"亲自编写代码并训练"），在 2×H100 上训练一个约 2B 参数的中英双语、支持流式的 Speech-LLM ASR。
- **DeepInfer 的用法有三档**（第 8 节）：
  - 档 1（必做，零成本）：模型的音频接入接口**逐一对齐 DeepInfer 的图像接入接口**（`SegmentInput` / `TokenType` / `downsample_mlp_gelu` projector / `replace_input_embeds` / 可学习特殊 token / 模态专属 MoE 路由偏置）。这样你的模型在结构上就是"DeepInfer 多模态栈的音频版"，面试时能逐行对照讲。
  - 档 2（推荐）：复刻 DeepInfer 的评测与 benchmark 方法论：golden-baseline 回归测试、batch-invariant 评测、encoder 单卡吞吐 sweep。
  - 档 3（可选，需租 SM100 卡约 8–16 GPU·小时）：给 DeepInfer 仓库写一个 audio 模态原型（processor + encoder + mapper），作为开源贡献 / 作品集。

**最终交付物：**
- 一个开源 repo：数据 pipeline + 训练 + 流式推理 + 评测，全部可复现。
- 两个模型：主模型约 2.0B（Whisper-medium 编码器 + Qwen3-1.7B）、小模型约 0.7B（Whisper-small 编码器 + Qwen3-0.6B）。
- 一份技术报告（WER/CER、延迟-精度曲线、数据清洗消融）。
- 可选：DeepInfer audio-VE 原型分支。

纯 GPU 时间约 3–5 天（含消融约 1 周），人力时间 6 周。

---

## 1. DeepInfer 可行性核查（证据表）

本地 clone 了 `deepseek-ai/DeepInfer`（main，2026-10-21）逐文件核对：

| 问题 | 结论 | 证据 |
|---|---|---|
| 能训练吗 | 不能。无 optimizer / loss / backward；所有前向都在 `torch.inference_mode()` 下 | README 自述 "inference framework"；`examples/_ve_helper.py` 的 `encode_images` 使用 `torch.inference_mode()`；全仓库无 `loss.backward` |
| 支持 H100 吗 | 不支持。只支持 SM100 与昇腾 950 | `README.md` "Supported hardware: NVIDIA SM100 GPUs / Huawei Ascend 950 NPUs"；`deepinfer/model_executor/models/vision_encoder/vit.py` 顶部 `from flash_attn.cute.interface import _flash_attn_fwd`（FA4，Blackwell 专属）、`import deep_gemm`、`from deep_cluster.hardware import is_sm_major_10`；`deep_cluster/hardware/device.py:160` `is_sm_major_10()` 判断 `capability[0] == 10`；`requirements-nvidia.txt`：`torch==2.13.0+cu132`、`flash-attn-4==4.0.0b34`；README 前置条件 CUDA 13 |
| 支持哪些模型 | 只有一个 | `deepinfer/model_executor/models/__init__.py` 注册表只有 `"DeepseekV41ForCausalLM": ("deepseek_vl", "DeepseekVLForCausalLM")`；`hugging_face/config.py` 强制 `model_type == "deepseek_v41"`，量化强制 `fp8` + 专家 `fp4` + `weight_block_size [32,32]` + `ue8m0`；`model_config_loader.py` 断言 `"vl_aligner" in model_cfg`（必须带视觉模块） |
| V4.1-Flash 多大 | 552B MoE（384 路由专家 + 1 共享，每 token 激活 6 个）+ 196B Engram；hidden 5120、40 层、vocab 129,280、1M 上下文 | HF `config.json`；DeepSeek 官方发布说明；`doc/inference_deployment_qa.md`："最小高效部署单元 64 卡 P + 64 卡 D" |
| 有音频吗 | 没有 | 全仓库 grep `audio|speech|asr|mel` 无命中；`ve/`（Vision Encoder）只有 `image_processor.py` |
| 多模态怎么接的 | ViT → 下采样 → 2 层 GELU MLP → 替换 placeholder token 的 embedding | `vision_encoder/vl_aligner.py` `VisionLanguageInstructTuner._encode_images`；`mlp_projector.py` 只支持 `downsample_mlp_gelu`；`multimodal_tokens.py` 的 `TokenType` / `ImageTokenOffset` |

**推论：** 想在 2×H100 上"用 DeepInfer 训 ASR"在工程上不成立。可成立的是"按 DeepInfer 的多模态设计范式训一个 Speech-LLM，并把 DeepInfer 的评测/部署思想复刻到 ASR 上"。

---

## 2. 项目定位与 JD 对齐

| JD 要求 | 本计划对应产出 |
|---|---|
| 亲自编写代码并训练大型 ASR/STT 模型 | 自研训练栈（非 icefall/wenet/NeMo 套壳）：数据加载、模型、loss、分布式、评测全部自写，约 5–8k 行 |
| 音频处理实战 | 自写 fbank 前端、SpecAugment、速度扰动、噪声/混响增强、VAD 切分、长音频流式窗口 |
| 流式（streaming）识别实战 | 编码器 chunk-causal 动态分块训练 + LLM 交错式读写策略，给出 WER–延迟曲线 |
| 覆盖数据、训练、评测全流程 | 第 4 / 5 / 7 节 |
| 大规模 LLM 训练数据 curation pipeline（清洗、去重、质量过滤、标注） | 第 4 节：含跨集泄漏检测、近重复去重、CTC/伪标签双重质量过滤、弱监督伪标注 |
| backend 工程、生产级系统 | 流式推理服务（vLLM 插件 + 自研流式 decoder）、golden 回归测试、batch-invariant 评测、吞吐 sweep |
| ASR/TTS 工程化训练（上线） | 第 6 / 8 节：服务化 + DeepInfer 接口对齐 |
| 顶会论文方向 | 第 5.5 节可选研究点（WER 奖励的 RL、流式策略、数据过滤消融）可以写成 Interspeech/ICASSP 投稿 |

---

## 3. 模型架构（对齐 DeepInfer 多模态接口）

### 3.1 整体结构

```
16kHz 波形
  └─> 80-dim log-mel（25ms 窗 / 10ms 帧移）                      [T×80, 100 Hz]
  └─> Audio Encoder（Whisper-medium 编码器初始化，chunk-causal）    [T/2×1024, 50 Hz]
        ├─> CTC head（辅助 loss + 对齐 + 数据过滤）
  └─> 帧堆叠下采样 ×4（相邻 4 帧 concat，50 Hz → 12.5 Hz，80 ms/token）[T/8×4096]
  └─> MLP projector：Linear(4096→2048) → GELU → Linear(2048→2048)   [T/8×2048]
  └─> 写入 LLM 输入 embedding 中 <audio> 占位 token 的位置
  └─> LLM：Qwen3-1.7B-Base（28 层，hidden 2048）自回归生成文本
```

30 秒音频 → 375 个音频 token，和 DeepInfer 中一张图最多 1024 个 image token 同一量级。

### 3.2 与 DeepInfer 图像通路的逐项对照（档 1 的核心）

| DeepInfer（图像） | 本计划（音频） | 说明 |
|---|---|---|
| `ve/processor/common.py`：`ImageInput(width, height, binary)` + `TextLengthInput(length)` 组成 `SegmentInput` | `AudioInput(sample_rate, num_samples, binary)` + `TextLengthInput` | 文本只传长度不传内容，token 本体在 engine 侧，这个契约照抄 |
| CPU worker 进程做 resize，产出裸 RGB bytes；GPU 侧 `rgb_pad_to_patches` 做 normalize/pad | CPU worker 解码到 16k PCM 并算 fbank；GPU 侧做 pad + batch | 把"CPU 预处理 / GPU 编码"边界切在同一位置 |
| `multimodal_tokens.py`：`ImageTokenOffset`（id = vocab_size + offset）、`TokenType` 2..6 | `AudioTokenOffset`：AUDIO_START=5, AUDIO=6, AUDIO_END=7；`TokenType` 扩展 7..9 | 特殊 token 不占词表，用 vocab_size 之后的 id |
| `vl_aligner.py`：`image_start/end/newline` 是 `nn.Parameter`，走 `replace_input_embeds` 写入 | `audio_start/audio_end` 同样是可学习参数 | 不是普通词表 token，是单独参数 |
| `batch_downsample(ratio=3)` → `MlpProjector(depth=2, GELU)` | 帧堆叠 ratio=4 → 同形 MLP | `hugging_face/config.py` `_build_vision` 里 projector 固定 depth=2 |
| `moe.py`：MoE gate 有独立的 `e_score_correction_bias_vl` 给图像 token；Engram 用 `images_seq_mask` 屏蔽图像 token | 若用 MoE 骨干（第 5.5 节），加 `bias_audio`；音频 token 不进任何 n-gram 记忆 | 模态专属路由偏置是 V4.1 的一个具体设计点 |
| `deepseek_vl.py`：`multi_modal_mapper`（processor 侧）+ `get_input_embeddings`（model 侧） | 训练代码的 `AudioLanguageInstructTuner` 保持同样两个入口 | 将来移植到 DeepInfer 只需要加一个 mapper 和一个 tuner 类 |
| `examples/_ve_helper.load_ve_only` + `sweep/ve_sweep.py`：只装编码器、单卡、无进程组跑吞吐曲线 | `tools/encoder_sweep.py` | 第 7.4 节 |
| 部署文档："VE 单卡部署，用便宜的卡" | 音频编码器同样可以 E/P/D 解耦，单独放在便宜卡上 | 写进技术报告的部署一节 |

### 3.3 各组件选型与理由

**音频编码器：Whisper-medium 编码器（24 层，d=1024，16 头，约 307M）。**
- 理由：预训练覆盖 68 万小时多语种，中英都强；307M 的体量在 2×H100 上可以全参微调。
- 要改三处才能流式（第 6 节）：Conv 下采样改因果 padding、正弦绝对位置编码换 RoPE（可做无限长流式）、注意力加 chunk-causal mask。这三处在 Stage 0 用 CTC 重新适配。
- 备选（从零自研路线）：Zipformer / Conformer-CTC 约 100–150M，从零训练。更"亲自编写"，但风险更高；作为 Plan B。

**Projector：帧堆叠 ×4 + 2 层 GELU MLP，约 12.6M 参数。** 消融 ratio ∈ {2, 4, 8}。

**LLM：Qwen3-1.7B-Base。** 小配置用 Qwen3-0.6B-Base。
- 可选"DeepSeek 亲和"骨干：DeepSeek-V2-Lite（MLA + DeepSeekMoE，16B 总参 / 2.4B 激活）。bf16 权重 32GB，用 LoRA（r=64，挂在 MLA 的 q/kv 低秩投影和共享专家上）+ 冻结路由专家，可以放进单卡 80GB。这是唯一开源的、带 MLA 的小 DeepSeek 模型，能让你实际处理 MLA 的 KV 压缩和 MoE 的模态路由偏置。放在 Stage 4（第 5.5 节）。

**Prompt 格式（Qwen chat template）：**
```
<|im_start|>user
<audio_start><audio>…<audio><audio_end>请把这段音频转写成文字。<|im_end|>
<|im_start|>assistant
{带标点、保留大小写的转写}<|im_end|>
```
输出保留标点和大小写（LLM 风格），评测时用标准化后的 WER/CER。

---

## 4. 数据 pipeline（对应 JD #6）

### 4.1 语料

**Phase A（快速迭代，约 3.2k 小时，全部免费直接下载）：**

| 语料 | 语种 | 时长 | 来源 | 说明 |
|---|---|---|---|---|
| LibriSpeech train-960 | EN 朗读 | 960 h | OpenSLR 12 | 约 60 GB flac |
| GigaSpeech M | EN 播客/有声书/YouTube | 1,000 h | 需在线签协议，通常当天过 | 若延迟则先用 MLS English 抽 1,000 h |
| AISHELL-1 train | ZH 朗读 | 150 h | OpenSLR 33 | |
| MAGICDATA | ZH 朗读 | 755 h | OpenSLR 68 | 非商用许可 |
| aidatatang_200zh | ZH 朗读 | 200 h | OpenSLR 62 | |
| ST-CMDS | ZH 朗读 | 100 h | OpenSLR 38 | |
| Common Voice 17 zh-CN / en（validated 子集各取部分） | ZH/EN | 约 100 h | Mozilla | 口音多样性 |

**Phase B（扩大规模，约 20k 小时）：**

| 语料 | 语种 | 时长 | 说明 |
|---|---|---|---|
| WenetSpeech L | ZH 多领域 | 10,000 h | 需签协议；opus 约 500 GB |
| GigaSpeech XL 或 MLS English | EN | 10,000 h | MLS 共 44k h 完全公开，抽 10k h |
| KeSpeech | ZH 8 种方言口音 | 1,542 h | 需申请；鲁棒性 |
| TALCS | 中英 code-switch | 587 h | 需申请 |
| Emilia（ZH 部分抽样） | ZH 野外伪标注 | 抽 2–5k h | CC-BY-NC；专门用来演示伪标签过滤 pipeline |

**增强素材：** MUSAN（噪声/音乐/人声，OpenSLR 17）、RIRS_NOISES（混响，OpenSLR 28）。

**评测集（全程只读，训练前做泄漏检测）：** LibriSpeech test-clean / test-other、GigaSpeech test、AISHELL-1 test、WenetSpeech test_net / test_meeting、FLEURS zh-CN / en、TALCS test、TED-LIUM 3 test（长音频）、ESC-50（非语音，测幻觉）。

### 4.2 Pipeline 步骤（每步一个 CLI，manifest 用 Lhotse CutSet，分片用 Lhotse shar / WebDataset）

1. **入库**：统一转 16 kHz 单声道；写 `recordings.jsonl` + `supervisions.jsonl`；记录来源、许可、原始采样率。
2. **文本标准化**：NFKC；繁转简（OpenCC）；全角转半角；中文数字规则统一（训练标签保留原文，另存一份标准化版本用于过滤和评测）；英文保留大小写与标点。
3. **基础过滤**：时长 0.3–30 s；字符/秒超出 [0.5, 25]（中文按字、英文按字母）剔除；空标签剔除；非目标语言字符比例 > 5% 剔除。
4. **去重**：
   - 精确：`sha1(音频 PCM)` 与 `sha1(标准化文本)` 双键。
   - 近重复：文本 MinHash-LSH（5-gram，Jaccard ≥ 0.9）；音频 Chromaprint 指纹（同一录音不同剪辑）。
   - **跨集泄漏**：训练集 vs 所有测试集，按文本 8-gram 命中与音频指纹双重检查，命中即删训练样本，并输出报告。
5. **质量过滤（两路打分，取并集）**：
   - 路 A：Stage 0 训练出的 CTC 编码器算每条样本的归一化 CTC loss（per frame），按来源分桶取 top 2% 剔除（标签错、音频损坏）。
   - 路 B：Whisper-large-v3 转写一遍，与参考标签算标准化 WER/CER；> 50% 剔除，20–50% 打"低置信"标签供课程学习降权。
   - 附加：削波比例 > 1%、估计 SNR < 0 dB、Whisper LID 与标注语种不一致 → 剔除。
6. **伪标注（Phase B 的 Emilia / 无标签数据）**：Whisper-large-v3 + 两个不同 ASR 系统投票；三者一致率（字级）≥ 90% 保留；逐条附置信度。这一步就是 JD 里"标注"环节的可复现实现。
7. **打包**：按时长桶（bucket）动态 batch；每个 shard 约 1 GB；分片内随机、分片间乱序。
8. **统计看板**：每个来源的时长分布、字符/秒分布、LID 分布、过滤各步剔除比例、dev 上分来源 WER。全部产出到一个 HTML 报告。

每一步输入/输出 manifest 都版本化（`data/manifests/v{n}/`），训练配置里引用版本号，保证可追溯。

### 4.3 存储预算

20k 小时 16 kHz 16-bit 约 2.3 TB 裸 PCM；按 flac/opus 分片存约 0.6–1 TB。建议机器挂 ≥ 4 TB NVMe。Phase A 约 300 GB。

---

## 5. 训练计划

### 5.1 算力预算（2×H100 80GB，bf16）

每 30 秒音频的训练 FLOPs 估算（6 × 参数 × token）：编码器 0.31B × 1500 帧 ≈ 2.8 TFLOP；LLM 1.7B × (375 音频 + 约 100 文本) ≈ 4.8 TFLOP；合计约 7.6 TFLOP / 30 s。按 2 卡有效 600 TFLOP/s（MFU 约 30%）算，约 80 条 30 秒样本/秒，即每小时墙钟时间可过约 2,400 小时音频；考虑数据加载和不等长 padding，保守按 1,200–1,800 h/h。

| 阶段 | 可训练参数 | 数据 | 轮数 | 预计 2×H100 时长 |
|---|---|---|---|---|
| S0 编码器流式适配 + CTC | 0.31B（编码器 + CTC head） | Phase A 3.2k h | 8 | 6–10 h |
| S1 Projector 对齐 | 12.6M + 2 个特殊 token | Phase A | 1 | 2–3 h |
| S2 全量 SFT | 2.0B | Phase A | 3 | 6–10 h |
| S3 流式交错 + 扩数据 | 2.0B | Phase B 约 20k h（混 Phase A） | 1.5 | 24–40 h |
| 消融（ratio / LoRA vs 全参 / 过滤策略 / 小模型） | — | — | — | 约 24 h |
| S4 可选（V2-Lite LoRA 或 RL） | 见 5.5 | — | — | 12–24 h |

合计约 3–5 天纯 GPU 时间。

### 5.2 显存方案

- 并行：2 卡 DDP 即可（2.0B 可训练参数全套 AdamW 状态约 32 GB/卡：权重 bf16 2B×2 + 梯度 2B×2 + fp32 master 2B×4 + Adam m/v 2B×8）。不需要 FSDP；保留 FSDP2 开关以便换更大骨干。
- 激活：梯度检查点全开（编码器每层 + LLM 每层）；fused CE（Liger-Kernel）避免 151k 词表的 logits 占显存。
- batch：按音频总时长组 batch，每卡 micro-batch 上限 240 s（例如 8×30 s 或 24×10 s），梯度累积 2 → 全局约 960 s 音频/步。Phase A 一轮约 12k 步。
- 注意力：编码器用 FlashAttention-2（Hopper）+ 自定义 chunk mask（用 `flex_attention` 或分块矩阵 mask）；LLM 用 FA2 varlen packing，多条短音频拼进同一序列（packing 时 attention mask 隔离）。

### 5.3 各阶段细节

**S0 编码器流式适配 + CTC（6–10 h）**
- 从 Whisper-medium 编码器加载权重；改动：两层 Conv1d 改左侧 padding（因果）；去掉 1500 位正弦绝对位置编码，换 RoPE；注意力改 chunk-causal（见第 6 节）。
- 头：CTC head（Linear 1024 → 词表），词表用 Qwen3 tokenizer 的 BPE 子集？不要，CTC 词表用一个独立的 6k 中英 sentencepiece（中文字 + 英文 BPE），与 LLM 解耦。
- 动态分块训练：每个 batch 随机选 chunk ∈ {16, 32, 64, 全上下文} 帧@50Hz（320 ms / 640 ms / 1.28 s / 非流式），各 25%；左上下文全部保留（RoPE 下流式时限制为最近 25 s）。
- 超参：AdamW(0.9, 0.98)，wd 0.01，峰值 lr 1e-4，warmup 2k 步，cosine，grad clip 1.0；SpecAugment（2 个频率 mask F=27，时间 mask 最多 5 个 T=50 帧，p=0.5）；速度扰动 0.9/1.0/1.1；噪声/混响各 p=0.3，SNR 均匀采样 [0, 20] dB。
- 产出：流式 CTC 模型（本身就是一个可独立评测的小型流式 ASR，作为 baseline 之一），以及第 4.2 节路 A 的质量打分器。

**S1 Projector 对齐（2–3 h）**
- 冻结编码器与 LLM；只训 projector + `audio_start/audio_end` 参数。
- loss：只在 assistant 文本 token 上算 CE。lr 1e-3，warmup 300，cosine，1 轮。
- 验收：LibriSpeech test-clean WER 应已 < 6%，否则说明 projector/token 布局有 bug。

**S2 全量 SFT（6–10 h）**
- 解冻全部。分组 lr：LLM 1e-5，编码器 2e-5，projector 1e-4；warmup 500，cosine，3 轮；CTC 辅助 loss 权重 0.3（保持编码器的对齐能力，后面流式交错要用）。
- 继续 S0 的动态分块（让 LLM 也习惯分块编码器的输出）。
- 数据混合：中英各 50%，每条样本以 p=0.1 把两条短音频拼成"多段音频"样本（练长上下文）。
- 对照组：LoRA（r=64，alpha=128，挂 q/k/v/o/gate/up/down）vs 全参；预期全参在 test-other / test_meeting 更好。

**S3 流式交错 + 扩数据（24–40 h）**
- 数据切到 Phase B（约 20k h）+ Phase A 重采样 30%。
- 50% 样本用第 6 节的交错格式，50% 用普通格式（保证非流式能力不退）。
- 加入 3% 非语音负样本（MUSAN 噪声/音乐片段、ESC-50 训练划分），标签为空串，压幻觉。
- 加入 5% 长音频样本（60–120 s，用 VAD 拼接 LibriSpeech 连续片段 / TED 长段），练 RoPE 外推。
- lr 比 S2 减半，1.5 轮。

### 5.4 小模型配置（并行跑，约 1/3 时间）

Whisper-small 编码器（12 层，d=768，88M）+ Qwen3-0.6B（hidden 1024）+ projector Linear(3072→1024)。总参约 0.7B。用于流式延迟 demo 和"数据过滤策略在小模型上更敏感"的消融。

### 5.5 Stage 4 可选研究点（选一个，每个 12–24 h）

1. **DeepSeek-V2-Lite 骨干（MLA + MoE）**：LoRA 微调；给 MoE gate 加音频 token 专属路由偏置 `bias_audio`（对应 V4.1 的 `e_score_correction_bias_vl`），统计音频 token 与文本 token 的专家分配分布。面试时可以讲 MLA 的 KV 压缩对流式 ASR 长上下文的意义。
2. **WER 奖励的 RL（GRPO / MWER）**：对 S3 模型采样 N=8 条假设，奖励 = −WER（标准化后），KL 约束 β=0.04；1.7B 在 2×H100 上完全可跑。预期 test-other / test_meeting 相对降 5–10%。
3. **数据过滤消融**：无过滤 / 只路 A / 只路 B / 并集，在 Phase A 上各训一个 S2，画 WER 曲线。这一条直接服务 JD #6 的"质量过滤"。

---

## 6. 流式识别方案（具体）

### 6.1 编码器侧：chunk-causal 增量编码

- chunk = 32 帧@50Hz = 640 ms（可配 16 帧 = 320 ms）。chunk 内双向注意力，chunk 间只看左边。左上下文最多 25 s（RoPE 下用相对位置，无需重算）。
- 因为第 i 个 chunk 的输出只依赖 ≤ i 的输入，编码器可以带 K/V cache 增量运行：新来 640 ms 音频 → 只算 32 帧的 Q，对 cache 里的 K/V 做注意力 → 输出 8 个 LLM token。
- 下采样 Conv 改因果后，没有右侧依赖；SpecAugment 的时间 mask 在训练时按 chunk 边界对齐，避免泄漏。

### 6.2 LLM 侧：交错式读写策略

把每条训练样本改写成音频块与文本块交错的序列：

```
<audio_start> [a_1 … a_8] <|txt|> 今天 <|wait|>
              [a_9 … a_16] <|txt|> 天气真 <|wait|>
              [a_17 … a_24] <|txt|> 不错 <|wait|>
              … <audio_end> <|txt|> 。<|im_end|>
```

- 词到 chunk 的分配：用 S0 的 CTC head 做强制对齐（`torchaudio.functional.forced_align`），词的结束时间落在第 i 个 chunk 且满足延迟 δ（默认 1 个 chunk，即允许 640 ms 前瞻）就分到第 i+δ 个文本块。
- loss 只算文本 token 与 `<|wait|>`；音频 token 位置不算 loss。
- 推理：每来一个 chunk，把 8 个音频 token 追加进 LLM 的 KV cache，自回归生成直到出 `<|wait|>`；VAD 判定句尾后追加 `<audio_end>`，生成到 `<|im_end|>`。LLM 的 KV cache 全程只追加不重算，流式开销就是每 chunk 一次 prefill 8 个 token + 若干 decode 步。
- 可调延迟：训练时 δ ∈ {0, 1, 2} 随机，推理时用一个控制 token `<|delay_k|>` 指定，得到同一个模型的多条 WER–延迟曲线。

### 6.3 Fallback 方案

若交错训练不稳定：用 S0 的 CTC 流式输出做实时中间结果，LLM 只在 VAD 句尾做一次非流式转写覆盖。延迟差一些但实现简单，一周内必能跑通。

### 6.4 流式指标

- 平均滞后（Average Lagging，AL）与 LAAL；首字延迟；句尾延迟（最后一个词出现时间 − 语音结束时间）。
- RTF（实时率）与每 chunk 的 P50/P99 计算耗时（单 H100、bf16、batch 1 与 batch 16）。
- 曲线：WER vs 平均滞后，chunk ∈ {320, 640, 1280 ms} × δ ∈ {0, 1, 2}。

---

## 7. 评测体系

### 7.1 指标与标准化

- 英文 WER：Whisper 的 EnglishTextNormalizer；中文 CER：去标点、繁转简、全角转半角、英文小写；code-switch：混合 MER（Mixed Error Rate，中文按字、英文按词）。
- 每个测试集报 greedy 与 beam=4 两组。

### 7.2 测试集与目标值（估计值，用来定"及格/良好"线，不是承诺）

| 测试集 | 及格（S2 非流式） | 良好（S3 非流式） | 流式 640 ms（S3） |
|---|---|---|---|
| LibriSpeech test-clean WER | ≤ 3.0 | ≤ 2.3 | ≤ 3.2 |
| LibriSpeech test-other WER | ≤ 7.0 | ≤ 5.0 | ≤ 7.0 |
| AISHELL-1 test CER | ≤ 5.0 | ≤ 3.8 | ≤ 5.0 |
| WenetSpeech test_net CER | — | ≤ 9.0 | ≤ 11.0 |
| WenetSpeech test_meeting CER | — | ≤ 14.0 | ≤ 17.0 |
| FLEURS zh / en | ≤ 6.0 / 6.0 | ≤ 4.5 / 4.5 | — |
| TALCS test MER | — | ≤ 10.0 | — |

Baseline 对照（同一套标准化脚本重新跑，不抄论文数字）：Whisper-large-v3-turbo、Whisper-medium、SenseVoice-Small、Paraformer-large、FireRedASR-AED-L、Qwen3-ASR 开源版（如可获取）、以及本项目 S0 的 CTC 模型。

### 7.3 鲁棒性与幻觉

- 噪声：test-clean 加 MUSAN 噪声，SNR ∈ {0, 5, 10, 20} dB。
- 混响：test-clean 卷 RIR。
- 非语音幻觉率：ESC-50 测试片段上非空输出的比例（目标 < 2%）。
- 长音频：TED-LIUM 3 test 整段（10–20 分钟）用 VAD 切分后 WER，以及 RoPE 外推不切分直接喂 120 s。

### 7.4 复刻 DeepInfer 的工程化评测（档 2）

- **Golden 回归**：参照 `tests/models/offline_decode_ppl.json` 的做法，固定 200 条 dev 样本的 greedy 输出与 loss 存成 `tests/golden/{model}.json`，CI 里逐条比对；数值变化需显式刷新并写明原因。
- **Batch-invariant 评测**：参照 `DEEPINFER_BATCH_INVARIANT=1` 的目标，验证 batch 1 / 8 / 32 下 greedy 转写逐字一致。做不到就排查（通常是 padding 位置进入了 attention 或 LayerNorm 的 reduce 顺序），并在报告里记录差异率。
- **编码器吞吐 sweep**：参照 `examples/sweep/ve_sweep.py`：只加载编码器、单卡、扫 batch ∈ {1, 4, 16, 64} × 时长 ∈ {5, 15, 30 s}，画"每秒处理音频秒数 vs 批大小"的 Pareto 图。对应 DeepInfer 部署文档里"VE 单卡部署"的论证方式。
- **确定性采样**：评测统一 `torch.use_deterministic_algorithms(True)` + 固定 seed，报告中注明。

---

## 8. 与 DeepInfer 的结合（三档）

### 档 1：接口对齐（必做）

训练仓库里 `model/audio_tuner.py` 的类与方法名、输入输出契约与 `vision_encoder/vl_aligner.py` 一一对应：

```python
class AudioLanguageInstructTuner(nn.Module):
    def __init__(self, cfg, embedding, audio_source): ...   # 接管 encoder / aligner / 特殊 token 参数
    def _encode_audio(self, feats, cu_seqlens, max_seqlen, num_downsampled) -> Tensor: ...  # encoder -> stack x4 -> aligner.layers
    def _merge_audio_embeds(self, input_embeds, audio_embeds, mm_token_type) -> Tensor: ...  # 原地替换 AUDIO 位置
    def forward(self, input_embeds, feats_raw, cu_feat_lens, ..., mm_token_type) -> Tensor: ...
```

权重命名也照 HF V4.1 的习惯：编码器 `audio.*`，projector `audio_aligner.w1 / w2`，特殊 token `audio_start / audio_end`。这样对照 `hugging_face/names.py` 的 `_VISION_PREFIX` / `_VISION_RENAME` 就能写出音频版映射表。

### 档 2：评测方法论复刻（推荐）

第 7.4 节的四项。

### 档 3：DeepInfer audio-VE 原型分支（可选）

需要租 1 张 SM100（B200 / GB200，市价每 GPU·小时几美元）约 8–16 小时，且 CUDA 13 + GCC 13 + Python 3.12 环境。范围严格限制在"编码器通路"，因为不存在听得懂音频的 V4.1-Flash，LLM 侧没法验证：

1. `deepinfer/ve/processor/audio_processor.py`：`AudioInput` → CPU worker 解码 + fbank；接入 `SegmentInput`；扩展 `ve/schema.py` 的 `MM_METADATA_KEYS`（`audio_feats_raw`, `cu_feat_lens`, `audio_starts`, `audio_lens`）。这部分纯 CPU，单元测试在任何机器上跑。
2. `deepinfer/multimodal_tokens.py`：加 `AudioTokenOffset` 与 `TokenType.AUDIO*`。
3. `deepinfer/model_executor/models/audio_encoder/`：把训练好的编码器移植到 DeepInfer 的算子栈（`deep_gemm.bf16_gemm_nt` + `flash_attn.cute` varlen + `tilelang_ops.norm`），参照 `vit.py` 的写法。
4. `deepinfer/model_executor/models/deepseek_al.py`：`DeepseekALForCausalLM(DeepseekVLForCausalLM)`，实现音频版 `multi_modal_mapper` 与 `get_input_embeddings`。
5. `examples/sweep/ae_sweep.py`：照 `ve_sweep.py` 写音频编码器吞吐 sweep，在租的 B200 上产出曲线；同一编码器在 H100（自研栈）与 B200（DeepInfer 算子栈）的吞吐对比写进报告。
6. 提交为 draft PR 或自己 fork 的分支，附设计文档。

---

## 9. 时间表（6 周）

| 周 | 内容 | 里程碑 |
|---|---|---|
| W1 | 环境；Phase A 下载/入库/清洗/去重/泄漏检测；评测框架 + 全部 baseline 重跑 | `data v1` 报告；baseline WER 表 |
| W2 | S0 编码器流式适配 + CTC；用 S0 跑路 A 质量过滤出 `data v2`；S1 | 流式 CTC 模型 WER/延迟；S1 通过验收 |
| W3 | S2 全参 vs LoRA；downsample ratio 消融；小模型配置 | 非流式主模型达"及格"线 |
| W4 | Phase B 申请/下载/过滤/伪标注；过滤策略消融 | `data v3`（约 20k h）+ 过滤消融图 |
| W5 | S3 流式交错训练；流式推理 decoder；WER–延迟曲线；vLLM 插件服务化 | 流式 demo + 曲线 |
| W6 | 档 1 接口对齐收尾；档 2 工程评测；（可选）档 3 原型；技术报告 + 开源 | 报告 + repo + 可选 PR |

---

## 10. 代码仓库结构、环境与关键命令

### 10.1 目录

```
speech-llm/
├── data/            # 4.2 节的每一步一个 CLI：ingest / normalize / filter / dedup / leak_check / score_ctc / score_whisper / pseudo_label / pack / report
├── frontend/        # fbank、SpecAugment、速度扰动、噪声/混响、VAD
├── model/
│   ├── encoder.py       # Whisper-medium 编码器改造：causal conv、RoPE、chunk mask、CTC head
│   ├── projector.py     # 帧堆叠 + MLP（downsample_mlp_gelu 同形）
│   ├── audio_tuner.py   # AudioLanguageInstructTuner（对齐 vl_aligner.py）
│   ├── tokens.py        # AudioTokenOffset / TokenType（对齐 multimodal_tokens.py）
│   └── speech_llm.py    # 组装 + get_input_embeddings + forward
├── train/           # 训练循环（DDP/FSDP2 开关）、分阶段冻结、分组 lr、packing、交错样本构造
├── streaming/       # 增量编码器 + LLM KV 追加式 decoder + VAD 端点
├── eval/            # WER/CER/MER、标准化、鲁棒性、幻觉、延迟、golden 回归、batch-invariant 测试
├── tools/           # encoder_sweep.py、forced_align.py、export_vllm.py
├── configs/         # s0_ctc.yaml s1_proj.yaml s2_sft.yaml s3_stream.yaml small_*.yaml
└── tests/
```

### 10.2 训练环境（注意：不要把 DeepInfer 的 requirements 装进训练环境，它要 CUDA 13 + FA4）

```
Python 3.11 / CUDA 12.8 / 驱动 ≥ 570
torch 2.9 (cu128)        flash-attn 2.8.x (Hopper)      transformers ≥ 4.57
peft                      liger-kernel                   lhotse ≥ 1.30
torchaudio                sentencepiece                  jiwer / opencc-python-reimplemented
whisper-normalizer        silero-vad                     wandb
```

### 10.3 关键命令（示意）

```bash
# 数据
python -m data.ingest   --src librispeech --out manifests/v1
python -m data.filter   --in manifests/v1 --out manifests/v1f --min-dur 0.3 --max-dur 30
python -m data.dedup    --in manifests/v1f --out manifests/v2 --text-minhash 0.9 --audio-fp
python -m data.leak_check --train manifests/v2 --tests manifests/test/* --report reports/leak_v2.html
python -m data.score_ctc  --ckpt ckpt/s0_ctc --in manifests/v2 --drop-top-pct 2 --out manifests/v3
python -m data.pack     --in manifests/v3 --shar out/shar_v3 --shard-gb 1

# 训练（2 卡）
torchrun --nproc_per_node 2 -m train.run configs/s0_ctc.yaml
torchrun --nproc_per_node 2 -m train.run configs/s1_proj.yaml  --init ckpt/s0_ctc
torchrun --nproc_per_node 2 -m train.run configs/s2_sft.yaml   --init ckpt/s1_proj
torchrun --nproc_per_node 2 -m train.run configs/s3_stream.yaml --init ckpt/s2_sft

# 评测
python -m eval.run --ckpt ckpt/s3 --sets librispeech_test_clean,aishell1_test,wenetspeech_test_net --beam 4
python -m eval.streaming --ckpt ckpt/s3 --chunk-ms 640 --delay 1 --sets librispeech_test_clean
python -m eval.batch_invariant --ckpt ckpt/s3 --batches 1,8,32
python -m tools.encoder_sweep --ckpt ckpt/s3 --batch 1,4,16,64 --dur 5,15,30
```

### 10.4 S2 配置示例

```yaml
# configs/s2_sft.yaml
data:
  shar: out/shar_v3
  max_duration_per_gpu: 240      # 秒
  grad_accum: 2
  mix: {zh: 0.5, en: 0.5}
  concat_prob: 0.1
  augment: {specaug: true, speed: [0.9, 1.0, 1.1], noise_p: 0.3, rir_p: 0.3, snr_db: [0, 20]}
model:
  encoder: {init: openai/whisper-medium, causal_conv: true, rope: true, chunk_frames: [16, 32, 64, -1]}
  projector: {downsample: 4, depth: 2, act: gelu}
  llm: {init: Qwen/Qwen3-1.7B-Base, finetune: full}   # 或 lora: {r: 64, alpha: 128}
  ctc_weight: 0.3
optim:
  name: adamw
  betas: [0.9, 0.98]
  weight_decay: 0.01
  lr: {llm: 1.0e-5, encoder: 2.0e-5, projector: 1.0e-4}
  warmup_steps: 500
  schedule: cosine
  epochs: 3
  clip_grad: 1.0
runtime:
  precision: bf16
  grad_checkpoint: true
  fused_ce: true
  ddp: true
```

---

## 11. 风险与对策

| 风险 | 概率 | 对策 |
|---|---|---|
| Whisper 编码器换 RoPE + 因果 conv 后 CTC 收敛差 | 中 | 先只加 chunk mask 保留绝对位置编码（30 s 窗口流式）做 S0'，RoPE 版本作为并行实验 |
| 交错式流式训练不稳定 | 中 | 第 6.3 节 fallback：CTC 实时结果 + LLM 句尾覆盖 |
| WenetSpeech / GigaSpeech 申请延迟 | 中 | Phase A 全是免登录下载；EN 用 MLS 替代；ZH 用 Emilia 抽样 + 伪标签 pipeline 顶上 |
| 空音频/噪声幻觉 | 高 | S3 加入 3% 非语音负样本；评测 ESC-50 幻觉率 |
| 词表 151k 的 CE 显存 | 低 | fused CE；或把 LLM 输出层算 loss 时分块 |
| 磁盘不够 | 中 | ≥ 4 TB NVMe；opus/flac 分片；Phase B 分批下载分批打包 |
| 档 3 找不到 SM100 卡 | 低 | 档 3 本来就是可选；CPU 侧 processor + 单元测试在 H100 机器上也能提交 |

---

## 12. 这份计划能在面试里讲的故事

1. "我读了 DeepInfer 源码，知道它为什么是推理框架、为什么只支持 SM100、V4.1-Flash 的多模态是怎么接进 LLM 的（ViT → 下采样 → 2 层 MLP → 替换 embedding，特殊 token 是单独参数，MoE 有图像专属路由偏置，Engram 屏蔽图像 token），然后按同样的范式做了音频版。"
2. "数据侧做了可复现的 curation pipeline：跨集泄漏检测、近重复去重、两路质量打分、投票式伪标注，并且用消融证明了过滤策略对 WER 的影响。"
3. "流式侧做了编码器增量编码 + LLM 交错读写，给出了 WER–延迟曲线，并且评测做到了 batch-invariant 和 golden 回归。"
4. "部署侧对标了 DeepInfer 的 VE/P/D 解耦思想，做了编码器单卡吞吐 sweep；（可选）给 DeepInfer 写了 audio 模态原型。"

---

## 13. 参考资料

- DeepInfer 仓库：https://github.com/deepseek-ai/DeepInfer （README、`doc/inference_deployment_qa.md`、`examples/README.md`）
- DeepInfer 发布推文：https://x.com/deepseek_ai/status/2112476496283775135
- DeepSeek-V4.1-Flash-1017 模型卡与 config.json：https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017
- DeepSeek V4.1 Flash 官方发布说明：https://www.deepseek.com/en/news/deepseek-v4-1-flash/
- NVIDIA 模型卡（V4.1-Flash 规模）：https://build.nvidia.com/deepseek-ai/deepseek-v4.1-flash/modelcard
- TNW 报道（DeepInfer 开源、硬件范围）：https://thenextweb.com/news/deepseek-deepinfer-open-source-inference-engine
- FlashMLA（SM90/SM100 说明）：https://github.com/deepseek-ai/FlashMLA
- CUTLASS Blackwell SM100 block-scaled MXFP4：https://docs.nvidia.com/cutlass/4.4.1/media/docs/cpp/blackwell_functionality.html
- DeepSeek-V4 技术报告（V4-Flash 284B/13B，用于对比）：https://arxiv.org/pdf/2606.19348
