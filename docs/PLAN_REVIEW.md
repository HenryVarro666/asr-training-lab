# 原方案审阅记录

审阅：Astra 6；日期：2026-10-22。替代执行方案见 [ASR_TRAINING_PLAN.md](../ASR_TRAINING_PLAN.md)。本记录不代表任何训练指标已实现。

## 核心结论

原方案对 DeepInfer 的主要背景并非虚构：当前官方仓库、唯一模型注册、依赖版本、支持硬件和“高效 64P+64D”表述均找到直接证据。但整体方案把数据准备、预训练结构改造、CTC 对齐、speech LLM 融合、流式状态机、大规模训练和部署兼容绑定成一次首跑，缺少可隔离的正确性检查和实测预算。

用户新增现代架构与两卡效率要求后，新的顺序是：**Whisper-small 工程对照 → 已预训练 Qwen3-ASR-1.7B 两卡100h SFT → 960h/中文与数据消融 → 独立流式验收 → 按实测需要做后端适配**。不再把自研Whisper→Qwen projector作为主线必经阶段。Qwen官方训练脚本已存在，但需修复padding mask等边界；详见主方案第11节。

## 逐项修订

| 原方案问题 | 判断 | 采用的修订 |
|---|---|---|
| 默认 2×H100 80GB、至少 4TB 可用 | 与主机实测不符 | 2×H100 NVL 95,830MiB；GPU0已有进程；初查/data约610GiB，用户后指定/mnt/raid实测约3.3T可用；统一/mnt/raid/asr-lab。Qwen恢复checkpoint扩大预算：当前campaign300GiB、挂载盘500GiB保留，替代初始120GB/200GB |
| “DeepInfer 不支持 H100，所以直接编译不过” | 前半有官方支持矩阵；后半不是本项目实测 | 写明“非官方支持目标”，不伪称已经做过编译验证 |
| “全仓库无 audio/speech/asr/mel 命中” | 本次仅验证路径无 audio/speech/asr | 限定证据范围；不重复未经执行的全文 grep 结论 |
| “64P+64D”当作绝对最小部署要求 | 文档区分可部署与高效配置 | 保留区别；本项目两卡 ASR 不依赖此后端 |
| 复制多模态接口必做、即兼容 | 不成立 | 先形成自己的清楚输入输出契约；真实转换和数值测试后才能称兼容 |
| 用代码行数和不使用成熟训练库证明能力 | 无效工程指标 | 能力证据用可复现数据、分布式正确性、完整训练、对照评测和可恢复运行 |
| 同时改 Whisper causal conv、RoPE、chunk attention | 风险高且难归因 | 首跑保留原架构；未来一次一个因素；做未来扰动与cache一致性检查 |
| 首跑 3.2k 小时、多源增强与过滤 | 数据授权/带宽/标准化复杂度过高 | 第一轮只使用公开 LibriSpeech train-clean-100 和独立 dev |
| 阶段A称全部免费直接下载，但表中有签协议 | 自相矛盾 | 逐来源核对访问、许可和可分发范围；申请型数据不占首跑关键路径 |
| 过滤阈值和文本去重一刀切 | 容易删掉口音、短句、姓名和code-switch | 保留原文；精确音频冲突硬检查，文本相似仅作候选；记录拒绝原因和分布 |
| 教师WER高即标签坏，多模型投票即真值 | 偏差未经验证 | 教师分歧是线索，抽样审阅；等数据量/等计算预算随机对照 |
| 用 test-clean <6% 验收 projector | test污染且目标无实验依据 | dev选择，test封存；先验证声学输入有效性和可过拟合性 |
| 3–5天总GPU时间与RL提升5–10% | 未测吞吐和收益 | 删除承诺；测稳态audio-seconds/s、评测耗时、coverage后更新ETA |
| DDP显存公式足以保证能跑 | 忽略现有占用、activation、logits与workspace | 从低microbatch实测，按GPU0约束，保护已有进程 |
| 任意 chunk mask 与 FA2 自动兼容 | 不能假定 | 首跑SDPA原模型；未来验证后端对实际mask的支持，禁止静默fallback改变语义 |
| Frozen LLM即可no_grad | 会切断projector梯度 | 冻结权重但保留对音频embedding的反向传播 |
| Base 模型直接套chat模板 | 模板存在不等于对话能力 | 明确纯文本prefix或验证实际模板；新token注册、embedding扩展、loss mask单独测试 |
| `vocab_size+offset`照搬到普通HF输入 | 可能越界 | 注册token或走受控inputs_embeds；不直接查越界embedding |
| 只关注encoder chunk attention的因果性 | 前端也可能看未来 | 审计STFT、整段mel归一化、卷积、VAD、resampler；做未来扰动测试 |
| KV cache只追加就自然等价 | 仅在过去表示不变时成立 | 比较full recompute与逐块cache，记录lookahead和状态机 |
| batch输出必须逐字完全一致否则就是bug | 浮点和解码边界也会影响 | 固定环境和容差，记录差异率、WER影响与原因，不提前保证bitwise等价 |
| vLLM插件、DeepInfer原型、服务上线统称交付 | 各有独立接口和运行证据 | 训练导出、推理demo、后端支持、生产部署分别验收 |

## 本次核查来源与范围

DeepInfer 固定提交为 `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`。读取了 README、模型注册表、`requirements-nvidia.txt`、部署 Q&A、vision encoder 部分源码和完整 Git tree；没有宣称全仓库语义审计。原计划的大模型具体参数量、每个私有/申请型数据集大小、所有第三方模型排名不影响首跑，因此未逐一重新认证，执行方案不依赖这些数字。

- [固定 README](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/README.md)
- [固定注册表](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/__init__.py)
- [固定依赖](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/requirements-nvidia.txt)
- [固定部署文档](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md)
- [Whisper官方模型](https://huggingface.co/openai/whisper-small)、[Whisper API](https://huggingface.co/docs/transformers/model_doc/whisper)
- [LibriSpeech](https://www.openslr.org/12)、[AISHELL-1](https://www.openslr.org/33/)
- [Qwen3-0.6B-Base配置](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/main/config.json)

## 实施交接要点

实施代理负责代码、真实环境锁、上传/远程训练、日志和运行状态；本次仅写计划与审阅记录。命令和目录名称以实现后的 README 为准，避免计划中的设想伪装成已经可执行的 CLI。任何偏离计划的默认超参、评测样本数、epoch数量或预算必须进入 resolved config 和运行报告。

当前未确认Qwen主实验完成。主代理已报告Whisper-small固定128条dev的真实基线WER3.1215%（83/2659）、约8.47秒、无截断，详见主方案11.6；此基线不等于Qwen或完整官方dev结果。正式训练成果仍须由实施代理根据真实checkpoint、日志与评测文件回填。

## Qwen adapter 独立只读审阅补充

检查文件：`src/asr_lab/qwen_data.py`、`qwen_train.py`、`qwen_evaluate.py`。本次未改实施代码。核心标签/因果loss/精度/生成解析未见阻断真实两卡smoke的问题；仍须观察真实多长度batch与首步更新。

训练前发现的配置差异已通知实施者：原配置lr2e-5、warmup3%、save200；主计划为1e-5、5%、100。分布式异常传播应统一finite flag并广播rank0磁盘/IO失败。微批均值loss在不同microbatch分组下与global-token CE不完全等价，吞吐矩阵不作为收敛等价性证据。

已查证HF4.57.6达到max_steps会强制保存steps checkpoint；Qwen官方generate返回`sequences`字段；当前Trainer缩放不会重复除梯度累积次数。Qwen full dev必须明确取消/放大默认128条limit。新增speaker-balanced共同128条manifest与最初随机128条不同，旧3.1215%基线不可直接用于新集合的模型比较。
