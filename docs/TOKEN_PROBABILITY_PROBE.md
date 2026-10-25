# 单句逐 token 概率探针合同

状态：2026-10-23，执行前冻结规格。用户已授权先规划并执行一次有界推理诊断；本文件不是执行收据。Astra负责策略与后续独立审阅，Sol负责实现，主代理负责有界执行。完成或失败即停止，不自动训练、扫描学习率或追加模型调用。

## 要回答的问题与范围

唯一音频 `652-129742-0006`；仅原始 base、Early100 step050、step100 三个现存权重。上一轮已证实 base/50 正确转录并 EOS 结束，100 生成 512 token 无 EOS 的 CULINARY 周期。本轮定位：在相同前缀下，三个模型如何给正确续写、循环续写和结束符分配概率；自由生成从哪个 token 开始不同，以及循环前缀中 EOS 是否被持续压低。

这是条件概率和生成轨迹诊断，不能直接识别学习率、训练数据、优化实现或参数更新机制的因果。只读保存的旧轨迹作为固定输入，不重新定义正确文本或重新 tokenize。大小写/标点不同且正常转录的两条轨迹都保留。

已先查代码图：`file_summary(src/asr_lab/qwen_truncation_probe.py)` 返回13节点；随后读取冻结旧探针、onset合同和原始序列。旧源码、旧报告、旧运行目录保持不变。

## 固定矩阵与调用计数

三个独立新进程，物理 GPU1 顺序 base→step050→step100。模型身份沿用 `reports/repetition-onset/inputs/model-reference.json`，原始模型 revision 为 `7278e1e70fe206f11671096ffdd38061171dd6e5`，检查点来自 `/mnt/raid/asr-lab/runs/qwen-early100-20261022-a/training/checkpoint-{50,100}`。

每个 worker 依次执行：

1. 同一 singleton prompt/audio 一次自由 greedy generate，采集全部实际 logits/scores，核对其 raw tokens 与该模型旧 onset singleton 一致。
2. 对固定 `correct_natural` 轨迹做一次完整 teacher-forced forward。
3. 对固定 `correct_upper` 轨迹做一次完整 teacher-forced forward。
4. 对固定 `loop` 轨迹做一次完整 teacher-forced forward。

| 固定轨迹 | 直接取自旧 onset raw continuation | 长度 | 末尾 |
|---|---|---:|---|
| correct_natural | base | 27 | EOS 151645 |
| correct_upper | step050 | 37 | EOS 151645 |
| loop | step100 | 512 | 无 EOS；不补 EOS |

所有模型使用完全相同的三组 token IDs。合计上限 **3 次 generate + 9 次显式 teacher-forced forward，1 个不同样本，0 次更新**。生成内部逐 token forward 不计作额外 teacher-forced 调用，但记录其实际 token 数。不能把12次顶层调用错误写成仅3次推理，也不能新增一次 forward 作为GPU smoke或数值重试。

## 生成观测与旧轨迹控制

保持旧 processor、audio、prompt、137-token 前缀、完整输入tensor、BF16、SDPA、eval/inference、设备映射、runtime和greedy决策不变。不 reset RNG、不引入额外 processor、温度、惩罚、停止条件或beam。决策 kwargs 仍为 `max_new_tokens=512, do_sample=False, num_beams=1, use_cache=True`。

本轮优先使用 HF 原生**仅观察输出**开关 `output_logits=True, output_scores=True`，由已验证官方 outer.generate 转发到 thinker；outer已设置 `return_dict_in_generate=True`，不得重复传这个参数。这是本轮明确允许的观测配置差异，完整记录，不能声称 kwargs 与旧实验逐字节相同。其余有效配置、EOS/PAD实际tensor、processor列表必须与旧singleton一致；输出开关不允许改写选择规则。使用已安装主源核对返回字段和语义，若不支持则停止实现准入，不临时换另一套解码。

`logits` 定义为官方生成循环返回的未施加 logits processors 的词表向量；`scores` 为该步选择token所用的处理后向量。完整记录返回dtype、shape、库源码hash及该版路径；不要将 BF16 权重模型的 FP32 cast 输出宣称为 FP32 模型精度。不得为了获得观察添加一个新 logits processor。返回值、配置和原始轨迹优先保存，然后再做身份/轨迹断言。

每个实际生成token必须对应恰好一行 logits 和 scores，词表宽度一致。对 scores 的 argmax 使用原库/torch 的 tie 行为，验证对应实际所选token；独立保存 top1/top2 margin、tie个数。不得用 raw-logit argmax 代替处理后选择检查。EOS集合固定 `{151645,151643}`，真实pad为151645；这里singleton没有结束后的额外有效生成步骤，但仍保留返回的全部token。

若与旧同模型 raw trajectory 不同：先保存本次全部可获得证据，标 `instrumented_trajectory_drift`，停止该模型剩余 teacher forcing 和后续worker，不补跑。不能直接将 CUDA 微差断言为观测开关改变语义或保存错误。仅“token相同”也不证明时间/内存开销没有变化。

## Teacher forcing 的精确定义

使用官方 `model.thinker` 的推理 forward，**不使用训练 outer-forward monkeypatch，不传 labels，不调用 loss/backward/optimizer**。每条轨迹从空cache开始、`use_cache=False`、`past_key_values=None`、`cache_position=None`、batch1。按已核验官方forward的重新计算分支处理 `rope_deltas` 等多模态位置状态：不得沿用先前generate或另一条轨迹的缓存位置；如需清空该属性，仅在TF前按官方语义清空并记入调用收据，不修改权重或生成配置。必须审查实际调用确保新序列重新计算位置，不能只凭 `use_cache=False` 推定所有状态都已清空。prompt/audio特征与generate相同；不从解码文本重新构造processor输入。

设已冻结输入前缀 `P` 长度为137，目标轨迹为 `Y=(y0,…,yL-1)`：

- forward 的 input IDs 是 `P || Y[:-1]`，attention mask 是原前缀mask加 L−1 个1；audio/input_features/feature_attention_mask不变。
- 由官方代码计算相应位置及多模态位置关系；不把旧cache_position或137长position_ids盲目复制到扩展序列。
- 全长输出第 `137-1+j` 行预测 `Y[j]`，j=0…L−1。保存的矩阵只有这L行全部词表 logits，以及明确的 input-position→target-position→target-ID 映射。
- 两条正确轨迹包含EOS作为最后一个**被预测**目标；它不出现在这次teacher-forced输入末尾。loop无EOS，不人为添加第513个预测或再多做一次forward。
- 输入前缀/延长mask/audio占位数和输出shape、索引必须断言；每次先保存原始取证，再生成派生指标。所有GPU forward在inference mode，model.eval，BF16/SDPA不变。

CPU toy-contract必须覆盖首目标、末目标/EOS、L=1、错一行拒绝和轨迹token原样保留。不能用 teacher-forced logits 与 free-generation 的位级相等作为索引正确的唯一证明。

同一模型自己旧轨迹上的 teacher-forced 行，与此次free-generation行条件前缀一致，可以比较实际差异；但 full-sequence no-cache 与 cached generation 的内核/数值路径不同。保存逐行 max-abs/mean-abs logit差、目标/EOS logprob差、argmax是否一致和margin；有限数值差本身作为观察，不能自动重跑、擅调容差或冒称等价。跨轨迹即使相同前缀也可能因总长度/内核不同有微差，明确报告。

## 必须保存的证据与派生量

新目录 `reports/token-probability/`，新run独立目录。除继承输入/模型/音频/来源hash、runtime、有效generation配置、raw sequences/continuation、EOS位置、双decode、parser前后与normalizer之外，保存：

- 自由生成的完整词表 raw logits 与 processed scores，按实际步数存储；9组teacher-forced的完整词表**目标对齐行** logits。原始返回dtype另记，所有持久化词表矩阵使用FP32；保存形状、row mapping、文件SHA。不截词表、不只保存topk。
- 固定三轨迹的原始token数组、来源文件SHA、tokenizer词表/特殊token身份；每次teacher-forced实际扩展input/mask指纹与完整IDs。不同模型权重身份不同，但三轨迹ID语义必须一致。
- CPU以FP32 stable log-softmax从保存矩阵逐行派生，不再模型调用。保存 target logprob/prob、rank、top10 IDs/pieces/logprobs、top1−top2 margin、entropy、两个EOS各自概率及**合计概率**。EOS合计用两个distinct token的概率求和或logsumexp，不只看其中一个，也不把pad等于EOS当作第三个事件。
- rank定义 `1 + count(logit > candidate_logit)`，同时记录并列数量，不把这个rank暗示为实际tie-break顺序。entropy处理合法零概率，保留极小事件的logprob，避免仅报告underflow为0的概率。
- 每一行的统一候选集为三条固定轨迹中出现过的token IDs与两个EOS的并集，所有模型/行按同一集计算；附加该行top10。这使“正确/循环候选”可以复核，不按模型事后挑有利词。候选单token只是该前缀下的下一token事件，不能将一个BPE片段概率称为完整单词CULINARY的概率。
- 每模型/轨迹的总与平均target NLL作为描述量；三种长度与大小写不同，不能用未经限定的总NLL判定哪种文本更好。主要跨模型比较应固定同一轨迹、同一个j、同一前缀。每条轨迹保存完整prefix/target数组；逐行提供prefix长度/hash和可复原的索引，关键分叉另列prefix IDs及decode，不能只给没有前缀的概率数字。

原始logits应全部有限。当前三个旧stage的processor列表均为空，因而不存在in-place processor污染returned raw logits的已知路径；必须继续核验空列表，不能将未来有processor时的别名行为推广为已验证。scores中的NaN/+inf、全词表−inf均停止；−inf仅在实际捕获的处理器明确合法屏蔽时可保留并以零概率处理。本合同旧路径processor为空，出现未解释−inf应停止。派生概率和entropy需有限/归一化一致。实际输出与必需行数或vocab不符、缺少返回字段即失败，不悄悄退为topk-only。

## 分叉、循环与EOS的报告规则

同时报告两个问题，不混为一谈：

1. **自由轨迹token分叉**：对每对模型算最长完全相同token前缀LCP，首次不同索引及双方下一token（也包括一方EOS/结束）。旧结果已知base对100在j=1分叉；base对50、50对100在j=0分叉。这是tokenization层位置，不自动等于首个语义错误。
2. **固定前缀概率变化**：在上述3×3矩阵内逐轨迹逐位置比较同一目标、EOS总概率、实际候选logprob差/排名。base自然大小写与50全大写都归一化为同一正确16词，早期case分叉不能标成转录错误；loop分支与正确分支的候选优势必须用同前缀行解释。

对loop轨迹完整512行报告EOS总概率、循环目标logprob/排名，逐周期汇总作为附加视图；保留所有行，不能只展示一个有利位置。可以指出哪些条件前缀下模型偏向继续循环、EOS是否较其他模型低；不能称低EOS为训练根因、把局部argmax margin解释为校准置信度，或把512窗口推广成无限循环。

没有新增词序列概率搜索、不同提示、强制停止实验、温度/惩罚干预或其他样本。没有“base在loop条件上也低EOS，所以训练无关”等反事实越界推断：被强制输入长异常前缀本身改变条件。

## 硬预算、存储与停止

| 项目 | 上限 |
|---|---:|
| 总墙钟（校验+3worker+汇总） | 1200秒 |
| 每模型stage（SHA+加载+生成+3forward+保存） | 240秒，且不超过总剩余时间 |
| 调用 | 3 generate + 9显式teacher-forced forward |
| 新增磁盘（含cache/临时写） | 2GiB |
| 项目总量 / 挂载可用 | ≤300GiB / ≥500GiB |

以vocab=151936估计，旧生成长度总576行：free raw+scores为1152行，teacher-forced为1728行，FP32合计约1.750GB（1.63GiB）。**实际以加载配置的vocab和预期行数复算**；不保存prompt部分全词表logits，不另存全词表probs/logprobs，不把完整矩阵嵌入JSON。远端大矩阵只保留独立 `/runs/` 目录，不再复制到远端repo；允许后续镜像到本地，报告用hash/路径引用，不将大矩阵纳入Git。若保存两份相同向量可内容寻址共用，但必须保存两种语义角色及相等验证，不能未经比较当作相同。优先逐矩阵原子落盘并释放CPU/GPU副本；预算必须计入写入中临时文件，不能先复制一份总矩阵再翻倍写盘。

按上次执行经验240秒有余量，但不保证；若实际vocab、预估持久化体积或剩余空间不满足上限，执行前停止，不下调证据要求或扩预算。若输出轨迹漂移或意外长输出导致更多矩阵，保存到预算允许范围并明确partial；不得为“完整”越界或补跑。

沿用全局锁、owned进程组清理、10秒资源监控、物理GPU1空闲检查（stage前及迁入GPU前）、GPU0 PersonaPlex保护。只在本项目根写数据，不停止其他任务，不自动等待/重试。每stage当前模型权重只做一次完整SHA，匹配上一onset已捕获hash（base现在已有历史锚点），分块检查deadline；SHA前后、worker前后stat与软链身份核验。其他共享来源/输入小文件正常hash。不能把stat匹配声称为结束后重新完整hash。

身份/输入/有效配置漂移（仅上述两个输出开关例外）、旧轨迹不一致、OOM、超时、非有限分布、I/O失败、缺证据、资源超限或异常退出，立即停止剩余调用；保留已有证据与失败阶段，清理自己进程组。teacher-forcing与generation之间有限数值差、概率高低、重复/截断是观察，不是自动重试理由。完成后保留旧Early100 failed，不补跑dev-other，不启动训练队列。

## 实现及验收边界

只新增独立runner/worker、CPU分析helper、针对性契约测试与本轮报告。复用冻结旧源码的纯函数；不改旧generate函数、训练patch或旧report来容纳新字段。需要新metadata，明确 `logits_collected=True`、观测开关、显式TF调用计数及各forward cache模式，不能照抄旧“未采logits”声明。

CPU测试至少覆盖：3模型×固定调用顺序/预算；轨迹与共享输入身份；teacher-forced行shift及EOS；官方输出字段和observational-only配置差异白名单；raw与scores职责/argmax/tie；FP32稳定概率、EOS求和及rank；异常/缺证据/漂移时停止、无隐藏retry；原子写与预计字节预算。模型调用不在CPU测试中伪装执行；远端仅CPU契约后，Astra明确代码review通过再进入已授权的一次GPU执行。

合同完成后Astra停止，等待主代理明确followup触发代码或结果审阅；本轮不扩成训练研究计划。


## 中断后恢复说明（2026-10-23）

Astra已完成本合同及代码核心路径的部分审阅，指出stage内产物校验也须受240秒deadline约束；它在最终代码签字前遇到usage限制而中断。恢复后由独立Sol reviewer核验修正、测试及最终源码SHA，结果保存于`reports/token-probability/validation/final-code-review.md`。本说明优先于上文必须取得Astra最终代码签字的执行顺序措辞；不把部分审阅记录为Astra最终通过。实验的调用、时间、磁盘、输入及空闲GPU门槛均不变。

恢复时GPU1已有其他用户任务，探针run尚不存在。先完成CPU准备和远端CPU契约验证；若最终资源快照仍占用，则本轮以`ready_blocked_gpu_busy`归档，不建立GPU等待队列，不自动重试，不伪造概率、审计结果或结果图。

## 2026-10-25已授权的编排修正与新attempt

本段优先于上文GPU1-only、仅worker启动后监控及旧run入口的执行措辞；科学问题、输入、轨迹、调用计数、512-token和数值容差均不变。用户已明确授权先修正报告、总时限、预检监控，再检查GPU0/GPU1，有真正空闲卡时启动一次新的A目录；0新增训练、0自动B、0自动重试。

- 总时限从进程入口起计，覆盖初始化/project扫描、锁、SHA、模型加载、调用、保存、验证、汇总及拥有进程的清理；原1200秒与每stage240秒上限不变。初始化失败也保存partial生命周期收据，不在构造时重置期限。
- 预检前立即启动只读资源采样，10秒周期，覆盖SHA及模型加载阶段；保留GPU0/GPU1的UUID、显存/利用率、compute PID、盘空闲、phase/elapsed及实际终态。不得将预检资源记录当作训练或推理进展。sampler故障、失败或超时须结束本次attempt并回收自己的采样进程。
- 报告分开计划与实际调用、存在/已验证/缺失的矩阵；pre-worker失败不得宣称完整词表矩阵已归档，已进入worker但缺计数的情况不得把未知填为0。
- `--physical-gpu {0,1}`选择一张真正无其他compute进程的物理卡，全程固定UUID；parent及worker迁入CUDA前均复查。GPU0的PersonaPlex同样属于占用，不能忽略或停止。资源竞态触发停止，不切换GPU补跑。
- 原固定runtime记录的`physical_gpu='1'`仅作为资源身份例外按本次选择卡比较，显式记录差异，不改原reference；包版本、dtype、attention、TF32等其他runtime字段仍严格相等。GPU0也必须逐token匹配旧同模型轨迹，若漂移按原合同保存并停止，不放宽数值或轨迹门槛。
- 原`qwen-token-probability-20261023-a`在worker前资源失败，6份partial已保留；本次必须新run，不修改旧原件。新attempt镜像到`reports/token-probability/attempts/<run-name>/`；独立审计与图亦显式输出到该attempt，不覆盖此前prepared记录。
- 实现按[修正计划](superpowers/plans/2026-10-25-a-supervision.md)测试先行、独立复核并以最终源码SHA重新冻结，再部署/远端CPU验证。此前旧版review只证明旧版，不能自动覆盖本次修改。
