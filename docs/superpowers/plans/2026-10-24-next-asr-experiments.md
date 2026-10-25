# ASR后续诊断与单变量验证实施计划

> **For agentic workers:** Use superpowers:subagent-driven-development for implementation and independent review. 本文件是分阶段设计，不是新增GPU或训练队列授权；现有逐token探针仍按其原合同执行。用户“暂不新增训练”的限制继续有效。

**Goal:** 先解释单句重复发生在哪个条件分支，再检查它与总体早期WER退化的关系；只有证据支持后才设计一次可解释的训练干预。

**Architecture:** CPU复用已有全量逐句证据，单卡执行已冻结概率探针；下一阶段只对预先冻结的16句采原始生成。未来如获授权，使用两条匹配的新训练分支检验一个学习率变量。

**Tech Stack:** 已固定的Qwen3-ASR-1.7B revision、PyTorch2.8.0/Transformers4.57.6、BF16/SDPA推理、现有normalizer、JSON/PT和10秒资源监控；不升级环境或更换推理后端。

**Spec:** [逐token概率合同](../../TOKEN_PROBABILITY_PROBE.md)、[已有Early100结果](../../../reports/early100/README.md)、[重复出现对照](../../../reports/repetition-onset/README.md)及用户当前规划请求。

## 当前判断：两件事分别验证

1. **总体退化已证实。** 同固定128／2905参考词，base为46错，25/50/100为122/108/93错。最早受测25步已退化，后续逐渐回落，但100仍差于base；不是误差随步数单调增加。
2. **单句循环已定位到生成阶段。** `652-129742-0006`在base/25/50正常，100到512 token无EOS并重复CULINARY；parser只把长串压短。受测阴性50、阳性100，不能宣称精确首次异常在100。
3. **单句循环不能解释大部分WER差异。** 该句贡献16/929=1.72%的完整clean新增词错误；配对排除后依然退化。该句不在固定128中。
4. **概率机制尚未测量。** 最新保存状态仍为`ready_blocked_gpu_busy`，3+9探针实际调用0。不能把下面分支预测写成现有发现。

CPU本轮已从保存的逐句编辑数复算：

| 相对base | 变差句 | 编辑数持平句 | 改善句 | 总词错误 |
|---|---:|---:|---:|---:|
| 25步 | 47 | 79 | 2 | 122 |
| 50步 | 45 | 79 | 4 | 108 |
| 100步 | 34 | 87 | 7 | 93 |

持平不等于文本完全一致或识别完全正确。完整记录与输入SHA见[设计证据](../../../reports/next-experiment-design/design-evidence.json)。本次只执行CPU规划计算，没有查询当前服务器、运行模型或训练；上次GPU占用快照不是当前状态保证。

## 全局约束

- 同模型权重来源、输入预处理、English提示、greedy、beam1、512-token上限和normalizer；新实验若改变其中一项，必须独立命名并重新冻结合同。
- 分阶段预算彼此独立。第一阶段省下的调用或时间不自动转给下一阶段；不自动串行推进，不自动失败重试。
- 推理选择一张符合资源门槛的卡，三个/四个模型始终在同一物理卡执行，记录UUID。用户允许GPU0或GPU1；现有worker固定GPU1，若选GPU0，先最小修改设备参数并做独立审阅及CPU契约验证。保留PersonaPlex及其他任务，不把显存余量当空闲授权。
- 不以这条样本修改repetition penalty、beam、EOS bias或token预算来声称训练问题解决；不在本轮加入LoRA、架构升级、数据扩容、伪标签或streaming。
- 未完成的test评测保持封存。开发集反复探索与分层面板不能当作独立最终泛化验证。

## Review Focus

- 不同条件前缀和大小写BPE不能直接作同事件概率比较；TF行`P-1+j`与EOS必须对齐。
- 固定batch16与singleton的padding/cache数值路径不同；即使同句，也不要求与旧singleton逐位相等。
- 16句是依据既有结果分层的机制面板，不能用其WER或循环比例估计总体发生率。
- 未来训练两条分支必须匹配实际逐rank样本顺序；相同seed不等于位级确定性。
- 磁盘预算包括导出临时文件；model-only不具有完整优化器恢复能力，不删除旧结果腾空间。

## 阶段A：完成现有逐token概率探针（立即优先项，0训练）

**已有实现：** `src/asr_lab/token_probability.py`、`scripts/run_token_probability.py`；分析工具在`reports/token-probability/validation/`。当前代码已审阅并通过本地109、远端9项CPU测试，不因再次等待GPU而重复整套开发。

**固定矩阵：** base、50、100各1次自由生成，再对三条相同保存轨迹各1次teacher forcing：自然大小写正确句27 token、全大写正确句37 token、循环512 token。合计**3 generate＋9显式forward，1个ID，0更新**。

**硬上限：** 总1200秒；每模型240秒，含SHA/加载/保存/校验；新增2GiB；项目≤300GiB、挂载空闲≥500GiB。事后CPU审计≤600秒。任一身份/配置漂移、raw轨迹与既有同模型记录不符、OOM、超时、非有限分布或缺证据，保存partial并停止；有限TF/cached数值差异只作为观察。

重点按顺序读结果：

| 检验 | 同条件比较 | 结果将指导什么 |
|---|---|---|
| 起始分支 | j0的C34/CA5049；固定C前缀下j1的UL1094/aul4943及候选排名 | 定量描述100在哪个前缀更倾向错误分支；不能把单片段概率叫整词置信度 |
| 正确句尾 | 自然句j26、全大写句j36，分别比较两个distinct EOS概率之和 | 若100仍正常偏向EOS，则不支持“所有句尾都不会停”的说法；若明显变化，再检查对应训练EOS监督 |
| 已进入循环 | 全512行目标续写和EOS概率，并保留每个周期 | 若base在同一强制循环前缀下也倾向继续，则重点是如何进入该前缀，不等于训练无关 |
| 实现一致性 | 实际生成token与processed-score argmax；input/row mapping与完整矩阵审计 | 原始序列/shift/身份不一致先排实现问题，不用新训练掩盖；TF与cached有限差异本身不判为实现错误 |

- [ ] 重新核对空闲卡/锁/磁盘/部署SHA，单次启动现有合同。
- [ ] 保存完整token、EOS、parser前后文本、全词表logits/scores、9组TF行、资源监控和退出状态。
- [ ] 全矩阵独立CPU复算后导出分支/EOS/全512位置图；写清观察能排除什么、不能确定什么。
- [ ] 阶段结束后汇报，不自动执行B或C。

## 阶段B：16句原始输出机制面板（下一份候选合同，0训练）

**为什么还需要它：** 现有128句已足以说明总体退化，但旧批量评测没有保存全部解析前token。这个小面板检查其他错误/持平样本是否也有被parser隐藏的重复或停止异常，避免无限钻研一条食谱句。

**现在已冻结样本，尚未运行模型：** 目标1条＋base→25步变差5条＋编辑数持平8条＋改善全部2条。改善池只有2条，因此没有虚构“改善5条”。从各池排序IDs，`random.Random(20261024)`先抽5变差、再抽8持平，最后16条按ID排序。50/100结果不参与选样。

[面板manifest](../../../reports/next-experiment-design/proposed-panel16.jsonl) SHA256：`d62bb111b9a79734bfc7066ae543bdae84f5c9473197851cdf4f62d8efa03753`。输入来源与分组在`design-evidence.json`，CPU生成脚本为`freeze_panel.py`。它是探索性结果分层面板，不能以其WER/循环率声称总体效果。

**候选预算：** base/25/50/100各对同一batch16生成一次，**4 generate、64条消费、16个ID、0额外TF、0更新**；保持greedy/512。单卡顺序、总1200秒、每模型240秒、增量1GiB。只采原始token/解码/parser/EOS/输入及来源，不再采全词表logits，防止批量矩阵大幅放大。

**输出：** 每ID一行，列出四模型的S/D/I、token数、EOS位置、周期重复、parser压缩程度、大小写变化。已知目标的循环/截断是诊断观察，不是悄悄重试或删除该行的理由。OOM、缺样本、身份漂移或超时才停止。若batch16下目标不复现旧singleton行为，原样报告，不额外追加singleton。

**准备时的文件边界：** 新增`src/asr_lab/panel_generation.py`、`scripts/run_panel_generation.py`、`tests/test_panel_generation.py`；复用旧纯函数，不改已冻结概率worker或历史report。

**接口：** 消费已冻结panel16 JSONL、四模型身份和相同normalizer；生成`run/<model>/{inputs,raw-tokens,decoded,effective-generation,metadata}.json`及必要PT/monitor，汇总`panel-comparison.jsonl`与SHA manifest。

- [x] CPU复算全128并冻结16句ID、顺序、来源与SHA，未采新模型结果。
- [ ] 若A结果和下一次阶段决策支持，先为B单独审阅上述合同；不借用A剩余预算。
- [ ] CPU契约覆盖缺失/重复ID拒绝、固定4次调用上限、EOS/批padding分离、无重试；完成独立review后运行。
- [ ] 对全部16句报告，不因新结果换样本；随后只选择一个训练假设或继续保留原模型。

## 阶段C：未来训练候选——仅检验学习率倍率（当前不执行）

只有A/B及现有标签/更新链审计没有未解决的实现问题，并且用户恢复训练授权后才制定实际启动单。先前已经审过右padding、prompt/audio mask、EOS监督和causal shift；不重复把这些检查当新发现。若A指出EOS监督问题，则先修并做零更新验证，**不同时改学习率**。

默认候选为一对新匹配分支：

| 参数 | 同期对照 | 候选 |
|---|---|---|
| 初始化 | 同一原始snapshot，fresh optimizer | 相同 |
| 唯一改变 | peak LR=1e-5 | peak LR=5e-6 |
| scheduler | 原892步linear，45步warmup | 相同形状，仅倍率0.5 |
| 实际训练 | 最多100更新 | 最多100更新 |
| checkpoint | 25/50/100，model-only | 相同 |
| 数据与执行 | seed42、同完整manifest、同逐rank实际IDs、micro8×2卡×accum2=32 | 相同 |
| 其他 | 原FP32参数/Adam、BF16 autocast、SDPA、loss reduction、clip1.0、weight decay0.01、checkpointing | 相同 |

两个分支均从base开始。**不能从checkpoint50“续训另一LR”**：50是model-only，缺少完整optimizer/RNG；也不能用`max_steps=100`把scheduler缩为100。以新低LR直接对旧历史run比较只能作探索，不能替代这对同期匹配控制。逐rank ID必须逐步匹配；导出保护RNG；同seed仍不声称CUDA位级一致。

**训练候选预算：** 两条分支合计最多200次更新，双卡DDP，分支串行。总训练＋限定评测≤3600秒，每条训练≤1200秒。两卡资源不满足就不运行，不能改成单卡后仍宣称与原配置只有LR不同。

**存储：** 六份model-only预计约45.65GiB；同时最多一份临时导出约7.61GiB，余量留给日志。硬新增≤60GiB，启动时项目必须≤240GiB且挂载空闲≥500GiB；每次保存都检查临时峰值和实际增长。禁止训练器额外自动保存完整optimizer/RNG checkpoint。本轮不具备中断后完整resume能力；若要求可恢复，另立容量方案，不删旧权重或悄悄突破300GiB总额。

**限定评测：** 两条分支的25/50/100各跑同fixed128、batch16、512，总768条消费／48次generate。另仅对两个step100跑同panel16，各1次generate，共32条消费；总800条消费／50次generate，无全词表TF评分。每次fixed128评测≤240秒，每次panel≤240秒，所有时间还受3600秒总上限约束。复用已有base预测，只在输入/生成条件完全匹配时比较。没有自动full-clean/other/test。

**主比较预先固定：** step100的fixed128配对词错误差值（低LR－同期对照）；25/50用于轨迹描述，不事后挑最佳点冒充主终点。附带低LR对base的比较；只比对照更好但仍差于base，报告为部分恢复。fixed128没有新增循环/预算耗尽才是候选质量门槛；面板中已知目标的负面对照循环须保留，不把它隐藏成runner成功，也不因为对照预期很差而跳过候选。

**继续条件：** 低LR主终点词错误少于同期对照且通过固定128生成质量门槛，才考虑另行完整dev-clean/other验证；否则停止这一假设，默认仍用base。开发集配对区间可以描述该seed下样本不确定性，不覆盖训练seed差异；单seed也不能证明LR是唯一根因。完整dev确认后，若候选仍有价值，再独立预算复现seed及最终test。先写验收阈值再运行，不用更低teacher-forced loss替代WER。

**实施边界：** 独立新runner、两份LR配置及针对性测试；先验证892/45调度、同ID顺序、无optimizer落盘和RNG保护，再冻结源码。当前文件不包含可直接启动新训练的命令。

## 执行顺序和停止点

```text
现有CPU证据＋固定面板（本次已保存）
  → A：3模型逐token概率（已实现，待资源）
  → 审计与解释；有实现疑点则先解决
  → B：固定16句原始输出（候选，0训练）
  → 选定单一假设；用户仍禁止训练时到此停止
  → C：fresh 1e-5 vs5e-6，各100更新（未来候选）
  → 有改善才另立完整dev、seed复现和最终test预算
```

现在最有价值的下一步是完成A，同时用本次CPU记录约束解释；不是继续重复同一句自由生成、追加checkpoint重训、提高512上限或马上引入更复杂训练架构。GPU长期被占用时应争取一个受控的单卡20分钟窗口；本文件不创建后台等待、自动调度或重试。

## 来源和解释边界

- 项目证据：[Early100](../../../reports/early100/README.md)、[重复对照](../../../reports/repetition-onset/README.md)、[训练链审计](../../diagnosis_model_audit.md)、本次`design-evidence.json`。预算中的训练时长以先前100步279秒为背景，不承诺未来共享主机耗时。
- [Qwen官方固定版本SFT源码](https://raw.githubusercontent.com/QwenLM/Qwen3-ASR/7c6daf77a2421100f5fb066495372c00129d39ff/finetuning/qwen3_asr_sft.py)提供训练接口来源；[官方模型卡](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)核对现有模型/推理接口范围。官方示例超参数不是本数据上的最优值证明，也不据此升级本项目依赖。
- 本路线和60GiB训练候选是当前设计判断，尚未执行。独立Sol规划复核认可分层面板、同轨迹单LR变量及逐次预留磁盘，要求保留上述因果和资源边界。
