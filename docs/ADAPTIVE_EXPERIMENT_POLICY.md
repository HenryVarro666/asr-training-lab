> 状态说明（2026-10-22）：代理因用量限制结束，本文件尚未完成最终验收或执行。首轮结果已产生并记录于 EXPERIMENT_STATUS.md；不得把本文草案称为已执行分析或已完成的预注册。

# Qwen ASR 结果判定与后续决策规则（历史草案）

> 最新用户要求：逐步规划，不再持续运行至额度耗尽。本文仅保留统计方法和候选假设，不构成自动执行授权；现行流程以 [NEXT_STEPS.md](NEXT_STEPS.md) 为准。

制定日期：2026-10-22。策略审阅：Astra 6。本文件只定义科学决策与执行边界，不启动GPU任务、不下载模型，也不表示下述后续流程已实现。

适用对象：`/mnt/raid/asr-lab/runs/qwen-campaign-20261022-a`，启动代码commit `cf30f45`；主模型与数据版本以该campaign的`campaign.json`、每个run的`run.json`和manifest hash为准。主epoch结果现已产生，详见EXPERIMENT_STATUS.md。本草案未形成已验证的事前登记，不应把其中门槛描述为确认性预注册结论。

## 1. 本轮要回答的问题

在相同数据、文本标准化、解码设置和评测ID上，Qwen3-ASR-1.7B进行100小时英语朗读数据全参微调后，是否比原始固定checkpoint更适合这项转写任务；较难的dev-other是否出现退化；选择的两卡配置是否在实测中有效。

模型较新、训练loss下降、两张H100持续高利用率，均不是质量提升的充分证据。允许最终选择未微调Qwen模型。Whisper-small是另一模型族的对照，不能用“Qwen优于Whisper”替代“本次微调优于原始Qwen”。

当前已由主代理核实的起始证据：

| 项目 | 已观察到的结果 | 解释边界 |
|---|---|---|
| Qwen固定共同128条基线 | batch16，46/2905，WER **1.5834768%** | 仅共同监控子集，不是完整dev成绩 |
| Whisper相同128条ID | WER **3.3390706%** | 可作同ID模型族对照，不是微调效果 |
| GPU恢复验证 | step2恢复至step4；音频累计与日程一致，707个已存tensor均有限；最大绝对差约8.90e-6 | 支持恢复流程可用；不宣称bitwise确定性 |

最初随机128条上的Whisper WER 3.1215%属于另一子集，不能混入上述比较。正式报告要链接这些结果对应的metrics与prediction文件。

## 2. 任何质量比较之前的硬检查

按以下顺序执行，任一未通过则状态为`invalid_comparison`，先修证据或实现；不得据此触发learning-rate试验。

1. 两侧模型分别明确为原始固定revision和本次实际训练checkpoint；记录权重/配置hash及训练起点。不能把一次benchmark末尾权重误当原始模型。
2. 参考音频与文本的manifest hash、样本ID集合及顺序相同；每个ID恰好一个预测。实际保留/排除条目及原因相同。
3. normalizer版本、语言条件、generation设置、batch size、最大输出长度与生成后处理相同。硬件计时条件另行记录，不能让速度比较悄悄改变解码质量。
4. 生成不使用参考转写作为prompt；没有输出截断、缺失预测、非有限指标或统计分母为零。异常样本不能在看到模型结果后单侧删除。
5. 训练完成状态、目标epoch/step、实际样本覆盖和终态checkpoint可核查。部分epoch可报告阶段结果，但不能与完整一轮使用同一完成标签。
6. 每条样本保存raw与normalized reference/hypothesis；按固定版本工具重新计算S、D、I、参考词数，与汇总metrics交叉核对。不能仅信任最终百分比。

当前短音频pipeline的dev-clean为原2703条中的2694条合格样本，9条超过30秒被排除。报告名称使用“dev-clean eligible ≤30s, n=2694”，而不是不加限定地声称完整官方dev。dev-other的实际合格样本数与排除量从当前manifest读取，不预填数字。若日后补回超长音频，所有候选需用同一新划分重评，另立结果表。

## 3. 指标优先级与汇总口径

### 3.1 主指标与保护指标

- **主指标**：全部合格dev-clean的corpus WER。
- **保护指标**：全部合格dev-other的corpus WER，判断在较难朗读条件下的回退。
- **监控指标**：固定speaker-balanced128条的WER、train/eval loss、空输出/截断/重复输出率。128条用于发现故障和观察方向，不独立决定“模型提升”，不作为反复查看后的显著性检验。
- **诊断指标**：分speaker的S/D/I，短/长音频、姓名/数字等错误样本；各桶定义在看结果前写入报告。不能只展示改善的桶。

WER必须先加总编辑数，再除以参考词数。设模型A为原始Qwen，B为候选：

```text
WER(A) = sum(S_A + D_A + I_A) / sum(reference_words)
delta_pp = 100 * (WER(B) - WER(A))
relative_error_reduction = (WER(A) - WER(B)) / WER(A)
```

`delta_pp < 0`表示候选更好。先报告绝对百分点与原始编辑数，再报告相对改善；基线为零时相对指标不定义。减少删除数但增加更多插入/替换，不算总体转写更好。

### 3.2 global / pooled WER如何使用

如果clean与other均通过硬检查，可以另报合并两集的**参考词数加权 pooled WER**：把两集编辑数与参考词数分别相加再相除。另可报两集WER的简单平均，但明确标为macro average，两种汇总不可混称。

pooled WER是次要摘要，不能掩盖dev-other退化。不得根据哪一种汇总更好看而临时选择权重。没有同条件dev-other结果时，不生成虚假的global指标，也不把clean结果外推成“整体泛化提升”。

## 4. 配对置信区间：只用已有预测，CPU即可完成

优先采用**配对、按speaker聚类的bootstrap**，而不是把同一speaker的句子全部视为独立样本。流程固定为：

1. 以样本ID匹配两模型预测，并保存每条样本的`S/D/I/reference_words/speaker_id/split`。
2. 每个split内按speaker聚合两模型的编辑数与共同参考词数。若speaker元数据缺失，先修复；确实无法获得时可用chapter聚类或句级bootstrap，但明确记录局限，不能悄悄替换。
3. 固定seed `20261022`、重采样次数 `10000`。每次从该split的speaker集合有放回抽取与原speaker数量相同的cluster；两模型使用**同一组**抽样索引。
4. 对抽到的cluster分别累加两模型编辑数和参考词数，计算本次`delta_pp`。取2.5%与97.5%分位数，作为配对差值的95% percentile区间。
5. 同时报告真实cluster数量、重采样单位、seed、次数、区间算法、工具版本和输入预测hash。bootstrap样本数增加不等于增加真实独立speaker数量。
6. pooled指标在clean/other内分别重采样，保持两集cluster数不变后合并；如果核查发现相同speaker跨split出现，应使用共享speaker重采样权重，以保留跨split关联。

区间直接针对两模型差值计算，不用“两个模型各自WER区间是否重叠”替代配对检验。句级bootstrap可作为敏感性分析，但不能从多个算法中挑出最窄区间报告。

这是对当前评测数据的抽样不确定性估计，不覆盖预训练数据污染、训练seed变化、真实生产分布或全部系统误差。因为dev用于适应性决策，这些区间是**开发集上的描述性证据**；经过挑选的最优checkpoint区间不能当成未选择过的确认性结论。

## 5. 预先固定的决策表

沿用主方案：`0.3`个绝对WER百分点是当前campaign的**行动门槛**，不是显著性阈值或普适行业标准。用`delta_clean`、`delta_other`表示候选减原始Qwen；负值为改善。

| 完整dev结果 | 决定 | 可使用的表述 |
|---|---|---|
| clean差值95%区间上界<0，other差值区间上界≤+0.3pp | 将候选列为优先部署/后续验证候选，保留原模型；不再扩大同100h训练 | “在当前合格dev划分观察到改善，other回退未超过预设容差的证据较充分”；部署本身仍需单独运行验证 |
| clean点估计改善，但其差值区间跨0；other无明确退化 | 保留原模型为默认，并列候选，补充错误分析；不自动新开LR搜索 | “观察到数值改善，证据仍不确定” |
| clean改善，other点估计恶化≥0.3pp或其差值区间下界>0 | 不发布“整体更好”；分析域适配与困难样本退化，优先原模型或可用早期checkpoint | “clean改善伴随other回退” |
| clean点估计恶化≥0.3pp，train loss下降，且硬检查/错误样本排除实现问题 | 允许进入一次低LR候选判断，见第6节 | “当前设置在dev退化；更低LR是待验证假设” |
| clean差值绝对值<0.3pp且区间跨0 | 结束本轮质量搜索，保留原模型；完成效率与可复现报告 | “未得到清楚的微调收益证据” |
| clean、other均明确退化 | 优先回到原模型；先检查目标格式、数据分布和训练强度，满足第6节才开展唯一后续训练 | “微调在本次评测上退化” |
| other未评完，或任一集合分母/设置不匹配 | 保持`pending_evaluation`或`invalid_comparison` | 不作整体质量结论 |

当other点估计没有恶化但区间上界>+0.3pp时，含义是样本证据不足以排除该幅度的退化，不能直接称为“已证明退化”，也不能称为“已证明无退化”。保留两侧数字和不确定性。

候选选择另看训练/推理成本，但不能为了降低计算成本修改质量指标。相同架构的两个checkpoint推理速度应在同条件重测，不能把训练吞吐当成在线推理延迟。

## 6. 第二次训练：默认只允许一个受控候选

首轮supervisor没有启动第二LR训练。用户现要求分阶段推进：先完成诊断和最小验证，再单独制定下一轮实验单；本文条件即使满足，也不自动触发训练。所有新增实验需有独立run ID和决策记录，不复用首轮12小时预算作为继续运行的依据。

### 6.1 优先方案：只把LR改为5e-6

只有下列条件全部成立才开展第二训练run：

1. 主run已完成指定一轮，原始模型/终态模型的完整合格dev-clean、dev-other均已有可比生成结果。
2. clean比原始模型恶化至少0.3pp，且train loss下降；审阅代表性新增错误后，未发现音频/目标错位、prefix/EOS、截断、normalizer等实现原因。
3. 若置信区间仍跨0，在决策中标明“退化证据不确定，进行一次保守诊断”；不能包装成已经找到过拟合原因。
4. 没有另一个后续训练候选已占用本轮唯一附加run额度。
5. 剩余已授权墙钟时间按实测吞吐足够覆盖一轮、结束评测和保存；campaign总占用≤300GiB、挂载盘保留≥500GiB；不得动PersonaPlex。预算不足则保存建议，不能不断新建campaign重置12小时预算。

第二run从**相同原始Qwen snapshot重新开始**，不是从已退化checkpoint继续。固定训练/评测manifest、seed、有效batch、microbatch、optimizer、warmup比例、训练epoch数、标准化和解码设置；**唯一计划变量为峰值LR 1e-5→5e-6**。记录相同训练曝光量和更新次数。效果不好时不再连续试2e-6、1e-6等参数。

第二run结束后同时保留原始模型、首run终态与第二run终态的全部对照结果。不能因为第二run看上去更好，就隐藏首run失败。若第二run仍未证明优于原始模型，最终选择原始模型是正确结论。

### 6.2 早期checkpoint优先作为低成本诊断

如果只剩终态与前一个完整checkpoint，最多对**一个事先明确的早期checkpoint**补做完整dev评测。选择原则是：优先现存、完整、可恢复、时间上最接近终态之前的保存点；不要先跑所有checkpoint的生成WER再挑最优。若只有teacher-forced eval loss，也不能把“loss最优”直接称为“WER最优”。

早期点好、终态差，可以支持“训练强度或时间相关退化”的假设，但未证明学习率是原因。它可以成为后续候选或使附加训练不再必要。此结果属于适应性选型，需标明选择过程；选定后不能再通过test挑checkpoint。不要为了找最佳epoch反复回溯或重新训练大量保存点。

本轮默认候选上限：原始模型 + 首run终态 + 至多一个现存早期点 + 至多一个追加训练终态。若追加训练内部又查看checkpoint，它只用于故障监控，不自动扩张最终候选集合。

### 6.3 LoRA何时值得考虑

**本轮不因全参结果不好就自动再开LoRA。** LoRA改变可训练参数集合和优化问题，不是对LR假设的单因素验证。当前campaign默认附加训练是6.1的低LR候选。

只有已有证据表明全参训练的显存/吞吐成本阻止任务完成，或跨域评测呈现持续退化且需要研究保留预训练能力，才将LoRA列入下一独立实验设计。仅凭dev-other退化不能宣称发生了中文/多语种遗忘；没有中文评测就没有该结论。

若剩余授权范围内决定用LoRA**替代**唯一低LR候选，必须在运行前写清方法变更理由、target modules、rank/alpha/dropout、冻结范围、可训练参数数、学习率、数据曝光量和恢复/加载合同。使用相同原始snapshot；修正与验证adapter路径后才允许训练。不得既跑低LR又继续开多个LoRA rank以寻找好看结果。LoRA结果称为方法探索，不能和低LR消融一样声称“只改变学习率”。

## 7. 防止反复比较造成过度结论

固定128条监控、多次teacher-forced loss与checkpoint检查都属于开发过程，不产生新的独立验证证据。不能每次都做95%检验，看到一次“通过”就停止并宣称显著提升。

这轮只设一个主要问题：首run终态相对原始Qwen的完整合格dev-clean差值；other用于预先声明的保护检查。pooled、错误桶、早期点与第二run均为次要/适应性结果，完整披露。所有结果表包含失败、不确定与被停止的试验。

若最终要对多个候选同时作确认性“优于基线”声明，应在独立test前固定候选数量及假设族；可用同一配对cluster重采样形成保守的Bonferroni simultaneous区间：对于K个预声明候选，使用每个候选`1-0.05/K`的双侧区间。或对有效配对检验的p值应用Holm校正。不能把`2*min(bootstrap_delta≤0比例, bootstrap_delta≥0比例)`未经论证当成万能p值。

默认更简单的做法是：**dev上选一个候选，与原始模型在封存test上做一次主要对照**。如果所有候选不如原始模型，则报告开发实验失败，不强行选一个微调checkpoint来“赢”。

## 8. test集与结束规则

test-clean/test-other不参与LR、LoRA、checkpoint、normalizer、长度过滤、beam或输出清理策略选择。看test之前冻结：候选checkpoint、评测manifest和过滤规则、decode/normalizer配置、主指标、CI方法与报告模板，并保存hash。

封存test上报告原始Qwen与唯一选定候选，必要时附预先声明的Whisper模型族对照。若沿用≤30秒过滤，必须在看模型预测前产生相同的eligible test划分与排除统计；不得单侧过滤。若test结果不支持dev改善，照实报告，不能返回相同test继续调参再称独立测试。

以下任一情形结束质量搜索，转入报告和可复现交付：

- 主run效果不确定、差值小于行动门槛，且没有明确可修实现错误。
- 已使用唯一后续训练额度，或已达到候选上限。
- 剩余墙钟/磁盘预算不足以完成训练、评测和恢复点保存。
- 代理token额度耗尽或任务被用户停止；保存当前真实状态和未执行队列，不把“计划继续”写成“已经完成”。

持续自主运行不等于持续训练同一数据直到得到改善。有效工作也包括误差归因、配对统计、模型恢复校验、效率报告和如实记录无收益结果。

## 9. 决策记录格式

每个适应性决定保存一份小型JSON，引用已有产物，不把大数据或私密主机信息放进公开仓库：

```json
{
  "policy_version": "2026-10-22",
  "policy_sha256": "<computed>",
  "campaign_id": "qwen-campaign-20261022-a",
  "decision": "keep_base | retain_candidate | evaluate_one_earlier | run_lr_5e_6 | propose_lora | stop_budget | invalid_comparison",
  "evidence": {
    "baseline_checkpoint": "<pinned revision>",
    "candidate_checkpoint": "<path and weight hash>",
    "dev_clean_metrics": "<path>",
    "dev_other_metrics": "<path>",
    "paired_bootstrap": "<path>",
    "error_audit": "<path>"
  },
  "delta_clean_pp": null,
  "delta_other_pp": null,
  "reason": "<observed evidence and remaining uncertainty>",
  "remaining_wall_seconds": null,
  "project_gib": null,
  "mount_free_gib": null,
  "additional_training_runs_used": 0,
  "test_accessed": false,
  "status": "decided_not_executed"
}
```

执行完成后增加实际run ID、退出状态、资源消耗和产物hash，而不是覆盖原决定。用“observed / uncertain / pending / failed / completed”区分证据状态，不预先承诺微调必胜。
