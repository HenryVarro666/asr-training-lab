# 单句逐token概率诊断：CPU数值修复与全量验收通过

2026-10-26当前为 **model_run_complete_cpu_reanalysis_passed**。原GPU0 A取证完成后，原独立审计曾因FP32派生概率超容差失败；该记录完整保留。现按用户另行授权，仅用原12矩阵完成CPU诊断／修复／2304行复算及原容差独立验收，新增模型／GPU／训练调用均为0，无B。根因是FP32绝对logsumexp相减的精度损失；FP64 fused log_softmax修复后目标概率最大误差1.52434e-13，原门槛仍1e-6。三阶段累计129.004秒，130原文件SHA均未变，三张图及PNG／SVG／PDF、资源监控、源码和原新证据已双端保存。

最新入口：[CPU修复报告](cpu-repair-20261026-r1/REPORT.md)、[机器状态](execution-summary.json)、[先行预算计划](CPU_NUMERIC_REPAIR_PLAN_2026-10-26.md)。[原GPU执行与失败报告](GPU0_EXECUTION_2026-10-26.md)保持历史身份，不能将原失败回写成通过。模型重复与WER退化的训练根因仍未确定。

## 准备阶段历史（下列资源快照不是当前状态）

- 原始模型、50步、100步的固定输入和三条旧token轨迹已冻结，正确自然大小写27个token、正确全大写37个、循环512个；同一句`652-129742-0006`、同137-token prompt。
- Sol完成独立worker/supervisor及概率取证：完整词表raw logits/processed scores、实际token/EOS/解析前后文本；三个固定轨迹按`P-1+j`评分，保存完整条件前缀和逐行候选/EOS指标。修正了产物重新哈希可能越过240秒后错误重置期限的问题，汇总也逐行检查总期限。
- 本地9项专项及109项全套CPU测试通过；远端9项专项CPU测试通过，CUDA均隐藏。这些包括mock调度与toy分布，不能当作真实模型调用。
- Astra完成策略及核心代码的部分审阅，最终签字因usage限制中断。恢复后由独立Sol完成最终SHA绑定复核：[审阅收据](validation/final-code-review.md)。没有把它记成Astra最终通过。
- 最终只读快照为`2026-10-24T03:06:54.347567+00:00`（Chicago 10月23日22:06），GPU1仍有3个其他用户任务（PID1427597、1501077、1533571）。按原空闲门槛未启动探针，未停止其他进程，也未建立自动等待/重试队列。GPU0既有PersonaPlex PID2154793保留；RAID空闲约2584.90GiB。
- 计划run `/mnt/raid/asr-lab/runs/qwen-token-probability-20261023-a`不存在。实际计数 **0 generate / 0显式teacher forward / 0 optimizer update**。旧Early100 campaign仍为原失败记录，文件SHA未变。

## 已固定的小规模预算

| 范围 | 上限 |
|---|---:|
| 模型及同一句音频 | base、step050、step100；1个ID |
| 自由生成 | 每模型1次，greedy，最多512新token |
| 固定轨迹评分 | 每模型3次显式teacher forward，合计9次 |
| GPU总墙钟／每模型stage | 1200秒／240秒（含SHA、加载、保存、校验） |
| 远端新增数据／项目总量／可用空间 | 2GiB／≤300GiB／≥500GiB |
| 事后独立CPU审计 | ≤600秒，无模型调用 |

GPU必须先空闲。身份、输入、有效生成决策、旧生成轨迹漂移、OOM、超时、非有限值、I/O或证据缺失均停止剩余调用；保存已产物，不重试或扩大预算。概率高低、重复本身及有限的cached/teacher-forced数值差异是观察。详见[实验合同](../../docs/TOKEN_PROBABILITY_PROBE.md)。

## 准备阶段的证据工具（历史）

执行时比较同前缀的`C`与`CA`、固定`C`后`UL`与`aul`；分别检查两种正确转录句尾EOS，再检查完整512-token循环各位置的继续概率和两个distinct EOS总概率。不能拿不同大小写或不同前缀下的单token概率当成完整词的概率，不能仅凭EOS变化断言训练根因。

独立CPU审计工具会由保存的完整词表矩阵以FP64重算全部行，检查token位移、排名、ties、top10、EOS和数值容差；绘图工具从已审计的保存数据导出PNG/SVG/PDF。准备时的[审计状态](independent-audit.json)为`prepared_not_run`，实际审计0行、图0张。合成CPU计时只测统计运算，不能证明GPU阶段会在240秒内完成。

本地tokenizer小文件缓存位于`.git/asr-cache/token-probability-tokenizer/`，已匹配旧模型SHA；它不含权重，远端固定snapshot仍是权威来源。正式运行才会保存并镜像大型PT；这些矩阵不入Git，也不在远端repo重复保存。

## 恢复入口与记录

当前旧run已存在且failed，不能直接重放[旧单次命令](validation/launch-command.sh)。以后明确恢复时，先取得稳定GPU窗口、核验空闲/锁/磁盘/源码，并为同预算建立新的run路径，保留本次partial；没有后台队列。完成真实run后才能同步矩阵、执行[独立审计](validation/audit_saved_evidence.py)及[绘图](validation/plot_evidence.py)；准备性日志不能升级为实验结果。

- [机器可读当前状态](execution-summary.json)
- [最终主机快照](validation/final-host.json)
- [部署文件SHA](validation/deployment.json)：远端Git HEAD仍为旧训练提交，执行身份以实际部署文件SHA和后续运行receipt为准。
- [本地测试](validation/local-unit-tests.log)、[远端CPU测试](validation/remote-unit-tests.log)
- [固定轨迹与来源](inputs/trajectory-reference.json)、[官方forward/generate来源核查](inputs/upstream-probability-source.json)

此前已执行的单句对照结论仍见[repetition-onset报告](../repetition-onset/README.md)：base/25/50正常，100在512窗口内重复。当时尚未提供逐token概率；最新验收数据与解释边界见本文顶部CPU修复报告。

准备包已提交为`484acae752e23513fc00b3b8180c1e27184016df`，31份允许文件已跨机逐一匹配SHA；Git bundle在本地与远端均verify且SHA一致。见[归档收据](validation/archive-sync.json)。该bundle覆盖准备代码/文档/小型收据，未包含任何尚不存在的本轮概率矩阵；本段与归档收据在payload提交后另行提交、同步。

## 再次资源核查

2026-10-23 22:14:13 CDT 按用户“空闲才执行”要求再次只读核查：GPU1仍有PID1427597、1501077、1533571三个外来compute进程，未满足既定空闲门槛，因此未启动；未停止任何其他任务。计划run仍不存在，计数仍为0生成／0显式评分forward／0训练更新。预算和512-token设置未改变，无自动等待或重试队列。[此次快照](validation/gpu-recheck-20261024T031412Z.json)记录剩余空间约2584.86GiB、锁可用、PersonaPlex保留及旧Early100 campaign SHA未变。

2026-10-23 23:02:24 CDT 再查：GPU1仍有三个外来compute进程（PID1427597、1501077、1591050），故继续保持启动前阻断。未停止其他任务，0模型调用、0新增训练，512-token及3+9／1200秒／240秒／2GiB预算不变。计划run不存在，旧Early100 SHA未变，PersonaPlex保留；RAID空闲约2582.12GiB。[此次只读快照](validation/gpu-recheck-20261024T040223Z.json)。

### 两卡资源核查

2026-10-23 23:25:01 CDT 按用户要求同时检查GPU0、GPU1：GPU0为PersonaPlex PID2154793及两个其他任务（1577799、1597780），GPU1有三个其他任务（1427597、1501077、1622504）；两卡利用率均100%。GPU0剩余32722MiB（31.96GiB），GPU1剩余29824MiB（29.13GiB），均有正在运行的其他任务，因此没有启动概率探针，也未修改设备选择或停止任何进程。计划run仍不存在，0生成／0显式评分forward／0训练更新；512-token和既定3+9／1200秒／240秒／2GiB上限保持。[此次两卡快照](validation/gpu-recheck-20261024T042501Z.json)。

2026-10-24 20:20:33 CDT 再次核查任一可用卡：GPU0仍有PersonaPlex及两个其他任务（2583428、2586966），利用率85%、空闲34198MiB（33.40GiB）；GPU1有三个其他任务（2581010、2586318、2587684），利用率100%、空闲31505MiB（30.77GiB）。两卡均未空闲，因此不启动、不停止其他任务。用户已授权任一空闲卡；若以后只有GPU0可用，先审查现有GPU1-only worker的必要设备参数改动，其他科学设置及预算不变。本次实际0生成／0评分forward／0训练更新，run不存在，旧Early100 SHA未变。[此次两卡快照](validation/gpu-recheck-20261025T012032Z.json)。

2026-10-24 20:26:29 CDT 用户明确继续A阶段、结果汇报后再决定B。实查两卡均100%利用率：GPU0含PersonaPlex和两个其他任务（2586966、2597576），GPU1含三个其他任务（2581010、2587684、2591635）。GPU0空闲23342MiB、GPU1空闲29776MiB，但均有其他计算任务，故A仍未启动。实际0生成／0显式评分forward／0训练更新；保留512-token与3+9／1200秒／240秒／2GiB合同，不进入B、不停止其他任务、不建自动等待队列。run不存在，原Early100 SHA未变。[本次A启动门槛快照](validation/gpu-recheck-20261025T012628Z.json)。

2026-10-25 00:03 CDT GPU1空闲，已按此前授权和冻结源码SHA启动唯一A阶段，budget保持3生成＋9显式TF、512 token、1200秒总/240秒stage/2GiB。GPU0仍有既有PersonaPlex及其他任务，未触碰；不新增训练、不自动进入B。启动与当时资源详见[launch receipt](validation/launch-receipt.json)，结果尚待终态及独立审计。

2026-10-25 00:04:42 CDT 终态更新：base SHA校验后，GPU1新外来任务PID2801775触发pre-worker空闲门槛；campaign failed，未进入execute，0模型调用。00:05:11退出核查：本项目进程已退出，锁可用，PersonaPlex仍在，旧Early100 SHA未变；[6份partial同步收据](validation/failed-pre-gpu-sync.json)。**原始run/REPORT.md中“rows are archived”是固定模板，本次实际没有矩阵**，原件保留，不能据此声称概率采集或连续worker监控完成。
