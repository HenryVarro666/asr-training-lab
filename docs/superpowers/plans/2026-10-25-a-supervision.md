# A阶段报告、全程时限与预检监控实施计划

> **For agentic workers:** Use superpowers:subagent-driven-development for implementation and independent review. 用户已授权本轮修正并在资源允许时执行一次新run；不再请求相同授权。

**Goal:** 修正三项已证实的编排记录缺口，然后在任一真正空闲GPU上完成一次有界A概率诊断。

**Architecture:** A专用runner与轻量stdlib监督helper覆盖初始化至收尾，保留原worker取证和数值计算；只最小参数化物理GPU资源门槛。旧run/报告只读，新attempt使用独立目录。先TDD、独立review、两端CPU验证和源码SHA，再查资源并单次执行。

**Tech Stack:** 现有Python/PyTorch环境、stdlib进程/信号/JSON、nvidia-smi只读采样、原10秒worker monitor；不升级依赖。

**Spec:** 用户2026-10-25当前请求；[已证实问题](../../EXPERIMENT_REVIEW_2026-10-25.md)第4节与[原A合同](../../TOKEN_PROBABILITY_PROBE.md)。当前用户请求覆盖原合同GPU1-only和仅worker监控的限制；科学矩阵及预算不变。

## Global Constraints

- base/50/100，单句652-129742-0006，3 generate＋9显式TF，0更新；固定27/37/512旧轨迹及512-token生成配置。
- 全程≤1200秒，模型stage≤240秒，新增≤2GiB，项目≤300GiB，盘空闲≥500GiB；事后CPU审计≤600秒。
- 不停止PersonaPlex或任何其他任务；任一外来compute进程表示该GPU未空闲；三模型用同一UUID。
- 0新增训练，0 B阶段，0自动重试或资源等待队列；失败保留partial，完成后报告。
- 不修改冻结的共享Campaign/onset/truncation源码、原始failed收据或外部未提交文档。

## Review Focus

- tree_size/构造/导入及收尾不得重置总时限；初始化异常也有状态、采样退出、锁正确关闭。
- SIGALRM/子进程超时及拥有的进程组清理不能杀其他任务或在总时限外长期等待。
- sampler立即采样、10秒周期及终态必须实际写盘；采样失败不得默默当正常；监控不是模型进展。
- pre-worker failed不误称矩阵已存；partial计数须来自实际收据，未知不能填0冒充已知。
- 选GPU0时worker和parent都检查同一张卡，不能只改CUDA_VISIBLE_DEVICES；已有run拒绝且不改旧文件。

## Task 1：监督与资源门槛（实现者独占源文件）

**Files:** 修改`scripts/run_token_probability.py`、`tests/test_token_probability.py`；必要时新增`src/asr_lab/probe_supervision.py`、`tests/test_probe_supervision.py`；`src/asr_lab/token_probability.py`只改资源gate和identity GPU字段。

**Interfaces:** CLI保留原参数，新增`--physical-gpu {0,1}`默认1；既有科学helper接口、固定输入和实际worker1+3调用不变。新增`run/preflight-monitor/metrics.jsonl,status.json`及初始化至收尾的lifecycle时限收据，字段明确elapsed、phase、资源/退出状态。

- [ ] 写并观测失败测试：pre-worker failed报告、partial产物计数、慢初始化超时、清理不超期限、预检初始/周期/终态采样与失败清理、已有run不改、GPU0/1 foreign-PID拒绝。
- [ ] 最小实现，保留现有3×3/漂移/行对齐/概率统计/预算门槛。
- [ ] 专项及完整CPU unittest，记录红→绿；不加载模型、不GPU smoke。
- [ ] 独立代码复核并按最终SHA冻结；任何修订重新审阅关联变更。

## Task 2：部署与一次新run（主代理独占）

**Files:** 新验证收据`reports/token-probability/validation/supervision-*`；新attempt镜像`reports/token-probability/attempts/<new-run-name>/`；状态与报告索引。

- [ ] 记录部署文件、合同修订及来源身份；远端仅CPU测试，核对SHA。
- [ ] 修正后再新查GPU0/1、全局锁、磁盘和原Early100 SHA；选择无其他compute任务的卡。
- [ ] 空闲才启动新run一次：不覆写20261023-a。无空闲则归档阻断，不排队。
- [ ] 监控实际lifecycle、worker阶段及调用计数；异常按预算停止，0重试。
- [ ] 同步全部已产物并SHA核对，旧权重/旧failed不变；若完成再运行已有独立CPU概率审计和绘图，0额外模型调用。
- [ ] 独立结果/partial复核，更新当前状态，提交并双端备份，汇报后停止，不进入B。

## 权限与实现裁定

用户已具体授权这三项修正和资源允许后的A执行，因此本计划只落实执行细节，不新增审批轮次。沿用当前`codex/token-probability`专用分支；保留无关未提交文件，不创建额外checkout以免重复本轮已冻结输入和本地缓存。
