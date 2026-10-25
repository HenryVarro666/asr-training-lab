# A监督修正与资源门槛（2026-10-25）

当前：**ready_blocked_gpu_busy**。三项修正已完成、测试与独立复核通过并部署；新A2尚未创建/启动，实际0生成/0显式TF/0更新/0矩阵/0结果图。无等待、重试、训练或B队列。

## 已完成的修正

- 报告分开计划、实际调用、未知调用、存在矩阵与size/SHA核验矩阵。pre-worker失败明确0，缺失worker计数保守unknown。
- 入口stdlib计时/硬中断覆盖heavy imports、初始化、SHA、worker、保存/验证、汇总、清理；仍1200秒总/240秒模型stage。清理前持久化主失败及cleanup_pending；模型完成先pending_cleanup，所有收尾成功后才提交complete。清理与失败记录预留均在原总限内。
- 预检监控立即采样、10秒周期和终态记录，两卡UUID/显存/利用率/compute PID与盘空闲；只清理本程序拥有的进程组。支持固定选择GPU0或1，PersonaPlex及任何外来任务均阻断；不自动切卡重试。

源提交：`6acd9dab4d6a8d8a8cd2aea23d7ea28484ca319e`（前版`7c174ca`）。独立review绑定runner/helper/worker/test SHA；模型、固定输入、生成和概率统计逻辑不改，仅physical_gpu资源身份有显式例外。

## 验证与已保留的失败

| 验证 | 实际结果 |
|---|---|
| 本地概率专项/helper专项 | 25 / 6通过 |
| 本地完整套件 | 131通过 |
| 独立边界复核 | 第一轮4项＋增量1项通过 |
| 远端完整已有套件 | 116通过，26.779秒，exit0 |
| 部署 | 12份允许文件SHA匹配 |

远端缺少本地独有`test_prediction_analysis.py`的15项，所以116不伪称等于131；共同测试文件SHA一致。[测试目录差异](validation/supervision-test-inventory.json)。本轮测试使用CPU、mock/临时fake nvidia-smi，不是A实际模型结果或训练验证。

前版远端115项中慢tree超时测试失败1项（主异常未在昂贵清理前落盘）；已保留[失败日志](validation/supervision-remote-validation-v1-failed.txt)和[失败部署](validation/supervision-deployment-v1-failed.json)。修正后新增确定性红→绿回归，隔离极短timer测试的外部采样/清理延迟，独立10秒真实fake-resource采样集成仍保留。没有因此扩大生产预算。

[最终本地原始输出](validation/supervision-local-validation-v2.txt) · [实现交接及红绿记录](validation/supervision-local-validation-v2-summary.txt) · [独立复核](validation/supervision-code-review.md) · [远端输出](validation/supervision-remote-validation.txt) · [部署SHA](validation/supervision-deployment.json)

## 修正后的两卡实查

时间：2026-10-25 00:33:07 CDT（05:33:07 UTC）。

| 设备 | 利用率 | 空闲显存MiB | 正在运行的compute PID |
|---|---:|---:|---|
| GPU0 | 100% | 49626 | PersonaPlex2154793、其他任务2836182 |
| GPU1 | 91% | 71882 | 其他任务2836827 |

[原始快照](validation/supervision-gpu-recheck-20261025T053307Z.json)。两卡均不空闲；显存余量不代替空闲条件。全局锁可用，挂载空闲2542.95GiB，旧Early100 SHA未变。未停止任何其他任务。

`/mnt/raid/asr-lab/runs/qwen-token-probability-20261025-a2`不存在；原`20261023-a`失败run保留，不重放旧路径。[新参数化单次入口](validation/supervision-launch.sh)及[原预算](../../docs/TOKEN_PROBABILITY_PROBE.md)已准备：未来明确恢复先重新查资源，选择一张真正空闲卡，全程同UUID，最多3生成＋9TF、512-token、1200秒/240秒/2GiB。当前没有真实A2预检时间序列或概率分布，不生成占位图片、不运行缺数据的概率审计。结果后再决定B，暂不新增训练。
