# GPU0请求与启动门槛（2026-10-26）

当前状态：**ready_blocked_gpu0_busy**。用户指定GPU0；本次未切换GPU1，没有创建或启动新run，实际0生成/0显式teacher forward/0更新、0概率矩阵/0结果图。没有停止其他任务，也没有等待、重试、训练或B队列。

## 本次只读验证

检查时间：2026-10-26 00:48:41 CDT（05:48:41 UTC）。[完整快照与执行脚本](validation/gpu0-request-recheck-20261026T054841Z.json)。

- 两端12份部署文件与已验证版本的SHA/字节数一致；沿用源提交`6acd9dab4d6a8d8a8cd2aea23d7ea28484ca319e`，没有改动源码或放宽GPU0保护。
- GPU0 UUID `GPU-eed83181-48a7-e9f3-cb49-e851dfaf647c`，利用率92%，空闲46702MiB（45.61GiB）。compute为既有PersonaPlex PID2154793和其他任务PID3984680；不满足已验证runner的空闲门槛，显存余量不代替该门槛。
- GPU1同时100%利用率，另有三个任务；本次请求仅GPU0，不作为切卡候选。
- 全局锁可用，RAID空闲2522.92GiB，旧Early100 campaign SHA保持`f6be4d5322fba1fb03d89b311017220506fff83ffa238d22614fb1e97ea1c279`。
- 新路径`/mnt/raid/asr-lab/runs/qwen-token-probability-20261026-gpu0-a`不存在。没有实际run的预检周期采样或模型进展，只有此次启动前快照；未生成占位概率图。

## 保留的执行合同

GPU0修正版已经完成[此前本地131/远端116项CPU验证及独立复核](SUPERVISION_UPDATE_2026-10-25.md)；本次因部署SHA不变不重复整套测试。物理GPU选择和runtime.physical_gpu例外已经验证，其他输入/运行时/旧raw轨迹门槛保持严格。

预算仍为同一样本、base/50/100各一次生成与三次固定轨迹评分，总3生成＋9显式TF、0训练；512-token，1200秒总/240秒每模型stage，新增≤2GiB，项目≤300GiB/盘空闲≥500GiB，完成后独立CPU概率审计≤600秒。入口硬期限与立即/10秒/终态监控均保留。

待GPU0满足原门槛后才能单次执行；不关闭PersonaPlex、不停止其他任务、不改用GPU1或自动重试。旧failed run及历史证据继续保留。本次证据经独立只读复核通过；未执行真实概率审计或模型。
