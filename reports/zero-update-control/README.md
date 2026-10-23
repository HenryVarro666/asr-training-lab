# 零更新保存／重载控制：通过

2026-10-22（Chicago；完成UTC为2026-10-23 04:16:44）。远端run为 `/mnt/raid/asr-lab/runs/qwen-zero-update-control-20261022`，源码版本 `c1d8ac3`，实际源码指纹由 `campaign.json` 记录。Sol实现，Astra在运行前及结果同步后分别只读审阅，未发现阻断项。

本次问题是：原始模型经过训练初始化、零更新保存和重新加载后，是否改变评测行为？**固定128条上没有发现变化。** 这支持保留原有保存／重载路径，不能证明训练更新逻辑全部正确，也没有确定微调退化机制。

## 实际结果

| 项目 | 结果 |
|---|---|
| optimizer更新／反向传播 | 0／0；没有调用 `Trainer.train()` |
| 完成状态 | prepare、base、reload全部exit 0；campaign complete |
| 墙钟时间 | 136.79秒，预算1200秒；只用物理GPU1 |
| 原始模型WER | 46/2905 = 1.5834768% |
| 重载模型WER | 46/2905 = 1.5834768% |
| 两侧S/D/I | 均为32/9/5；128条，无生成截断 |
| 生成token | 128条完全一致，包含原始batch padding；规范化前的预测记录也完全一致 |
| 参数、输入、有效解码规则 | 全部一致，8批输入；保存前后以及生成前后持久状态不变 |
| 加载诊断 | missing/unexpected/mismatched/error均为空 |

状态指纹包含708个state_dict键，共2,349,217,408个元素。`thinker.lm_head.weight`与`thinker.model.embed_tokens.weight`共享存储，重复计入311,164,928个元素；去重后为实际2,038,052,480个参数。当前模型未记录额外持久buffer键；比较逻辑也覆盖存在此类buffer的情况。不能把state_dict的带别名元素总数当成唯一参数数目。

输入、BF16值、加载状态及最终thinker生成规则均来自实际执行。两侧使用相同revision、软件版本、固定ID/顺序、batch16和greedy解码。FP32保存前后状态一致，BF16投影分别与两侧实载模型一致。没有因原始配置中的无效采样字段差异而改变实际greedy规则；完整原始配置保留于 `generation_raw.json`。

## 资源和监控

14次资源采样没有记录告警；三阶段终态以成功stage receipt为准。通用monitor的 `status.json` 是最后一次采样，所以仍显示当时进程running；其中 `latest_checkpoint=inputs-07.pt` 是诊断输入tensor，不是训练checkpoint，不能据此宣称训练或恢复进展。

完成快照确认GPU1无计算进程、控制supervisor及tmux session均已退出；GPU0保留PersonaPlex PID2154793。项目约195.6GiB，控制run约7.72GiB，挂载盘空闲约3.04TiB，均在既定预算内。原训练checkpoint800/892及optimizer、scheduler、两rank RNG仍保留。

TensorBoard训练事件已经同步到 `../monitoring/main-epoch1-tensorboard/`，事件解析覆盖892条train loss和8条eval loss。SSH转发存活时，可在 http://127.0.0.1:16006/ 查看原训练曲线；本控制没有新的训练曲线。

## 证据与边界

- [独立复核记录](verification.json)、[机器判定](comparison.json)、[执行记录](campaign.json)、[最终主机快照](final-host-snapshot.json)。
- `prepare/` 保存状态指纹与零更新收据；`base/`、`reload/` 保存预测、指标、输入指纹、实际生成规则和资源采样。
- 大型FP32快照和8批输入tensor留在服务器；本地只同步小型证据，没有复制权重或音频进Git。
- 运行前本地完整60项测试及远端11项控制测试通过，日志位于 `../zero-update-control-validation/`。CPU测试与本次真实GPU结果分开记录。

此控制未使用独立test或人工听音，也未测试优化器更新、全量输入、双卡训练数值等价或原生streaming。主实验微调退化的结论保持不变；当前选择原始Qwen作为开发集质量基线。

## 下一项问题

Astra建议后续短程复现原训练前段，判断退化是否在前100步已出现。必须保留原始892步学习率调度及相同seed、batch、数据顺序，在预定检查点做generation；不能直接设 `max_steps=100` 而同时重设调度。先写独立实验单和停止机制，再决定启动。本轮没有排队或启动第二轮训练。
