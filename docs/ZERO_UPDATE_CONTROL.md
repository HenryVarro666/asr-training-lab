# 零更新保存／重载控制

设计并执行于2026-10-22（Chicago），依据Astra 6审阅建议拟定。Sol实现独立控制脚本；本地完整60项测试及远端11项专属测试通过。**真实GPU控制已通过，用时136.79秒；Astra结果复核无阻断项。** 原始／重载两侧128条raw token、输入、模型状态和有效生成规则完全一致，WER均为46/2905=1.5835%。详见[执行报告](../reports/zero-update-control/README.md)。以下保留原实验合同，不代表新的训练队列。

## 问题与范围

在调学习率前，检查原始权重经过本项目训练初始化、序列化和重新加载之后，是否保持相同的评测行为。设计时已有静态配置审阅和两条音频的processor对照，本实验补齐真实权重的保存／重载验证。结果只支持本次受测路径一致，不证明训练更新一定是退化机制。

唯一变化是是否经过零更新保存／重载路径。optimizer更新数严格为0；不调用`Trainer.train()`，不执行反向传播，不创建新的训练数据或修改既有checkpoint。

## 固定条件

- 从revision `7278e1e70fe206f11671096ffdd38061171dd6e5`的原始Qwen3-ASR权重开始。
- 控制侧按首轮实际流程FP32加载，应用相同forward适配、generation sanitize和token ID对齐后保存；使用独立新目录。
- 原始直接加载与重载侧均使用相同软件版本、同一物理GPU1、BF16、SDPA、batch16、固定128条及相同批次顺序，显式greedy参数不变。两侧在本次控制中重新运行，不仅与历史预测比较。
- 核对全部实际参数和持久buffer的名称、shape以及评测dtype下的值；整数buffer保持原dtype。输入tensors、有效生成配置也逐项核对，不能只对比JSON文件的表面差异。
- 保留生成token IDs、原始和规范化文本、WER、加载诊断、输入与模型指纹。正式test仍保持隔离。

## 资源和停止条件

预计FP32模型快照约8.15GB，申请的磁盘增量上限20GiB；实际启动前检查项目总量不超过300GiB、挂载盘至少保留500GiB。所有模型、临时文件和缓存位于`/mnt/raid/asr-lab`。

总墙钟上限20分钟。只使用GPU1，复用project lock、独立进程组、十秒资源采样及超时清理；不操作GPU0服务，不并发其他实验。OOM、非有限值、磁盘不足、源文件变化、异常退出即停止并留存证据，不自动重试或扩大预算。

## 判定与分支

1. 参数、输入、有效解码规则一致，并且固定128的生成token IDs一致：此控制通过。结论仅覆盖本次roundtrip和受测输入，不等于证明所有训练实现正确。
2. 非预期参数差异、加载缺失／多余权重、输入变化或解码规则不同：停止后续训练设计，先定位这一差异。
3. 仅生成有差异：先检查原模型自身重复运行，并记录首个分歧位置两侧logits、top-2 margin。greedy模式也可能对接近并列的argmax敏感，不能把CUDA数值微差直接命名为序列化错误。
4. 若差异可重复且超出自身重跑波动，仍无法解释：记录“控制未通过／原因未定”，不启动第二轮训练。不要用放宽阈值或修改normalizer消除差异。

控制通过后，再编写一份单变量训练实验单。可选学习率、转录目标格式或参数更新范围中的一个；其余条件固定。此文件没有选择获胜方案，也未授权脚本自动遍历所有选项。

## 实施验收

由Sol实现独立控制脚本，复用首轮模型初始化和评测逻辑，避免复制逻辑造成另一套不一致的路径。先检查源代码与首轮指纹，完成必要的CPU契约测试和只读review，才启动一次有界控制。产物保留在独立run，并同步小型报告至`reports/`；大权重继续留在服务器。

实际入口（使用新的run目录）：

```bash
/mnt/raid/asr-lab/venv/bin/python scripts/run_zero_update_control.py \
  --run-dir /mnt/raid/asr-lab/runs/qwen-zero-update-control-20261022 \
  --receipt /mnt/raid/asr-lab/runs/qwen-campaign-20261022-a/campaign.json \
  --training-receipt /mnt/raid/asr-lab/runs/qwen-campaign-20261022-a/main-epoch1/run.json
```

准备、原始评测、重载评测为独立子进程，避免class-level forward patch污染对照。保存采用实际`SafeTrainer._align_special_tokens()`与`save_model()`；不调用`train()`。比较覆盖全部参数、持久buffer和共享别名，保存前后及生成前后状态、实际BF16值、输入tensors、最终thinker解码规则和生成token。EOS之后的batch padding从语义token比较中移除，原始token仍保存。若仅token不一致，最多再做一次base-repeat，仍标记控制未通过；完整logits因果定位另行规划。
