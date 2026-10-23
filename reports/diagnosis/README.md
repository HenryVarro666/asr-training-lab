# 退化诊断与checkpoint800验证

2026-10-22。用户已更新目标：继续有证据支撑的受控实验，逐阶段复核，不以耗尽额度为目标。CPU诊断、checkpoint800 GPU评测及随后[零更新保存／重载控制](../zero-update-control/README.md)均已完成，未新增训练。控制通过，仅覆盖本次路径及固定128输入，退化原因尚未确定。

## 本轮结果

| 合格开发集 | 原始模型WER | checkpoint800 WER | checkpoint892 WER | 样本数 |
|---|---:|---:|---:|---:|
| dev-clean | 1.6122% | 2.4192% | 2.3729% | 2694 |
| dev-other | 3.0850% | 5.0418% | 4.9850% | 2857 |

**退化在第800步时已经存在。** 最后92步的点估计略有改善，未支持它们造成主要退化的假设；退回800不能解决问题。只保留了800和892两个很晚的保存点，不能定位前800步的退化起点、推导最佳早停时刻，或据此确定学习率是原因。当前继续保留原始模型作为质量基线。

## 已核实的结论

原始/微调预测的样本ID、原始参考文本、normalizer版本、真实manifest hash、解码batch16和512-token预算一致，零截断。逐句重算S/D/I、参考词数和corpus WER与已存指标全部一致。对全部原始reference/hypothesis再次运行固定Whisper EnglishTextNormalizer，保存的normalized文本逐项相同。

| 集合 | 样本数 | 改善 | 退化 | 编辑数持平 | WER增加百分点 | 差值95%区间（百分点） |
|---|---:|---:|---:|---:|---:|---:|
| dev-clean | 2694 | 125 | 434 | 2135 | +0.7607 | [0.5885, 0.9307] |
| dev-other | 2857 | 181 | 732 | 1944 | +1.9000 | [1.5171, 2.3379] |

区间为按说话人配对的percentile bootstrap，clean40人、other33人，seed20261022，10000次；属于当前开发集的描述性不确定性分析，不覆盖训练seed差异、预训练污染或真实应用分布，也不能确定因果。原始模型与末步模型两侧均通过comparison_valid检查。

## 输出模式及尚未确定的原因

- 微调后clean有2688/2694条、other有2853/2857条输出全大写；原模型两集均为0条。参考训练文本本身为大写。统一normalizer已经消除大小写，WER退化并非仅由大小写计分造成。
- 全部样本中，没有检测到规范化空输出、解析后特殊前缀残留、三连trigram重复且参考没有该重复、输出词数超过参考1.5倍或生成截断。这些是明确限定的规则检测，不代表排除了所有异常或幻觉。
- 固定样本可见新增专名/罕见词拼写、近音普通词和词边界错误。例如clean的1673-143397-0002把animadversion改为animated version；2086-149220-0028把Phoebe改为phi b。其余改善/持平样本也保留，详见逐句清单。
- teacher-forced固定128条dev loss从step100的0.0805降至step800的0.0601，没有“dev loss上升”的经典过拟合证据。该loss与自由生成WER不同，不能用前者替代后者。
- “全参学习大写/少标点的转写风格导致适配漂移”是候选解释，尚未建立因果。尚无证据直接认定学习率过大、声学标签错误或多语种遗忘。没有进行人工听音验证。

## 可复现产物

- `text-audit.json`：全部raw/normalized核对、规则计数、固定抽样ID及输入hash。
- `dev-clean.paired.json` / `dev-other.paired.json`：完整S/D/I、差值、配对区间、版本与源码hash。
- `*.all-utterances.jsonl` / `*.edit-counts.jsonl`：全部样本的差异证据。
- `*.review-samples.jsonl`：每集固定抽20退化、5改善、5持平，共60条；seed20261022，非只挑最大错误。
- `inputs/`：从服务器复制的原manifest与固定spelling-normalizer；原始音频和训练文件未修改。
- `SAMPLE_REVIEW.md`：固定60条样本文字审阅记录；`../../docs/diagnosis_model_audit.md`：已完成的独立模型链条审阅，包含两条不等长音频的CPU processor、标签和生成输入对照。

分析工具的15项CPU测试已通过，包括metadata不匹配时拒绝有效比较、stored S/D/I校验、cluster重采样、输出分母和真实CLI。本地完整49项测试通过，见`validation.json`和`unit-tests.log`。冻结训练src没有变更。

## checkpoint800实验：已完成

只评估现存checkpoint-800，对照原始Qwen及checkpoint-892；两集全部合格dev、相同batch16/manifest/normalizer/greedy设置，0次optimizer更新。预先记录的问题是：**末尾92次更新是否主要造成已观察到的退化？** 决策和执行证据见`checkpoint800-decision.json`、`checkpoint800/campaign.json`及两个stage receipt。

| 对照（候选减基线） | clean差值pp [95%区间] | other差值pp [95%区间] |
|---|---:|---:|
| 原始→800 | +0.8070 [0.6260, 0.9861] | +1.9568 [1.5783, 2.3896] |
| 800→892 | −0.0463 [−0.1098, 0.0223] | −0.0568 [−0.1404, 0.0307] |

两集800→892区间均覆盖零；这不是等效性证明，也不排除小效应。所有区间使用与上文相同的speaker配对bootstrap，属于开发集诊断，不是独立test确认。四份新增比较均通过`comparison_valid`条件、实际manifest哈希和已存S/D/I复核。

| 集合 | 原始S/D/I（总错误） | 800 S/D/I（总错误） | 892 S/D/I（总错误） | 参考词数 |
|---|---|---|---|---:|
| clean | 643/157/71（871） | 1057/103/147（1307） | 1038/109/135（1282） | 54027 |
| other | 1251/200/124（1575） | 2128/167/279（2574） | 2096/199/250（2545） | 51053 |

另对800的全部5551条raw reference/hypothesis再次运行固定normalizer，零差异；零截断、规范化空输出或所检查的解析前缀/重复异常。clean/other分别2687/2854条全大写。证据见`checkpoint800/output-audit.json`，这仍不是人工音频正确性判定。

本实验实际479.49秒（7.99分钟），预算1200秒，仅GPU1，未触碰test。两stage exit code均为0，campaign状态complete。资源记录包含clean正常退出时的一次`training_process_exited`通用告警：它发生于成功stage receipt前约0.45秒，随后other正常执行完成，不能称为零告警或训练崩溃。没有OOM、NaN或磁盘阈值失败。

结束后只剩既有GPU0 PersonaPlex进程2154793，GPU1无本项目计算进程；挂载盘约3.1TiB可用。互斥锁、独立进程组和监控记录保留。Sol只读验收未发现阻断项。

## 可复算命令

在仓库根目录，以本地既有环境运行；以下均为CPU分析，不启动训练：

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/audit_saved_predictions.py \
  --campaign reports/qwen-campaign-20261022-a \
  --inputs reports/diagnosis/inputs --output reports/diagnosis
.venv/bin/python scripts/analyze_asr_predictions.py \
  --baseline reports/diagnosis/checkpoint800/dev-clean/predictions.jsonl \
  --candidate reports/qwen-campaign-20261022-a/evaluation/final-dev_clean/predictions.jsonl \
  --manifest reports/diagnosis/inputs/dev-clean.jsonl \
  --bootstrap-unit speaker --replicates 10000 --seed 20261022 --require-valid \
  --output reports/diagnosis/dev-clean.800-vs892.paired.json \
  --details reports/diagnosis/dev-clean.800-vs892.edits.jsonl
```

other使用对应other路径；原始→800比较的baseline改为原campaign的`evaluation/baseline-dev-clean`或`baseline-dev-other`，candidate使用本次800预测。每个输出记录输入文件和分析源码的SHA256及包版本。GPU实际命令、环境约束、源文件与manifest指纹由`checkpoint800/campaign.json`记录，不需重复GPU评测来复算统计。

## 下一阶段

零更新控制实际136.79秒通过，原始与重载权重、输入、有效生成规则和128条raw token全部一致，双方WER1.5835%；Astra独立复核无阻断。具体证据见[控制报告](../zero-update-control/README.md)。下一项优先问题是“退化是否在前100步已出现”：保留原892步scheduler、lr、seed、batch和数据顺序，短程复现早期轨迹并在固定检查点生成评测。需先落实独立实验单与停止机制；本轮未排队第二轮训练。
