# Repetition onset 独立 CPU 审计

状态：**passed**。四个保存阶段的张量、切分、EOS、tokenizer/parser/normalizer、S/D/I及文件指纹通过独立复算。

| 阶段 | 返回token数 | 首EOS位置 | 强重复 | 512无EOS loop | 解析结果 |
|---|---:|---:|---|---|---|
| base | 27 | 26 | False | False | Cauliflower mayonnaise: Take cold boiled cauliflower, break into branches, adding salt, pepper, and vinegar to season. |
| step025 | 35 | 34 | False | False | Cauliflower Mayonnaise TAKE COLD BOILED CAULIFLOWER BREAK INTO BRANCHES ADDING SALT PEPPER AND VINEGAR TO SEASON |
| step050 | 37 | 36 | False | False | CAULIFLOWER MAYONNAISE TAKE COLD BOILED CAULIFLOWER BREAK INTO BRANCHES ADDING SALT PEPPER AND VINEGAR TO SEASON |
| step100 | 512 | None | True | True | CULINARY CUL |

暴力枚举每个起点和完整周期，独立核对源码helper与保存统计；未将合成重复测试当作实际token证据。

4次generate / 4次样本消费 / 0训练更新；本审计没有新增推理。Base shard SHA是本轮首次捕获；25/50/100对照历史prefix-complete。

仅观察保存采样点，不能确定精确首发update、持续性、单调性或训练根因。旧singleton参考不等于原Early100全dev原始token重放。

step100完整sequences与旧singleton一致：True。

[完整命令、环境、逐项证据与SHA](independent-audit.json)
