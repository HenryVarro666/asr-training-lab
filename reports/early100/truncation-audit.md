# Early100 截断失败审计

状态：部分完成且质量 gate 失败；comparison_valid=false。未重跑、未修改训练或 runner。
实际保存四份预测；step100-dev_other 未启动。完整五阶段控制没有完成。

唯一截断 ID：`652-129742-0006`。
- REF：CAULIFLOWER MAYONNAISE TAKE COLD BOILED CAULIFLOWER BREAK INTO BRANCHES ADDING SALT PEPPER AND VINEGAR TO SEASON
- step100 解析后 HYP：CULINARY CUL
- baseline HYP：Cauliflower mayonnaise: Take cold boiled cauliflower, break into branches, adding salt, pepper, and vinegar to season.
- step100：S=2、D=14、I=0；参考16词；baseline同句0词错误且未截断。

冻结 evaluator 的58–67行逐条检查continuation，判定为长度≥512且没有EOS151645/151643。长度使用batch补齐后的宽度，但每句保留正常EOS时不会仅因同batch长输出而被标截断。该句位于0-based第2203条、batch137；同batch其余15条均未截断。没有证据支持仅由整batch统一长度导致误标；原token未保存，仍不能独立验证EOS或长度。

实际 token IDs/长度及 parse_asr_output 之前的生成文本未被 evaluator 保存，不能从两词 HYP 推断其内容。按冻结源码的512上限与截断标记逻辑，可推断continuation触达512且无识别的EOS；这不是独立保存的token长度证据，也不能断言发生了哪种循环。

已完成 clean 的2694条逐句 raw REF/HYP→固定 normalizer 与保存文本一致：True；所有四份已保存预测文本完整性通过：True。
clean 复算 S/D/I=1463/149/188，1800/54027=3.331667% WER；baseline=1.612157%。原零截断质量 gate 仍失败。

该截断句计16词错误，占candidate总错误0.8889%，对完整clean WER的算术贡献为0.029615个百分点；占相对baseline新增错误的1.7223%。
同时从两侧移除此句后，剩余2693条/54011参考词：candidate=3.303031%，baseline=1.612634%。这只是配对排除敏感性，不是替代正式指标或通过gate；该句也不能解释其余词错误差异。

style/uppercase、前缀、重复、长度、空输出和截断 flags 的计数保存在JSON（多标签可重叠）。这是文本诊断，不能判定音频真值、灾难性遗忘或学习率因果。
原完整五项 audit_early100_outputs.py 仍保持原严格完成条件。

复算命令：`/Users/caochao/Documents/ChatGPT/train/.venv/bin/python scripts/audit_early100_partial.py`。输入、脚本和复用依赖SHA256见truncation-audit.json。
