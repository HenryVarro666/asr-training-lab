# 固定 60 条样本的文本审阅

本审阅逐条阅读两份已固定 JSONL，未重新抽样、未读取全量预测作自动归因、未听音频，也未启动 GPU 或训练。REF 是提供的参考转写，BASE/CAND 是输入文件中的基线/候选输出；三者均不自动等于录音真值。`improved/regressed/tied` 沿用输入在既定正常化与词级编辑对齐下的分组，不能据此断言标签的音频正确性、灾难性遗忘或学习率因果。

输入：[dev-clean.review-samples.jsonl](dev-clean.review-samples.jsonl)；[dev-other.review-samples.jsonl](dev-other.review-samples.jsonl)。下表逐条引用正常化后的局部原串，省略未变化上下文；明确注明“原始”的说明来自相应 raw 字段。Δ 是 CAND 编辑总数减 BASE 编辑总数，负数为现行评分改善。

## 样本范围与分类口径

每个 split 固定 20 条退化、5 条改善、5 条持平，共 60 条。这是按结果配额挑选的案例审阅，不是随机样本；任何比例或编辑和都不应外推为全量频率、总体 WER 或显著性证据。

分类由逐条人工式文本判断给出，可多标签：

- **专名/拼写**：名称、标题或字母级正写差异；并不保证已确认某一写法正确。
- **普通词近音候选**：普通词替换中值得听音核对的疑似近音/词形现象；未做音标或声学验证，不能把“候选”省去。
- **词边界**：合写、分写或短语内部重新切分；词数不变也可有边界变化。
- **插删**：明确额外/缺失的计分 token，含由边界变化带来的词级插删；不能等同于实际说话增删或音素遗漏。
- **正常化争议**：缩写、正写、空格、口语记法或归一化产物可能影响计分解释；仅提示审查，不在此修改规则、重算分数或裁定等价。
- **持平一致**：该样本三者正常化后整句相同；可与正常化争议共存。

`audio-needed` 表示词语/读音需回听才能判断；`unknown` 表示正写、源文本、标签或正常化政策尚未核验。纯空格或同音专名拼法往往还需源文本，不能指望音频独自裁决。全部 60 条均未做音频验收。

| split | 退化 | 改善 | 持平 | 样本内 BASE 编辑和 | 样本内 CAND 编辑和 |
|---|---:|---:|---:|---:|---:|
| dev-clean | 20 | 5 | 5 | 26 | 44 |
| dev-other | 20 | 5 | 5 | 26 | 59 |

下列计数是**人工多标签的样本覆盖数**，不是自动错误分解；同一行可同时计入多列，各列不得相加当作样本数或独立错误数。专名/拼写也包括本行未改善的共同差异。

| split / 分组 | 条数 | 专名/拼写 | 普通词近音候选 | 词边界 | 插删 | 正常化争议 | 持平一致 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev-clean / 退化 | 20 | 14 | 8 | 6 | 5 | 5 | 0 |
| dev-clean / 改善 | 5 | 1 | 1 | 4 | 4 | 4 | 0 |
| dev-clean / 持平 | 5 | 0 | 0 | 0 | 0 | 0 | 5 |
| dev-other / 退化 | 20 | 13 | 11 | 6 | 8 | 4 | 0 |
| dev-other / 改善 | 5 | 3 | 3 | 3 | 3 | 3 | 0 |
| dev-other / 持平 | 5 | 0 | 0 | 0 | 0 | 1 | 5 |

## dev-clean：逐条审阅

### 退化 20 条

| ID / Δ | REF → BASE → CAND（局部） | 可观察分类与说明 | 未决项 |
|---|---|---|---|
| `1272-128104-0004` / +1 | REF: `linnell` / `idylls` / `birket` / `carker` / `up guards and at em`<br>BASE: `linell` / `idylls` / `birkit` / `carker` / `upguards and adam`<br>CAND: `linnell` / `ittles` / `burket` / `carkrer` / `upguards and adam` | **专名/拼写、普通词近音候选、词边界、插删**。一处 linnell 更接近 REF，但 idylls 和 carker 新增偏差，birket 的不同拼写仍在；两者均未匹配 up guards and at em。净退化不表示每个局部都更差。 | audio-needed；多处专名需源文本 |
| `1673-143397-0002` / +2 | REF: `animadversion`<br>BASE: `animadversion`<br>CAND: `animated version` | **普通词近音候选、词边界、插删**。一个词变成两个词，REF 原已匹配；文本可见替换并多出一个计分 token。 | audio-needed（词语与分界未核） |
| `1673-143397-0020` / +1 | REF: `blemmyes` / `nestorius`<br>BASE: `blemys` / `nestorius`<br>CAND: `blemis` / `nostrorhos` | **专名/拼写**。族名拼写两者均异于 REF；新增 nestorius → nostrorhos 差异。 | audio-needed；名称正写 unknown |
| `1993-147966-0003` / +1 | REF: `ambrosch` / `mama`<br>BASE: `ambrosch` / `mamma`<br>CAND: `ambrose` / `mamma` | **专名/拼写、正常化争议**。新增专名 ambrosch → ambrose；mama/mamma 是共同拼写差异，不是新增退化。 | audio-needed（专名）；mama/mamma 规范 unknown |
| `2086-149214-0003` / +2 | REF: `though slight`<br>BASE: `though slight`<br>CAND: `those light` | **普通词近音候选、词边界**。词数不变但内部词边界与词形重新切分；原匹配的短语变为两词替换。 | audio-needed（不能仅凭拼字确认声学来源） |
| `2086-149220-0022` / +2 | REF: `phoebe` / `daguerreotype`<br>BASE: `phoebe` / `daguerreotype`<br>CAND: `phebe` / `daguerrotype` | **专名/拼写**。名称及普通名词各出现一处拼写变化。文本不足以分清命名惯例与读音识别。 | unknown（专名/正写）；audio-needed |
| `2086-149220-0028` / +2 | REF: `phoebe`<br>BASE: `phoebe`<br>CAND: `phi b` | **专名/拼写、词边界、插删**。名称拆成 phi b，增加一个计分 token。 | audio-needed；名称正写 unknown |
| `2412-153948-0004` / +1 | REF: `bred`<br>BASE: `bred`<br>CAND: `breed` | **普通词近音候选**。过去分词形式变为 breed；其余摘录上下文匹配。 | audio-needed（时态/元音未核） |
| `2412-153954-0001` / +1 | REF: `eyeing`<br>BASE: `eyeing`<br>CAND: `eying` | **专名/拼写、正常化争议**。仅见 eyeing/eying 的正写差异，正常化后仍不同；是否作等价拼写应单独复核，不应由本例重定义评分。 | unknown（正写/评测规范；音频不能单独决定拼法） |
| `2428-83699-0000` / +1 | REF: `christmases`<br>BASE: `christmases`<br>CAND: `christmasses` | **专名/拼写、正常化争议**。Christmases 的 s 数量发生变化并被现行评分计错；没有据此证明读音或语义发生变化。 | unknown（正写规范）；未听音频 |
| `251-137823-0011` / +2 | REF: `nearby`<br>BASE: `nearby`<br>CAND: `near by` | **词边界、插删、正常化争议**。nearby 拆成 near by，现行词级对齐增加替换和插入；空格变化本身不证明声学损失。 | unknown（复合词边界/评分规范） |
| `2803-161169-0007` / +1 | REF: `aslant`<br>BASE: `aslant`<br>CAND: `aslan` | **专名/拼写、普通词近音候选**。普通词末尾字母 t 不再出现；这是词串替换，不能把字母缺失直接当成音素漏读。 | audio-needed |
| `3081-166546-0022` / +1 | REF: `gryce`<br>BASE: `gryce`<br>CAND: `grice` | **专名/拼写**。侦探姓名 y/i 拼写改变，REF 原已匹配。 | unknown（名称正写）；audio-needed |
| `5694-64029-0023` / +1 | REF: `cheatham`<br>BASE: `cheatham`<br>CAND: `cheedem` | **专名/拼写**。专名改成另一串拼写；仅凭文本无法认定该串对应何种实际读音。 | audio-needed；名称源文本 needed |
| `6241-61946-0016` / +1 | REF: `these sacred`<br>BASE: `these sacred`<br>CAND: `they sacred` | **普通词近音候选**。句首 these → they；属普通词替换。 | audio-needed（辅音/词尾未核） |
| `6313-66129-0022` / +1 | REF: `tad is` / `horsemanship`<br>BASE: `ted is` / `horsemanship`<br>CAND: `ted is` / `horsemenship` | **专名/拼写**。新增 horsemanship 的拼写差异；Tad/Ted 是两者共同偏差。原文所有格经正常化成为 is，不能误当成新增词。 | unknown（正写）；Tad/Ted 需 audio-needed |
| `652-129742-0008` / +3 | REF: `mayonnaise` / `over all`<br>BASE: `mayonnaise` / `over all`<br>CAND: `maitrenais` / `overall` | **专名/拼写、普通词近音候选、词边界、插删、正常化争议**。mayonnaise 新变为 maitrenais，over all 同时并为 overall；本例混有词语及边界变化。 | audio-needed（mayonnaise）；边界规范 unknown |
| `777-126732-0009` / +1 | REF: `terrorist`<br>BASE: `terrorist`<br>CAND: `terror` | **普通词近音候选**。terrorist → terror 为词替换；不能因字面缩短而直接判定整词删除。 | audio-needed |
| `8842-302196-0001` / +1 | REF: `rossetti`<br>BASE: `rossetti`<br>CAND: `rosetti` | **专名/拼写**。专名少一个 s；文本可见正写变化，音频未核。 | unknown（专名正写）；audio-needed |
| `8842-302196-0010` / +1 | REF: `vita nuova`<br>BASE: `vita nuova`<br>CAND: `vita nova` | **专名/拼写**。标题/专名片段 nuova → nova；是否涉及发音或译写需另查。 | unknown（标题规范）；audio-needed |

### 改善 5 条

| ID / Δ | REF → BASE → CAND（局部） | 可观察分类与说明 | 未决项 |
|---|---|---|---|
| `174-50561-0010` / -2 | REF: `played` / `bed time`<br>BASE: `play it` / `bedtime`<br>CAND: `play it` / `bed time` | **词边界、插删、正常化争议**。改善来自 bed time 与 REF 的词边界对齐；played → play it 这一共同差异仍在。不能把整句称为已纠正。 | unknown（分词规范）；played/play it 需 audio-needed |
| `5694-64025-0002` / -2 | REF: `et cetera`<br>BASE: `etc`<br>CAND: `et cetera` | **词边界、插删、正常化争议**。缩写 etc 改成 REF 的两词展开式；现有正常化未将二者合并。这是文本表示改善证据。 | unknown（缩写/正常化规范）；非听辨改善证据 |
| `5895-34629-0008` / -2 | REF: `off hand` / `fibi` / `vinos`<br>BASE: `offhand` / `phoebe` / `phoebus`<br>CAND: `off hand` / `phoebe` / `phoenix` | **专名/拼写、词边界、插删、正常化争议**。off hand 的边界改善；两处名称仍异于 REF，末名由 phoebus 换成 phoenix，不能把名称变化也计作正确。 | audio-needed；名称规范还需源文本 |
| `6313-66129-0007` / -1 | REF: `free to admit`<br>BASE: `afraid to admit`<br>CAND: `free to admit` | **普通词近音候选**。普通词 afraid → free 后与 REF 相同。 | audio-needed（尚未确认实际读音） |
| `777-126732-0063` / -2 | REF: `passer by` / `gas lamp to gas lamp`<br>BASE: `passerby` / `gaslamp to gaslamp`<br>CAND: `passer by` / `gaslamp to gaslamp` | **词边界、插删、正常化争议**。只修复 passer by 边界；两处 gaslamp 与 REF gas lamp 的共同差异保留。 | unknown（复合词分写规范） |

### 持平 5 条

| ID / Δ | REF → BASE → CAND（局部） | 可观察分类与说明 | 未决项 |
|---|---|---|---|
| `2078-142845-0051` / +0 | REF: `the sifted sugar should be omitted`<br>BASE: `the sifted sugar should be omitted`<br>CAND: `the sifted sugar should be omitted` | **持平一致**。三者正常化后整句相同；摘录为句尾。 | unknown（音频未核） |
| `2412-153948-0012` / +0 | REF: `each must cry louder and wander farther`<br>BASE: `each must cry louder and wander farther`<br>CAND: `each must cry louder and wander farther` | **持平一致**。三者正常化后整句相同。 | unknown（音频未核） |
| `2803-154328-0011` / +0 | REF: `the jailer may forget`<br>BASE: `the jailer may forget`<br>CAND: `the jailer may forget` | **持平一致**。三者正常化后整句相同，不因 CAND 大写而计作识别变化。 | unknown（音频未核） |
| `3081-166546-0019` / +0 | REF: `george had returned to my side`<br>BASE: `george had returned to my side`<br>CAND: `george had returned to my side` | **持平一致**。三者正常化后整句相同；包含相同姓名 george。 | unknown（音频未核） |
| `84-121550-0017` / +0 | REF: `3 maidens at the right wheel`<br>BASE: `3 maidens at the right wheel`<br>CAND: `3 maidens at the right wheel` | **持平一致**。原始三者均为 Three/THREE，正常化为 3；此处格式归一未造成分差。 | unknown（音频未核） |


## dev-other：逐条审阅

### 退化 20 条

| ID / Δ | REF → BASE → CAND（局部） | 可观察分类与说明 | 未决项 |
|---|---|---|---|
| `116-288046-0005` / +1 | REF: `preacher or priest`<br>BASE: `preacher or priest`<br>CAND: `preacher or preach` | **普通词近音候选**。priest → preach；近音只是待验证候选，不是已做听音判断。 | audio-needed |
| `1585-131718-0042` / +1 | REF: `to notice`<br>BASE: `to notice`<br>CAND: `to de notice` | **插删**。新增 token de；不能仅凭文本确定它是填充音、错误词边界还是解码产物。 | audio-needed；机制 unknown |
| `1585-157660-0008` / +5 | REF: `oft expectation fails` / `there where most it promises` / `oft it hits`<br>BASE: `aft expectation fails` / `there where most it promises` / `aft it hits`<br>CAND: `aft exhortation fowls` / `there were most a promises` / `aft a hits` | **普通词近音候选**。两者已有三处 oft/aft 共同偏差；CAND 再出现 expectation/exhortation、fails/fowls、where/were、it/a 等普通词替换。 | audio-needed；不由文本推断噪声/口音 |
| `1686-142278-0064` / +1 | REF: `aunt shaw`<br>BASE: `aunt shaw`<br>CAND: `aunt shaugh` | **专名/拼写**。姓名 Shaw 的拼写变化；声音真值和姓名正写均未核。 | unknown（专名规范）；audio-needed |
| `2506-11267-0006` / +1 | REF: `in private`<br>BASE: `in private`<br>CAND: `in privacy` | **普通词近音候选**。private → privacy，末尾词形变化。 | audio-needed |
| `3660-172182-0011` / +2 | REF: `geraint greets`<br>BASE: `geraint greets`<br>CAND: `jarain greeted` | **专名/拼写、普通词近音候选**。专名 geraint 及 greets/greeted 同时变化。 | audio-needed；专名规范 unknown |
| `3660-172182-0030` / +4 | REF: `told geraint the cause` / `said geraint be it`<br>BASE: `told geraint the cause` / `said geraint be it`<br>CAND: `told gerai not the cause` / `said gerai not be it` | **专名/拼写、词边界、插删**。两处 geraint 各拆为 gerai not，新增两个 not token；不等于已证明说话人说了否定词。 | audio-needed；名称与切分 unknown |
| `3663-172528-0041` / +2 | REF: `pietro pagolo`<br>BASE: `pietro pagolo`<br>CAND: `piero paglio` | **专名/拼写**。两个人名 token 均改变；不据此判断哪种姓名书写有权威性。 | unknown（专名源文本）；audio-needed |
| `3915-57461-0014` / +1 | REF: `ghastliness of mien`<br>BASE: `ghastliness of mien`<br>CAND: `ghastliness of mine` | **普通词近音候选、专名/拼写、正常化争议**。mien/mine 是现行正常化保留的词串替换；读音相近或等音需听音核对，单听音也可能需上下文/源文本区分。 | audio-needed；语义/正写 unknown |
| `3915-98647-0020` / +1 | REF: `american tombs`<br>BASE: `american tombs`<br>CAND: `american tumbs` | **专名/拼写**。tombs → tumbs 的拼写差异；字母改变并不直接证明声学偏差。 | unknown（正写）；audio-needed |
| `4515-11057-0002` / +1 | REF: `too ignoble`<br>BASE: `too ignoble`<br>CAND: `too ignomable` | **专名/拼写、普通词近音候选**。普通词变为另一串拼写；只记录观察，不给非词来源下结论。 | audio-needed |
| `4572-112375-0006` / +4 | REF: `he ceased` / `prince replied` / `fraught` / `and least of all` / `0 prince`<br>BASE: `he ceased` / `prince replied` / `fraught` / `and list of all` / `0 prince`<br>CAND: `he sees` / `prince reply` / `frowd` / `unlist of all` / `0 prince` | **普通词近音候选、词边界、插删、正常化争议**。新增多处普通词及 and list/unlist 边界变化；共有的 thus/does、shouldst/should 等未修复。原始 O prince 三者均归一为 0，不是新增口述数字。 | audio-needed；O/0 正常化需政策复核 |
| `4831-25894-0012` / +1 | REF: `bella monica` / `merry and` / `tommo`<br>BASE: `bellamonica` / `marian` / `tommo`<br>CAND: `belle amonica` / `marian` / `tomo` | **专名/拼写、普通词近音候选、词边界、插删**。姓名/词组换一种切分，tommo 另少一个 m；merry and → marian 为共同差异。本例是组合变化，不能只按 token 数解释。 | audio-needed；名称和词边界 unknown |
| `6841-88294-0045` / +1 | REF: `holding down`<br>BASE: `holding down`<br>CAND: `holdin down` | **专名/拼写、正常化争议**。词尾 g 的书写消失；是否反映口语转写风格、读音或误拼无法从文本决定。 | audio-needed；转写规范 unknown |
| `7601-101619-0002` / +1 | REF: `metrodorus`<br>BASE: `metrodorus`<br>CAND: `metridor` | **专名/拼写**。专名由长串变短串，但词级仍是替换而非整词删除。 | unknown（专名规范）；audio-needed |
| `7697-105815-0051` / +5 | REF: `the jaw bone` / `the sling and stone` / `goliath of gath`<br>BASE: `a jaw bone` / `the slain and stone` / `goliath of gath`<br>CAND: `a jawn bone` / `the slain and the stone` / `golioth of ga ath` | **专名/拼写、普通词近音候选、词边界、插删**。在共有 the/a、sling/slain 差异之外，又改变 jaw、Goliath，并增加 the、将 Gath 拆成 ga ath。 | audio-needed；专名/分词 unknown |
| `8173-294714-0004` / +2 | REF: `van brandt stung`<br>BASE: `van brandt stung`<br>CAND: `van brant is stung` | **专名/拼写、插删**。Brandt/Brant 拼写改变，另增 is。不能仅凭语法猜测 is 的声学来源。 | unknown（名称规范）；audio-needed |
| `8254-84205-0004` / +2 | REF: `every one was busy`<br>BASE: `every one was busy`<br>CAND: `everyone was busy` | **词边界、插删、正常化争议**。合并 every one 后现行词级计两编辑；需区分标签分词习惯与实际词语选择。 | unknown（词边界/评分规范） |
| `8288-274150-0003` / +2 | REF: `saint aignan`<br>BASE: `saint agnan`<br>CAND: `sight in agnon` | **专名/拼写、普通词近音候选、词边界、插删**。BASE 已有专名拼写差异；CAND 将短语变为 sight in agnon，多一个 token，词语及专名界限更不同。 | audio-needed；专名规范 unknown |
| `8288-274162-0060` / +1 | REF: `poor mouse`<br>BASE: `poor mouse`<br>CAND: `poor mouth` | **普通词近音候选**。mouse → mouth 为普通词替换；不能由字面推断录音的末辅音。 | audio-needed |

### 改善 5 条

| ID / Δ | REF → BASE → CAND（局部） | 可观察分类与说明 | 未决项 |
|---|---|---|---|
| `1650-167613-0016` / -1 | REF: `skillful`<br>BASE: `skilful`<br>CAND: `skillful` | **专名/拼写、正常化争议**。单/双 l 拼写改成 REF 形式；这是正写匹配改善，不足以说明声音识别改善。 | unknown（等价拼写/正常化规范） |
| `1686-142278-0022` / -2 | REF: `to night`<br>BASE: `tonight`<br>CAND: `to night` | **词边界、插删、正常化争议**。按 REF 分写 to night 后少两个词级编辑；尚不能判定合写本身是听辨错误。 | unknown（边界/评测规范） |
| `2506-11278-0033` / -1 | REF: `confidence savors`<br>BASE: `confident savers`<br>CAND: `confidence saves` | **普通词近音候选**。confident → confidence 修复一词；savers → saves 后仍异于 savors。改善与残余错误同时存在。 | audio-needed |
| `6455-67804-0029` / -1 | REF: `fate` / `broadoaks`<br>BASE: `fate` / `broad oaks`<br>CAND: `fay` / `broadoaks` | **专名/拼写、普通词近音候选、词边界、插删、正常化争议**。名称 broadoaks 的并词改善，另新增 fate → fay；净改善掩盖了局部普通词退化。 | audio-needed（fate/fay）；名称边界 unknown |
| `700-122868-0020` / -1 | REF: `seeming an hour` / `lily maid`<br>BASE: `seeming like an hour` / `lilymain`<br>CAND: `seeming like an hour` / `lily mae` | **专名/拼写、普通词近音候选、词边界、插删**。lilymain 分成 lily mae，减少一个词级编辑；maid 仍未匹配，额外 like 也仍在。不能把 Lily Mae 的人名式外观当作已确认人名。 | audio-needed；词/姓名身份 unknown |

### 持平 5 条

| ID / Δ | REF → BASE → CAND（局部） | 可观察分类与说明 | 未决项 |
|---|---|---|---|
| `116-288045-0012` / +0 | REF: `it had only one fault`<br>BASE: `it had only one fault`<br>CAND: `it had only one fault` | **持平一致**。三者正常化后整句相同。 | unknown（音频未核） |
| `1630-73710-0017` / +0 | REF: `do not be angry with your henrietta`<br>BASE: `do not be angry with your henrietta`<br>CAND: `do not be angry with your henrietta` | **持平一致**。三者正常化后整句相同；姓名未变化。 | unknown（音频未核） |
| `1701-141760-0023` / +0 | REF: `0 dear` / `rostov`<br>BASE: `0 dear` / `rostov`<br>CAND: `0 dear` / `rostov` | **持平一致、正常化争议**。原始三者是 Oh/OH dear，正常化同为 0 dear；不能把它当成三者都“听成数字0”的证据。 | unknown（正常化语义）；音频未核 |
| `4570-102353-0006` / +0 | REF: `visitors withdrew dissatisfied`<br>BASE: `visitors withdrew dissatisfied`<br>CAND: `visitors withdrew dissatisfied` | **持平一致**。三者正常化后整句相同。 | unknown（音频未核） |
| `6123-59150-0040` / +0 | REF: `his work like his face`<br>BASE: `his work like his face`<br>CAND: `his work like his face` | **持平一致**。三者正常化后整句相同。 | unknown（音频未核） |

## 从这些案例可以保留的观察

退化案例不是单一类型：例如 clean 的 `1673-143397-0002` 出现 animadversion → animated version，other 的 `1585-157660-0008` 出现一组普通词替换；另有 Gryce/Grice、Rossetti/Rosetti、Geraint/gerai not 等名称拼写和分界问题。`4572-112375-0006`、`7697-105815-0051` 的多处变化值得回听，但本审阅不据此推断噪声、口音、说话人特征或机制。

改善也真实存在于既定文本计分：clean 的 `6313-66129-0007` 将 afraid 改成匹配 REF 的 free；other 的 `2506-11278-0033` 修复 confidence，却仍未匹配 savors。`6455-67804-0029` 以名称边界改善抵消新增 fate/fay 差异，说明句级净改善不能代替局部审阅。

若干分差直接落在正写/分词：bedtime/bed time、offhand/off hand、passerby/passer by、nearby/near by、tonight/to night、everyone/every one、etc/et cetera，以及 skilful/skillful。它们应作为正常化敏感性检查的明确案例，同时继续保留原始主指标；不能选择性豁免退化而保留同类改善。

这 10 条持平样本全部是三者正常化后整句相同，并非“错误相互抵消”的持平。原始 O/Oh 在 `1701-141760-0023`、`4572-112375-0006` 的正常化文本中成为 0，提醒后续审阅区分归一化产物与原始解码。该现象在相关 BASE/CAND 中相同，不能用来解释两者分差。

输入标志显示 CAND 的全大写样式在 clean 为 30/30、other 为 29/30，BASE 为 0/30、0/30；这是样本内格式观察，正常化文本已小写，不能直接作为 WER 变化、语言能力损失或“遗忘”的证据。输入的空串、前缀泄漏、非参考三元组重复、超长词数及截断标志在这 60 条均未置位；这些标志本身也不替代听音或全量审查。

下一步若做音频复核，应同时保留退化、改善与持平对照，优先覆盖普通词替换、名称及边界的混合案例；回听时将“实际说了什么”“应采用何种正写/词界”“现行正常化如何计分”分别记录。在有复核前，音频真值和变化机制仍为 unknown。

## 输入指纹

| 文件 | SHA-256 |
|---|---|
| `dev-clean.review-samples.jsonl` | `2e2d9fd876c769e909163cf714f77dc7816d6b3f7fc0ada3b12713293789b7b7` |
| `dev-other.review-samples.jsonl` | `e8ecd16a283dfa88e4b34c95b1e004e63e460b229e764bb98249c9083cc43b36` |
