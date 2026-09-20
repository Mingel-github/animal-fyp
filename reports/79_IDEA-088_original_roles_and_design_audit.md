# IDEA-088 原始角色恢复与设计独立审查

日期：2026-09-20  
状态：角色冻结完成；正式训练未获授权

## 审查结论

IDEA-088 可以在不重新随机分猫的前提下恢复原 baseline 风格角色。角色的唯一来源是
2026-08-25 本机复现日志中实际打印的、完成 `000A`/`046A` 交换之后的 5 seeds ×
4 folds 训练与测试猫清单。清洗投影只从每个角色中删除重复别名 `049A`；没有恢复重复
音频，没有重新运行分折器，也没有把后续 111 猫固定协议代替原角色。

冻结产物：

- `runs/meowagenet_idea088_original_style_clean_v1/roles/original_clean_roles.json`
- 解析器：`scripts/freeze_idea088_original_roles.py`
- 独立测试：`tests/test_idea088_original_roles.py`

角色 JSON SHA-256：`a0ce4989985989ebdd982bbf565e33c080b4e314e232eb875f30fbdb4eac34b8`。

冻结 JSON 同时绑定原复现日志、音频 manifest、猫 manifest 与官方 VGGish CSV 的
SHA-256。每个 cell 保存删除 `049A` 后、已排序去重的 pre-swap 审计名单和 post-swap
实际训练名单；原始 112-ID 证据由逐 cell 日志行号与源日志哈希绑定，不把 049A 写回任何
可被 runner 消费的角色数组。

## 数据边界

三种数量不能混写：

| 视图 | 样本单位 | 行数 | 猫 ID |
| --- | --- | ---: | ---: |
| 官方音频 | cropped call | 793 | 112 published IDs |
| 官方 VGGish | 0.96 秒 embedding window | 937 | 112 published IDs |
| IDEA-088 清洗音频 | unique call | 792 | 111 analysis IDs |
| IDEA-088 清洗 VGGish | embedding window | 936 | 111 analysis IDs |

`0Y-049A.wav` 与 `0Y-041A-01.wav` 字节相同。IDEA-088 按用户选择保留 `041A`、排除
`049A`。937 行 VGGish CSV 没有 call filename；因此 VGGish 只能把同猫 embedding-row
概率直接平均，不能声称先恢复为 call 概率再做猫均值。

原 categorical 脚本先向 CSV 追加 `age_group`，再以 `dataframe.iloc[:, :-4]` 取输入。
CSV 原列为 128 维 embedding 加 `mean_freq, gender, target, cat_id`，所以原 final 实际保留
`0..127 + mean_freq`，输入是 **129 维**，不是仅 128 维。IDEA-088 的 VGGish 必须保留
`mean_freq`，且所有 129 维标准化参数只由当前训练角色拟合。

类别边界继续使用原实际三分类定义：`0≤age<0.5` 为 kitten、`0.5≤age<10` 为 adult、
`10≤age<20` 为 senior。清洗后猫数为 15/62/34，call 数为 134/405/253；这与现有 AST
任务标签一致。原 sklearn `LabelEncoder` 的整数顺序是 `adult, kitten, senior`，而本轮
统一概率列顺序是 `kitten, adult, senior`。只要保存并正确应用映射，整数顺序变化不改变
语义标签；不要求人为追求历史 VGG 70.0136% 的 bitwise 重现。

每个清洗后 cell 均满足：训练/测试猫不交；并集严格为 111 猫；训练与测试均含 kitten、
adult、senior；两侧合计严格覆盖 792 calls 与 936 VGGish rows；`000A`、`046A` 始终在
训练角色且从不进入测试。20 个 cell 的训练猫数范围为 81～86，测试猫数范围为 25～30；
这些是原日志清洗投影后的实际范围，不能套用 IDEA-087 的角色数量。

## 原交换造成的非 OOF 边界

原始交换规则有意把 `000A`、`046A` 保留在训练集，并把同龄替换猫放入测试集。因此，
即使删除 `049A`，每个 seed 的四折也并非完整 OOF：

- `000A`、`046A` 的 test 次数均为 0；
- 每个 seed 有两只替换猫 test 两次；
- 其余猫 test 一次；
- 四折共有 111 个 test entries，但只有 109 个 unique test cats。

重复测试猫依次为：seed 7270 的 `026C`/`110A`，seed 860 的 `087A`/`110A`，seed
5390 的 `022A`/`048A`，seed 5191 的 `010A`/`045A`，seed 5734 的 `002A`/`111A`。
因此主汇总必须对 20 个 fold metric 等权平均，并另列五个 seed 的四折均值；不得把跨折
拼接预测描述成完整 OOF，也不得用 pooled-cat 指标替换主汇总。

## 三模型固定设计

本轮只有 VGGish、A0、C1，共 `5 × 4 × 3 = 60 fits`，无额外 arms、HPO 或结果后搜索。
三者固定采用原 final 风格训练配方：

- VGGish 使用 129 维输入与 128 单元 ReLU 分类头、BatchNorm、Dropout、三类输出，
  优先走原 Keras CPU 实现；A0/C1 保持已有 Torch 结构；
- hidden 128，dropout `0.44571035356880917`；
- Adamax，learning rate `0.003109800273709165`，epsilon `1e-7`，betas `0.9/0.999`，
  无 weight decay；
- batch size 128；VGGish 的 batch 单位为 embedding rows，A0/C1 为 calls；
- 每 epoch shuffle，最多 1500 epochs；
- 仅监控 training loss，`min_delta=0.001`、`patience=30`、恢复 training-loss 最佳权重；
- 不设 validation，不调参，不以 test 指标早停；
- class weights 仅由本 cell 的 training rows/calls 估计；
- class-weighted CE 必须以 batch 样本数为分母，复现 Keras 语义，不能使用 PyTorch
  `CrossEntropyLoss(weight=...)` 默认的权重和归一化；
- C1 保留已确认的 width 60、cap 0.25，不改变结构。

这里的训练配方与 IDEA-087 不同。现有使用 validation checkpoint、不同 learning rate、
不同 batch size 或 PyTorch 默认 weighted-mean CE 的代码不能未经修改直接复用。

## 指标契约

公共主指标是每折 **cat probability-mean macro-F1**：

- VGGish：同猫 embedding-row softmax 概率直接平均；
- A0/C1：同猫 call softmax 概率平均；
- 每折独立计算后，对 20 folds 等权平均；另列五个 seed 均值与配对 `C1−A0`。

各模型 native-unit macro-F1 另列：VGGish 是 embedding-row 单位，A0/C1 是 call 单位，
二者不可混为同一观测单位。旧复现的约 0.7001 是 window-row macro-F1，也不能直接当作
本轮 cat probability-mean 主指标。

## 预检决定

当前只批准冻结角色和审查 runner。正式 60 fits 仍未获授权。牛马2的 runner 到达后，独立
预检至少要确认：只消费冻结 JSON；没有分折器；三模型同 cell 猫角色一致；049A 不可达；
VGGish 实际读取 129 维且标签映射可逆；训练损失按样本数跨 batch 汇总；EarlyStopping 的
`min_delta`、patience、最佳 checkpoint epoch 与恢复权重符合 Keras 语义；VGGish/AST
输入单位与公共猫级指标分别实现；
首个完整 fit 的日志、checkpoint 与预测可由独立复算脚本验证。满足这些条件后，才可向
研究总监申请正式训练放行。
