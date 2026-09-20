# IDEA-086 独立 CPU 预检与结果审计

日期：2026-09-19  
审计角色：牛马1（独立只读审查与复算）  
结论：**工程完整性 PASS；两个预设主 classification gate 均 FAIL。**

## 1. 锁定对象

- Protocol SHA-256：`df576b4c0eb28d7dc524202aa15b40a563d1909a57e9460dbcbf21cf7ee4a527`。
- Runner SHA-256：`33e02d175f42c3d799bc4578f6c0dbd59b5eb747f3398821b8d91137dfdbc472`。
- Tests SHA-256：`a9533145f2d6abb63afb25914b34f47577e0aa045ba0613bc0aac9eed59d118b`。
- CPU preflight SHA-256：`fb1f49ee1f279e9d79631d068ffec862854a179225d0f0a7abdd60d5c69f968c`。
- IDEA-085 reuse manifest SHA-256：`afec92014e172615433e76a5128e060f1dae6da7ecbb21f4905c985020891898`。
- 方法/文献审查见 `reports/71_IDEA-086_literature_history_and_method_review.md`，SHA-256 `db834cb34f6af373fe189deadedb569429f53407909a56ee32bcd30ba2cf3e85`。

上述 protocol 锁定 runner/tests 自身哈希，并执行 IDEA-085 的完整父依赖验证链。正式 manifest 与 protocol、runner、CPU preflight、reuse manifest 和 director authorization 一致；outer-test 预测或指标未访问。

## 2. 独立 CPU 预检

### 模型、初始化与预处理

- SET1 结构为共享逐帧 `Linear(12,48)+GELU → Linear(48,48)+GELU → masked mean → zero-init Linear(48,128)`；没有 Conv1d、位置编码、attention 或 variance pooling，也不叠加 C1 分支。
- 分支参数独立复算为 `9,248`，总可训练参数为 `108,323`，比 T1/J1 少 `32`。
- common AST head 与 A0、T1 的初态一致；zero-init SET1 相对 A0 的最大 logit 差为 `0`。训练角色的 median/mean/std 与 IDEA-085 T1 相同，post-build RNG reset 探针一致。
- 六个 finite indicators 由对应原始帧在排列后同行重建；lookup 列只用于 ragged retrieval，不进入 encoder。轨迹仍为 792 calls、57,848 frames，长度 `9/70/443`，缺失 F0 帧保留。

### 非空洞排列不变性

CPU 使用非零随机 frame encoder 和非零 projection 探针，而非只比较零残差 logits：

- 原序 context norm `1.12174`，residual norm `0.034724`；active projection norm `1.35787`。
- 真实 mixed-length 序列为 `16/177/75` 帧；原序、逆序和 IDEA-085 J1 固定联合排列均通过 `atol=rtol=2e-5`。
- 最大 context 差为 `2.98e-8`，最大 residual 差为 `4.66e-10`，最大 logit 差为 `0`。
- padding 审计覆盖 9 帧与 443 帧调用，单独/混合 batch context 最大差 `2.98e-8`。

训练后的 36 个 SET1 checkpoints 也全部保留实际排列不变性：所有 probe 均 PASS；projection norm 最小 `1.2900`，native context/residual norm 最小 `1.3147/0.4412`，三类最大差分别为 context `2.38e-7`、residual `1.19e-7`、logit `4.77e-7`，均低于锁定容差。该不变性只针对固定 frozen AST call embedding 下的辅助声学帧集合；AST 主路径仍可携带时序。

### 梯度、残差上限与复用保护

- zero-init projection bias 梯度 norm `33.94`；两层 frame Linear 在 zero-init 时梯度为 `0`，非零 projection 探针下梯度 norm 为 `10.88/21.15`，符合零初始化学习路径。
- 饱和探针最大相对残差 `0.25000006`，符合共享 `0.25×RMS(h)` 上限的浮点容差。
- 144 个 IDEA-085 fit summaries、288 个预测文件，共 432 个复用文件逐项哈希通过；CPU 重新完成 144 次 call→cat 重建、角色/标签/概率检查。
- 总监冻结的 450 文件清单也逐文件独立重算，数量仍为 450，组合 SHA-256 仍为 `161e2357e544eaffed62b458f8d25eedc4b360a60000f02ca36c09fa7db89af3`。
- 首个训练 fit 的实际 checkpoint permutation probe、预测、call→cat 重建与共同 epoch batch 顺序均通过；随后全量 36 fits 的 initialization 与共同 batch history 也由独立 verifier 逐 cell 复核。

CPU/首 fit 阶段未发现工程阻断；总监的分阶段 GPU 授权与最终运行均发生在独立结果解释之前，不以首 fit 成绩择停。

## 3. 最终预测完整性

独立 verifier 不导入 IDEA-086 runner，也不调用其 `aggregate`。它读取原始预测并完成：

- 36 个新 SET1 fits + 144 个 IDEA-085 只读 fits = `180` fit summaries；
- 72 个新预测文件 + 288 个旧预测文件 = `360` prediction CSV；
- `180/180` 次由 call probabilities 独立重建 cat probabilities、call count 与 argmax；
- 所有概率有限、在 `[0,1]` 内、行和为 1；所有 role、call/cat identity、标签、full seed、checkpoint reload 与 outer-test 标志通过；
- 五条管线在每个 cell 的 validation cats/labels 完全配对；
- 独立复算 Accuracy、Macro-F1、BA、三类 recall、CE、Brier、3/9/12 稳定性、四组纠错转移和两条主 gate；
- 与正式 `initial_evaluation_summary.json` 逐字段差异 `0`，数值容差 `1e-12`。

## 4. 九个 seed×repeat 等权均值

| Pipeline | Accuracy | Macro-F1 | BA | CE | Brier | kitten recall | adult recall | senior recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SET1 acoustic set | 0.71242 | 0.72743 | 0.76204 | 0.70782 | 0.42449 | 0.91667 | 0.68611 | 0.68333 |
| A0 AST only | 0.72222 | 0.73622 | 0.75741 | 0.68992 | 0.41245 | 0.88889 | 0.71667 | 0.66667 |
| C1 summary residual | 0.71242 | 0.72547 | 0.75093 | 0.70649 | 0.41817 | 0.88889 | 0.70278 | 0.66111 |
| T1 local temporal | 0.72059 | 0.73328 | 0.77037 | 0.71653 | 0.42573 | 0.93056 | 0.69722 | 0.68333 |
| J1 frame shuffled | 0.71895 | 0.73371 | 0.76852 | 0.71238 | 0.42179 | 0.93056 | 0.69722 | 0.67778 |

## 5. 两条主 classification gate

| 比较 | mean ΔMacro-F1 | 3个 base-seed 均值为正 | 9个 seed×repeat 为正 | 12个 split 非负 | worst split | Gate |
|---|---:|---:|---:|---:|---:|---|
| SET1−A0 | −0.00879 | 0/3 | 4/9 | 6/12 | −0.11143 | **FAIL** |
| SET1−C1 | +0.00195 | 2/3 | 5/9（另 1 tie） | 7/12 | −0.08148 | **FAIL** |

锁定门槛要求 mean delta `≥0.005`、至少 `2/3` base-seed 均值为正、至少 `6/9` seed×repeat 为正、至少 `8/12` split 非负、worst split `≥−0.03`，且每条比较五项同时满足。

- SET1−A0 五项全部失败；各 base-seed 均值为 `−0.02467/−0.00056/−0.00116`。
- SET1−C1 只有 base-seed 条件通过；平均增益不足，稳定性和 worst-split 防线均未满足。
- 不存在单一全局 gate；Accuracy、BA、recall、CE、Brier 均为辅助剖面，不改变主 gate 判定。

## 6. 预设辅助比较

| 比较 | mean ΔMacro-F1 | seed×repeat 正/平/负 | base-seed 均值为正 | split 非负 | worst split | Gate |
|---|---:|---:|---:|---:|---:|---|
| SET1−T1 | −0.00586 | 2/2/5 | 1/3 | 6/12 | −0.05109 | 不适用 |
| SET1−J1 | −0.00628 | 3/2/4 | 0/3 | 6/12 | −0.05109 | 不适用 |

这两条比较按预注册只作辅助描述，不能事后套用主 gate。SET1 与 T1/J1 是少 32 参数的近似匹配方法比较，不是保持同一结构只删除时间的纯因果消融；SET1 较低也不能反向证明时序普遍有益。

## 7. 辅助指标与纠错

- SET1−A0：Accuracy `−0.00980`、BA `+0.00463`、CE gain `−0.01789`、Brier gain `−0.01204`；纠正 11、引入 17、净 `−6`。
- SET1−C1：Accuracy `0`、BA `+0.01111`、CE gain `−0.00132`、Brier gain `−0.00632`；纠正 19、引入 19、净 `0`。
- SET1−T1：Accuracy `−0.00817`、BA `−0.00833`、CE gain `+0.00871`、Brier gain `+0.00125`；纠正 9、引入 14、净 `−5`。
- SET1−J1：Accuracy `−0.00654`、BA `−0.00648`、CE gain `+0.00456`、Brier gain `−0.00270`；纠正 8、引入 12、净 `−4`。

负 CE/Brier gain 表示 SET1 的对应概率质量更差。SET1 的 kitten recall 较 A0/C1 各高 `0.02778`，但 adult recall 分别低 `0.03056/0.01667`，局部类别交换没有转化成主指标 gate。

## 8. 重复出现与解释边界

每条管线 pooled validation 均为 `612` 个重复 animal occurrences、`97` 个 unique cats；每只猫出现次数最小/中位数/最大值为 `3/6/15`。612 行不是 612 个独立动物，数据全集仍为 111 cats。

结果仅说明当前固定的共享逐帧 MLP + mean 残差没有达到 A0/C1 的预注册成功标准。非线性帧 embedding 的均值是经验分布摘要，但不完整保留所有分布性质，也不能恢复全轮廓；本轮不能声称完整声学分布、普适集合逼近、新发明 Deep Sets、时间信息有害或外部确认。

## 9. 产物与验证

- 独立 verifier：`scripts/verify_idea086_acoustic_set_results.py`，SHA-256 `aa4d10c3e55994dd237509b5c9d9f587318e31a5281a491487b051ef931ac888`。
- 独立 verifier tests：`tests/test_verify_idea086_acoustic_set_results.py`，SHA-256 `645b9b6a66585ae06d64fbf572767e7f4f98f3d5368ab0ca5fd17aa1f04fbfc4`。
- 独立结果 metadata：`runs/meowagenet_idea086_acoustic_set_residual_v1/independent_results_audit.json`，SHA-256 `7f05d5c66da9e62b4af983de63be893f848972add2b51996e38ab0795c027252`。
- 正式 summary SHA-256：`487e0ad0f86570cbc9ac44d52a11bbf4f17baa7033d6a9df437a1139aeb0f253`；run summary SHA-256：`0ee04d2b498a79fe483a99e48b0b71f881eeee93f49da9450ab9bec66039a192`；run manifest SHA-256：`6ad421f337f937995b44c60acb9dccc2880bc120cceb82bfb64eab178bc169bd`。
- IDEA-086 runner 与独立 verifier 专项测试：`15 passed in 10.51s`；连同 IDEA-085 依赖测试的联合运行此前为 `36 passed in 13.74s`。

最终审计决定：**接受 IDEA-086 产物为完整、可复现的探索性负结果；不保留 SET1 为通过主 gate 的候选，不做结果驱动的架构、seed 或门槛修补。**
