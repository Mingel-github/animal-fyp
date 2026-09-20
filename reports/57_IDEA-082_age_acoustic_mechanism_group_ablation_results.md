# IDEA-082：MeowAgeNet 年龄声学机制分组消融结果

日期：2026-09-18  
正式结论：**四个候选组均未通过“双门”，不能确认完整 C1 增益来自任何单一预注册机制组。**  
最强探索性线索：G3 spectral-energy/voice-quality 通过 real−A0 实用门，但未通过 real−shuffled 信息门。

## 1. 执行与审计

- 完成 9 pipelines × 3 base seeds × 3 repeats × 4 folds = **324/324 fits**。
- 每个 seed/repeat/fold 只运行一次 A0，并由八条分组管线共享；完整 20 维 C1 未重跑。
- checkpoint 固定为最低 animal-level validation CE；训练损失仍为 globally class-balanced call CE。
- 首个完整 cell 的九条管线在下一 cell 前强制暂停，身份、预测哈希、共同初始化、四对 real/shuffled state、role-local derangement、非有限值、checkpoint reload、batch coverage 与 outer-test 标志全部通过。
- 独立审计重新读取 324 份 fit summary 和 648 份预测文件，独立重算 36 个 fold、9 个 seed×repeat、12 个 split-cell 以及所有 gate；与 runner summary 一致。
- 未生成或读取 outer-test prediction/metric；未发现 outer-test-like 输出。

## 2. 九条管线的 seed×repeat 等权均值

| Pipeline | Macro-F1 | Balanced accuracy | Animal CE ↓ | Brier ↓ |
|---|---:|---:|---:|---:|
| A0 AST only | 0.739914 | 0.765741 | 0.686022 | 0.407149 |
| G1 F0 real | 0.747839 | 0.770370 | 0.697082 | 0.409960 |
| G1 F0 shuffled | 0.740041 | 0.763889 | 0.681794 | 0.404699 |
| G2 stability real | 0.745958 | 0.770370 | 0.685984 | 0.405316 |
| G2 stability shuffled | 0.745779 | 0.775000 | 0.681334 | 0.403396 |
| G3 spectral-energy real | **0.756020** | **0.782407** | **0.679533** | **0.398036** |
| G3 spectral-energy shuffled | 0.736361 | 0.780556 | 0.696874 | 0.412096 |
| G12 F0+stability real | 0.741861 | 0.776852 | 0.696001 | 0.411226 |
| G12 F0+stability shuffled | 0.749788 | 0.780556 | 0.683051 | 0.404908 |

这些是 9 个等权 seed×repeat 估计的均值；不能把重复动物出现或 324 个 fit 当作独立样本。

## 3. 预注册机制门

每组必须同时通过：信息门 `real−matched shuffled` 与实用门 `real−A0`，才能标记 `source_supported=true`。

| 组 / 对照 | ΔMacro-F1 | 正 seed×repeat | 正 base seed | 非负 split-cell | 最差 split | Gate |
|---|---:|---:|---:|---:|---:|---|
| G1 real−shuffled | +0.007799 | 5/9 | 1/3 | 7/12 | −0.072274 | FAIL |
| G1 real−A0 | +0.007925 | 5/9 | 2/3 | 6/12 | −0.068955 | FAIL |
| G2 real−shuffled | +0.000179 | 4/9 | 1/3 | 6/12 | −0.031898 | FAIL |
| G2 real−A0 | +0.006044 | 5/9 | 1/3 | 7/12 | −0.030341 | FAIL |
| G3 real−shuffled | **+0.019658** | **5/9** | **2/3** | **8/12** | **−0.051064** | **FAIL** |
| G3 real−A0 | **+0.016106** | **6/9** | **2/3** | **10/12** | **−0.014579** | **PASS** |
| G12 real−shuffled | −0.007927 | 5/9（另 1 tie） | 1/3 | 7/12 | −0.067754 | FAIL |
| G12 real−A0 | +0.001947 | 4/9 | 2/3 | 7/12 | −0.091488 | FAIL |

### G1：F0 level/contour

两个平均 ΔMacro-F1 都超过 +0.005，但 seed-repeat、split-cell、最差 split 和校准门未通过。real−A0 的 base-seed senior recall 最差为 −0.0333，也低于 −0.02 安全界。因此不能把完整 C1 的方向性信号归因给 F0 level/contour。

### G2：source stability/periodicity/voicing/HNR

real−shuffled 几乎为零（+0.000179）；real−A0 虽为 +0.006044，但只有 1/3 base seed 与 5/9 seed×repeat 为正，最差 split 为 −0.030341，刚刚越过预注册的 −0.03 下界，senior recall 亦在一个 base seed 上为 −0.0333。G2 不受支持。

### G3：spectral-energy/voice-quality

G3 是唯一通过完整实用门的组：real−A0 为 +0.016106，2/3 base seed、6/9 seed×repeat、10/12 split cells 支持，最差 split −0.014579；平均 CE、Brier、balanced accuracy 均不劣于 A0，所有 base seed 的 senior recall 安全条件也通过。

但核心信息门仍失败。real−shuffled 虽有更大的平均增益 +0.019658，且 CE/Brier/BA 均不劣、2/3 base seed 和 8/12 split cells 支持，却只有 5/9 seed×repeat 为正，且最差 split 为 −0.051064，低于 −0.03 下界。因此按预注册规则，不能排除该优势含有容量、优化路径或不稳定配对效应；`source_supported=false` 必须保留。

### G12：F0 + stability

G12 的 real−shuffled 均值为负，real−A0 均值也低于 +0.005，且 split 稳定性、校准和 senior safety 均未形成完整支持。组合并未恢复完整 C1 的稳定性。

## 4. 组合的描述性对照

这些对照预先声明为描述性，不参与 gate：

| 对照 | 平均 ΔMacro-F1 | 正 / tie / 负 |
|---|---:|---:|
| G12 real − G1 real | −0.005978 | 3 / 1 / 5 |
| G12 real − G2 real | −0.004097 | 3 / 0 / 6 |
| G12 − G1 − G2 + A0 | −0.012022 | 4 / 0 / 5 |

没有观察到 G1+G2 的一致增量或正向冗余交互；这进一步反对把完整 C1 的平均正向结果简单解释为 F0 与稳定性信息相加。

## 5. 结论与下一步边界

1. **正式结论是 NO-GO for source attribution**：G1、G2、G3、G12 均未同时通过信息门和实用门。
2. **G3 是唯一值得保留的探索性线索**：它对 A0 的性能、校准和 senior safety 全面通过，但 matched shuffled 稳定性不足，所以只能描述为候选线索，不能称为确认的年龄声学机制。
3. 结果不支持在同一数据上继续通过改组、挑 seed、调宽度或重设阈值修复结论。若继续，应使用外部/前瞻数据，或在新数据上原样验证冻结的 G3 对照，而不是回到 MeowAgeNet 结果驱动调参。
4. 本实验只回答 20 维既有代理特征的分组归因；`period_variation_proxy`、`amplitude_variation_proxy`、spectral tilt/flatness 等并非纯粹、直接的生理测量，因此即使 G3 将来通过，也仍需谨慎表述其机制含义。

## 6. 固定结果哈希

- protocol：`0b5e4a4c0be591925811a42769555cab7e5ac9ba8cd2781ac91d1d341e782022`
- runner：`2ce03190d3079cbfe34fbdd140288b2309dfa12ace54f8d21279e66fa7f2c415`
- initial evaluation summary：`2fb10f2969efc577a3004562eccd4c97d35db41568eff6caf2bd9cc768c1fe44`
- compact run summary：`b450cdb623115c877a14c815b47228bf8faab6a79f4327d1b474cfd9e7abc51b`
- run manifest：`b55fa4e1206b66780f160979447f655dc57269ce19ecdc62cb75cc9af45befce`
- first-cell audit：`6449debdb08b7ac2589a07060e3d8134a20ab70420fc503225fc4b247d1514fc`
- independent results audit：`beb59534d4c65f1fcf4209c16e90e96efc28bef40e27621b2f306a443d94506a`
- independent verifier：`73a9a4d28be51328dc230d1c7df25a9f0a034dcc6ae7fe892184c3f11d04fc0c`

