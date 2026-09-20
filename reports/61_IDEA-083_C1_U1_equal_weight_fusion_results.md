# IDEA-083：C1 / U1 固定等权概率融合结果

日期：2026-09-19  
分析性质：**事后探索性、只读历史 inner-validation predictions、零训练、零 GPU、outer-test 未访问**  
固定公式：`pE = 0.5 × pC1 + 0.5 × pU1`

## 1. 结论

IDEA-083 得到一个**可作为“分类—概率质量折中”保留的探索候选**；它具有实用的平均折中信号，但不构成已确认晋级。

在唯一主分析 IDEA-076 的 18 个 `base_seed × repeat` 单元中，固定等权融合 E 的平均 Macro-F1 为 `0.752846`，几乎保持 U1 的 `0.753434`，差值仅 `−0.000588`；同时 E 的 animal CE 为 `0.694126`，比 U1 的 `0.714473` 数值上降低并接近 C1 的 `0.693010`，Brier 为四条管线最低的 `0.408584`。因此，“接近 U1 分类表现，并在平均数值上恢复 C1 概率质量”的中心趋势成立。本轮没有独立动物层面的显著性检验。

但它不是全面或稳定胜出：E−U1 的 18 个 Macro-F1 差值为 `6` 正、`7` 平、`5` 负，最差 seed×repeat 为 `−0.051056`；E 的 CE 仍比 C1 略差 `+0.001116`。E 相对 A0 虽平均提高 `+0.013903`、`13/18` 为正，最差共同 split-cell 仍为 `−0.047226`。所以最准确的判读是：**固定等权融合能在平均意义上部分兼顾 U1 的 F1 与 C1 的概率质量，但没有形成跨单元稳定支配，也不改变 IDEA-076 的 gate 失败和 v3 模型层级。**

错误互补也有限。1,224 个 animal occurrences（实际涉及 97 只 unique cats，动物会跨 seed/repeat/fold 重复）中，C1 错/U1 对为 34，U1 错/C1 对为 26，但两者同时错为 295，错误 Jaccard 重叠为 `0.8310`。E 相对 C1 实际纠正 21、损坏 11，净增 10；相对 U1 纠正 15、损坏 13，净增仅 2。这说明平均融合主要改善概率折中，离散分类互补空间较小。

## 2. 锁定范围与统计单位

主分析只读取 `runs/meowagenet_idea076_C1_final_seed_confirmation_v1/` 中已保存的 A0、C1、U1 validation predictions：

- 6 个 base seeds × 3 repeats × 4 folds = 72 个 paired fold cells；
- 18 个 `base_seed × repeat` 是主要等权汇总单位；
- 12 个 `repeat × fold` 共同 split-cells 先在 6 个 seeds 上平均；
- 1,224 个 animal occurrences 来自 97 只 unique cats，同一只猫可跨 seed、repeat，甚至不同 fold 再次出现，不能把 1,224 当作独立猫样本；
- 行主键使用 `base_seed / repeat / fold / cat_id`，call 表再加入 `call_index / call_id`。

IDEA-072 与 IDEA-073 只作两个分开报告的历史敏感性分析，不参与选择主轮次、权重或结论。融合权重始终为 `0.5 / 0.5`，没有 alpha 搜索、类别权重、按猫路由或 oracle 路由。

## 3. 主分析：IDEA-076

### 3.1 四条管线的 seed×repeat 等权均值

| 管线 | Macro-F1 | Balanced accuracy | Animal CE ↓ | Brier ↓ | Senior recall |
|---|---:|---:|---:|---:|---:|
| A0 | 0.738943 | 0.769444 | **0.690224** | 0.413578 | 0.677778 |
| C1 | 0.747932 | 0.778241 | 0.693010 | 0.410765 | 0.683333 |
| U1 | **0.753434** | **0.786574** | 0.714473 | 0.414171 | 0.688889 |
| E = 0.5 C1 + 0.5 U1 | 0.752846 | 0.786111 | 0.694126 | **0.408584** | **0.691667** |

E 与 U1 的 Macro-F1 相差不到 `0.0006`，balanced accuracy 相差不到 `0.0005`；E 的 CE 比 U1 降低 `0.020347`，Brier 降低 `0.005587`。相对 C1，E 的 Macro-F1 提高 `0.004914`、balanced accuracy 提高 `0.007870`、Brier 改善 `0.002181`，但 CE 增加 `0.001116`。因此 E 的确把两个 parent 的不同优势拉近，却没有在每项指标上同时不劣。

### 3.2 配对方向与 split 尾部

差值定义为 `E − reference`；CE/Brier 的负值表示 E 更好。

| 比较 | Δ Macro-F1 | seed×repeat 正/平/负 | Δ BA | Δ CE | Δ Brier | Δ senior recall | 非负 split | 最差 split |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E−A0 | +0.013903 | 13/0/5 | +0.016667 | +0.003902 | −0.004993 | +0.013889 | 9/12 | −0.047226 |
| E−C1 | +0.004914 | 8/2/8 | +0.007870 | +0.001116 | −0.002181 | +0.008333 | 10/12 | −0.008707 |
| E−U1 | −0.000588 | 6/7/5 | −0.000463 | −0.020347 | −0.005587 | +0.002778 | 6/12 | −0.017844 |

E−C1 的均值为正，但严格正与严格负各 8 个；E−U1 的均值近零且 7 个单元完全同分。固定平均不是一个稳定的离散分类提升器。它最清楚的收益在概率质量：相对 U1 的 CE/Brier 明显改善，相对 C1 的 Brier 继续改善。

概率平均的凸性 sanity 在 18/18 个单元通过：每个单元均满足 `CE(E) ≤ [CE(C1)+CE(U1)]/2` 和 `Brier(E) ≤ [Brier(C1)+Brier(U1)]/2`。这验证了实现，但不等于 E 必须胜过任一单模型。

### 3.3 预定义 12 个 split-cell 的较差划分

这里的“绝对 split Macro-F1”先在每个 `repeat × fold` 内对 6 个 base seeds 的 fold Macro-F1 等权平均。四条管线各自最差的 cell 不一定相同，因此下表用于描述尾部水平，不能把不同 cell 的最差值作严格配对效应。

| 管线 | 最差 cell | 最差绝对 Macro-F1 |
|---|---|---:|
| A0 | repeat 2 / fold 0 | 0.621060 |
| C1 | repeat 2 / fold 3 | 0.640583 |
| U1 | repeat 1 / fold 1 | 0.661775 |
| E | repeat 2 / fold 3 | **0.674358** |

E 的最差绝对值高于 A0、C1、U1 各自的最差值，是一个有利的描述性尾部信号，但各自最差 cell 不同，不能称为配对保护门通过。

在 **U1 自己最差**的 repeat 1 / fold 1，A0、C1、U1、E 的 Macro-F1 分别为 `0.673522 / 0.677156 / 0.661775 / 0.677694`；E 相对 U1 改善 `+0.015919`，相对 C1 也高 `+0.000538`。这说明固定平均在 U1 的最弱 cell 确实由 C1 拉回。

反过来，在 **E 自己最差**的 repeat 2 / fold 3，C1、U1、E 分别为 `0.640583 / 0.692201 / 0.674358`：E 相对 C1 改善 `+0.033775`，却相对 U1 损失 `−0.017844`。这正是固定平均的取舍——它可抬高较弱 parent，却也会把较强 parent 向中间拉回。该观察只回答“较差划分发生了什么”，不用于改权重或选择性路由。

## 4. 错误互补与实际纠正/损坏

### 4.1 Animal-occurrence 总表

| C1 / U1 状态 | Occurrences |
|---|---:|
| 两者都对 | 869 |
| C1 错、U1 对 | 34 |
| U1 错、C1 对 | 26 |
| 两者都错 | 295 |
| **合计** | **1,224** |

两者预测标签不同的 occurrence 仅 65 个，预测相同为 1,159 个；在 295 个共同错误中，290 个甚至给出相同错误标签。C1 与 U1 的错误交集为 295、并集为 355，Jaccard 重叠 `295/355 = 0.8310`。这解释了为什么概率平均能改善 CE/Brier，却只带来很小的离散正确数变化。

### 4.2 E 的实际作用

| 参照 | E 纠正 parent 错误 | E 损坏 parent 正确 | 净正确变化 |
|---|---:|---:|---:|
| C1 | 21 | 11 | +10 |
| U1 | 15 | 13 | +2 |

至少一个 parent 正确的 occurrence 有 929 个；E 实际正确 905 个，因此仍有 24 个“某一 parent 正确、固定平均却错误”的 occurrence。`929/1224 = 0.7590` 只是 oracle 潜力覆盖上限：它假设事后知道该信任哪个 parent，**不是模型成绩、不是可部署准确率，也没有参与模型排名或路由设计。**

### 4.3 Seed×repeat 分表

18 个单元的互补、纠正与损坏计数完整保存在 `seed_repeat_complementarity.csv`。主报告不把这些 repeated occurrences 当作独立动物做显著性推断；分表只用于定位哪些初始化/划分产生有限互补。

## 5. 历史敏感性分析（不参与选择）

### 5.1 IDEA-072

| 管线 | Macro-F1 | CE ↓ | Brier ↓ |
|---|---:|---:|---:|
| A0 | 0.717597 | **0.709855** | 0.420344 |
| C1 | 0.726709 | 0.714212 | 0.417395 |
| U1 | 0.728289 | 0.729000 | 0.421963 |
| E | **0.731901** | 0.715155 | **0.416597** |

E−A0 为 `+0.014304`（7/9 正），E−C1 为 `+0.005192`（4 正/2 平/3 负），E−U1 为 `+0.003612`（5/9 正）。E 的 Brier 最低，但 CE 仍比 C1 高 `0.000943`。在 612 个 occurrences（97 unique cats）中，E 相对 C1 净增 2 个正确，相对 U1 反而净减 1 个。该轮提供有限正向敏感性支持，但不能用于选择 IDEA-072 作为有利主轮次。

### 5.2 IDEA-073

| 管线 | Macro-F1 | CE ↓ | Brier ↓ |
|---|---:|---:|---:|
| A0 | 0.745749 | **0.690063** | 0.413613 |
| C1 | **0.770587** | 0.694195 | **0.409858** |
| U1 | 0.755491 | 0.713831 | 0.421555 |
| E | 0.765983 | 0.697748 | 0.412533 |

E−A0 为 `+0.020234`（9/9 正），E−U1 为 `+0.010492`（6 正/2 平/1 负），但 E−C1 为 `−0.004604`；E 的 CE/Brier 也分别比 C1 差 `0.003554/0.002675`。在 612 个 occurrences（97 unique cats）中，E 相对 C1 净减 3 个正确，相对 U1净增 4 个。该轮说明当 C1 本身明显领先时，固定平均会向 U1 拉回并损害 C1；因此历史敏感性不支持“融合始终保留最优 parent”。

### 5.3 敏感性总判读

IDEA-072、073 与主 IDEA-076 的共同点是：E 相对 U1 通常改善 CE/Brier，概率平均凸性全部通过。不同点是 E 相对 C1 的分类和概率结果随轮次变化；特别是 IDEA-073 中 E 明确低于 C1。三个历史轮次必须分别保留，不能看完结果后选择最有利的一轮，也不能据此调 alpha。

## 6. 数据与实现审计

- 主分析核验 216 份 fit summaries 与 432 份 validation prediction 文件；三组分析合计核验 432 份 fit summaries、864 份 prediction 文件和 3 份 source manifests，共 1,299 个输入文件哈希，0 失败。
- 每个 cell 的 A0/C1/U1 `call_index`、`call_id`、`cat_id`、label、run/seed/repeat/fold 与 validation role 一致；概率列顺序固定为 kitten/adult/senior。
- 历史保存的 animal probabilities 可由 call predictions 重建，最大差 `2.22e-16`。
- `call 概率先融合→猫均值` 与 `C1/U1 猫概率直接等权平均` 在全部三轮的最大差 `3.33e-16`。
- 独立 verifier 不导入主分析脚本，重新计算公式、seed×repeat 指标、fold/split、互补计数和凸性；状态为 PASS。
- 6 项数值测试全部通过。
- 没有读取或生成 outer-test prediction/metric；没有训练、没有加载模型、没有调用 GPU。

## 7. 科学边界与决策

1. **作为分类—概率质量折中的探索候选保留，不升级模型层级。** E 在 IDEA-076 平均上近似 U1 Macro-F1，并使 U1 的 CE/Brier 在数值上改善；这是有价值的后处理观察。
2. **不称稳定胜出。** E 没有稳定超过 U1，CE 也未完全达到 C1；相对 A0 的最差共同 split 仍低于 `−0.03`。
3. **不改变 v3。** A0 仍是唯一 primary / 当前主模型；C1 仍是 exploratory-positive 主要研究候选。IDEA-083 不修改 IDEA-076 gate，也不修改 `reports/58_Formal_Research_Summary_v3.md`。
4. **不启动 alpha 搜索。** 本轮没有证据授权在同一数据上选择更有利的 C1/U1 权重、按猫路由、按类别路由或 oracle 路由。
5. **Oracle 只作上限。** 它说明有限的理论纠错空间，不是可实现模型结果。

## 8. 产物与哈希

| 产物 | SHA-256 |
|---|---|
| 分析计划 | `7889efc7df0565286ebbc9ab82c9a5ee2325c05129fa522cee4d7bc94e0e83a5` |
| 主分析脚本 | `c962e07cca66305ade95e1722d26463cf3e973b72283abe97ee24bb68b57f236` |
| 独立 verifier | `afd176ad695b949676fe8a7f1dc0e0d74efdfe372adea9635e3fced87f3aef1d` |
| 数值测试 | `391d477d54be7270795d173b8ff36412d8cc96244a7377eb0a5632293d7c7cdb` |
| `analysis_summary.json` | `a8d940ea3060eb20e5cd7409b3429d549fbca3c4f72b9cb273a5a14effc4c0f0` |
| `independent_verification.json` | `74d701a6621cfea2cf18b954aa28eaf41979126e19e24a41dab875a407f35b1b` |
| `run_manifest.json` | `397fc5dd2fd6cb11a241eb08fa7a458a7a57f1a0cd8ed966a1afad9fcb22e481` |
| animal occurrences | `309e61f07a1d5cf16e6a653e626b6521b866fadf0b4adfd94740067e7a9018c3` |
| seed×repeat metrics | `2375ac341d448871aff29c49601fa00cf41e9665b3c1f3e198bdbe4523a640ef` |
| split cells | `0cdf85440af0d7580a9ebb4ca844a6df6f9c4731a5f7dbeeb6279eddd7a74923` |
| seed×repeat complementarity | `dc09ab1c1a7bd7b3952201606ac0543131ca0a39b5ca0dcc8753adc9f219dac2` |
| ensemble call predictions | `9cbde0a59138933bcc96010c7b673bbd4cd8f8445918d7e540ddf8bf5a9ba806` |
| input hash ledger | `1bfd17eb92ba3b033deb7baad2321a0256657db3b76f343f73777966e9f0a881` |

完整运行产物位于 `runs/meowagenet_idea083_C1_U1_equal_weight_fusion_v1/`。
