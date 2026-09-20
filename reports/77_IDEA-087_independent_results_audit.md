# IDEA-087 嵌套调参独立结果审计

日期：2026-09-19  
审计角色：牛马1（独立只读复算）  
结论：**PASS（训练、选择锁、outer 完整 OOF 与正式汇总均可由原始预测独立重建）；本轮没有产生可替代原参数 A0 的分类赢家。**

## 1. 审计范围与独立性

本审计脚本不导入 IDEA-087 runner，也不调用其 selection 或 aggregate。它从锁定 protocol、角色表、fit summaries 和原始 call prediction CSV 独立完成：

- 重建全部 `576` 个 inner fits、`1,152` 个 inner prediction files 和 `12` 个 selection locks；
- 逐 search seed 把三个 inner validation folds 拼成完整 outer-dev OOF，再按锁定的 Macro-F1 候选池、Brier、F1 seed SD、Accuracy 和 config-ID 顺序重做选择；
- 重建 `72` 个逻辑 outer records，其中 `54` 个实体拟合、`18` 个 alias records；验证 `108` 个唯一 outer prediction files；
- 对 `72/72` 个逻辑 outer records 从 call probabilities 平均重建 cat probabilities、call count 和 argmax；
- 每个 pipeline/policy/refit seed 将四个互斥 outer-test folds 拼成完整 `111` 只猫 OOF，再独立计算 Accuracy、Macro-F1、balanced accuracy、三类 recall、animal CE 与 Brier；
- 验证 outer authorization、selection-lock SHA、final-stage manifest 和“只在选择锁之后用于最终评分”的 outer-test 边界。

最大 call→cat 概率重建差为 `2.220446049250313e-16`，最大保存指标差为 `5.995204332975845e-15`，均低于锁定数值容差。独立审计状态为 `PASS`。

## 2. 选择锁

`original` 是 q00 原超参数在同一 nested-refit 框架下的对照；其 refit epoch 也只由本 pipeline、outer fold 的 inner 结果推导。`selected` 是八格搜索的锁定选择。两者都不使用 outer-test 选配置、选 epoch 或早停。

| Pipeline | Outer fold | Selected | selected epoch | q00 epoch | selected=q00 alias |
|---|---:|---|---:|---:|---|
| A0 | 0 | q01 | 9 | 11 | 否 |
| A0 | 1 | q00 | 3 | 3 | 是 |
| A0 | 2 | q05 | 9 | 7 | 否 |
| A0 | 3 | q01 | 15 | 15 | 否 |
| U1 | 0 | q00 | 9 | 9 | 是 |
| U1 | 1 | q00 | 3 | 3 | 是 |
| U1 | 2 | q00 | 7 | 7 | 是 |
| U1 | 3 | q02 | 10 | 12 | 否 |
| C1 | 0 | q01 | 9 | 9 | 否 |
| C1 | 1 | q03 | 6 | 3 | 否 |
| C1 | 2 | q00 | 7 | 7 | 是 |
| C1 | 3 | q00 | 14 | 14 | 是 |

q00/q01/q02/q03/q05 分别为 `(lr, dropout, Adamax weight decay)`：`(.006,.44571035,0)`、`(.006,.44571035,.001)`、`(.006,.25,0)`、`(.006,.25,.001)`、`(.003,.44571035,.001)`。12 个锁中 `6` 个选择 q00，因此 selected policy 的 `18` 个 seed-fold records 通过严格同一身份 alias 到 original，而不是重复训练或重复计作独立证据。

## 3. 六个 policy 的 outer complete-OOF 结果

下表每项先对一个 refit seed 拼接四折得到 `111` 只猫的完整 OOF 指标，再对三个 refit seeds 等权平均。它不等同于旧实验的 inner-validation 均值，绝对数值不能跨评估角色直接比较。

| Pipeline / policy | Accuracy | Macro-F1 | BA | CE | Brier | kitten recall | adult recall | senior recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **A0 original** | **0.762763** | **0.772163** | **0.796690** | 0.695362 | 0.391903 | 0.866667 | 0.709677 | 0.813725 |
| A0 selected | 0.747748 | 0.758528 | 0.788917 | 0.694026 | 0.387945 | **0.888889** | 0.693548 | 0.784314 |
| U1 original | 0.750751 | 0.757812 | 0.779479 | 0.711029 | 0.400457 | 0.844444 | 0.709677 | 0.784314 |
| U1 selected | 0.744745 | 0.750693 | 0.774418 | 0.701857 | 0.392386 | 0.844444 | 0.704301 | 0.774510 |
| C1 original | 0.753754 | 0.760230 | 0.782747 | 0.705173 | 0.393559 | 0.844444 | 0.709677 | 0.794118 |
| C1 selected | 0.759760 | 0.764904 | 0.774812 | **0.675427** | **0.382225** | 0.822222 | **0.747312** | 0.754902 |

最重要的排序事实是：

- **原参数 A0 仍是六组中 Accuracy、Macro-F1 和 balanced accuracy 最高者。**因此 IDEA-087 没有找到分类指标上可替代它的方案。
- C1 selected 是六组中 CE 与 Brier 最低者，并有最高 adult recall；这是真实的概率质量/类别取舍，但不是总体分类胜出。
- 不能只写“C1 selected 优于 A0 selected”。C1 selected 相对 **A0 original** 的 Accuracy/Macro-F1/BA 分别为 `−0.003003/−0.007258/−0.021878`；CE/Brier gain 为 `+0.019934/+0.009678`，kitten/adult/senior recall 变化为 `−0.044444/+0.037634/−0.058824`。

## 4. 调参是否改善各自模型

下表均为 `selected − original`；CE/Brier 使用 `control − candidate`，正值表示 selected 的概率误差更低。

| Pipeline | ΔAccuracy | ΔMacro-F1 | ΔBA | CE gain | Brier gain | Δkitten | Δadult | Δsenior | 三个 refit-seed F1 方向 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A0 | −0.015015 | −0.013635 | −0.007773 | +0.001335 | +0.003958 | +0.022222 | −0.016129 | −0.029412 | 0 正 / 0 平 / 3 负 |
| U1 | −0.006006 | −0.007119 | −0.005060 | +0.009172 | +0.008071 | 0 | −0.005376 | −0.009804 | 0 正 / 1 平 / 2 负 |
| C1 | +0.006006 | +0.004674 | −0.007934 | +0.029746 | +0.011334 | −0.022222 | +0.037634 | −0.039216 | 3 正 / 0 平 / 0 负 |

解释如下：

- A0 调参在三个 refit seeds 上都降低 Macro-F1；平均 CE/Brier 略好，但 CE 的 seed 方向为 `1/0/2` 正/平/负，不能称为稳定概率改进。
- U1 调参没有改善分类，Brier 在 `3/3` seeds 改善，CE 为 `2/0/1`；这是概率质量与分类成绩的方向分离。
- C1 调参是唯一在 `3/3` seeds 提高 Macro-F1 的 pipeline，且 CE/Brier 都在 `3/3` seeds 改善。不过其 F1 增益只有 `+0.004674`，同时 BA `−0.007934`；kitten 和 senior recall 下降、adult recall 上升。它说明当前 inner 目标可为 C1 找到较好的概率解，却没有形成跨类别一致的提升。

## 5. 结构比较必须分别看 fixed 与 tuned

正 CE/Brier gain 表示表中前者优于后者；方向计数基于三个完整 refit-seed OOF。

| 比较 | ΔMacro-F1 | F1 正/平/负 | ΔBA | CE gain | Brier gain |
|---|---:|---:|---:|---:|---:|
| U1 original − A0 original | −0.014351 | 0/0/3 | −0.017211 | −0.015667 | −0.008553 |
| C1 original − A0 original | −0.011932 | 1/0/2 | −0.013943 | −0.009812 | −0.001656 |
| C1 original − U1 original | +0.002418 | 2/0/1 | +0.003268 | +0.005856 | +0.006897 |
| U1 selected − A0 selected | −0.007834 | 1/0/2 | −0.014499 | −0.007831 | −0.004441 |
| C1 selected − A0 selected | +0.006376 | 2/0/1 | −0.014105 | +0.018599 | +0.005720 |
| C1 selected − U1 selected | +0.014211 | 2/0/1 | +0.000394 | +0.026430 | +0.010160 |

在 fixed/original 条件下，A0 的 Macro-F1、BA、CE 和 Brier 都优于 U1/C1；C1 只相对 U1 有小幅分类与概率质量优势。在 tuned/selected 条件下，C1 优于同样经过选择的 A0/U1 的 F1 和概率质量，但 C1−A0 的 BA 仍为负，且 tuned A0 已被自身调参明显削弱。因而“C1 selected > A0 selected”不能外推为“C1 selected > 原参数 A0”。

## 6. 这轮实验说明什么

1. 当前锁定的八格 HPO 与 inner 选择目标**不能稳定转移到 outer 分类表现**：A0、U1 的外层 F1 下降，只有 C1 小幅上升。
2. 搜索更稳定地改善了概率质量，尤其是 C1；但 CE/Brier 改善并不保证 Macro-F1、BA 与三类 recall 同时改善。
3. C1 selected 的特点是 adult recall 与概率质量改善，同时牺牲 kitten/senior recall 和 BA。这是有意义的取舍证据，不是无条件赢家。
4. 本轮没有新的全局 pass/fail gate，也没有授权部署或替换主模型。分类上仍应保留原参数 A0 为当前最强参照；C1 selected 可作为概率质量与类别权衡的后续研究材料。

## 7. 解释边界

- 本轮使用的是历史上已反复研究的同一 `111` 只猫、`792` 个 calls。nested HPO 隔离了本轮 inner 选择与本轮 outer 评分，但没有创造新动物、外部数据或独立研究队列。
- 三个 refit seeds 重复使用同一 111 只猫，只刻画训练随机性；不能把它们当作三个独立数据集，也不适合据此提出窄置信区间或强显著性声明。
- outer-test 在 selection lock 后才被访问，并且只用于事先锁定的最终评分；看到 outer 结果后不得追加配置、seed、门槛或重新选择模型。
- IDEA-076 的历史停止结论、IDEA-083/084/085/086 的结论和 v3 证据不因本轮结果被改写。IDEA-087 是用户新授权的内部优化检验，不是独立外部确认。

## 8. 保护、测试与可复现性

- 总监最终复核的旧 `2030` 文件只读快照仍为 `2df36fb271262695b16457777b69cb73570ecc9a645f2bd88fb6a93cbb1382e4`，与训练前一致。
- 主 runner 测试与独立 verifier 测试最终为 `18 passed`。训练前为 `17 passed`；首个授权 fit 完成后新增的第 18 项只从保存的原始 calls 重建该 fit 的身份、cat predictions 与指标，用于加强独立回归检查。该新增测试没有修改 protocol、runner、训练、选择或聚合逻辑。
- Protocol：`ff74e49579afa5a6cad6d4e1ad14894817286924a84af78787dc00191ace2786`。
- Runner：`6152100d5a2e53420aa4877674a63d4163b8cc7a26d157d227491b35e1d57fe7`。
- 主 tests：`8ae5687f96abdb0fc1d587d2ab7b8e26cec582118dae34db94cba8b8da089ee1`。
- CPU preflight：`9f0b3c494679fbcc41deaafc7ea99a577a9990ff96906d618876b414779253be`；inner roles：`70a2f95d65a9a1b584e6d5b45ee9eb450d2c535c3e8c44a4bc9ae0fdf7fd3c11`。
- 独立设计审查 report75：`25e40ce2459c96b89c36ddf2f45e0d8e1e41368b56a52be40a1a9982a25b0268`。
- Selection lock：`04a7bbff803414962d88f0985890fdc4d5bd1797963d42c1c96664a90a274166`；独立 selection audit：`67b79cf77c37e34a3f90a4e197f87247bbbc365fad238448332e50dbc4959b0f`。
- 正式 `initial_evaluation_summary.json`：`8ce1e95bc36c4c38812c40c8ae088688d8f705a96797984e7b69b850c39bf4ae`；outer authorization record：`96750c6b4c0827d130dfb1567a5d48fa8102b7506c79dd1b92add2d72522995c`；final-stage manifest：`d02b857f39673c4e21939c085573dce5fbd849c0dcda1a94ea891b6bed73434e`。
- 独立 verifier：`811d62a1ef63c11c141576706d08800c75188b0e50dfb1dd63373c83efae6eeb`；独立 tests：`fa6676dd28e2985d0c23b317a135aa468a3d6e328d3cca64a07d0a974cb7592f`。
- 独立完整结果审计 JSON：`434f52e194fce757a7727b49c3c64e66e71f813ba324eea46c99fb0e8cd1dc20`。

最终审计决定：**接受 IDEA-087 为工程与统计复算均完整的内部 nested-HPO 结果；不把任何 tuned policy 升级为超过原参数 A0 的分类赢家，不做结果驱动的二次搜索。**
