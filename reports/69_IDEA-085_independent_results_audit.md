# IDEA-085 独立结果审计

日期：2026-09-19  
审计角色：牛马1（独立只读复算）  
结论：**PASS（结果完整性与正式汇总一致）；T1 对 C1、A0、J1 的三个预设分类 gate 均未通过。**

## 独立性与完整性

- 独立复算脚本不导入 IDEA-085 runner，也不调用其 `aggregate`；只读取锁定协议、manifest、fit summary 及原始 validation call/animal CSV。
- 核对完整 `3 base seeds × 3 repeats × 4 folds × 4 pipelines = 144 fits`，共验证 288 个预测文件 SHA；每个 fit 的状态、full seed、角色调用/动物全集、参数数目、checkpoint reload、无 outer-test 标志均通过。
- 对 144 个 fit 全部由 call 概率平均重建 cat 概率、argmax 和 call count；与保存的 animal CSV 完全一致。各 cell 四管线的 animal ID 与标签成对一致。
- 独立复算 Accuracy、Macro-F1、BA、kitten/adult/senior recall、CE、Brier、9 个 seed×repeat、12 个 split cell、3 个 base-seed 稳定性、三组纠错转移及三条独立 classification gate。
- 与正式 `initial_evaluation_summary.json` 逐字段比较，容差 `1e-12`，差异数为 `0`。outer-test 预测和指标未访问。

## 九个 seed×repeat 等权均值

| Pipeline | Accuracy | Macro-F1 | BA | CE | Brier | kitten recall | adult recall | senior recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 AST only | 0.72222 | 0.73622 | 0.75741 | 0.68992 | 0.41245 | 0.88889 | 0.71667 | 0.66667 |
| C1 summary residual | 0.71242 | 0.72547 | 0.75093 | 0.70649 | 0.41817 | 0.88889 | 0.70278 | 0.66111 |
| T1 real-order temporal | 0.72059 | 0.73328 | 0.77037 | 0.71653 | 0.42573 | 0.93056 | 0.69722 | 0.68333 |
| J1 joint-frame shuffled | 0.71895 | 0.73371 | 0.76852 | 0.71238 | 0.42179 | 0.93056 | 0.69722 | 0.67778 |

CE、Brier、Accuracy、BA 和各类 recall 是辅助剖面，不参与预设 Macro-F1 classification gate。

## 三条预设 classification gate

| 比较 | mean ΔMacro-F1 | 3个 base-seed 均值为正 | 9个 seed×repeat 为正 | 12个 split 非负 | worst split | Gate |
|---|---:|---:|---:|---:|---:|---|
| T1−C1 | +0.00781 | 3/3 | 5/9 | 8/12 | −0.05777 | **FAIL** |
| T1−A0 | −0.00294 | 1/3 | 4/9 | 8/12 | −0.06337 | **FAIL** |
| T1−J1 | −0.00043 | 2/3 | 2/9（另 4 ties） | 8/12 | −0.03190 | **FAIL** |

门槛要求依次为 mean delta `≥0.005`、至少 `2/3` base-seed 均值为正、至少 `6/9` seed×repeat 为正、至少 `8/12` split 非负、worst split `≥−0.03`。逐项结论：

- T1−C1 的均值、base-seed 与 split 数条件通过，但只有 `5/9` seed×repeat 为正，且 worst split 为 `−0.05777`，因此失败。
- T1−A0 只有 split 数条件通过；均值为负、仅 `1/3` base-seed 均值为正、`4/9` seed×repeat 为正，worst split 为 `−0.06337`，因此失败。
- T1−J1 的 base-seed 与 split 数条件通过；均值略负、仅 `2/9` seed×repeat 为正，worst split `−0.03190` 也略低于锁定下限，因而失败。
- 本轮明确不存在单一全局 gate；三条结果必须分别解释，不能合并成一个总 pass/fail。

## 辅助剖面与纠错转移

- T1−C1：Accuracy `+0.00817`，BA `+0.01944`，CE gain `−0.01004`，Brier gain `−0.00757`；kitten/adult/senior recall 变化为 `+0.04167/−0.00556/+0.02222`。612 次成对动物出现中，纠正 21、引入 16、净 `+5`；9 个 seed×repeat 的净纠错方向为 `5/0/4`（正/平/负）。
- T1−A0：Accuracy `−0.00163`，BA `+0.01296`，CE gain `−0.02660`，Brier gain `−0.01329`；recall 变化为 `+0.04167/−0.01944/+0.01667`。纠正 12、引入 13、净 `−1`；方向 `2/3/4`。
- T1−J1：Accuracy `+0.00163`，BA `+0.00185`，CE gain `−0.00415`，Brier gain `−0.00395`；recall 变化为 `0/0/+0.00556`。纠正 7、引入 6、净 `+1`；方向 `2/4/3`。

负 CE/Brier gain 表示 T1 的概率质量比比较对象更差；这些辅助结果不改变三个 Macro-F1 gate 的判定。

## 重复出现次数与独立动物数

每条管线的 pooled validation 包含 `612` 个重复动物出现，但只有 `97` 个 unique cats；每只猫出现次数最小/中位数/最大值为 `3/6/15`。这 612 行来自 seed、repeat 与 fold 的重复评估，不能当作 612 个独立动物；数据全集仍为 111 cats，而本轮 validation 角色并集为 97 cats。

## 解释边界

- 结果只支持对这个锁定的轻量局部时序残差设计作判断，不是独立外部验证，也不改变既有 v3 结论。
- T1−C1 同时改变了从固定 summary 到局部轨迹编码的表示方式，不能把差异纯归因于“帧顺序”。
- T1−J1 是真实顺序与一个固定联合帧置乱控制的比较；它同时保留/打乱六通道值与缺失性、voicing 的局部组织，不是纯 F0 顺序隔离，也不是完整因果证明。
- 按预注册规则记录全部结果，不因三个 gate 失败而进行事后调参、改 seed、改置乱或改门槛。

## 产物与验证

- 独立首 cell 审计：`runs/meowagenet_idea085_acoustic_temporal_residual_v1/independent_first_cell_audit.json`，SHA-256 `e10001a9b2a37c2ab31eac69b0045f11717fac714c11421bd6b3a9b8030553db`。
- 独立完整结果审计：`runs/meowagenet_idea085_acoustic_temporal_residual_v1/independent_results_audit.json`，SHA-256 `c2d7d55ca0460acd6d564f3807b83d7de9226ed1a503247783083656824b7a51`。
- 正式汇总：`initial_evaluation_summary.json` SHA-256 `4aa2ce4862c81fda49e35e4e77a2901fac01bcea9d52feeab4fb80fbc58c0f7a`；`run_summary.json` SHA-256 `41d6e9df66e07bc30961e0cd308a674519e2eef8c72bd4df4e912dc63d048c48`。
- 独立复算脚本 SHA-256 `3b1eb9ea3b5ea363cdae63fddb509ce06f2d5f0b359bc78921e019657277c773`；专项测试 SHA-256 `11e308e06df3521788786e7fd0169ae6a24bb65bb07a58d948e2bb924401a9d8`。
- 独立复算器、IDEA-085 模型测试与轨迹测试联合执行：`21 passed in 8.60s`。
