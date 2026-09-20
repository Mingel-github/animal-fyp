# IDEA-081：AST 尾层完整 ConvPass × C1 因子实验结果

日期：2026-09-18  
状态：144/144 fits 完成；独立审计 PASS；三个预注册 gate 与 full factorial gate 全部失败；outer test 未访问。

## 1. 结论

本次实验不保留固定的 block-12 full ConvPass，也不支持其与 C1 存在正交互。

- `V1−A0` 的九个等权 seed-repeat Macro-F1 平均差为 **+0.00635**，达到了均值阈值，CE 与 Brier 的平均值也略有改善；但只有 5/9 seed-repeat 为正，最差 split 为 −0.03361，并且两个 base seed 的 senior recall 下降超过容忍界限，因此主 gate 失败。
- `CV1−C1` 的平均 Macro-F1 差仅为 **+0.00083**，只有 1/3 base seed 为正、4/9 seed-repeat 为正；CE、Brier、balanced accuracy 与 senior recall 的平均方向均较差，因此组合 gate 失败。
- 交互 `I=(CV1−C1)−(V1−A0)` 为 **−0.00552**，只有 3/9 seed-repeat 为正；CE 与 Brier 交互也为负，因此没有正协同证据。

虽然 CV1 的原始平均 Macro-F1 最高（0.73956），但它对 C1 的预注册增益只有 +0.00083，且概率质量和稳定性不通过。不得按最高单点均值选择 CV1。

## 2. 实验范围

- 数据：MeowAgeNet，792 calls、111 cats、843 segments。
- 表征：只读复用 IDEA-078 block-11 token cache `[843,146,768]`。
- 四管线：A0、C1、V1 full ConvPass、CV1 C1+ConvPass。
- `3 base seeds × 3 repeats × 4 folds × 4 pipelines = 144 fits`。
- base seeds：`[9217,7339,4211]`；与 IDEA-065 至 IDEA-080 的 base/full seeds 无碰撞。
- 独立单位：`cat_id`；primary 单位是九个等权 base-seed × repeat 估计，每个估计先汇集四折 validation animals。
- outer-test predictions：`false`。

封存协议 SHA-256：`50d30df1ece31dacfa33bc33307f8c7992b2d2f0c5d9863b08214821c0dbccd4`。  
封存 runner SHA-256：`def87e10852506c3ac26d1133a232594f92768d9d81912d47de1d47227f97903`。

## 3. 四管线平均结果

| 管线 | Macro-F1 | Balanced accuracy | CE | Brier | Senior recall |
|---|---:|---:|---:|---:|---:|
| A0 | 0.73018 | 0.77315 | 0.69020 | 0.41174 | 0.67222 |
| C1 | 0.73873 | 0.78333 | 0.69314 | 0.41471 | 0.68333 |
| V1 | 0.73652 | 0.76944 | 0.68591 | 0.40768 | 0.65000 |
| CV1 | 0.73956 | 0.77593 | 0.70035 | 0.41715 | 0.65556 |

这些是九个 seed-repeat 的等权平均，不是把 144 个 fold fit 当作独立样本。

## 4. 预注册对比

| 对比 | Mean Δ Macro-F1 | 正/平/负 seed-repeat | 正 base seed | 非负 split | 最差 split | CE gain | Brier gain | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| V1−A0 | +0.00635 | 5/0/4 | 2/3 | 8/12 | −0.03361 | +0.00428 | +0.00406 | FAIL |
| CV1−C1 | +0.00083 | 4/1/4 | 1/3 | 7/12 | −0.09900 | −0.00720 | −0.00245 | FAIL |
| Interaction | −0.00552 | 3/0/6 | 1/3 | 6/12 | −0.07388 | −0.01149 | −0.00651 | FAIL |

这里 CE/Brier gain 定义为 baseline 减 candidate；正值表示更好。interaction 的 probability gain 定义与 Macro-F1 相同，比较 ConvPass 在 C1 背景与 A0 背景下的增益差。

### 4.1 V1−A0

- Mean/SD/median Macro-F1：`+0.00635 / 0.04299 / +0.00672`。
- base-seed mean：9217 `−0.01732`；7339 `+0.02956`；4211 `+0.00680`。
- senior recall delta：9217 `−0.05000`；7339 `−0.03333`；4211 `+0.01667`。
- 通过：均值、正 base seeds、非负 split 数、平均 CE、平均 Brier。
- 失败：正 seed-repeat 仅 5/9（要求 6/9）、最差 split 低于 −0.03、两个 base seed 的 senior recall 低于 −0.02。

解释：存在弱的平均增益信号和较好的平均概率指标，但跨 seed/repeat/split 不稳定，并伴随 senior recall 风险，不能保留。

### 4.2 CV1−C1

- Mean/SD/median Macro-F1：`+0.00083 / 0.03301 / 0.00000`。
- base-seed mean：9217 `−0.00044`；7339 `+0.02696`；4211 `−0.02403`。
- senior recall delta：9217 `−0.06667`；7339 `0.00000`；4211 `−0.01667`。
- 八项 gate 条件全部失败，包括平均 F1、seed/repeat/split 稳定性、CE、Brier 与 senior recall。

解释：ConvPass 加到固定 C1 背景后没有可复现的实用增益。

### 4.3 交互

- Mean/SD/median Macro-F1：`−0.00552 / 0.04316 / −0.00509`。
- base-seed mean：9217 `+0.01687`；7339 `−0.00260`；4211 `−0.03083`。
- senior recall interaction：9217 `−0.01667`；7339 `+0.03333`；4211 `−0.03333`。
- Macro-F1、CE 与 Brier 的平均交互全部为负；八项 interaction gate 条件全部失败。

解释：没有证据支持 C1 与尾层 full ConvPass 协同；平均方向更接近轻微拮抗。

## 5. 完整性与复现审计

GPU 启动前验证了封存 protocol、runner、CPU preflight 与研究总监授权。首个完整 fold 的四管线完成后强制中断并审计，确认无异常才使用 `resume` 继续。

最终独立审计结果：

- 144 个唯一 fit summary，覆盖 36 个 seed-repeat-fold 单元，每单元四管线齐全。
- 144 组 animal/call prediction 哈希全部匹配。
- 每个单元四管线的 validation cats、calls、true labels 与行顺序完全相同。
- validation role、每 epoch training call/cat 覆盖与 coverage hash 全部正确。
- 所有概率有限且归一化；call-to-animal 均值重算完全一致。
- 参数量分别为 99,075、108,143、126,371、135,439。
- 所有 checkpoint reload 最大概率差为 `0.0`。
- 最大 peak VRAM：A0 555,219,456；C1 555,331,584；V1 676,032,000；CV1 676,144,128 bytes。
- 独立 aggregate 重放与保存 summary 的 canonical bytes 完全一致。
- aggregate SHA-256：`9ded82a815b081ecc9fb9fb800ce76eb8972b9be63223b44cd621b614f1ecfbd`。
- `outer_test_accessed=false`。

## 6. 方法结论

本次负结果不否定 ConvPass 在其他数据或全层部署中的价值；它只拒绝本实验预注册的固定配置：**冻结 AST，仅在 block 12 放置 width-8、scale-0.1 的双支路 full ConvPass，并用于 MeowAgeNet 年龄分类**。

不能在看到结果后改 scale、width、层数、只保留 MSA 分支或选择有利 seed 来修复本 gate。若以后重新研究卷积旁路，应作为全新的、由独立文献或机制假设驱动的实验，而不是 IDEA-081 的结果后调参。

## 7. 关键产物

- `configs/protocol/meowagenet_idea081_ast_tail_convpass_c1_factorial_v1.json`
- `scripts/run_meowagenet_idea081_ast_tail_convpass_c1_factorial.py`
- `runs/meowagenet_idea081_ast_tail_convpass_c1_factorial_v1/cpu_preflight.json`
- `runs/meowagenet_idea081_ast_tail_convpass_c1_factorial_v1/gpu_startup_audit.json`
- `runs/meowagenet_idea081_ast_tail_convpass_c1_factorial_v1/first_complete_fold_audit.json`
- `runs/meowagenet_idea081_ast_tail_convpass_c1_factorial_v1/initial_evaluation_summary.json`
- `runs/meowagenet_idea081_ast_tail_convpass_c1_factorial_v1/independent_audit.json`
- `scripts/verify_idea081_ast_tail_convpass_c1_results.py`

文献与真实拓扑依据见 `reports/52_IDEA-081_ConvPass_AST_literature_and_preflight_basis.md`。
