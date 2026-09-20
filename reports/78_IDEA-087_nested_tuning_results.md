# IDEA-087：A0/U1/C1 严格嵌套调参结果

日期：2026-09-19  
正式结论：**原参数 A0 保持当前分类首选；调参 C1 是本轮最好的 tuned policy 与概率评分方案，并且是唯一相对自身原参数同时提高 Accuracy、Macro-F1 并降低 CE、Brier 的模型。两者体现不同优势：A0 original 的总体分类最高，C1 selected 的概率质量最好且 adult recall 更高。**

## 1. 这轮研究了什么

本轮没有改变 A0、U1、C1 的结构，而是在完全相同的严格 nested-HPO 预算下，分别搜索三项训练参数：learning rate `{0.006, 0.003}`、dropout `{0.44571035356880917, 0.25}`、Adamax weight decay `{0, 0.001}`，共 8 个配置。`q00=(0.006, 0.44571035356880917, 0)` 是原参数。

- A0：冻结 AST 表征加共同分类头。
- U1：声学分支固定为 `20→60→128`，把输出 `r` 自由加到主路 128 维表示 `h`，即 `h+r`。
- C1：使用相同声学分支，但先把残差限幅为 `0.25×stopgrad(RMS(h))×tanh(r)`，再加到 `h`。声学宽度 60 与限幅 0.25 在本轮固定，不参与搜索。

每个 outer fold 只在 outer-dev 内做三折猫级分层搜索。两个 search seeds 各自产生一套完整 outer-dev OOF；先保留平均 Macro-F1 距最佳不超过 `0.002` 的配置，再依次按 Brier、F1 seed 样本标准差、Accuracy 和配置编号选择。最终 refit epoch 取该配置六个 inner best epochs 的中位数并 half-up 取整。也就是说，本轮调参同时允许训练参数和由 inner 结果决定的固定训练 epoch 发生变化。

这只是锁定的 8 配置局部搜索，结果含义是“在当前小网格与当前选择规则下的最优选择”，不表示全局超参数最优。

## 2. Inner 选择结果

下表只描述 **inner selection lock**，不是 outer 成绩。`q00 epoch` 是同一 pipeline、同一 outer fold 下原参数对照自己的 inner-derived refit epoch。

| Pipeline | Outer fold | Selected | lr | dropout | weight decay | selected epoch | q00 epoch | F1 容差候选池 |
|---|---:|---|---:|---:|---:|---:|---:|---|
| A0 | 0 | q01 | 0.006 | 0.44571035 | 0.001 | 9 | 11 | q01 |
| A0 | 1 | q00 | 0.006 | 0.44571035 | 0 | 3 | 3 | q00 |
| A0 | 2 | q05 | 0.003 | 0.44571035 | 0.001 | 9 | 7 | q05 |
| A0 | 3 | q01 | 0.006 | 0.44571035 | 0.001 | 15 | 15 | q01 |
| U1 | 0 | q00 | 0.006 | 0.44571035 | 0 | 9 | 9 | q00, q05 |
| U1 | 1 | q00 | 0.006 | 0.44571035 | 0 | 3 | 3 | q00, q07 |
| U1 | 2 | q00 | 0.006 | 0.44571035 | 0 | 7 | 7 | q00 |
| U1 | 3 | q02 | 0.006 | 0.25 | 0 | 10 | 12 | q02 |
| C1 | 0 | q01 | 0.006 | 0.44571035 | 0.001 | 9 | 9 | q01 |
| C1 | 1 | q03 | 0.006 | 0.25 | 0.001 | 6 | 3 | q03, q00 |
| C1 | 2 | q00 | 0.006 | 0.44571035 | 0 | 7 | 7 | q00, q05 |
| C1 | 3 | q00 | 0.006 | 0.44571035 | 0 | 14 | 14 | q00 |

其中 `q01=(.006,.44571035,.001)`、`q02=(.006,.25,0)`、`q03=(.006,.25,.001)`、`q05=(.003,.44571035,.001)`。12 个 model×fold 锁中有 6 个选择 q00；对应的 18 个 refit-seed 结果严格 alias 到同一个 q00 physical fit，避免重复训练和重复证据。

## 3. Outer 完整 OOF 主结果

每个 refit seed 都把四个互斥 outer-test folds 拼成同一套 `111` 只猫完整 OOF，再对 3 个 refit seeds 等权平均。表中 `±` 后为三个完整 OOF 估计的样本标准差。

| Pipeline / policy | Accuracy | Macro-F1 | BA | CE ↓ | Brier ↓ |
|---|---:|---:|---:|---:|---:|
| **A0 original** | **0.762763 ± 0.020805** | **0.772163 ± 0.021455** | **0.796690 ± 0.011356** | 0.695362 ± 0.032698 | 0.391903 ± 0.016064 |
| A0 selected | 0.747748 ± 0.023836 | 0.758528 ± 0.021527 | 0.788917 ± 0.006678 | 0.694026 ± 0.021113 | 0.387945 ± 0.007285 |
| U1 original | 0.750751 ± 0.005201 | 0.757812 ± 0.012678 | 0.779479 ± 0.011137 | 0.711029 ± 0.032263 | 0.400457 ± 0.016439 |
| U1 selected | 0.744745 ± 0.013761 | 0.750693 ± 0.017210 | 0.774418 ± 0.013697 | 0.701857 ± 0.024355 | 0.392386 ± 0.015161 |
| C1 original | 0.753754 ± 0.013761 | 0.760230 ± 0.003545 | 0.782747 ± 0.014929 | 0.705173 ± 0.044485 | 0.393559 ± 0.018236 |
| **C1 selected** | 0.759760 ± 0.010403 | 0.764904 ± 0.000926 | 0.774812 ± 0.008630 | **0.675427 ± 0.041689** | **0.382225 ± 0.011678** |

结果呈现出两条清楚的主线：

- **A0 original 的 Accuracy、Macro-F1 和 BA 都是六组最高。**它仍是当前最强分类参照。
- **C1 selected 的 CE 与 Brier 都是六组最低，且三个 refit seeds 的 Macro-F1 离散程度最小。**它是本轮概率评分与 tuned-policy 的最佳方案。

本表的 complete outer OOF 与旧实验的 inner-validation 均值属于不同评估角色；绝对数值不直接横比。IDEA-087 的主要证据是同一轮、同一 outer cats、同一 refit seeds 下的配对差。

## 4. 三类 recall 的收益与取舍

| Pipeline / policy | kitten recall | adult recall | senior recall |
|---|---:|---:|---:|
| A0 original | 0.866667 | 0.709677 | **0.813725** |
| A0 selected | **0.888889** | 0.693548 | 0.784314 |
| U1 original | 0.844444 | 0.709677 | 0.784314 |
| U1 selected | 0.844444 | 0.704301 | 0.774510 |
| C1 original | 0.844444 | 0.709677 | 0.794118 |
| C1 selected | 0.822222 | **0.747312** | 0.754902 |

C1 selected 把重点移向 adult：相对 C1 original，adult recall `+0.037634`，同时 kitten `−0.022222`、senior `−0.039216`，所以 BA 下降 `0.007934`。A0 selected 的 kitten recall 最高，但 adult/senior recall 和总体分类分数有所下降。U1 selected 的类别轮廓变化较小，主要收益体现在概率评分而非分类。

## 5. 调参相对各自原参数

下表为 `selected − original`；CE/Brier gain 定义为 `original − selected`，正值表示调参后概率误差更低。

| Pipeline | ΔAccuracy | ΔMacro-F1 | ΔBA | CE gain | Brier gain | F1 seed 正/平/负 | CE seed 正/平/负 | Brier seed 正/平/负 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | −0.015015 | −0.013635 | −0.007773 | +0.001335 | +0.003958 | 0/0/3 | 1/0/2 | 2/0/1 |
| U1 | −0.006006 | −0.007119 | −0.005060 | +0.009172 | +0.008071 | 0/1/2 | 2/0/1 | 3/0/0 |
| C1 | **+0.006006** | **+0.004674** | −0.007934 | **+0.029746** | **+0.011334** | **3/0/0** | **3/0/0** | **3/0/0** |

- A0：当前搜索改善了平均 CE/Brier 和 kitten recall；Accuracy 的 seed 方向为 0 正/1 平/2 负，Macro-F1 为 0 正/0 平/3 负，分类应用继续用 original。
- U1：selected 的 Brier 在 3/3 seeds 改善、CE 在 2/3 改善，同时 Accuracy/Macro-F1 较低；本轮分类同样优先 original。既往 U1 的正向结构证据继续保留，本轮只回答这套局部 HPO 是否改善 U1。
- C1：是唯一在 3/3 seeds 同时提高 Macro-F1、降低 CE 与 Brier 的 pipeline；平均 Accuracy 也上升。收益集中在 adult recall 与概率质量，代价是 kitten/senior recall 和 BA。

## 6. 模型之间如何选

| 同轮比较 | ΔAccuracy | ΔMacro-F1 | ΔBA | CE gain | Brier gain |
|---|---:|---:|---:|---:|---:|
| C1 selected − A0 selected | +0.012012 | +0.006376 | −0.014105 | +0.018599 | +0.005720 |
| C1 selected − U1 selected | +0.015015 | +0.014211 | +0.000394 | +0.026430 | +0.010160 |
| C1 selected − A0 original | −0.003003 | −0.007258 | −0.021878 | +0.019934 | +0.009678 |

因此建议按目标选用：

- 若首要目标是 Accuracy、Macro-F1 或跨类均衡，继续以 **A0 original** 为主模型/基准。
- 若首要目标是概率评分质量，或业务更重视 adult recall，可保留 **C1 selected** 作为明确的候选方案。
- U1 的本轮 selected policy 没有形成新的优势；在 IDEA-087 的相同 outer 口径下，U1 original 优先于 U1 selected。

这轮没有设置新的单一 global gate，也不把不同指标的取舍压缩成一个“全赢/全输”标签。

## 7. 完整性与独立复算

- Inner 完成 `576/576` fits；selection lock 有 `12/12` 个 model×fold 身份。
- Outer 完成 `72/72` 个逻辑结果：`54` 个 physical fits、`18` 个 aliases；3 个 refit seeds 各有完整 111-cat OOF，共 333 个重复动物出现，但实际仍是同一 111 只猫。
- outer-test 只在 selection lock 固定并经独立复算、总监绑定精确 SHA 授权后访问；它只用于最终评分。
- 独立 verifier 从原始预测重建了 576 个 inner fits、12 个选择锁、72 个 outer 逻辑记录和所有汇总。最大 call→cat 概率差 `2.22e-16`，最大保存指标差 `5.995e-15`。
- 专项主测试与独立 verifier 测试为 `18 passed`。
- 完成后再次执行 outer `--resume`：验证 `72/72`，新增 physical fits `0`、aliases `0`；180 个 outer fit/prediction 核心文件复跑前后组合 SHA 均为 `9176b7868ad6d1ff49e9fc710f0f5cb99cab6c3b495a6cf588e65b5cd385fa2a`。
- 旧 2030 文件保护快照仍为 `2df36fb271262695b16457777b69cb73570ecc9a645f2bd88fb6a93cbb1382e4`。

完整独立方法与结果审计见 [报告 77](./77_IDEA-087_independent_results_audit.md)。

## 8. 证据边界与决定

这是在历史上已反复研究的同一 `111` 只猫、`792` 个 calls 上完成的内部 nested-HPO。严格嵌套隔离保证了本轮 inner 选择与 outer 最终评分分离；它没有增加新动物、外部队列或独立研究数据。三个 refit seeds 反映训练随机性，不是三个独立数据集。

**决定：保留 A0 original 作为分类首选；保留 C1 selected 作为概率质量/adult-recall 候选，不替换 A0 的分类地位；U1 original 优先于本轮 U1 selected。停止当前 8 配置搜索，不根据 outer 结果追加参数、种子或阈值。IDEA-076、IDEA-083/084/085/086 与 v3 的既有结论保持不变。**

## 9. 固定产物哈希

- plan：`5770ade41409f28340513d0f6a282a8561cb89adf44222b453a5b64cc1f3faf0`
- protocol：`ff74e49579afa5a6cad6d4e1ad14894817286924a84af78787dc00191ace2786`
- runner：`6152100d5a2e53420aa4877674a63d4163b8cc7a26d157d227491b35e1d57fe7`
- main tests：`8ae5687f96abdb0fc1d587d2ab7b8e26cec582118dae34db94cba8b8da089ee1`
- independent verifier：`811d62a1ef63c11c141576706d08800c75188b0e50dfb1dd63373c83efae6eeb`
- independent verifier tests：`fa6676dd28e2985d0c23b317a135aa468a3d6e328d3cca64a07d0a974cb7592f`
- design review report75：`25e40ce2459c96b89c36ddf2f45e0d8e1e41368b56a52be40a1a9982a25b0268`
- CPU/preflight report76：`5a888f0dd8a34b92ffb4a7c48ad6f6c67327e2210f5bfaa55fda4ae88b7308a2`
- independent results audit report77：`d17cce52fb042b24815301d3ea651128b28ace464b8bf11c7c8da2b40234aa6e`
- CPU preflight：`9f0b3c494679fbcc41deaafc7ea99a577a9990ff96906d618876b414779253be`
- inner roles：`70a2f95d65a9a1b584e6d5b45ee9eb450d2c535c3e8c44a4bc9ae0fdf7fd3c11`
- first-fit audit：`90be1640ff9996ebd579f5c40ffc0f3e430dc424e7a531115107d28bc35bd852`
- independent first-fit audit：`272de71adbfc42fdff34ce93893933b0fbe45bae1f4c4b9e7a386fd2fb4fbc0b`
- selection lock：`04a7bbff803414962d88f0985890fdc4d5bd1797963d42c1c96664a90a274166`
- independent selection audit：`67b79cf77c37e34a3f90a4e197f87247bbbc365fad238448332e50dbc4959b0f`
- outer authorization record：`96750c6b4c0827d130dfb1567a5d48fa8102b7506c79dd1b92add2d72522995c`
- final-stage manifest：`d02b857f39673c4e21939c085573dce5fbd849c0dcda1a94ea891b6bed73434e`
- initial evaluation summary：`8ce1e95bc36c4c38812c40c8ae088688d8f705a96797984e7b69b850c39bf4ae`
- independent full results audit：`434f52e194fce757a7727b49c3c64e66e71f813ba324eea46c99fb0e8cd1dc20`
