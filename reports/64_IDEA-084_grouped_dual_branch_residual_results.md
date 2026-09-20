# IDEA-084：分组双分支声学残差结果

日期：2026-09-19  
正式结论：**NO-GO for grouped dual-branch advantage。P1 对 C1、A0 与固定随机分组 R1 的平均 Macro-F1 均未改善，三条预注册主比较均失败。**

## 1. 执行与完整性

- 完成 A0、同期原样 C1、真实分组 P1、固定哈希随机分组 R1 × 3 base seeds × 3 repeats × 4 folds = **144/144 fits**。
- 首个四管线 cell 在总监限定范围内单独运行并暂停；身份、预测哈希、初始化、checkpoint reload、参数量、批次覆盖、总残差预算和 outer-test 标志全部通过后，才按总监续跑授权完成其余 fits。
- 全量结束后再次执行 canonical `--resume` 完整性检查：没有重训已有 fit，144/144 产物通过 manifest、fit identity 与预测哈希校验，正式 aggregate 可重复生成。
- 独立 verifier 未导入 runner 或 aggregate，而是从 144 份 fit summary、144 份 call-level 和 144 份 animal-level CSV 重建结果；共检查 **288 份预测文件**，逐调用重建逐动物概率，与正式汇总 `difference_count=0`。
- IDEA-084 专项测试为 **13/13 passed**。未生成、读取或评分 outer-test prediction/metric；`outer_test_accessed=false`。

## 2. 四条管线的 seed×repeat 等权均值

| Pipeline | Macro-F1 | Balanced accuracy | Animal CE ↓ | Brier ↓ |
|---|---:|---:|---:|---:|
| A0 AST only | 0.727352 | 0.761111 | 0.699483 | 0.420760 |
| C1 bounded wide additive | 0.729273 | 0.762963 | **0.697506** | **0.417705** |
| P1 grouped dual branch | 0.726634 | 0.766667 | 0.698193 | 0.418447 |
| R1 fixed hash-random dual branch | **0.733229** | **0.771296** | 0.701961 | 0.420721 |

这些是 9 个等权 seed×repeat 估计的均值。36 个 fold 差值和 612 个重复动物出现只用于描述；它们重复使用同一批 111 只猫，不能当作独立样本扩大证据量。

## 3. 三条预注册主比较

CE/Brier “gain” 定义为 comparator 减 candidate，因此正值表示 P1 更低、更好。

| 比较 | ΔMacro-F1 | seed×repeat 正/tie/负 | 正 base seed | 非负 split-cell | 最差 split | ΔBA | CE gain | Brier gain | 纠正/新增/净纠正 | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P1−C1 | −0.002639 | 4/0/5 | 1/3 | 6/12 | −0.036159 | +0.003704 | −0.000687 | −0.000742 | 14/15/−1 | FAIL |
| P1−A0 | −0.000718 | 3/3/3 | 2/3 | 8/12 | −0.028042 | +0.005556 | +0.001291 | +0.002313 | 8/9/−1 | FAIL |
| P1−R1 | −0.006595 | 1/2/6 | 0/3 | 7/12 | −0.061250 | −0.004630 | +0.003768 | +0.002274 | 6/10/−4 | FAIL |

### P1−C1：未显示方法改进

P1 的平均 Macro-F1 比同期原样 C1 低 0.002639，9 个 seed×repeat 中 5 个为负，三个 base-seed 均值只有 3583 为正。P1 的 balanced accuracy 高 0.003704，但 CE 与 Brier 均略差，纠错总账也是新增错误比纠正错误多 1 次。

Senior recall 的 base-seed 差值为 `−0.0167 / −0.0333 / +0.0667`，方向不稳定且 5080 越过预注册安全界。因此，额外 84 个参数和 15/5 双分支结构没有形成相对 C1 的稳健收益。

### P1−A0：保留概率质量与类别平衡正向结果，但宏F1增益未建立

P1 相对 A0 的平均 Macro-F1 为 −0.000718，方向计数正/tie/负恰为 3/3/3，因此主效用门失败。与此同时，有几项预注册的正向结果必须保留：balanced accuracy `+0.005556`、CE gain `+0.001291`、Brier gain `+0.002313`，且三个 base seed 的 senior-recall 差值为 `+0.0333 / 0 / +0.0167`。

合并的重复验证出现中，P1 的 kitten recall 从 A0 的 0.9306 升至 0.9444，senior recall 从 0.6444 升至 0.6611，而 adult recall 从 0.7083 降至 0.6944。这解释了 balanced accuracy 改善但 Macro-F1、总体准确率和净纠错没有同步改善：收益偏向边缘年龄类别，同时牺牲了部分 adult 判断。该剖面是有价值的探索性信息，但不能覆盖失败的主效用结论。

### P1−R1：不支持冻结的真实分组优于容量匹配对照

P1 相对 R1 的平均 Macro-F1 为 −0.006595，三个 base-seed 均值全部为负，只有 1/9 seed×repeat 为正，最差 split 为 −0.061250；balanced accuracy 也低 0.004630，纠错总账净少 4 次。P1 的 CE 与 Brier 分别比 R1 好 0.003768 与 0.002274，但不足以抵消分类性能与稳定性的反向结果。

R1 是结果前固定的一次哈希随机分组，只用于控制双分支容量与优化路径。P1−R1 失败说明本次 15 维 `source-related` / 5 维 `spectral-energy` 切分没有显示特异优势；它既不能证明随机分组具有生理意义，也不能把 R1 事后升级为新机制候选。

## 4. 同期描述性正向结果

这些结果不改变三条主比较的失败，也不构成新的事后 gate：

- 同期 C1−A0：Macro-F1 `+0.001921`、balanced accuracy `+0.001852`、CE gain `+0.001978`、Brier gain `+0.003055`。这保留了 C1 的小幅平均正向方向，但幅度有限，不能重开 IDEA-076 已冻结的结论。
- R1−A0：Macro-F1 `+0.005877`、balanced accuracy `+0.010185`，为本轮最高；但 CE gain 为 `−0.002477`，Brier 仅改善 `+0.000039`。这是预定控制的描述性正向结果，不是预注册主比较，也没有外部确认。

## 5. 结论与边界

1. **正式结论是 NO-GO**：冻结的 P1 分组双分支没有显示相对 C1、A0 或 R1 的稳健 Macro-F1 优势。
2. **正向结果仍完整保留**：P1 相对 A0 的 balanced accuracy、CE、Brier 与 senior recall 有小幅改善；C1 与 R1 也分别有同期描述性正向指标。
3. **不能从 R1 的最高 F1 推导随机分组机制**：只有一个结果前固定的随机对照，且其概率质量没有同步领先。CE/Brier 是综合概率评分，不能仅凭下降就声称独立的校准机制已获证明。
4. **本轮冻结，不事后调参追分**：本次144 fits不追加分组、宽度、权重、种子或门槛搜索。未来若有新声学假设，可继续以MeowAgeNet为首要任务，事前锁定设计后另立探索实验；重复利用当前验证数据不能包装为独立确认。外部猫或狗benchmark是可选的后续扩展，不是用户当前任务的强制成功条件。
5. 文献只支持“分开建模声学族”作为合理假设，而不验证本项目的具体切分。哺乳动物发声的声源—滤波框架及其交互见 [Taylor & Reby (2010)](https://doi.org/10.1111/j.1469-7998.2009.00661.x)；小型、理论驱动声学参数集的组织原则见 [GeMAPS](https://doi.org/10.1109/TAFFC.2015.2457417)。本实验没有 formant，spectral tilt 亦是混合代理，因此不作纯生理分解声明。

## 6. 固定结果哈希

- protocol：`0c4b50f2a6903bb5e55157591e5a93400c56602161fa2b1f8c8fd0d112bb7eef`
- runner：`fa3913fd7b2a7a66f3a4e61deb150ef7358cc32ee5b65eafdfe9d2c7e2abe12f`
- run manifest：`f773401e39778a1207020201acba4a1bea5c8228ed951c2b9deab551a9478c37`
- initial evaluation summary：`fb4dea0b4b452561a800b6cb4c9f3a7c29ac0f1f79a74bd67c1b3c674092cd46`
- compact run summary：`6bc04fe30d4facbfec09868c22da96ad6660b1fb35f3c2a6307d60cf7a684429`
- first-cell audit：`88611fc002048de1b09b54143e01077ca789f37015014335fd3edbd484f9f4d6`
- independent results audit：`8ebe20a875a6f3b766141551d6914820e5994a101703d2edf2ce696324c82b6a`
- independent verifier：`649ca0dcdbdc20a2e9265de8b8edc810fb87f5857f434e8013422f4060cd603a`
