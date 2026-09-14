# IDEA-056 按猫级验证损失选择训练轮次的确认结果

## 结论摘要

本轮在 frozen AST、`768 → 128 → 3` 分类头、训练 loss、数据划分和训练轨迹完全匹配的
条件下，只比较两种 checkpoint 选择规则：

- **C0-call**：选择 inner-validation 叫声级交叉熵最低的 epoch；
- **C1-animal**：先平均同一只猫各条叫声的预测概率，再选择猫级交叉熵最低的 epoch。

正式实验使用新的 base seeds 43 和 101，每个 seed 包含 3 个 repeat、每个 repeat 包含 4 个
cat-ID-disjoint folds。24 条共享训练轨迹生成 6 组覆盖全部 111 只猫的 complete-OOF 配对。

| 选择规则 | Animal macro F1，mean ± SD | Balanced accuracy | QWK | 普通 accuracy | Animal CE，越低越好 |
| --- | ---: | ---: | ---: | ---: | ---: |
| C0-call | 0.7368 ± 0.0321 | **0.7614** | **0.6508** | 0.7222 | 0.7781 |
| C1-animal | **0.7426 ± 0.0186** | 0.7413 | 0.6480 | **0.7372** | **0.7521** |
| C1 − C0 | **+0.0059** | −0.0201 | −0.0029 | **+0.0150** | **−0.0260** |

预设主门槛要求平均 macro F1 增量至少为 `+0.005`，并且至少 4/6 组配对为正。本轮实际
取得 `+0.005855` 和 4/6，因此 **IDEA-056 的确认门槛通过**。C1 同时提高普通 accuracy
1.50 个百分点、把最差一组 macro F1 提高 0.0171，并将 repeat 间 macro-F1 SD 从 0.0321
降到 0.0186。后续 tuned frozen-AST reference 采用 C1 的猫级验证 CE 选择规则。

这项收益具有明确的类别结构：C1 在 6 组完整 OOF、合计 666 次猫级评价中多识别正确
27 次 adult，同时 kitten 和 senior 分别少识别正确 8 次和 9 次，总正确数净增加 10 次。
因此普通 accuracy 与 macro F1 提高；balanced accuracy 因三类 recall 等权计算而下降
0.0201，QWK 小幅下降 0.0029。该结果支持把 C1 表述为一项小幅、已通过预设门槛的流程
改进，并同步记录类别侧重点和 seed 差异。

## 1. 为什么要比较这两种 checkpoint 选择规则

最终任务的评价单位是猫：同一只猫可能包含多条叫声，系统先预测每条叫声，再平均这些
概率得到一只猫的三分类结果。C0 在选择训练轮次时把每条叫声看作一个验证样本，叫声较多
的猫会在平均损失中占更大份量。C1 在每个 epoch 先聚合同一只猫的所有叫声概率，每只猫
只贡献一次验证损失，因此 checkpoint 选择过程与最终 animal-level 评价单位一致。

每折只训练一条 inner trajectory，并在每个 epoch 同时记录 call CE 和 animal CE；随后在
同一条 outer trajectory 上保存两个规则选中的 epoch。模型初始化、batch 顺序、训练数据、
优化器和最大候选 epoch 均相同，观测差异可以归因于“选择哪个训练轮次”。

## 2. 六组新的 complete-OOF 配对

每一行都合并该 repeat 的四个 outer folds，得到覆盖全部 111 只猫的一份 complete-OOF。

| Base seed | Repeat | C0 macro F1 | C1 macro F1 | C1 − C0 | 方向 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 43 | 0 | 0.7357 | **0.7630** | **+0.0273** | C1 提高 |
| 43 | 1 | **0.7274** | 0.7246 | −0.0028 | C0 略高 |
| 43 | 2 | 0.7184 | **0.7487** | **+0.0303** | C1 提高 |
| 101 | 0 | **0.7919** | 0.7527 | −0.0392 | C0 较高 |
| 101 | 1 | 0.7496 | **0.7520** | **+0.0024** | C1 略高 |
| 101 | 2 | 0.6977 | **0.7148** | **+0.0171** | C1 提高 |
| **平均** |  | **0.7368** | **0.7426** | **+0.0059** | **4/6 为正** |

seed 43 的三组均值由 0.7272 提高到 0.7454，差值 `+0.0183`；seed 101 的三组均值由
0.7464 变为 0.7398，差值 `−0.0066`。两个 seed 各有 2/3 组配对为正，主要差异来自
seed 101/repeat 0：该组 C0 达到全轮最高的 0.7919，C1 为 0.7527。其余五组中，C1 有
四组提高、一组仅下降 0.0028。

以 cat_id 为重采样单位的 5,000 次 paired bootstrap 得到平均差 `+0.0053`、95% 区间
`[−0.0301, +0.0388]`，62.48% 的重采样差值高于零。中心方向与主结果一致，区间宽度反映
111 只猫和不同训练 seed 下仍然存在可见波动。预先声明的判断规则依据六组 complete-OOF
的均值与正向数量，本轮两项条件均已达到。

## 3. 指标变化说明了什么

六组结果的类别支持数合计为 kitten 90、adult 372、senior 204。两条规则的混淆矩阵合计为：

| 真实类别 | C0：预测 kitten / adult / senior | C1：预测 kitten / adult / senior | 正确数变化 |
| --- | ---: | ---: | ---: |
| Kitten | 78 / 8 / 4 | 70 / 15 / 5 | −8 |
| Adult | 20 / 252 / 100 | 13 / 279 / 80 | **+27** |
| Senior | 4 / 49 / 151 | 2 / 60 / 142 | −9 |

C1 改变 68 次猫级判断，其中 37 次从错误变为正确，27 次从正确变为错误，净增加 10 次
正确判断。adult recall 从 0.6774 提高到 0.7500；kitten recall 从 0.8667 变为 0.7778；
senior recall 从 0.7402 变为 0.6961。

- **普通 accuracy** 按 666 次评价中的总正确数计算。adult 数量最多，新增的 27 次 adult
  正确抵消了 kitten 和 senior 的变化，因此 accuracy 提高 0.0150。
- **Balanced accuracy** 先计算三类 recall，再给三类相同权重。kitten 与 senior 的 recall
  回落大于 adult recall 的增量，最终下降 0.0201。
- **Macro F1** 对三类 F1 等权平均，同时考虑 precision 和 recall。C1 减少 adult 被错分为
  kitten 或 senior 的次数，adult F1 从 0.7397 提到 0.7685；senior F1 近似持平，kitten F1
  下降，综合后 macro F1 提高 0.0059。
- **QWK** 对 kitten、adult、senior 的顺序距离加权。C1 的平均变化为 −0.0029，幅度很小，
  表示总体顺序一致性近似持平、中心略向 C0 倾斜。
- **Animal CE** 衡量聚合概率给真实类别分配的概率，数值越低越好。C1 从 0.7781 降到
  0.7521，说明按猫选择 checkpoint 与按猫评价概率具有更直接的目标一致性。

因此，本轮结果同时呈现两类信息：C1 提高主要指标 macro F1、总体正确率和结果稳定性，
并改善猫级概率损失；C0 保留更均衡的三类 recall 和极小的 QWK 优势。

## 4. 选择 epoch 的变化

C0 选中 epoch 的平均值为 4.71，C1 为 11.92。在 24 个 folds 中，C1 有 20 折选择更晚的
epoch、2 折相同、2 折更早。这个分布说明叫声级 CE 通常较早达到最低点，而同一只猫多条
叫声聚合后的概率在后续训练中仍继续改善。

C1 的作用属于训练流程优化：模型结构与参数量保持不变，最终使用的 checkpoint 更贴近
animal-level 目标。这一结论为后续 AST 内部结构实验提供统一参考流程，也让新的模块比较
建立在相同的 checkpoint 选择规则上。

## 5. 与 seed 17 历史观察的关系

seed 17 的三组历史结果为：C0 均值 0.7464，C1 均值 0.7570，差值 `+0.0106`，2/3 组为正。
该结果用于提出 IDEA-056，按计划只作为历史背景。新的 seeds 43/101 独立承担确认判断。

将历史 seed 17 与本轮两个新 seed 合并作描述时，9 组结果为：C0 均值 0.7400，C1 均值
0.7474，平均差 `+0.0074`，6/9 组为正。这个联合描述与新确认结果方向一致，同时保留
seed 101 均值略偏向 C0 的信息。

## 6. 阶段决策与下一步

本轮按 IDEA-056 计划形成以下阶段决策：

1. 后续 tuned frozen-AST reference 使用 **animal-level validation CE checkpoint selection**；
2. 论文将其记录为评价单位对齐带来的训练流程改进，主要确认结果为 `+0.0059` 和 4/6；
3. C0 保留为匹配消融，用于展示 checkpoint 选择单位本身的贡献；
4. 下一阶段先执行 AST 内部诊断，定位局部 patch、中间层表示和预训练领域差异中最值得
   推进的方向；
5. 依据诊断选择一个 AST 内部模块，与参数量匹配对照及原始 AST reference 完成首轮比较。

当前结论完成 reference 流程的阶段性确认，同时保持后续 architecture idea 的开放性。

## 7. 运行与完整性审计

Smoke 使用 seed 43/repeat 0/fold 0，包含 517 条 inner-training calls 与 112 条
inner-validation calls，`outer_test_accessed=false`。两条规则在 smoke 中都选择 epoch 2；
99,075 个 head 参数参与训练，6 个参数 tensor 均发生更新，概率和最大误差为
`9.01e−8`，最早并列 epoch 与双指标 patience 测试均通过。

正式运行完成 24/24 条共享训练轨迹、48 份 pipeline-fold predictions、12 份 complete-OOF
评价和 6 组主要配对。内部训练累计约 64.46 秒，outer 训练与预测累计约 41.48 秒；设备为
NVIDIA GeForce RTX 4060 Ti，记录到的峰值显存为 20,240,384 bytes。全套测试结果为
`128 passed, 1 warning`。C0 对 seeds 43/101 的历史 global-weighting 结果最大绝对差为
`1.60e−11`，确认匹配参考得到精确复现。

- Idea Card：`plan/IDEA-056_animal_level_checkpoint_selection_confirmation.md`；
- Protocol：`configs/protocol/meowagenet_idea056_checkpoint_selection_confirmation_v1.json`；
- Runner：`scripts/run_meowagenet_idea056_checkpoint_selection.py`；
- 机器可读结果：`metadata/experiments/meowagenet_idea056_checkpoint_selection_confirmation_v1_results.json`；
- 完整运行目录：`runs/meowagenet_idea056_checkpoint_selection_confirmation_v1/`；
- 执行代码 commit：`8018ccbff00a9a08b6b24ccb29cc34f12b0a2d86`；
- Idea Card SHA-256：`81bf3ba0baf5db6ed3fa33eb141a9a04ae64ebcce878df4f5056b7f81f651791`；
- Protocol SHA-256：`71bc38c05237fddf25724c7822f9c706c489453b03129a625d5357f2703069c3`；
- Runner SHA-256：`94186639855c46ea52c04142d135eaaba5cf0a36dd51d7ba19fa4d7bc35b768d`；
- Smoke summary SHA-256：`405d28fad447ea5eb3300f5057532188b703bab045d1f5580e08261a68370c39`；
- Execution lock SHA-256：`d9828e447a6d304f2b2d701c5bafed10dd06b99dffec80fecea78cf102fb3201`；
- Environment lock SHA-256：`6bc5a8f2e8b2224de7e14bd18e039c67f6c2d6565980545e77afe023f547ef26`；
- Evaluation summary SHA-256：`c82329943495cea148fab4a0d529aec6e98694acf5b65a6443a326cb8121c533`；
- Raw-prediction inventory SHA-256：`0615432a0ddc61036719df04e4463de3b2ebdf8f3661284bf970ba5d12c805ac`；
- Raw-prediction aggregate SHA-256：`b62e9c9b5c0cf0157aadf121afb5369f55565e1329c3fb4e445f842519f41001`。

运行目录共 140 个文件、5,441,198 bytes。31 个 JSON 结果与审计文件进入版本控制；109 个
CSV 原始预测保留在本地研究环境。inventory 覆盖全部 CSV，并提供聚合哈希用于后续核对。
