# IDEA-053 | AST–VGGish Nested Probability Fusion

## Provenance

- Origin: repository priority-3 route, activated by paired AST–VGGish complementarity diagnosis
- Stage: post-evidence exploratory candidate
- Date: 2026-09-13

## Primary angle

- 主要角度：H. Knowledge combination
- 支持角度：E. Feature extraction / J. Evaluation and robustness

## Located evidence

### 当前 AST reference

IDEA-052 的 R0 使用 frozen AST 768 维 call embeddings、`768 → 128 → 3` head、strict
global class-balanced call loss、learning rate 0.006 和 animal-level checkpoint selection。
三份 seed-17 complete OOF 的 animal macro F1 为 `0.7659、0.7565、0.7487`，均值
`0.7570`。

### VGGish reference

Formal-v2.1 的 VGGish+MLP 使用官方 128 维 embedding rows、`128 → 128 → 3` head 和原始
baseline recipe。与当前 AST 对齐的 base-seed-17 三份 complete OOF macro F1 为
`0.7098、0.6694、0.6593`，均值 `0.6795`。

### 配对互补性诊断

`metadata/experiments/meowagenet_idea053_ast_vggish_complementarity_v1.json` 对齐相同三个
repeat、相同 111 只猫，共 333 次猫级判断：

- 两者都正确 209 次；
- 只有 AST 正确 43 次；
- 只有 VGGish 正确 21 次；
- 两者都错误 60 次；
- 两者预测标签不同 66 次；
- 任一模型正确的描述性上界为 0.8198 plain accuracy；
- 诊断性固定 `0.7 × AST + 0.3 × VGGish` 的 mean macro F1 为 0.7630，相对 AST
  `+0.0060`，三次中两次提高。

固定 0.7 权重来自已观察 OOF，只用于确认概率融合值得进入下一实验。IDEA-053 在每个 outer
fold 内重新训练两条 head，并只用 inner-validation cats 选择融合权重。

## Observed bottleneck

AST 总体更强，同时仍有 21/333 次判断由 VGGish 独占正确。VGGish 的 AudioSet CNN
embedding 与 AST transformer embedding 对声学事件的压缩方式不同；单一 AST head 无法
直接访问 VGGish 保留的互补信息。简单 confidence selection 的方向随 repeat 变化，因此
需要在训练数据内部选择两条概率路径的相对权重。

## Idea claim

保留两条已建立的 frozen embedding classifiers，在猫级概率空间使用 inner-validation
animal cross-entropy 选择 convex fusion weight，可以吸收 VGGish 的独占正确信息，同时
让较强的 AST 保持主要贡献，从而提高 animal-level macro F1。

## Proposed mechanism

每个 repeat/fold 中独立执行：

1. AST path 按 IDEA-052 R0 recipe 在 inner train 上训练，由 inner-validation animal
   cross-entropy 选择 epoch，再在 outer train 上按该 epoch 数重训并预测 outer cats；
2. VGGish path 按 formal-v2.1 baseline recipe 在 inner train embedding rows 上训练，由
   unit-level validation loss 选择 epoch，再在 outer train 上按该 epoch 数重训并预测
   outer cats；
3. 将两条 inner-validation 预测都平均到 cat level；
4. 在 `α ∈ {0.0, 0.1, ..., 1.0}` 中最小化 inner-validation animal cross-entropy；
5. outer fused probability 为 `α × P_AST + (1 − α) × P_VGGish`。

`α=1` 等于纯 AST，`α=0` 等于纯 VGGish。自适应融合具有自动回退能力，不增加神经网络
参数。

## Predictions

1. 若两种 embedding 的概率误差具有可利用互补性，inner-selected fusion 将在至少 2/3
   complete OOF repeats 上提高 macro F1，且平均增益达到 0.005；
2. 若互补性主要来自偶然的 outer composition，inner alpha 会跨 fold 大幅变化，fusion
   总体接近或低于 AST；
3. 若 VGGish 主要改善 AST 的边界样本，fusion 会改变少量 cats，同时提高 balanced
   accuracy、QWK 或某个弱类别 recall；
4. 若概率 calibration 差异主导融合，selected alpha 与两条 inner animal CE 的相对大小
   将呈现清楚关系。

## Minimum discriminating experiment

### 数据与 splits

- AST：792 个 768 维 frozen call embeddings；
- VGGish：936 个 128 维官方 embedding rows；
- 三个年龄类别、111 只猫；
- 复用 `meowagenet_formal_v2_nested_roles.csv` 的 animal-ID-disjoint nested splits；
- 首轮 base seed 17，repeat 0/1/2，每个 repeat 四折。

### Pipeline

- `A0_ast_reference`；
- `V0_vggish_reference`；
- `F1_inner_ce_probability_fusion`。

每折训练一条 AST head 和一条 VGGish head，共 24 个 model fits；fusion 直接组合两条猫级
概率，不增加训练 fit。最终生成三条 pipeline × 三个 repeat，共 9 份 111-cat complete OOF。

### Primary 与支持指标

- primary：animal macro F1；
- supporting：balanced accuracy、QWK、plain accuracy、逐类 recall/F1、confusion matrix；
- mechanism：每折 selected alpha、inner CE curve、outer changed/gained/lost cats；
- uncertainty：paired cat bootstrap。

### 继续条件

F1 相对 A0 的 mean macro-F1 delta 至少 `+0.005`、至少 2/3 repeats 为正，并由 balanced
accuracy、QWK、最差 repeat 或关键类别 recall 中至少一项支持时，扩展 base seeds 43/101。
若首轮只形成小幅或不稳定变化，保留为融合消融并转向 feature-level fusion 或
LayerNorm/SSF/BitFit。若概率融合稳定提高，再以 F1 为 teacher/ensemble reference 评估
distillation 是否能将双模型知识压缩回单模型。

## Novelty location

- task：面向多叫声猫级年龄分类的跨音频 backbone 互补；
- method：nested animal-level calibration/fusion，而非直接使用外层结果固定权重；
- efficiency：复用 frozen embeddings，只训练两个轻量 classification heads；
- interpretation：独占正确动物、alpha 与类别变化可直接解释融合收益来源。

## Status

- Status: shortlisted / implementation authorized
- Human decision owner: project team
