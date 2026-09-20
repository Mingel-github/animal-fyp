# AST Adapter 确定性复现实验结果

## 结论

Probe-guided Adapter 相对于匹配的冻结 AST head-only 对照，没有表现出稳定、独立的增益。这个结论只针对当前两层瓶颈 Adapter，并不否定冻结 AST 表示本身的价值：在既有实验中，冻结 AST 仍优于较早的 VGGish 基线。

## 同种子复跑与确定性诊断

我们使用原 formal-v2.1 的 base seed 17，重新完成了 3 repeats × 4 folds 的实验。

- Head-only 完全复现：12/12 个预测文件逐位一致。
- Adapter 未完全复现：12/12 个预测文件都不相同；三次重复的平均 Macro-F1 从 0.7348 变为 0.7246。
- Adapter 相对 Head-only 的差值由 +0.0011 变为 -0.0091。

模型 checkpoint 和配置哈希均未变化。原因是原运行脚本虽然设置了随机数种子，但没有强制使用确定性的 CUDA 算法，也没有关闭可能非确定性的 scaled-dot-product attention kernel。

启用确定性算法、设置 cuBLAS workspace，并固定使用 math attention kernel 后，同一 repeat、fold 和 seed 的两次连续 Adapter 拟合产生了完全一致的 epoch history 和预测概率。因此，前述同种子差异应被视为计算确定性诊断，而不是额外的效果估计。

## 新种子确定性复现

在冻结原有训练配方后，我们使用此前未使用的 base seeds 151、307 和 509，并在相同的 3 组重复分组四折划分上重新评估。这属于在同一批 111 只猫上的事后稳健性复现，不是新的外部验证或确认性证据。

| Pipeline | 平均 OOF Macro-F1 | 标准差 |
|---|---:|---:|
| AST head-only | 0.7253 | 0.0385 |
| Probe-guided AST Adapter | 0.7195 | 0.0405 |
| Adapter − Head | -0.0057 | 0.0375 |

在 9 组配对 OOF 比较中，Adapter 有 5 组为正、4 组为负；单组差值范围为 -0.0823～+0.0320。部分划分中确实出现了正向提升，因此这些结果仍可作为后续 Adapter 设计和稳定性研究的依据；但符号不稳定且总体均值为负，不能支持“当前 Adapter 能稳定优于 Head-only”的结论。

## 科学解释与可复用价值

原 formal-v2.1 中 Adapter 的平均优势本来就很小（+0.0052）。新的独立种子组使差值反向，而确定性设置排除了计算随机性的歧义。当前最准确的表述是：

- 冻结 AST 相对于旧 VGGish 基线仍然有价值；
- 当前两层 probe-guided bottleneck Adapter 在部分 split 上有正向结果，但总体不构成稳健改进；
- 后续工作不应把 AST 的整体提升归因于该 Adapter；
- 正向 split 和失败 split 都应保留，可用于分析什么条件下 Adapter 有效，并指导更有明确声学先验的结构设计。

完整产物位于 `runs/meowagenet_idea065_adapter_seed_replication_v1/`。
