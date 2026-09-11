# AST Cat-Balance Global Weighting 修正版复验计划

## 1. 计划状态与边界

状态：pre-execution protocol design。

本计划处理 AST accuracy-enhancement 阶段发现的 loss implementation 问题。既有 protocol、runner、结果、报告和 SHA-256 归档继续作为历史证据保存，不作覆盖或改写。新实验只回答一个问题：严格按全局目标定义的 cat-balanced loss 是否优于匹配的 class-balanced loss。

最终模型选择由团队成员和导师结合本计划的结果决定。

## 2. 已观察到的结果

以下数字直接来自已完成报告，不是本计划的新结果。

| 范围 | A0 class-balanced Macro F1 | A1 cat-aware Macro F1 | A1 − A0 |
|---|---:|---:|---:|
| base seed 17，3 repeats | 0.7488 | 0.7765 | +0.0277 |
| base seeds 43/101，6 repeats | 0.7333 | 0.7242 | −0.0091 |
| 三个 base seeds，9 repeats | 0.7385 | 0.7416 | +0.0031 |

九组 paired comparison 中有六组为正。A1 的 accuracy 较高，A0 的 balanced accuracy 与 QWK 略高。不同 seed 范围的方向发生变化，因此当前证据支持“效果很小且 seed-dependent”。

全 12 层 uniform-start scalar fusion 在两个 sampling/loss 条件下均明显低于 final-layer AST。该结果关闭的是这一种具体 fusion 实现，不外推到所有 cross-layer fusion 方法。

## 3. 实现审计发现

原 runner 的 lookup weight 定义为：

$$
w_i=\frac{N}{3\,N_{\mathrm{cats},y_i}\,N_{\mathrm{calls},c_i}}
$$

其中 \(N\) 是训练集 call 总数，\(N_{\mathrm{cats},y_i}\) 是类别 \(y_i\) 的 cat 数，\(N_{\mathrm{calls},c_i}\) 是 cat \(c_i\) 的 call 数。这个 lookup table 在全局求和时具有预期的 equal-class、within-class equal-cat 总量。

原 runner 随后在每个 micro-batch 内计算：

$$
L_B=\frac{\sum_{i\in B}w_i\ell_i}{\sum_{i\in B}w_i}
$$

分母随 micro-batch 的 cat 和 class 组成变化。经过随机 batching 后，optimizer 实际接收的累计系数不再严格保持全局 equal-cat 总量。原实验因此准确描述为 **per-mini-batch normalized cat-aware weighting**，不能作为严格 global cat-balanced loss 的最终验证。

## 4. 研究问题与可证伪假设

研究问题：在相同 AST backbone、head、split、seed、优化器和训练预算下，严格 global cat-balanced loss 能否稳定提升 animal-level age classification 的 Macro F1？

目标假设 H1：global cat-balanced loss 减少多-call cat 对梯度的支配，使 animal-level Macro F1 高于 global class-balanced control。

竞争解释：

- R1：先前增益来自 seed variation，修正后两组仍接近。
- R2：cat balancing 改善少数 cat 的影响，却削弱 class boundary 的有效样本量，Macro F1 下降。
- R3：数据集中 call count 与年龄类别或录音条件相关，weighting 改变的是 class/context mixture，而非独立的 cat contribution。

## 5. 配对实验组

### C0：Global class-balanced AST

每个 call 的固定权重：

$$
w_i^{C0}=\frac{N}{3\,N_{\mathrm{calls},y_i}}
$$

### C1：Global cat-and-class-balanced AST

每个 call 的固定权重：

$$
w_i^{C1}=\frac{N}{3\,N_{\mathrm{cats},y_i}\,N_{\mathrm{calls},c_i}}
$$

C0 与 C1 使用同一 frozen AST backbone、同一 classification head、同一超参数、同一 fold、同一 model seed 和同一 batch order。当前 unweighted call-level validation cross-entropy 继续用于 checkpoint selection，以隔离 training-loss weighting 的影响。

## 6. Loss 修正规范

有效 accumulation window 固定为 32 calls。micro-batch size 固定为 8，gradient accumulation steps 固定为 4。

每个完整 micro-batch 计算：

$$
L_{micro}=\frac{\sum_{i\in B}w_i\ell_i}{32}
$$

四个 micro-batch 的梯度累积后再执行 optimizer step。epoch 末的 partial accumulation window 继续使用固定分母 32，并记录其中的有效 call 数。禁止使用当前 micro-batch 的 `weights.sum()` 作为分母。

runner 必须额外保存每个 epoch 的实际累计 effective coefficient，至少按 cat、class、fold 和 pipeline 汇总。这样可以从日志核验目标权重与 optimizer 实际接收的权重是否一致。

## 7. 运行前测试

正式 outer fit 启动前必须全部通过：

1. 全量 lookup test：C0 的三个 class 总权重相等；C1 的三个 class 总权重相等，且每个 class 内各 cat 总权重相等。
2. Deterministic epoch test：固定 batch order 后，日志累计的 cat/class effective coefficient 与解析目标一致，允许浮点误差。
3. Denominator test：训练 loss 路径不得出现以当前 `weights.sum()` 为分母的归一化。
4. Unit-weight equivalence test：全部权重设为 1 时，gradient 与普通 mean cross-entropy 在相同 32-call accumulation window 上一致。
5. Smoke test：完成一个短 fold 的 train、validation、checkpoint reload 和 animal-level aggregation；smoke 输出不得进入正式汇总。

## 8. 正式运行矩阵

- pipelines：C0、C1；
- base seeds：17、43、101；
- split repeats：0、1、2；
- outer folds：0、1、2、3；
- 总计：\(2\times3\times3\times4=72\) 个 outer fits。

每一对 C0/C1 必须共享 fold membership、model seed、batch order 和训练预算。缺失、重跑和失败 fit 单独记录，不能静默替换 seed。

## 9. 指标与分析单位

Primary endpoint：111-cat complete-OOF animal-level Macro F1。

Secondary endpoints：balanced accuracy、QWK、plain accuracy、三类 recall、per-cat call count 与预测置信度/错误的关系。

九组 complete-OOF 是在同一 111 只 cat 上的重复测量，不把它们表述为 999 个独立 animal observations。推断采用以 cat 为 resampling unit 的 paired hierarchical bootstrap，同时保留九个 seed/repeat paired deltas、三组 base-seed means 和 descriptive interval。若层级 bootstrap 实现无法通过审计，只报告 descriptive uncertainty，不提供伪精确的显著性结论。

## 10. 预先定义的结果解释

以下规则用于约束解释，不自动替团队作最终选择：

- **支持稳定改进**：C1 − C0 的平均 Macro F1 至少为 +0.010，三个 base-seed mean 至少两个为正，九组 paired comparison 至少六组为正，paired-cat hierarchical bootstrap 95% interval 的下界高于 0。
- **小效应或混合证据**：平均绝对差小于 0.010，或不同 base seed 方向不一致，或 paired interval 覆盖 0。
- **缺乏改进证据**：平均差不大于 0，且多数 base-seed mean 不为正。

accuracy、balanced accuracy、QWK 和 class recall 用于解释 trade-off。模型选择优先依据 primary endpoint，同时完整报告 secondary metrics，避免只挑选有利指标。

## 11. 预期产物

- executable protocol：`configs/protocol/meowagenet_ast_cat_balance_global_weighting_v1.json`
- independent runner：`scripts/run_meowagenet_ast_cat_balance_global_weighting_v1.py`
- run directory：`runs/meowagenet_ast_cat_balance_global_weighting_v1/`
- machine-readable result：`metadata/experiments/meowagenet_ast_cat_balance_global_weighting_v1_results.json`
- result report：`reports/23_AST_cat_balance_global_weighting_results.md`

报告必须同时引用本计划、代码 commit、environment lock、protocol hash、runner hash、raw prediction hash 和 summary hash。

## 12. 历史证据保护

以下既有材料保持只读：

- `reports/21_AST_cat_balancing_and_multilayer_fusion_results.md`
- `reports/22_AST_cat_balancing_seed_expansion_results.md`
- `configs/protocol/meowagenet_ast_accuracy_enhancement_v1.json`
- `configs/protocol/meowagenet_ast_cat_balance_seed_expansion_v1.json`
- 对应 runner、metadata、logs、predictions 与 hash inventory。

新报告需要明确说明旧结果测量的是 per-mini-batch normalized cat-aware weighting，并将严格 global cat-balanced loss 的结论限定在本次修正版复验结果内。
