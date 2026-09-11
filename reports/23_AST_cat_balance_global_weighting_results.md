# AST Cat-Balance 全局加权修正版复验结果

## 结论摘要

本轮完成严格 global weighting 修正版复验：2 条 pipeline × 3 个 base seed × 3 个
repeat × 4 折，共 **72/72 个 outer fit**，形成 18 份 complete OOF 和 9 组逐猫配对比较。
每份 OOF 均覆盖同一批 792 条 call、111 只猫以及 kitten/adult/senior 三个年龄类别。

主结果为：

| Pipeline | Animal macro F1，mean ± SD | Balanced accuracy | QWK | 普通 accuracy |
| --- | ---: | ---: | ---: | ---: |
| C0 global class-balanced | **0.7400 ± 0.0280** | **0.7613** | **0.6543** | **0.7287** |
| C1 global cat+class-balanced | 0.7326 ± 0.0278 | 0.7468 | 0.6353 | 0.7197 |
| C1 − C0 | **−0.0074** | −0.0145 | −0.0191 | −0.0090 |

C1 在 9 组配对中胜出 3 组；三个 base-seed mean 分别为 −0.0224、−0.0011、
+0.0014，1/3 为正。以 cat 为重采样单位的 10,000 次 paired hierarchical bootstrap
给出 95% 区间 **[−0.0380, +0.0238]**，31.57% 的重采样结果偏向 C1。

按照预先写入计划的判定规则，本轮归入 **“缺乏改进证据”**。效应幅度较小，整体方向
偏向 C0；C1 在 adult recall 上保留 +0.0072 的局部优势，C0 在其余总体指标、kitten
recall 和 senior recall 上领先。C0 继续作为 tuned frozen-AST reference，严格 global
cat-and-class weighting 作为已完成的 loss ablation 写入论文证据链，后续 AST 新思路
继续保持开放。

## 1. 修正了什么

旧 A0/A1 runner 先生成了具有正确全局含义的权重 lookup table，随后在每个 8-call
micro-batch 内计算：

$$
L_B=\frac{\sum_{i\in B}w_i\ell_i}{\sum_{i\in B}w_i}.
$$

分母随着 batch 中猫和年龄类别的组成而改变。四个 micro-batch 累积时，各批获得近似
相同的总贡献，原先设计的“同类中每只猫总权重相等”会被 batch composition 改写。
因此，旧结果准确对应 **per-mini-batch normalized cat-aware weighting**。

本轮固定有效 accumulation window 为 32 calls。每个 micro-batch 使用：

$$
L_{micro}=\frac{\sum_{i\in B}w_i\ell_i}{32},
$$

连续四个 micro-batch 累积梯度后更新一次参数。epoch 尾部的 partial window 同样使用
固定分母 32，并记录实际 call 数。这样，每条 call 的 optimizer-effective coefficient
始终为其全局权重除以 32，batch composition 只改变处理顺序。

两条修正版 pipeline 为：

| Pipeline | 每条 call 的固定权重 | 全局目标 |
| --- | --- | --- |
| C0 | $N/(3N_{calls,y_i})$ | 三个年龄类别的 loss 总权重相等 |
| C1 | $N/(3N_{cats,y_i}N_{calls,c_i})$ | 三类总权重相等，同类中每只猫总权重相等 |

例如某只 adult 猫有 39 条训练 call，另一只 adult 猫有 1 条训练 call，C1 会让前者每条
call 获得较小权重、后者的单条 call 获得较大权重，最终两只猫在该年龄类中的总权重
相同。C0 则让 adult 类的所有 call 共享同一个类别权重。

## 2. 固定条件与运行审计

C0 和 C1 共享 frozen AST final embedding、`768 → 128 → 3` head、dropout 0.4457、
Adamax、学习率 0.006、相同 animal-independent nested splits、model seed、batch order、
训练预算与 unweighted inner-validation checkpoint selection。

运行前测试覆盖以下内容：

- C0 三类全局权重总量相等；
- C1 三类总量相等，且同类每只猫的总量相等；
- 固定 batch order 下，epoch 日志的 effective coefficients 与解析目标一致；
- loss 路径使用固定 32-call 分母；
- unit weights 下，4 × 8 micro-batch 的梯度与 32-call mean CE 梯度一致；
- smoke 完成 inner train、validation、checkpoint 保存/重载和 animal aggregation。

Smoke 仅访问 inner train/validation，产生 2 个短 fit，并从正式汇总中排除。两条 pipeline
重载 checkpoint 后的最大概率差均为 0.0。正式阶段完成 72 个 fit；36 对 C0/C1 fold
在所有共同 inner/outer epochs 上的 batch-order SHA-256 均一致。每个 epoch 的日志保留
完整 call 覆盖、batch size、optimizer step、partial window 以及按 class/cat 汇总的实际
effective coefficient。

## 3. 九组 complete-OOF 主结果

| Base seed | Repeat | C0 macro F1 | C1 macro F1 | C1 − C0 |
| ---: | ---: | ---: | ---: | ---: |
| 17 | 0 | **0.7674** | 0.7280 | −0.0395 |
| 17 | 1 | **0.7479** | 0.7267 | −0.0212 |
| 17 | 2 | **0.7240** | 0.7174 | −0.0066 |
| 43 | 0 | **0.7357** | 0.7318 | −0.0040 |
| 43 | 1 | 0.7274 | **0.7574** | +0.0300 |
| 43 | 2 | **0.7184** | 0.6890 | −0.0294 |
| 101 | 0 | **0.7919** | 0.7693 | −0.0225 |
| 101 | 1 | 0.7496 | **0.7685** | +0.0189 |
| 101 | 2 | 0.6977 | **0.7056** | +0.0079 |
| **九组平均** |  | **0.7400** | 0.7326 | **−0.0074** |

按 base seed 汇总：

| Base seed | C0 macro F1 | C1 macro F1 | C1 − C0 | C1 胜出次数 |
| ---: | ---: | ---: | ---: | ---: |
| 17 | **0.7464** | 0.7240 | −0.0224 | 0/3 |
| 43 | **0.7272** | 0.7261 | −0.0011 | 1/3 |
| 101 | 0.7464 | **0.7478** | +0.0014 | 2/3 |

seed 17 的三组均偏向 C0；seed 43 两者均值接近；seed 101 的均值轻微偏向 C1。九组
差值范围为 −0.0395 到 +0.0300，说明这项 weighting 对训练轨迹有可见影响，同时缺少
稳定的跨 seed 正方向。

## 4. 各指标与类别变化

Macro F1 分别计算 kitten、adult、senior 的 F1 后取平均，同时反映各类 precision 和
recall。C0 高 0.0074；两组 sample SD 只差 0.0003，稳定性相近。

Balanced accuracy 是三类 recall 的平均。C0 高 0.0145，表示三个年龄类别的平均识别
比例更高。

QWK 根据年龄顺序给错误分配距离代价。kitten 与 senior 的跨两级误判受到更重惩罚。
C0 高 0.0191，表示其预测与年龄顺序的一致性更好。

普通 accuracy 是每份 111-cat OOF 中预测正确的猫所占比例。九组累计形成 999 次猫级
评估，C0 判对 728 次，C1 判对 719 次，净差 9 次。

| Pipeline | Kitten recall | Adult recall | Senior recall |
| --- | ---: | ---: | ---: |
| C0 | **0.8519** | 0.6935 | **0.7386** |
| C1 | 0.8370 | **0.7007** | 0.7026 |
| C1 − C0 | −0.0148 | **+0.0072** | −0.0359 |

九组累计正确数进一步展示了变化位置：

- kitten：C0 115，C1 113；
- adult：C0 387，C1 391；
- senior：C0 226，C1 215。

C1 多识别正确 4 次 adult，同时减少 2 次 kitten 和 11 次 senior 的正确预测。adult 的
局部改善因此没有转化成总体 macro F1、balanced accuracy、QWK 或 accuracy 的提升。

## 5. 配对 bootstrap 与 call 数诊断

层级 bootstrap 每次以 `cat_id` 为单位重采样 111 只猫，并在每只重采样猫上保留九组
C0/C1 配对 OOF 预测。10,000 次迭代得到：

- C1 − C0 macro F1 的 bootstrap mean：−0.00739；
- 95% descriptive interval：[−0.03796, +0.02377]；
- 差值高于 0 的重采样比例：31.57%。

区间覆盖 0，反映当前 111-cat 样本下仍存在两种方向；区间中心和多数重采样更偏向 C0。

每只猫的 call 数与九组平均预测置信度呈中等负相关：C0 的 Spearman $\rho=-0.3492$
（p=0.00017），C1 为 $\rho=-0.3173$（p=0.00069）。call 较多的猫在本轮通常拥有更低
的平均置信度，可能同时包含更多录音变化或更难判断的声音。call 数与九组正确率的相关
较弱：C0 为 $\rho=0.1251$（p=0.1909），C1 为 $\rho=0.0961$（p=0.3157）。因此，更多
call 与“更容易判对”之间没有形成清晰的单调关系。

## 6. 与旧 A0/A1 结果的关系

旧 per-mini-batch normalized 实验的九组均值为 A0 0.7385、A1 0.7416，A1 − A0 为
+0.0031，且 6/9 配对为正。本轮 strict global 实验得到 C0 0.7400、C1 0.7326，
C1 − C0 为 −0.0074，且 3/9 配对为正。

两个阶段共同给出一条清晰的方法结论：

1. per-mini-batch normalized cat-aware weighting 曾产生一个很小、seed-dependent 的
   正向信号；
2. 固定 32-call 分母后，严格 equal-class、within-class equal-cat 目标在当前数据上没有
   延续该信号；
3. 逐猫均衡对 adult recall 有局部帮助，同时对 senior recall 的影响更大，最终 primary
   macro F1 平均下降 0.0074。

因此，论文可将旧结果作为 implementation audit 触发点，将本轮作为目标函数的正式
修正版复验。C0 是当前更合适的 tuned frozen-AST reference，C1 提供完整、有价值的
loss ablation 证据。

## 7. 阶段结论与后续位置

预设“稳定改进”需要同时达到四项：平均增益至少 +0.010、至少 2/3 base-seed means
为正、至少 6/9 配对为正、bootstrap 区间下界高于 0。实际结果为 −0.0074、1/3、3/9、
区间下界 −0.0380。按预设分类，本轮结论为 **strict global cat balancing 缺乏改进
证据**。

本轮完成的是当前 weighting 路线的阶段性收尾：

- C0 保留为 tuned frozen-AST reference；
- C1 保留为已完成的 strict global weighting ablation；
- AST backbone 的总体优势、已有 adapter/LoRA/fusion 证据继续保持原有位置；
- 后续新的 AST accuracy-enhancement idea、超参数调整和最终重复实验仍可独立推进。

## 8. 文件、环境与哈希

- 计划：`plan/AST_cat_balance_global_weighting_retest.md`；
- Protocol：`configs/protocol/meowagenet_ast_cat_balance_global_weighting_v1.json`；
- Runner：`scripts/run_meowagenet_ast_cat_balance_global_weighting_v1.py`；
- 机器可读结果：
  `metadata/experiments/meowagenet_ast_cat_balance_global_weighting_v1_results.json`；
- 完整运行目录：`runs/meowagenet_ast_cat_balance_global_weighting_v1/`；
- 代码 commit：`20077e4c941280d0f00fb3d6dc1ab0e805fb98e7`；
- plan SHA-256：`c8504eedf0ec874ecaa2e8b7b3e16900ece78908e1344a6bc746a336c9cc72e6`；
- protocol SHA-256：`d377bcaf5e3c2187c2ad97792bdf1f2a759f8bcf95358128b91d698d20c87244`；
- runner SHA-256：`a0a217e76a99f2d9f3e00d5caa2d21c81c5e37e53237c6ac311318ad8c39674a`；
- execution-lock SHA-256：`4b8c75d46176e98223a4f6e0b3f83efe1055a808cc7106f6ab4baa837ae805bc`；
- environment-lock SHA-256：`57107ad3851c39d827a5b3862d5d348b20a54a441a9323585050e02cfae27940`；
- evaluation-summary SHA-256：`46affeead827f172eb9acd9dae0ffc26dac7a5d4e9ee5866d0c7f42bc0cfbbe7`；
- raw-prediction inventory SHA-256：`560dba579c9a6f14882eabee1746c8deaec4a692aa6908f3b0bc531ab8dd9584`；
- raw-prediction aggregate SHA-256：`14f01ff60d0a5cde03ef6156af815308291717849f555f1ba66bd31dbc9dc568`。

运行目录共有 176 个文件、9,283,808 bytes：81 个 JSON 审计文件进入版本控制；93 个
CSV 和 2 个 smoke checkpoint 保留在本地研究环境。raw-prediction inventory 单独覆盖
72 个 call-level outer predictions、18 个 animal-level complete OOF 和 1 个逐猫变化表，
合计 91 个文件、1,433,017 bytes。
