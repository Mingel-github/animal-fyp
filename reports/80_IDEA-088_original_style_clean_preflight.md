# IDEA-088 clean111 原 baseline 风格对照：CPU 预检

> 状态：本地 CPU preflight **GO**；独立训练前复核 **GO**；GPU/正式训练未授权、未启动。  
> 预检产物 SHA-256：`fb4fa4b55762bed639528387858b7f417f452083cde7e778792a40c4392eb449`

## 1. 本轮锁定内容

IDEA-088 使用项目自行清洗的 792 calls / 111 analysis cats，把原 categorical notebook 日志恢复出的 post-swap 猫角色删除 alias `049A` 后投影到 clean111。runner 只读取冻结角色清单，不重新执行 SGKF。

三条模型固定为：

- VGGish MLP：TensorFlow/Keras，原 final 实际 129 维输入，即 128 个 embedding 列加 `mean_freq`；
- A0：现有 frozen-AST call embedding 与 128 hidden head；
- C1：现有 20→60→128 bounded acoustic residual，cap 0.25。

所有模型采用原 final 风格配方：Adamax lr `0.003109800273709165`、epsilon `1e-7`、betas `(0.9,0.999)`、dropout `0.44571035356880917`、batch 128 prediction units、最多 1500 epochs、训练 loss early stopping、`min_delta=0.001`、patience 30、恢复记录的最佳权重；无独立 validation、无 HPO、无 gradient clip、无 AMP。

## 2. 冻结身份

| 产物 | SHA-256 |
|---|---|
| protocol | `c12bd2a5a4abe6b4f0af425a153747a7d6c96b6c9214023684ffc33625d21847` |
| runner | `f1cd6c9993b505e3e3f5f6d19488ef1b755c55eec8c63ffb68b4918e8d0db5cc` |
| tests | `7cebb432801827280783d65e1dab57d3efe95dc42fc67e6a4e212af10610bf20` |
| plan | `a03f2f0ee3449c0d013e006d5fbb99b58567dc53826991cfd8c5e8d791de58b3` |
| recovered roles | `a0ce4989985989ebdd982bbf565e33c080b4e314e232eb875f30fbdb4eac34b8` |
| CPU preflight | `fb4fa4b55762bed639528387858b7f417f452083cde7e778792a40c4392eb449` |

协议另外绑定了 data/cat manifests、937-row VGGish CSV、AST embedding cache、20 维声学 cache，以及 VGGish/A0/C1 三条被复用实现的准确哈希。协议状态为 `locked_before_cpu_preflight`；训练入口还要求总监显式授权并提供准确 preflight SHA。

## 3. 数据、角色与标签审计

- 原 VGGish CSV：937 rows；删除 `049A` 后 936 rows、111 cats。
- AST：792 calls、111 cats；标签顺序统一为 `0=kitten, 1=adult, 2=senior`。
- clean111 猫类别数：15/62/34；calls 类别数：134/405/253。
- VGGish 逐猫 target-derived label 与 AST label 全部一致。
- 20 个 `(split_seed, fold)` 身份完整且唯一；每 cell 的 train/test 猫零交集，并集严格等于 clean111。
- `049A` 在全部角色中缺席；`000A`、`046A` 在全部 20 cells 中始终位于训练侧。
- 强制换猫导致每 seed 均有两只猫不测试、两只替换猫测试两次，因此明确禁止 `111-cat complete OOF` 表述。

## 4. 模型与训练语义审计

- VGGish 输入列准确为 `0..127 + mean_freq`，`gender/target/cat_id/age_group` 均不进入模型；参数量 17,539。
- A0 参数量 99,075；C1 参数量 108,143。
- 在相同 cell 与 model seed 下，A0/C1 公共初态 SHA-256 均为 `dd1a9347db188080907052dd83b723a286cfc46413ecaaf133a4555566c817a8`，zero-init C1 与 A0 的初始 logits 逐位相同。
- A0/C1 的 model seed 固定为 `split_seed + fold`；post-build training seed 再加 1,000,000。每 epoch call 顺序和 dropout RNG 流配对。
- class weight 只按当前训练 units 计算：`N/(3*N_class)`；每 batch 加权 CE 除以 batch 样本数，epoch loss 按样本数累计。
- TensorFlow 分支在 patience 没有触发、训练达到最大预算时，也会显式加载 callback 保存的 `best_weights`，避免误留最后 epoch。
- TensorFlow/PyTorch 遇到 NaN/Inf loss 立即失败。
- 每 fit 必须先写 checkpoint lock、保存并复载权重且 state digest 一致，之后才能进行一次 test prediction。

本 runner 专项测试结果：**16 passed**。连同独立角色恢复测试共 **23 passed**。覆盖角色只读消费、129 维 VGG 输入、标签计数、batch-N loss、Keras min-delta/patience、预算耗尽 best restore、NaN fail-fast、A0/C1 配对初态、unit/cat 指标分离、训练授权 fail-closed 和 dependency hash。

## 5. AST tail-batch 审计

所有训练 calls 每 epoch 完整覆盖且不重复；不使用 `drop_last`。20 cells 均无单 call 尾批：

| seed | fold 0 | fold 1 | fold 2 | fold 3 |
|---:|---:|---:|---:|---:|
| 7270 | 633 / tail 121 | 603 / tail 91 | 579 / tail 67 | 643 / tail 3 |
| 860 | 651 / tail 11 | 608 / tail 96 | 591 / tail 79 | 607 / tail 95 |
| 5390 | 578 / tail 66 | 620 / tail 108 | 665 / tail 25 | 587 / tail 75 |
| 5191 | 583 / tail 71 | 603 / tail 91 | 649 / tail 9 | 618 / tail 106 |
| 5734 | 642 / tail 2 | 587 / tail 75 | 616 / tail 104 | 590 / tail 78 |

最小尾批为 2，不触发 PyTorch BatchNorm 的单样本限制。

## 6. 评价与解释边界

公开主指标为 cat probability-mean Macro-F1，同时报告 Accuracy、BA、三类 recall、CE、Brier。原生输入单位 F1 分开列出：VGGish row F1 与 AST call F1 不互相混同。

每个 seed 先对四折等权平均，再对五个 seed-level 估计报告均值和样本 SD。C1−A0 只在相同 seed×fold 内配对。20 folds、重复测试猫与五个 split seeds 都不当作新增独立动物。

IDEA-087 已经在每折完整 outer-dev 83/84 猫上重训；IDEA-088 不增加训练动物。潜在差异来自角色、强制 swap、训练配方、输入粒度和 checkpoint 规则。

## 7. 独立复核

独立复核重新运行两组测试，结果为 **23 passed**；重新计算 6 个冻结工件与 11 个协议依赖，均无哈希不一致。另以不等 batch 的 TensorFlow 合成样例实测，确认 `class_weight` 下 history loss 为 `sum(weighted CE)/N`，差异仅 `2.48e-8`；Adamax 默认 beta1/beta2/epsilon 与锁定值一致。

独立审查同时确认：

- Torch batch-N 与 epoch-N 聚合语义正确；
- 每 epoch 顺序唯一、完整覆盖训练 calls；
- normalization、声学填补与类别权重均 train-only；
- checkpoint lock 后才执行一次 test prediction；
- 60-fit 身份和五个 seed-level 汇总没有把 20 folds 当独立实验；
- 当前尚无 `fits` 目录或训练产物。

独立结论：**GO，仅建议先放行 A0 / seed 7270 / fold 0 / max-fits 1；首 fit 后仍须独立审计，再决定剩余 59 fits。**

## 8. 门禁结论

本地 CPU preflight 判定 **GO**，但这不是训练授权。下一步须先取得独立复核结论，再由总监绑定准确 preflight SHA，仅授权 `A0 / seed 7270 / fold 0` 一个技术 fit。该 fit 审计通过后，剩余 59 fits 需要第二次明确授权；不得按成绩择停或扩展。
