# IDEA-058｜受约束 AST 顶层适配结果

> 执行日期：2026-09-15
> 状态：首轮独立验证完成，按预设 gate 形成阶段性收尾
> 锁定代码提交：`517169bac797bca3c2a372e3ec7dd82483c99b3e`

## 1. 本轮研究问题

AST internal diagnosis 曾观察到 Last-2 adaptation 的 inner-only Macro F1 从 `0.7189` 提高到
`0.7439`。IDEA-058 将这条 mixed signal 转化为受约束实验：仅更新 AST 顶部一个或两个
Transformer blocks，并用较小 encoder learning rate 适应猫叫年龄分类。

正式比较包含三条同期配对 pipeline：

- `R0_tuned_frozen_ast`：冻结 AST，只训练标准 `768 → 128 → 3` 分类头；
- `M1_top_block_adaptation`：更新候选选择确定的顶部 block、final LayerNorm 与同一分类头；
- `C1_bottom_block_control`：更新同样数量的底部 block、final LayerNorm 与同一分类头。

C1 与 M1 的可训练参数量完全相同，因此两者的差异直接回答“更新位置是否重要”。三条 pipeline
共享 111 只猫、792 条 call、animal-ID-disjoint roles、全局类别平衡 call 权重、animal-level
validation CE checkpoint selection 和猫内 call probability 算术平均。

## 2. Inner-only 候选选择

四个候选各运行 3 repeats × 4 folds，共 48 个 inner-only fit。候选阶段只读取 outer training
role 内的 train/validation 数据，`outer_test_accessed=false`。

| 候选 | 更新范围 | Encoder LR | Inner animal Macro F1 | Inner animal CE | 可训练参数 |
| --- | --- | ---: | ---: | ---: | ---: |
| `top1_lr1e-5` | 最后 1 个 block + final LN | `1×10⁻⁵` | **0.7460 ± 0.0881** | 0.7251 | 7,188,483 |
| `top1_lr3e-6` | 最后 1 个 block + final LN | `3×10⁻⁶` | 0.7353 ± 0.0997 | **0.7023** | 7,188,483 |
| `top2_lr3e-6` | 最后 2 个 blocks + final LN | `3×10⁻⁶` | 0.7275 ± 0.0866 | 0.7225 | 14,276,355 |
| `top2_lr1e-5` | 最后 2 个 blocks + final LN | `1×10⁻⁵` | 0.7092 ± 0.0913 | 0.7276 | 14,276,355 |

预设选择规则以平均 inner animal Macro F1 为首要依据，因此锁定 `top1_lr1e-5`。这一结果同时
完成了结构收缩：最后一个 block 的表现高于 Last-2 候选，并将可训练参数从约 1428 万降到约
719 万。

## 3. Smoke 与执行锁

- R0 可训练参数为 `99,075`；
- M1/C1 均为 `7,188,483`，其中 encoder `7,089,408`、head `99,075`；
- M1 更新 zero-based block `11`，C1 更新 block `0`，两者均更新 final LayerNorm；
- M1/C1 使用相同预训练 AST 状态、分类头初始状态和 post-build RNG；
- smoke 初始 logits 最大差为 `0`，checkpoint 重载后概率最大差为 `0`；
- 12 个正式 fold group 的共同 epoch batch-order hashes 全部匹配；
- execution lock 在 outer-test 评估前固定候选、代码、环境和数据哈希。

## 4. Complete-OOF 结果

每条 pipeline 完成 3 repeats × 4 folds。每个 repeat 拼接四个 outer folds 后覆盖全部 111 只猫
一次，共 9 个 complete-OOF evaluations、36 个正式 fit。

| Pipeline | Animal Macro F1 | Balanced accuracy | QWK | Accuracy | Animal CE |
| --- | ---: | ---: | ---: | ---: | ---: |
| R0 tuned frozen AST | 0.7570 ± 0.0086 | 0.7645 | 0.6721 | 0.7568 | **0.7160** |
| M1 top-block adaptation | **0.7607 ± 0.0371** | **0.7739** | **0.6725** | **0.7568** | 0.7248 |
| C1 bottom-block control | 0.7385 ± 0.0218 | 0.7440 | 0.6413 | 0.7417 | 0.7406 |

M1 相对 R0 的平均变化为：Macro F1 `+0.0037`、Balanced Accuracy `+0.0095`、QWK
`+0.0003`、Accuracy `0.0000`、Animal CE `+0.0089`。顶部适配主要改善类别平衡指标；整体
正确率保持相同，概率交叉熵有小幅增加。

### 4.1 Repeat-level 配对结果

| Repeat | R0 Macro F1 | M1 Macro F1 | C1 Macro F1 | M1−R0 | M1−C1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7659 | 0.7726 | 0.7499 | +0.0067 | +0.0227 |
| 1 | 0.7565 | 0.7904 | 0.7522 | +0.0339 | +0.0381 |
| 2 | 0.7487 | 0.7190 | 0.7133 | −0.0297 | +0.0057 |
| **平均** | **0.7570** | **0.7607** | **0.7385** | **+0.0037** | **+0.0222** |

M1 在 `2/3` 个 repeats 中超过 R0，并在 `3/3` 个 repeats 中超过参数量匹配的 C1。配对
cat-cluster bootstrap 给出的 M1−R0 Macro F1 95% 区间为 `[-0.0224, +0.0355]`；M1−C1 的
对应区间为 `[-0.0062, +0.0592]`。方向上，顶部更新优于底部更新的信号更一致；相对 frozen
R0 的净增益受到 repeat 2 回落影响。

### 4.2 类别表现

| Pipeline | Kitten recall | Adult recall | Senior recall | Kitten F1 | Adult F1 | Senior F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R0 | 0.8000 | 0.7581 | **0.7353** | 0.7914 | 0.7853 | **0.6943** |
| M1 | **0.8667** | **0.7688** | 0.6863 | **0.8212** | **0.7877** | 0.6731 |
| C1 | 0.7778 | 0.7581 | 0.6961 | 0.7695 | 0.7761 | 0.6699 |

M1 将 Kitten recall 提高 `0.0667`，Adult recall 提高 `0.0108`，同时 Senior recall 回落
`0.0490`。因此平均 Balanced Accuracy 获得 `+0.0095`，Macro F1 仅获得 `+0.0037`：较大的
Kitten 收益与 Senior 回落相互抵消。QWK 几乎保持原水平，说明年龄顺序上的严重错分代价总体
稳定。

## 5. 稳定性与计算成本

| Pipeline | Macro F1 repeat SD | 平均选中 epoch | 平均 inner 时间/fit | 平均 outer 时间/fit | 峰值 VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| R0 | **0.0086** | 10.17 | 2.24 s | 1.37 s | 20.2 MB |
| M1 | 0.0371 | 7.58 | 17.70 s | 9.25 s | 600.7 MB |
| C1 | 0.0218 | 9.00 | 29.15 s | 17.35 s | 1,279.3 MB |

M1 的平均分数略高于 R0，同时 repeat SD 从 `0.0086` 增加到 `0.0371`，表明约 719 万参数的
更新增强了 split 敏感性。C1 与 M1 参数量相同，但梯度需要从输出穿过后续 11 个冻结 blocks
回传到 block 0，因此训练时间和峰值显存均更高；这个差异来自计算图位置。

## 6. 预设 gate 与阶段结论

seed expansion gate 要求 M1 相对 R0 平均 Macro F1 至少 `+0.005`。本轮实际为 `+0.003657`，
距离阈值 `0.001343`。其余七项检查全部通过：正向 repeat 数达到 `2/3`，M1 平均高于 C1 且
`3/3` repeats 为正，Balanced Accuracy、QWK、Animal CE 和逐类别 recall 的变化均处在预设
容许范围内。

本轮形成三条可直接用于论文的结论：

1. 受约束顶部单 block 适配取得与 frozen AST 相当且均值略高的 complete-OOF Macro F1，
   Balanced Accuracy 提高约 `0.0095`；
2. M1 在三个 repeats 中持续高于同参数量底部 block 对照，平均 Macro F1 优势为 `0.0222`，
   支持 AST 更新位置具有实际影响；
3. M1 的相对 R0 增益呈 `两正一负`，repeat SD 增至 `0.0371`，当前主要限制来自稳定性以及
   Kitten 收益与 Senior 回落之间的权衡。

按预设规则，IDEA-058 作为 constrained top-block 与 update-location ablation 阶段性收尾，
本轮省略 seeds 43/101 扩展。`top1_lr1e-5` 保留为有支持信号的适配候选；未来出现新的稳定化
方案或独立实验计划时，可以从该 recipe 继续迭代。

## 7. 执行过程中的顺序修正

第一次 outer evaluation 草稿把 `train + validation` 合并索引排序后再交给确定性 batch
permutation。样本集合保持相同，但位置变化使同一个 permutation 对应到不同 call，R0 因此得到
`0.7419`，偏离锁定参考 `0.7570`。该锚点差异在结果接收前触发审查。

修正将合并顺序恢复为历史 runner 的“train 索引后接 validation 索引”，并加入回归测试。旧草稿
整体移动到
`runs/meowagenet_idea058_constrained_top_block_v1/audit/superseded_sorted_outer_order_2026-09-15/`，
随后在提交 `517169b` 上重新签发 execution lock 并重跑全部 36 个 fit。

正式 R0 的五项聚合指标与锁定参考逐位一致；333 条 cat-level OOF 标签也全部一致。与 IDEA-057
当时重算 global embedding 的预测概率相比，最大绝对差为 `2.45×10⁻⁴`，且预测标签保持一致。
正式报告与机器可读记录只引用修正后的 evaluation。

## 8. 关键文件

- 计划：`plan/IDEA-058_constrained_top_block_adaptation.md`；
- Protocol：`configs/protocol/meowagenet_idea058_constrained_top_block_v1.json`；
- Runner：`scripts/run_meowagenet_idea058_top_block_adaptation.py`；
- 机器可读结果：
  `metadata/experiments/meowagenet_idea058_constrained_top_block_v1_results.json`；
- 候选选择记录：
  `runs/meowagenet_idea058_constrained_top_block_v1/selection/selection_record.json`；
- 执行锁：`runs/meowagenet_idea058_constrained_top_block_v1/execution_lock.json`；
- 原始预测文件：`91` 个，总计 `1,429,779` bytes；
- 原始预测清单聚合 SHA-256：
  `98261cea320ca4275a12ae760403d80d4ea9b35e16b3e48cd4bd77c72b9dc203`。
