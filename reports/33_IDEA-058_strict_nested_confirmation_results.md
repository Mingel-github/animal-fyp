# IDEA-058｜严格 nested 复核结果

> 执行日期：2026-09-15
> 状态：阶段 A 完成；严格复核形成正式结论
> 锁定代码提交：`9fe2b02676fa4e01b3be979c588a923dc25fd05f`

## 1. 本轮复核回答什么

原 IDEA-058 在全部 `3 repeats × 4 folds` 的 inner-validation 结果上选出一个全局 recipe，随后将
该 recipe 用于所有 outer folds。同一只猫会在一个 fold 中担任测试样本、在其他 folds 中进入训练
或验证，因此全局汇总让该猫能够间接影响测试它时采用的 recipe。

本轮将选择边界收紧到每个 `repeat × outer fold`：每折只使用本折的 train/validation 猫比较
四个既有候选，锁定本折 recipe 后再训练 R0、M1 和 C1，并在最后一次性读取本折测试猫。这个
修正保持候选范围、数据、split、seed、分类头、权重、checkpoint rule 和聚合方法不变，集中检验
selection boundary 对 IDEA-058 结论的影响。

## 2. 数据与执行规模

- 数据：111 只猫、792 条 call、843 个 segment；
- 评价单位：animal；每只猫在每个 repeat 的 complete-OOF 中出现一次；
- 候选选择：4 个 recipe × 3 repeats × 4 folds，共 48 个 inner-only fit；
- 正式评价：3 条 pipeline × 3 repeats × 4 folds，共 36 个 outer fit；
- 最终评价：每条 pipeline 形成 3 组 111-cat complete-OOF，共 9 组结果；
- 失败运行：0；所有 selection、smoke、evaluation 产物状态均为 `complete`。

四个候选保持原定义：顶部 1 或 2 个 AST Transformer blocks，encoder learning rate 为
`3×10⁻⁶` 或 `1×10⁻⁵`。M1 更新顶部 block，C1 更新相同数量的底部 block；两者同时更新 final
LayerNorm 和同一个 `768 → 128 → 3` 分类头。

## 3. 每折 inner-only recipe 锁

| Repeat | Outer fold | 选中 recipe | Blocks | Encoder LR | Inner Macro F1 | Inner animal CE |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0 | 0 | `top1_lr1e-5` | 1 | `1×10⁻⁵` | 0.7474 | 0.8721 |
| 0 | 1 | `top1_lr1e-5` | 1 | `1×10⁻⁵` | 0.8491 | 0.3496 |
| 0 | 2 | `top1_lr3e-6` | 1 | `3×10⁻⁶` | 0.8864 | 0.2909 |
| 0 | 3 | `top2_lr3e-6` | 2 | `3×10⁻⁶` | 0.7000 | 0.9219 |
| 1 | 0 | `top1_lr1e-5` | 1 | `1×10⁻⁵` | 0.7608 | 0.8455 |
| 1 | 1 | `top1_lr1e-5` | 1 | `1×10⁻⁵` | 0.6393 | 0.7213 |
| 1 | 2 | `top1_lr3e-6` | 1 | `3×10⁻⁶` | 0.7608 | 0.6861 |
| 1 | 3 | `top2_lr3e-6` | 2 | `3×10⁻⁶` | 0.8000 | 0.7999 |
| 2 | 0 | `top1_lr3e-6` | 1 | `3×10⁻⁶` | 0.6547 | 0.7315 |
| 2 | 1 | `top2_lr1e-5` | 2 | `1×10⁻⁵` | 0.8643 | 0.8127 |
| 2 | 2 | `top2_lr3e-6` | 2 | `3×10⁻⁶` | 0.8148 | 0.7076 |
| 2 | 3 | `top2_lr1e-5` | 2 | `1×10⁻⁵` | 0.6801 | 0.8689 |

12 个 locks 覆盖 12 个唯一 repeat/fold，`outer_test_accessed=false`。选择次数分别为：
`top1_lr1e-5` 4 折、`top1_lr3e-6` 3 折、`top2_lr3e-6` 3 折、`top2_lr1e-5` 2 折。
这说明不同训练侧子集实际支持不同容量和学习率，也说明单个全局 recipe 会抹平这种折间差异。

## 4. Smoke 与执行审计

- Smoke 使用 repeat 0 / fold 0 锁定的 `top1_lr1e-5`；
- M1/C1 的预训练 AST 初态、分类头初态、post-build RNG 和可训练参数量相同；
- M1/C1 初始 logits 最大绝对差为 `0`；
- 三条 pipeline 的 checkpoint 重载后概率最大绝对差均为 `0`；
- 12 个 fold groups 的共同 epoch batch-order hashes 全部匹配；
- execution lock 在 outer evaluation 前固定代码、环境、数据和 12 个 per-fold recipes；
- R0 的聚合结果与历史锁定 reference 逐位一致，Macro F1 均为
  `0.7570206188850608`。

## 5. Complete-OOF 主结果

| Pipeline | Animal Macro F1 | Balanced accuracy | QWK | Accuracy | Animal CE |
| --- | ---: | ---: | ---: | ---: | ---: |
| R0 tuned frozen AST | **0.7570 ± 0.0086** | **0.7645** | **0.6721** | **0.7568** | 0.7160 |
| M1 strict top-block adaptation | 0.7493 ± 0.0211 | 0.7564 | 0.6629 | 0.7508 | **0.6924** |
| C1 strict bottom-block control | 0.7337 ± 0.0139 | 0.7433 | 0.6391 | 0.7357 | 0.7204 |

M1 相对 R0 的平均变化为：Macro F1 `−0.0077`、Balanced Accuracy `−0.0080`、QWK
`−0.0092`、Accuracy `−0.0060`，Animal CE 改善 `0.0235`。因此，M1 给真实类别分配的平均
概率更合理，离散 argmax 分类在三个 repeats 的均值上略低于 R0。这里的 CE 与 F1 共同说明：
概率校准方向有收益，同时少数接近分类边界的猫改变了最终类别，拉低了按类别计算的分数。

M1 相对 C1 的平均 Macro F1 优势为 `+0.0156`，Balanced Accuracy 优势为 `+0.0131`，QWK
优势为 `+0.0238`。严格边界下仍保留了“顶部更新的位置优于底部更新”的平均信号。

### 5.1 Repeat-level 配对结果

| Repeat | R0 Macro F1 | M1 Macro F1 | C1 Macro F1 | M1−R0 | M1−C1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7659 | 0.7510 | 0.7202 | −0.0149 | +0.0308 |
| 1 | 0.7565 | 0.7696 | 0.7480 | +0.0131 | +0.0215 |
| 2 | 0.7487 | 0.7275 | 0.7330 | −0.0213 | −0.0055 |
| **平均** | **0.7570** | **0.7493** | **0.7337** | **−0.0077** | **+0.0156** |

M1 在 `1/3` repeats 中高于 R0，在 `2/3` repeats 中高于 C1。按猫配对的 cluster bootstrap
给出 M1−R0 Macro F1 95% 区间 `[-0.0244, +0.0077]`，M1−C1 区间
`[-0.0074, +0.0405]`。三次 repeat 中，M1 相对 R0 平均改变 3.33 只猫的判定，平均新增
1.33 只正确猫、损失 2.00 只正确猫；净变化约为每个 repeat 少 0.67 只正确猫。

### 5.2 类别表现

| Pipeline | Kitten recall | Adult recall | Senior recall | Kitten F1 | Adult F1 | Senior F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R0 | 0.8000 | 0.7581 | **0.7353** | **0.7914** | **0.7853** | **0.6943** |
| M1 | **0.8000** | **0.7634** | 0.7059 | 0.7833 | 0.7821 | 0.6826 |
| C1 | 0.7778 | 0.7366 | 0.7157 | 0.7620 | 0.7673 | 0.6719 |

M1 保持 Kitten recall，Adult recall 提高 `0.0054`，Senior recall 回落 `0.0294`。Adult recall
的收益与 Senior recall 的回落共同形成 Macro F1 和 Balanced Accuracy 的小幅净下降。M1 的 QWK
仍高于 C1，说明顶部更新在年龄顺序相关的错误代价上也优于底部更新；R0 继续取得本轮最高 QWK。

## 6. 与探索性 IDEA-058 的并列比较

| Pipeline | 探索性 Macro F1 | Strict Macro F1 | Strict−探索性 |
| --- | ---: | ---: | ---: |
| R0 tuned frozen AST | 0.7570 | 0.7570 | 0.0000 |
| M1 top-block adaptation | **0.7607** | 0.7493 | −0.0114 |
| C1 bottom-block control | 0.7385 | 0.7337 | −0.0048 |

R0 完全一致，为数据、split、baseline 实现和聚合方法提供了锚点。M1 的变化集中反映 recipe
选择边界从“跨 12 个 folds 的全局汇总”改为“当前 outer fold 内独立选择”。原 `0.7607` 继续
作为探索性结果记录；严格估计 `0.7493` 用于本阶段的正式决策。

## 7. 参数量、epoch 与计算成本

| Pipeline | 可训练参数 | 平均选中 epoch | Epoch 范围 | 平均 inner 时间/fit | 平均 outer 时间/fit | 峰值 VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R0 | 99,075 | 10.17 | 1–19 | 2.31 s | 1.43 s | 19.3 MiB |
| M1 | 7,188,483–14,276,355 | 8.83 | 1–20 | 20.07 s | 11.27 s | 736.4 MiB |
| C1 | 7,188,483–14,276,355 | 10.25 | 1–25 | 31.89 s | 19.93 s | 1,334.7 MiB |

M1/C1 的参数范围来自每折选择 1 或 2 个 blocks。相同 recipe 下两者参数量相同；C1 位于网络
底部，反向传播需要穿过后续冻结 blocks，因此时间与显存均高于 M1。正式 outer 训练阶段约用
17.4 分钟，随后完成 complete-OOF 聚合与 5,000 次 cat-cluster bootstrap。

## 8. Gate 与阶段结论

严格 seed-expansion gate 的八项检查中，六项通过：M1 对 C1 的平均优势、`2/3` positive
repeats，以及 Balanced Accuracy、QWK、animal CE、逐类别 recall 的容许代价均满足要求。
两项主条件为 M1 相对 R0 平均至少 `+0.005`，并在至少 `2/3` repeats 中为正；本轮分别为
`−0.0077` 和 `1/3`，因此 gate 状态为 `passed=false`。

本轮形成三条论文层面的结论：

1. 严格 nested 选择给出 M1 Macro F1 `0.7493 ± 0.0211`，R0 为
   `0.7570 ± 0.0086`；当前 top-block 方法保持接近 baseline 的表现，严格复核支持 R0 作为
   更稳定的主模型；
2. M1 相对同参数量 C1 仍有平均 `+0.0156` Macro F1、`+0.0238` QWK，并在 `2/3`
   repeats 中领先，继续支持“AST 更新位置具有实际影响”这一结构性观察；
3. M1 取得更低的 animal CE（`0.6924` 对 `0.7160`），同时 Senior recall 回落 `0.0294`；
   顶层适配改善概率质量的信号值得记录，当前 argmax 分类收益尚未形成稳定优势。

按照预设路线，IDEA-058 以“探索性弱正、严格复核未确认”阶段性收尾，Stage B top-block
stabilization 暂停。下一条独立性能路线为 Stage C：IDEA-039 grouped augmentation policy；
Stage D nuisance-variable robustness 保留为随后可并行解释现有 OOF 预测的低成本诊断。

## 9. 关键文件

- 阶段计划：`plan/POST_IDEA058_next_stage_plan.md`；
- Strict protocol：`configs/protocol/meowagenet_idea058_strict_nested_v1.json`；
- Runner：`scripts/run_meowagenet_idea058_strict_nested.py`；
- 每折 recipe locks：
  `runs/meowagenet_idea058_strict_nested_v1/selection/per_fold_recipe_locks.json`；
- Execution lock：`runs/meowagenet_idea058_strict_nested_v1/execution_lock.json`；
- 机器可读结果：
  `metadata/experiments/meowagenet_idea058_strict_nested_v1_results.json`；
- 原始预测：91 个文件、1,430,727 bytes；
- 原始预测清单聚合 SHA-256：
  `48f60d63bb06e459fbd69c4d432f0051833bb5301be21f6fccace4b55426be8f`。
