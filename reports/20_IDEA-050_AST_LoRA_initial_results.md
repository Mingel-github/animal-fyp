# IDEA-050｜AST LoRA 初始嵌套配对实验结果

> 完成日期：2026-09-02
> 实验状态：`complete_for_initial_nested_paired_evaluation`
> 候选状态：`screened_below_matched_head_only`
> 评价单位：animal-level complete OOF，111 只猫
> 阶段决策：当前五候选 Q/V LoRA 完成阶段性收尾，保留为论文 ablation，暂缓 seeds 43/101 扩展

## 1. 本轮回答的问题

IDEA-050 检查低秩适配 LoRA 能否在当前最强的 tuned AST head-only 上继续增加年龄分类能力。
两条 pipeline 使用相同的 MIT AST AudioSet checkpoint、同一套 792 条叫声、111 个 cat-ID、
三个 repeat、四个 outer folds、classifier head 和训练规则。唯一核心差别是 LoRA pipeline
在 AST self-attention 的 query/value projection 上增加低秩可训练参数。

本轮共完成：

- implementation audit 1 次；
- repeat 0 / fold 0 / seed 17 inner-only smoke 1 次；
- 5 个 LoRA 候选 × 3 repeats × 4 folds = 60 个 inner-only candidate fits；
- 在 outer prediction 前写入 12 份逐折 selection lock；
- AST head-only 与 AST LoRA 各 12 个 outer fits，共 24 个；
- 两条 pipeline 各 3 套、合计 6 套覆盖 111 只猫的 complete OOF。

这 84 个正式训练 fit 完整实现了计划中的 Stage A–D。Formal-v2.1 的协议、代码和结果保持
原状；本轮定位为 post-formal exploratory PEFT study。

## 2. LoRA 配置与实现审计

LoRA 把冻结 attention 权重的任务更新表示为两个小矩阵的乘积：

\[
W' = W + \frac{\alpha}{r}BA
\]

原始 AST 权重 \(W\) 保持冻结，训练更新集中在 \(A\)、\(B\) 和 `768 → 128 → 3`
classifier head。`A` 使用 Kaiming 初始化，`B` 从 0 开始，因此首次前向时低秩增量为 0。

审计以 L1 为例，将 LoRA 注入第 9–12 个 transformer blocks 的 query/value，共 8 个
projection。审计结果如下：

| 审计项目 | 结果 |
|---|---:|
| AST 总参数 | 85,515,267 |
| L1 LoRA 参数 | 49,152 |
| L1 总可训练参数，含 head | 148,227 |
| 非目标 backbone 可训练参数 | 0 |
| 零增量 LoRA 与 frozen AST 最大 logit 差 | 0.0 |
| Smoke 后全部 LoRA 与 head 参数发生更新 | 通过 |
| Trainable-only checkpoint | 609,368 bytes |
| Checkpoint 重载最大 logit 差 | 0.0 |
| Smoke 峰值显存 | 761,573,376 bytes |

这组结果确认 LoRA 注入位置、冻结范围、梯度更新和轻量 checkpoint 链路均按设计工作。

## 3. 五个候选与逐折选择

共享 head 使用 dropout `0.4457103536`、Adamax 和 head learning rate `0.006`。所有 LoRA
候选固定 Q/V projections、LoRA dropout `0.05` 与 `alpha = 2 × rank`：

| Candidate | Blocks | Rank | LoRA LR | LoRA 参数 | 总可训练参数 |
|---|---|---:|---:|---:|---:|
| L1 | last 4 | 4 | 3e-4 | 49,152 | 148,227 |
| L2 | last 4 | 8 | 3e-4 | 98,304 | 197,379 |
| L3 | all 12 | 4 | 3e-4 | 147,456 | 246,531 |
| L4 | last 4 | 4 | 1e-4 | 49,152 | 148,227 |
| L5 | last 4 | 4 | 1e-3 | 49,152 | 148,227 |

每个 candidate 先按最低 inner-validation cross-entropy loss 选择 epoch，再比较该 epoch 的
animal macro F1；并依次使用 balanced accuracy、QWK、validation loss 和 candidate ID
处理同分。各 fold 的锁定结果为：

| Repeat | Fold | Candidate | Epoch | Inner macro F1 | BA | QWK |
|---:|---:|---|---:|---:|---:|---:|
| 0 | 0 | L5 | 6 | 0.7474 | 0.8000 | 0.5103 |
| 0 | 1 | L2 | 9 | 0.8491 | 0.9000 | 0.6502 |
| 0 | 2 | L4 | 2 | 0.7270 | 0.8333 | 0.7038 |
| 0 | 3 | L1 | 5 | 0.7222 | 0.7333 | 0.5785 |
| 1 | 0 | L3 | 1 | 0.6105 | 0.6667 | 0.5103 |
| 1 | 1 | L4 | 3 | 0.7251 | 0.8000 | 0.5333 |
| 1 | 2 | L1 | 6 | 0.8148 | 0.8333 | 0.7190 |
| 1 | 3 | L4 | 6 | 0.8000 | 0.8000 | 0.6909 |
| 2 | 0 | L2 | 4 | 0.8148 | 0.8333 | 0.7190 |
| 2 | 1 | L3 | 3 | 0.8643 | 0.9000 | 0.7984 |
| 2 | 2 | L2 | 6 | 0.8148 | 0.8333 | 0.7190 |
| 2 | 3 | L5 | 2 | 0.7222 | 0.7333 | 0.5785 |

选择频次为 L1=2、L2=3、L3=2、L4=3、L5=2。五个候选均被至少两个 folds 选中，L2 与
L4 各领先一次。该分布表明不同 cat-ID development folds 偏好不同的 rank、层范围或学习率，
当前候选空间中尚未形成单一稳定配方。

## 4. Animal-level complete-OOF 主结果

| Repeat | Head-only macro F1 | LoRA macro F1 | LoRA − head | Head BA | LoRA BA | Head QWK | LoRA QWK |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7816 | 0.7773 | -0.0043 | 0.7965 | 0.7955 | 0.6797 | **0.7044** |
| 1 | 0.7373 | 0.6908 | -0.0465 | 0.7563 | 0.6976 | 0.5999 | 0.5508 |
| 2 | 0.7275 | 0.6842 | -0.0433 | 0.7403 | 0.6815 | 0.6612 | 0.5797 |
| Mean | **0.7488** | **0.7174** | **-0.0313** | **0.7644** | **0.7249** | **0.6469** | **0.6116** |

Head-only 三个 repeat 的 macro F1 sample SD 为 0.0288；LoRA 为 0.0519。repeat 0 中两者
macro F1 相差 0.0043，处于非常接近的水平，同时 LoRA QWK 提高 0.0247。repeat 1 和 2
的 LoRA macro F1 分别低 0.0465 与 0.0433。三组配对共同形成平均差 -0.0313，且 LoRA
的 repeat 间波动更大。

10,000 次 repeat-and-class-stratified animal paired bootstrap 得到：

- observed mean difference：-0.03135；
- bootstrap mean：-0.03149；
- 95% percentile interval：[-0.06785, 0.00212]；
- `P(LoRA − head-only > 0)`：0.0358。

约 96.4% 的 bootstrap 重采样支持 matched head-only 的 macro F1 更高。区间上界接近 0，
说明 repeat 0 体现了两者可接近的情形；整体方向由 repeat 1、2 的一致差距决定。

## 5. 各年龄类别发生了什么

三次 repeat 的平均 recall 与 F1 如下：

| Pipeline | Kitten recall | Adult recall | Senior recall | Kitten F1 | Adult F1 | Senior F1 |
|---|---:|---:|---:|---:|---:|---:|
| AST head-only | 0.8222 | 0.7258 | 0.7451 | 0.7948 | 0.7735 | 0.6780 |
| AST LoRA | 0.7778 | 0.7204 | 0.6765 | 0.7690 | 0.7492 | 0.6342 |
| LoRA − head | -0.0444 | -0.0054 | -0.0686 | -0.0258 | -0.0243 | -0.0439 |

将三个 repeat 的混淆矩阵相加，行是真实年龄，列是预测年龄：

| Head-only true \ pred | Kitten | Adult | Senior |
|---|---:|---:|---:|
| Kitten | 37 | 5 | 3 |
| Adult | 8 | 135 | 43 |
| Senior | 3 | 23 | 76 |

| LoRA true \ pred | Kitten | Adult | Senior |
|---|---:|---:|---:|
| Kitten | 35 | 8 | 2 |
| Adult | 8 | 134 | 44 |
| Senior | 3 | 30 | 69 |

Adult recall 基本持平，主要差距来自 kitten 与 senior，尤其是 7 个额外 senior 被分到相邻的
adult 类别。LoRA 的 kitten↔senior 跨两级错误总数为 5，head-only 为 6；这说明 LoRA
保持了一部分年龄顺序结构，同时在 senior/adult 相邻边界上损失了更多区分能力。repeat 0 的
QWK 优势与较少远距离错误一致，后两个 repeat 的整体正确排序与类别识别仍由 head-only 领先。

## 6. 参数与运行资源

LoRA 各折包含 49,152–147,456 个低秩参数，加上 head 后共有 148,227–246,531 个可训练
参数，只占 85.5M AST 模型约 0.17%–0.29%。因此，本轮实现达到了低参数适配目标。

12 个 LoRA outer fits 的平均 train-and-predict 时间为 6.80 秒，峰值显存为
1,160,909,824 bytes。Head-only 使用已经缓存的 frozen AST embedding，平均 outer fit 为
0.60 秒、峰值显存 20,236,800 bytes；两组资源数字反映“在线 AST 前向与低秩更新”和“缓存
embedding 上训练 head”两种实际计算路径。LoRA 的轻量训练状态约 0.61 MB，模型保存成本很低。

## 7. 本轮结论与阶段决策

1. Q/V LoRA 工程链路完整可用。低秩参数正确注入，backbone 冻结范围正确，初始化等价、
   梯度更新和 checkpoint 重载全部通过。
2. 当前五候选 LoRA pipeline 的平均 animal macro F1 为 0.7174；matched tuned AST
   head-only 为 0.7488。Head-only 在三次 repeat 均领先，平均优势为 0.0313。
3. LoRA 在 repeat 0 达到接近 head-only 的 macro F1，并取得更高 QWK，说明低秩 attention
   adaptation 能形成有价值的顺序性变化；该变化在 repeat 1、2 上转化为更低的 senior
   recall 和更大的 split sensitivity。
4. 当前 seed-expansion signal 三项均未满足：mean paired macro F1 为负、正向 repeat 为
   0/3、BA 与 QWK 的均值同时下降。本阶段据此暂缓 seeds 43/101，节省 48 个新增 outer fits。
5. 论文主性能候选继续采用 tuned AST head-only。IDEA-050 作为完整 PEFT ablation 进入论文，
   支持“少量 Q/V 低秩更新在当前 111-cat 数据上增加了适配容量，同时未增加稳定泛化收益”的
   实验结论。未来仍可基于明确机制尝试 K、attention output、MLP LoRA 或新的正则化方案。

作为历史上下文，IDEA-049 中相同 base seed 与 repeats 的 VGGish mean macro F1 为 0.6795。
本轮 LoRA 的 0.7174 仍高约 0.0379，说明 AST representation 的总体优势继续存在；本次阶段结论
聚焦于 LoRA 相对更强 matched AST head-only 的附加价值。

## 8. 代码、结果与审计材料

- 协议：`configs/protocol/meowagenet_idea050_ast_lora_v1.json`；
- 独立 runner：`scripts/run_meowagenet_idea050_ast_lora.py`；
- 机器可读结果：
  `metadata/experiments/meowagenet_idea050_ast_lora_initial_v1_results.json`；
- 参数审计：`runs/meowagenet_idea050_ast_lora_v1/audit/audit_summary.json`；
- inner-only smoke：`runs/meowagenet_idea050_ast_lora_v1/smoke/smoke_summary.json`；
- 60 个 candidate summaries、12 份 locks 与总索引：
  `runs/meowagenet_idea050_ast_lora_v1/selection/`；
- 24 个 outer fit summaries 与完整指标：
  `runs/meowagenet_idea050_ast_lora_v1/evaluation/`；
- 静态与结果测试：`tests/test_idea050_lora_runner.py`。

仓库保存 104 个精简 JSON 文件，共 961,044 bytes。逐 call predictions、OOF CSV、模型
checkpoint、音频和本地环境继续留在实验机；JSON fit histories、参数审计、selection locks、
OOF aggregate 和 checksum 足以复查本轮流程、选择与结论。
