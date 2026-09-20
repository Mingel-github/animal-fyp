# IDEA-079：AST 空间 patch-token adapter 结果

## 结论

预注册的 `S1_spatial_patch` 没有优于精确缓存尾部基线 `A0_cached_tail`，候选 gate 未通过；相对参数、初始化和计算量匹配的 `C1_pointwise_patch`，S1 虽有小幅平均优势，但稳定性条件未通过，因此空间机制 gate 也失败。`full_gate_passed=false`。

九个等权 `base_seed × repeat` 单元中，S1 相对 A0 的平均 Macro-F1 差为 `-0.00678`，只有 `3/9` 为正；三个 base seeds 中只有一个平均为正。S1 的 animal CE 和 Brier 均劣于 A0。S1 相对 C1 的平均 Macro-F1 差为 `+0.00320`，达到预注册均值阈值，但仅 `4/9` 个 seed×repeat 为正，不能支持稳定的二维局部邻接收益。

因此，IDEA-079 按预注册 fail action 终止。不得据此改 kernel、宽度、层位、gate 或种子，也不保留当前固定 S1 公式作为后续确认候选。

## 实验边界与完成状态

- 数据：MeowAgeNet 792 calls、111 cats、843 个 1.28 秒 segments；42 个 calls 含多个 segments。
- 角色：formal-v2 nested roles；只生成 train/validation 产物，`outer_test_accessed=false`。
- 管线：A0 缓存尾部基线、C1 逐点 patch control、S1 空间 patch candidate。
- 新 base seeds：`8058, 2495, 2473`；repeats 为 `0,1,2`，folds 为 `0,1,2,3`。
- 预算：`3 pipelines × 3 seeds × 3 repeats × 4 folds = 108 fits`；全部完成。
- S1/C1：都只适配 144 个 patch tokens，两个特殊 tokens 精确旁路；固定 width 9、14,691 个 adapter 参数。
- 总可训练参数：A0 `99,075`；C1/S1 各 `113,766`。AST block 12 始终冻结。
- 共享缓存：只读使用 IDEA-078 的 block-11 token cache，形状 `843×146×768`、float32 mmap，没有重复抽取 AST tokens。
- 执行环境：Python 3.10.12、PyTorch 2.2.2+cu121、CUDA 12.1、NVIDIA GeForce RTX 4060 Ti。

## 主要结果

以下指标先在每个 `base_seed × repeat` 单元合并四个 validation folds，再对九个单元等权平均。

| Pipeline | Macro-F1 | Balanced accuracy | Animal CE | Animal Brier |
|---|---:|---:|---:|---:|
| A0 cached tail | 0.74576 | 0.78241 | 0.67871 | 0.40304 |
| C1 pointwise patch | 0.73578 | 0.76759 | 0.68206 | 0.40336 |
| S1 spatial patch | 0.73898 | 0.77130 | 0.69116 | 0.41155 |

配对 Macro-F1：

| Comparison | Mean delta | SD | Positive / tied / negative | Worst | Best |
|---|---:|---:|---:|---:|---:|
| S1 − A0 | -0.00678 | 0.02602 | 3 / 1 / 5 | -0.06160 | +0.02525 |
| C1 − A0 | -0.00998 | 0.02418 | 2 / 2 / 5 | -0.05381 | +0.02022 |
| S1 − C1 | +0.00320 | 0.02843 | 4 / 1 / 4 | -0.04533 | +0.05381 |

S1 相对 A0 的 balanced accuracy 低 `0.01111`，animal CE 高 `0.01245`，Brier 高 `0.00852`。这不是单一离散阈值造成的 Macro-F1 波动：分类平衡性与概率质量也同时变差。

## 候选 gate 审计：S1 对 A0

八项预注册条件中，只有每个 seed 的 senior recall 保护条件通过。

| 条件 | 预注册阈值 | 观察值 | 结果 |
|---|---:|---:|---|
| 平均 Macro-F1 增益 | ≥ +0.005 | -0.00678 | Fail |
| 正向 base seeds | ≥ 2/3 | 1/3 | Fail |
| 正向 seed×repeat | ≥ 6/9 | 3/9 | Fail |
| 非负 split-cells | ≥ 8/12 | 6/12 | Fail |
| 最差 split-cell | ≥ -0.03 | -0.09708 | Fail |
| Animal CE 不劣 | S1 ≤ A0 | 0.69116 > 0.67871 | Fail |
| Animal Brier 不劣 | S1 ≤ A0 | 0.41155 > 0.40304 | Fail |
| 每个 seed senior recall | ≥ -0.02 | 最差 -0.01667 | Pass |

三个 base-seed 的 S1−A0 Macro-F1 均值分别为：

- seed 8058：`+0.00806`
- seed 2495：`-0.01700`
- seed 2473：`-0.01141`

对应的 senior recall 差值为 `+0.01667、-0.01667、0.00000`。12 个共同 split-cells 为 3 正、3 平、6 负，范围 `-0.09708～+0.08840`。最差 cell 明显越过 `-0.03` 防线，说明局部退化风险较大。

## 空间机制 gate：S1 对 C1

| 条件 | 预注册阈值 | 观察值 | 结果 |
|---|---:|---:|---|
| 平均 S1−C1 Macro-F1 | ≥ +0.002 | +0.00320 | Pass |
| 正向 S1−C1 seed×repeat | ≥ 5/9 | 4/9 | Fail |

S1 相对 C1 的均值优势说明 3×3 深度卷积并非完全没有信号，但其 9 个独立单元呈 4 正、1 平、4 负，正负对称且波动较大。由于预注册机制 gate 要求两项同时通过，本轮不能作出“二维局部邻接优于逐点混合”的机制结论。

## 缓存、初始化与训练审计

- 缓存尾部重建相对锁定 A0 embedding 的平均绝对差为 `4.02e-7`、最大绝对差为 `8.46e-6`；正式前运行时复核最大差为 `9.89e-6`。
- C1、S1 的 `up` 层为零初始化；正式前使用真实缓存检查时，两者相对 A0 的初始 logit 最大差均为 `0`。C1/S1 共用 head 初态，并具有匹配的 adapter 随机张量。
- 108 个最优 checkpoint 全部重载一致，最大概率差为 `0`；没有 NaN、OOM 或参数冻结违规。
- 36 个 A0/C1/S1 fits 的平均训练时间分别为 `11.02、13.05、12.09` 秒；最大记录显存分别约为 `466、758、758` MiB。

## 独立审计与可复现性

- 锁定协议 SHA-256：`690b38a7e258a44435398151ce1a114be7ca58a253e6428c85fd0e57c6f53421`。
- 锁定 runner SHA-256：`0b3623c7a0ab7021e36659bdacfd5d8aed351a394dc6c8a6035e48242a7d1aa0`。
- 原正式命令使用 `--resume` 再运行时在 7.4 秒内完成验证，仍报告 108/108，没有重训。
- 独立审计检查 108 个 fit summaries、216 个预测文件、12,015 条 call 预测和 1,836 条 animal 预测。
- 从 call 预测重新聚合到 animal 的最大数值差为 `1.11e-16`。
- 独立重算的 fold、seed×repeat、split-cell、每 seed senior recall、两组 gate 与锁定汇总完全一致。
- 正式 summary SHA-256：`f1453f7ff560fb1220bd4df56438fde1993f71bdbe33b6f3c7429637fd9efd10`。

完整正式产物位于 `runs/meowagenet_idea079_spatial_patch_adapter_v1/`，独立审计为其中的 `independent_audit.json`。

## 决策

IDEA-079 的候选 gate、空间机制 gate 和 full gate 均失败。停止当前固定的 block-11→12、width-9、3×3 depthwise spatial patch-token adapter，不进行任何结果驱动的 kernel、宽度、层位或种子修补。A0 cached-tail 仍是本轮三种内部表示路径中最稳健的参照。
