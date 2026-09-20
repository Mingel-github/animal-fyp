# IDEA-087：A0/U1/C1 公平预算嵌套训练参数优化

日期：2026-09-19  
状态：用户已授权设计与 CPU 实现；GPU 尚未授权。

## 1. 目标与边界

在现有 111 只猫上，以严格 outer-test 隔离的嵌套流程分别优化 A0、U1、C1 的共同训练参数。A0 是冻结 AST 的 768→128→3 分类头；U1 是 20→60→128 的自由加法声学残差；C1 使用相同声学分支和固定 `0.25 × stopgrad(RMS(h)) × tanh` 上限。

本轮只搜索 learning rate、dropout 和 Adamax weight decay，不搜索声学宽度、C1 cap、特征、损失、checkpoint 规则或模型结构。它是在反复使用过的同一 111-cat 数据上的内部嵌套优化评估，不是新动物确认，也不改变 IDEA-076、083、085、086 或 v3 的既有结论。

## 2. 数据角色

复用 `splits/meowagenet_formal_v2_nested_roles.csv` 的 repeat 0、四个 outer folds。每个 outer fold 的原 train 与 validation 合并为 outer-train；原 test 是唯一 outer-test。outer-test 在全部 inner fits 与 selection lock 冻结前不可进入预测、选择、早停、epoch 决策或补跑逻辑。

每个 outer-train 内，以猫为单位、按年龄标签分层、固定 seed 新建 3 个 inner folds。同猫的全部 calls 永远同角色。每个 inner cell 必须满足：train/validation 猫与 call 均不相交、两侧三类齐全、合并恰好覆盖 outer-train。每个 search seed 的三份 inner validation 预测必须组成 outer-train 的完整且无重复猫级 OOF。

repeat 0 的角色规模为：outer-train `83/83/83/84` cats，outer-test `28/28/28/27` cats。outer-train 类别计数分别为 `46/11/26`、`46/11/26`、`47/11/25`、`47/12/25`（adult/kitten/senior），因此 3 折分层在四个 outer folds 中均可行。

## 3. 固定搜索空间与预算

三种模型使用完全相同的 8 行笛卡尔网格；循环顺序为 learning rate → dropout → weight decay：

- learning rate：`[0.006, 0.003]`
- dropout：`[0.44571035356880917, 0.25]`
- Adamax weight decay：`[0, 0.001]`

`q00` 为原训练配置。分支学习率倍率固定 1，U1/C1 声学宽度固定 60，C1 cap 固定 0.25。不得在看到 inner 或 outer 结果后增加网格、种子或阈值。

- Inner：`8 configs × 3 models × 4 outer folds × 3 inner folds × 2 search seeds = 576 fits`。
- Outer：`selected/original × 3 models × 4 outer folds × 3 refit seeds = 72 fits` 上限。
- 总上限：648 fits。若 selected 恰为 q00，同一 pipeline/outer fold/refit seed 只训练一次，并把 selected 记录为指向 original fit 的别名；不得把别名计作独立 fit。

IDEA-076 同类 GPU fits 的历史均值/中位/P90 为约 `2.07/1.91/3.20` 秒，648 fits 的纯训练粗估约 0.37 GPU 小时；即使计入预测、I/O 与审计，预算仍可行。

## 4. 训练与公平性

保持原分类头、猫 batch 4、Adamax `eps=1e-7`、gradient clip 1、最多 50 epochs、patience 8、最低未加权 inner-validation animal CE checkpoint。原“globally class-balanced call CE”的准确实现为：只用当前 fit training calls 的标签计数，并令每类 call 权重为 `n_train_calls / (3 × n_train_calls_in_class)`；不是全数据权重，也不是猫级权重。

AST 标准化、声学缺失值 median/mean/std 与类别权重均只拟合当前 fit training calls。相同 stage/cell/seed 下，A0/U1/C1 与八个配置共享相同主头初态和 post-build RNG reset；同一 epoch 的猫顺序与 call coverage 必须相同。AST backbone 始终冻结。

## 5. 选择与 refit epoch

每个 outer fold、pipeline、config、search seed 先拼接 3 个 inner-validation fold，得到完整 outer-train 猫级 OOF；再对两个 search seeds 的指标等权。

1. 找到最高 mean Macro-F1。
2. 候选池包含 `mean Macro-F1 ≥ best − 0.002` 的配置。
3. 池内按 mean Brier 升序、Macro-F1 sample SD 升序、mean Accuracy 降序、config ID 升序唯一选择。

每个配置的 outer refit epoch 由其 `2 seeds × 3 inner folds = 6` 个 1-based best epochs 取 median，并以 half-up 规则取整，限定到 `1..50`。q00 与 selected 分别使用自己的六个 epoch；outer refit 在完整 outer-train 上训练固定 epoch，不再早停或查看 outer 标签决定轮数。

## 6. 分阶段门禁与产物

1. CPU preflight：生成并核对 inner roles、参数、初始化、训练专属变换、class weights、选择规则、half-up、resume 与 source hashes。
2. 总监仅授权 1 个 inner fit 作工程检查；通过后授权其余 575 个 inner fits。
3. 全部 inner fits 完成后一次性写入 `selection/selection_lock.json`，其中包含 12 个 pipeline×outer 选择、q00 与 selected refit epochs、所有候选分数和逐 fit hashes。任何一个 inner fit 缺失都不得生成 lock。
4. 独立审计 selection；outer 入口同时要求总监授权与显式匹配的 selection-lock SHA-256。
5. 最多 72 个 outer refits 完成后，从原始 call/animal predictions 独立复算并报告结果。

resume 必须强校验 protocol、runner、tests、source、stage、pipeline、config、seed、outer/inner role identity、训练角色 hashes、预测 hashes 与 selection SHA；不得静默接受部分或错位结果。

## 7. 最终报告范围

分别报告 A0、U1、C1 的 tuned−fixed（q00）以及 tuned U1/C1−tuned A0；指标包括 Accuracy、Macro-F1、BA、kitten/adult/senior recall、CE、Brier、三 refit seeds 与 12 个 seed×outer-fold 配对稳定性。各 outer fold 的 selected config 与 refit epoch 必须完整列出。

本轮不设新的全局 gate，也不从 outer 分数事后挑选一个全局赢家配置。若未来需要完整 111-cat 部署参数，必须另行授权并在看相应结果前锁定同类 inner-only 选择流程。
