# IDEA-087：A0 / U1 / C1 嵌套参数优化独立设计与泄漏审查

日期：2026-09-19  
审查阶段：结果产生前、CPU 设计审查  
结论：**GO（仅进入实现与 CPU preflight；不构成 GPU 训练授权）**

## 1. 独立结论

IDEA-087 的最终方案可以在现有 111 只猫上形成一套统计上闭合、预算公平且不使用当前外层结果选参的 nested-HPO 流程。核心条件已经明确：

1. 每个 outer fold 的开发集是 repeat 0 中原 `train + validation` 的并集；该 fold 的 `test` 完全排除在配置选择、checkpoint、epoch 选择、标准化、缺失填补和类别权重估计之外。
2. 内层为固定、共享的猫级分层 3 折；两个 search seeds 只表示训练随机性，不改变内层划分。每个 seed 先把三折 held-out 预测拼成覆盖整个 outer-development set 的猫级 OOF，再计算一次指标；配置分数是两个 seed-level OOF 指标的等权平均。
3. 每个 outer fold、每种模型独立选择一个配置；所有 12 个选择以及 selected/original 的 refit epochs 必须先写入不可变 selection lock，之后才能进行任何 outer-test 预测或计分。
4. selected 与原配置 `q00` 分别用各自 6 个 inner best epochs 推导定长 outer refit epoch；outer refit 不早停，也不查看 outer 指标。
5. 三种模型使用同一 8 行训练参数网格、相同划分、search seeds、refit seeds 和 fit 数。这里成立的是**搜索机会和计算预算公平**，不是参数量相同：A0 为 99,075 个可训练参数，U1/C1 各为 108,143 个。

在这些条件下，内层 validation 同时用于 checkpoint 选择和配置评分是标准 nested CV 中的内层选择，不是 outer leakage。其代价是内层分数会有选择乐观偏差；该偏差只能由完全隔离的当前 outer fold 评估，而不能通过把六个 inner fits 当成六个独立样本来消除。

本轮仍只能称为**既有 111 猫上的内部嵌套优化评估**。这些动物及其历史 outer 结果已经影响过方法开发；nested HPO 能保护 IDEA-087 本轮的局部选择过程，但不能把该数据恢复成从未看过的外部验证集，也不能推翻 IDEA-076 的历史停止结论。

## 2. 角色表与少数类可行性

独立读取 `splits/meowagenet_formal_v2_nested_roles.csv` 的 repeat 0 后，四个 outer-test 折互斥，且合并后恰好覆盖 111 个不同 `cat_id`，每只猫出现一次。数据总计 792 calls，类别为 15 kitten、62 adult、34 senior。

| Outer fold | Development cats | Development calls | Kitten / Adult / Senior | Test cats | Test calls | Kitten / Adult / Senior |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 83 | 629 | 11 / 46 / 26 | 28 | 163 | 4 / 16 / 8 |
| 1 | 83 | 554 | 11 / 46 / 26 | 28 | 238 | 4 / 16 / 8 |
| 2 | 83 | 569 | 11 / 47 / 25 | 28 | 223 | 4 / 15 / 9 |
| 3 | 84 | 624 | 12 / 47 / 25 | 27 | 168 | 3 / 15 / 9 |

三折猫级分层在每个 outer-development set 中是可行的；理想情况下每个 held-out inner fold 只有 3–4 只 kitten。这个数量足以保证三类覆盖，却意味着 kitten recall 和 Macro-F1 很离散。由此得到三项边界：

- 必须在唯一猫表上分层，再让一只猫的所有 calls 跟随该猫；不能在 call 行上做普通分层。
- 每个 inner train/validation 均需在 preflight 中断言 cat 交集为空、三类齐全、开发猫完整覆盖且每只猫恰好 held out 一次。
- `0.002` 的 F1 候选带是预先规定的决策偏好，不应解释为统计等价界限；只有两个 search-seed OOF 值，F1 SD 也只是确定性 tie-break，不是方差推断。

## 3. 固定搜索空间与预算公平

三种模型必须使用完全相同的 8 行配置表，配置 ID 和行序在结果前固定：

| Config | Learning rate | Dropout | Adamax weight decay | 说明 |
|---|---:|---:|---:|---|
| q00 | 0.006 | 0.44571035356880917 | 0 | 原配置 |
| q01 | 0.006 | 0.44571035356880917 | 0.001 |  |
| q02 | 0.006 | 0.25 | 0 |  |
| q03 | 0.006 | 0.25 | 0.001 |  |
| q04 | 0.003 | 0.44571035356880917 | 0 |  |
| q05 | 0.003 | 0.44571035356880917 | 0.001 |  |
| q06 | 0.003 | 0.25 | 0 |  |
| q07 | 0.003 | 0.25 | 0.001 |  |

分支 learning-rate multiplier 固定为 1；U1/C1 宽度固定为 60，C1 cap 固定为 0.25。结构专属参数不进入本轮搜索，不能看完 outer 结果再追加网格、种子或局部修补。

预算核算正确：

```text
inner = 3 models × 4 outer folds × 8 configs × 2 search seeds × 3 inner folds
      = 576 fits

outer upper bound = 3 models × 4 outer folds × 2 arms × 3 refit seeds
                  = 72 fits

total upper bound = 648 fits
```

若某个 `model × outer_fold` 的 selected config 为 q00，tuned 与 fixed 是同一策略，三次 refit 可在完整身份校验后复用。实际 outer fits 为：

```text
72 - 3 × count(selected_config == q00 across 12 model-by-fold cells)
```

复用时必须把两条结果标为 identity/alias；不能把相同预测复制后当成两组随机证据。

## 4. 无歧义的内层选择规则

对每个 `outer_fold × model × config × search_seed`：

1. 在共享的 3 个 inner folds 上分别训练；fit 的 best checkpoint 由该 fit held-out inner fold 上最低的未加权 animal-level CE 选择，epoch 为 1-based 的 1–50。
2. 读取三个 best-checkpoint held-out 预测，按 `cat_id` 拼接成覆盖当前完整 outer-development set 的一次 OOF；每只猫必须恰好出现一次。
3. 在该 seed 的整套 OOF 上计算 Macro-F1、Brier、Accuracy，而不是先算三个 fold 指标再平均。
4. 对两个 search-seed OOF 指标等权平均。六个 fold fits 是生成两个配对 OOF 估计的组成部分，不是六个独立观察。

指标定义锁定为：

- Macro-F1：`labels=[0,1,2]`、`zero_division=0`；
- Brier：先对每只猫计算 `sum_class (p - onehot)^2`，再对猫取算术平均；
- F1 SD：两个 seed-level OOF Macro-F1 的 sample SD，`ddof=1`；
- Accuracy：两个 seed-level OOF ordinary Accuracy 的等权平均，只作最末级确定性 tie-break。

唯一选择键如下：

1. 令 `best_mean_f1` 为 8 配置中最高 mean Macro-F1；候选池为 `mean_f1 >= best_mean_f1 - 0.002 - 1e-12`，边界闭合，`1e-12` 只作浮点保护。
2. 池内依次按 mean Brier 升序、F1 sample SD 升序、mean ordinary Accuracy 降序、固定 config ID 升序选出唯一配置。

ordinary Accuracy 会受 adult 多数类影响，因此它不能被解释为少数类改善证据；但其位置在 F1 带、Brier 和 F1 稳定性之后，作为已锁定的末级确定性 tie-break 可以接受。正式结果仍必须单独报告 balanced accuracy 与三类 recall。

## 5. Early stopping 与 outer refit

每个配置在一个 outer fold 中有 6 个 inner best epochs（2 search seeds × 3 folds）。对任一将被 outer refit 的配置 `q`，固定：

```text
E(q, model, outer_fold)
  = clip(round_half_up(median(its six 1-based best epochs)), 1, 50)
```

因此：

- tuned arm 使用 selected config 自己的 6 个 epochs；
- fixed arm 使用 q00 自己的 6 个 epochs；
- 若 selected=q00，两者的配置、epoch 和 refit 完全相同；
- outer refit 在全部 outer-development cats 上从头训练恰好 `E` 个 epoch，不设 early stopping，不在训练过程中加载或计算 outer-test metric。

这里必须使用真正的 decimal half-up，而不能依赖 Python 的 bankers rounding。selection lock 至少包含全部候选的两个 seed-level OOF 指标、聚合分数、6 个 best epochs、选择键、selected/q00 refit epochs、输入预测哈希和配置/代码/划分哈希。

把 early stopping 的 inner-validation CE 与候选 Macro-F1 分开是合理的：CE 决定单个 fit 的 checkpoint，预先锁定的多指标规则决定配置。但这仍是在同一 inner held-out predictions 上作两层选择；因此 inner 数字只能用于选择和诊断，不能作为无偏的最终成绩。

## 6. 泄漏屏障与每-fit 估计量

每个 fit 必须满足以下训练边界：

- 冻结 AST embedding 的标准化均值和标准差只由该 fit 的训练 calls 计算；
- 20 维声学特征的标准化、缺失填补及任何数据依赖统计只由该 fit 的训练数据计算；
- “global class-balanced call CE”中的 global 只表示当前 fit 训练 calls 的全局类别权重，不得读取 inner validation 或 outer test 的标签/频数；
- outer refit 的所有这些统计只由完整 outer-development set 计算；
- 猫级 validation/test 概率为该猫 calls 概率的算术平均，统计单位始终是猫；
- 内层 split assignment 由单独固定 split seed 产生，与两个 search seeds 和三个 refit seeds 分离，并对三模型、八配置完全共享；
- search/refit 的基础种子、完整 fit seeds、历史 seeds 必须做碰撞审计。主头初始化、post-build RNG 与 cat-batch 顺序应在可比较管线间配对并留下哈希。

最稳妥的工程边界是 selection 阶段根本不生成 outer-test predictions；只有完整 selection lock 写入并校验后才允许独立 outer scorer 读取测试角色。outer 结果不能触发淘汰、补跑、增添 seed、修改候选表或挑选全局配置。

## 7. Fixed、tuned 与结构贡献的报告口径

对每个 `model × arm × refit_seed`，先把四个 outer folds 的预测拼成一次 111 猫 complete OOF，再计算指标。这样每条管线/策略得到 3 个配对的 complete-OOF 估计。主要报告单位是这三个 refit seeds；12 个 folds 和 333 个 repeated cat occurrences 只能作描述，不能当成独立动物扩大样本量。

贡献应分开写：

- **训练参数优化增量**：同一模型 tuned policy 减去 fresh-refit q00 policy；
- **结构增量**：在 fixed-q00 对 fixed-q00、tuned policy 对 tuned policy 的同层比较中计算 U1−A0、C1−A0；不能用 tuned U1/C1 对 fixed A0 混合归因；
- **bound 机制差异**：C1−U1 最接近参数匹配的比较；
- **容量边界**：A0 参数更少，U1/C1 相对 A0 的差异仍包含额外分支容量，不能只归因于年龄声学信息。

q00 的 outer 结果必须是 IDEA-087 下按相同 nested epoch 规则 fresh refit 的结果，不能复用 IDEA-076 的历史 validation predictions。若每折选出的 tuned config 不同，tuned 成绩代表“嵌套选择流程”的表现，不代表已经得到一个可部署的全局固定配置。本轮禁止依据 outer 成绩挑选全局赢家；若以后需要全数据部署配置，应另行授权，并只用已锁规则在全数据内部选择。

## 8. 解释边界

即使 outer complete-OOF 显示 tuned 改善，也必须保留下列限制：

1. 111 只猫及其标签、历史 inner/outer 结果已参与多轮方法发展，IDEA-087 不是新动物或新数据集确认。
2. 两个 search seeds 和三个 refit seeds 重复使用同一批猫；它们刻画训练随机性，不增加独立动物样本量。
3. 每个 inner held-out fold 仅约 3–4 只 kitten，单折 Macro-F1/kitten recall 波动很大；完整 seed-level OOF 汇总能减少折权重偏差，但不能创造更多少数类动物。
4. 只有三个 refit-seed OOF 差值，不适合把常规显著性检验或窄置信区间作为主要证据；应完整报告三次配对值、均值、SD、正/平/负方向和逐类结果。
5. IDEA-076 的“停止同数据 seed 扩展”仍是该固定实验的历史结论。IDEA-087 来自用户的新授权，检验的是预先锁定的 nested HPO 流程，不能把 IDEA-076 改写成通过。

## 9. 进入训练前的独立验收条件

实现与 CPU preflight 至少应证明：

1. roles 哈希、111 cats、792 calls、四个互斥 outer-test 折和上表类别/调用数完全一致；
2. 12 组 inner 3-fold assignment 的 train/validation 猫零交集、三类覆盖、完整 OOF 覆盖和跨模型/配置共享均通过；
3. 8 配置表逐字段一致，q00 是原参数，576 inner fits 身份唯一且预算精确；
4. search/split/refit seeds 已从结果前固定材料导出，相互及历史完整种子无碰撞；
5. 人工小例验证 seed-level OOF 聚合、Brier、F1 标签全集、`ddof=1`、候选池边界、完整 tie-break 和 half-up epoch；
6. 每-fit normalization/imputation/class weights 的来源索引严格等于训练索引；
7. outer-test 在 selection lock 前没有预测或计分路径，selection lock 可由保存的 inner predictions 独立重建；
8. selected=q00 的 alias 不重复计证据，actual-fit 公式和 72 上限正确；
9. resume 只接受身份、输入、配置、代码及预测哈希完全一致的完成 fit；
10. 既有 076/083/084/085/086 与非 087 文件通过总监只读快照复核，组合 digest 保持 `2df36fb271262695b16457777b69cb73570ecc9a645f2bd88fb6a93cbb1382e4`。

CPU preflight 全部通过后，仍只意味着实现可以提交总监作下一阶段判断。一个 inner fit 的技术检查以及其余 GPU fits 都需总监另行授权；技术检查不得根据模型成绩修改协议。

## 10. 审查依据与哈希

| 工件 | SHA-256 |
|---|---|
| 总监 IDEA-087 锁定草案 | `2f99c3d7e577e7dc94884e0d6619bbde524ba1c107b259e31d87023b15db6db8` |
| repeat 0 roles | `87deda39808297e1af5b71283e1d7487a7b88d9288cb492c488e3e64fb91c433` |
| IDEA-076 plan | `bc61029d55fc3aa557ea88db82234350133e13e180ab6998ec0a9ef938585c30` |
| IDEA-076 protocol | `3fede90e88d3fbecc8f87996b53337244502a24eaffd05e8f30902c554dc47f6` |
| IDEA-076 results report | `2f7159ddbb88e5f8ccf2fb9d8818aacc4ba95a8d5ea6cf291cd25913f4aace69` |
| IDEA-071 plan | `c7a5745bd3e6fd28c646b441f64b2f4c41ff10654739be18e0c3d8e966aefaee` |
| IDEA-071 protocol | `3050b16a0c3c63f8ea9b672f5bf0c5ad3771e9696bdf7a3c4773351490fcf79a` |

最终判定：**设计 GO；实现和 CPU preflight 待独立复核；GPU 未授权。**
