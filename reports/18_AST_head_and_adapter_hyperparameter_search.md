# AST head-only 与 Probe-guided adapter 超参数搜索报告

> 完成日期：2026-09-01
> 实验状态：`complete_for_exploratory_hpo`
> 实验性质：formal-v2.1 之后的探索性超参数优化
> 评价单位：animal-level complete OOF，111 只猫

## 1. 核心结果

本轮为 AST head-only 与 Probe-guided AST adapter 建立独立调参协议。搜索只使用 inner-train
与 inner-validation，两个 pipeline 各比较 8 组参数，共完成 64 个 inner fit。候选配置在
读取 outer-test 前写入 selection lock；随后对两个锁定配置执行 repeats 0–2 × 4 folds ×
base seed 17，共 24 个 fit 和 6 个 complete OOF。

本轮最明确的收益来自 AST head-only。head 学习率从 `0.0031098` 提高到 `0.006` 后，三次
complete-OOF animal macro F1 平均值达到 **0.7488**，比 formal-v2.1 中相同三个 repeat、
相同 seed 17 的历史平均值 **0.7337** 高 **0.0151**。QWK 同时提高 **0.0125**，balanced
accuracy 变化为 **-0.0085**。这组结果表示新的学习率提高了类别 F1 与年龄顺序一致性，
类别平均召回率出现小幅交换。

adapter 搜索最终选回原配置：head LR `0.0031098`、adapter LR `0.001`、dropout `0.4457`。
它在本轮 8 组 adapter 参数中排名第一，说明现有 adapter 参数已经位于本次小网格的最佳点。

## 2. 搜索设计

两条 pipeline 共享以下固定条件：

- AST backbone 与既有 embedding/fbank 缓存；
- `768 → 128 → 3` 分类 head、ReLU、BatchNorm；
- Adamax、balanced class weights、micro-batch 8、gradient accumulation 4；
- 最大 50 epochs、patience 8、gradient clip 1.0；
- 每折按最低 inner-validation cross-entropy loss 选择 epoch；
- cat-ID-disjoint split bank 与 animal-level 概率聚合。

head-only 搜索 dropout `{0.25, 0.4457}` 与 head LR
`{0.0005, 0.0015, 0.0031098, 0.006}` 的组合。adapter 在 width 32、每折 inner-train-only
probe 选出的两个层上，搜索 dropout `{0.25, 0.4457}`、head LR
`{0.0015, 0.0031098}` 与 adapter LR `{0.0003, 0.001}`。

选择指标为 repeat 0 四个开发折的平均 inner-validation animal macro F1。并列时依次比较
balanced accuracy、QWK、validation loss 和 trial ID。该流程属于**超参数优化 / 模型选择**：
它优化训练配置，保持 AST backbone 和 adapter 结构定义不变。

## 3. Inner-only 选择结果

| Pipeline | 锁定配置 | Inner macro F1 | 原配置 | 搜索阶段变化 |
|---|---|---:|---:|---:|
| AST head-only | dropout 0.4457，head LR 0.006 | **0.7729** | 0.6810 | **+0.0920** |
| Probe-guided adapter | dropout 0.4457，head LR 0.0031098，adapter LR 0.001 | **0.7869** | 0.7869 | 0.0000 |

head-only 在开发折上的提升很大；外层 OOF 的实际提升为 0.0151。两者共同说明学习率方向有效，
同时也显示开发折中的优势只有一部分转移到完整 OOF。adapter 的原配置在当前搜索范围内保持
第一名，较低 adapter LR 或较低 head LR 均未形成更高的四折平均 macro F1。

## 4. 锁定配置的 complete-OOF 结果

| Repeat | Tuned AST head-only | Probe-guided adapter | Adapter − head-only |
|---:|---:|---:|---:|
| 0 | **0.7816** | 0.7317 | -0.0500 |
| 1 | 0.7373 | **0.7576** | +0.0203 |
| 2 | **0.7275** | 0.7208 | -0.0066 |
| Mean | **0.7488** | 0.7367 | **-0.0121** |
| Sample SD | 0.0288 | **0.0189** | — |

| Pipeline | Macro F1 | Balanced accuracy | QWK |
|---|---:|---:|---:|
| Tuned AST head-only | **0.7488** | **0.7644** | **0.6469** |
| Probe-guided adapter | 0.7367 | 0.7592 | 0.6441 |

本轮 tuned head-only 在三个平均指标上均领先，macro F1 优势为 0.0121，balanced accuracy
优势为 0.0051，QWK 优势为 0.0028。按 repeat 观察，head-only 赢得 2 次，adapter 赢得 1 次。
head-only 的平均性能更高，adapter 的 sample SD 更低；两条路线分别呈现更高均值与更小波动。

## 5. 与 formal-v2.1 seed-17 历史结果的关系

### 5.1 Head-only 调参收益

| Repeat | Formal head-only | Tuned head-only | 差值 |
|---:|---:|---:|---:|
| 0 | 0.7182 | 0.7816 | +0.0634 |
| 1 | 0.7616 | 0.7373 | -0.0243 |
| 2 | 0.7212 | 0.7275 | +0.0063 |
| Mean | 0.7337 | **0.7488** | **+0.0151** |

较高学习率在 repeat 0 与 repeat 2 获得正向变化，在 repeat 1 出现回落。平均提升达到 1.51
个百分点，效果规模属于**小而有价值的改进**。它支持将 LR `0.006` 作为下一阶段的
head-only performance candidate，并为后续 seeds 43/101 扩展提供明确候选。

### 5.2 Adapter 重跑波动

adapter 锁定回与 formal-v2.1 完全相同的超参数，12 个 fit 的 probe 层组合也逐一相同。
它的历史 seed-17 平均 macro F1 为 0.7348，本次为 0.7367，平均变化 **+0.0019**，总体水平
高度接近。12 个 fit 中有 5 个选择了不同的最低验证损失 epoch。

当前训练代码固定 Python、NumPy 与 PyTorch seed；CUDA 计算保持默认高性能算法选择，未强制
deterministic algorithms。浮点计算轨迹的细小变化会改变 validation loss 的最低点，进而改变
outer retrain epoch。该现象解释了单个 repeat 的上下波动；三次平均值仍然稳定复现 adapter
的约 0.735–0.737 水平。formal-v2.1 继续作为已冻结的正式历史记录，本轮记录探索性重跑结果。

## 6. 阶段结论

本轮形成四条直接结论：

1. AST head-only 对 head learning rate 较敏感，`0.006` 比原 `0.0031098` 更适合当前
   seed-17 三-repeat 协议，macro F1 平均提升 0.0151；
2. Probe-guided adapter 的原参数在本轮小网格中排名第一，当前 dropout 与两组学习率组合
   保持为 adapter 的首选配置；
3. tuned head-only 当前以 0.7488 的 macro F1 位列两条 AST 路线第一，adapter 以 0.7367
   提供接近的性能和更小的 repeat 波动；
4. 本轮属于阶段性超参数优化结果。未来可用 seeds 43/101 验证 head-only 的 0.0151 平均
   收益，并根据论文篇幅决定是否加入 deterministic CUDA 重跑消融。

因此，项目目前保留两条清晰角色：**tuned AST head-only 是阶段性性能候选，Probe-guided
adapter 是已完成机制研究并保留的低参数适配候选**。这次调参提升了简单 head 的竞争力，
也把 adapter 的可调空间收敛回原配置。

## 7. 可复核文件

- 协议：`configs/protocol/meowagenet_ast_hpo_v1.json`
- 独立 runner：`scripts/run_meowagenet_ast_hpo_v1.py`
- 搜索锁：`runs/meowagenet_ast_hpo_v1/search/selection.json`
- OOF 汇总：`runs/meowagenet_ast_hpo_v1/evaluation/summary.json`
- 机器可读结果：`metadata/experiments/meowagenet_ast_hpo_v1_results.json`
