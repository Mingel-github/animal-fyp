# IDEA-080：AST 最后一层 LoRA × C1 因子实验结果

## 结论

IDEA-080 的 144/144 个正式 fits 全部完成，但预注册的主效应、组合效应和交互效应 gate 均失败，结论为 **NO-GO**。固定的 AST block 12 Q/V rank-6 LoRA 不作为候选保留，也不保留其与 C1 的组合；不得根据本轮结果改 rank、target、layer、gate 或 seeds 后重试。

九个等权 `base_seed × repeat` 估计中，L1 相对 A0 的平均 Macro-F1 仅增加 `+0.00257`，低于 `+0.005` 门槛，且 CE、Brier 均变差；CL1 相对 C1 的平均 Macro-F1 为 `-0.01156`。预注册交互

`(CL1 − C1) − (L1 − A0)`

为 `-0.01413`，不是正交互。C1 仍维持既有候选身份；IDEA-080 不改变 IDEA-076/078 的冻结结论。

## 文献与固定设计

参数在任何 MeowAgeNet 结果产生之前锁定。LoRA 采用 Hu et al. 的冻结基础权重加低秩更新 `ΔW = (α/r)BA`，并依据原论文与官方实现使用随机 Kaiming A、零 B、`α/r=1`、dropout 0。PETL-AST 在 AST 上将 LoRA 用于 attention 的 query/value，并在匹配设置中使用 rank 6；AST 本身为 12 层、hidden size 768。因此本轮固定：

- 仅 AST one-based block 12；仅 query、value projection；
- rank `6`、alpha `6`、scaling `1.0`、dropout `0`；
- LoRA 参数 `18,432`，bias 冻结；
- C1 保持 20 个年龄特征、width `60`、cap `0.25`、参数 `9,068`；
- A0/C1/L1/CL1 总可训练参数分别为 `99,075 / 108,143 / 117,507 / 126,575`。

依据与锁定说明见 `reports/51_IDEA-080_LoRA_AST_literature_basis.md`。主要来源为 [LoRA 原论文](https://arxiv.org/abs/2106.09685)、[Microsoft 官方 LoRA 实现](https://github.com/microsoft/LoRA)、[PETL-AST](https://arxiv.org/abs/2312.03694) 与 [AST](https://arxiv.org/abs/2104.01778)。

## 实验边界与完成状态

- 数据：MeowAgeNet 792 calls、111 cats、843 segments；独立单位为 cat。
- 角色：formal-v2 nested roles；只产生 train/validation 结果，`outer_test_accessed=false`。
- 共享表示：只读使用 IDEA-078 的 `[843,146,768]` float32 block-11 token cache，没有重复提取。
- base seeds：`59, 7031, 1855`；repeats 为 `0,1,2`；folds 为 `0,1,2,3`；36 个 full seeds 唯一。
- 预算：`4 pipelines × 3 seeds × 3 repeats × 4 folds = 144 fits`；全部完成。
- 执行环境：Python 3.10.12、PyTorch 2.2.2+cu121、CUDA 12.1、NVIDIA GeForce RTX 4060 Ti。

## 正式前验收

CPU preflight 为 `GO_FOR_FORMAL_GPU_RUN`。完整 843-segment cache 经冻结 block 12 重建 A0，和锁定 final embedding 的平均绝对差为 `5.41925e-7`、最大绝对差为 `9.89437e-6`，分别低于 `2e-6` 与 `2e-5` 门槛。

四条 pipeline 的共享 head 初态相同；C1/CL1 的年龄分支状态相同；L1/CL1 的 LoRA 状态相同。零初始化 B 与年龄输出层使 C1、L1、CL1 相对 A0 的初始 logits 最大差都为 `0`，四条初始 loss 均为 `1.179112195968628`。LoRA B 与年龄输出层在零影响起点均可收到梯度；固定非零 probe 后 LoRA A 与年龄 hidden 也均可收到梯度。冻结 tail 无参数获得梯度。

## 主要结果

以下指标先在每个 `base_seed × repeat` 单元合并四个 validation folds，再对九个单元等权平均。

| Pipeline | Macro-F1 | Balanced accuracy | Animal CE | Animal Brier |
|---|---:|---:|---:|---:|
| A0 frozen tail | 0.73570 | 0.76852 | 0.67675 | 0.40381 |
| C1 bounded age | 0.74659 | 0.77593 | 0.68348 | 0.40433 |
| L1 final-block Q/V LoRA | 0.73827 | 0.76389 | 0.68471 | 0.40749 |
| CL1 C1 + LoRA | 0.73503 | 0.76019 | 0.69711 | 0.41411 |

配对 Macro-F1：

| Comparison | Mean delta | SD | Positive / tied / negative | Worst | Best |
|---|---:|---:|---:|---:|---:|
| L1 − A0 | +0.00257 | 0.04221 | 5 / 0 / 4 | -0.05797 | +0.07682 |
| CL1 − C1 | -0.01156 | 0.04273 | 2 / 0 / 7 | -0.07004 | +0.08199 |
| Interaction | -0.01413 | 0.03723 | 4 / 0 / 5 | -0.08639 | +0.02766 |
| C1 − A0（描述性） | +0.01089 | 0.03966 | 5 / 0 / 4 | -0.04076 | +0.07983 |
| CL1 − L1（描述性） | -0.00324 | 0.03813 | 5 / 0 / 4 | -0.06083 | +0.04944 |

C1−A0 在本轮 seeds 上为正是描述性结果，不是新的 C1 confirmation，也不替代 IDEA-076 的既有证据。关键问题是：LoRA 在 A0 背景只有小而不稳定的增益，在 C1 背景反而退化，并没有形成互补。

## 主效应 gate：L1 对 A0

| 条件 | 预注册阈值 | 观察值 | 结果 |
|---|---:|---:|---|
| 平均 Macro-F1 增益 | ≥ +0.005 | +0.00257 | Fail |
| 正向 base seeds | ≥ 2/3 | 2/3 | Pass |
| 正向 seed×repeat | ≥ 6/9 | 5/9 | Fail |
| 非负 split cells | ≥ 8/12 | 5/12 | Fail |
| 最差 split cell | ≥ -0.03 | -0.05780 | Fail |
| Animal CE 不劣 | L1 ≤ A0 | +0.00797 更差 | Fail |
| Animal Brier 不劣 | L1 ≤ A0 | +0.00367 更差 | Fail |
| 每个 seed senior recall | ≥ -0.02 | 最差 -0.06667 | Fail |

三个 base-seed 的 L1−A0 Macro-F1 均值为 `-0.01758、+0.00904、+0.01623`；对应 senior-recall 差为 `-0.05000、-0.06667、+0.03333`。主效应 gate 仅 1/8 条件通过。

## 组合 gate：CL1 对 C1

| 条件 | 预注册阈值 | 观察值 | 结果 |
|---|---:|---:|---|
| 平均 Macro-F1 增益 | ≥ +0.005 | -0.01156 | Fail |
| 正向 base seeds | ≥ 2/3 | 0/3 | Fail |
| 正向 seed×repeat | ≥ 6/9 | 2/9 | Fail |
| 非负 split cells | ≥ 8/12 | 7/12 | Fail |
| 最差 split cell | ≥ -0.03 | -0.08504 | Fail |
| Animal CE 不劣 | CL1 ≤ C1 | +0.01363 更差 | Fail |
| Animal Brier 不劣 | CL1 ≤ C1 | +0.00977 更差 | Fail |
| 每个 seed senior recall | ≥ -0.02 | 最差 -0.08333 | Fail |

三个 base-seed 的 CL1−C1 Macro-F1 均值为 `-0.02993、-0.00159、-0.00316`，三个都为负。对应 senior-recall 差为 `-0.08333、0、-0.06667`。组合 gate 0/8 条件通过。

## 交互 gate

交互均值为 `-0.01413`；0/3 base seeds 为正、4/9 seed×repeat 为正、6/12 split cells 非负，最差 split cell 为 `-0.09991`。五项 raw 条件全部失败。由于主效应与组合效应也未通过，交互按预注册规则不可解释，`interaction_gate_passed=false`。

## 完整性、资源与可复现性

- 144 个 fit identities 唯一；四条 pipeline 各 36 个。
- 检查 144 个 fit summaries、288 个 prediction files、16,020 条 call predictions 和 2,448 条 animal predictions。
- prediction 哈希、validation 身份、outer-test 隔离、可训练参数、共享初始化、逐 epoch cat/call/segment coverage 均通过；没有 NaN、Inf 或 OOM。
- call→animal 重新聚合的最大差为 `0`；概率和的最大误差为 `1.44e-7`；checkpoint 重载最大概率差为 `0`。
- 只读 `--resume` 复核没有重训任何 fit，canonical summary 保持不变。
- A0/C1/L1/CL1 的累计训练时间分别约 `228.64 / 255.39 / 229.12 / 232.63` 秒；最大记录显存约 `493 / 493 / 730 / 730` MiB。
- 正式进程结束后 GPU 回落至桌面基线；没有训练进程残留。

## 锁定产物

- protocol SHA-256：`c2aa57e9dbab8dc955ec1e96ce453566807464f6f0e4380d57987e4a40feb925`
- runner SHA-256：`2bb7359a6c88544d262d5836b66f099fd62acfc028c4789c9a439978fca140da`
- tests SHA-256：`52e614e8d52f6d9c8087516b37468bc9967157537dfd06734057341ea8d6c18a`
- CPU preflight SHA-256：`d3b71a557dd5c8e48ee27407db6d51313b600d119c333f6962b884d81c83e641`
- formal summary SHA-256：`94b1e66f4743089d3f0e15a2f51c872d2e56e598a7d6a2b2c561c92223372204`
- independent audit SHA-256：`08b05e6dc665eb69c16db89e18e332cd91ff3f1c8a0476423dd8dd9230779e7d`
- results metadata SHA-256：`24d53f7042fa1c9a508bbdbeb802543a2e71a25b263bb68f42053bf4bc9aaee9`

## 决策

`main_gate_passed=false`、`combination_gate_passed=false`、`interaction_gate_passed=false`、`factorial_gate_passed=false`。停止当前固定 final-block Q/V rank-6 LoRA 与 LoRA×C1 组合，不作结果驱动修补。保留 IDEA-076 的 C1 决策和 IDEA-078 的只读 cache；本轮所有输出冻结为负结果证据。
