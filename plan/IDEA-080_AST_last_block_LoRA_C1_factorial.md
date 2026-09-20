# IDEA-080：AST 最后 block LoRA × C1 因子实验计划

## 1. 研究问题

在 MeowAgeNet 主任务上，使用已有 label-free block-11 token cache，只训练 AST block 12 的 Q/V LoRA，检验：

1. `L1−A0`：最后 block 的低秩 attention 适配是否优于冻结 AST；
2. `CL1−C1`：同一 LoRA 是否能在冻结 C1 年龄残差背景下提供增量；
3. `(CL1−C1)−(L1−A0)`：C1 与 LoRA 是否出现正交互。

本轮是内部新种子 train/validation 模块筛选，不是外部验证。IDEA-076、IDEA-078 与 IDEA-079 的结果全部冻结，犬数据不参与。

## 2. 文献锁定

中文文献依据见 `reports/51_IDEA-080_LoRA_AST_literature_basis.md`。结果前固定：block 12、Q/V、rank 6、alpha 6、scaling 1.0、LoRA dropout 0、A Kaiming-uniform/B zero、bias frozen。

## 3. 四管线

| Pipeline | 冻结 AST tail | C1 | block-12 Q/V LoRA | 可训练参数 |
|---|---|---|---|---:|
| `A0_frozen_tail` | 是 | 否 | 否 | 99,075 |
| `C1_bounded_age` | 是 | 是 | 否 | 108,143 |
| `L1_lastblock_qv_lora` | 除 Q/V LoRA 外冻结 | 否 | 是 | 117,507 |
| `CL1_C1_plus_lastblock_qv_lora` | 除 Q/V LoRA 外冻结 | 是 | 是 | 126,575 |

C1 严格保持 20 维年龄声学特征、`20→60→128`、`0.25 × stopgrad(RMS(h)) × tanh(...)`。LoRA 严格保持 `ΔW=(6/6)BA`，只作用于 block 12 的 query/value。

## 4. 初始化与配对

- 四管线共享完全相同的 classifier head tensors；
- L1/CL1 共享完全相同的 LoRA tensors；
- C1/CL1 共享完全相同的年龄分支 tensors；
- C1 output 与 LoRA B 均为零，所以四管线优化前 logits/loss 精确相等；
- 每个 pipeline 使用相同 base seed、repeat、fold、cat order 与 call/segment coverage；
- 模型构建后用固定 offset 重置训练 RNG，避免组件数量改变 dropout/batch 随机轨迹。

## 5. 数据与训练

- MeowAgeNet：792 calls、111 cats、843 segments；independence unit 为 cat；
- formal-v2 nested roles；仅 train/validation，`outer_test_accessed=false`；
- 复用 `runs/ast_prelast_tokens_idea078_v1`，不得重复提取；
- 冻结 AST block 1–11，block 12 除 Q/V LoRA 外冻结；
- head/C1/LoRA 统一 Adamax LR `0.006`，epsilon `1e-7`；
- 全局类别平衡 call-level CE，complete-cat batch size 4；
- 最高 50 epochs，patience 8，gradient clip 1.0；
- checkpoint 只按最低未加权 inner-validation animal CE 选择，改善阈值 `1e-6`；
- animal prediction 为 cat 内 call probabilities 算术平均。

PETL-AST 对 adapter/LoRA 使用约 0.005 的初始 LR；本轮不引入 LoRA 专用 LR 搜索，而使用已冻结 head/C1 recipe 的 0.006，保持四管线配对与单一配方。

## 6. 新种子与预算

固定字符串：

```text
IDEA-080-Meow-last-block-LoRA-x-C1-v1
```

UTF-8 SHA-256：

```text
8753401b4be127f74a464f7f3277f1b729be3cef42dc7de997f80a8b4b16c393
```

按连续、互不重叠的 big-endian uint32 取模 10000，并排除 IDEA-068 至 IDEA-079 的 base/full-seed 碰撞，前三个合格值为：

```text
59 / 7031 / 1855
```

完整种子为 `base_seed + 10000×repeat + 100×fold`，36 个均与既往实验无碰撞。

```text
4 pipelines × 3 base seeds × 3 repeats × 4 folds = 144 fits
```

## 7. 主要统计单位

- primary metric：animal-level Macro-F1；
- 九个等权 `base_seed × repeat` 单元，每个单元合并四 folds；
- 12 个 `repeat × fold` split-cells，每个 cell 对三个 base seeds 等权平均；
- secondary：balanced accuracy、animal CE、animal Brier、senior recall；
- 重复出现的猫只作描述，不视为独立样本扩增。

## 8. 预注册 gates

### 主门：L1−A0

全部条件必须同时满足：

1. 九单元平均 Macro-F1 差 ≥ `+0.005`；
2. 至少 `2/3` base-seed 均值严格为正；
3. 至少 `6/9` seed×repeat 严格为正；
4. 至少 `8/12` split-cells 非负；
5. 最差 split-cell ≥ `-0.03`；
6. L1 平均 animal CE 不劣于 A0；
7. L1 平均 animal Brier 不劣于 A0；
8. 每个 base seed 的 senior-recall 差值 ≥ `-0.02`。

### 组合门：CL1−C1

采用与主门完全相同的八项阈值，只把 contrast 换为 CL1−C1。

### 交互门

交互定义：

```text
I = (CL1 − C1) − (L1 − A0) = CL1 − C1 − L1 + A0
```

raw positive-interaction gate 要求：平均 I 严格 > 0；至少 `2/3` base seeds 为正；至少 `5/9` seed×repeat 为正；至少 `8/12` split-cells 非负；最差 split-cell ≥ `-0.03`。只有主门与组合门都通过时，交互门才可解释；否则只作描述。

另外完整报告 `C1−A0` 与 `CL1−L1` 两个年龄分支简单效应，但不为它们新设成功门，避免重开 IDEA-076。

## 9. CPU preflight 与正式授权

CPU-only preflight 必须检查：文献／依赖哈希、cache 和角色边界、完整 A0 tail 重建、参数量与可训练名字、Q/V-only scope、四路初始 logits/loss、配对状态、零影响梯度日程、checkpoint state schema。任何一项失败均为 NO-GO。

preflight 通过只表示工程与预注册合同可执行，不表示模型有效。正式 144 fits 需要研究总监另行授权；本阶段不得使用 GPU。

## 10. 停止与解释规则

- 主门失败：不保留单独 L1；
- 组合门失败：不保留 CL1 作为 C1 增强；
- 主门或组合门失败：不得解释正交互；
- 不根据结果修改 rank、alpha、dropout、target projection、layer、C1 cap/width、seed 或 gate；
- 不增加同数据 seed bank 修复结论；
- outer test 始终关闭，犬结果不参与。
