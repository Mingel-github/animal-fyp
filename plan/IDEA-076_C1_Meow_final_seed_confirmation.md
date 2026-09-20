# IDEA-076：C1 在 MeowAgeNet 上的最终独立新种子确认

## 1. 决策背景

MeowAgeNet 主数据集优先级最高；其他猫数据次之，犬数据最低。IDEA-075 的犬结果只限定犬外部探索结论，不能用于否定 C1 在 MeowAgeNet 上的候选资格。

IDEA-071～074 的 C1 对匹配 A0 的 36 个 `base_seed × repeat` 单元描述性合并为：平均 Macro-F1 差值 `+0.0050580097`，`21 正／3 平／12 负`；12 个基础种子为 `7 正／5 负`。但四轮复用同一 792 calls、111 cats 和 nested roles，且综合分析不是事前确认性分析。C1 的描述均值刚过 `+0.005`，同时 CE、Brier 和种子方向存在明显异质性，因此只授权一次最终、预先锁定的新种子确认。

完整历史数值冻结在 `metadata/experiments/meowagenet_C1_evidence_ledger_pre_IDEA076.json`。本轮不得根据结果改变公式、特征、cap、宽度、训练配方、种子、聚合或 gate。

## 2. 研究问题

1. 在六个从未使用、且完整训练种子不与 IDEA-068～075 碰撞的基础种子上，C1 能否稳定超过 A0？
2. 仅当第一问全部过 gate 时，C1 的优势能否归因于 RMS-relative `tanh` bound，而不是宽分支容量，即 C1 能否超过参数匹配 U1？

## 3. 锁定管线

共同主路径：

```text
h = ReLU(W_ast x)
```

- `A0_ast_only`：`h_A0 = h`。
- `U1_wide_unbounded_additive`：`c=GELU(W_age a), 20→60; u=W_r c, 60→128; h_U1=h+u`。
- `C1_bounded_wide_additive`：

```text
q = stopgrad(sqrt(mean_j(h_j²) + 1e-8))
r = 0.25 × q × tanh(W_r c)
h_C1 = h + r
```

U1 与 C1 的宽度、参数量和初始状态完全相同；所有年龄输出头零初始化，三管线优化前 logits 必须逐元素相同。

## 4. 数据、训练与独立性

- 固定使用 792 calls、111 cats、MeowAgeNet formal-v2 nested roles。
- 固定使用 IDEA-068 的 20 维无标签年龄声学特征和冻结 AST embedding。
- 固定复用 IDEA-068 的训练超参、确定性设置、动物级验证 CE checkpoint 选择规则。
- 只使用 inner train/validation roles，`outer_test_accessed=false`。
- 统计独立单位仍是猫；跨 seed/repeat 重复出现的猫不能计作独立动物样本。

## 5. 新种子与预算

种子字符串为 `IDEA-076-C1-Meow-final-seed-confirmation-v1`，UTF-8 SHA-256：

```text
6fe04567db046cacdb62d343fb3b6dfa4b7158222675f3c27accc0f2f8bd6bbd
```

按 digest 连续非重叠大端 uint32 对 10000 取模，并排除 IDEA-068～075 已使用基础种子或任何完整训练种子碰撞，首六个合格基础种子固定为：

```text
8807, 268, 6915, 5994, 9330, 4322
```

每个基础种子使用 3 repeats × 4 folds。完整种子为 `base_seed + 10000×repeat + 100×fold`；72 个完整种子全部唯一且与 IDEA-068～075 不碰撞。

```text
3 pipelines × 6 base seeds × 3 repeats × 4 folds = 216 fits
```

## 6. 主要统计口径

- 18 个 `base_seed × repeat` 单元：每个单元先合并 4 folds，再等权平均；这是 Macro-F1、CE、Brier、balanced accuracy 的主要口径。
- 12 个共同 split-cell：同一 `repeat × fold` 先对六个基础种子求平均。
- 72 个逐 fold 配对和 pooled animal occurrences 仅作描述，不视为独立样本。

## 7. 预注册 gate

### 第一层：C1−A0 主 gate，全部条件同时满足

1. 平均 Macro-F1 差值 ≥ `+0.005`；
2. 至少 `4/6` 个基础种子均值严格为正；
3. 至少 `12/18` 个 seed×repeat 严格为正；
4. 至少 `8/12` 个共同 split-cell 非负；
5. 最差共同 split-cell ≥ `-0.03`；
6. C1 等权 CE 不劣于 A0；
7. C1 等权 Brier 不劣于 A0；
8. C1 等权 balanced accuracy 不劣于 A0；
9. 每个基础种子的 senior recall 差值 ≥ `-0.02`。

### 第二层：C1−U1 机制 gate

仅当第一层通过后可解释。全部条件同时满足：

1. 平均 Macro-F1 差值严格 > 0；
2. 至少 `4/6` 个基础种子均值严格为正；
3. 至少 `10/18` 个 seed×repeat 严格为正；
4. C1 等权 CE 不劣于 U1；
5. C1 等权 Brier 不劣于 U1。

若第一层失败，第二层正式状态必须为 `uninterpretable/false`，但完整描述性数字仍保留。

## 8. 执行与审计

1. 在任何训练前锁定 plan、evidence ledger、protocol、runner 和 tests 哈希。
2. 独立 preflight 核对依赖哈希、数据规模、角色隔离、种子无碰撞、参数量、初始 logits、U1/C1 完整初始状态和 C1 预算。
3. preflight 只有 `GO` 才能用 CUDA 正式运行；支持逐 fit `--resume`，已完成 fit 必须校验身份与预测哈希。
4. 完成后再次 `--resume`，分别审计 JSON 语义一致与规范 LF 字节一致；不得因 Windows CRLF 产生伪失败。
5. 中文报告与 metadata 必须完整记录正、平、负结果和 gate；无论结果如何，不得结果后调参或改 gate。

## 9. 决策规则

- 主 gate 通过：C1 获得 MeowAgeNet 最终新种子确认资格；机制 gate 决定能否把收益归因于 bound。
- 主 gate 失败：C1 保留历史探索性证据，但停止在当前 MeowAgeNet 配方上继续追加 seed bank 或结果驱动修补。
- 犬结果不参与本轮 gate，也不改变 MeowAgeNet 优先级。
