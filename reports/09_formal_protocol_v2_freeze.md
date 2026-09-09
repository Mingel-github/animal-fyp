# MeowAgeNet Formal Protocol v2 冻结记录

> 日期：2026-08-27
> Protocol ID：`meowagenet-formal-v2`
> 状态：正式结果尚未运行；协议、划分和对照矩阵已冻结

## 1. 已确认的主线

正式阶段以 **IDEA-048** 为研究目标，以 **IDEA-019 的低参数 AST adaptation** 为主要方法路线。当前冻结的 primary candidate 是 Probe-guided AST adapter。

本阶段拟检验的宽口径主张是：

> 在 MeowAgeNet 的 cat-ID-disjoint 内部评价中，低参数 AST adaptation 能否相对复现的 VGGish+MLP 和匹配的 frozen-AST head-only control，提高 animal-level 猫龄三分类性能。

Probe-guided placement 是当前候选的实现方式。正式 v2 不预设“layer probe 找到了独特年龄机制”，该解释只由 Random 和 Distributed placement 的诊断结果约束。

IDEA-003 暂停。现有 pilot 中，标准 CORAL 没有提高整体性能；formal v2 不加入 ordinal objective，也不继续搜索 ordinal variants。历史报告继续保留，不改写为失败之外的结论。

## 2. Pilot 与正式 v2 的证据边界

Pilot 已经使用全部111只猫完成多轮 outer-fold 评价，IDEA-019 也是在查看 pilot 结果后进入主线。因此 formal v2 是 **pilot-informed prospective evaluation**，不是从未接触过数据的独立验证。

Formal v2 使用五套新生成的四折 cat-ID-disjoint partitions，并在运行前冻结全部模型、seed、指标和解释规则。新 partitions 可以减少对 v1 精确 folds 的依赖，但不会产生新动物；最终论文必须把结果称为 repeated internal grouped validation，不称为 external replication。

## 3. 继承自 v1 的稳定基础

- 公开 MeowAgeNet 数据、manifest 和 checksum；
- 792段唯一叫声、111只猫；
- 排除重复 alias `049A`；
- kitten、adult、senior 三分类；
- cat-ID 为划分和独立性单位；
- 同一只猫的概率先聚合，再计算指标；
- primary metric 为 animal-level macro F1；
- practical threshold `δ_task = 0.03`；
- MIT AudioSet standard AST checkpoint；
- 16 kHz、1.28秒窗口、0.64秒 hop、standard `10 × 10` stride。

IDEA-012 的 geometry search、IDEA-013 的 temporal pooling search、IDEA-017 的完整 architecture comparison 和 full fine-tuning 均不进入 formal v2。

## 4. 正式研究问题与比较

### H048：完整性能目标

`Probe-guided AST adapter − VGGish+MLP`

该比较回答当前模型升级路线是否超过 MeowAgeNet baseline。`0.03 macro F1` 继续作为实践意义阈值。

### H019：adapter 净贡献

`Probe-guided AST adapter − frozen AST head-only`

该比较回答性能变化是否来自低参数 adapter，而不是只来自新的 AST head recipe。只有 H048 与 H019 都产生稳定的正向差值，才适合使用“低参数 AST adaptation 改善性能”的完整表述。

### 诊断比较

- Distributed adapter：固定第3层和第10层；
- Random adapter controls：每个 repeat-fold 事前随机生成两个 layer pairs，分别训练和报告，不把两者 ensemble；
- Last-2 fine-tuning：作为常规部分微调与参数效率对照。

这些比较解释收益来源，不取代两个事前规定的主要比较。

## 5. 正式矩阵

| 类型 | Pipeline | 角色 |
|---|---|---|
| Confirmatory | VGGish+MLP | IDEA-048 baseline |
| Confirmatory | Frozen AST head-only | Adapter 净效应对照 |
| Confirmatory | Probe-guided AST adapter | Primary candidate |
| Diagnostic | Distributed adapter，layers 3+10 | 固定跨深度 placement |
| Diagnostic | 每个 repeat-fold 两个 Random placements | Generic placement variability |
| Diagnostic | Last-2 AST fine-tuning | 常规微调与参数效率对照 |

每个 pipeline 使用5个 split repeats、每个 repeat 4个 outer folds、3个 model seeds。一个 pipeline 形成15套覆盖全部111只猫的 complete out-of-fold evaluations。

## 6. Probe-guided 选层边界

每个 repeat-fold 中，layer probe 只能使用该 fold 的 inner-train cats：

1. 对12层 frozen AST representations 分别训练 balanced logistic regression；
2. 使用3折 stratified cat-level CV；
3. 按 mean animal-level macro F1 排序；
4. 选择前两层；
5. 并列时选择层号较小者；
6. 选层完成后才能训练该 fold 的 adapter 和生成 outer-test prediction。

Outer-test cats 不参与选层、epoch selection、StandardScaler 或 class weight 估计。

## 7. 评价与统计规则

每个 repeat和model seed将4个outer folds的预测拼成一套111-cat OOF predictions。正式报告包含：

- 15个完整 OOF macro F1，不只报告最佳 seed；
- mean、standard deviation、minimum 和 maximum；
- 每个预定 contrast 的15个 paired differences；
- 正向差值出现次数；
- 10,000次 hierarchical paired bootstrap 的95%区间；
- balanced accuracy、QWK、逐类 precision/recall/F1、confusion matrix 和 ordinal MAE；
- trainable parameters、显存和运行时间。

Hierarchical bootstrap 在每次重复中按真实年龄类别重采样 cats，并对 split repeats 和 model seeds 重采样；同一次 bootstrap 对所有 pipelines 使用完全相同的 animals、repeat 和 seed 索引。

## 8. 结果解释规则

完整路线支持需要同时满足：

1. H048 与 H019 的 mean paired difference 均大于0；
2. 两个比较都至少有12/15套 OOF evaluations 为正；
3. 两个95%区间均排除0；
4. H048 mean difference 达到0.03时，才称为具有实践意义的提升。

若 mean H048 difference 达到0.03，但区间仍包含0或方向不足12/15，报告为“具有实践幅度的点估计，但稳定性仍不确定”。`0 < Δ < 0.03` 只称为小幅方向性结果；`Δ ≤ 0` 不支持对应的性能提升 hypothesis。

Probe-guided 与 Random/Distributed 的差异不决定 IDEA-048 是否成功。若它们表现接近，论文主张收敛到一般的低参数 AST adaptation，不声称 probe-guided 具有独特机制。

## 9. 完整性、停止和偏差记录

- 必须运行完整冻结矩阵，不能因中途结果有利或不利而提前结束；
- 不能只报告最佳 fold、seed 或 random placement；
- 任一 pipeline 缺少猫预测时，formal result 不完整；
- 技术失败只能用相同配置重跑；修改配置必须登记 deviation；
- formal v2 运行后新增的模型、loss、placement 或调参均标记为 exploratory，并使用新 ID；
- 不覆盖本协议、split manifests 或原始 pilot 记录。

## 10. 正式阶段必须保留的材料

- protocol、split manifests 和 SHA256；
- source revision、训练脚本及其 SHA256；
- environment 和硬件信息；
- 每个 repeat、fold、seed、pipeline 的逐猫概率；
- layer-probe rankings 与最终选层；
- epoch-selection logs 和技术失败记录；
- 完整统计输出；
- dated deviation log。

这些材料保存在实验 Git 项目或其受控 artifact storage 中。原始音频、模型权重和大型 run artifacts 继续遵守仓库的 data policy，不直接提交 Git。

## 11. 外部实验端的下一步

1. 运行 `scripts/freeze_meowagenet_formal_v2.py` 验证 config、manifest、splits 和 checksum；
2. 扩展训练 runner，使其读取 repeat、outer fold 和 model seed，不硬编码 v1 roles；
3. 在读取 formal-v2 aggregate outcomes 前，完成 dry run、单元测试并记录训练脚本 SHA256；
4. 一次性运行冻结矩阵；
5. 输出完整 OOF predictions、统计报告和 deviation log；
6. 根据预定解释规则形成结论，不根据结果替换 primary candidate。

机器可读配置位于 `configs/protocol/meowagenet_formal_v2.json`，冻结记录位于 `metadata/experiments/meowagenet_formal_v2_freeze.json`。
