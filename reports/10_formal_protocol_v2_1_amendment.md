# MeowAgeNet Formal Protocol v2.1 修订记录

> 日期：2026-08-27
> 状态：formal outcome 尚未运行；证据底座已冻结，具体执行范围待 execution lock
> 本文件取代 v2 作为执行依据，v2 继续保留为审计记录

## 1. 术语先说明

本文中的 **H** 是 **Hypothesis，即研究假设**。研究假设是等待实验检验的命题，不是模型名称，也不是已经成立的结论。

- **H048** 对应 IDEA-048：选定的低参数 AST adaptation pipeline 能否比 VGGish+MLP 提高 MeowAgeNet 预测性能；
- **H019** 对应 IDEA-019：adapter 本身能否比 matched frozen-AST head-only control 带来增益；
- **IDEA** 表示研究方向。一个 IDEA 可以产生多个 research questions、hypotheses、模型实现和实验。

为了提高可读性，后文优先写“性能假设 H048”和“adapter 贡献假设 H019”，避免只出现编号。

## 2. 为什么修订 v2

v2 正确锁定了数据、cat-ID 独立性和评价口径，但把以下内容同时设为硬要求：

- 5个 split repeats；
- 3个 model seeds；
- 6类 pipeline；
- 每个 repeat-fold 两个 Random placements；
- 420次 fold-level fits；
- 固定的12/15正向次数和 interval gate。

这些要求适合作为理想完整矩阵，不适合作为资源有限毕业设计的最低进入条件。它还容易造成一种误解：只有全部420次训练完成，实验才算正式。

v2.1 将“正式性”重新定义为：研究问题、数据独立性、核心对照、主指标和结果报告规则事前明确。模型开发和诊断规模可以调整，但调整发生的时间和依据必须留下记录。

## 3. 继续冻结的证据底座

以下内容不随模型尝试改变：

- 792段叫声、111只猫及现有 manifest/checksum；
- 按 cat-ID 隔离 train、validation 和 test；
- VGGish+MLP 是 IDEA-048 的正式 baseline；
- matched frozen AST head-only 是 adapter 净贡献对照；
- animal-level macro F1 是 primary metric；
- balanced accuracy、QWK 和逐类指标用于解释；
- `0.03 macro F1` 是实践幅度参考，不是机械通过线；
- outer-test 结果不能选择模型、层、epoch、超参数或 repeat 数量；
- 不能只报告最佳 fold、seed 或 placement。

这些条目决定证据是否可比较。改变其中任何一项都需要新的 protocol version。

## 4. 保留的模型开发空间

在创建 execution lock 之前，团队可以继续：

- 修复 runner 和数据处理问题；
- 使用 synthetic data、smoke test 或 inner-training roles 检查实现；
- 在 IDEA-019 范围内比较 adapter 实现；
- 将当前 Probe-guided reference 替换为另一个有明确记录的低参数 AST adapter；
- 为显存调整 batch size 或 gradient accumulation，同时保持有效优化设定可比。

如果没有新决定，Probe-guided AST adapter 继续作为默认 primary candidate。若要替换，execution lock 必须写明 candidate ID、完整 recipe、选择依据和日期。

Formal outer-test aggregate outcome 一旦开始生成，primary candidate 不再根据结果替换。后续新模型仍可实验，但使用新的 exploratory ID 或下一版本 protocol。

## 5. 最小正式 core

| Pipeline | 作用 |
|---|---|
| VGGish+MLP | IDEA-048 性能 baseline |
| Frozen AST head-only | Adapter 净贡献对照 |
| Selected IDEA-019 adapter | 主要低参数 AST candidate |

最低执行范围为：

- split repeats 0、1、2；
- 每个 repeat 4个 outer folds；
- model seeds 17、43、101；
- 每个 pipeline 9套完整的111-cat OOF evaluations；
- 三个 core pipelines 共108次 fold-level fits。

Split repeats 3、4已经预生成，可以在 execution lock 中直接启用。启用全部5个 repeats 时，core 为180次 fold-level fits。

Exact repeat scope 应在查看 formal aggregate outcome 前写入 execution lock。之后如果因资源或精度需要增加 repeats，必须登记追加原因、当时是否已经查看结果以及对解释的影响；不能因为当前结果接近预期而选择停止。

## 6. 可选诊断模块

以下模块不再是“formal core 完成”的前提：

- Distributed adapter，layers 3+10；
- 事前生成的 Random placement controls；
- Last-2 AST fine-tuning。

它们分别回答 fixed cross-depth placement、generic placement variability 和 parameter efficiency。团队在 execution lock 中选择本轮启用哪些模块。

未启动的模块可以后置。已经启动的模块必须完成其声明的 paired scope，并报告全部有效结果，不能只留下有利配置。

## 7. H048 与 H019 如何解释

### 性能假设 H048

比较：`selected IDEA-019 adapter − VGGish+MLP`。

它回答完整模型升级是否改善 MeowAgeNet 性能。报告 mean paired difference、不同 repeats/seeds 的变化和 paired uncertainty interval。`0.03` 用于说明效果是否达到预先认为有实际意义的幅度，不自动决定论文成功或失败。

### Adapter 贡献假设 H019

比较：`selected IDEA-019 adapter − frozen AST head-only`。

它区分 adapter 增益与 AST head recipe 增益。如果 H048 为正而 H019 接近0，结果更支持“AST pipeline 改进”，不足以把提升归因于 adapter。

最终解释同时考虑方向、幅度、区间和跨 repeat/seed 稳定性。协议不再规定“必须12/15为正”这种机械门槛：

- 点估计为正且跨条件稳定、区间主要位于0以上，可以形成较强支持；
- 点估计为正但区间跨0，应写成“观察到正向结果，但幅度仍不确定”；
- 平均差值接近0，说明没有识别到稳定增益；
- 平均差值为负，挑战对应的改善假设。

结论由团队和导师结合全部结果判断，不由单一阈值自动生成。

## 8. Execution lock 的作用

v2.1 不再把所有实验细节永久锁死，而是在正式运行前增加一份短 execution lock。它需要写明：

- 本轮 selected IDEA-019 adapter 及完整 recipe；
- 使用3个还是5个 split repeats；
- model seeds；
- 启用的 optional modules；
- runner revision 和 SHA256；
- environment、hardware 和决定日期；
- 在锁定前是否已经查看 formal outcome。

模板位于 `configs/protocol/meowagenet_formal_v2_1_execution_lock_template.json`。外部实验端完成 runner 和 smoke test 后填写该文件，再开始正式 outer-test aggregation。

## 9. 当前版本关系

- v1：最初的单次 pilot 协议；
- IDEA-048 checkpoint：保存 pilot 结论；
- formal v2：第一版严格矩阵，保留为设计历史；
- **formal v2.1：当前执行依据**；
- execution lock：外部实验开始前确定本轮实际范围。

截至本次修订，没有运行或查看 formal-v2/v2.1 outcome。已有的0.6846和0.7575仍是 pilot evidence。
