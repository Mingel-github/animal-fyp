# IDEA-057｜AST 结构化局部 patch 分支

> 阶段：诊断后候选，等待 executable protocol
> 来源：AST internal diagnosis 的局部 patch 证据
> 主要 claim 类型：predictive / mechanistic candidate

## 1. 已观察结果

这些结果来自 inner-only AST internal diagnosis：

- final global probe 的 animal Macro F1 为 `0.7649`；
- AST 频率网格中间区域单独取得 `0.7437`，差值为 `-0.0213`；
- 中频区域相对 global probe 独有 7 次正确，global 独有 15 次正确；
- 均值替换中段时间区域后，真实类别概率平均下降 `0.1781`，12/12 splits 为正；
- IDEA-052 的全时间 temporal-mean residual 平均差为 `-0.0009`，说明简单时间均值已经完成
  筛选，保留二维局部位置是本轮的新区别。

局部诊断的完整支持规则没有通过。以上结果支持建立候选实验，不构成模块有效性的既有证据。

## 2. 研究问题与假设

**研究问题：** 在保留 final AST global embedding 主路径时，对中频和相对时间位置进行
结构化读取的轻量分支，能否提高 unseen-cat 年龄分类的 Macro F1？

**H057：** AST 的 final patch tokens 中存在未被 global pooling 稳定利用的条件信息。
位置感知的局部分支可以在保持 global prediction 的基础上修正一部分错误。

**H057-R1，冗余解释：** 中频与中段时间很重要，但相关信息已经进入 global embedding，
新分支只重复现有信息。

**H057-R2，容量解释：** 性能变化来自新增参数，而不是局部时间—频率结构。

**H057-R3，干扰变量解释：** 分支主要学习 call duration、padding、能量或切分边界，在某些
split 上形成短期收益。

**H057-R4，遮蔽伪影解释：** 均值替换产生训练分布之外的输入，较大性能下降高估了对应区域
的机制作用。

## 3. 候选模块边界

M1 必须同时满足：

- 保留 IDEA-056 的 final global AST 主路径；
- 读取 AST patch tokens，并保留相对时间—频率位置；
- 重点使用诊断指出的中频与有效时间结构；
- 使用轻量 pooling、gating、separable projection 或同等复杂度结构；
- 不重复 IDEA-052 的全时间 token mean residual；
- 在 inner-only 阶段将具体结构收缩到一个 recipe，再访问 outer-test。

候选搜索保持有界，优先比较不超过两种局部读取方式。最终 token 区域、隐藏维度、dropout、
初始化和正则化在 executable protocol 中列明。诊断结果不能继续用于反复调整 outer 模型。

## 4. 核心对照

| 编号 | Pipeline | 需要回答的问题 |
| --- | --- | --- |
| R0 | IDEA-056 tuned frozen AST reference | 原始 AST 在相同运行条件下达到什么水平 |
| M1 | final global path + structured local patch branch | 局部结构能否提供增量 |
| C1 | 与 M1 使用相同分支和近似参数量，但将局部 token 变为无位置结构的汇总输入 | 增量来自局部位置机制还是普通新增容量 |

C1 的具体无位置实现可采用 call 内 patch mean 重复输入或其他保持维度一致的确定性变换；选择
必须在 outer-test 前冻结。报告 M1 与 C1 的实际 trainable parameters 和计算量差异。

## 5. 可区分预测

- H057 获得支持性证据：M1 跨 repeats 高于 R0，并且相对 C1 保留同方向增量；
- 容量解释更相容：M1 与 C1 同时提高且两者接近；
- 冗余解释更相容：M1/C1 都接近或低于 R0；
- 干扰变量解释更相容：收益集中在特定 duration、valid-frame fraction 或能量组，跨 split
  方向明显改变；
- 局部机制代价：Macro F1 提高但 Kitten/Senior recall、Balanced Accuracy 或 CE 系统下降。

## 6. 评价与继续条件

首轮使用现有 animal-ID-disjoint roles，方法选择只访问 inner-training / inner-validation。
主要结果为 complete-OOF animal Macro F1；同时报告 Balanced Accuracy、QWK、普通 accuracy、
CE、各类别 recall、逐猫预测改变、call-duration 分组和参数量。

初始继续条件采用双方向总计划中的共同规则：M1 相对 R0 平均 Macro F1 至少 `+0.005`，三个
complete-OOF repeats 至少两个为正，并且 M1 相对 C1 的结果支持局部结构解释。满足条件后
锁定 recipe 并扩展新的 random seeds；未满足时 IDEA-057 作为结构化局部 patch 消融收尾。

## 7. 最小交付物

- executable protocol 与 runner；
- R0/M1/C1 参数、初始化和 feature-shape audit；
- inner-only selection record 与 outer execution lock；
- complete-OOF 主要指标、逐类别结果和 nuisance-variable analysis；
- 中文报告、机器可读 JSON 和是否 seed expansion 的团队决定。

## 8. 术语说明

| 术语 | 含义 |
| --- | --- |
| structured local patch branch（结构化局部分支） | 读取 AST 的局部 patch tokens，并保留它们在时间和频率网格中的相对位置。该名称是本项目对候选模块的描述，不是现成算法名称。 |
| global path（全局主路径） | 使用 AST 最终汇总向量完成分类的原参考路径。新模块保留这条路径，使局部信息只作为增量。 |
| position-aware（位置感知） | 模型能够区分一个 token 位于叫声的前、中、后段或不同频率区域，而不是只看到全部 token 的平均值。 |
| separable projection（可分离投影） | 分别处理时间和频率关系的轻量映射，用较少参数保留二维结构。它是可选实现类别，最终结构需要在 protocol 中明确。 |
