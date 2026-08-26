# IDEA-003 Ordinal Learning 实验报告

> 日期：2026-08-26
> 状态：五种 objective、四折 GPU pilot 与 animal-level paired bootstrap 已完成
> 主指标：animal-level quadratic weighted kappa（QWK）

## 1. 结论摘要

IDEA-003 得到**部分支持**：年龄类别的真实顺序确实包含有效结构，但当前标准 CORAL
没有把这种结构转化为相对三分类 CE 的整体性能提升。

1. 正确顺序 CORAL 的 QWK 为 **0.5434**，与同框架 nominal CE 的 0.5568 接近；
   配对差为 -0.0134，95% bootstrap 区间为 `[-0.1042, +0.0867]`。
2. 正确顺序 CORAL 比真正打乱顺序的 CORAL 高 **0.3093 QWK**，区间
   `[+0.1013, +0.5251]`。这给“kitten → adult → senior 的自然顺序有信息”提供了
   清楚的支持。
3. CORAL 把 kitten↔senior 极端翻转从 nominal CE 的 2/111 降到 1/111，同时产生
   更多 adult→端点的一步误差。它维持了 QWK，却使 macro F1 从 0.6381 降至 0.5260，
   ordinal MAE 从 0.3874 升至 0.4865。
4. Quadratic-cost CE 的 macro F1 为 **0.6741**、准确率为 **0.6937**、ordinal MAE
   为 **0.3333**，均是本轮 objective 对照中的最好结果；它的 QWK 为 0.5397，较
   nominal CE 低 0.0171。该方法主要改善 adult 分类和普通一步误差，代价是多 1 个
   kitten↔senior 极端翻转。

与锁定 VGGish baseline 相比，quadratic-cost CE 的 macro F1 低 0.0105、balanced
accuracy 低 0.0035，QWK 则高 0.0133。两者总体接近且各有优势：锁定 baseline 保持
更高的三分类 F1，quadratic-cost CE 更强调年龄距离一致性。

因此，本轮不把标准 CORAL 纳入最终性能 pipeline。锁定的 VGGish + MLP 三分类
baseline 保持不变；quadratic-cost CE 作为“改善 macro F1/MAE 的可选损失消融”保留，
不会另立成第二个 baseline。路线可继续进入 **IDEA-019**。

## 2. 本轮到底比较了什么

所有方法读取相同的 128 维 VGGish embeddings，使用相同 cat-ID folds、train-fold
StandardScaler、`128 → Dense(128) → ReLU → BatchNorm → Dropout` trunk、balanced
class weights、Adamax 和 early stopping。五种方法只改变输出 objective 或类别顺序：

| 名称 | 定义 | 检验目的 |
|---|---|---|
| Nominal CE | 普通三分类交叉熵 | 同一 PyTorch 实现中的 objective 参照 |
| Quadratic-cost CE | CE + 预测概率的归一化平方距离代价 | 检验“错误距离有代价”本身是否有效 |
| CORAL correct | `kitten < adult < senior` | 检验真实年龄序关系 |
| CORAL reversed | `senior < adult < kitten` | 检验同一年龄轴反向参数化的数学对称性 |
| CORAL shuffled | `kitten < senior < adult` | 真正破坏年龄邻接关系的顺序对照 |

CORAL 使用一个共享 latent score 和两个有序 thresholds，学习两个累计问题：

- 是否高于第一个年龄等级；
- 是否高于第二个年龄等级。

这样生成的三类概率天然位于一条有序轴上。Quadratic-cost CE 仍保留三个独立 softmax
输出，只在损失中让跨两级的错误比相邻一级错误更贵。

### “反向顺序”为什么不是错误顺序

`kitten < adult < senior` 与 `senior < adult < kitten` 描述的是同一条直线，只是坐标
方向相反。对共享-score CORAL 而言，score 和 thresholds 做相应镜像后，模型族完全
等价。因此，反向顺序用于核验实现和训练稳定性；真正的顺序特异性证据来自把 adult
移到端点的 shuffled control。

单折三 epoch smoke test 中，correct 与 reversed 的损失和离散预测完全一致。正式
独立 GPU 长程训练中，两者因浮点累积、dropout/矩阵规约的数值路径以及 val-loss
选 epoch 的放大作用出现 QWK -0.0408 的差异，其区间 `[-0.1040, +0.0112]` 包含 0。
这项差异作为单 seed 长程训练的稳定性信息记录，不作为“反向年龄轴更差”的证据。

## 3. 数据与训练协议

- 分析范围：936 个 VGGish prediction units、111 只猫；排除 alias `049A`；
- 单元标签分布：kitten 170、adult 460、senior 306；
- animal 标签分布：kitten 15、adult 62、senior 34；
- 年龄边界：`<0.5` 岁为 kitten，`0.5–<10` 岁为 adult，`≥10` 岁为 senior；
- 固定 4 个 cat-ID-disjoint outer folds；test 单元数为 209、248、253、226；
- 每折在 inner validation objective loss 选择 epoch；
- 随后使用 outer train + validation cats 从头训练相同 epoch 数；
- batch size 128，Adamax learning rate `0.003109800273709165`；
- balanced class weights，最大 500 epochs，patience 30；
- GPU：NVIDIA GeForce RTX 4060 Ti，PyTorch 2.2.2+cu121；
- 主指标：animal-level QWK；同时报告 macro F1、balanced accuracy、ordinal MAE、
  expected-rank MAE、极端翻转率、adult prediction rate 和混淆矩阵。

五种方法的 trunk 初始化逐参数一致，初始三类概率均为 1/3。这样可以把训练后的差异
集中解释为 objective 与顺序约束的差异。

## 4. 总体结果

| Pipeline | Macro F1 | Balanced accuracy | QWK | Accuracy | Ordinal MAE | 极端翻转率 | Adult 预测率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 锁定 VGGish + MLP baseline（论文主 baseline） | **0.6846** | 0.6754 | 0.5263 | — | — | — | — |
| Nominal CE objective control | 0.6381 | **0.6768** | **0.5568** | 0.6306 | 0.3874 | 0.0180 | 0.3874 |
| Quadratic-cost CE | **0.6741** | 0.6719 | 0.5397 | **0.6937** | **0.3333** | 0.0270 | 0.5135 |
| CORAL correct | 0.5260 | 0.6300 | 0.5434 | 0.5225 | 0.4865 | **0.0090** | 0.2162 |
| CORAL reversed | 0.5152 | 0.5963 | 0.5026 | 0.5225 | 0.4955 | 0.0180 | 0.2432 |
| CORAL shuffled | 0.5517 | 0.5543 | 0.2341 | 0.5946 | 0.4595 | 0.0541 | 0.6757 |

第一行继续是项目唯一的 VGGish 三分类 baseline。Nominal CE 行是本轮为了让不同
objective 共享 PyTorch 初始化和训练代码而设置的**实验内参照**。它采用均匀输出
初始化，而锁定 baseline 使用原 Keras/Glorot 初始化；所以两者的 F1/QWK 权衡不同，
实验内参照不会替代或复制论文 baseline。

## 5. QWK、F1 和 MAE 为什么会给出不同侧面

QWK 会按错误距离加权：adult→kitten 或 adult→senior 是一步错误，kitten→senior
是两步错误，后者受到更重惩罚。Macro F1 则分别计算三类 F1 后等权平均，更关心每类
是否都被识别好。Ordinal MAE 直接把一步记为 1、两步记为 2，再取平均。

因此，CORAL correct 虽然只正确识别 17/62 只 adult，却把两端极端翻转控制在 1 只，
并正确识别 30/34 只 senior。它的 macro F1 和 MAE明显变差，QWK 仍保持在 nominal
CE 附近。这个组合说明 CORAL 学到了一条平滑年龄轴，但 decision boundaries 把 adult
区域压得过窄。

Quadratic-cost CE 保留三个独立类别区域，正确识别 44/62 只 adult，所以准确率、
macro F1 和 MAE更好；它正确识别 9/15 只 kitten，并出现 3 个极端翻转，所以 QWK
没有同步提高。两种 ordinal 思路各自强调了不同优势：

- CORAL：最少的跨两级错误，senior recall 最高；
- Quadratic-cost CE：更多整体正确分类，更强的 adult recall，更低的一般等级误差；
- Nominal CE：本轮最高 QWK 和 balanced accuracy，三类边界相对均衡。

## 6. 混淆矩阵揭示的具体机制

行是真实类别，列是预测类别。

### Nominal CE

| 真实 \ 预测 | Kitten | Adult | Senior |
|---|---:|---:|---:|
| Kitten | 11 | 3 | 1 |
| Adult | 6 | 33 | 23 |
| Senior | 1 | 7 | 26 |

三类 recall 分别为 0.7333、0.5323、0.7647。它保留了较宽的 adult 区域，同时有
23 只 adult 被判成 senior。

### Quadratic-cost CE

| 真实 \ 预测 | Kitten | Adult | Senior |
|---|---:|---:|---:|
| Kitten | 9 | 4 | 2 |
| Adult | 3 | 44 | 15 |
| Senior | 1 | 9 | 24 |

adult recall 提高到 0.7097，adult→senior 从 23 降到 15；kitten recall 降到 0.6000。
平方距离代价主要扩大了中间 adult 的吸引区域，因此普通一步误差减少。

### CORAL correct

| 真实 \ 预测 | Kitten | Adult | Senior |
|---|---:|---:|---:|
| Kitten | 11 | 4 | 0 |
| Adult | 12 | 17 | 33 |
| Senior | 1 | 3 | 30 |

senior recall 达到 0.8824，kitten→senior 为 0；adult recall 为 0.2742，adult 预测率
只有 0.2162。当前 CORAL 的主要问题是 middle-class compression，而不是年龄顺序
本身缺少信息。

### CORAL shuffled

| 真实 \ 预测 | Kitten | Adult | Senior |
|---|---:|---:|---:|
| Kitten | 10 | 0 | 5 |
| Adult | 3 | 49 | 10 |
| Senior | 1 | 26 | 7 |

打乱后 adult 成为 ordinal 轴端点，模型把 26/34 只 senior 推到 adult，并产生 5 个
kitten→senior 错误。其 QWK 降到 0.2341、极端翻转增至 6/111，说明正确邻接关系对
有序模型非常重要。

## 7. 配对 bootstrap

使用同一组 111 只 outer-test cats，按真实类别分层重采样 2,000 次。差值均为
candidate − reference。

| 对比 | ΔQWK（95% 区间） | ΔMacro F1（95% 区间） | ΔOrdinal MAE（95% 区间） | Δ极端翻转率 |
|---|---:|---:|---:|---:|
| CORAL correct − Nominal CE | -0.0134 `[-0.1042,+0.0867]` | -0.1121 `[-0.2043,-0.0240]` | +0.0991 `[+0.0180,+0.1802]` | -0.0090 |
| Quadratic cost − Nominal CE | -0.0171 `[-0.1105,+0.0669]` | +0.0360 `[-0.0346,+0.1067]` | -0.0541 `[-0.1171,+0.0090]` | +0.0090 |
| CORAL correct − CORAL shuffled | **+0.3093** `[+0.1013,+0.5251]` | -0.0257 `[-0.1391,+0.0836]` | +0.0270 `[-0.0811,+0.1351]` | **-0.0450** |

最稳定的信号是正确顺序相对 shuffled 的 QWK 与极端翻转优势。CORAL 与 nominal CE
的 QWK 差位于偶然波动范围；其 F1/MAE 差异则方向清楚。Quadratic cost 的 F1 和 MAE
点估计有优势，当前 111-cat 区间仍覆盖 0。

## 8. 四折稳定性

| Pipeline | Fold 0 QWK | Fold 1 QWK | Fold 2 QWK | Fold 3 QWK | 合并 QWK |
|---|---:|---:|---:|---:|---:|
| Nominal CE | 0.7030 | **0.5969** | 0.6696 | 0.3226 | **0.5568** |
| Quadratic-cost CE | **0.7375** | 0.5116 | 0.5227 | **0.4146** | 0.5397 |
| CORAL correct | 0.6111 | 0.5473 | **0.6752** | 0.3460 | 0.5434 |
| CORAL shuffled | 0.5333 | 0.3053 | 0.0939 | 0.0597 | 0.2341 |

三个合理 objectives 在不同 folds 各有优势：quadratic cost 赢 fold 0 和 fold 3，nominal
CE 赢 fold 1，CORAL correct 赢 fold 2。它们的合并 QWK 位于 0.5397–0.5568 的窄区间，
可以理解为主指标不相上下；混淆矩阵进一步说明它们通过不同类别权衡达到相近 QWK。
Shuffled control 在 fold 1–3 持续落后，顺序破坏的影响远大于前三者之间的差异。

## 9. 路线判断

本轮形成三条可直接用于论文的结论：

1. 年龄的自然顺序具有可学习价值：正确 CORAL 的 QWK 显著高于 shuffled CORAL；
2. 标准 CORAL 的强单轴约束压缩了 adult decision region，当前实现不进入最终模型；
3. 距离敏感 CE 相对本轮 nominal CE control 提高 macro F1 和准确率，并降低普通
   ordinal MAE，是有价值的 loss ablation；它的 macro F1 尚未超过锁定 VGGish
   baseline，但 QWK 略高于锁定 baseline。相对本轮 nominal CE control，它没有提升
   primary QWK。

项目主 baseline 继续使用锁定的三分类 VGGish + MLP。IDEA-003 作为“有序结构有效、
标准 CORAL 约束过强、soft cost 更适配当前小数据”的完整实验归档。下一条既定路线是
**IDEA-019**。

## 10. 产物与完整性

- 正式汇总：`runs/idea003_ordinal_learning_v1/summary.json`
- 五种方法的 unit/animal predictions：`runs/idea003_ordinal_learning_v1/`
- 训练脚本：`scripts/run_idea003_ordinal_learning.py`
- 机器可读元数据：`metadata/experiments/meowagenet_idea003_ordinal_learning_v1_results.json`

完整性检查确认：每种方法包含 936 个唯一 prediction units、111 个唯一 cats；每只猫
只进入一个 outer-test fold；四折规模保持 209/248/253/226；所有概率有限、非负且
三类和为 1，最大数值误差小于 `1.4e-7`。
