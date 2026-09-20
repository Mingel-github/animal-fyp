# IDEA-089 基线、数据与统一协议独立设计审计

日期：2026-09-20  
状态：**设计审计通过；仅可进入实现/CPU 预检，未授权真实训练**

## 1. 独立结论

在不导入 IDEA-089 runner、不开启任何训练的条件下，我从冻结源文件重新核对了数据、
角色、标签、VGG 输入、两套缓存、历史结果资格、参数量与 fit 预算。结论如下：

1. `formal-v2.1` 的 repeats 0/1/2 均由 4 个猫隔离 folds 构成；每个 repeat 的 test
   角色恰好覆盖 111 猫各一次，可以形成完整 OOF。
2. 清洗数据是 792 个 unique calls、111 猫；猫标签为 kitten/adult/senior =
   15/62/34，call 标签为 134/405/253；重复别名 `049A` 不可达。
3. 官方 VGG CSV 是 937 rows/112 published IDs；按 clean111 过滤后是 936
   embedding/window rows/111 猫。其列为 `0..127 + mean_freq + gender + target + cat_id`。
   因而旧 formal-v2.1 的 VGG 输入确为 128 维，而作者风格输入是 128 个 embedding
   列加 `mean_freq`，共 129 维。
4. VGG clean111 的猫级语义标签与 call manifest 对 111 猫全部一致；但 VGG row 与
   AST call 没有可靠的一一映射，二者只能在猫 ID、标签、角色和最终猫级指标上对齐。
5. IDEA-089 的主表必须全部重新训练。历史角色和无标签特征缓存可复用；历史权重、
   checkpoint、预测、best epoch 和成绩只能进入有协议标签的背景表。
6. 六个 pipelines、3 repeats、4 folds、3 seeds 对应 216 个 selection fits 与 216 个
   outer refits，总计 432 个真实 fits。此算术及 A0/D0/U1/C1 参数量均复核通过。

因此，冻结设计在统计单位和实验预算上可执行；下一门禁是 runner、协议、合成测试和
CPU 预检的独立审核。本报告不是训练授权。

## 2. 输入与标签事实

### 2.1 清洗音频与缓存

| 资产 | 独立核对结果 | 单位/边界 |
| --- | ---: | --- |
| clean data manifest | 792 calls / 111 cats | call |
| 冻结 AST | `792 × 768` | 与 manifest filename 顺序完全一致 |
| 冻结声学特征 | `792 × 20` | call ID 顺序与 AST 完全一致 |
| 官方 VGG CSV | 937 rows / 112 IDs | embedding/window row |
| clean VGG | 936 rows / 111 cats | 无 row→call 键 |

20 维声学缓存由标签盲提取生成，但不是全有限矩阵：其中 20 个 calls 在 17 个声学维度
各缺 1 值，共 340 个 NaN，772 个 calls 全维有限，没有无穷值。故每个 selection 阶段
只能以 train 角色拟合逐维有限中位数和标准化量；outer refit 必须在 train+validation
上重新拟合，不能沿用 selection 的统计量。test 不能参与填补、标准化或类别权重估计。

统一语义标签顺序固定为 `kitten=0, adult=1, senior=2`，年龄边界为：

- kitten：`0 ≤ age < 0.5`；
- adult：`0.5 ≤ age < 10`；
- senior：`10 ≤ age < 20`。

旧 sklearn `LabelEncoder` 的整数顺序 `adult, kitten, senior` 只属于历史实现；本轮只能
通过语义标签映射比较，不能把旧整数直接当作新概率列索引。

### 2.2 VGG128 与作者 VGG129 的来源

`scripts/run_meowagenet_formal_v2_1.py` 明确构造 `0..127` 共 128 个输入列；其历史
VGG best epoch 由 Keras row-level `val_loss` 选择。`scripts/freeze_idea088_original_roles.py`
从原作者 categorical 数据切片语义恢复出 `0..127 + mean_freq`，即 129 个输入列。

clean VGG 的 row 标签为 kitten/adult/senior = 170/460/306。`mean_freq` 在 936 rows
均有限，均值 625.893、样本 SD 254.490、范围 33.51–1973.95；这些是源字段的数值审计。
作者相关仓库的
[`vggish_inference_demo.py`（commit `6c5054b`）](https://github.com/aster-droide/audioset-thesis-work/blob/6c5054baf9731f6bd70a09eac546aae70f647bba/audioset/vggish/vggish_inference_demo.py)
把 pitch 注释为 mean F0，并注明文件名中的 pitch 由 CREPE 生成；但该版本实际调用
`extract_pitch_from_filename` 的语句被注释，当前导出的 `all_embeddings.csv` 也只含
embedding、gender、target、cat_id。因此，它提供了 `mean_freq` 的作者语义来源，却不能
单独证明本地带 `mean_freq` CSV 的确切生成版本、帧筛选/平均规则或逐 row→call 映射。
严谨名称仍应是“作者提供的 `mean_freq` 平均 F0/频率标量”或“F0/mean_freq 控制”，
不能声称已独立复原为严格的逐 call F0 测量；该 CREPE 线索也不得与本轮独立、基于 pYIN
等算法提取的 20 维声学缓存混为同一来源。

## 3. roles 与完整 OOF 审计

角色文件共有 5 repeats × 4 folds × 111 cats = 2,220 行；本轮只取 repeats 0/1/2，
共 1,332 行。repeat 0/1/2 的 outer seeds 分别为 104729、130363、155921；model base
seeds 固定为 17、43、101，训练 full seed 为
`base_seed + 10000 × repeat + 100 × fold`。

每个 cell 的 train/validation/test 猫集合两两不交，并集严格为 clean111，三侧均含三类。
每个 repeat 的四个 test 集对 111 猫恰好各覆盖一次。逐 cell 数量如下：

| repeat | fold | train cats/calls | validation cats/calls | test cats/calls |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 66 / 517 | 17 / 112 | 28 / 163 |
| 0 | 1 | 66 / 458 | 17 / 96 | 28 / 238 |
| 0 | 2 | 66 / 445 | 17 / 124 | 28 / 223 |
| 0 | 3 | 67 / 460 | 17 / 164 | 27 / 168 |
| 1 | 0 | 66 / 515 | 17 / 86 | 28 / 191 |
| 1 | 1 | 66 / 498 | 17 / 114 | 28 / 180 |
| 1 | 2 | 66 / 467 | 17 / 81 | 28 / 244 |
| 1 | 3 | 67 / 534 | 17 / 81 | 27 / 177 |
| 2 | 0 | 66 / 462 | 17 / 95 | 28 / 235 |
| 2 | 1 | 66 / 472 | 17 / 130 | 28 / 190 |
| 2 | 2 | 66 / 480 | 17 / 115 | 28 / 197 |
| 2 | 3 | 67 / 485 | 17 / 137 | 27 / 170 |

fold 0/1 的猫标签分配为 train 36/9/21、validation 10/2/5、test 16/4/8；fold 2 为
37/9/20、10/2/5、15/4/9；fold 3 为 37/10/20、10/2/5、15/3/9，顺序均为
adult/kitten/senior。不同 repeat 改变具体猫而不改变这组猫数/类别数设计。

## 4. 哪些历史材料可复用

| 历史材料 | IDEA-089 用法 | 不得复用到主表的部分 | 原因 |
| --- | --- | --- | --- |
| formal-v2.1 roles repeats 0/1/2 | **直接复用** | 无 | 已核对猫隔离和完整 OOF |
| clean data/cat manifests | **直接复用** | 无 | 身份、标签和 049A 排除边界已冻结 |
| AST `792×768` cache | **逐字节复用** | 无 | 无标签预计算，call 顺序已核对 |
| 20 维声学 cache/feature names | **逐字节复用** | 旧阶段的填补/标准化量 | cache 标签盲；预处理必须按本阶段重拟合 |
| VGG 官方 CSV | **作为原始输入复用** | 历史 scaler/weights | 新协议需训练侧重拟合 |
| 已审计结构、指标与训练代码 | **可复用实现思想/代码** | 未通过本轮 hash/测试的运行产物 | 本轮须重新冻结和预检 |
| v2.1/076/087/088 结果 | **背景表** | 权重、预测、成绩、best epoch | 协议、角色、monitor 或配方不同 |

新统一主表只能由 IDEA-089 新产生的六模型结果填充。历史最佳值应保留原日期、原协议名
和原统计单位，不能拼成一个看似同协议的总表。

## 5. 与 v2.1、IDEA-087、IDEA-088 的差异

| 方面 | formal-v2.1 | IDEA-087 | IDEA-088 | IDEA-089 |
| --- | --- | --- | --- | --- |
| 主要角色 | repeats 0/1/2，4 folds | repeat 0 outer 内再构造 3-fold inner | 原作者 5 seeds×4 folds 清洗投影，非完整 OOF | v2.1 repeats 0/1/2，4 folds |
| VGG 输入 | 128 | 非本轮配对 VGG 设计 | 129，含 `mean_freq` | 128/129 严格配对控制 |
| AST 配方 | v2.1 head/adapter 配方 | HPO/original 路径 | 原 final 风格、batch128、lr 0.0031098 | IDEA-076/087 original q00 配方 |
| epoch monitor | VGG row CE；AST call CE | animal CE，inner 规则 | training loss，无 validation | 全六臂未加权 validation 猫 CE |
| outer 机制 | train→选 epoch；train+val refit | 三 inner folds 后锁 epoch再 refit | 无 validation；直接训练/test | 每个模型/折/seed 独立选 epoch；从头 refit |
| 可进 089 主表 | 否 | 否 | 否 | 是，且必须新跑 |

IDEA-076 可作为固定 AST q00 配方来源，但它的旧 validation 结果、六个不同 seeds 和旧
预测不能代替本轮 outer-test OOF。IDEA-087 可作为猫级 CE 选 epoch 与从头固定 epoch
refit 的方法先例，但本轮不采用其跨 inner folds 的 epoch 汇总规则。IDEA-088 证明作者
风格实际使用 129 维并恢复了原角色，但其 training-loss early stop、非完整 OOF 角色和
原 final 配方均不适用于本轮主表。

## 6. 冻结训练与 refit 设计审核

### 6.1 pipelines 与参数量

六臂为 VGG128、VGG129、A0、D0 direct concat、U1、C1；其中“五核心”不含额外的
VGG128 控制。独立参数算术如下：

| AST pipeline | 参数计算 | 可训练参数 |
| --- | --- | ---: |
| A0 | `768×128+128 + BN(256) + 128×3+3` | 99,075 |
| D0 | `788×128+128 + BN(256) + 128×3+3` | 101,635 |
| U1/C1 | A0 + `20×60+60 + 60×128+128` | 108,143 |

D0 是简单输入拼接控制，不与 U1/C1 等容量；U1/C1 才是严格同结构同容量的约束对照。

### 6.2 训练、epoch 锁与 outer refit

- AST 四臂固定 Adamax lr 0.006、epsilon `1e-7`、weight decay 0、dropout
  0.44571035356880917、4 猫一批且每猫全部 calls、max 50、patience 8、CUDA AMP、
  gradient clip 1.0，并保留已有“批内类别权重和归一”的 call-balanced CE 语义。
- VGG 两臂固定 Keras 结构、Adamax lr 0.003109800273709165、epsilon `1e-7`、相同
  dropout、128 rows/batch、max 500、patience 30，无 AMP/clip，保留 Keras
  `class_weight` 的样本数分母语义。
- 六臂统一按未加权 validation 猫 CE 选 1-based best epoch；只有
  `new_ce < best_ce - 1e-6` 才算改善并清零 stale。
- best epoch 必须针对每个 pipeline×repeat×fold×seed 独立选择，不采用 IDEA-087 的
  跨 inner-fold 中位数。selection checkpoint 可留作审计，但 outer 不能加载它继续训练。
- 216 个 selection fits 全部完成、epoch lock 写盘并独立核验后，才可启动 216 个 outer
  refits。outer 在 train+validation 上从头构建模型，重新拟合预处理/类别权重，并严格训练
  已锁 epoch 数；checkpoint 与预处理锁定后才可一次读取 test 产生预测。

### 6.3 配对初始化边界

- AST 四臂公共 head 初态、猫/call 顺序和 post-build 随机流配对；D0 新增 20 个输入列
  零初始化但保持可学习，使优化前 logits 与 A0 相同。
- U1/C1 共享完整声学分支初态；需要零初始化的是声学分支最后的 **60→128 投影层**，
  不是公共 128→3 分类器。预检需同时证明初始 logits 对齐和零列/零投影参数梯度可学习。
- VGG129 的前 128 列、bias、BN、输出层与 VGG128 相同，新增 `mean_freq` 输入列权重
  零初始化且可学习，row 顺序和 dropout 流配对。这是 IDEA-089 为归因而设的初始化，
  不等于作者 129 维模型的随机初始化复现。

这些配对只支持家族内部的因果式差分解释。A0/C1 与 VGG129 使用不同原生观察行、框架和
损失分母，跨家族差值应称为统一角色下的 pipeline 比较，而非仅由某个输入因素造成。

## 7. 主表、九组预定比较与独立性

每个模型按 3 repeats×3 seeds 得到 9 个完整 111 猫 OOF 集；每个 OOF 集需先合并四折，
再计算 Accuracy、Macro-F1、Balanced Accuracy、三类 recall、CE 与 Brier。报告 9 个
分数的均值和样本 SD。禁止用逐折 F1 均值替代完整 OOF F1，也禁止先跨 seeds 平均概率
而把 ensemble 当作单模型分数。

训练前预定的九组比较为：C1−A0、C1−U1、C1−D0、U1−A0、D0−A0、U1−D0、
VGG129−VGG128、A0−VGG129、C1−VGG129。主方法比较仍是 C1−A0。U1−D0 是两种
融合方案的**整体比较**：容量、非线性和融合位置同时不同，不能把差值单独归因于“加法”
或其中一个部件；C1−U1 才是同分支、同容量下整组“RMS 相对缩放 + tanh 限幅”约束的
配对对照，但本设计也没有把 RMS 与 tanh 分别拆臂，不能声称识别二者各自贡献。本轮不设
新的机械提升门槛。

每个模型有 9×111=999 次猫预测出现，但它们来自同一 111 只猫，独立动物数仍是 111。
固定 seed 的 10,000 次年龄组内猫级分层配对 bootstrap，应在每次抽样中把同一猫索引
同时用于全部模型和全部 9 个 OOF 集；所得 2.5%/97.5% 分位数只是给定已有预测、同一
开发数据历史下的描述性区间。训练 seed 与划分波动另由 9 个 OOF 分数和配对差报告；
不得把 999 当独立样本做 t 检验或根据显著性选择性报告比较。

## 8. 是否需要单独的历史 v2.1 F0 bridge

本轮不需要增加一个“逐字 v2.1”实验。VGG128/VGG129 已在同一新协议、同一角色、同一
训练随机流和配对初始化下提供内部 `mean_freq` 控制；这是更直接的本轮 F0/mean_freq
差分。它只能回答 IDEA-089 锁定条件下额外 `mean_freq` 列的贡献，不能把所得差值解释为
历史 v2.1 约 65.25% 与其他历史分数差异的唯一原因。若未来确需解释历史差距，应另行
预注册逐字历史桥接，而不能混入本轮 432-fit 预算。

## 9. 可复核产物与门禁决定

独立脚本：`scripts/audit_idea089_baseline_protocol_design.py`。它不导入 IDEA-089 runner，
逐项锁定源文件 SHA-256，重算角色完整性、标签、VGG 维度、缓存对齐、缺失值、参数量、
历史源证据和 fit 预算。测试：`tests/test_idea089_baseline_protocol_design.py`，5 项通过。
机器可读输出：
`runs/meowagenet_idea089_unified_core_v1/independent_design_audit.json`，状态为
`PASS_DESIGN_PREFLIGHT_ONLY_NO_TRAINING_AUTHORIZATION`。

当前决定：**设计 GO，训练 NO-GO**。只有 runner/protocol、公共初始化、可学习梯度、
checkpoint 保存/重载、selection 不触碰 test、outer 从头 refit、预测身份与恢复机制均经
CPU/合成预检和独立审核后，才可申请预定 6 个首批 inner fits 的技术放行；这 6 fits 必须
计入原 432 预算，且放行依据不得包含成绩高低。
