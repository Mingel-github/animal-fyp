# IDEA-089 独立工程审计

日期：2026-09-20  
状态：**CPU preflight、全 216 个 selection 与 epoch lock、全 216 个 outer fits、54 组完整
OOF、9 项配对比较、10,000 次 bootstrap、metadata 与最终报告数值/边界独立审计全部通过**

## 1. 判定

牛马1在不修改 runner、protocol、plan、牛马2测试或 fit 产物的条件下，对冻结输入、CPU
preflight、全 216 个 selection、epoch-selection lock，以及预定首批
`repeat=0 / fold=0 / base_seed=17` 六臂各一 outer fit 做了独立复算。

判定为：

- CPU preflight：PASS；
- 首批 6 selection 工程链路：PASS；
- 全 216 selection 与 epoch-selection lock：PASS；
- 首批 6 outer 工程链路：PASS；
- 全 216 outer、机械 aggregate 与 metadata：PASS；
- 最终独立结论：`PASS_FULL_OUTER_AND_AGGREGATE_AUDIT`。

本判定只依据角色、预处理、随机顺序、early stopping、checkpoint、预测聚合和文件完整性，
没有用 validation 或 test 的 Accuracy、Macro-F1、CE 或任一模型的相对成绩决定是否续跑。

## 2. 冻结身份

| 产物 | SHA-256 |
| --- | --- |
| protocol | `651a568ae48b89c9966345c0293572bbfed02e8a236a472a9fe12df45dff3195` |
| runner | `44f28e1bc809035c4f6793bc250c3d6566f2944c6c3dc6c05378df4d69328e70` |
| 牛马2 tests | `589df957618d3bab3e9bd922c140b3af4a6f4776e14c94a6ccdadf2f3946b605` |
| plan | `29402baed28b28491e913414096590ec7e765d55a509df73a5d4731ea426692b` |
| 报告84设计审计 | `cb9809b50054f8d81672e2e2ae4df1a0e384590156aea2bfc360f85dfac2cf08` |
| CPU preflight | `f667ad7233c98974bc5fde0a5e03e776735f2f17ba7eee9b182f80d7ccc80c90` |

protocol 中的 12 个依赖 hash 由独立 verifier 逐文件重算后全部一致。牛马1独立设计测试
5 项与牛马2 runner 测试 12 项合并为 17 项，独立复跑全部通过；10 条 warning 均为现有
Python/pandas 的弃用提示，不是数值、角色或训练失败。

## 3. CPU preflight 独立复核

preflight 明确记录 `training_started=false`、`outer_test_accessed=false`，没有真实 fit。
以下项目经独立读取并复核：

- 12 个 repeat/fold cells 均有预期 train/validation/test 猫数；
- 总预算为 216 selection + 216 outer = 432 physical fits；
- VGG128/VGG129 可训练参数为 17,155/17,283，含 BN moving state 的总参数为
  17,411/17,539；A0/D0/U1/C1 可训练参数为 99,075/101,635/108,143/108,143；
- A0/U1/C1 公共头 state 精确一致，U1/C1 完整初态精确一致；
- D0 前 768 列、bias、BN、输出与 A0 配对，新增 20 列为零；
- D0、U1、C1 的新增声学参数以及 VGG129 的 `mean_freq` 输入列均收到有限、非零梯度；
- D0−A0 优化前 logit 最大绝对差为 `1.9371509552e-7`，低于预定 `1e-6` 容差；
  U1−A0 与 C1−A0 为 0，VGG129−VGG128 为 0；
- VGG 配对检查使用各自训练侧标准化后的真实输入，不是只在未标准化合成输入上比较。

当前 Windows/TensorFlow 2.15 环境实测无可见 TensorFlow GPU，VGG 实际在 CPU 执行；
AST 正式 selection 可按授权使用 CUDA。此处应在最终环境记录中分开表述 VGG 与 AST
设备，不能把单一 `device` 字段误读为六臂共同硬件。

## 4. 首批 6 fits 的独立复算方法

独立 verifier `scripts/verify_idea089_unified_core_results.py` 不导入 IDEA-089 runner，而是
从 protocol、roles、VGG CSV、AST cache 和 20 维声学 cache 重新完成以下检查：

1. 验证 `initial-six-selection` 授权记录只对应固定 r0/f0/seed17 六臂；
2. 重建 VGG row 与 AST call 的 train/validation indices；
3. 重算训练角色猫 ID hash、unit index hash、类别计数和类别权重；
4. 从当前 train 侧独立重算 VGG/AST 均值和标准差，以及 D0/U1/C1 声学中位数填补、
   均值和标准差，并与保存的 NPZ 逐数组精确比较；
5. 逐 epoch 重放 VGG `default_rng(training_seed+epoch)` row 顺序；用独立
   `torch.Generator(full_seed)` 重放 AST 猫顺序及 call coverage hash；
6. 从 history 按 `new_CE < best_CE - 1e-6` 独立恢复 best epoch 和 stale，核对 patience
   与 maximum epoch；
7. 从 validation unit CSV 重新做逐猫概率算术平均，核对猫身份、unit 数、预测类别、概率
   归一化，并重算 Accuracy、Macro-F1、Balanced Accuracy、三类 recall、CE、Brier；
8. 重算每个 fit 引用的 weights、preprocessing、history、unit prediction、cat prediction
   共 5 个产物 SHA，核对 checkpoint state/prediction reload 审计和 test 访问计数。

## 5. 首批工程结果

| pipeline | train 单位 | validation 单位 | best/stopped epoch | reload 最大概率差 |
| --- | ---: | ---: | ---: | ---: |
| VGG128 | 66 猫 / 597 rows | 17 猫 / 137 rows | 47 / 77 | 0 |
| VGG129 | 66 猫 / 597 rows | 17 猫 / 137 rows | 47 / 77 | 0 |
| A0 | 66 猫 / 517 calls | 17 猫 / 112 calls | 16 / 24 | 0 |
| D0 | 66 猫 / 517 calls | 17 猫 / 112 calls | 16 / 24 | 0 |
| U1 | 66 猫 / 517 calls | 17 猫 / 112 calls | 11 / 19 | 0 |
| C1 | 66 猫 / 517 calls | 17 猫 / 112 calls | 11 / 19 | 0 |

六个 fit 均满足：

- `outer_test_accessed=false` 且 `test_prediction_calls=0`；
- 训练角色和训练侧预处理与独立重算逐项一致；
- VGG 两臂 77 epochs 的 row-order hash 全部一致并符合冻结 seed 规则；
- AST 四臂共同的前 19 epochs 猫顺序与 call coverage 完全一致；继续训练到第 24 epoch
  的 A0/D0 仍分别符合相同确定性规则；
- VGG 的停止点与 best epoch 相差 patience 30，AST 相差 patience 8；独立严格改善重放
  恢复出完全相同的 best/stopped epochs；
- 六臂各 5 个引用产物的文件 SHA 全部匹配；checkpoint state reload 通过，重载前后
  validation 概率最大差均为 0；
- unit→cat 概率平均、17 猫身份和所有保存指标与独立重算一致。

这里故意不展示首批 validation 分数，避免把工程放行误解成结果筛选。

## 6. 全 216 selection 与 epoch lock

全 216 个 selection fits 完成后，独立 verifier 对每个 fit 重复第 4 节所列的身份、训练侧
预处理、逐 epoch 顺序、严格 early stopping、unit→cat 聚合、指标与 5 个引用产物 hash
检查；合计核对 1,080 个 fit 产物 hash，未发现 test 访问。随后对 216 条 lock entries
逐一核对 selection summary 路径与 hash、完整 seed、best epoch、weights/preprocessing/
unit/cat prediction hash，坐标集合与冻结的 6 pipelines × 3 repeats × 4 folds × 3 seeds
完全相同。

epoch-selection lock SHA-256 为
`15c052ed953a4482dac3968f30bca78139cf570868ec2fe7b4383f0566ede2a4`。各 pipeline
锁定 epoch 范围为：VGG128 `20–128`、VGG129 `23–118`、A0 `1–29`、D0 `1–30`、
U1 `1–44`、C1 `1–30`。这些是工程身份信息，不是放行所依据的成绩比较。

## 7. 首批 6 outer 独立复核

在总监仅放行 `initial-six-outer` 后，六臂固定执行 r0/f0/seed17；首次完成 6 个 fit，随后
`--resume` 为 0 new / 6 validated。独立复核确认：

- 六个 outer fit 都将 train+validation 合并为 83 猫，并只在该训练侧重新拟合预处理；
- refit epoch 与 lock 完全相同，依次为 `47/47/16/16/11/11`；训练 history 长度与之
  一致，没有 validation 指标或 early-stopping 分支；
- 从头构建的 seed、post-build seed、参数量和每 epoch 确定性训练顺序均符合协议；
- checkpoint 在 test prediction 前锁定，锁前 test 调用数为 0；每个 fit 锁定后只调用
  test prediction 1 次；
- 六个 test fold 均是冻结的 28 猫；VGG 为 202 rows，AST 为 163 calls；unit→cat
  算术平均、保存指标与猫身份都能独立重算；
- 保存权重的 state digest、重载状态、预处理、history、checkpoint、unit/cat prediction
  共 36 个引用产物 hash 全部一致。

该阶段只作工程门禁，续跑建议不读取、不比较首批 test 分数。

## 8. 全量 OOF、配对与 bootstrap 结果复算

全量 `--resume` 门禁返回 0 new / 216 validated。独立 verifier 逐一复核 216 个 outer fit，
并从各折猫级预测重新拼接 54 组完整 OOF。每组恰有 111 个不重复猫 ID；每个 pipeline
有 9 组 OOF，即 999 次预测出现，但独立动物仍只有 111 只。

下表为 9 组完整 OOF 的均值 ± 样本标准差，所有数值均由独立 verifier 从预测文件重算：

| pipeline | Accuracy | Macro-F1 | Balanced Accuracy | CE | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| VGG128 | 0.670671 ± 0.036348 | 0.663676 ± 0.043201 | 0.668372 ± 0.049496 | 0.730196 ± 0.021048 | 0.446392 ± 0.012870 |
| VGG129 | 0.700701 ± 0.039470 | 0.688178 ± 0.044970 | 0.691213 ± 0.057360 | **0.681879 ± 0.036777** | 0.420272 ± 0.025560 |
| A0 | 0.721722 ± 0.038074 | 0.728678 ± 0.036318 | 0.740893 ± 0.026280 | 0.735894 ± 0.067882 | 0.414372 ± 0.037848 |
| D0 | 0.724725 ± 0.032828 | 0.732954 ± 0.033531 | 0.744953 ± 0.026576 | 0.729135 ± 0.064456 | 0.411093 ± 0.034089 |
| U1 | **0.738739 ± 0.039269** | **0.744721 ± 0.037941** | **0.750064 ± 0.026696** | 0.730326 ± 0.073617 | **0.400832 ± 0.038420** |
| C1 | 0.729730 ± 0.046377 | 0.734879 ± 0.047258 | 0.743704 ± 0.036581 | 0.726186 ± 0.053306 | 0.403834 ± 0.032053 |

Macro-F1 的 9 组配对差与共享猫分层 bootstrap 区间如下；符号数按正/平/负报告：

| 比较（左−右） | 均值 ± SD | 符号数 | 95% 描述性区间 |
| --- | ---: | ---: | ---: |
| C1−A0 | +0.006201 ± 0.018726 | 5/0/4 | [-0.006331, +0.018932] |
| C1−U1 | -0.009842 ± 0.025785 | 3/0/6 | [-0.025756, +0.005816] |
| C1−D0 | +0.001925 ± 0.019897 | 5/0/4 | [-0.011355, +0.014977] |
| U1−A0 | +0.016043 ± 0.023623 | 7/0/2 | [-0.000192, +0.033062] |
| D0−A0 | +0.004276 ± 0.011739 | 5/0/4 | [-0.009463, +0.018122] |
| U1−D0 | +0.011767 ± 0.021310 | 7/0/2 | [-0.004228, +0.027680] |
| VGG129−VGG128 | +0.024503 ± 0.024988 | 7/0/2 | [-0.007029, +0.057833] |
| A0−VGG129 | +0.040500 ± 0.036475 | 8/0/1 | [-0.023650, +0.107822] |
| C1−VGG129 | +0.046701 ± 0.041682 | 8/0/1 | [-0.016096, +0.112556] |

U1 的平均 Macro-F1 最高；C1 相对 A0 仍是小幅正差，但仅 5/9 为正，描述性区间跨 0。
所有九个 Macro-F1 配对区间都跨 0，不能宣称稳定优势。本轮无正向差异门槛，也未据结果
追加实验。U1−D0 同时包含容量、非线性和融合位置差异，只能解释为整套融合设计比较；
C1−U1 是同容量下 RMS 缩放与 tanh 约束的整体比较，不能拆成单一因素效应。bootstrap
区间只条件于这 111 只历史复用猫、当前预测和既往开发过程，不是外部确认。

独立审计还精确复算了共享的 10,000 次、seed `20260920`、按年龄类别分层的猫 ID 抽样，
九项比较的全部指标区间均与 aggregate 一致；metadata 除 aggregate 路径与 hash 外，内容
与 aggregate 完全相同。

## 9. 独立产物与结论边界

- 独立 verifier：`scripts/verify_idea089_unified_core_results.py`；
- verifier tests：`tests/test_verify_idea089_unified_core_results.py`，9 项通过；
- 独立 preflight JSON：
  `runs/meowagenet_idea089_unified_core_v1/independent_preflight_audit.json`，SHA-256
  `0f6fb0d34156b4d799e66fb4f968788d3a20fc083793dde808621f6ac0cccb26`；
- 首批独立审计 JSON：
  `runs/meowagenet_idea089_unified_core_v1/independent_first_six_selection_audit.json`，
  SHA-256 `3bd4db1c4829826e6a7941056fe60d7485323b33c9741703769af74dd875a192`。
- 全 selection/lock 独立审计 JSON：
  `runs/meowagenet_idea089_unified_core_v1/selection/independent_selection_lock_audit.json`，
  SHA-256 `e519a03b834c5bf9717b78cb24f6208f13e49d109d24734b8ac0c9224a44c9ff`；
- 首批 outer 独立审计 JSON：
  `runs/meowagenet_idea089_unified_core_v1/outer/independent_first_six_outer_audit.json`，
  SHA-256 `559eab2cb2874a6ce4241df712ee6dca3fb8549671114a4e744068020d4be2ba`。
- aggregate：`runs/meowagenet_idea089_unified_core_v1/idea089_aggregate_results.json`，
  SHA-256 `e4a2f88a90ace6fdbf7e6cbb91e0b5f047e25677349b24952eb68eac014be884`；
- metadata：`metadata/experiments/meowagenet_idea089_unified_core_v1_results.json`，
  SHA-256 `66ecb307b0a043add8997b6d1a6dcd636cc96266d22affe5324b09036993b8da`；
- 全量独立审计 JSON：
  `runs/meowagenet_idea089_unified_core_v1/independent_full_audit.json`，SHA-256
  `72df9d25d0f9ded28e2068409cc80faae46cf20c9d9c17c551da2c3fc2010adf`；
- 独立 verifier/test SHA-256 分别为
  `b0fa08340cb2a3a4128809af4341ce0ad4a6f024f24ba1855ad4ff713785ecdd` 与
  `cdc1a86d55426a4e3b4a7ba21517ca95e495ef911cd83de54f30bef107e37d28`，专项测试 9 项通过。
- 最终结果报告：`reports/87_IDEA-089_unified_core_results.md`，SHA-256
  `ec9fd759bb43b2bde9eee3b51d558b2d3f1cb9cfd9e19ec6d869259c29af51f9`；六模型主表、
  类别 recall、九项比较的均值/样本 SD/符号数/bootstrap 区间、训练时间、关键 hash 与
  本报告独立复算一致，且统计与来源边界均已核对。

本结果只支持在历史复用的 clean111 猫上的统一内部比较，不支持新动物或外部泛化主张。
VGG129 的 `mean_freq` 只表述为作者提供的频率标量；现有材料未独立恢复其严格 F0 提取实现
或逐 row 音频映射，不能把它与本项目 20 维 pYIN-derived 声学 cache 视为同源特征。
