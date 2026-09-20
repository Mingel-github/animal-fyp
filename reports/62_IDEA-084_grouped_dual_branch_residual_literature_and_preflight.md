# IDEA-084：分组双分支声学残差的文献、协议与 CPU 预检

日期：2026-09-19  
状态：协议/runner/tests 已锁定；CPU preflight GO；GPU 未授权

## 结论先行

IDEA-084 提出一个新的、固定的分组双分支残差：15 维 `source-related` 与 5 维 `spectral-energy` 各自经 `d→32→128` 支路后等权合成，同时保持 C1 的总 RMS 相对上限。该设计只是假设驱动的探索性变体，不是已有论文方法的原样复现，也不构成对 IDEA-076 的重新确认。

正式矩阵固定为 A0、原样 C1、真实分组 P1、固定哈希随机分组 R1，共 144 fits；结果后不得追加权重、宽度、分组或种子调参。

## 原始文献动机与边界

Taylor 与 Reby 的哺乳动物声源—滤波器综述指出，喉部声源及其 F0 与声道滤波后的谱包络可以承载年龄、性别等个体信息；该框架适合提出“不同声学族可能需要分开建模”的假设。原文同时讨论 source–filter interaction，因此不支持把实际代理量强行解释为完全独立的生理部件：[Taylor & Reby, 2010](https://doi.org/10.1111/j.1469-7998.2009.00661.x)。

GeMAPS 原始论文把最小声学描述符按 frequency、energy/amplitude、spectral 等类别组织，并强调参数的理论意义、自动可提取性和小型标准集合的可复用性。这支持用结构化声学族而非无约束大特征集来建立新变体：[Eyben et al., 2016](https://doi.org/10.1109/TAFFC.2015.2457417)。

但 IDEA-084 的 15/5 切分不是物理 source–filter 解耦：

- 15 维组只能称 `source-related`，其中 voicing、HNR 与 amplitude variation 都是算法或混合代理；
- 5 维组只能称 `spectral-energy`，没有 formant；
- spectral tilt 会受声源、声道滤波与录音增益/频响共同影响；
- P1 的双分支、32 宽度、固定 0.5/0.5 权重和 RMS 总预算均为本项目提出的新工程变体。

## 既有实验证据边界

IDEA-076 的 C1 平均方向为正但正式稳定性、安全性与 CE 门失败，故 C1 仍是 exploratory-positive，而非已确认方法。IDEA-082 的 G3 `real−A0` 实用门通过，但 `real−shuffled` 信息门失败，所有 `source_supported` 都为 false。IDEA-084 只把这些结果当作提出新组合对照的依据，不重写二者结论。

## 结果盲冻结设计

| 管线 | 声学结构 | 声学支路参数 | 总参数 | 角色 |
|---|---|---:|---:|---|
| A0 | 无 | 0 | 99,075 | 稳健参照 |
| C1 | 20→60→128 单支路 | 9,068 | 108,143 | 当轮配对原样 C1 |
| P1 | 15→32→128 与 5→32→128 | 9,152 | 108,227 | 新方法 |
| R1 | 哈希固定混合 15→32→128 与 5→32→128 | 9,152 | 108,227 | 容量/分组对照 |

P1 与 R1 的所有可训练参数形状、初始化顺序与固定权重完全一致；训练角色统计缓冲区随各自特征成员确定，因此数值不要求相同。两条支路末层均零初始化，四管线预优化 logits 必须逐位一致。

R1 的固定分组由 `IDEA-084-fixed-random-grouping-v1` 的逐索引 SHA-256 排名一次性生成：A=`[0,1,2,4,5,8,9,10,11,13,14,15,17,18,19]`，B=`[3,6,7,12,16]`。A、B 都含两类真实声学族，且不等于 P1；不得运行多个随机分组后择优。

## 评价与解释

三条预注册主比较为 P1−C1、P1−A0、P1−R1，C1−A0 只作同期描述性上下文。每条主比较将分别报告：

- 平均 Macro-F1、方向计数与 split 尾部；
- balanced accuracy、animal CE 与 Brier；
- senior recall 安全性；
- baseline 错误被纠正、candidate 新增错误与净纠正数。

每条比较有独立条件清单，但报告不得用一个布尔 gate 代替完整价值剖面。即使 P1−R1 通过，也只能说明该固定真实分组优于这一个结果前随机分组，不能证明完整生理因果机制。

## CPU preflight 放行标准

必须同时通过：依赖哈希、792 calls/111 cats、20 维 schema、12 个 train/validation role cell 无动物或调用泄漏、新 seed 无冲突、固定分组可重建、P1/R1 同参数同可训练初态、四管线零初始差、双支路梯度可达、总残差上限、144-fit 预算、outer-test=false、canonical resume 约束。CPU GO 仍不等于 GPU 授权。

## CPU preflight 结果

结果：**GO（仅限工程预检）**。这不代表 GPU 放行，也不代表方法有效。

- 数据：792 calls、111 cats；12 个 repeat×fold role cell 全部通过动物与调用隔离检查；outer test 未加载、未预测、未计分。
- 种子：`[3583,5080,9355]`，36 个 full seeds 唯一且与 IDEA-068 至 IDEA-082 已知 seed bank 无碰撞。
- 参数：A0 `99,075`；C1 `108,143`；P1/R1 各 `108,227`。
- 初始化：C1/P1/R1 相对 A0 的最大预优化 logit 差均为 `0.0`；共享 AST 状态一致；P1/R1 可训练初态一致；三个残差候选的末层均为零。
- 训练角色预处理：P1/R1 两分支的 median/mean/std 均与仅用当前 train calls 重算结果逐位一致；validation/test 未参与统计。
- 梯度：两个分支的零初始化输出层均可达；当末层权重进入非零探针后，两分支 hidden 层均可达。
- 总预算：饱和探针的最大相对扰动为 `0.25000006`，在浮点容差内满足单一总上限 `0.25×RMS(h)`。
- 测试：IDEA-084 专项 `13/13` 通过；连同 IDEA-071/076/082 回归测试共 `35/35` 通过。
- 运行安全：正式 run 同时要求 `--director-authorized` 与匹配当前 protocol/runner SHA 的 CPU GO。`--max-cells 1` 只运行一个完整四管线 cell，写 partial summary 后返回且不生成全量 aggregate；审查通过后才能 `--resume` 完成 144 fits。

## 锁定产物

| 产物 | SHA-256 |
|---|---|
| `configs/protocol/meowagenet_idea084_grouped_dual_branch_acoustic_residual_v1.json` | `0c4b50f2a6903bb5e55157591e5a93400c56602161fa2b1f8c8fd0d112bb7eef` |
| `scripts/run_meowagenet_idea084_grouped_dual_branch_acoustic_residual.py` | `fa3913fd7b2a7a66f3a4e61deb150ef7358cc32ee5b65eafdfe9d2c7e2abe12f` |
| `tests/test_idea084_grouped_dual_branch_acoustic_residual.py` | `d37865e4ded83079e8bd4daa2ce5789b8dd47657d40d3f2df5b20428b6a602a3` |
| `plan/IDEA-084_grouped_dual_branch_acoustic_residual.md` | `e28cb551803186e922093f05621338b31193f583c54103ab0ff549fa5784a74d` |
| `runs/meowagenet_idea084_grouped_dual_branch_acoustic_residual_v1/cpu_preflight.json` | `f7e0bfbfcd1ec0670e98664973b45deac4e6df8f9ab9fcf13d2a41e3630f09ca` |

## 当前停止点

GPU 仍未授权。下一步是总监与牛马1只读检查协议、runner、tests 和 CPU preflight；只有收到明确 GO 后才允许先运行一个四管线 cell。固定 144 fits 完成后不得追加 weight、seed、width 或 grouping 调参。
