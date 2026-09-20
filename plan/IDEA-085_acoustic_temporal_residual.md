# IDEA-085：声学时序残差

日期：2026-09-19  
状态：结果盲工程锁定；等待逐帧缓存与 CPU 预检，GPU 未授权。

## 问题

IDEA-068/071/084 使用的 20 维声学摘要包含部分斜率与波动统计，但丢失逐帧局部组织。IDEA-085 以最小两层 1-D CNN 检验：冻结 AST 表示之上，原始逐帧声学局部动态能否优于同期 C1 摘要分支，并区分真实顺序与一次结果盲的逐叫声联合帧打乱。

这是一项受既有结果启发的新探索，不是独立外部确认，也不重写 IDEA-076、IDEA-082、IDEA-084 或 v3。两层 `k=5`、`k=3` CNN 的理论局部感受野为 7 帧；10 ms hop 下中心跨度约 60 ms，而每个输入声学帧本身使用 64 ms 窗。masked mean 汇总局部动态，不能表述为已建模整段叫声的长程轮廓或阶段先后关系。

## 冻结矩阵

| 管线 | 结构 | 角色 |
|---|---|---|
| A0 | 原样 AST-only | 稳健参照 |
| C1 | 原样同期 20→60→128 bounded residual | 摘要特征方法参照 |
| T1 | 真实帧顺序的时序 CNN residual | 候选方法 |
| J1 | 每个 call 固定联合帧排列、与 T1 同结构同初态 | 顺序对照 |

固定 3 个新 base seeds `[2713, 5395, 5226]` × 3 repeats × 4 folds × 4 pipelines = 144 fits。三条主比较分别为 T1−C1（方法增量）、T1−A0（实用效用）和 T1−J1（真实顺序相对固定打乱）。C1−A0 只作同期描述背景。

## 逐帧输入

原始通道固定为 `log_f0`、`voiced_probability`、`periodicity`、`log_rms`、`spectral_tilt`、`spectral_flatness`。每通道添加一个 finite indicator，共 12 维输入。

- 保留 native 10 ms 网格，不裁剪、不重采样、不删除 F0 缺失帧；padding mask 与 finite indicators 分离。
- 当前 train 角色内逐通道计算 finite median；以 median 填充后计算 mean/std。验证与 test 不参与统计。
- 不逐 call 去均值，不输入 call ID/index/hash 或原时间 index。call index 仅允许作为轨迹查找键。
- C1 仍使用原 20 维缓存及原样预处理，不能用时序统计替换。

## 时序编码器与边界

`Conv1d(12,32,k=5,pad=2) → GELU → Conv1d(32,32,k=3,pad=1) → GELU → masked mean → Linear(32,128)`。每层 GELU 后把 padding 位置清零，避免 padding 激活回灌；末层 Linear 权重和偏置均零初始化。

时序支路参数为 9,280，总参数为 108,355；C1 声学支路 9,068、总参数 108,143。融合固定为：

`h' = h + 0.25 × stopgrad(sqrt(mean(h²)+1e-8)) × tanh(r_seq)`。

T1/J1 只替换声学编码器，不叠加 C1 分支，不重训 AST 骨干。

## 固定联合帧打乱

J1 使用 material `IDEA-085-joint-frame-permutation-v1` 与完整 call ID 派生一个确定性排列。同一帧的 6 个通道及其 6 个 finite indicators整体移动，保持跨通道关系、缺失元组、长度和数值多重集。训练和验证都使用 J1；padding 不参与，不扫描或挑选多个排列。

T1−J1 检验的顺序信息包括 F0 缺失、voicing、周期性和谱能量代理的共同时间组织，不能简化为“纯 F0 顺序”。

## 评价与停止规则

9 个 seed×repeat 为主要等权汇总单位，12 个 repeat×fold split 描述稳定性。猫级 Macro-F1 为主，同时报告普通 accuracy、balanced accuracy、kitten/adult/senior recall、animal CE、Brier 与方向性纠错。

每条主比较单独报告分类证据：平均 Macro-F1 差至少 `+0.005`、至少 2/3 base seeds 正向、至少 6/9 seed×repeat 正向、至少 8/12 split 非负、最差 split 不低于 `−0.03`。CE、Brier、BA 和各类 recall 是兼容性/安全性剖面，不纳入分类 gate，也不能因轻微变差抹掉 F1 正向结果。不得只给单一 global pass。

本轮先完成 CPU cache/角色/参数/初始化/梯度/mask/shuffle/断点安全预检，再由独立审计决定是否向总监申请首个 GPU cell。结果后不追加权重、宽度、排列、种子或门槛搜索。

