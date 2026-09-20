# IDEA-086：排列不变声学集合残差

日期：2026-09-19  
状态：结果前锁定；仅授权 CPU 实现与预检，GPU 未授权。

## 问题

IDEA-085 的 T1 局部时序残差未超过 A0 或联合帧乱序 J1，T1 与 J1 的 Macro-F1 近似相同。IDEA-086 因此检验一个更窄的问题：把六条声学轨迹及六条 finite indicators 当作无序帧集合，能否通过逐帧共享映射与 masked mean 获得比 A0 或同期 C1 更稳定的分类收益。

本轮复用 IDEA-085 的同 seed A0/C1/T1/J1 预测作为只读配对参照，只新增训练 36 个 SET1 fit。由于候选与门槛是在已知 IDEA-085 结果后提出，本轮是探索筛查，不是独立确认。

## 锁定模型

- 候选：`SET1_acoustic_set_residual`。
- AST 主干、共同分类头、训练规则及残差上限与 IDEA-085 相同；不叠加 C1 分支。
- 输入：IDEA-085 的 6 条原始轨迹与 6 条 finite indicators；只用训练角色拟合 median/mean/std，padding 不参与池化，lookup 只用于检索 ragged 序列。
- 编码器：逐帧 `Linear(12,48)+GELU -> Linear(48,48)+GELU -> masked mean -> zero-init Linear(48,128)`。
- 明确禁止：时间卷积、位置编码、attention、variance pooling、逐叫声中心化。
- 残差：`h' = h + 0.25 * stopgrad(RMS(h)) * tanh(r_set)`。
- 分支参数 9,248；总可训练参数 108,323，比 T1/J1 少 32。
- 因逐帧映射共享且只做 masked mean，模型应对任意整帧联合排列保持不变；六通道值与 finite indicator 必须始终同行移动。

## 运行矩阵与只读复用

- Base seeds：`2713, 5395, 5226`；repeats：`0,1,2`；folds：`0,1,2,3`。
- 新训练：`3 × 3 × 4 = 36 fits`。
- 只读参照：IDEA-085 的 A0/C1/T1/J1 共 144 fit summaries 与 288 份预测文件。
- 复用前必须核对 IDEA-085 protocol、runner、summary、manifest、fit identity、角色、标签与预测哈希；任何不兼容均 fail closed，不重训旧模型。
- 必须证明 SET1 与 IDEA-085 common AST head 初态、训练角色预处理以及 post-build RNG/cat batch/call coverage 兼容。

## 评价

主分类 screen 分别为 `SET1−A0` 与 `SET1−C1`。每条独立要求：平均 ΔMacro-F1 `>=0.005`、至少 `2/3` base-seed 均值为正、至少 `6/9` seed×repeat 为正、至少 `8/12` split-cell 非负、最差 split `>=−0.03`。

`SET1−T1` 与 `SET1−J1` 为预设辅助比较，不设分类 gate；不存在单一全局 gate。四条比较均报告 Accuracy、Macro-F1、BA、kitten/adult/senior recall、CE、Brier、纠错总账与 3/9/12 稳定性。CE/Brier、BA 与 recalls 不进入主分类 gate。

## 门禁

CPU 预检必须覆盖参数量、zero-init、梯度可达、残差 cap、padding、lookup 不入模、无 outer、只读复用完整性和 resume fail-closed。排列不变性不能只看 zero-init logits：必须在非零随机逐帧 encoder 上比较原序、逆序、IDEA-085 J1 固定联合排列的 context，并在非零 projection 探针下比较 residual 与最终 eval logits；覆盖真实与随机输入及混合长度 padding，按声明浮点容差判断。首个已训练 SET1 fit 还须再次记录实际排列不变性。

CPU GO 只允许请求总监的首 fit GPU 授权，不等于正式训练授权。不得访问 outer-test，不得修改 IDEA-085、v3、提交或推送。
