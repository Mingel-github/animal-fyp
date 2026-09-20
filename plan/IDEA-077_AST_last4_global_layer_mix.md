# IDEA-077：AST 最后四层全局 LayerMix

## 1. 研究边界

本轮只研究 MeowAgeNet 的冻结 AST 表示本体，不接触犬数据、IDEA-076 文件或训练结果。它是同一批 792 calls、111 cats 和 formal-v2 nested roles 上的内部模块筛选，不是独立外部验证。所有层数、结构、种子、训练配方、聚合和 gate 在首次验证结果前锁定。

## 2. 代码与文献审查结论

- 标准 AST 提取链使用 128 维 log-Mel fbank、标准 10×10 stride 几何、`ASTModel.pooler_output`，再对同一 call 的多个 segment 作算术平均。
- HuggingFace AST 的 pooler 是最终 LayerNorm 后两个特殊 token（CLS 与 distillation token）的均值。仓库已有逐层缓存提取器对每个 block 输出施加同一最终 LayerNorm，再作相同双 token 和 segment 均值，因此 12 层处于可比较的 768 维几何中。
- 现有缓存为 792×12×768 float32；缓存末层与锁定 `pooler_output` 的平均绝对差为 `1.3647904e-6`、最大绝对差为 `1.8119812e-5`。这说明缓存足以支持本轮，无需新的 hidden-state GPU 提取。
- 原 AST 论文给出 12 层、768 维编码器，并用分类 token 作为音频表示；其 DeiT 初始化同时保留两个特殊 token。PETL-AST 文献显示简单线性瓶颈 adapter 对任务类型敏感，语音任务上可能过于简单；Conformer adapter 的局部卷积结构和 kernel 选择会引入额外方法与超参。Soft-MoA 又需要 adapter 数量、slot 数和位置等选择。因此本轮优先测试只有 5 个新增标量参数的全局层融合，不重试 IDEA-065 的两层瓶颈 adapter。
- 桌面目录中名为 `2026_Miron_Multi_Layer_Attentive_Probing...pdf` 的文件没有 PDF header/EOF，不能作为已核验的一手来源；不据此选择层数或 gate。

## 3. 固定表示

令 `z_l ∈ R^768` 为第 `l` 个 AST encoder block 的 call-level 表示，`l∈{9,10,11}` 使用既有逐层缓存；令 `z_12` 直接取锁定的标准 AST `pooler_output` call embedding，而不是重新提取的近似值。所有中间层都已经过 AST 最终 LayerNorm、双特殊 token 均值和 call 内 segment 均值。

固定比较三条管线：

1. `A0_final`：`r_A0 = z_12`。
2. `M0_uniform_last4`：`r_M0 = (z_9 + z_10 + z_11 + z_12) / 4`。这是无参数次要机制对照。
3. `L1_global_layermix`：

```text
w = softmax(alpha), alpha = [0, 0, 0, 0]
m = sum_l w_l z_l
g = tanh(gamma), gamma = 0
r_L1 = z_12 + g (m - z_12)
```

`alpha` 是跨所有样本共享的四个全局 logits，`gamma` 是一个共享的有符号 residual gate。初始化时 `g=0`，所以 L1 与 A0 的表示、logits 和 loss 必须逐元素相同；LayerMix 只增加 5 个训练参数。第一步允许 gate 接收梯度，LayerMix logits 在 gate 离开零点后接收梯度。这一行为在 preflight 和单元测试中显式审计。

三条管线都使用由当前 training role 的精确 `z_12` 拟合的同一类标准化方式，以及相同的 `768→128→3` ReLU、BatchNorm、dropout head。M0 不另拟合专属 scaler，避免把尺度变化混入表示对照。

## 4. 固定训练与数据边界

- 只用每个 repeat/fold 的 `train` 和 `validation` roles；不读取 `test` 预测，`outer_test_accessed=false`。
- 复用当前 MeowAgeNet A0 配方：training-role 标准化、dropout `0.44571035356880917`、Adamax、学习率 `0.006`、epsilon `1e-7`、gradient clip `1.0`、最多 50 epochs、patience 8。
- 完整 cat batching，每 batch 4 cats；损失是 training-role calls 上计算的全局 class-balanced call CE；checkpoint 由未加权 validation animal CE 最小值选择。
- 每条管线、每个 split 和 seed 的 head 使用相同初始化；模型构建后以 `full_seed + 1,000,000` 重置训练 RNG。
- 固定 seeds 为 `6917, 1398, 5934`，来自 UTF-8 字符串 `IDEA-077-AST-last4-global-layer-mix-v1` 的 SHA-256 连续大端 uint32 对 10000 取模。它们及 36 个 full seeds 均不与 IDEA-068～076 碰撞。

预算：

```text
3 pipelines × 3 base seeds × 3 repeats × 4 folds = 108 fits
```

其中主要 A0/L1 配对是 72 fits；M0 的 36 fits 只用于机制解释，不改变主 gate。

## 5. 预注册统计口径与 gate

主要口径是 9 个 `base_seed × repeat` 单元：每个单元合并四个 validation fold 的动物记录后计算 Macro-F1、balanced accuracy、CE、Brier 和 senior recall，再对九个单元等权。12 个 `repeat × fold` split-cell 先对三个 seeds 的配对差求均值。逐 fold 和重复动物 occurrence 只作描述，不视为独立样本。

`L1_global_layermix − A0_final` 主 gate 要求全部满足：

1. 九个 seed×repeat 的平均 Macro-F1 增益至少 `+0.005`；
2. 三个 base-seed 均值全部严格为正；
3. 至少 `6/9` 个 seed×repeat 严格为正；
4. 至少 `8/12` 个 split-cell 非负；
5. 最差 split-cell 不低于 `-0.03`；
6. 等权 validation animal CE 不劣于 A0；
7. 等权 validation animal Brier 不劣于 A0；
8. 每个 base seed 的 pooled senior recall 差值不低于 `-0.02`。

M0 与 A0、L1 的差值全部报告，但只作机制解释：M0 好说明多层平均本身有价值；L1 额外好才支持学习全局层权重和 gate。不得根据 M0 结果改层数、删层或改 gate。

## 6. 执行顺序

1. 锁定 plan、protocol、runner、roles、final embedding、layer cache 和依赖哈希。
2. CPU preflight 核对缓存几何、call/cat/label 顺序、末层差异、角色隔离、种子碰撞、参数量、共享 head 初态、A0/L1 初始 logits/loss 以及两阶段梯度可达性。
3. IDEA-076 正式拟合完成并获得 GPU 使用许可后才可执行 108 fits；支持逐 fit `--resume` 并校验预测哈希。
4. 完成后生成中文报告和 metadata，完整保留正、平、负结果、learned weights/gate、CE/Brier/senior 与 split 稳定性；无论结果如何不得追加结果驱动调参。
