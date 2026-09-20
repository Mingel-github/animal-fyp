# IDEA-079：AST 内部空间 patch-token Adapter

## 1. 研究边界与当前状态

本轮只研究 MeowAgeNet 的标准 AST 本体，保持 IDEA-077 的 plan、protocol、runner、tests 和结果文件完全冻结，不接触 IDEA-078 的候选方法文件。IDEA-079 当前只锁定方法、统计协议、runner、tests 与 CPU 结构预检；在 IDEA-078 发布共享 block-11 token cache 的最终 manifest 与哈希之前，正式 cache preflight 和 108 个拟合必须硬停止。当前不提取全量 cache、不占 GPU。

数据仍是 792 calls、111 cats、843 个 1.28 s 标准 AST segments 与 formal-v2 nested roles。843 多于 792 的唯一原因是 42 个较长 call 被切成 2～6 个重叠 segment；每个 call 至少一个 segment，`segment_counts.sum()==843`。cache 必须保存且逐元素核对 segment→call 映射、792 个 call 的 segment 覆盖、call ID 和源音频路径，不能把 segment 当独立叫声。

## 2. IDEA-065 只读复盘与停止重复方案的理由

IDEA-065 的真实实现不是 head adapter，而是 `scripts/run_idea019_peft_placement.py` 中的 encoder 内部 adapter：

- `ResidualBottleneckAdapter` 对完整 token 序列执行 `x + up(GELU(down(x)))`；`up.weight` 与 `up.bias` 为零。
- `ASTLayerWithAdapter` 先运行选定 encoder block，再对 `outputs[0]` 的全部 token 施加 adapter。
- `AdapterASTClassifier` 替换两个被 probe 选中的 encoder layer，之后仍执行 AST 最终 LayerNorm、CLS/蒸馏 token pooling 和 call 内 segment 均值。
- 每个 width-32 adapter 有 49,952 个参数；两个合计 99,904；连同 99,075 参数 head 共 198,979 个可训练参数。adapter LR 为 `1e-3`，head LR 为 `0.003109800273709165`。
- 12 个 repeat×fold placement 为：`[8,11]` 三次、`[10,12]` 两次、`[11,12]` 四次、`[5,7]` 一次、`[6,7]` 一次、`[8,11]` 另一次，即唯一组合 `(5,7),(6,7),(8,11),(10,12),(11,12)`；层号为 one-based。
- deterministic seed replication 的九个 seed×repeat OOF `adapter-head` Macro-F1 差为 `-0.08230,-0.00233,+0.02685,+0.03203,+0.01064,+0.00366,-0.03740,-0.02891,+0.02609`；均值 `-0.00574`，5 正 4 负。head 平均最佳 epoch `3.389`，adapter `2.556`，表现出很早的 checkpoint 选择；adapter 峰值显存均值约 605 MiB、最大约 834 MiB，而 head-only 均值约 67 MiB。

因此，“最后 block 输出后、最终 LayerNorm 前，再对每个 token 加一个普通零初始化 bottleneck adapter”在 IDEA-065 选择第 12 层的 cell 中已经实际发生；单 adapter 只是其子集而不是新方法。本轮明确停止该方案，不为得到新编号而硬跑。

## 3. 文献核验与方法选择

本地 PETL-AST 论文的结构图和方法页确认：AST block 是 Pre-LN MHSA/FFN 残差结构；常规 adapter 是 down-project、非线性、up-project，可串行或并行放在 MHSA/FFN 周围。论文同时指出简单线性 bottleneck 对 speech/audio 可能过于简单，并以含局部卷积的 Conformer adapter 引入空间归纳偏置。Soft-MoA AST 论文也把普通 adapter、Convpass 与并行 MHSA/FFN adapter 区分为不同结构族。这支持测试局部二维 patch 邻域，但不支持从同一数据里搜索 kernel、层位或宽度。

本轮固定一个候选和一个严格机制控制：

1. `A0_cached_tail`：从共享的 block-11 输出 token 直接运行冻结 block 12、最终 LayerNorm、双 special-token 均值、segment→call 均值和既定 MLP head。
2. `C1_pointwise_patch`：在 block 11 与 block 12 之间，仅对 144 个 patch tokens 施加逐位置通道混合 adapter；两个 special tokens 在注入点逐元素原样保留。
3. `S1_spatial_patch`：同一位置、同一 down/up、同一参数量、同一初始化抽样和同一训练流，但把控制的 dense `1×1` mixer 换成 depthwise `3×3` mixer，从而只新增固定的频率×时间邻域归纳偏置。

HF AST patch projection 先产生 `[B,C,12,12]`，再按 frequency-major、time-minor flatten 为 144 个 patch tokens。因此 runner 把 token 2～145 恢复为 `[B,9,12,12]`；`3×3` 邻域对应 AST 的真实 12×12 频率×时间 patch 网格，不是任意 token 排列。

## 4. 固定 Adapter 公式、参数和计算

对 block-11 输出 `H=[s_cls,s_dist,P]`，其中 `P∈R^(144×768)`：

```text
U = GELU(Down(LayerNorm_no_affine(P)))       # 768 -> 9
C1: V = Conv1x1_dense_9x9(U)
S1: V = Conv3x3_depthwise_groups9(U)
P' = P + Up(GELU(V))                         # 9 -> 768
H' = [s_cls, s_dist, P']
Z  = frozen_block12(H')
e  = mean(final_LayerNorm(Z)[:, 0:2], dim=token)
```

`Up.weight=0` 且 `Up.bias=0`。所以 C1/S1 初始化时 `P'=P`，全部 146 tokens、block-12 输出、pooler、call embedding、head logits 与 CE 都与 A0 精确相同；special tokens 在 adapter 注入点始终旁路且逐元素不变。第一步只有 Up 能获得非零梯度，Down/mixer 的梯度按设计为零；Up 离开零点后 Down/mixer 可达，CPU preflight 和单元测试同时锁定这一梯度日程。

宽度固定为 9，不作数据驱动选择。两种 mixer 的参数都恰好是 90：

```text
Down: 768×9 + 9       = 6,921
C1 mixer: 9×9×1×1+9  =    90
S1 mixer: 9×1×3×3+9  =    90
Up: 9×768 + 768       = 7,680
Adapter total         = 14,691
```

相等来自 `r²+r = 10r` 的正整数解 `r=9`。C1 与 S1 的 mixer 均有 81 个 weight 和 9 个 bias，fan-in 同为 9；在同 seed、同构建顺序下，flatten 后的初始随机数逐元素相同，只是张量形状/连接拓扑分别为 `[9,9,1,1]` 和 `[9,1,3,3]`。两者每 segment 的 down、mixer、up 合计约 2,002,320 MACs，均只有 144×9 的瓶颈激活。A0 可训练参数 99,075；C1/S1 都是 113,766。

## 5. 为何属于 AST 本体而不是 head

Adapter 位于 encoder block 11 与 12 之间，输入仍是每个 1.28 s segment 的 146-token AST 序列。它在最终 pooling 之前改变 patch tokens，随后冻结 block 12 会用这些新 token 重新计算 Q/K/V、自注意力和 FFN；patch 改动因此可经 attention 传播进 CLS 与蒸馏 token。该变换不能从已经 pooled 的 768 维 call embedding 或分类 logits 重建，也不能在 head 上等价实现。分类 head 仍是原 `768→128→3` MLP，只接收 block 12 与最终 LayerNorm 后的 call embedding。

与 IDEA-065 的非重复边界逐项固定如下：

| 维度 | IDEA-065 | IDEA-079 |
| --- | --- | --- |
| 位置 | probe 每 split 选两个 block，adapter 在各 block 输出后 | 固定且仅在 block 11→12 之间 |
| token 范围 | 全部 token，含两个 special tokens | 仅 144 patch tokens；special tokens 旁路 |
| mixer | 每 token 独立的线性 bottleneck | 候选为 12×12 上的 3×3 depthwise 空间邻域 |
| 机制控制 | head-only | 额外有同参数、同初始化、同算量的 dense 1×1 patch 控制 |
| 参数 | 两个 width-32，共 99,904 adapter 参数 | 一个 width-9，共 14,691 adapter 参数 |
| 层位选择 | inner-train probe，位置随 split 变化 | 无 probe、无层位搜索 |
| cache/在线计算 | 每 epoch 在线运行完整 AST | 只读共享 block-11 cache，在线运行 adapter+冻结 block 12 |

## 6. 共享 cache 依赖与硬审计

IDEA-079 不定义第二套 cache schema，不提取第二套 cache，也不要求 IDEA-078 使用大 NPZ。唯一允许的依赖入口是 IDEA-078 最终发布的：

`runs/ast_prelast_tokens_idea078_v1/` 下 cache manifest。

当前 protocol 只保存 `cache_manifest_path` 占位，并以 `cache_manifest_sha256=null`、`resolved=null` 硬阻断 cache preflight/formal run。IDEA-078 落盘后，只允许一次机械性 dependency amendment：填入 manifest、mmap-compatible float32 tokens、独立 index、extractor 的实际路径与 SHA-256、index key 映射，以及 manifest 已报告的 A0 重建误差；不得同时改方法、seed、gate、训练配方或容差。

最终 cache 必须满足：

- tokens 为 `(843,146,768)` float32，使用 `.npy` mmap 只读加载；裸 tensor 约 378,095,616 bytes（360.58 MiB）。
- index 不含 label/cat 字段，只含 segment→call、per-call segment counts、call IDs 和源音频路径；IDEA-079 的 labels/cat IDs 只能从锁定 fbank/roles 读取。
- index 与 `ast_fbank_128.npz` 的 `segment_call_indices`、`segment_counts`、`call_ids`、`source_paths` 逐元素相同；每个 call 有 1～6 个 segment，总和 843，42 个 call 多于一个 segment。
- 源 fbank SHA-256 为 `007d07f7c236ba44ae76a1e867cee5cc49b774ddde2f54064c68190cd01d60c7`；正式 roles 为 `87deda39808297e1af5b71283e1d7487a7b88d9288cb492c488e3e64fb91c433`。
- 原始音频由固定 commit `3d02295bef1500d2b2500a124596f77010181391`、audio tree `6a10ede379615a52441b0b80e9f4783f7b182ebf`、manifest SHA-256 `68e5131dc5d3cd611ecdda30e5176a6dcc90c0ea2500a6d8a4d9b066ce11a72f` 和 checksum-list SHA-256 `7c1b51ce1a18b1253d3099e9ce3ea034385cd1851ed121a1a5b82a50a20e6a12` 对账。
- 提取完全不读 labels/roles，保持既有 1.28 s window、0.64 s hop、128×128 fbank、stride 10×10 与 12×12 patch grid，不重新切音频。
- 用 cache token 经过冻结 block 12、最终 LayerNorm、双 special-token 均值和 segment→call 均值重建 A0；相对锁定 `ast_standard_call_embeddings.npz` 的 mean absolute error 必须不大于 `2e-6`，maximum absolute error 不大于 `2e-5`。IDEA-079 cache preflight 必须自己对 843 个 segments 重算，不能只信 manifest 中的摘要；在固定 repeat-0/fold-0 非 test probe 上，cached-tail 与 locked-A0 的共同随机 head 最大 logit 差还必须不大于 `5e-4`，且 A0/C1/S1 的 cached-tail 初始 logits 仍须逐元素相同。三条正式管线均走相同 cached-tail 路径，不把锁定 embedding 与 cached-tail 输出混用。

## 7. 固定训练、种子与统计口径

三条管线都使用当前 training role 的锁定标准 AST call embedding 计算同一类 scaler，并使用相同 `768→128→3` ReLU、BatchNorm、dropout `0.44571035356880917` head。完整 cat batching，每 batch 4 cats；call-level class-balanced CE；checkpoint 由未加权 validation animal CE 最小值选择；最多 50 epochs、patience 8、gradient clip 1.0、Adamax epsilon `1e-7`。head LR 为 `0.006`，C1/S1 adapter LR 为 `0.001`。模型构建后用 `full_seed+1,000,000` 重置训练 RNG。

base seeds 固定为 `8058,2495,2473`，来自 UTF-8 字符串 `IDEA-079-AST-spatial-patch-token-adapter-v1` 的 SHA-256 连续大端 uint32 对 10000 取模后的前三个合格值。它们及其 36 个 full seeds 排除 IDEA-065～078 的 Meow seeds，特别排除 IDEA-078 的 `9763,3230,9726` 及其 full seeds，也排除 IDEA-075 dog seeds。

预算：

```text
3 pipelines × 3 base seeds × 3 repeats × 4 folds = 108 fits
```

只读 train/validation roles，不生成 test 预测。主统计单元为 9 个 base-seed×repeat：每个单元合并四个 validation folds 的动物记录后计算指标，再九个单元等权。12 个 repeat×fold split-cell 先对三个 seeds 的配对差求均值。逐 fold 与重复动物 occurrence 只作描述，不视为独立样本。

## 8. 预注册 gate 与结论边界

`S1−A0` 候选 gate 要求全部满足：九单元平均 Macro-F1 至少 `+0.005`；至少 2/3 base-seed 均值为正；至少 6/9 seed×repeat 为正；至少 8/12 split-cell 非负；最差 split-cell 不低于 `-0.03`；平均 animal CE 与 Brier 均不劣于 A0；每个 base seed 的 pooled senior recall 差不低于 `-0.02`。

空间机制 gate 另外要求：九单元平均 `S1−C1` Macro-F1 至少 `+0.002`，且至少 5/9 seed×repeat 为正。

- 两组 gate 都通过：保留 S1 为后续独立确认候选，并允许“局部二维邻域优于等预算逐位置通道混合”的有限机制结论。
- 只通过 S1−A0、不通过 S1−C1：只能说该内部 patch adapter 容量可能有用，不能归因于空间邻域；本固定结构停止，不结果驱动搜索 kernel/宽度/层位。
- S1−A0 不通过：完整报告并停止 IDEA-079。

## 9. 执行顺序

1. 当前只运行 CPU `structural-preflight`：源哈希/segment 覆盖、真实冻结 block-12 CPU 前向、参数量、special-token 旁路、A0/C1/S1 初态等价、C1/S1 成对初始化和两阶段梯度可达性；不得读取 GPU 状态或提取全量 token cache。
2. 等 IDEA-078 发布实际 cache schema/manifest/hash；IDEA-079 只做 dependency amendment，并重新运行 tests。
3. 运行只读 `cache-preflight`，核对 mmap、label-free index、843→792 映射、12×12 geometry、源哈希、roles 隔离和 A0 重建误差。
4. 向研究总监汇报并取得 GPU 排队许可后，才允许执行 108 fits；formal runner 对 CPU formal run、未解析 cache、hash 漂移和 outer test 访问全部硬报错。
5. 完成后无论正负都保留 A0/C1/S1 全部 predictions、checkpoint audit、CE/Brier/senior、split 稳定性和 gate；不得追加结果驱动变体。
