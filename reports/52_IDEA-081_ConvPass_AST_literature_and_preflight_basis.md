# IDEA-081：ConvPass/AdaptFormer 式 AST 尾层旁路的文献依据与实现边界

日期：2026-09-18  
状态：方法锁定；等待 CPU 预检结果写入独立 JSON；未使用 GPU；未访问 outer test。

## 1. 结论先行

IDEA-081 可以在 IDEA-078 的 block-11 token cache 上忠实实现，当前方法判定为 **GO**，不需要以近似模块冒充 ConvPass。实现会完整重放冻结的 AST 第 12 层，并在同一层的两个 pre-LN 残差子层中各加入一个 ConvPass：

\[
x_1=x_0+\operatorname{MSA}(\operatorname{LN}_1(x_0))
    +s\,C_{attn}(\operatorname{LN}_1(x_0)),
\]

\[
x_2=x_1+\operatorname{MLP}(\operatorname{LN}_2(x_1))
    +s\,C_{mlp}(\operatorname{LN}_2(x_1)).
\]

这里选用原论文与官方代码所称的完整 `convpass`，而不是只并联 MSA 的 `convpass_attn`。冻结 block 12 和最终 LayerNorm 后，cache-tail 路径仍能逐算子表达上述公式。

## 2. 原始来源与核查结果

### 2.1 ConvPass

- 原论文：Jie & Deng, *Convolutional Bypasses Are Better Vision Transformer Adapters*, ECCV 2022，arXiv 2207.07039。
- 官方实现：`JieShibo/PETL-ViT`，本次核查 commit `026e4f12cfbe1e46bfb4d3ed5ba09f2d9f83e91e`。
- 原论文图 3 与公式把卷积分支并联在 MHSA 或 MLP 残差分支；完整 ConvPass 在每个 block 放两个模块。
- 官方模块为 `Linear(768,8) -> QuickGELU -> dense Conv2d(8,8,3,padding=1) -> QuickGELU -> Dropout(0.1) -> Linear(8,768)`。3×3 卷积是 dense 卷积，不是 depthwise。
- 官方代码分别还原二维 patch grid，并把 CLS 当作单独的 1×1 图通过同一个卷积；因此 AST 的 CLS 与 distillation token 也必须各自独立处理，而不能像 IDEA-079 那样旁路。
- 官方默认初始化为：down Xavier、conv 全零后在中心写入通道单位阵、up 全零。零 up 使初始旁路输出严格为零，同时保留可达的分阶段梯度。
- 原论文/代码按任务搜索或配置 scale；本实验在看结果前固定 `s=0.1`。这是官方配置中实际使用的值，也与 AdaptFormer 官方固定 scalar 0.1 一致，但不声称它是 ConvPass 唯一标准值。

### 2.2 AdaptFormer

- 原论文：Chen et al., *AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition*, NeurIPS 2022。
- 官方实现：`ShoufaChen/AdaptFormer`，本次核查 commit `6967d676c1a5e5a11be2e2768a6e5c604bb043ed`。
- AdaptFormer 把可训练 bottleneck 作为冻结 MLP 的平行分支，而不是串接在整个 Transformer block 之后；其公式为冻结 MLP、scaled adapter 与残差三者相加。
- 官方默认配置使用 parallel FFN adapter、fixed scalar 0.1、zero-initialized up projection。这支持 IDEA-081 的“冻结主干 + 平行旁路 + 零初始”的实验控制，但 V1 的具体二维卷积拓扑仍严格取自 ConvPass，而不是把 AdaptFormer 名称套在自创模块上。

### 2.3 AST

- 原论文：Gong, Chung & Glass, *AST: Audio Spectrogram Transformer*, Interspeech 2021。
- 官方 AST 以 DeiT 为骨干，使用 CLS 与 distillation 两个 special token，并平均两者得到最终表示。
- 当前锁定输入为 128×128 fbank，16×16 kernel、10×10 stride，得到 12×12、共 144 个 patch token；加两个 special token 后 cache 形状为 `[843,146,768]`。
- Hugging Face AST 的 patch projection 先把输入转成 `[batch,1,frequency,time]`，随后 `Conv2d(...).flatten(2).transpose(1,2)`，所以 `[B,144,h] -> [B,12,12,h]` 的行主序是 frequency-major、time-minor。实现不额外交换频率和时间轴。

### 2.4 桌面论文材料

本次逐页核查了下列本地原始/正式 PDF：

- `2021_Gong_AST_Audio_Spectrogram_Transformer.pdf`，SHA-256 `c237e981755e9ba5047e7290ffcb7d38d8292e166730ae3cd180b7833f184303`。
- `2024_Cappellazzo_PETL_AST.pdf`，SHA-256 `9ff59f6b4569b2202a320048a561884032fa86f4175458032b8935ec1d0b108d`。其 AST 图与正文明确展示 adapter 可平行于 MHSA，或同时平行于 MHSA 与 FFN。
- `2024_Cappellazzo_Soft_Mixture_of_Adapters_AST.pdf`，SHA-256 `345f0542e26f61da92dbb70a70632d8e6c9310a48834f70354a54366b4b9ba3b`。其图再次区分 parallel adapter 与 ConvPass 等卷积 adapter。

这些二手到 AST 场景的材料只用于确认兼容性；决定 V1 真实拓扑的是 ConvPass 原论文与官方实现。

## 3. 与 IDEA-079 的不可混淆边界

IDEA-079 是项目自定义的、block 11 与 block 12 之间的单个串行 patch-only adapter：special token 完全旁路，包含无仿射 LayerNorm，空间 mixer 是 depthwise 3×3，只放一个模块。它既不平行于 MSA/MLP，也不是 dense 3×3，不能称作 ConvPass。

IDEA-081 的 V1 则位于 block 12 内部，分别并联 MSA 与 MLP；两个 special token 各自经共享 ConvPass；每个分支使用 dense 3×3。IDEA-077 与 IDEA-079 的文件和结论保持冻结。

## 4. 固定参数与计算量

单个 width-8 ConvPass 参数量：

- down：`768×8+8 = 6,152`
- dense 3×3：`8×8×3×3+8 = 584`
- up：`8×768+768 = 6,912`
- 合计：`13,648`

完整 block-12 双分支为 `27,296` 个可训练参数。每段音频的计算量按实际 dense Conv2d 运算计：

- down：`146×768×8 = 897,024` MAC
- 12×12 patch 卷积：`144×8×8×9 = 82,944` MAC
- 两个 1×1 special 卷积（含 padding 的实际 kernel 计算）：`2×8×8×9 = 1,152` MAC
- up：`146×8×768 = 897,024` MAC
- 每个模块：`1,878,144` MAC；双分支：`3,756,288` MAC/segment。

四管线总可训练参数分别为：A0 `99,075`，C1 `108,143`，V1 `126,371`，CV1 `135,439`。

## 5. 公平性与预检要求

- A0/V1 共用完全相同的 head；V1 的两个 up projection 为零，因此初始 logits 与 loss 必须逐元素相同。
- C1/CV1 共用相同的 age branch state；V1/CV1 共用相同的两个 ConvPass state；四管线 common head state 完全相同。
- C1 的 age output 也为零，因此四管线在优化前都应与 A0 完全一致。
- CPU 预检必须验证 dense 3×3 的局部支持、special token 独立处理、参数/MAC、zero-up 梯度时序、真实 frozen block-12 重放、完整 cache 与 locked A0 的重构误差、age feature 顺序及所有 role 不变量。
- 任一拓扑、cache、state pairing 或重构检查失败即为 **NO-GO**；禁止用近似结构代替后继续。

## 6. 来源链接

- ConvPass 原论文：https://arxiv.org/abs/2207.07039
- ConvPass 官方代码：https://github.com/JieShibo/PETL-ViT
- AdaptFormer 原论文：https://papers.nips.cc/paper_files/paper/2022/hash/69e2f49ab0837b71b0e0cb7c555990f8-Abstract-Conference.html
- AdaptFormer 官方代码：https://github.com/ShoufaChen/AdaptFormer
- AST 原论文：https://arxiv.org/abs/2104.01778
- AST 官方代码：https://github.com/YuanGongND/ast
