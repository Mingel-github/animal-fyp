# IDEA-054｜LayerNorm、SSF 与 BitFit 受约束 AST 校准

## 1. Idea claim

MeowAgeNet 只有 111 只猫，完整或 last-2 fine-tuning 会同时移动大量 AST 参数。若年龄线索
已经存在于 AudioSet 预训练表示中，更新归一化、逐维尺度/平移或 bias 可能比增加较大的
adapter/LoRA 分支更稳定地把表示调整到 kitten、adult、senior 三分类任务。

本轮问题是：

> 在复用当前 tuned frozen-AST head、animal-ID-disjoint splits、修正后的全局 class-balanced
> call loss 和 animal-level checkpoint selection 时，LayerNorm、block-output SSF 或 BitFit
> 能否稳定提高 111-cat complete-OOF animal macro F1？

## 2. 与已有实验的关系

- formal-v2.1、IDEA-019 和 IDEA-050 继续作为历史 adapter / LoRA / fine-tuning 证据；
- IDEA-051/052/053 已三次复现 tuned frozen-AST reference：三个 repeat 的 macro F1 为
  `0.7659、0.7565、0.7487`，平均 0.7570；
- 本轮只改变 AST backbone 中允许更新的参数集合，分类头、损失、splits、epoch selection
  和猫级聚合保持一致；
- 当前实验属于 post-formal exploratory method screening，不改写既有 formal-v2.1 文件。

## 3. 四条 pipeline

| Pipeline | AST 中可训练部分 | AST 适配参数 | 研究问题 |
| --- | --- | ---: | --- |
| A0 tuned frozen AST | 无；使用匹配的 frozen call embedding | 0 | 当前强参考 |
| P1 LayerNorm tuning | 12 个 blocks 的 24 个 LayerNorm，加最终 LayerNorm；weight 与 bias | 38,400 | 只调整各层归一化尺度和中心是否足够 |
| P2 block-output SSF | 每个 block 输出后的 768 维 scale 与 shift，共 12 组 | 18,432 | 逐层特征重标定能否保留主干并改变任务边界 |
| P3 BitFit | AST 内全部 98 个现有 bias tensors | 102,912 | 只平移现有神经元响应是否比结构化新分支稳定 |

四条 pipeline 均训练同一个 `768 → 128 → 3` head，共 99,075 个 head 参数。P1 和 P3
直接从 pretrained 参数开始；P2 的 scale 初始化为 1、shift 初始化为 0，因此三种适配模型
在训练前都与同一 online frozen AST 表示一致。

SSF 在本轮特指 **block-output scale and shift**：

`h_l' = gamma_l ⊙ h_l + beta_l`

其中每个 Transformer block 有一组 768 维 `gamma_l` 和 `beta_l`。它不替换 attention 或
feed-forward，只重新标定该层输出的每个特征维度。

## 4. 固定训练条件

- checkpoint：`MIT/ast-finetuned-audioset-10-10-0.4593` 的仓库锁定 revision；
- standard AST geometry 与现有 frozen embeddings 相同；
- head dropout：0.44571035356880917；
- head learning rate：0.006；
- adaptation learning rate：0.0003；
- optimizer：Adamax，epsilon `1e-7`；
- 8-call micro-batch，4-step gradient accumulation，固定 32-call loss denominator；
- 每个 outer fold 的 class weights 只由对应训练 calls 计算；
- maximum 50 epochs，patience 8；
- inner-validation 的 unweighted animal-level cross-entropy 选择 outer retrain epoch；
- 一只猫的全部 call probabilities 取算术平均，得到猫级预测。

首轮统一 adaptation learning rate，重点比较“更新哪里”。若某一方法出现明确正信号，后续
再围绕该 family 做小范围 learning-rate refinement。

## 5. Smoke 与实现审计

正式评价前在 `repeat 0 / fold 0 / seed 17` 完成 inner-only smoke：

1. 核对 792 calls、111 cats、fbank 与 frozen embeddings 的顺序和 hash；
2. 核对 P1/P2/P3 的 trainable parameter allowlist 和准确数量；
3. 比较 cached frozen embedding 与 online frozen AST 的输出；
4. 比较 P1/P2/P3 与 online frozen AST 的初始 logits；
5. 训练两个 epoch，确认适配参数与 head 参数获得更新；
6. 保存并重载 trainable state，核对预测；
7. smoke 全程不请求 outer-test indices。

## 6. 初始评价

- base seed：17；
- repeats：0、1、2；
- 每个 repeat：4 个 animal-ID-disjoint outer folds；
- pipelines：A0、P1、P2、P3；
- 规模：每条 pipeline 12 个 outer fits，共 48 个 pipeline-fold fits；
- 输出：12 份 complete OOF 评价，即每条 pipeline 3 份 × 111 cats；
- primary metric：animal macro F1；
- supporting metrics：balanced accuracy、QWK、plain accuracy、分类别
  precision/recall/F1、混淆矩阵、paired prediction changes、参数量、时间与显存。

P1/P2/P3 各自与同 repeat 的 A0 配对比较。不同 PEFT family 之间的横向排序属于支持信息，
主要结论聚焦于每种受约束校准相对 frozen AST 的增量。

## 7. 预期与竞争解释

### 主要假设

至少一种受约束校准方式能在三个 repeat 中稳定调整 AST 的年龄判别边界，并以很少的 AST
参数达到平均 macro-F1 正增量。

### 竞争解释

1. frozen AST 表示加 tuned head 已经提取了当前数据可稳定学习的大部分信息，额外校准主要
   增加 split sensitivity；
2. LayerNorm 和 bias 更新会共同影响大量下游激活，小数据中的方向估计仍会波动；
3. SSF 的逐层逐维参数容量较小，但 12 层同时调整可能积累偏移；
4. 某种方法改善 macro F1，同时重新分配 adult/senior 边界，使 QWK 或类别 recall 呈现
   不同优势。

## 8. 扩展条件与阶段决策

任一 candidate 满足以下条件时，再把 A0 与该 candidate 扩展至 base seeds 43/101：

- 三个 repeat 的平均 macro-F1 增量至少 `+0.005`；
- 至少 2/3 repeats 为正；
- balanced accuracy、QWK、worst-repeat macro F1 或可解释的类别 recall 至少一项支持。

达到 `+0.01` 且三次方向一致时记为强信号。首轮未达到扩展条件时，本轮仍形成“更新位置与
容量”的 PEFT 对照，workflow 随后进入 checkpoint averaging / calibration。某一 family
出现局部正信号时，可在后续独立 idea 中开展小范围 learning-rate 或 layer-scope refinement。

## 9. 预期产物

- `configs/protocol/meowagenet_idea054_constrained_ast_calibration_v1.json`；
- `scripts/run_meowagenet_idea054_constrained_ast_calibration.py`；
- `tests/test_idea054_constrained_ast_calibration_runner.py`；
- `runs/meowagenet_idea054_constrained_ast_calibration_v1/`；
- `metadata/experiments/meowagenet_idea054_constrained_ast_calibration_v1_results.json`；
- `reports/27_IDEA-054_constrained_AST_calibration_results.md`。
