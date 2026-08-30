# IDEA-049｜MeowAgeNet 预训练主干模型筛选计划

> 纳入日期：2026-08-29
> 来源：U 盘 `IDEA-049_Backbone_Screening_Plan.md`
> 来源 SHA-256：`49f9b04d03f16f9142403245abc58b6458437279c7ad6aece791067ea0f15824`
> 状态：SSAST、PaSST、PANNs CNN14、AVES 四个候选的 initial screening 已完成；本阶段在 AVES 后收尾

## 1. 定位

IDEA-049 是 formal-v2.1 完成后的独立探索性 backbone screening。它沿用同一套
MeowAgeNet 数据、cat-ID-disjoint split bank 和 animal-level 指标，回答公开预训练
audio pipeline 中是否存在能够稳定超过现有 AST head-only 的候选。

formal-v2.1 继续作为不可改写的历史参照。IDEA-049 使用独立协议、runner、run ID 和
结果报告，服务于 IDEA-048 的总体性能提升目标。

## 2. 历史参照

| Pipeline | Animal macro F1，mean ± SD | Balanced accuracy | QWK |
|---|---:|---:|---:|
| VGGish + MLP | 0.6525 ± 0.0462 | 0.6525 | 0.5334 |
| AST head-only | 0.7238 ± 0.0335 | 0.7597 | 0.6374 |
| Probe-guided AST adapter | 0.7290 ± 0.0428 | 0.7419 | 0.6373 |

Initial screening 只使用 base seed 17。与之配对的 AST head-only 三个 repeat macro F1
分别为 0.7182、0.7616 和 0.7212，平均 0.7337。Expanded screening 增加 seeds 43、
101 后，候选才与 formal AST 的全部九组 complete OOF 对齐。

## 3. 共同评价边界

- 分析视图：792 段唯一叫声、111 只猫；
- splits：`meowagenet_formal_v2_outer_folds.csv` 和
  `meowagenet_formal_v2_nested_roles.csv`；
- primary unit：animal；
- primary metric：animal-level macro F1；
- secondary metrics：balanced accuracy、QWK、per-class recall 和混淆矩阵；
- 同一只猫的全部叫声始终处于同一 partition；
- inner validation 选择 epoch，outer-test 只形成该候选的 exploratory OOF；
- 每段叫声产生一个 frozen call embedding，分类概率在 cat_id 内算术平均。

## 4. 统一分类 head

所有 frozen candidate 使用：

```text
embedding dimension -> 128 -> 3
```

具体结构为 feature standardization、Linear、ReLU、BatchNorm、Dropout、Linear。输入维度
随 backbone 改变，隐藏宽度固定为 128，优化器、学习率、class weighting、early stopping
和 animal aggregation 沿用 formal AST head-only。这样可以直接复用已完成的 AST 参照。

## 5. 候选顺序

1. SSAST；
2. PaSST；
3. PANNs CNN14；
4. AVES；
5. 具有适合公开预训练权重时的 Conformer。

BEATs、HTS-AT 和 Perch 保留在第二波候选池。

## 6. SSAST v1 锁定

第一候选采用 SSAST-Base-Patch-400 的 Hugging Face 转换版权重。上游定义来自
`YuanGongND/ssast`，BSD-3-Clause，官方主干 commit
`a1a3eecb94731e226308a6812f2fbf268d789caf`。官方 Dropbox 当前临时停用，实际运行锁定
`Simon-Kotchou/ssast-base-patch-audioset-16-16` revision
`f67df8e895e787bad0aec434563e8f0a1f61c794`。

- pretraining：AudioSet + LibriSpeech，joint discriminative/generative MSPM；
- architecture：base，12 blocks，768 hidden dimensions；
- input：16 kHz，128 mel bins，1024 frames；
- patch/stride：16×16 / 16×16；
- normalization：mean −4.2677393，std 4.5689974；
- pooling：Hugging Face AST `pooler_output`，即 CLS 与 distillation token 的平均；
- checkpoint SHA-256：
  `a31a7f70fea7648847882ecd278369f6f1b1cb0050364c4dff7d3c8cf8aabfe6`。

该实现被报告为“SSAST Hugging Face converted pipeline”，从而保留与原生官方实现的
来源边界。

## 7. 执行阶段

### Stage A｜资源审查

锁定论文、上游代码、license、checkpoint、revision、checksum、输入前端和 pooling。

### Stage B｜embedding cache

从 792 段音频提取一个候选专属维度的 call-level frozen embedding，保存数据顺序、标签、
cat_id、duration、模型 revision 和 checksum。

### Stage C｜smoke

运行 repeat 0、fold 0、base seed 17 的 inner train/validation 数据流；限制为两个 epoch，
检查维度、loss、animal aggregation、GPU 和日志。

### Stage D｜initial screening

运行 repeats 0–2 × folds 0–3 × base seed 17，共 12 个 fold-level fits，形成三组完整
111-cat OOF，并与同 repeat、同 seed 的 AST head-only 配对。

### Stage E｜expanded screening

候选经团队确认后增加 seeds 43 和 101，总量达到 36 fits 和九组 complete OOF。

SSAST 的三组 seed-17 OOF 已完成，macro F1 均值为 0.6292；配对 AST head-only
均值为 0.7337。SSAST 记录为 `screened_not_better`，因此本轮保留初筛证据并停止
seed 扩展。PaSST 已于 2026-08-30 获准进入第二候选执行。

## 8. PaSST v1 锁定

第二候选采用官方 `hear21passt==0.0.26` 提供的
`passt_s_swa_p16_128_ap476`。权重来自 PaSST 官方 GitHub release
`v0.0.1-audioset`，checkpoint SHA-256 为
`302903fa8c4aee817b11dc982da0b29aaf8d11a3e722420476d0a12c9db70c2c`。

- pretraining：ImageNet 初始化后进行 supervised AudioSet multi-label training，使用
  structured Patchout 与 SWA；
- input：32 kHz mono、128 mel、800-sample window、320-sample hop、1024 FFT；
- architecture：PaSST-S，16×16 patch、10×10 stride、768 hidden dimensions；
- pooling：final CLS 与 distillation token 的平均；
- output：`embed_only` 768 维，排除 527 维 AudioSet logits；
- inference Patchout：0，形成确定性 frozen embedding；
- short-call policy：保留 call 原始时长，仅将短于 160 ms 的 11 条 call 右侧补零至
  5120 samples；其余 781 条保持原始时长；
- license：Apache-2.0。

PaSST 使用独立 recipe、runner 和 run ID，继续复用同一 split bank、共享
`embedding dimension → 128 → 3` head 与 animal-level evaluation。

PaSST 的三组 seed-17 complete OOF macro F1 为 0.6319、0.6741 和 0.6619，均值
0.6560；配对 AST head-only 均值为 0.7337，平均差值为 −0.0777。PaSST 记录为
`screened_not_better`，保留完整初筛证据并停止 seed 扩展。它比 SSAST 的初筛均值高
0.0268，且 repeat 间 SD 从 0.0734 降至 0.0217。详细结果见
`reports/15_IDEA-049_PaSST_initial_screening_results.md`。

## 9. PANNs CNN14 v1 锁定

第三候选采用官方 AudioSet `Cnn14_mAP=0.431.pth`。权重来自 Zenodo record 3987831，
checkpoint SHA-256 为
`0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31`。

- pretraining：supervised AudioSet multi-label training，balanced sampling 与 mixup；
- input：32 kHz mono、64 mel、1024-sample Hann window、320-sample hop、50–14000 Hz；
- architecture：六个 convolution blocks，前五个执行 2×2 average pooling；
- pooling：frequency mean 后执行 temporal max-plus-mean，再通过 `fc1` ReLU；
- output：2048 维 frozen embedding，排除 527 维 AudioSet logits；
- short-call policy：将短于 310 ms 的 68 条 call 右侧补零至 9920 samples；其余 724 条
  保留原始时长；
- code license：MIT；checkpoint license：CC BY 4.0。

PANNs 的三组 seed-17 complete OOF macro F1 为 0.6121、0.5991 和 0.5540，均值
0.5884；配对 AST head-only 平均差值为 −0.1453。PANNs 记录为 `screened_not_better`，
保留完整初筛证据，seeds 43/101 停留在待扩展池。详细结果见
`reports/16_IDEA-049_PANNs_CNN14_initial_screening_results.md`。随后按计划执行 AVES。

## 10. AVES-base-bio v1 锁定与结果

第四候选采用官方 TorchAudio port 的 `AVES-base-bio`，运行包为 `esp-aves==1.0.0`。
checkpoint 为 377,570,545 bytes，SHA-256 为
`7a7dfaff2ea0b617cae1d82d7831e766be2a9ac00e37962a26f0a1b285be2530`。

- pretraining：AVES core 数据加 AudioSet、VGGSound 动物子集，共 360 小时；
- input：16 kHz mono，保留每条 call 的原生时长；
- architecture：12 层 HuBERT/Wav2Vec2-style encoder，768 hidden dimensions；
- pooling：最后层全部有效时间帧算术平均，与官方分类器一致；
- batch policy：每条 call 独立前向，避免 padding 改变逐帧 embedding；
- short-call policy：最短卷积输入 400 samples；本数据 792 条均超过该长度，补零数量为 0；
- license：MIT。

AVES 的三组 seed-17 complete OOF macro F1 为 0.6675、0.6649 和 0.6865，均值
0.6730，sample SD 0.0118；balanced accuracy 均值 0.7203，QWK 均值 0.6033。
它在四个新 backbone 中综合排名第一，相对 PaSST 的 macro F1 平均提高 0.0171，并与
VGGish 的 0.6795 接近。配对 AST head-only 平均差值为 −0.0607。详细结果见
`reports/17_IDEA-049_AVES_initial_screening_and_stage_closeout.md`。

## 11. 阶段收尾

四个新 backbone 的 macro F1 均值排序为 AVES 0.6730、PaSST 0.6560、SSAST 0.6292、
PANNs 0.5884。AVES 同时取得四候选最高 balanced accuracy、QWK 与最低 repeat 波动；
AST head-only 以 0.7337 继续保持最高参照。

本阶段在 AVES 后收尾，seeds 43/101、Conformer、BEATs、HTS-AT 与 Perch 进入未来工作池。
已有四项完整初筛结果保留为 IDEA-049 的阶段性证据。

## 12. 结果解释

IDEA-049 报告完整 pipeline 在共享评价边界下的性能。preprocessing、pretraining、
pooling 和 backbone 共同构成 pipeline，候选排名不延伸为纯 architecture 因果结论。
所有完成的 OOF、资源代价与失败记录均进入结果文档。
