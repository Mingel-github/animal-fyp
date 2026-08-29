# IDEA-049｜SSAST initial screening 结果报告

> 完成日期：2026-08-29
> 实验状态：`complete_for_initial_screening`
> 候选状态：`screened_not_better`
> 评价单位：animal-level complete OOF，111 只猫

## 1. 本轮完成范围

本轮完成 SSAST 的资源审查、792 条 call-level frozen embedding 缓存、inner-only smoke
test，以及 repeat 0–2 × outer fold 0–3 × base seed 17 的 12 个 fit。三个 repeat 均形成
覆盖 111 只猫的 complete OOF 预测。

IDEA-049 使用独立协议、recipe、runner 和 run 目录，复用 formal-v2.1 的 cat-ID-disjoint
split bank 与 animal-level evaluation。formal-v2.1 文件保持原样。

## 2. SSAST 资源审查与运行身份

| 项目 | 锁定内容 |
|---|---|
| 官方论文 | SSAST，AAAI 2022，<https://arxiv.org/abs/2110.09784> |
| 官方代码 | `YuanGongND/ssast`，commit `a1a3eecb94731e226308a6812f2fbf268d789caf`，<https://github.com/YuanGongND/ssast> |
| License | BSD-3-Clause，<https://github.com/YuanGongND/ssast/blob/main/LICENSE> |
| 官方推荐模型 | SSAST-Base-Patch-400，AudioSet + LibriSpeech，joint discriminative/generative MSPM |
| 实际 checkpoint | `Simon-Kotchou/ssast-base-patch-audioset-16-16`，revision `f67df8e895e787bad0aec434563e8f0a1f61c794`，<https://huggingface.co/Simon-Kotchou/ssast-base-patch-audioset-16-16> |
| Checkpoint SHA-256 | `a31a7f70fea7648847882ecd278369f6f1b1cb0050364c4dff7d3c8cf8aabfe6` |
| 输入规范 | 16 kHz；128 mel bins；1024 frames；mean −4.2677393；std 4.5689974 |
| 模型结构 | patch 16×16，stride 16×16；12 blocks；hidden dimension 768 |
| Frozen embedding | Hugging Face AST `pooler_output`：CLS 与 distillation token 的均值 |

官方 Dropbox checkpoint 在审查时处于临时停用状态。本轮锁定可校验的 Hugging Face
转换版权重，因此实验名称明确写为 `SSAST Hugging Face converted pipeline`。这个命名
保留模型来源与具体运行实现之间的边界。

## 3. 数据流与模型设置

- 数据：792 段唯一叫声，111 只猫；
- embedding cache：`792 × 768`，所有数值有限；
- feature SHA-256：
  `cfffa85ac117767a0d47118bf7bd0f5f7576214250a167ab210530f426a1784c`；
- 分类 head：`768 → 128 → 3`，共 99,075 个可训练参数；
- head 结构：standardization、Linear、ReLU、BatchNorm、Dropout、Linear；
- optimizer：Adamax；class-balanced cross entropy；inner validation loss 选择 epoch；
- animal aggregation：同一 cat_id 的 call-level 概率取算术平均；
- primary metric：animal macro F1；secondary metrics：balanced accuracy、QWK、
  per-class recall、混淆矩阵。

Embedding 提取使用 NVIDIA GeForce RTX 4060 Ti，耗时 12.48 秒，峰值显存
659,125,248 bytes。smoke test 使用 repeat 0 / fold 0 / seed 17 的 inner train/validation，
运行两个 epoch，`outer_test_accessed=false`；数据维度、loss、animal aggregation、GPU
和日志链路全部通过。

## 4. 三组 complete OOF 结果

| Repeat | VGGish + MLP macro F1 | AST head-only macro F1 | SSAST macro F1 | SSAST − AST |
|---:|---:|---:|---:|---:|
| 0 | 0.7098 | 0.7182 | 0.6371 | −0.0811 |
| 1 | 0.6694 | 0.7616 | 0.6983 | −0.0633 |
| 2 | 0.6593 | 0.7212 | 0.5521 | −0.1691 |
| Mean | 0.6795 | 0.7337 | 0.6292 | −0.1045 |

SSAST 的 macro F1 均值为 **0.6292**，sample SD 为 **0.0734**，范围为
0.5521–0.6983。三组配对结果均低于同 repeat 的 AST head-only。对三组完整 OOF 做
5,000 次配对 bootstrap，SSAST − AST 的均值为 −0.1049，95% percentile interval 为
[−0.2033, −0.0075]。

SSAST 相对 seed-17 VGGish 均值低 0.0504；相对 seed-17 probe-guided AST adapter
均值低 0.1057。初筛结果支持当前 AST 路线继续作为已完成候选中的性能参照。

## 5. Balanced accuracy、QWK 与逐类表现

| Repeat | Balanced accuracy | QWK | Kitten recall | Adult recall | Senior recall |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.6653 | 0.6081 | 0.6667 | 0.5645 | 0.7647 |
| 1 | 0.7348 | 0.6331 | 0.8667 | 0.6613 | 0.6765 |
| 2 | 0.6063 | 0.4014 | 0.8000 | 0.5484 | 0.4706 |
| Mean | 0.6688 | 0.5475 | 0.7778 | 0.5914 | 0.6373 |

SSAST 对 kitten 的平均召回率达到 0.7778，说明 frozen representation 对幼猫类别保留了
较强的可分信息。adult 是样本最多的类别，其平均召回率为 0.5914；adult 被分到 senior
的数量分别为 20、15、19。repeat 2 的 senior recall 降至 0.4706，同时 QWK 降至
0.4014，体现该 split 下相邻年龄等级的排序一致性减弱。这两部分共同拉低了 macro F1
并扩大 repeat 间波动。

## 6. 阶段判断

预注册式筛选规则要求候选先完成 seed 17 的三组 complete OOF，再决定是否扩展 seeds
43/101。SSAST 在三组配对中取得 0/3 正向差值，平均差值为 −0.1045，因此本轮状态写为
`screened_not_better`，保留完整初筛证据并停止额外 seed 扩展。

这项结果评价的是当前完整 pipeline：转换版权重、固定输入前端、CLS/distillation-token
pooling、frozen backbone 与共享 MLP head。它为 SSAST 当前配置提供了清楚的筛选结论，
同时给未来更换原生 checkpoint、pooling 或有限 fine-tuning 留下可追踪的实验入口。
IDEA-049 计划中的下一候选为 PaSST，启动前由团队确认。

## 7. 审计材料

- 协议：`configs/protocol/meowagenet_idea049_backbone_screening_v1.json`；
- recipe：`configs/experiment/idea049/ssast_base_patch400_frozen_v1.json`；
- checkpoint card：`metadata/models/idea049/ssast_base_patch400.json`；
- runner：`scripts/run_meowagenet_idea049.py`；
- 机器可读结果：
  `metadata/experiments/meowagenet_idea049_ssast_initial_v1_results.json`；
- feature manifest：
  `runs/idea049_ssast_base_patch400_v1/features/feature_manifest.json`；
- smoke logs：`runs/idea049_ssast_base_patch400_v1/smoke/*.json`；
- initial summary 与 12 个 fit summaries：
  `runs/idea049_ssast_base_patch400_v1/initial/`。

仓库保存精简 JSON 审计链。792×768 embedding cache 与逐 call prediction CSV 保留在本机，
其内容身份由 manifest/checksum 和 complete-OOF 审计记录固定。12 个 prediction 文件按
repeat 合并后均包含 792 条唯一 call、111 个唯一 cat_id；概率和最大误差低于
1.29×10⁻⁷；12 个 fit 全部完成，cat_id partition overlap 为 0。
