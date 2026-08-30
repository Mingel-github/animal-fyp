# IDEA-049｜AVES initial screening 与阶段收尾

> 完成日期：2026-08-30
> 实验状态：`complete_for_initial_screening`
> 候选状态：`screened_not_better`
> 阶段状态：四个计划候选完成，IDEA-049 initial backbone screening 收尾
> 评价单位：animal-level complete OOF，111 只猫

## 1. 本轮完成范围

本轮完成 AVES 官方资源审查、AVES-base-bio checkpoint 与配置校验、792 条 call-level
frozen embedding 提取、inner-only smoke test，以及 repeats 0–2 × outer folds 0–3 ×
base seed 17 的 12 个 fit。三个 repeat 均形成覆盖 792 条叫声和 111 只猫的 complete OOF。

AVES 使用独立 recipe、runner、feature cache 和 run 目录，继续复用 IDEA-049 已锁定的
cat-ID-disjoint split bank、`embedding dimension → 128 → 3` head 和 animal-level
evaluation。AVES 完成后，IDEA-049 的 SSAST、PaSST、PANNs CNN14、AVES 四项初筛均有
完整运行记录；Conformer 留作未来候选，本阶段在 AVES 后收尾。

## 2. 官方资源与候选选择

| 项目 | 锁定内容 |
|---|---|
| 论文 | AVES: Animal Vocalization Encoder based on Self-Supervision，ICASSP 2023，<https://arxiv.org/abs/2210.14493> |
| 官方代码 | `earthspecies/aves` revision `fd1b660…a4c89`，<https://github.com/earthspecies/aves> |
| 代码与包许可 | MIT |
| 推理实现 | 官方 TorchAudio port，`esp-aves==1.0.0` |
| 模型 | `AVES-base-bio`，12 层、768 hidden dimensions、12 attention heads |
| Checkpoint | 377,570,545 bytes；MD5 `f2124a57e2ce7ef005dfd084cf2e81f9` |
| Checkpoint SHA-256 | `7a7dfaff2ea0b617cae1d82d7831e766be2a9ac00e37962a26f0a1b285be2530` |
| Pretraining | 153 小时 core 数据，加 AudioSet 与 VGGSound 的动物子集，共 360 小时自监督预训练 |
| 输出 | 最后一个 encoder layer 的逐时间帧 768 维表示，再执行有效时间帧算术平均 |

AVES 论文比较了 core、bio、non-bio 与 all 等预训练数据组合，并指出任务相关数据的价值。
`base-bio` 的动物声音预训练先验与猫叫年龄分类最接近，因此本轮优先选择它。`base-all`
拥有更大的 5,054 小时宽领域音频规模，任务相关性更分散，继续保留为未来模型版本消融。

## 3. 输入、pooling 与 embedding cache

每条叫声通过 `torchaudio.load` 读取，多声道取算术平均，并重采样到 16 kHz。官方 AVES
说明 padding 会改变 HuBERT 每个时间帧的 embedding，因此 792 条叫声均以 batch size 1、
原生时长独立前向。卷积前端的最短输入为 400 samples（25 ms）：400 samples 产生一个
时间帧，399 samples 无法通过最后一层卷积。数据最短叫声为 0.084375 秒，重采样后约
1,350 samples，所以本轮补零数量为 0。

最后层输出形状为 `1 × temporal frames × 768`。本轮对全部有效时间帧求算术平均，形成
每条叫声一个 768 维向量。该规则与官方 `AVESClassifier` 的 temporal mean pooling 一致，
并在读取 outer-test 结果前锁定。

| 项目 | 数值 |
|---|---:|
| Embedding shape | 792 × 768 |
| 时间帧数，minimum / median / maximum | 3 / 34 / 221 |
| 提取耗时 | 20.28 秒 |
| 峰值显存 | 462,640,128 bytes |
| 冻结 backbone 参数 | 94,370,944 |
| Feature SHA-256 | `48ed3a6b…0b80cc` |

## 4. Inner-only smoke

smoke 使用 repeat 0 / fold 0 / seed 17 的 517 条 inner-train 与 112 条
inner-validation，运行两个 epoch，`outer_test_accessed=false`。第 2 epoch 的 validation
loss 最低，为 0.8295；对应 17 只 validation 猫的 macro F1 为 0.6258、balanced
accuracy 为 0.7333、QWK 为 0.4706。

共享 768→128→3 head 包含 99,075 个可训练参数。特征标准化、BatchNorm、loss、
animal aggregation、GPU 和日志链路全部通过。

## 5. AVES complete-OOF 结果

| Repeat | Macro F1 | Balanced accuracy | QWK | Kitten recall | Adult recall | Senior recall |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.6675 | 0.7116 | 0.5969 | 0.8000 | 0.6290 | 0.7059 |
| 1 | 0.6649 | 0.7036 | 0.5966 | 0.7333 | 0.6129 | 0.7647 |
| 2 | 0.6865 | 0.7455 | 0.6164 | 0.8000 | 0.6129 | 0.8235 |
| Mean | **0.6730** | **0.7203** | **0.6033** | **0.7778** | **0.6183** | **0.7647** |

三个 macro F1 的 sample SD 为 0.0118，范围 0.6649–0.6865。它是四个 IDEA-049
候选中最低的 repeat 间波动，表明 AVES-base-bio 在三套 cat-ID folds 上形成了较稳定的
frozen representation。

三组混淆矩阵合计如下；行是真实年龄，列是预测年龄：

| True \ Pred | Kitten | Adult | Senior |
|---|---:|---:|---:|
| Kitten | 35 | 7 | 3 |
| Adult | 27 | 115 | 44 |
| Senior | 4 | 20 | 78 |

AVES 对年龄两端的召回较强：kitten 35/45，senior 78/102。adult 为 115/186，且 27 次
进入 kitten、44 次进入 senior。预测为 adult 的 precision 达到 115/142 = 0.8099；预测为
kitten 的 precision 为 35/66 = 0.5303，说明部分 adult 叫声在 AVES 空间中靠近 kitten。
这组 precision–recall 分布解释了 balanced accuracy 0.7203 与 macro F1 0.6730 的差异。

QWK 0.6033 表明预测保留了较好的年龄顺序一致性。kitten↔senior 的跨两级错误合计为
7 次，adult 与相邻两端之间的错误占主要部分，因此顺序加权评分高于普通分类 F1 的数值。

## 6. 四个新 backbone 与历史参照

以下均为相同 seed 17、三个 repeat、相同 split bank 和 animal-level complete OOF 的均值：

| Pipeline | Macro F1 | Balanced accuracy | QWK | Macro F1 sample SD |
|---|---:|---:|---:|---:|
| PANNs CNN14 | 0.5884 | 0.6623 | 0.5491 | 0.0305 |
| SSAST | 0.6292 | 0.6688 | 0.5475 | 0.0734 |
| PaSST | 0.6560 | 0.6973 | 0.5812 | 0.0217 |
| **AVES-base-bio** | **0.6730** | **0.7203** | **0.6033** | **0.0118** |
| VGGish + MLP | 0.6795 | 0.6795 | 0.5772 | 0.0267 |
| AST head-only | **0.7337** | **0.7729** | **0.6344** | 0.0240 |

AVES 在四个新 backbone 中同时取得最高 macro F1、balanced accuracy、QWK 和最低
repeat 波动。相对 PaSST，macro F1 平均提高 0.0171，三组配对中有 2 组为正；相对
PANNs 平均提高 0.0846，三组全部为正。

AVES 与 VGGish 的 macro F1 均值相差 0.0065，处于非常接近的水平。两个模型呈现清楚的
优势分工：VGGish 的 adult recall 与 kitten precision 更高，因此 macro F1 略高；AVES
对 kitten、senior 的覆盖更均衡，使 balanced accuracy 高 0.0407、QWK 高 0.0261。
AVES 第 2 个 repeat 的 macro F1 高于同组 VGGish 0.0272。

AST head-only 仍是本阶段最高参照。AVES 相对 matched AST 的三个 macro F1 差值为
−0.0507、−0.0967、−0.0347，平均 −0.0607。5,000 次 animal-level paired bootstrap
均值为 −0.0603，95% percentile interval 为 [−0.1482, 0.0280]。三组点估计方向一致，
同时 111 只猫带来的区间宽度保留了结果的不确定范围。

## 7. 阶段结论

1. AVES-base-bio 是 IDEA-049 四个新 frozen backbone 中最强、最稳定的候选，并在
   balanced accuracy 与 QWK 上超过 VGGish。
2. AVES 与 VGGish 的 macro F1 基本相当，两者分别强调年龄两端召回与类别 precision；
   这为论文提供了清楚的模型能力差异。
3. AST head-only 继续保持综合领先，尤其在 macro F1 和 balanced accuracy 上形成约
   0.06 与 0.05 的均值优势。
4. 本阶段保留 AVES 全部初筛证据，seeds 43/101 留作未来重复实验选项；Conformer 与
   第二波候选进入未来工作池。IDEA-049 initial backbone screening 在 AVES 后完成收尾。

## 8. 审计材料

- recipe：`configs/experiment/idea049/aves_base_bio_frozen_v1.json`；
- checkpoint card：`metadata/models/idea049/aves_base_bio.json`；
- 官方配置语义副本：`metadata/models/idea049/aves_base_bio_torchaudio_config.json`；
- 环境补充：`environment/idea049-aves-inference-v1.txt`；
- runner：`scripts/run_meowagenet_idea049_aves.py`；
- 机器可读结果：
  `metadata/experiments/meowagenet_idea049_aves_initial_v1_results.json`；
- feature manifest：`runs/idea049_aves_base_bio_v1/features/feature_manifest.json`；
- smoke logs：`runs/idea049_aves_base_bio_v1/smoke/*.json`；
- initial summary 与 12 个 fit summaries：`runs/idea049_aves_base_bio_v1/initial/`。

每个 repeat 均由 4 个 outer-fold 文件组成，合并后覆盖 792 条唯一 call 和 111 个唯一
cat_id；概率和最大误差低于 1.27×10⁻⁷；12 个 fit 全部完成，cat_id partition overlap
为 0。仓库保存 18 个精简 JSON 审计文件，embedding cache 与逐 call CSV 留在本机并由
checksum 和 complete-OOF 审计固定身份。
