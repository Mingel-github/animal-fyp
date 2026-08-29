# IDEA-049｜PANNs CNN14 initial screening 结果报告

> 完成日期：2026-08-30
> 实验状态：`complete_for_initial_screening`
> 候选状态：`screened_not_better`
> 评价单位：animal-level complete OOF，111 只猫

## 1. 本轮完成范围

本轮完成 PANNs CNN14 官方资源审查、792 条 call-level frozen embedding 提取、
inner-only smoke test，以及 repeats 0–2 × outer folds 0–3 × base seed 17 的 12 个 fit。
三个 repeat 均形成覆盖 792 条叫声和 111 只猫的 complete OOF。

PANNs 使用独立 recipe、runner、feature cache 和 run 目录，复用 IDEA-049 已锁定的
cat-ID-disjoint split bank、`embedding dimension → 128 → 3` head 和 animal-level
evaluation。formal-v2.1、SSAST 与 PaSST 的历史文件保持原样。

## 2. 官方资源与运行身份

| 项目 | 锁定内容 |
|---|---|
| 论文 | PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition，<https://arxiv.org/abs/1912.10211> |
| 官方代码 | `qiuqiangkong/audioset_tagging_cnn` 的 `Cnn14`，<https://github.com/qiuqiangkong/audioset_tagging_cnn> |
| 代码许可 | MIT |
| 推理实现 | `panns-inference==0.1.1`，wheel SHA-256 `97f6b5…27fc2` |
| 频谱依赖 | `torchlibrosa==0.1.0`，wheel SHA-256 `89b65f…a38cb` |
| Checkpoint | Zenodo record 3987831，`Cnn14_mAP=0.431.pth`，CC BY 4.0 |
| Checkpoint checksum | MD5 `541141fa2ee191a88f24a3219fff024e`；SHA-256 `0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31` |
| Pretraining | supervised AudioSet multi-label training，full training set、balanced sampling、mixup |
| 官方 AudioSet 结果 | mAP 0.431 |
| 输出 | 全局时频聚合与 `fc1` ReLU 后的 2048 维 embedding；527 维 AudioSet logits 留在分类头之外 |

官方 Zenodo 文件为 327,428,481 bytes。本地下载采用带相同 SHA-256 的镜像副本完成，
随后以 Zenodo 的文件大小和 MD5、镜像公开的 SHA-256、PyTorch state-dict 全键匹配三层
校验固定身份。Cnn14 共 81,837,071 个参数，全部在 embedding 提取阶段冻结。

## 3. 输入前端、pooling 与短叫声处理

官方前端使用 32 kHz mono、1024-sample Hann window、320-sample hop、64 mel bins、
50–14,000 Hz、centered reflect padding 和 torchlibrosa log-mel。Cnn14 包含六个卷积块；
前五个卷积块在时间和频率方向各做一次 2×2 average pooling，第六个保持尺寸。

卷积输出先对频率维取平均，再对时间维分别取最大值和平均值并相加，最后经过 2048→2048
的 `fc1` 与 ReLU 形成 frozen embedding。这个 pooling 同时保留最强时间响应与整段平均响应，
适合把可变长度叫声压缩为固定的 2048 维向量。

五次时间减半要求 STFT 至少产生 32 帧。实测 9,920 个 samples（0.31 秒）可以输出
`1 × 2048` embedding，9,919 个 samples 会在第五次 pooling 后失去时间位置。因此 68 条
短于 0.31 秒的 call 只在右侧补零至 9,920 samples，其余 724 条保留原始时长。该规则在
访问 PANNs outer-test 结果前锁定。

792×2048 embedding 提取耗时 7.36 秒，峰值显存 488,786,944 bytes；feature SHA-256 为
`3f39f5416111f5f9a551de0ac2fceded18c40efa42098b1426a039a82275eaa4`。

## 4. Smoke test

smoke 使用 repeat 0 / fold 0 / seed 17 的 517 条 inner-train 与 112 条 inner-validation，
运行两个 epoch，`outer_test_accessed=false`。validation loss 在第 2 epoch 达到最低值
0.9193，因此选择 epoch 2；对应的 validation animal macro F1 为 0.5806、balanced
accuracy 为 0.7000、QWK 为 0.4333。

PANNs 的 2048 维输入使共享 head 包含 262,915 个可训练参数。数据维度、feature
standardization、BatchNorm、loss、animal aggregation、GPU 和日志链路全部通过。

## 5. Complete-OOF 主结果

| Repeat | SSAST macro F1 | PaSST macro F1 | PANNs macro F1 | VGGish macro F1 | AST head-only macro F1 | PANNs − AST |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.6371 | 0.6319 | 0.6121 | 0.7098 | 0.7182 | −0.1062 |
| 1 | 0.6983 | 0.6741 | 0.5991 | 0.6694 | 0.7616 | −0.1625 |
| 2 | 0.5521 | 0.6619 | 0.5540 | 0.6593 | 0.7212 | −0.1672 |
| Mean | 0.6292 | 0.6560 | **0.5884** | 0.6795 | **0.7337** | −0.1453 |

PANNs 的 macro F1 sample SD 为 0.0305，范围为 0.5540–0.6121。它在 repeat 2 略高于
SSAST 0.0019；相对 PaSST、VGGish 和 AST 的三组差值均为负，平均差值分别为 −0.0676、
−0.0912 和 −0.1453。

相对 matched AST head-only 的 5,000 次 paired bootstrap 均值为 −0.1450，95%
percentile interval 为 [−0.2325, −0.0559]。三组 repeat 在方向上保持一致，显示当前
PANNs frozen representation 与共享 MLP 的组合在本数据边界上形成稳定性能差距。

## 6. Balanced accuracy、QWK 与逐类信息

| Repeat | Balanced accuracy | QWK | Kitten recall | Adult recall | Senior recall |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.6935 | 0.5837 | 0.9333 | 0.5000 | 0.6471 |
| 1 | 0.6491 | 0.5600 | 0.8667 | 0.5806 | 0.5000 |
| 2 | 0.6444 | 0.5038 | 0.8667 | 0.4194 | 0.6471 |
| Mean | **0.6623** | **0.5491** | **0.8889** | **0.5000** | **0.5980** |

PANNs 最清楚的优势落在 kitten：45 个 kitten 判断中有 40 个正确，平均 recall 0.8889，
高于本轮 PaSST 的 0.8000。三个 repeat 的 adult 混淆矩阵合计显示，186 个 adult 判断中
93 个正确，40 个被分到 kitten，53 个被分到 senior。adult recall 0.5000 因而成为
macro F1 的主要限制，同时两端类别接收较多 adult 预测，降低 kitten 和 senior precision。

QWK 0.5491 高于本候选 macro F1 的数值水平，说明年龄顺序信息仍有可学习信号：大量错误
落在相邻年龄区间，跨越两级的 kitten↔senior 错误相对少。它低于 PaSST 的 0.5812 和
AST 的 matched seed-17 水平，表示 PANNs 当前的顺序一致性也尚未抵消 adult decision
region 的压缩。

## 7. 阶段判断

IDEA-049 的 primary comparison 是 matched AST head-only。PANNs 在三组配对中取得
0/3 个正向 macro F1 差值，平均差值 −0.1453，因此状态记录为 `screened_not_better`，
本轮保留完整初筛证据，seeds 43/101 停留在待扩展池。

三个已完成的新 backbone 中，PaSST 当前排名第一（0.6560），SSAST 第二（0.6292），
PANNs 第三（0.5884）。PANNs 的高 kitten recall 提供了明确的类别特征信息；综合指标与
matched AST 的差距支持把下一轮资源投入既定第四候选 AVES。

## 8. 审计材料

- recipe：`configs/experiment/idea049/panns_cnn14_frozen_v1.json`；
- checkpoint card：`metadata/models/idea049/panns_cnn14_audioset.json`；
- runner：`scripts/run_meowagenet_idea049_panns.py`；
- 机器可读结果：
  `metadata/experiments/meowagenet_idea049_panns_initial_v1_results.json`；
- feature manifest：`runs/idea049_panns_cnn14_v1/features/feature_manifest.json`；
- smoke logs：`runs/idea049_panns_cnn14_v1/smoke/*.json`；
- initial summary 与 12 个 fit summaries：`runs/idea049_panns_cnn14_v1/initial/`。

每个 repeat 均由 4 个 outer-fold 文件组成，合并后覆盖 792 条唯一 call 和 111 个唯一
cat_id；概率和最大误差低于 1.28×10⁻⁷；12 个 fit 全部完成，cat_id partition overlap
为 0。仓库保存 18 个精简 JSON 审计文件，embedding cache 与逐 call CSV 留在本机并由
checksum 和 complete-OOF 审计固定身份。
