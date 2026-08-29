# IDEA-049｜PaSST initial screening 结果报告

> 完成日期：2026-08-30
> 实验状态：`complete_for_initial_screening`
> 候选状态：`screened_not_better`
> 评价单位：animal-level complete OOF，111 只猫

## 1. 本轮完成范围

本轮完成 PaSST 官方资源审查、792 条 call-level frozen embedding 提取、inner-only
smoke test，以及 repeats 0–2 × outer folds 0–3 × base seed 17 的 12 个 fit。三个 repeat
均形成覆盖 792 条叫声和 111 只猫的 complete OOF。

PaSST 使用独立 recipe、runner、feature cache 和 run 目录，复用 IDEA-049 已锁定的
cat-ID-disjoint split bank、`embedding dimension → 128 → 3` head 和 animal-level
evaluation。formal-v2.1 与 SSAST 历史文件保持原样。

## 2. 官方资源与运行身份

| 项目 | 锁定内容 |
|---|---|
| 论文 | Efficient Training of Audio Transformers with Patchout，Interspeech 2022，<https://arxiv.org/abs/2110.05069> |
| 官方代码 | `kkoutini/PaSST`，revision `2a5c818afcc2a215b2a1aaf1ed8be71f89d43201`，<https://github.com/kkoutini/PaSST> |
| 推理实现 | `kkoutini/passt_hear21`，revision `5f1cce6a54b88faf0abad82ed428355e7931213a`，<https://github.com/kkoutini/passt_hear21> |
| License | Apache-2.0 |
| 运行包 | `hear21passt==0.0.26`，wheel SHA-256 `a3a737…25550` |
| Checkpoint | `passt_s_swa_p16_128_ap476`，官方 release `v0.0.1-audioset` |
| Checkpoint SHA-256 | `302903fa8c4aee817b11dc982da0b29aaf8d11a3e722420476d0a12c9db70c2c` |
| Pretraining | ImageNet 初始化、supervised AudioSet multi-label training、structured Patchout、SWA |
| 输出 | `embed_only`，final CLS/distillation-token mean，768 维 |

`embed_only` 选择保留 PaSST backbone 的 768 维表示。官方 wrapper 的 `all` 模式还会
拼接 527 维 AudioSet logits，形成 1295 维 HEAR embedding；本轮排除该任务专用 logits，
使分类 head 继续接收单一 backbone representation，并与 AST/SSAST 的 768 维输入一致。

## 3. 输入前端与短叫声处理

PaSST 官方前端使用 32 kHz mono、128 mel bins、800-sample window、320-sample hop、
1024 FFT、0.97 pre-emphasis 和 `(log(mel + 1e-5) + 4.5) / 5` normalization。推理阶段
三种 Patchout 均设为 0，保证 frozen embedding 的确定性。

分析数据的 call 时长范围为 0.084–4.425 秒，中位数为 0.698 秒。PaSST 的 16-frame
时间 patch 需要至少 160 ms 输入，因此 11 条更短的 call 只在右侧补零至 5120 samples；
其余 781 条保留原始时长。这个规则在访问 PaSST outer-test 结果前锁定。

792×768 embedding 提取耗时 10.40 秒，峰值显存 383,693,312 bytes；feature SHA-256 为
`d63561358fe349f722bd6bf5e121b8df2ee74df5d36ed0950977118091e1606f`。

## 4. Smoke test

smoke 使用 repeat 0 / fold 0 / seed 17 的 517 条 inner-train 与 112 条 inner-validation，
运行两个 epoch，`outer_test_accessed=false`。第二个 epoch 的 validation animal macro F1
为 0.6242、balanced accuracy 为 0.7333、QWK 为 0.4917。数据维度、loss、animal
aggregation、99,075 个 head 参数、GPU 与日志链路全部通过。

## 5. Complete-OOF 主结果

| Repeat | SSAST macro F1 | PaSST macro F1 | VGGish macro F1 | AST head-only macro F1 | PaSST − AST |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.6371 | 0.6319 | 0.7098 | 0.7182 | −0.0863 |
| 1 | 0.6983 | 0.6741 | 0.6694 | 0.7616 | −0.0875 |
| 2 | 0.5521 | 0.6619 | 0.6593 | 0.7212 | −0.0593 |
| Mean | 0.6292 | **0.6560** | 0.6795 | **0.7337** | −0.0777 |

PaSST 的 macro F1 sample SD 为 **0.0217**，范围为 0.6319–0.6741。它比 SSAST 均值
高 0.0268，repeat 间波动明显收窄；repeat 2 对 SSAST 的增量达到 0.1098。相对 matched
VGGish，PaSST 在 repeats 1、2 分别高 0.0047 和 0.0025，repeat 0 低 0.0779，最终均值
差为 −0.0236。

PaSST 相对 AST head-only 的三组配对差值均为负，平均 −0.0777。5,000 次配对 bootstrap
得到均值 −0.0782，95% percentile interval 为 [−0.1435, −0.0136]。

## 6. Balanced accuracy、QWK 与逐类信息

| Repeat | Balanced accuracy | QWK | Kitten recall | Adult recall | Senior recall |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.6609 | 0.5788 | 0.6667 | 0.5806 | 0.7353 |
| 1 | 0.7089 | 0.5692 | 0.8667 | 0.6129 | 0.6471 |
| 2 | 0.7222 | 0.5956 | 0.8667 | 0.5645 | 0.7353 |
| Mean | **0.6973** | **0.5812** | **0.8000** | **0.5860** | **0.7059** |

Matched seed-17 VGGish 的 balanced accuracy 为 0.6795、QWK 为 0.5772。PaSST 分别高
0.0178 和 0.0041，显示它在三个年龄类别的平均召回和年龄顺序一致性上略占优势。
PaSST 的 kitten/senior recall 分别为 0.8000/0.7059，高于 VGGish 的 0.6667/0.6569。

PaSST 的 adult recall 为 0.5860，VGGish 为 0.7151。三个 repeat 中分别有 26、24、27
只 adult 被分到 kitten 或 senior，这会同时降低 adult recall，并降低两端预测的 precision。
因此 PaSST 可以取得更高 balanced accuracy 和略高 QWK，同时 macro F1 均值仍低于
VGGish。这组结果清楚呈现 PaSST 的优势集中在年龄两端和跨 repeat 稳定性，当前主要缺口
集中在 adult decision region。

## 7. 阶段判断

IDEA-049 的 primary comparison 是 matched AST head-only。PaSST 在三组配对中取得 0/3
正向 macro F1 差值，平均差值 −0.0777，因此状态记录为 `screened_not_better`，本轮保留
完整证据并停止 seeds 43/101 扩展。

在两个已完成的新候选中，PaSST 提供了更高、更稳定的 macro F1，并在 balanced accuracy、
QWK 和年龄两端 recall 上形成可解释信息。下一候选按既定顺序进入 PANNs CNN14，用来检验
强 CNN AudioSet representation 在相同边界下的表现。

## 8. 审计材料

- recipe：`configs/experiment/idea049/passt_s_ap476_frozen_v1.json`；
- checkpoint card：`metadata/models/idea049/passt_s_ap476.json`；
- runner：`scripts/run_meowagenet_idea049_passt.py`；
- 机器可读结果：
  `metadata/experiments/meowagenet_idea049_passt_initial_v1_results.json`；
- feature manifest：`runs/idea049_passt_s_ap476_v1/features/feature_manifest.json`；
- smoke logs：`runs/idea049_passt_s_ap476_v1/smoke/*.json`；
- initial summary 与 12 个 fit summaries：`runs/idea049_passt_s_ap476_v1/initial/`。

每个 repeat 均由 4 个 outer-fold 文件组成，合并后覆盖 792 条唯一 call 和 111 个唯一
cat_id；概率和最大误差低于 1.35×10⁻⁷；12 个 fit 全部完成，cat_id partition overlap
为 0。仓库保存 18 个精简 JSON 审计文件，embedding cache 与逐 call CSV 留在本机并由
checksum 和 complete-OOF 审计固定身份。
