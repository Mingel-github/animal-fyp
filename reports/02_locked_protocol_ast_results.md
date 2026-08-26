# 固定协议与 AST Pilot 结果

日期：2026-08-25

## 结论先行

本轮已完成固定 cat-ID folds、统一 animal-level 指标、标准 AST frozen
representation、IDEA-012 time-fine geometry、frequency-fine 计算量对照和一次性
outer comparison。

主要结果是：VGGish+MLP 的 animal-level macro F1 为 **0.6846**；标准 AST
为 **0.6525**；time-fine AST 为 **0.6469**。time-fine 相对标准 AST 的差值
为 **-0.0056**，没有出现 IDEA-012 预期的提升。只依据 inner validation 逐折
选择的 nested AST 为 **0.6666**，比 VGGish 低 0.0180，点估计仍落在事前设置
的 `±0.03` 实践差异范围内，但没有超过 VGGish。

因此，当前 frozen-encoder pilot 不支持“减小 time stride 会提高性能”的
主张；VGGish 仍是本轮单一 pipeline 中 primary metric 点估计最高者。

## 1. 固定协议

- 数据：792 段唯一音频、111 个分析 `cat_id`；排除重复别名 `049A`。
- 类别动物数：kitten 15、adult 62、senior 34。
- Outer evaluation：固定 4 折，每折测试 27～28 只猫；同一只猫的全部叫声
  始终位于同一 fold。
- Inner validation：只从当前 outer-training cats 中划出17只猫，用于确定早停
  epoch；随后使用全部 outer-training cats 重训相同 epoch 数。
- 聚合：同一猫的类别概率取算术平均，再取最大概率类别。
- Primary metric：animal-level macro F1。
- Secondary：balanced accuracy、QWK、per-class recall、prediction-unit macro F1。
- 实践差异阈值：`δ_task = 0.03` macro F1。
- 训练 seed：`42 + outer_fold`，所有表示保持相同。

原论文 notebook 的强制训练猫规则没有沿用；`000A`、`046A` 与其他猫一样可
进入测试 fold。

## 2. 模型与预训练兼容性

本轮使用官方
[MIT AST AudioSet checkpoint](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)，
标准配置依据
[Hugging Face AST 文档](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer)。
checkpoint revision固定为
`f826b80d28226b62986cc218e5cec390b1096902`。`model.safetensors` 的 SHA-256
为 `ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d`。

2026-08-26重新核验确认本机配有NVIDIA GeForce RTX 4060 Ti 8GB。2026-08-25
这轮特征提取仍实际运行在CPU上，原因是当时Conda环境安装了`torch 2.2.2+cpu`，
且提取脚本没有将模型和batch迁移到CUDA；这属于环境与执行路径选择，不是硬件
缺失。本轮采用冻结AST encoder、训练统一Dense(128) MLP head，仍然是标准AST
geometry的预训练表示pilot，不等同于完整encoder fine-tuning。

音频统一重采样到16 kHz，以1.28秒窗口、0.64秒 hop 覆盖完整叫声；792段叫声
共形成843个片段，长叫声的多个片段 embedding 在 call 内取平均。三种 AST
共享相同 log-Mel、checkpoint、输入窗口、encoder 和分类头：

| 变体 | frequency stride | time stride | Patch tokens | 用途 |
| --- | ---: | ---: | ---: | --- |
| Standard | 10 | 10 | 144 | 标准 geometry |
| Time-fine | 10 | 5 | 276 | IDEA-012 主候选 |
| Frequency-fine | 5 | 10 | 276 | 与 time-fine token 数匹配的方向对照 |

三种变体均保留16×16 patch kernel。patch projection 和 Transformer encoder
权重全部复用；只将预训练 positional grid 双线性插值到新输入/stride 对应的
网格，没有随机重建 patch projection。encoder 可训练参数为0。

## 3. Animal-level 结果

| Pipeline | Macro F1 | Balanced accuracy | QWK | Kitten recall | Adult recall | Senior recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VGGish+MLP | **0.6846** | 0.6754 | 0.5263 | 0.6667 | **0.7419** | 0.6176 |
| AST standard | 0.6525 | 0.7063 | **0.5723** | **0.8000** | 0.6129 | **0.7059** |
| AST time-fine | 0.6469 | 0.7018 | 0.5474 | **0.8000** | 0.6290 | 0.6765 |
| AST frequency-fine | 0.6167 | 0.6670 | 0.4729 | **0.8000** | 0.6129 | 0.5882 |
| AST nested-selected | 0.6666 | **0.7082** | 0.5432 | **0.8000** | 0.6774 | 0.6471 |

标准 AST 的 macro F1 低于 VGGish，但 balanced accuracy 和 QWK 更高。它提高
了 kitten、senior recall，同时明显降低 adult recall；因此“哪个更好”会随
指标改变，这正是事前指定 primary metric 的原因。

四折 macro F1 如下：

| Pipeline | Fold 0 | Fold 1 | Fold 2 | Fold 3 |
| --- | ---: | ---: | ---: | ---: |
| VGGish+MLP | 0.8125 | 0.5937 | 0.6966 | 0.6183 |
| AST standard | 0.7737 | 0.6232 | 0.6653 | 0.5844 |
| AST time-fine | 0.7737 | 0.7078 | 0.5788 | 0.5934 |
| AST frequency-fine | 0.6595 | 0.5974 | 0.6016 | 0.5844 |

fold 差异较大，说明111只猫仍不足以稳定区分几个百分点的模型差异。

## 4. Inner selection 与主要差值

只依据各 outer fold 的 inner validation macro F1，选择结果为：

- Fold 0：AST standard；
- Fold 1：AST time-fine；
- Fold 2：AST frequency-fine；
- Fold 3：AST standard。

| Contrast | Macro F1 差值 | 动物分层配对 bootstrap 95%区间 |
| --- | ---: | ---: |
| Time-fine − Standard | -0.0056 | [-0.0557, 0.0421] |
| Nested AST − VGGish | -0.0180 | [-0.1100, 0.0805] |
| Frequency-fine − Standard | -0.0359 | [-0.0904, 0.0128] |

time-fine 比 frequency-fine 高约3.03个百分点，说明时间方向至少优于同 token
数的频率方向；但它仍未超过 standard，因此不足以构成 IDEA-012 的性能支持。

## 5. 与原论文复现结果的关系

此前约0.70的三分类结果来自官方937行 VGGish CSV、作者 split 修改和以
embedding/call 为主的指标。本轮0.6846使用清洗后的936行 VGGish 视图、固定
111猫 folds、validation-loss 早停和 animal-level macro F1。两组数字回答的
问题不同，不能把0.6846解释为原论文复现退化。

本轮没有建立第二个正式 baseline：原 notebook 结果继续作为历史复现参考；
统一协议下的 VGGish+MLP 是 IDEA-048 的正式比较基线。

## 6. 当前判断与限制

- IDEA-012：当前 frozen-encoder pilot 没有提升信号，结果更接近 N0/H4，即
  geometry 差异位于小样本训练和 fold 方差内。
- IDEA-048：nested AST 点估计比 VGGish 低1.8个百分点，按当前务实阈值可暂视
  为性能接近，但不能声称超过 baseline。
- 这是冻结encoder、单seed-per-fold的pilot；其中2026-08-25版AST特征在CPU上
  提取。GPU重跑用于校验设备迁移的一致性并为后续fine-tuning建立运行环境。
- 每个 inner validation 只有17只猫，variant 排名不稳定；bootstrap 区间也较宽。
- “standard”指标准16×16 kernel和10×10 stride；为匹配短叫声及首轮pilot预算，
  输入位置网格从原10.24秒设置插值到1.28秒窗口。

按照主路线的判定规则，本轮已经到达人工决策点：不再继续无边界调stride。
2026-08-26已收到IDEA-003、013和019的完整候选材料；下一候选应结合GPU重跑、
当前接口可行性和主性能目标另行冻结。

## 7. GPU硬件纠正与一致性重跑（2026-08-26）

重新核验确认本机配有NVIDIA GeForce RTX 4060 Ti 8GB，驱动595.79。原环境中的
PyTorch为`2.2.2+cpu`，且提取脚本没有设备迁移逻辑，这解释了为什么8月25日记录
的是CPU耗时。环境现已替换为`torch 2.2.2+cu121`，脚本增加`auto/cuda/cpu`
设备选择和设备审计。

保持checkpoint、revision、输入、三种geometry、folds、seed和MLP完全不变后，
完整重跑792段叫声、843个segments：

| 变体 | 原CPU推理 | GPU推理 | 加速比 |
| --- | ---: | ---: | ---: |
| AST standard | 43.93秒 | 2.81秒 | 15.63× |
| AST time-fine | 90.79秒 | 5.19秒 | 17.51× |
| AST frequency-fine | 92.21秒 | 5.21秒 | 17.68× |

CPU与GPU embedding的平均绝对差约为`2.54e-6`～`2.66e-6`，最大绝对差不超过
`3.86e-5`。重新训练相同MLP后，五条pipeline的111只猫最终预测标签全部一致，
animal-level macro F1、balanced accuracy、QWK、逐类recall、nested fold选择和
主要bootstrap差值均与原报告相同。

因此，硬件审计错误影响了资源判断和运行速度说明，但没有影响本轮frozen AST
结论。更重要的新信息是：本机现在具备有限AST fine-tuning、PEFT和token-level
pooling实验的硬件条件，下一阶段不再受“仅CPU”这一假设限制。

GPU重跑材料：

- AST审计与embedding：`runs/ast_locked_v1/gpu_rerun_2026-08-26/`；
- 重新训练与预测：`runs/locked_comparison_gpu_verified_2026-08-26/`。
