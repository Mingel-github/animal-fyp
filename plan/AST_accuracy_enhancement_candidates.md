# AST 准确率增强候选计划

## 状态与目标

- 当前状态：候选发现，实验协议尚未冻结。
- 起点：tuned frozen AST head-only。
- 主指标：complete out-of-fold prediction 上的 animal-level Macro F1。
- 评价边界：cat-ID-disjoint outer evaluation、inner-only 模块选择、同轮匹配对照，以及111只猫均获得一次完整预测。

当前 post-formal tuned AST head-only 的探索性锚点为：三次 seed-17 complete OOF 的 animal-level Macro F1 均值0.7488、Balanced Accuracy 0.7644、QWK 0.6469。这些数值用于后续匹配比较，不代表新模块预期能够达到的性能。

## 优先研究路线

当前首选候选是 **Cat-balanced Multi-Layer AST**，由两个可以独立消融的机制组成：

1. Cat-balanced loss 或 sampling：虽然每只猫拥有1–45条 call，训练时仍让每只猫的期望梯度贡献接近一致。
2. Frozen multi-layer scalar fusion：读取若干 AST layer 的 pooled representation，只学习少量 layer weights 和 MLP，而非只读取 final representation。

待检验的 idea claim 是：

> 使训练贡献与 animal-level 评价单位一致，并向分类头开放 frozen AST 不同层的互补表示，可以相对容量匹配的 final-layer head 提高 unseen-cat 年龄分类性能。

这是一条 hypothesis。两个机制都可能单独失败，也可能产生负交互。

## 训练新模块之前的诊断

1. 检查每只猫的 call 数与正确率、置信度和错误类别之间的关系。
2. 计算 tuned AST 与 matched VGGish/CNN 的 animal-level 错误重合度。
3. 只利用 inner validation，在单层和分组 AST layers 上训练 probe。
4. 检查 call duration、有效帧比例和 padding 比例与错误率的关系。

诊断结果控制后续分支：错误互补支持 fusion/distillation；时长相关错误支持 padding-aware processing；不同层的验证差异支持 multi-layer fusion。

## 核心消融矩阵

| Run | Cat balancing | AST 表示 | 目的 |
|---|---|---|---|
| A0 | 当前 recipe | Final representation | tuned head-only 同轮对照 |
| A1 | 启用 | Final representation | 单独检验训练单位对齐 |
| A2 | 当前 recipe | Learned scalar fusion | 单独检验跨层信息 |
| A3 | 启用 | Learned scalar fusion | 检验组合收益和交互 |

A2和A3还需要一个 trainable parameter 数量近似匹配的 final-layer wider MLP 对照，用于区分“获得了多层信息”和“分类头容量增加”两种解释。

## 次级与条件候选

| 候选方法 | 启动条件 | 主要对照 |
|---|---|---|
| AST–VGGish/CNN probability fusion | 两类模型都有足够的 exclusive-correct cats | 两个单模型 |
| Cross-model knowledge distillation | Probability fusion 出现稳定正信号 | 相同 recipe、无 distillation 的 AST |
| LayerNorm head | 当前 micro-batch BatchNorm 存在跨 seed 波动 | 当前 BatchNorm head |
| Checkpoint averaging | 多个相邻 late checkpoints 的 validation 表现接近 | 单个 selected checkpoint |
| Padding-aware/native-length AST | 错误率随 padding 比例稳定升高 | 当前固定1.28秒输入 |
| Acoustic residual branch | 手工声学特征在 AST 之外仍提供 inner-validation 信息 | AST-only 和 feature-only |
| Patch-Mix contrastive regularization | 已有理由更新 encoder | 相同 recipe、无 Patch-Mix/contrastive loss |
| Animal-level class-bias calibration | 表示改造后仍存在可重复的类别偏差 | 未校准的 animal probabilities |

## 进入组合实验的条件

- 所有超参数只在各 outer training role 内选择。
- 单模块先获得一致的 inner-validation 正信号，再完成 matched outer evaluation，随后才进入组合实验。
- 同时报告 Macro F1、Balanced Accuracy、QWK、各类别 precision/recall、confusion matrix，以及每只猫的 paired prediction changes。
- time-fine patch geometry、temporal attention、full fine-tuning、adapter 和已测试 Q/V LoRA 的负结果继续作为设计证据保留。
- 团队选定进入外部执行的候选后，再冻结带编号的正式 protocol。

## 文献依据

- Miron et al., *Multi-layer attentive probing improves transfer of audio representations for bioacoustics*, arXiv:2605.10494。
- Gong et al., *CMKD: CNN/Transformer-Based Cross-Model Knowledge Distillation for Audio Classification*, arXiv:2203.06760。
- Feng and Schuller, *ElasticAST*, Interspeech 2024，DOI `10.21437/Interspeech.2024-1890`。
- Bae et al., *Patch-Mix Contrastive Learning with Audio Spectrogram Transformer on Respiratory Sound Classification*, Interspeech 2023，DOI `10.21437/Interspeech.2023-1426`。
- Ghosh et al., *MAST: Multiscale Audio Spectrogram Transformers*, ICASSP 2023，arXiv:2211.01515。
- He et al., *Multi-View Spectrogram Transformer for Respiratory Sound Classification*, ICASSP 2024，arXiv:2311.09655。
