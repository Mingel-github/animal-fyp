# IDEA-086：排列不变声学集合残差模型与 CPU 预检

日期：2026-09-19  
状态：**CPU GO；GPU 仍未授权。**

## 1. 锁定方案

新候选 `SET1_acoustic_set_residual` 复用 IDEA-085 的六条原始声学轨迹及六条 finite indicators，但不使用时间卷积、位置编码、attention 或 variance pooling。每帧通过共享的 `Linear(12,48)+GELU -> Linear(48,48)+GELU`，再做排除 padding 的 masked mean，最后经零初始化 `Linear(48,128)` 注入冻结 AST 的共同分类头。残差上限保持 `0.25 × stopgrad(RMS(h)) × tanh(r_set)`，不叠加 C1 分支。

分支参数为 `9,248`，总可训练参数为 `108,323`，比 IDEA-085 的 T1/J1 少 32。只新增训练 `3 seeds × 3 repeats × 4 folds = 36 fits`；A0/C1/T1/J1 的 144 个旧 fits 只读复用。

## 2. 非空洞排列不变性

预检不是在零初始化输出层上只比较恒为零的残差。它保留随机非零逐帧编码器，并临时设置非零 projection，使用混合长度 `16/177/75` 帧的真实调用比较原序、逆序和 IDEA-085 J1 固定联合帧排列：

- frame encoder 权重范数：`3.861227`；非零 projection 探针范数：`1.357866`。
- native context 范数：`1.121740`；native residual 范数：`0.034724`。
- 逆序最大差：context `1.49e-8`、residual `4.66e-10`、eval logits `0`。
- J1 排列最大差：context `2.98e-8`、residual `4.66e-10`、eval logits `0`。
- 判定容差预先固定为 `atol=rtol=2e-5`，结论 PASS。

这里的排列不变性只指同一冻结 AST call embedding 条件下，辅助声学帧集合分支对联合帧排列不变；它不表示打乱原始音频后整个 AST+SET1 系统不变，AST 主路径仍可携带时序信息。

## 3. 模型与训练兼容性

- SET1 与 IDEA-085 A0/T1 的 common AST head 初态逐张量相等；zero-init 时相对 A0 的最大 logit 差为 `0`。
- SET1 的训练角色 median/mean/std 与同 cell 的 IDEA-085 T1 完全相等；lookup 只用于 ragged 检索，不进入 12 维逐帧编码器。
- post-build RNG reset 探针相等；36 个 seed×repeat×fold cell 的旧四管线首 epoch 共核对 144 份 cat-order/call-coverage 哈希，全部一致。
- padding 混合长度 `9/443` 帧时，短调用单独与成批 context 最大差 `2.98e-8`。
- zero-init projection bias 梯度范数 `33.9411`；此时两层 frame encoder 梯度严格为 0。非零 projection 探针下两层梯度范数分别为 `10.8800/21.1482`，梯度可达。
- cap 饱和探针的相对扰动最大值 `0.25000006`，在数值容差内满足 0.25 上限。
- runner 与 protocol 显式复验 IDEA-085 的完整 068/071/082/084 依赖链、训练超参、损失、checkpoint 规则、RNG 与确定性字段；formal run 还会重新核对 CPU preflight、reuse manifest 和自身 runner/tests 哈希。

## 4. 旧结果只读完整性

- 核对 IDEA-085 `144 fit summaries + 288 prediction files`，每个文件均匹配 SHA 固定的独立审计逐文件证据；432 个文件全部 PASS。
- 对 144 个旧 fit 全部检查概率有限、范围、归一、argmax、role identity 与标签，并从 call 概率重建 cat 概率、call count 和 argmax；`144/144` 一致。
- 按总监原始 PowerShell 算法重算包含完整 IDEA-085 run root、085 protocol/runner/tests/report/metadata 与 v3 两文件的 450 文件基线，得到 `161e2357e544eaffed62b458f8d25eedc4b360a60000f02ca36c09fa7db89af3`，与预设值完全一致。有序 relative-path/SHA 清单已固化在新 reuse manifest。
- outer-test prediction/metric 未生成、读取或评分；CPU 预检期间 CUDA 未初始化。

## 5. 测试与结论

IDEA-086 专项测试 `10/10 passed`；连同 IDEA-085 模型与轨迹、IDEA-084 和 IDEA-071 回归测试共 `42/42 passed`。

因此，本地 CPU 预检结论为 **GO**。这只表示实现与只读复用门禁满足锁定方案；GPU 仍未授权，下一步必须等待独立 CPU 审计与研究总监明确授权，且首次只运行一个 SET1 fit 后暂停，检查实际训练 checkpoint 的排列不变性、batch coverage、reload 与预测哈希。

## 6. 产物哈希

- plan：`421e41781fea33adc77814b5a5d0c22f11f219b537c02c3bee4f5eecf347f759`
- protocol：`df576b4c0eb28d7dc524202aa15b40a563d1909a57e9460dbcbf21cf7ee4a527`
- runner：`33e02d175f42c3d799bc4578f6c0dbd59b5eb747f3398821b8d91137dfdbc472`
- tests：`a9533145f2d6abb63afb25914b34f47577e0aa045ba0613bc0aac9eed59d118b`
- literature/history/method review：`db834cb34f6af373fe189deadedb569429f53407909a56ee32bcd30ba2cf3e85`
- reuse manifest：`afec92014e172615433e76a5128e060f1dae6da7ecbb21f4905c985020891898`
- CPU preflight：`fb1f49ee1f279e9d79631d068ffec862854a179225d0f0a7abdd60d5c69f968c`
