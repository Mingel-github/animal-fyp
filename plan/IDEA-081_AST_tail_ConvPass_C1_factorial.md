# IDEA-081 — AST 尾层完整 ConvPass × C1 因子实验

## 决策

GO（仅方法与 CPU 预检）。V1 可在共享 block-11 token cache 后忠实实现为 AST block 12 内的双 ConvPass 旁路，无需近似。正式 144-fit GPU 实验尚未获本任务授权。

## 问题

在 MeowAgeNet 的固定 inner train/validation 角色中，文献忠实的尾层 ConvPass 是否：

1. 在无 C1 背景下优于 A0（V1−A0）；
2. 在固定 C1 背景下仍提供增益（CV1−C1）；
3. 与 C1 产生正交互 `I=(CV1−C1)−(V1−A0)`。

## 冻结项

- IDEA-077、IDEA-079 全部冻结，不更改文件、结果或命名。
- 只读复用 IDEA-078 block-11 cache `[843,146,768]`；不二次提取。
- frozen AST 与 final LayerNorm 不训练。
- outer test 始终为 false。
- 不根据结果调整 block、width、scale、初始化、seed 或 gate。

## 模型矩阵

- `A0_frozen_tail`：冻结 block 12 + final LN + 两 special 平均 + segment-to-call 平均 + 固定 head。
- `C1_bounded_age`：A0 加已经锁定的 C1 bounded RMS-relative age residual。
- `V1_tail_full_convpass`：A0 加 block 12 的 MSA/MLP 两个完整 ConvPass 旁路。
- `CV1_C1_plus_tail_full_convpass`：同时启用 C1 与 V1。

ConvPass 的公式、12×12 reshape、两个 special token 处理、QuickGELU、dense 3×3、width 8、scale 0.1、dropout 0.1、identity-center conv 与 zero-up 均在 protocol 中锁定。

## 初始化配对

- 四管线 common head state 相同。
- C1 与 CV1 的 age state 相同。
- V1 与 CV1 的两个 ConvPass state 相同。
- zero-up 与 C1 zero-output 必须使四管线初始 logits/loss 精确相同。

## 种子与预算

- 派生文本：`IDEA-081-Meow-tail-ConvPass-x-C1-v1`
- SHA-256：`12f7cae12bf10b3bcfcaf31330ea5abb96f3e692034e33858e3e33182fc6947ca9`
- 排除 IDEA-065 至 IDEA-080 的所有已用/已保留 base seeds 与 full seeds。
- 固定 base seeds：`[9217,7339,4211]`。
- 3 seeds × 3 repeats × 4 folds × 4 pipelines = 144 fits；每管线 36 fits。

## Gate

V1−A0 与 CV1−C1 分别要求：九个 seed-repeat 的 mean Macro-F1 ≥ +0.005；至少 2/3 base seed 为正；至少 6/9 seed-repeat 为正；至少 8/12 split cell 非负；最差 split ≥ −0.03；平均 CE 与 Brier 不劣；每个 base seed 的 senior recall delta ≥ −0.02。

交互 gate 要求：mean interaction Macro-F1 > 0；至少 2/3 base seed 为正；至少 5/9 seed-repeat 为正；至少 8/12 split cell 非负；最差 split ≥ −0.03；交互 CE gain 与 Brier gain 非负；每个 base seed 的 senior recall interaction ≥ −0.02。正交互只在两个简单效应 gate 都通过时作机制解释。

## 执行顺序

1. 运行测试与 CPU-only preflight。
2. 若 topology、cache、initial-state pairing、reconstruction 或 gradient audit 任一失败，记录 NO-GO 并停止。
3. 只有 CPU preflight 写出 `GO_FOR_FORMAL_GPU_RUN` 且研究总监另行授权后，才能开始 144-fit GPU run。
4. 正式结果必须完整报告 V1−A0、CV1−C1、交互、CE、Brier、senior recall 与 split stability；不得访问 outer test。
