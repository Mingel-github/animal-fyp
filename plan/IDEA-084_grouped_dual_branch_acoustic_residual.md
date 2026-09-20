# IDEA-084：分组双分支声学残差

日期：2026-09-19  
状态：结果盲协议准备；仅允许 CPU preflight，GPU 未授权

## 研究问题

IDEA-082 显示 15 维 `source-related` 组与 5 维 `spectral-energy` 组各自都不足以完成来源归因，但 G3 提供了最强探索性方向。IDEA-084 不再问“哪一组单独有效”，而问：在保持 C1 总 RMS 上限不变时，把两类声学输入交给独立的小分支、再做固定等权合成，能否比原样 C1 更好地利用互补信息。

本实验是受既有结果与文献启发的新探索性方法，不是外部确认，不重开 IDEA-076 的失败结论，也不是 Taylor–Reby 或 GeMAPS 论文原配方的复现。

## 固定四管线

1. `A0_ast_only`：冻结 AST 表示与原样分类头。
2. `C1_bounded_wide_additive`：原样 20→60→128 的 C1，当轮配对重跑。
3. `P1_grouped_dual_branch`：真实 15/5 分组双分支。
4. `R1_hash_random_dual_branch`：固定哈希产生的 15/5 混合分组双分支。

三个新 base seeds 为 `[3583, 5080, 9355]`，每个 `3 repeats × 4 folds`。总预算为 `4 × 3 × 3 × 4 = 144 fits`。outer test 不加载、不预测、不计分。

## P1 与 R1

P1：

- `source-related`：索引 `0..14`，包含 F0、周期性、voicing、HNR 与 amplitude variation 代理；
- `spectral-energy`：索引 `15..19`，包含 RMS、spectral tilt 与 spectral flatness。

上述名称是操作性分组，不是物理 source–filter 解耦。5 维组没有 formant；spectral tilt 同时受声源、声道滤波与录音链影响。

R1 由固定字符串 `IDEA-084-fixed-random-grouping-v1` 在结果前生成。对每个特征索引 `i` 计算 `SHA256(material + "|feature_index=" + i)`，按 digest 排序；前 15 个为 A 组，其余 5 个为 B 组：

- A：`[0,1,2,4,5,8,9,10,11,13,14,15,17,18,19]`；
- B：`[3,6,7,12,16]`。

两组都同时含 `source-related` 与 `spectral-energy` 特征，且与 P1 真分组不同。R1 只控制双分支容量与固定分组结构，单个随机分区不能构成完整因果证明。

## 冻结公式与预算

两分支均为 `d→32→128`：

\[
u_A=W_{2,A}\operatorname{GELU}(W_{1,A}a_A+b_{1,A})+b_{2,A},
\]

\[
u_B=W_{2,B}\operatorname{GELU}(W_{1,B}a_B+b_{1,B})+b_{2,B},
\]

\[
r=0.25\,\operatorname{stopgrad}(\operatorname{RMS}(h))
\left(0.5\tanh(u_A)+0.5\tanh(u_B)\right),\qquad h'=h+r.
\]

两个末层均零初始化。每一通道的总写入上限仍为 `0.25×RMS(h)`，不是每分支各自拥有完整 0.25 预算。P1/R1 的声学支路参数为 `9,152`，总可训练参数 `108,227`；C1 声学支路为 `9,068`，总参数 `108,143`，差 `84`。

## 预注册比较

- `P1−C1`：回答分组双分支是否优于原样 C1；这是主方法比较。
- `P1−A0`：回答新方法是否有实用效用。
- `P1−R1`：回答文献/机制启发的特定 15/5 分组是否优于同容量固定随机分组。
- `C1−A0`：只作当轮同时对照的描述性上下文。

每条主比较分别报告：Macro-F1 均值/中位数/标准差、正平负、base-seed 方向、12 个 split cell、最差 split、balanced accuracy、animal CE、Brier、senior recall 安全性，以及 baseline 错误被纠正、candidate 新增错误和净纠正数。不得只用一条总 gate 覆盖平均收益、纠错、概率质量与稳定性。

## 禁止事项

- 不把同一 111 猫上的新 seed 称为独立样本或外部验证；
- 不用历史 C1 数值替代本轮配对 C1；
- 不在结果后改分组、权重、宽度、门槛或 seed；
- 固定 144 fits 后不追加 weight/seed 搜索；
- 不因 P1 优于一个随机分组便声称完整因果机制成立；
- 不覆盖 v3、IDEA-076 或 IDEA-082 的既有协议和结论。

## 执行顺序

1. 文献、协议、runner、tests 与 CPU preflight；
2. 总监与牛马1只读审查；
3. 只有总监明确授权后才能用 `--director-authorized --device cuda` 启动正式运行；
4. 正式结果完成后不追加结果驱动实验。
