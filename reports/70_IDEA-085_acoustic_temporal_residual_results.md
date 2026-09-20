# IDEA-085：声学局部时序残差结果

日期：2026-09-19  
正式结论：**本轮不升级主模型。T1 相对同期静态声学 C1 有小幅 Macro-F1 收益，但没有超过 AST-only A0，也没有优于固定联合帧乱序对照 J1；三条预注册比较各自判定为 FAIL。**

## 1. 执行与完整性

- 完成 A0、同期静态声学 C1、真实顺序时序残差 T1、固定联合帧乱序 J1 × 3 base seeds × 3 repeats × 4 folds = **144/144 fits**。
- T1/J1 使用 6 条逐帧声学轨迹及 6 条有限值指示器；训练折内完成填补与标准化。时序分支为两层轻量一维卷积，局部感受野 7 帧，中心跨度约 60 ms，输入帧约 64 ms；因此它是局部动态编码器，不是长程轮廓或发声阶段模型。
- 首个四管线 cell 先单独运行并暂停；首 cell 工程审计与总监检查通过后，按总监授权续跑其余 fits。独立首 cell 审计随后在已有 27 fits 时完成并确认无阻断。
- 全量结束后再次执行 canonical `--resume`：没有重训任何 fit，144 份 fit summary 及全部预测文件组合哈希保持不变，正式汇总可重复生成。
- 独立审计不导入 runner 或 aggregate，而是从原始 validation call/animal CSV 复算。共核对 144 份 fit summary、288 份预测文件，逐 call 重建 144/144 份 cat 概率；正式汇总逐字段差异数为 0（容差 `1e-12`）。联合专项测试 **21/21 passed**。
- 未生成、读取或评分 outer-test prediction/metric；`outer_test_accessed=false`。

## 2. 模型原理与文献来源

AST 主干在四条管线中冻结。C1 把既有 20 维逐叫声声学摘要投影为有界残差；T1/J1 则输入六条逐帧轨迹及对应 finite mask，经两层一维卷积、masked mean 和零初始化的 128 维投影后注入 AST 表征。三条声学分支都沿用 `0.25 × stopgrad(RMS(h)) × tanh(residual)` 的残差上限。T1 与 J1 的结构、参数和初始状态相同，唯一区别是 T1 保留真实联合帧顺序，J1 使用按 call-ID 固定的联合帧置乱。

完整文献与缓存依据见 [报告 66](./66_IDEA-085_trajectory_literature_and_cache.md)。[Ryu et al. (Interspeech 2025)](https://www.isca-archive.org/interspeech_2025/ryu25_interspeech.html) 为显式 pitch contour 建模提供人类语音任务的工程动机；[Schötz and van de Weijer (Speech Prosody 2014)](https://www.isca-archive.org/speechprosody_2014/schotz14_speechprosody.html) 显示猫叫语境间存在可感知的 F0 轮廓差异。这些文献支持“保留时间顺序值得测试”，但都不是猫年龄预测的直接证据。

## 3. 四条管线的 seed×repeat 等权均值

| Pipeline | Accuracy | Macro-F1 | BA | kitten recall | adult recall | senior recall | CE ↓ | Brier ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 AST only | **0.72222** | **0.73622** | 0.75741 | 0.88889 | **0.71667** | 0.66667 | **0.68992** | **0.41245** |
| C1 summary residual | 0.71242 | 0.72547 | 0.75093 | 0.88889 | 0.70278 | 0.66111 | 0.70649 | 0.41817 |
| T1 real-order temporal | 0.72059 | 0.73328 | **0.77037** | **0.93056** | 0.69722 | **0.68333** | 0.71653 | 0.42573 |
| J1 joint-frame shuffled | 0.71895 | 0.73371 | 0.76852 | **0.93056** | 0.69722 | 0.67778 | 0.71238 | 0.42179 |

这些是 9 个 seed×repeat 估计的等权均值，也是本报告的主表。Accuracy、BA、三类 recall、CE 与 Brier 是预注册辅助剖面，不参与 Macro-F1 分类 gate。

## 4. 三条独立的预注册分类 gate

每条比较分别要求：平均 `ΔMacro-F1 ≥ 0.005`；至少 `2/3` 个 base-seed 均值为正；至少 `6/9` 个 seed×repeat 为正；至少 `8/12` 个 split-cell 非负；最差 split `≥ −0.03`。本实验不存在单一“全局 gate”。

| 比较 | mean ΔF1 | seed×repeat 正/tie/负 | 正 base seed | 非负 split | 最差 split | 结果 |
|---|---:|---:|---:|---:|---:|---|
| T1−C1 | +0.007810 | 5/0/4 | 3/3 | 8/12 | −0.057772 | **FAIL** |
| T1−A0 | −0.002938 | 4/0/5 | 1/3 | 8/12 | −0.063373 | **FAIL** |
| T1−J1 | −0.000427 | 2/4/3 | 2/3 | 8/12 | −0.031898 | **FAIL** |

### T1−C1：有小幅分类收益，但稳定性不足

T1 的平均 Macro-F1 比本轮 C1 高 `0.007810`，三个 base-seed 均值全部为正，并同时改善 Accuracy `+0.008170`、BA `+0.019444`、kitten recall `+0.041667` 与 senior recall `+0.022222`；adult recall 下降 `−0.005556`。612 次成对重复动物出现中，T1 纠正 21 次、引入 16 次，净 `+5`。

这项正向证据必须保留，但只有 `5/9` 个 seed×repeat 为正，且最差 split 为 `−0.057772`，所以 gate 明确失败。概率质量也没有同步改善：CE gain `−0.010036`、Brier gain `−0.007569`，负值表示 T1 更差。

### T1−A0：没有超过主基线

T1 相对 A0 的平均 Macro-F1 为 `−0.002938`，仅 `1/3` 个 base-seed 均值为正，`4/9` 个 seed×repeat 为正，最差 split 为 `−0.063373`，因此失败。BA `+0.012963`、kitten recall `+0.041667` 与 senior recall `+0.016667` 是保留的辅助正向结果；但 Accuracy `−0.001634`、adult recall `−0.019444`、CE gain `−0.026604`、Brier gain `−0.013288`，纠错总账为 12/13/−1，不能支持替换 A0。

### T1−J1：真实顺序没有优于乱序对照

T1 与 J1 的平均 Macro-F1 几乎相同，但略低 `−0.000427`；只有 `2/9` 个 seed×repeat 为正，另有 4 个 ties，最差 split `−0.031898` 也略低于锁定下限。T1 的 BA 高 `0.001852`、senior recall 高 `0.005556`，纠错总账净 `+1`，但 CE 与 Brier 分别更差 `0.004150` 与 `0.003948`。因此没有建立真实局部顺序相对容量匹配乱序控制的优势。

## 5. 同期上下文与解释边界

- 本轮 C1−A0 的 Macro-F1 为 `−0.010748`，Accuracy `−0.009804`、BA `−0.006481`，CE/Brier 也更差。这只是当前随机种子与训练矩阵的同期上下文，不能据此推翻历史 C1 结论。
- T1−C1 同时改变了固定 summary 与局部轨迹编码方式，差异不能纯归因于“时间顺序”。
- T1−J1 的固定联合帧置乱会同时改变六通道数值、缺失性与 voicing 的局部组织；单个固定控制既不是纯 F0 顺序隔离，也不是完整因果证明。
- 这轮只检验约 60 ms 中心跨度的局部动态。它不支持“时序/F0 方向整体无用”的结论，也没有测试更长程的轮廓、阶段结构或其他事前定义的时序模型。
- 每条管线 pooled validation 的 612 行是跨 seed、repeat、fold 重复评估产生的 **612 次动物出现**，实际为 97 只 unique cats，不能当成 612 只独立猫。

## 6. 决策

1. **不升级主模型，主候选仍为 A0/C1**：T1 未超过 A0，也未超过 J1，且 CE/Brier 更差。
2. **冻结本轮结果**：不根据结果追加卷积宽度、残差权重、种子、置乱方式或 gate 搜索。
3. **保留有限正向证据**：相对本轮 C1，T1 的平均 Macro-F1、BA、kitten/senior recall 与净纠错均为正，但稳定性门槛未通过。
4. **不作机制声明**：本实验没有证明真实帧顺序、F0 顺序或声学缺失模式是改进来源。
5. **不改变既有 v3 结论**，也不启动新的 IDEA；任何后续时序假设都应另行事前锁定，并视为探索而非本轮的独立确认。

## 7. 固定结果哈希

- protocol：`592b8fccf721aeeab8f2bf20839534b703c9e5caa582c93dc220854d77f66bca`
- runner：`471bb5bd0e007be696decd86dec9317b3ef8c3f732490718a4eed99f28e821b4`
- trajectory extractor：`c2ae23b65afe6c4086fcd62271d67312573437e6d8b8ebbcd3c1ee048177df97`
- trajectory cache：`a69b246eba8d773f12a605f2caa9086e2ba0a2d87a3eeeb6ec57175c551a46a7`
- CPU preflight：`c76fcb1177ca4a14b432993efaf8b8ab2b4df99c148c3ca80405ad3456b67387`
- run manifest：`ddfc3436ade7ac4d8733cf36a6e288510d3900de6fcdd77a23eb7e334ff4ef90`
- initial evaluation summary：`4aa2ce4862c81fda49e35e4e77a2901fac01bcea9d52feeab4fb80fbc58c0f7a`
- compact run summary：`41d6e9df66e07bc30961e0cd308a674519e2eef8c72bd4df4e912dc63d048c48`
- first-cell audit：`ecbd0f339df888b815542b27bab7030c0c22199ebfa75e8738cc40561a06bf91`
- independent first-cell audit：`e10001a9b2a37c2ab31eac69b0045f11717fac714c11421bd6b3a9b8030553db`
- independent results audit：`c2d7d55ca0460acd6d564f3807b83d7de9226ed1a503247783083656824b7a51`
- independent audit report：`aa659db0ac106f7f35856b77aeb8f21c6843c05ed5f4964484245e2c562ffb39`
- second-resume combined fit/prediction snapshot：`01828a9856b94cfc7348a2081bfe22465ef653a861ba02bef9500ba0887856e8`
