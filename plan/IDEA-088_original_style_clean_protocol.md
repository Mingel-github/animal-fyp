# IDEA-088：clean111 投影的原 baseline 风格对照

## 1. 目的与证据边界

本实验回答：在同一组由原 categorical notebook 日志恢复的 post-swap 猫角色上，采用原 final 训练配方时，当前 frozen-AST A0、声学 bounded residual C1 与 VGGish MLP 的表现如何；尤其检验同 cell 的 `C1 − A0`。

这不是原论文的精确复现，也不是外部验证。原因包括：

- 数据使用项目自行清洗的 792 条 calls / 111 个 analysis cat IDs，删除与 `041A` 字节相同的 alias `049A`；
- 原 VGGish 输入单位是 936 个 clean embedding rows，AST 输入单位是 792 个 calls；
- 角色由原 937-row 日志恢复后删除 `049A` 投影而来；不重新运行 SGKF；
- VGGish 保留 TensorFlow/Keras 原结构，A0/C1 保留 PyTorch 实现，跨框架初始化和数值轨迹不可能逐位一致；
- `000A`、`046A` 被原代码强制留在训练侧，替换猫可能重复进入测试，因此结果不是 111-cat complete OOF。

历史 VGGish 三分类 F1 `0.700136` 只作背景，不当成本实验 clean 分组的新成绩。

## 2. 冻结数据与角色

- AST：`runs/ast_locked_v1/gpu_rerun_2026-08-26/ast_standard_call_embeddings.npz`，792 calls、111 cats。
- 声学特征：IDEA-068 固定的 20 维 call 特征；缺失填补及 mean/std 每 fit 只由训练 calls 计算。
- VGGish：原 CSV 删除 `049A` 后 936 rows。输入严格使用 128 个 embedding 列加 `mean_freq`，共 129 维；`gender`、`target`、`cat_id` 和派生 `age_group` 不作为特征。
- 标签统一为 `0=kitten, 1=adult, 2=senior`，使用原 target 半开边界：`[0,0.5)`、`[0.5,10)`、`[10,20)`；逐猫必须与 AST 标签一致，clean111 类别数必须为 15/62/34。
- 角色清单由独立协作者从固定旧日志恢复，路径为 `runs/meowagenet_idea088_original_style_clean_v1/roles/original_clean_roles.json`。runner 只消费清单，不导入或调用随机分组器。
- 五个 split seeds：`7270, 860, 5390, 5191, 5734`；每 seed 四折，共 20 cells。
- 每 cell 必须 train/test 猫零交集且并集等于 clean111；`049A` 不得出现；`000A`、`046A` 不得进入 test。

## 3. 三条固定模型

1. `vggish_mlp`：129→128 Dense、ReLU、BatchNorm、dropout、3 类输出。直接复用现有 formal runner 的 TensorFlow/Keras 构建函数。
2. `A0_ast_only`：现有 frozen 768 维 AST embedding、train-only 标准化、128 hidden head。
3. `C1_bounded_wide_additive`：A0 加现有 `20→60→128` 声学 residual，`0.25 × stopgrad(RMS(h)) × tanh(r)` 限幅；声学输出层零初始化。

A0/C1 在同 cell 使用相同公共主头初态、相同每 epoch call 顺序和相同 dropout RNG 流。基础 model seed 固定为 `split_seed + fold`，post-build training seed 固定加 `1,000,000`。这是 IDEA-088 的显式适配，不冒充原 TensorFlow seed 的逐位复现。

## 4. 原 final 风格训练配方

三模型统一：

- Adamax：lr `0.003109800273709165`、epsilon `1e-7`、betas `(0.9,0.999)`、weight decay `0`；
- dropout `0.44571035356880917`；
- batch size 128 prediction units：VGGish 为 rows，A0/C1 为 calls；
- 最多 1500 epochs；无独立 validation；
- early stopping 监控训练过程累计的 epoch class-weighted CE，`min_delta=0.001`、`patience=30`、restore recorded best；
- 每条训练 unit 权重为 `N/(3*N_class)`，每 batch 的加权 CE 除以 batch 样本数；epoch loss 再按 batch 样本数加权；
- FP32、无 AMP、无 gradient clip、无 HPO；
- normalization、声学缺失填补、类别权重只拟合当前训练角色；
- NaN/Inf loss 立即作为工程失败；不得按分数择停、补 seed 或扩展设置。

CPU preflight 必须列出 20 cells 的 AST 训练 calls 与实际 batch sizes。若任何 cell 出现单 call 尾批，禁止开训并先由总监确定 tail policy；不得 drop_last。

## 5. 测试访问与保存

每 fit 先完成训练、选择 best training-loss epoch、恢复并持久化 checkpoint；新建同结构模型复载并核对 state digest 后，才能对 test units 预测一次。测试标签可保留用于最终评分，但不得参与归一化、权重、epoch 或模型选择。

每 fit 保存：

- checkpoint lock 与模型权重；
- unit/cat 概率；
- train-only normalization、声学填补和类别权重；
- 每 epoch 完整 unit 顺序、loss、best/stopped epoch；
- 初态、复载、输入角色及所有关键文件哈希。

## 6. 指标与汇总

公开主指标统一为：每猫对 rows/calls 概率作算术平均后计算 Macro-F1。同步报告 Accuracy、balanced accuracy、三类 recall、cross-entropy、Brier。

原生输入单位指标另列：VGGish row Macro-F1 与 AST call Macro-F1，二者不相互混同，也不与猫级主指标混同。

汇总顺序：

1. 每 pipeline 的 20 个 fold 结果；
2. 每个 split seed 对四折等权平均，得到 5 个 seed-level 估计；
3. 报 5 个 seed-level 均值和样本 SD；
4. 对 A0/C1 在相同 seed×fold 内作差，再形成 5 个 seed 的四折平均差及其均值、样本 SD、正/平/负方向。

20 folds、重复测试猫和 5 seeds 都不当作新增独立动物。由于强制训练猫导致测试覆盖不完整，禁止使用 `complete OOF` 表述。

## 7. 分阶段门禁

1. 代码、协议、角色 schema、专项测试与 CPU preflight 完成并交叉审查；不训练。
2. 总监绑定准确 preflight SHA 后，仅授权 `A0_ast_only / seed 7270 / fold 0` 一个技术 fit。
3. 技术 fit 的 checkpoint、全训练覆盖、tail batch、复载、预测与受保护历史文件审计通过后，总监另行授权剩余 59 fits。
4. 仅工程错误可恢复；不能根据成绩停止、替换或扩展。
5. 完成 60/60 后才运行 aggregate，生成 IDEA-088 metadata 与报告 82；旧报告、旧 splits、旧 runs 和旧 metadata 不修改。
