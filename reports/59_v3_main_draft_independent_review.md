# MeowAgeNet v3 主稿独立证据审阅

日期：2026-09-18  
审阅角色：牛马2（独立证据审计；未编辑主稿）  
审阅对象：`reports/58_Formal_Research_Summary_v3.md`  
审阅对象 SHA-256：`23c628a2e09fcd19871588363aa447b2d3d984ce7aefcba4a1425d0327b7c748`  
metadata SHA-256：`465f7311039d7f67c6cd89605f59def17db8665e13328ca93d32b86a22b91828`  
独立账本 SHA-256：`489346cf3f82dc60d5223fdea2836db2c624ef91629db0938c5da3db0702eb3a`

## 审阅结论

**REVISE REQUIRED，当前不予 PASS。**

主稿的实验数值主体与源文件一致：A0/C1 各轮、IDEA-077 至 IDEA-082、CatMeows、犬 benchmark 和 IDEA-081 的 C1 重算均未发现数值冲突。阻止 PASS 的问题集中在证据术语、Adapter 确定性诊断的命名、全轮次 C1 描述性汇总完整性、链接锚点和审阅状态管理。

## 必须修订项

### R1：按四类术语重新标注证据状态

当前稿把 C1 标为“主要研究候选 / Tier 2”，把 probe-guided Adapter 仅列为“探索性历史证据”。这与预先锁定的独立术语账本不一致。

统一口径应为：

- **primary candidate**：冻结 AST representation + head 路线，相对 VGGish 的总体路线证据；A0 可继续称“当前主模型/稳健参照”。
- **exploratory positive**：C1 的同一 111 猫多轮正均值信号，以及 G3 的 utility 正向线索；必须同时写明 IDEA-076 或信息门失败。
- **negative result**：probe-guided Adapter 的独有/独立增量，以及 IDEA-077 至 IDEA-081 预注册候选或机制门失败；局部正值不改变该状态。
- **external boundary**：CatMeows 情境迁移和犬年龄子集；不得与 MeowAgeNet 年龄主任务数值合并。

可以保留“C1 是下一步外部验证中优先配对的研究对象”这一研究优先级，但不能让“主要研究候选”被误读为 `primary candidate` 或确认性模型层级。

### R2：更正 Adapter 的两个负差口径

`−0.0091` 与 `−0.0057` 的统计身份不同：

- `−0.0091`：原 formal-v2.1 base seed 17 在原实现未强制 deterministic CUDA / math attention 时的 **same-seed nondeterministic rerun**。它用于暴露计算非确定性，报告 35 明确规定不得作为额外效果估计。随后开启确定性算法后的同一 fit 才逐位一致。
- `−0.0057`：base seeds 151/307/509 的 **deterministic new-seed replication**，head `0.7253`、Adapter `0.7195`，`5/9` 正、`4/9` 负；这是可用于稳定增量判读的复验结果。

因此，执行摘要第 14 行、表格第 138 行及 metadata 中的 `same_seed_deterministic_rerun` 均应改成“同种子非确定性复跑/计算确定性诊断”，并明确 `−0.0091` 不是独立效应估计。

### R3：补齐 formal-v2.1 的两条锁定 contrast

主稿已经正确给出均值与方向，但完整证据解释应同时列出：

- H048，Adapter−VGGish：`+0.0764758`，`9/9` 为正，hierarchical 95% CI `[-0.006354, +0.168513]`。
- H019，Adapter−head-only：`+0.0051946`，`5/9` 为正，hierarchical 95% CI `[-0.045810, +0.058502]`。

两条层级 bootstrap CI 均跨零。H048 支持的是 AST 路线的正向证据，不能据此把整体增量归因于 Adapter；H019 与新种子确定性复验共同构成 Adapter 独有增量的负结果。

### R4：说明跨轮 C1 汇总截止范围

现稿逐行覆盖 IDEA-071/072/073/074/076/080/081，IDEA-081 `C1−A0=+0.008553493977`、`6/0/3` 经九个 seed×repeat 配对重算正确。

现有汇总只到 IDEA-076：54 个同数据单元、`+0.006368181557`、`30/6/18`。若 v3 声称给出“截至 IDEA-081 的完整跨轮描述性汇总”，还应列出纳入 IDEA-080/081 因子实验控制臂后的 72 单元描述性值：`+0.007207031785`、`41/6/25`。无论是否补 72 单元，都必须明确这些单元反复使用相同 111 猫，只能作描述性库存，不能作独立样本、显著性或确认性证据。

### R5：修复 6 个失效锚点

95 个本地链接的文件目标全部存在，但 83 个带锚点链接中有 6 个失效，归为 3 种目标：

- `07_IDEA-019_PEFT_placement_results.md#结果总览` 出现 2 次，应改为 `#1-结论摘要`。
- `09_formal_protocol_v2_freeze.md#证据身份与限制` 出现 3 次，应改为 `#2-pilot-与正式-v2-的证据边界`。
- `57_IDEA-082_age_acoustic_mechanism_group_ablation_results.md#g12f0--stability` 出现 1 次，应改为 `#g12f0-stability`。

### R6：更正独立审阅状态

当前 metadata 在独立审阅完成前使用 `complete_source_audited`。修订稿进入二审前应使用 `pending_independent_review`；只有全部修订项核销并重新核对主稿、metadata、链接和哈希后，才可改为 `complete_independently_audited`。

## 已通过项目

- 主稿 SHA 与 metadata 中记录的 report SHA 一致。
- metadata 中 27 个源文件 SHA 均与当前文件一致，无缺失。
- 95 个本地链接的文件路径全部存在。
- IDEA-081 C1−A0 精确重算为 `+0.008553493977`，方向 `6/0/3`。
- IDEA-077 至 IDEA-081 的主比较和 gate 结论与源文件一致。
- IDEA-082 G3 `real−A0=+0.016105606`、`real−shuffled=+0.019658297`，utility PASS、information FAIL、`source_supported=false` 的表述正确。
- CatMeows 被限定为 440 录音、21 猫、三类情境分类，不被当作猫年龄验证。
- 犬 benchmark 被限定为 2,290 units、125 狗、五年龄阶段的低优先级跨物种子集；没有用 dummy 下限结果挽救 C1 主 gate。
- 主稿未把 seed、repeat、fold 或重复出现的猫写成新增独立动物。
- 本次审阅未启动 GPU，也未覆盖 v2/v2.1 或直接修改 v3 主稿。

## 二审 PASS 条件

牛马1修订后，必须以新 SHA 重新核对 R1–R6。只有六项全部关闭，且 report hash、metadata hash、源哈希与锚点检查均通过，才能给最终 PASS。
