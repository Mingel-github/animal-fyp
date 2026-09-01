# IDEA-050｜AST LoRA 低参数适配计划

> 制定日期：2026-09-01
>
> 状态：`proposal`，等待外部实验端实现与团队确认
>
> 阶段：formal-v2.1、IDEA-049 与 AST-HPO-v1 之后的 exploratory PEFT study
>
> 决策责任：团队成员与导师

## 1. 计划定位

IDEA-050 检验 LoRA 是否能在当前最强 AST head-only pipeline 上提供稳定的附加收益。该计划
只引入一种新 PEFT family，保留现有 probe-guided bottleneck adapter 作为历史参照，控制
毕业设计的实验规模。

核心问题是：

> 在相同 AST checkpoint、短叫声输入、classifier head、cat-ID splits 和 animal-level
> evaluation 下，attention Q/V projections 上的 LoRA 能否提高 tuned AST head-only 的
> macro F1，并保持 balanced accuracy 与 QWK？

## 2. 已有结果与证据边界

**Formal-v2.1 observation**：AST head-only 的 mean animal macro F1 为 `0.7238`；
probe-guided adapter 为 `0.7290`。Adapter 相对 head-only 的平均增量为 `0.0052`，9 次配对
中5次为正。

**AST-HPO-v1 observation**：inner-only小网格选择了dropout `0.445710...`和head learning
rate `0.006`。在base seed 17的三个complete OOF中，tuned head-only达到macro F1
`0.7488`、balanced accuracy `0.7644`、QWK `0.6469`；同轮adapter分别为`0.7367`、
`0.7592`和`0.6441`。

AST-HPO-v1的工程链条通过以下审计：

- search阶段只请求inner-train与inner-validation indices；
- selection在outer prediction前写入并保存protocol与runner hashes；
- evaluation覆盖792条calls和111只cats；
- 每条pipeline使用相同8-trial预算；
- formal-v2.1文件和结果保持原状。

该HPO结果属于可靠的exploratory candidate evidence。它仍有三个解释边界：

1. `0.006`位于本轮learning-rate search的上边界；
2. 完整评价当前只有base seed 17和三个split repeats；
3. 四个development folds复用同一批111只猫，每只outer-test cat也会在其他fold中承担
   train或validation角色。

因此，IDEA-050采用`0.006`作为provisional shared head learning rate，并在新的LoRA
execution中同期重跑matched head-only，使主要比较来自同一runner、环境、split和seed。

## 3. 假设与竞争解释

**Claim type**：predictive performance and efficiency comparison。

**主要假设 H050-L**：inner-only selection得到的LoRA recipe在配对complete-OOF evaluation
中取得高于matched tuned AST head-only的mean animal macro F1。

**Null model H050-0**：LoRA与head-only的paired differences以0为中心，观察变化来自split、
seed和有限动物样本的波动。

**Mechanism proposal**：LoRA通过低秩更新attention projection，重新组合pretrained AST
已经学到的time-frequency关系，同时将训练参数维持在远低于full fine-tuning的范围。

**竞争解释**：

1. Frozen AST representation与tuned MLP已经包含主要可用信息；
2. 新增低秩参数提高training fit，同时增加unseen-cat variance；
3. 收益来自head learning rate、epoch selection或CUDA轨迹；
4. 特定rank或layer范围偶然适配development splits；
5. LoRA改变类别precision/recall trade-off，macro F1与BA/QWK呈现不同方向。

## 4. 固定条件与matched control

以下项目固定：

- 数据：792条唯一calls、111只cats、kitten/adult/senior三分类；
- independence unit：`cat_id`；
- AST checkpoint与revision：沿用formal-v2.1；
- waveform：16 kHz mono、1.28秒segment、0.64秒hop；
- AST geometry：16×16 patch、frequency/time stride 10×10；
- classifier head：`768 → 128 → 3`、ReLU、BatchNorm、dropout `0.445710...`；
- head optimizer：Adamax，head learning rate `0.006`；
- class weights：由当前training calls计算balanced weights；
- epoch selection：最低inner-validation cross-entropy loss；
- animal aggregation：同一cat的call probabilities算术平均后取argmax；
- primary metric：animal-level macro F1；
- secondary metrics：balanced accuracy、QWK、per-class precision/recall/F1和confusion matrix。

核心pipelines为：

| Pipeline | Trainable components | 作用 |
|---|---|---|
| Tuned AST head-only | classifier head | matched control |
| AST LoRA | Q/V LoRA + 相同classifier head | primary candidate |

Formal-v2.1 adapter和AST-HPO-v1 adapter继续作为历史上下文。LoRA核心完成后，团队可决定是否
使用shared head LR `0.006`同期重跑adapter。

## 5. LoRA原理与注入位置

LoRA将冻结矩阵的更新写为低秩分解：

\[
W' = W + \frac{\alpha}{r}BA
\]

其中原AST矩阵 \(W\) 保持冻结，\(A\) 与 \(B\) 为可训练矩阵，rank \(r\) 控制adaptation
capacity。初始低秩增量设为0，使首次前向与matched frozen AST一致。

第一轮将LoRA注入self-attention的`query`和`value`projections。Q/V是常见的低参数起点：
query影响token选择关系，value影响被聚合的信息。Key、attention output和MLP projections进入
后续候选池。

## 6. 有界候选空间

为控制组合数量，第一轮固定`alpha = 2r`和LoRA dropout `0.05`，重点检查rank、作用层与
LoRA learning rate。

| Candidate | Target blocks | Projections | Rank | LoRA LR |
|---|---|---|---:|---:|
| L1 | last 4 | Q/V | 4 | `3e-4` |
| L2 | last 4 | Q/V | 8 | `3e-4` |
| L3 | all 12 | Q/V | 4 | `3e-4` |
| L4 | last 4 | Q/V | 4 | `1e-4` |
| L5 | last 4 | Q/V | 4 | `1e-3` |

该设计围绕L1分别改变一个主要因素，便于解释rank、placement和learning-rate signal。Smoke
阶段可以根据显存与数值稳定性缩小范围；最终candidate table在读取outer aggregate前冻结。

## 7. 执行流程

### Stage A｜Implementation audit

保存全部trainable parameter names、shapes和counts。检查AST backbone其余参数的
`requires_grad=false`。使用零增量初始化验证LoRA model与head-only在相同输入上的logits差异
处于声明的数值容差内。

### Stage B｜Inner-only smoke

使用repeat 0、fold 0、seed 17运行两个epochs，检查forward、backward、gradient、mixed
precision、checkpoint save/load、cat aggregation和metrics。日志记录
`outer_test_accessed=false`。

### Stage C｜Inner-only candidate selection

五个LoRA candidates在相同development partitions与seed上运行。选择指标为mean
animal-level macro F1；balanced accuracy、QWK、validation loss、trainable parameters和
训练稳定性用于tie-break与人工审阅。全部候选结果随selection lock一起保存。

### Stage D｜Initial paired evaluation

冻结一个LoRA recipe后，同期运行LoRA和tuned head-only：

- repeats：0、1、2；
- outer folds：0、1、2、3；
- base seed：17；
- 总量：2 pipelines × 3 repeats × 4 folds = 24 fits。

每个pipeline形成三套111-cat complete OOF。Initial结果用于判断LoRA是否具有继续扩展的信号。

### Stage E｜Seed expansion

团队确认后增加base seeds 43和101，使两条pipeline各形成9套complete OOF。扩展阶段保持
LoRA recipe、head recipe、runner和analysis不变。

## 8. 主要比较与评价

Primary contrast为：

```text
selected AST LoRA - matched tuned AST head-only
```

报告内容包括：

- 每套complete OOF的macro F1、balanced accuracy与QWK；
- paired differences、正向次数、mean、SD、range；
- hierarchical paired bootstrap interval；
- kitten、adult、senior的precision、recall与F1；
- confusion matrix与跨两级年龄错误；
- trainable/total parameters、peak VRAM、wall time和checkpoint size。

同期head-only控制运行环境与训练随机性。AST-HPO-v1 tuned head结果保留为历史参考，并用于
检查新runner是否处于相近性能区间。

## 9. 可区分结果

| 观察结果 | 支持的解释 |
|---|---|
| LoRA在多数paired OOF中提高macro F1，BA/QWK保持或提高 | Q/V低秩attention adaptation提供稳定附加信息 |
| LoRA与head-only集中在相同区间 | frozen AST与tuned head已利用主要任务信息 |
| LoRA均值提高且repeat/seed方差增大 | adaptation增加capacity，也提高split sensitivity |
| Last-4优于all-layer | 后层task-facing representation更适合低数据适配 |
| All-layer优于last-4 | 年龄任务需要跨深度调整attention representation |
| Macro F1提高而QWK下降 | 类别precision/recall改善伴随更多远距离年龄错误 |

结果解释同时考虑方向、幅度、uncertainty和效率。最终候选由团队与导师确认。

## 10. Adversarial review

第一个风险是同一批111只猫经过多阶段开发后形成adaptive reuse。报告使用“post-formal
exploratory repeated internal validation”定位，并保留完整search与decision provenance。

第二个风险是LoRA search获得新的选择机会。控制措施是有界五候选空间、outer前锁定、完整
报告全部trial，以及同期运行matched head-only。

第三个风险是HPO-v1的head LR `0.006`位于搜索上界。IDEA-050将其视为已选候选，并通过
matched control保持LoRA归因；更宽的head LR搜索属于独立后续实验。

## 11. 停止条件与交付物

完成LoRA与matched head-only的initial paired evaluation后达到最低完成范围。Initial signal
经团队确认后执行seed expansion。Q/K/V、attention output、MLP LoRA、prompt tuning和其他
PEFT families进入未来候选池。

外部实验端回传：

- LoRA protocol、candidate table和selection lock；
- runner revision、environment和hardware；
- trainable parameter audit与initialization-equivalence test；
- inner-only smoke和全部candidate summaries；
- matched head-only与LoRA fit summaries；
- complete-OOF probabilities和aggregate metrics；
- paired uncertainty、资源指标和deviation records。

正向结果形成新的低参数AST candidate；接近head-only的结果则界定LoRA在MeowAgeNet小数据
条件下的附加收益范围。
