# IDEA-087：A0/U1/C1 嵌套参数优化 CPU 预检

日期：2026-09-19  
状态：**CPU GO；GPU 尚未授权。**

## 1. 锁定目标与预算

本轮在相同预算下分别为 A0、U1、C1 搜索共同训练参数，不改变三种模型结构。固定 8 行网格为：learning rate `{0.006, 0.003}` × dropout `{0.44571035356880917, 0.25}` × Adamax weight decay `{0, 0.001}`；循环顺序为 learning rate → dropout → weight decay，`q00` 是原配置。

只搜索上述三项。U1/C1 声学宽度固定 60，C1 cap 固定 0.25，分支学习率倍率固定 1；不追加网格、种子或结果驱动阈值。

- Inner：`8 configs × 3 pipelines × 4 outer folds × 3 inner folds × 2 search seeds = 576 fits`。
- Outer：`original/selected × 3 pipelines × 4 folds × 3 refit seeds = 72 fits` 上限。
- 总上限 648 fits；selected=q00 时用显式 alias 指向同一 fresh q00 refit，不制造第二个物理 fit。

IDEA-076 同类 GPU fit 的历史均值/中位/P90 训练时间约 `2.07/1.91/3.20` 秒，按均值估算 648 fits 的纯训练约 0.37 GPU 小时；加上预测、I/O 与审计后仍属于可执行的小型矩阵。

## 2. 严格数据隔离

使用既有 repeat 0 的四个 outer folds。每折把原 train+validation 合并为 outer-dev，原 test 只用于最终 outer 评分：

| outer fold | outer-dev cats | outer-test cats | outer-dev adult/kitten/senior |
|---:|---:|---:|---:|
| 0 | 83 | 28 | 46/11/26 |
| 1 | 83 | 28 | 46/11/26 |
| 2 | 83 | 28 | 47/11/25 |
| 3 | 84 | 27 | 47/12/25 |

每个 outer-dev 先按 `cat_id` 稳定排序，再用独立固定 seed 做猫级三折分层。生成的 `inner_roles.csv` 有 999 行：每只 outer-dev 猫在三个 inner folds 中恰好一次 validation、两次 train。同猫全部 calls 始终同角色；12 个 inner validation 均含三类，validation 猫数为前三个 outer folds 的 `28/28/27` 和第四折的 `28/28/28`。

预检逐 cell 核对 train/validation 猫与 calls 无交集、合并完整覆盖 outer-dev、三类齐全，并固定训练/验证 cat-ID hash、call-index hash、call 类别计数与训练 call 权重。outer-test 预测没有生成。

## 3. 损失、选择与 refit 语义

原版“globally class-balanced call CE”已核实为当前 fit 内计算：只用 training calls 的标签计数，类别权重为 `n_training_calls / (3 × n_training_calls_in_class)`；不是全数据权重，也不是猫级权重。AST 标准化、U1/C1 声学缺失填补与 mean/std 同样只拟合当前 fit 的 training calls。

Inner checkpoint 仍以未加权 validation animal CE 最低为准，epoch 为 1-based、范围 1..50。每个 config 在一个 pipeline×outer fold 中产生 `2 seeds × 3 folds = 6` 个 best epochs；取 median 后 half-up 四舍五入，作为完整 outer-dev 的固定 refit epoch。q00 与 selected 分别使用各自的六个 epochs。

每个 search seed 先拼接三折 validation，必须形成无重复、完整覆盖 outer-dev 的猫级 OOF；随后两个 seed 等权。选择规则固定为：

1. 候选池满足 `mean Macro-F1 ≥ best − 0.002 − 1e-12`；
2. 池内按 mean Brier 升序；
3. Macro-F1 sample SD（`ddof=1`）升序；
4. ordinary Accuracy 降序；
5. config ID 升序。

Macro-F1 明确使用 `labels=[0,1,2], average=macro, zero_division=0`；BA 为三类显式 recall 的算术均值；Brier 为每只猫三类概率与 one-hot 平方差之和再对猫取均值。

## 4. 模型与种子审计

- 参数量：A0 `99,075`，U1/C1 各 `108,143`。
- 三模型共同 AST head 初态哈希相同；U1/C1 完整初态相同；zero-init 下 U1/C1 相对 A0 初始 logit 最大差均为 `0`。
- q00 的 learning rate、dropout、weight decay 三项与原训练设置完全一致；原 Adamax 未显式传 weight decay，等价于 `0`。
- 2 个 search base seeds 派生 24 个 outer×inner full seeds；3 个 refit base seeds 派生 12 个 outer full seeds。两组各自唯一、彼此无交集，与 IDEA-068 至 IDEA-086 的保守历史 seed registry 零碰撞。
- 同一 stage/cell/seed 下，pipeline 与 config 复用共同 seed；模型构建后统一 reset RNG，训练 loader 使用相同 seed。实际 epoch batch history 会逐 fit 保存，首个 GPU fit 后再核对共同 epoch 的 cat-order 与 call-coverage hashes。

## 5. 分阶段 fail-closed 门禁

- Inner GPU 入口必须显式给出总监授权；支持 `--max-fits 1` 首 fit 技术检查和严格 resume。
- 只有 576 个 inner fit 全部存在、预测哈希/角色/身份/call→cat 重建均通过时，才可一次性生成含 12 个 pipeline×outer-fold 身份的 `selection_lock.json`。
- Outer 入口必须同时具备总监授权和命令行提供的精确 selection-lock SHA-256；还会复核 selection 顶层 protocol/runner/tests/inner-role hashes、12 个唯一身份及 `outer_test_accessed=false`。
- outer 授权与阶段转换另写持久记录；初始 manifest 明确只是创建时 outer 未访问，最终另写 `final_stage_manifest.json` 并把 outer access 记录为 true。
- selected=q00 alias 会逐项核对源 physical fit 的 pipeline/fold/seed/config/epoch、预测路径与哈希、audit 完全一致。
- fixed-epoch outer refit 不宣称 checkpoint reload；其 audit 明确标为 `checkpoint_reload_applicable=false`。

## 6. 测试与决定

专项测试覆盖网格/预算、seed 材料、稳定 inner roles、三类/OOF 覆盖、half-up、选择键与数值边界、指标公式、train-only call weights、模型参数/初态、q00 原版等价、outer 授权拒绝、selection 身份拒绝和 fixed-epoch reload 语义。结果：**12/12 passed**。

CPU preflight 结论为 **GO**。这只表示锁定实现满足嵌套隔离和工程门禁；GPU 仍需独立只读复审与总监单独授权，且首次只允许一个 inner fit，不根据成绩择停。

## 7. 固定产物哈希

- plan：`5770ade41409f28340513d0f6a282a8561cb89adf44222b453a5b64cc1f3faf0`
- protocol：`ff74e49579afa5a6cad6d4e1ad14894817286924a84af78787dc00191ace2786`
- runner：`6152100d5a2e53420aa4877674a63d4163b8cc7a26d157d227491b35e1d57fe7`
- tests：`8ae5687f96abdb0fc1d587d2ab7b8e26cec582118dae34db94cba8b8da089ee1`
- independent design review：`25e40ce2459c96b89c36ddf2f45e0d8e1e41368b56a52be40a1a9982a25b0268`
- inner roles：`70a2f95d65a9a1b584e6d5b45ee9eb450d2c535c3e8c44a4bc9ae0fdf7fd3c11`
- CPU preflight：`9f0b3c494679fbcc41deaafc7ea99a577a9990ff96906d618876b414779253be`
