# Formal-v2.1 Runner 与 Inner-only Smoke 记录

> 日期：2026-08-27  
> 基础 Git 提交：`36f2914bf52afc1c3949f8d35dc8d7efc087955d`  
> 协议：`meowagenet-formal-v2.1`  
> 状态：runner 已通过 inner-only smoke，等待 execution lock 共同确认

## 1. 本轮完成内容

新增以下执行材料：

- `scripts/run_meowagenet_formal_v2_1.py`：三个正式核心 pipeline 的统一 runner；
- `configs/experiment/meowagenet_formal_v2_1_probe_guided_candidate_v1.json`：候选 recipe；
- `tests/test_formal_v2_1_runner.py`：runner 边界、recipe 和依赖 hash 测试。

Runner 支持两个明确分开的 scope：

- `inner-only`：只使用当前 repeat-fold 的 inner-train 和 inner-validation；
- `formal`：执行 inner epoch/layer selection、outer train+validation refit 和 outer-test prediction。

`formal` scope 必须先读取状态为 `locked_before_formal_outcomes` 的 execution lock，并核对 recipe SHA256、runner SHA256 和非空环境字段。尚未锁定的模板会在创建运行目录前被拒绝。

## 2. Recipe 已明确的实现口径

三个 core pipelines 为：

1. `vggish_mlp`：唯一性能 baseline，继承 locked-v1 VGGish 预处理和 MLP recipe；
2. `ast_head_only`：冻结 AST，只训练 matched classifier head；
3. `ast_probe_guided_adapter`：冻结 backbone 主参数，在 inner-train 选出的两个 block 后训练 width-32 residual bottleneck adapters，并训练 matched classifier head。

AST segment pooling 已明确为：先对同一 call 的 AST `pooler_output` embeddings 做算术平均，再送入分类头。该定义消除了父协议中“embeddings or probabilities”的实现歧义。

模型 seed 使用：

```text
full_seed = base_model_seed + 10000 × repeat + 100 × outer_fold
```

三个 pipeline 在相同 repeat、fold 和 base seed 下使用相同 full seed。

## 3. Inner-only smoke 范围

本轮 smoke 配置：

- split repeat：0；
- outer fold：0；
- base model seed：17；
- full seed：17；
- pipeline：三个 core pipelines；
- smoke epoch：每个 pipeline 1 epoch；
- AST device：NVIDIA GeForce RTX 4060 Ti；
- outer-test access：`false`。

Probe 只使用该 fold 的 66 只 inner-train cats、517 个 calls，通过三折 cat-level CV 选出 AST 第 8、11 层。

## 4. Smoke 结果

| Pipeline | Inner-validation animal macro F1 | 主要执行核验 |
| --- | ---: | --- |
| VGGish + MLP | 0.2929 | TensorFlow 数据流、标准化、class weight、MLP 训练和验证通过 |
| AST head-only | 0.5395 | 冻结 AST embedding、matched head 和 PyTorch 验证通过 |
| Probe-guided adapter | 0.6296 | AST 前向、adapter 反向传播、梯度累积和验证通过 |

这些数值来自单个 inner-validation fold 且只训练1 epoch，用途是确认代码链路健康。它们不进入 formal OOF 比较，也不承担模型选择证据。

Adapter smoke 审计与 pilot recipe 对齐：

- adapter trainable parameters：99,904；
- adapter + head trainable parameters：198,979；
- peak allocated CUDA memory：611,393,536 bytes，约583 MiB；
- 1 epoch adapter inner training：约1.61秒。

## 5. 边界核验

smoke 输出目录中只有：

- run manifest；
- layer-probe 记录；
- 三个 inner fit summary；
- run summary。

没有 outer-test prediction 文件。`run_summary.json` 明确记录：

```json
{
  "outer_test_predictions_produced": false,
  "formal_aggregate_written": false
}
```

正式 guard 也已单独测试：把 `template_not_locked` 的 execution-lock 模板交给 `formal` scope 时，runner 返回错误并保持运行目录不存在。

## 6. 测试与当前 gate

仓库测试结果：

```text
16 passed
```

当前 runner 和 candidate recipe 已达到 execution-lock review 阶段。下一 gate 是共同确认：

- primary adapter 是否继续使用 probe-guided recipe；
- repeats 是否使用最低正式 core `[0, 1, 2]`；
- model seeds 是否保持 `[17, 43, 101]`；
- optional modules 是否保持空列表；
- runner、recipe、Git revision 和硬件信息是否写入正式 lock。

在该 gate 完成前继续使用 `inner-only` scope；锁定后才启动 formal outer-test aggregation。
