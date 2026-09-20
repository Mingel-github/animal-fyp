# IDEA-085 声学时序残差：模型锁定与 CPU 预检

日期：2026-09-19  
状态：**CPU 预检 GO；正式 GPU 运行仍未授权。**

## 结论

IDEA-085 的 A0/C1/T1/J1 四管线、三个新基础种子、固定联合帧打乱、训练设置、汇总轴与分类门槛均已在查看任何新模型成绩之前锁定。实际数据 CPU 预检通过：`792 calls / 111 cats / 12 role cells`，预计 `144 fits`；CUDA 未初始化，outer-test 预测与指标均未访问。

该预检只证明实现符合锁定设计，不是模型效果结果，也不授权正式运行。

## 锁定比较

- `A0_ast_only`：原始 AST-only 对照。
- `C1_bounded_wide_additive`：同时代原始 C1；输入仍是旧 20 维逐叫声摘要。
- `T1_acoustic_temporal_residual`：六条原始声学轨迹加六条 finite indicators，经两层局部 1D 卷积、masked mean 和零初始化投影形成有界残差。
- `J1_frame_shuffled_temporal_residual`：与 T1 参数、初态、训练和预处理完全相同；唯一差异是每个叫声使用一个由完整 `call_id` 哈希固定的联合整帧排列，训练与验证均打乱。

主要比较为 `T1-C1`、`T1-A0`、`T1-J1`；`C1-A0` 只作同时代背景描述。`T1-J1` 检验真实局部顺序相对一个固定联合排列的价值，但不是完整因果证明，也同时涉及 F0 缺失/voicing 的时间组织，不可表述为纯 F0 效应。

## 模型边界

时序分支为：

`Conv1d(12,32,k=5,pad=2) + GELU + padding归零 -> Conv1d(32,32,k=3,pad=1) + GELU + padding归零 -> masked mean -> Linear(32,128)`。

投影层权重与偏置均零初始化。融合固定为：

`h' = h + 0.25 * stopgrad(sqrt(mean(h^2)+1e-8)) * tanh(r_seq)`。

两层卷积的局部感受野为 7 帧；10 ms hop 对应中心跨度 60 ms，单帧分析窗本身为 64 ms。这是最小局部时序编码，不建模长程整段轮廓、阶段顺序或全叫声状态转移。

参数审计结果：

| 管线 | 可训练参数 |
|---|---:|
| A0 | 99,075 |
| C1 | 108,143 |
| T1 | 108,355 |
| J1 | 108,355 |

时序分支为 9,280 参数，仅比 C1 总模型多 212 参数。

## 输入与预处理审计

轨迹缓存 SHA-256 为 `a69b246eba8d773f12a605f2caa9086e2ba0a2d87a3eeeb6ec57175c551a46a7`，包含 57,848 帧；每叫声帧数 `min/median/max = 9/70/443`。最长序列按 `443 × 10 ms` 计覆盖长度为 4.43 秒，而末帧时间点是 4.42 秒，两者只是口径不同。六通道顺序为 `log_f0, voiced_probability, periodicity, log_rms, spectral_tilt, spectral_flatness`。

- F0/periodicity 有限率均为 `0.8165537270`，其余四通道为 `1.0`。
- 20 个没有有效 F0 的叫声仍保留原生网格和其他通道，没有被删除。
- 每个原始通道只用当前训练角色帧拟合 finite median；插补后再拟合 mean/std。
- 六条 finite indicators 不标准化；不做逐叫声中心化。
- `time_seconds` 只供审计，不加载进模型。
- 尾部数值 call index 只用于 ragged retrieval，不进入 C1 或时序编码器；C1 前 20 列与旧缓存逐字节/NaN 一致。
- T1/J1 的 train-only median/mean/std 完全相同。
- `roles`、冻结 AST embedding、旧 20 维摘要和新轨迹缓存均在 `verify_protocol` 中直接按协议 SHA-256 fail-closed；本 runner/tests 的 SHA-256 也写入协议并直接核验。

缓存提取、旧 20 维统计重建与文献边界见 `reports/66_IDEA-085_trajectory_literature_and_cache.md`。

## CPU 预检结果

实际预检文件：`runs/meowagenet_idea085_acoustic_temporal_residual_v1/cpu_preflight.json`。

通过项：

- 角色隔离：3 repeats × 4 folds 全部 cat-disjoint；train/validation call 不重叠。
- 初始化：C1/T1/J1 相对 A0 最大 logit 差均为 `0.0`；共同 AST state 一致；T1/J1 完整可训练初态一致。
- 零初始化：C1 与 T1/J1 残差输出层均为零。
- 梯度：零投影时投影偏置有非零梯度而两层卷积梯度为零；把投影改成非零探针后，两层卷积梯度均严格大于零。
- 扰动上限：T1/J1 饱和探针的最大相对扰动均为 `0.2500000596`，低于 `0.25 + 1e-5` 容差界。
- padding：最短 9 帧与最长 443 帧同批时，短序列表示相对单独推理的最大差为 T1 `3.73e-8`、J1 `8.94e-8`。
- 打乱：固定排列可重复、非 identity、穷尽全部有效帧；六通道值与 finite 模式作为整帧元组共同移动，padding 不参与，未使用标签。
- 设备/边界：`device=cpu`、`cuda_initialized=false`、`outer_test_predictions_or_metrics_accessed=false`。

## 汇总接口与门槛冒烟测试

除真实预检外，使用合成历史预测占位完成了完整 `144-fit` 汇总冒烟测试。它验证 IDEA-084 执行骨架可正确消费 IDEA-085 的以下输出：

- Macro-F1、plain accuracy、balanced accuracy；
- kitten/adult/senior 三类召回；
- cross-entropy、Brier、paired error corrections；
- 三个主要比较各自独立的五项分类门槛条件及兼容 `gate_passed` 字段。

分类门槛只使用预注册的 Macro-F1 稳定性条件：均值至少 `+0.005`、至少 `2/3` 基础种子均值为正、至少 `6/9` seed-repeat 为正、至少 `8/12` split cell 非负、最差 split 不低于 `-0.03`。CE、Brier、BA、召回和纠错只作辅助质量/安全画像，不进入分类门槛，也不存在单一全局 pass。

测试结果：IDEA-085 专项与轨迹测试 `14/14 PASS`；连同 IDEA-071、IDEA-084 回归测试共 `32/32 PASS`。

## 锁定产物哈希

| 产物 | SHA-256 |
|---|---|
| `plan/IDEA-085_acoustic_temporal_residual.md` | `a327ff78ee790f7f96b39bf8712032cff793ca883affda729ef578fecc05e2fc` |
| `configs/protocol/meowagenet_idea085_acoustic_temporal_residual_v1.json` | `592b8fccf721aeeab8f2bf20839534b703c9e5caa582c93dc220854d77f66bca` |
| `scripts/run_meowagenet_idea085_acoustic_temporal_residual.py` | `471bb5bd0e007be696decd86dec9317b3ef8c3f732490718a4eed99f28e821b4` |
| `tests/test_idea085_acoustic_temporal_residual.py` | `e14cd7e25ffec97a8034e8a2967d9ac21e7d1b3223dd1ed2213a4f961610633d` |
| `runs/.../features/acoustic_trajectories.npz` | `a69b246eba8d773f12a605f2caa9086e2ba0a2d87a3eeeb6ec57175c551a46a7` |
| `runs/.../cpu_preflight.json` | `c76fcb1177ca4a14b432993efaf8b8ab2b4df99c148c3ca80405ad3456b67387` |

## 执行门禁

当前只可把结论记录为“CPU 预检 GO”。协议中的 `gpu_authorized` 保持 `false`；runner 还要求显式 `--director-authorized`，且正式运行必须匹配上述协议与 runner 哈希的 CPU preflight。下一步应由独立审查者复核实现与预检，然后由研究总监决定是否只授权首个 GPU cell。未经新的明确授权，不启动任何 GPU 训练。
