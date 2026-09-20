# IDEA-082 CPU preflight

日期：2026-09-18  
结论：**GO（仅工程预检）**；正式 GPU 仍未获授权

## 冻结设计

- 9 条管线：1 个 A0，加 G1/G2/G3/G12 各自的 real 与等参数 shuffled 控制。
- G1/G2/G3 为互斥 5/10/5 分组并穷尽 20 维；G12 是预先声明的 G1+G2 15 维重叠组合。
- 3 个新 base seed（8694、5378、5945）× 3 repeats × 4 folds；每 cell 只拟合一次共享 A0，总计 324 fits。
- 完整 C1 不重跑；IDEA-076 只作历史参考。
- shuffled 特征在 train 与 validation role 内分别确定性置换，test 行不物化；同一 cell 的四组共享同一行映射。

## CPU-only 检查结果

| 检查 | 结果 |
|---|---|
| 数据身份 | 792 calls / 111 cats |
| role cells | 12/12 动物不跨 train/validation/test，train/validation calls 不交 |
| seed | 36 个 full seeds 唯一，且与既有 bank 无碰撞 |
| 分组 | 5/10/5 互斥穷尽；G12 精确等于 G1∪G2 |
| 置换 | 12×2 role mappings 均固定点为 0；不跨 role；各组映射 hash 一致 |
| 多重集/统计 | 12/12 cells、4/4 groups 的 real/shuffled 行多重集与 train-only median/mean/std 完全一致 |
| 参数 | A0=99,075；G1/G3=108,181；G2=108,099；G12=108,131；每对 real/shuffled 严格相等 |
| 初始化 | 八条分组管线相对 A0 的最大 logit 差均为 0；共同 AST 状态一致；4 对 real/shuffled 完整 state 一致 |
| 梯度 | 所有组在零初始化时 age-output 可达、age-hidden 梯度为 0；非零 output probe 后 age-hidden 均可达 |
| C1 上限 | 四组 saturation probe 的最大相对扰动均为 0.2500000596，处于 0.25+1e-5 容差内 |
| 设备 | `cpu`；preflight 前后 CUDA 均未初始化 |
| outer test | 未生成或读取 outer-test prediction/metric |
| 测试 | 10 passed |

## 固定哈希

- protocol：`0b5e4a4c0be591925811a42769555cab7e5ac9ba8cd2781ac91d1d341e782022`
- runner：`2ce03190d3079cbfe34fbdd140288b2309dfa12ace54f8d21279e66fa7f2c415`
- tests：`b91e4f02a28a98aa6aa7223c4dbc0f5a7a37b9f93f81c3e60f3f697e94fdb00a`
- CPU preflight artifact：`b0967358ed610ce41441cf3bb7bde2cbaa950d38727bb2359eeef7724d783952`
- locked feature cache：`ba951db756436ea00adb5c7241ea0264de9fa0f6674a84f274812434d93e26ba`
- independent grouping review：`0fcd9f735750b7ac753ca95eaae3d7c10c70dfcb03a647f2e3f28dc56f4a23dd`

## GO 的严格含义

GO 只表示协议、数据身份、配对初始化、容量对照、置换、梯度和有界融合在 CPU 上通过了结果盲审计。它不是统计结论，也不是 GPU 放行。runner 的 formal stage 在没有显式 `--director-authorized` 时 fail closed；当前未开始任何 IDEA-082 epoch。

