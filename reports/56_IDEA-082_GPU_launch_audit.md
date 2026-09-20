# IDEA-082 正式 GPU 启动审计

日期：2026-09-18  
状态：研究总监已明确放行；首个完整 cell 强制暂停审计

## 授权与冻结范围

- 研究总监线程：`01a0a8e4-831f-7872-9f7a-c04ca7ea0f02`
- base seeds：8694、5378、5945
- 9 pipelines × 3 base seeds × 3 repeats × 4 folds = 324 fits
- `outer_test=false`；完整 C1 不重跑；A0 每 cell 只运行一次并共享
- 启动必须同时经过 authorization artifact、冻结哈希、`--resume` 与 runner 的 `--director-authorized`
- 首个 cell 固定为 base seed 8694 / repeat 0 / fold 0；九条管线全部完成后、下一 cell 之前强制暂停

## 启动前 GPU 状态

- GPU：NVIDIA GeForce RTX 4060 Ti
- UUID：`GPU-01226efb-464f-773b-2dfe-807ed093cd0f`
- driver：595.79
- 显存：8188 MiB total；启动检查时 6979 MiB free
- Python：`D:\animalFYP\animal-fyp\environment\.conda\meowagenet-repro\python.exe`
- PyTorch：2.2.2+cu121；CUDA runtime 12.1
- `torch.cuda.is_available()`：true；device count：1
- 启动前未发现 IDEA-081/IDEA-082 训练进程；GPU utilization 2%

## 锁定哈希

- protocol：`0b5e4a4c0be591925811a42769555cab7e5ac9ba8cd2781ac91d1d341e782022`
- runner：`2ce03190d3079cbfe34fbdd140288b2309dfa12ace54f8d21279e66fa7f2c415`
- tests：`b91e4f02a28a98aa6aa7223c4dbc0f5a7a37b9f93f81c3e60f3f697e94fdb00a`
- CPU preflight：`b0967358ed610ce41441cf3bb7bde2cbaa950d38727bb2359eeef7724d783952`
- authorization launcher：`871707b4a6e640a158e7d389b3c6b386cce181fdecd8d835f2c2f629a945bc03`
- first-cell auditor：`07c4eee829e00080d1d65d7fb0bb2e5b351dd7faf84d0e8ec01155dcf9db602d`

## 首 cell 强制审计条件

九条管线须同时通过：fit 身份与预测文件哈希；共享初始 logits/AST state；四对 real/shuffled 完整初始 state；train/validation role-local derangement hash 与 CPU preflight 一致；概率、历史与指标全为有限值且归一化；animal/call population 完全一致；checkpoint reload；参数量；成对 batch coverage；所有 outer-test 标志为 false。任一失败立即停止，不进入下一 cell。

