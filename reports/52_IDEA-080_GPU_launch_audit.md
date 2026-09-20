# IDEA-080：正式 GPU 启动审计

## 授权与边界

- 日期：2026-09-18
- 研究总监已明确放行正式 GPU 实验。
- 固定范围：MeowAgeNet、`outer_test=false`、4 pipelines、3 base seeds × 3 repeats × 4 folds，共 144 fits。
- base seeds：`[59, 7031, 1855]`；36 个 full seeds 唯一。
- 启用 `--resume`；formal runner 另要求 `--director-authorized`。

## 启动前 GPU 就绪检查

- GPU：NVIDIA GeForce RTX 4060 Ti（UUID `GPU-01226efb-464f-773b-2dfe-807ed093cd0f`）
- driver：595.79
- 显存：8188 MiB total；启动前 7067 MiB free
- Python 环境：`D:\animalFYP\animal-fyp\environment\.conda\meowagenet-repro\python.exe`
- PyTorch：2.2.2+cu121
- CUDA runtime：12.1
- `torch.cuda.is_available()`：true
- device count：1

## 锁定依赖

- protocol SHA-256：`c2aa57e9dbab8dc955ec1e96ce453566807464f6f0e4380d57987e4a40feb925`
- runner SHA-256：`2bb7359a6c88544d262d5836b66f099fd62acfc028c4789c9a439978fca140da`
- tests SHA-256：`52e614e8d52f6d9c8087516b37468bc9967157537dfd06734057341ea8d6c18a`
- CPU preflight SHA-256：`d3b71a557dd5c8e48ee27407db6d51313b600d119c333f6962b884d81c83e641`
- 定向测试：9 passed，0 failed。
- CPU preflight：`GO_FOR_FORMAL_GPU_RUN`；完整 cache A0 重建 MAE `5.41924578101316e-07`、最大误差 `9.894371032714844e-06`。

## 精确启动命令

```powershell
& 'D:\animalFYP\animal-fyp\environment\.conda\meowagenet-repro\python.exe' scripts\run_meowagenet_idea080_ast_last_block_lora_c1_factorial.py --stage run --device cuda --resume --director-authorized
```

启动后首先检查 base seed 59、repeat 0、fold 0 的四条 pipeline：fit 完整性、NaN、可训练参数量、outer-test 标志、共享初始化与成对 batch coverage。任一异常立即停止。
