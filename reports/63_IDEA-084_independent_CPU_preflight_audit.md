# IDEA-084 独立 CPU 预检审计

日期：2026-09-19  
审计角色：牛马1（独立只读复核）  
结论：**GO，仅限锁定的首个四管线 cell；GPU 仍未授权。**

## 核心复核

- 锁定哈希与总监给定值一致：protocol `0c4b50f2a6903bb5e55157591e5a93400c56602161fa2b1f8c8fd0d112bb7eef`，runner `fa3913fd7b2a7a66f3a4e61deb150ef7358cc32ee5b65eafdfe9d2c7e2abe12f`，tests `d37865e4ded83079e8bd4daa2ce5789b8dd47657d40d3f2df5b20428b6a602a3`，CPU preflight `f7e0bfbfcd1ec0670e98664973b45deac4e6df8f9ab9fcf13d2a41e3630f09ca`，report 62 `1e5568583fcebed82c35f217a3e11d3d2269ca6ced38ac0142dcbfe0a9599ddf`，plan `e28cb551803186e922093f05621338b31193f583c54103ab0ff549fa5784a74d`。
- P1 的 `15/5` 分组与 R1 的固定 SHA-256 随机分组均排他、完备且可由锁定 material 重建；R1 不是从多个随机划分中择优。
- 双分支声学参数为 `9,152`，原样 C1 声学支路为 `9,068`；P1/R1 总参数均为 `108,227`，C1 为 `108,143`。
- 实现采用一个共享总预算：`0.25×RMS(h)×(0.5 tanh(u_A)+0.5 tanh(u_B))`，不是两个分支各自拥有 `0.25` 上限。CPU 饱和探针最大相对扰动 `0.25000006`，在锁定浮点容差内。
- P1/R1 可训练参数形状与初始数值相同；四管线零残差初始 logits 一致，共享 AST 状态一致，残差末层为零初始化且两个分支梯度可达。
- 新 base seeds `[3583,5080,9355]` 对应 36 个 full seeds，内部唯一且不与锁定的既往 seed bank 冲突。
- 12 个 repeat×fold role cell 的动物角色互斥；训练/验证调用无交叉。P1/R1 的 median/mean/std 均由当前 train calls 独立拟合，validation/test 不参与预处理统计。
- C1 是同期运行的原始 `BoundedWideAdditiveClassifier`，不是历史结果替代；四管线在每个 cell 使用同一 full seed，并核对训练 cat 顺序与 call coverage。
- P1−C1、P1−A0、P1−R1 分别报告 Macro-F1、稳定性、BA、CE、Brier、senior recall 与方向性纠错剖面；C1−A0 仅为描述背景。实现明确禁止用单一全局 pass 覆盖三条主比较。
- outer test 不加载、不预测、不计分。正式运行同时要求明确 `--director-authorized` 和与当前 protocol/runner 哈希匹配的 CPU GO。
- `--max-cells 1` 只在一个完整四管线 cell 结束后写 partial summary 并返回，`aggregation_generated=false`；续跑会校验 manifest、fit identity 与预测文件哈希，只有完整 144 fits 才进入聚合。

## 独立测试

在项目锁定复现环境中执行 IDEA-084 专项测试：`13 passed in 4.34s`。本审计未修改 IDEA-084 文件，未启动 CUDA/GPU，也未访问 outer-test 预测或指标。

## 放行边界

本结论只支持由研究总监明确授权后运行首个四管线 cell，并在 cell 产物通过后再决定是否 `--resume` 完成预注册的 144 fits。它不授权 GPU、不改变固定分组/权重/宽度/种子/门槛，也不预判 P1 的实证结果。
