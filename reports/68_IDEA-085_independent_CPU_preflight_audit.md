# IDEA-085 独立 CPU 预检审计

日期：2026-09-19  
审计角色：牛马1（独立只读复核）  
结论：**GO，仅限锁定的首个四管线 cell；GPU 仍未授权。**

## 锁定对象与数据依赖

- 当前 protocol SHA-256：`592b8fccf721aeeab8f2bf20839534b703c9e5caa582c93dc220854d77f66bca`。
- 当前 runner SHA-256：`471bb5bd0e007be696decd86dec9317b3ef8c3f732490718a4eed99f28e821b4`。
- 当前 tests SHA-256：`e14cd7e25ffec97a8034e8a2967d9ac21e7d1b3223dd1ed2213a4f961610633d`。
- CPU preflight SHA-256：`c76fcb1177ca4a14b432993efaf8b8ab2b4df99c148c3ca80405ad3456b67387`；文件内记录的 protocol/runner 哈希与当前文件一致，状态为 `GO`、`cuda_initialized=false`、`outer_test_predictions_or_metrics_accessed=false`。
- 独立从磁盘重算四项数据依赖：roles `87deda39808297e1af5b71283e1d7487a7b88d9288cb492c488e3e64fb91c433`，frozen AST `1c763169e9a9306cc46898808c571b0de7e41db0272e963c30fb88d9b4142399`，C1 summary `ba951db756436ea00adb5c7241ea0264de9fa0f6674a84f274812434d93e26ba`，trajectory cache `a69b246eba8d773f12a605f2caa9086e2ba0a2d87a3eeeb6ec57175c551a46a7`；全部与协议相同。当前 runner 也会在正式执行入口直接校验这些数据哈希以及自身/tests 哈希。

## 核心复核

- 预注册四管线为 A0、同期 C1、T1 和固定联合帧置乱控制 J1；每条管线 36 fits，总计 144 fits。三个新 base seeds `[2713,5395,5226]` 生成的 36 个 full seeds 内部唯一。
- T1/J1 使用完全相同的参数形状、初始权重和训练流程，总参数均为 `108,355`；时序分支 `9,280` 参数，只比 C1 总模型多 `212`。四管线在零残差初始化时 logits 一致，共享 AST 状态一致。
- 数字 call lookup 列只用于从 ragged cache 取回轨迹，不进入编码器；可选 `time_seconds` 也不载入模型。C1 所用前 20 个声学 summary 列与来源逐字节一致。
- 六个原始轨迹通道与六个 finite indicator 组成 12 通道输入。原始通道的 finite median、mean、std 只在当前 cell 的 training calls 全部原生帧上拟合；validation/test 不参与，不做逐调用居中，indicator 不标准化。T1/J1 的训练统计一致。
- 两层卷积后均对 padding 清零，池化只使用有效帧。真实 cache 上 9 帧与 443 帧调用的 padding-invariance 最大差分别为 `3.73e-08` 和 `8.94e-08`；均为浮点误差量级。
- J1 对每个完整 call_id 由锁定 material 派生一个 PCG64 种子，并对六通道值及其缺失性联合置乱；长度与联合 tuple 多重集保持，padding 和标签不参与。审计样本 92 帧置乱确定、非恒等。
- T1/J1 的残差投影为零初始化；预检既确认起点输出为零，也用非零投影探针确认卷积参数梯度可达。共享残差预算的饱和探针最大相对扰动为 `0.25000006`，符合 `0.25` 上限的浮点容差。
- 每个 cell 内四管线共享 full seed、动物/调用覆盖和 batch 顺序。checkpoint 按未加权 inner-validation animal CE 选取；主要分类门槛仅对 T1−C1、T1−A0、T1−J1 的 cat-level Macro-F1 分别判断，不允许单一全局 pass。BA、各类 recall、CE、Brier 和纠错转移均单独报告而不进入主门槛。
- 最长序列 443 个 10 ms 帧对应 `4.43 s` 覆盖长度，最后一帧时间点为 `4.42 s`；二者语义不同但一致，不构成阻断项。

## 独立测试

在项目锁定复现环境中，对当前加固版 IDEA-085 模型测试与轨迹测试联合执行：`14 passed in 6.91s`。本审计未修改 protocol、runner、tests 或预检产物，未初始化 CUDA/GPU，也未读取 outer-test 预测或指标。

## 放行边界

本结论支持研究总监在显式授权后只运行首个完整四管线 cell，并在该 cell 的 manifest、fit、预测及 partial summary 产物通过审查后，再决定是否续跑。它不授权 GPU，不允许修改模型、训练、种子、联合置乱规则或门槛，也不预判 T1 的实证表现。
