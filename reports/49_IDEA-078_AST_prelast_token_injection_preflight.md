# IDEA-078：AST 最后一层前 special-token 年龄注入预检

## 结论

当前技术状态为 **`GO_FOR_FORMAL_GPU_RUN`**。唯一共享 cache 已提取并完成 GPU 与 CPU 双重全量重建验收；初始化公平性、梯度可达性、参数量、种子与 108-fit 预算也全部通过，外层 test prediction 未访问。

本轮授权范围仅到 cache 落盘、manifest 哈希锁和全量验收，没有启动 108 个正式 fits。`GO_FOR_FORMAL_GPU_RUN` 表示技术依赖已经闭合，不代表本报告自行扩大执行权限。

cache 完成后只做了机械锁定：正式 protocol 状态改为 `locked_for_formal_evaluation`，回填 manifest/token/index 哈希和两项 GPU 重建误差。cache manifest 继续保留生产它的 pre-cache protocol SHA，以避免 protocol 与 manifest 相互包含哈希造成循环；正式 runner 同时核对 producer protocol、最终 manifest、token 和 index 四个哈希。方法、种子、训练、gate 和统计单位均未改变。

## 唯一共享 cache 契约

IDEA-078 已发布唯一 schema：

- schema：`metadata/experiments/ast_prelast_token_cache_idea078_v1_schema.json`
- schema SHA-256：`5183b9d5e7e689a0337e0e6b0cfb2d5cf6dc7c6402af39ccafdbc11964dc0f3b`
- token：`runs/ast_prelast_tokens_idea078_v1/prelast_tokens.float32.npy`
- index：`runs/ast_prelast_tokens_idea078_v1/prelast_token_index.npz`
- 最终 manifest：`runs/ast_prelast_tokens_idea078_v1/cache_manifest.json`
- manifest SHA-256：`af14eb7b8a2f9c362f37484d06af4381fe89f1a497a89c62f1142ab49baf8e94`
- token SHA-256：`9fb7b32973560d31ad1906c07349e745054683d5a6a3486e3e33182fc6947ca9`
- index SHA-256：`aa46820aa15a0812ef4437c80c29f28fd515f0228a9d0fd10bac266b734b9bb2`
- owner：IDEA-078；只读消费者：IDEA-078、IDEA-079；禁止第二次提取或另建 schema。

token tensor 固定为 `[843,146,768]` float32 `.npy`，可 mmap，原始 payload 为 378,095,616 bytes（360.580078125 MiB）。位置 0、1 是 CLS 和 distillation special tokens；2～145 是按 AST 原生 frequency-major/time-minor 顺序展开的 12×12 patch tokens。边界固定为 one-based block 11 输出／block 12 输入，尚未执行最终 LayerNorm 或 segment pooling。

index 仅保存 `segment_call_indices`、`segment_counts`、`call_ids`、`source_paths` 和边界常量，使用 `allow_pickle=False`。不得保存 labels、cat IDs 或 roles。已确认 843→792 映射闭合、每 call 有 1～6 segments；关键数组哈希如下：

| 项目 | SHA-256 |
| --- | --- |
| segment_call_indices | `a191970d575e9bacda3bc454bca1120bdd5a0054ca5068d59a28a41ee7e21c73` |
| segment_counts | `62880eaf08cf1891103d646caa2eee731633ccd1f936ccba5e6ea17deb0f60a9` |
| call_ids | `d48bf93391588611190f59d9b73bca7d943ed0a5866c1b90ea5028a2ab035f3f` |
| source_paths | `62c52dc181a2f95af0ee03a9cb233bb520b10d934332c746e0755c8171a45cfe` |

最终 manifest 已锁定 protocol、schema、extractor、token、index、fbank、data manifest、audio checksums、roles、锁定 final embedding 和模型权重哈希，并记录窗口、hop、geometry、block 边界、843→792 映射和全量 A0 重建误差。

提取时出现一个已透明记录的 Windows 收口异常：843 个 segment 的提取和严格重建均完成后，只读 mmap 句柄尚未释放，第一次临时文件重命名被操作系统拒绝。进程退出后没有重新提取或覆盖；独立 verifier 对原临时 token/index 完成 shape、hash、index keys、与锁定 fbank 的逐元素映射和全量 GPU 重建复核，全部通过后仅把同一两份文件原子改为正式名。正式文件哈希与临时文件哈希一致，manifest 的 `recovery_note` 记录了该过程。

## CPU A0 端到端重建

预检固定使用 call indices 0、24、114，共 3 calls、8 segments，覆盖 1、2、5 segments/call。CPU 上：

1. fbank 经过原生 embeddings 和前 11 个 encoder blocks 得到 `[8,146,768]` pre-last tokens；
2. 手工执行冻结 block 12、最终 LayerNorm 和两个 special tokens 的均值；
3. 按锁定 segment→call 映射做 call 内算术平均；
4. 与相同 CPU 模型的原生完整前向比较，最大绝对误差为 **0.0**；
5. 与历史锁定 GPU final embedding 比较，均值绝对误差为 **2.1557375e-6**，最大绝对误差为 **1.3113022e-5**。

原先考虑的 CPU↔GPU 均值阈值 2e-6 被观测值轻微越过，而 CPU 原生前向与手工 tail 又严格相等。因此在任何正式评价结果产生之前，CPU 跨设备探针阈值透明地锁为 mean≤3e-6、max≤2e-5；这不是放宽最终 cache 标准。GPU 生成的完整 cache 仍必须在 792 calls 上达到 mean≤2e-6、max≤2e-5，与 IDEA-079 的共享依赖保持一致。

最终 cache 在全量 843 segments→792 calls 上的 GPU A0 重建误差为 mean **4.0224327e-7**、max **8.4638596e-6**；独立 CPU 全量复核为 mean **5.4192458e-7**、max **9.8943710e-6**。两者均通过 mean≤2e-6、max≤2e-5，故共享 cache 依赖正式闭合。

## 模块、参数和必要容量对照

年龄分支固定为 `20→10→768`：`u=GELU(W1a+b1)`，`d=W2u+b2`，其中 `W2` 和 `b2` 零初始化。

- A0：不改 tokens，经同一冻结 tail 重建；99,075 个可训练参数。
- P1：把 `d` 加到最终 pooled 768 维表示；107,733 参数。
- T1：只把同一个 `d` 加到 block 12 前的 token 0、1；144 个 patch tokens 在注入点逐元素不变；107,733 参数。

P1 是必要的等参数容量对照。A1/C1 曾把年龄信息注入可训练分类器的 128 维隐藏层；T1 位于冻结 AST 最后一层之前，允许 special tokens 与未直接修改的 patch tokens 在 self-attention/FFN 中重新交互，因此不是旧 A1/C1 的重复。T1/P1 比旧 C1 少 410 个参数，未用无作用参数伪造匹配。

三条 pipeline 的初始 shared head 完全相同，P1/T1 年龄分支完全相同；初始 logits 相对 A0 的最大差均为 0，初始 CE 均为 1.1518301964。零输出层时 output projection 有梯度（最大值 0.0252193）、hidden 层梯度为 0；固定非零 output probe 后 hidden 层梯度变为非零。冻结 AST tail 没有任何参数获得梯度。

## 种子、预算和主判据

基础种子固定为 9763、3230、9726。3 seeds×3 repeats×4 folds 产生 36 个唯一 full seeds，与 IDEA-068～077 的 base/full seed 无碰撞。三管线总计 108 fits；主要 A0/T1 对比为 72 fits。

主目标是 MeowAgeNet animal-level Macro-F1。T1−A0 gate 同时要求：均值≥+0.005、3/3 base seeds 为正、至少 6/9 seed×repeat 为正、至少 8/12 split cells 非负、最差 split≥−0.03、CE 和 Brier 均不劣、每个 base seed 的 senior-recall 差≥−0.02。T1−P1 机制 gate 只有在主 gate 通过时才可解释。

## 已验证文件

- 正式 protocol SHA-256：`54080718a798655e3ebd3bd7a9bd0c346a554154f1ec8001ed15dbfe6122e7f8`
- cache producer protocol SHA-256：`d2d2397ab9fed1a1b6302bc248ff2a43f56f511c592d9b28097d79b63a364e13`
- runner SHA-256：`36e75b63e37a485dec05b1b2def993f137aa3dafb4857c222a0f41c80a85f399`
- extractor SHA-256：`ad9b596cd39f7502c8d4cfe7a63192cb71dbde10e3151cd808186f710ad8b698`
- tests SHA-256：`e77575fcb425d519627ef5432f19790b608bfe0fad247a797932188d52dd11cd`
- 定向测试：9 passed，0 failed。

本报告确认共享无标签 cache 与 manifest 已完成且技术上可供 IDEA-078/079 只读使用；本轮没有启动 108-fit 正式训练。
