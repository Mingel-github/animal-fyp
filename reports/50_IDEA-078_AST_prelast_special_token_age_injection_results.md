# IDEA-078：AST 倒数第二层 special-token 年龄注入结果

## 结论

IDEA-078 已按结果产生前锁定的协议完成 `108/108` 个 fits：3 条管线 × 3 个新基础种子 × 3 repeats × 4 folds。正式运行、`--resume` 规范字节复核和独立 CPU-only 全工件审计均通过，且 `outer_test_accessed=false`。

最终结论为：

1. **T1 相对 A0 的主 gate：FAIL。** T1−A0 的九个 `base_seed × repeat` 等权 Macro-F1 均值为 `-0.0201088`；只有 `1/3` 个基础种子均值和 `5/9` 个 seed×repeat 为正，12 个共同 split-cell 中只有 `6/12` 非负，最差 split-cell 为 `-0.1464354`。T1 的 CE 和 Brier 也都劣于 A0。唯一通过的是三个基础种子的 senior-recall 差值均不低于 `-0.02`。
2. **T1 相对等参数 P1 的机制 gate：不可解释／FAIL。** T1−P1 的平均 Macro-F1 为 `-0.0103537`，只有 `1/3` 个基础种子和 `3/9` 个 seed×repeat 为正；`7/12` 个 split-cell 非负，但最差单元为 `-0.1039567`。T1 的 CE 略优于 P1，Brier 却更差。主 gate 未通过，因此机制 gate 按预注册规则不可解释；其 raw 条件本身也为 false。
3. **不保留 T1，也不做结果驱动修补。** 本结果不授权事后修改 bottleneck 宽度、注入层、token 范围、种子或 gate。当前 frozen-AST 表示参考仍为 A0；IDEA-076 的冻结结论不变。共享倒数第二层 token cache 已通过独立审计，可继续由 IDEA-079 只读使用。

这是一项可信的负结果：所有配对初始化、参数量、冻结尾部、注入 token 范围、checkpoint 重载、validation-only 边界、预测文件哈希和缓存哈希均通过独立检查，失败不能归因于已知实现偏离或 outer-test 泄漏。

## 实验边界与设计

- 数据：MeowAgeNet，792 calls、111 cats、843 个 1.28 秒 segments。
- 角色：formal-v2 nested roles；只访问 train/validation，outer test 始终关闭。
- 基础种子：`9763 / 3230 / 9726`；相对 IDEA-068 至 IDEA-077 的基础种子及完整种子无碰撞。
- 主要单位：九个等权 `base_seed × repeat` validation 估计，每个估计合并四个 folds。
- split 稳定性单位：12 个 `repeat × fold` 单元，每个单元对三个基础种子等权平均。
- 重复出现的猫只作描述，不能视为独立样本量扩大或外部确认。

三条锁定管线为：

| 管线 | 结构 | 可训练参数 | Fits |
|---|---|---:|---:|
| `A0_token_reconstruction` | 原样 block-11 tokens → 冻结 block 12 → 最终 LayerNorm → 两个 special tokens 均值 → call 内 segment 均值 → 公共分类头 | 99,075 | 36 |
| `P1_postpool_age_injection` | 在冻结 AST 最终 pooled representation 后加入 `20→10→768` 年龄残差，再进入公共分类头 | 107,733 | 36 |
| `T1_prelast_special_token_injection` | 在冻结 block 12 前，仅向 token 0/1 加入同一个 `20→10→768` 年龄残差；144 个 patch tokens 不直接改动 | 107,733 | 36 |

P1 与 T1 的年龄分支完全等参数，共 8,658 个参数。两者的输出投影权重和偏置均从零开始，因此优化前 A0/P1/T1 logits 与 loss 完全相同；P1 是区分“增加年龄分支容量”与“经最后一个冻结 transformer block 交互”的必要对照。

## 主要结果：九个 seed×repeat 等权均值

| 管线 | Macro-F1 | Balanced accuracy | Animal CE ↓ | Animal Brier ↓ |
|---|---:|---:|---:|---:|
| A0 | **0.7408654** | 0.7712963 | **0.6669666** | **0.4029924** |
| P1 | 0.7311103 | **0.7731481** | 0.7203434 | 0.4208184 |
| T1 | 0.7207567 | 0.7629630 | 0.7187377 | 0.4230000 |

两种年龄分支都没有超过 A0。P1−A0 的 Macro-F1 已为 `-0.0097551`，说明年龄分支容量／年龄线索整合本身不稳定；T1 在此基础上相对 P1 又下降 `-0.0103537`。

### 配对效应

| 比较 | Macro-F1 Δ 均值 | SD | 中位数 | seed×repeat 正／平／负 | 最差／最好 |
|---|---:|---:|---:|---:|---:|
| T1−A0 | -0.0201088 | 0.0584930 | +0.0017013 | 5／0／4 | -0.1589322／+0.0367174 |
| P1−A0 | -0.0097551 | 0.0402422 | -0.0102944 | 3／0／6 | -0.0945148／+0.0405021 |
| T1−P1 | -0.0103537 | 0.0392150 | -0.0037847 | 3／0／6 | -0.0709838／+0.0550849 |

## 九个 seed×repeat 单元

| Seed | Repeat | A0 F1 | P1 F1 | T1 F1 | T1−A0 | T1−P1 |
|---:|---:|---:|---:|---:|---:|---:|
| 9763 | 0 | 0.7698645 | 0.7997494 | 0.7802739 | +0.0104094 | -0.0194755 |
| 9763 | 1 | 0.6698284 | 0.6669009 | 0.6715297 | +0.0017013 | +0.0046288 |
| 9763 | 2 | 0.7942857 | 0.6997710 | 0.6353535 | -0.1589322 | -0.0644174 |
| 3230 | 0 | 0.7877697 | 0.7771192 | 0.7629731 | -0.0247966 | -0.0141461 |
| 3230 | 1 | 0.6701754 | 0.7106775 | 0.7068928 | +0.0367174 | -0.0037847 |
| 3230 | 2 | 0.7423629 | 0.7598852 | 0.6889014 | -0.0534615 | -0.0709838 |
| 9726 | 0 | 0.7737883 | 0.7634939 | 0.7842034 | +0.0104151 | +0.0207095 |
| 9726 | 1 | 0.7151173 | 0.6773016 | 0.7323865 | +0.0172692 | +0.0550849 |
| 9726 | 2 | 0.7445967 | 0.7250945 | 0.7242958 | -0.0203008 | -0.0007986 |

## 三个基础种子

| 基础种子 | T1−A0 F1 | P1−A0 F1 | T1−P1 F1 | T1−A0 senior recall |
|---:|---:|---:|---:|---:|
| 9763 | -0.0489405 | -0.0225191 | -0.0264214 | +0.0833333 |
| 3230 | -0.0138469 | +0.0157913 | -0.0296382 | 0.0000000 |
| 9726 | +0.0024611 | -0.0225374 | +0.0249986 | -0.0166667 |

T1−A0 与 T1−P1 都只有 seed 9726 为正。三个基础种子的 T1−A0 senior recall 均高于锁定的 `-0.02` 下限；这项安全条件通过，但不足以覆盖主要性能和稳定性失败。

## 12 个共同 split-cell

每个单元先对三个基础种子的 fold-level Macro-F1 差值取均值。

| Repeat | Fold | T1−A0 | T1−P1 |
|---:|---:|---:|---:|
| 0 | 0 | +0.0532341 | +0.0175439 |
| 0 | 1 | -0.0366555 | -0.0366555 |
| 0 | 2 | +0.0280423 | 0.0000000 |
| 0 | 3 | -0.0412055 | +0.0067857 |
| 1 | 0 | -0.1087457 | +0.0530864 |
| 1 | 1 | +0.1058526 | +0.0177574 |
| 1 | 2 | +0.0153169 | -0.0092683 |
| 1 | 3 | +0.0685675 | +0.0147064 |
| 2 | 0 | +0.0317621 | +0.0545986 |
| 2 | 1 | -0.1160270 | -0.1039567 |
| 2 | 2 | -0.1464354 | -0.0866504 |
| 2 | 3 | -0.1147688 | -0.0784252 |

T1−A0 有 `6/12` 个非负单元，最差为 `-0.1464354`；T1−P1 有 `7/12` 个非负单元，最差为 `-0.1039567`。两项都低于预注册的至少 `8/12` 非负、最差不低于 `-0.03` 的要求。

## 预注册 gate 判定

### 主 gate：T1 相对 A0——FAIL

| 条件 | 观察结果 | 判定 |
|---|---|---|
| 平均 T1−A0 Macro-F1 ≥ +0.005 | `-0.0201088` | 未通过 |
| 3/3 基础种子均值严格 > 0 | `1/3` | 未通过 |
| 至少 6/9 seed×repeat 严格 > 0 | `5/9` | 未通过 |
| 至少 8/12 split-cell ≥ 0 | `6/12` | 未通过 |
| 最差 split-cell ≥ -0.03 | `-0.1464354` | 未通过 |
| T1 平均 CE 不劣于 A0 | `0.7187377 > 0.6669666` | 未通过 |
| T1 平均 Brier 不劣于 A0 | `0.4230000 > 0.4029924` | 未通过 |
| 每个基础种子的 senior-recall 差值 ≥ -0.02 | 最差为 `-0.0166667` | 通过 |

只有 1/8 项通过，因此 `main_gate_passed=false`。

### 机制 gate：T1 相对 P1——不可解释／FAIL

| 条件 | 观察结果 | 条件本身 |
|---|---|---|
| 平均 T1−P1 Macro-F1 严格 > 0 | `-0.0103537` | 未通过 |
| 至少 2/3 基础种子均值 > 0 | `1/3` | 未通过 |
| 至少 5/9 seed×repeat > 0 | `3/9` | 未通过 |
| 至少 8/12 split-cell ≥ 0 | `7/12` | 未通过 |
| 最差 split-cell ≥ -0.03 | `-0.1039567` | 未通过 |
| T1 平均 CE 不劣于 P1 | `0.7187377 < 0.7203434` | 通过 |
| T1 平均 Brier 不劣于 P1 | `0.4230000 > 0.4208184` | 未通过 |

机制 raw 条件只有 1/7 通过，`mechanism_gate_raw_passed=false`。主 gate 又未通过，所以 `mechanism_gate_interpretable=false`、`mechanism_gate_passed=false`。

## 为什么 T1 比等参数 P1 更差

这是对锁定对照结果的机制性解释，不是事后调参建议，也不是已被单独验证的因果结论。

1. **年龄分支本身已经不稳定。** P1 直接在最终 pooled representation 后注入年龄信息，仍比 A0 低 `0.0097551`。因此 T1 的失败不能只归因于“注入位置不够好”；当前年龄线索／分支容量在这些新种子上没有形成稳定收益。
2. **冻结 block 12 无法适应被改变的 special-token 分布。** T1 把同一个稠密 768 维残差加到 CLS 与 distillation token，再送入按无条件 token 几何预训练且完全冻结的 block 12。虽然 144 个 patch tokens 没有被直接改写，special tokens 改变后的 LayerNorm、自注意力 query/key/value 和 FFN 轨迹仍会与原来的 patch-token 分布交互；冻结尾部无法针对这个条件分布重新适配。
3. **T1 的年龄信号必须穿过固定非线性尾部。** 固定的注意力、残差和 FFN 可以旋转、压缩或抵消学习到的年龄方向。P1 则保留锁定的 AST pooled representation，只在其后增加一条直接加法路径，发生表示干扰的机会更少。
4. **证据更符合不稳定干扰，而非绝对不可行。** T1−P1 平均为负、仅 `3/9` 个 seed×repeat 为正，且两个基础种子为负；但 seed 9726 的 T1 同时略优于 A0 和 P1，T1 的 CE 也略优于 P1。这说明效果不是处处同向的灾难，而是缺乏可复现的性能收益并带有较大 split 尾部风险。

因此，当前证据支持的最窄结论是：**在预注册的零影响 `20→10→768`、只注入两个 special tokens、冻结 block 12 的设计下，pre-last-block 路由没有把同等容量的年龄分支转化为稳健收益，反而进一步降低了 Macro-F1。** 不能由此声称所有内部年龄注入都无效，也不能用结果来反推并立即搜索新的层、宽度或 token 范围。

## 缓存与重建审计

共享 cache 位于 `runs/ast_prelast_tokens_idea078_v1/`，由 IDEA-078 唯一生产，IDEA-079 只读复用。

| 项目 | 结果 |
|---|---|
| token shape／dtype | `[843, 146, 768]`／float32 |
| 存储 | 可 memory-map 的 `.npy`；文件 378,095,744 bytes，原始 float payload 378,095,616 bytes |
| index | `.npz`，`allow_pickle=false` |
| 标签／角色／cat ID | 均不写入 cache |
| GPU 全量 A0 重建误差 | mean `4.02243e-7`；max `8.46386e-6` |
| CPU 独立重建误差 | mean `5.41925e-7`；max `9.89437e-6` |
| 锁定容差 | mean ≤ `2e-6`；max ≤ `2e-5` |

cache 提取完成后，Windows 上打开的 memory map 阻止了临时文件原子改名。进程退出后，完整临时文件先按 shape、token/index/source mapping 哈希和全量 GPU 重建结果独立核验，再只执行改名；没有重新计算或覆盖内容。`cache_manifest.json` 明确记录了这一恢复过程。

## 实现前置修正

正式结果产生前发现并修正了两处纯实现问题，研究合同、数据、种子、模型、训练和 gate 均未改变：

1. 初始化／梯度审计的 probe tensors 留在 CPU，而冻结 tail 位于 CUDA；将 probes 和对应审计模型移动到 tail device。修正前完成 fits 为 0。
2. 第一轮正式启动在聚合前停止，因为 call prediction 输出键为 `label`，而冻结的聚合器要求 `true_label`；只改正该输出键。修正前完成 fits 为 0。

两项修正均在任何可用结果形成前完成，并经研究总监确认后才进入正式运行。它们不构成结果驱动协议变更。

## 正式运行与独立审计

| 审计项 | 结果 |
|---|---|
| 完成 fits | 108/108；三管线各 36 |
| fit summaries | 108/108 通过 |
| validation 预测文件 | 216/216 哈希一致 |
| 检查的 call 预测行 | 12,015 |
| validation cells | 36/36 角色、calls、cats 与标签一致 |
| `outer_test_accessed` | manifest、summary 和全部 fits 均为 false |
| 初始化审计 | 108/108；共享 head、P1/T1 年龄分支状态和零影响 logits/loss 均符合 |
| trainable parameters | A0 99,075；P1/T1 107,733；全部符合 |
| frozen tail parameters | 7,089,408；全部符合 |
| 注入 token | 仅 T1 为 `[0,1]`；patch tokens 未直接修改 |
| checkpoint reload | 108/108；最大概率差为 0 |
| 共同 epoch 批次覆盖 | 546 个配对审计全部一致 |
| canonical aggregate rebuild | 与锁定 summary 语义和规范字节完全一致 |
| 独立工件清单 | 327 entries；SHA-256 `e8230f25febcc6b5ec06948661d332e7feffe726e6acc38ac03f857e47d275d8` |

正式运行结束后执行相同 runner 的 `--resume`，没有重新训练；聚合 summary 和 compact run summary 的语义与规范字节保持一致。独立结果 verifier 在 `CUDA_VISIBLE_DEVICES=''` 下运行，未重新占用 GPU。

## 可复现性与哈希

| 工件 | SHA-256 |
|---|---|
| 锁定计划 | `e53e2a77d66ffcdbf90a4bd4322bc7e35954f5d14e8b2213326844c7c265998b` |
| 锁定协议 | `54080718a798655e3ebd3bd7a9bd0c346a554154f1ec8001ed15dbfe6122e7f8` |
| cache schema | `5183b9d5e7e689a0337e0e6b0cfb2d5cf6dc7c6402af39ccafdbc11964dc0f3b` |
| cache extractor | `ad9b596cd39f7502c8d4cfe7a63192cb71dbde10e3151cd808186f710ad8b698` |
| 训练 runner | `36e75b63e37a485dec05b1b2def993f137aa3dafb4857c222a0f41c80a85f399` |
| 测试 | `e77575fcb425d519627ef5432f19790b608bfe0fad247a797932188d52dd11cd` |
| cache verifier | `f8e757fa938b0602fdc45586e212376d4a2c54c8093445db2e095b15f705e0a7` |
| 结果 verifier | `359fb776bf3cbf47b6f10981e1421e5cc5bd5a4fb933b0f15a69f4e767517360` |
| cache manifest | `af14eb7b8a2f9c362f37484d06af4381fe89f1a497a89c62f1142ab49baf8e94` |
| cache token | `9fb7b32973560d31ad1906c07349e745054683d5a6a3486e3e33182fc6947ca9` |
| cache index | `aa46820aa15a0812ef4437c80c29f28fd515f0228a9d0fd10bac266b734b9bb2` |
| run manifest | `b6dd5095beb142f537d69a5e4475058f1224214e94c49f3752e38a384fe7c3c5` |
| 初始评估汇总 | `bcbe31dc438fcb6e8b88dc6067fd4cf03bfc19dfb52479cd876108be29cdd9c0` |
| compact run summary | `da9b38c86016a81172aeb81f24354d06888dbc7873945f3886ad053ac7a1a007` |
| 独立审计 | `cdecafc4225415a23b2815ce7713602e7d409c6d63731e66a2f63cf0f93b7b27` |
| roles | `87deda39808297e1af5b71283e1d7487a7b88d9288cb492c488e3e64fb91c433` |
| 冻结 AST embedding | `1c763169e9a9306cc46898808c571b0de7e41db0272e963c30fb88d9b4142399` |
| AST fbank | `007d07f7c236ba44ae76a1e867cee5cc49b774ddde2f54064c68190cd01d60c7` |
| 20 维年龄声学特征 | `ba951db756436ea00adb5c7241ea0264de9fa0f6674a84f274812434d93e26ba` |

测试结果为 `9 passed`。正式训练环境为 Python 3.10.12、PyTorch 2.2.2+cu121、CUDA 12.1、NVIDIA GeForce RTX 4060 Ti；Git revision 为 `0e9c31ee5150795d7bf4c35ce2794c05a2123036`。

## 最终研究状态

- `main_gate_passed = false`
- `mechanism_gate_raw_passed = false`
- `mechanism_gate_interpretable = false`
- `mechanism_gate_passed = false`
- `overall_gate_passed = false`
- `retain_T1_as_candidate = false`
- `post_outcome_tuning_authorized = false`

按预注册 fail action，停止当前 pre-last special-token 年龄注入公式，不追加结果驱动种子或结构修补。A0 继续作为 frozen-AST 表示参考；共享、无标签的 block-11 token cache 保持冻结并供 IDEA-079 只读复用。
