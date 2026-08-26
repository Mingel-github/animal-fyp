# MeowAgeNet 数据来源与锁定规则

核验日期：2026-08-25

## 官方来源

- 仓库：<https://github.com/aster-droide/feline-age-prediction>
- 固定提交：`3d02295bef1500d2b2500a124596f77010181391`
- 原始裁剪音频目录：`dataset/raw_audio/AudioCropped`
- 目录 Git tree SHA-1：`6a10ede379615a52441b0b80e9f4783f7b182ebf`
- 上游许可：CC BY 4.0；使用时需引用 van Toor、Qazi 与 Paladini（2025）

原始音频不提交到本项目 Git。运行以下命令可从固定提交下载或重新核验：

```powershell
node .\scripts\build_meowagenet_manifest.mjs
node .\scripts\build_meowagenet_manifest.mjs --verify-only
```

本地音频保存在
`data/meowagenet/official-3d02295bef15/AudioCropped`，由 `.gitignore` 排除。

## 已版本化的审计文件

| 文件 | 用途 |
| --- | --- |
| `data_manifest.csv` | 793 个官方文件的路径、Git blob、SHA-256、WAV 参数、时长、年龄、年龄组、发布 ID 与清洗标记 |
| `checksums.sha256` | 793 个本地音频的逐文件 SHA-256 |
| `cat_id_manifest.csv` | 发布 `cat_id`、分析 ID、类别、原始文件数及 VGGish embedding 行数的对账表 |
| `dataset_summary.json` | 来源提交、总数、聚合校验值与异常摘要 |

当前 `data_manifest.csv` 的 SHA-256 为
`68e5131dc5d3cd611ecdda30e5176a6dcc90c0ea2500a6d8a4d9b066ce11a72f`；
`checksums.sha256` 文件自身的 SHA-256 为
`7c1b51ce1a18b1253d3099e9ce3ea034385cd1851ed121a1a5b82a50a20e6a12`。

## 数量对账

| 层级 | 行数/文件数 | ID 数 | 含义 |
| --- | ---: | ---: | --- |
| 官方 `AudioCropped` | 793 | 112 个发布 ID | 每行是一段人工裁剪的原始叫声文件 |
| 唯一音频内容 | 792 | 111 个清洗后分析 ID | 排除一个跨 ID 的字节级重复副本 |
| 官方 VGGish CSV | 937 | 112 个发布 ID | 每行是一个 0.96 秒 VGGish 窗口 embedding，不是一段新的叫声 |
| 清洗后的 VGGish 视图 | 936 | 111 个分析 ID | 从官方 CSV 排除重复别名 `049A` 的一行，用于后续公平比较 |

793 段原始叫声的年龄组计数为 kitten 135、adult 405、senior 253；平均
时长 0.7247 秒，范围 0.0844～4.4253 秒，与论文报告一致。937 比 793
多出的 144 行来自长叫声产生的额外重叠 VGGish 窗口；短叫声通过循环补齐
至少产生一个窗口。因此 937 不能作为“独立叫声数量”。

## 111 与 112 的解释

官方文件名和 VGGish CSV 实际都能恢复出 112 个发布 ID。审计发现：

- `0Y-041A-01.wav` 与 `0Y-049A.wav` 的 Git blob、SHA-256、WAV 内容和
  VGGish embedding 完全相同，只有文件名中的 `cat_id` 不同；
- 保留官方 793 行不改动，但在清洗分析视图中保留 `041A`、排除重复别名
  `049A`，恰好得到 792 个唯一音频和 111 个分析 ID；
- 这能数值上解释论文的 111，但论文和上游仓库没有明确说明该别名关系，
  因而应在论文方法中写成“本地数据审计与清洗规则”，不要声称是作者明示规则。

另外，文件名使用 `020AB`，而 embedding CSV 使用 `020A`；manifest 将前者
规范映射到后者。`000B` 同一只猫包含 2 岁和 5 岁录音，两者都属于 adult，
不会影响三分类标签，但连续年龄任务必须保留这个纵向事实。

## 本项目使用规则

1. 原论文数值复现继续使用官方 937 行 VGGish CSV 和 112 个发布 ID，保证
   与上游 notebook 输入一致；该结果只作为“原论文复现参考”。
2. 后续标准 AST、IDEA-012 与 IDEA-048 的锁定比较使用
   `data_manifest.csv` 中 `analysis_include=true` 的 792 段唯一音频、111 个
   分析 ID。
3. 同一锁定比较中的 VGGish 基线使用排除 `049A` 后的 936 行视图，确保
   与新模型采用同一批分析动物；这不是另立第二个 baseline，而是把复现参考
   转换到统一数据协议。
4. 937 行 embedding 与 793 段音频不可拼接、相加或当作两套独立数据；前者
   是后者经过 VGGish 窗口化后的派生表示。
