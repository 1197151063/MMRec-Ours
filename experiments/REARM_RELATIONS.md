# REARM：关系分支与用户门控，第一轮实验

## 要验证的假设

语义图的提前融合可能混合不同关系；不同用户可能需要不同关系权重。
先验证这两点，再决定是否实现训练内部遮边的预测效用监督。**本轮没有效用监督、
weighted BPR、额外 SSM 或新对齐损失，也不预设会涨点。**

起点为固定版本的完整 REARM。官方源码保持不变；扩展在 `rearm_relations.py`，
每个分支调用原始 `REARM.forward`，只临时替换 item 同构图。
保留原有 UU、UI 传播、attention、meta、ID embedding、可训练模态表、优化器及早停。

## 六组对照

| 模式 | Item 图 | 分支 | 融合 |
|---|---|---|---|
| original | 原始 REARM 图 | 1 | 原始预测 |
| merged_single | 四类新图等权合并 | 1 | 原始预测 |
| merged_matched | 四份相同的新合并图 | 4 | 用户门控，容量/计算对照 |
| typed_fixed | 四类关系各自成图 | 4 | 固定 1/4 |
| typed_global | 同上 | 4 | 所有用户共享可学习权重 |
| typed_user | 同上 | 4 | 根据训练历史得到用户权重 |

四类关系依次为：TRAIN 共现、图文共同邻居、视觉独有邻居、文本独有邻居。
共现图沿用官方 `item_co_graph`（train-only fallback），语义图由初始原始特征
计算正余弦 top-k，排除自身。共同邻居取两个余弦的平均值；独有仅表示不在另一侧
top-k 中，不等价于负关系。关系图逐行归一化；没有邻居的行保留自身，记录覆盖率。
图固定不更新，无 ID 顺序、分组或 PE。

这里重新启用了两个模态构图，并改变了原始图的权重/归一化方式，**必须先比较
original 与 merged_single，不能把后续全部收益都归因于分支化**。

## 模型和损失

所有分支共享同一套 REARM 参数；不是四个独立模型，也不新增四套 user ID 表。
不同 item 图经原始 UI 传播产生不同的用户表示，实现对应关系的用户偏好。

    z^r = REARM_forward(A_ii = A_r)
    s(u,i) = sum_r g(u,r) * dot(z_u^r, z_i^r)

通过拼接 `g(u,r)*z_u^r` 和 `z_i^r` 计算内积，避免分支间交叉项。
合并图单分支完全保持原式；四分支权重和为 1，避免仅因拼接就将分数放大四倍。
每个分支内部保留官方归一化；不额外改成 cosine 打分。

门控输入是 TRAIN 历史图文均值经**初始** user 投影层映射并归一化后的固定 128 维表示，
不读 held-out 交互。门控仅为 Linear(128,4)+softmax，共 516 个参数，初始化为零。
typed_global 将所有用户 logits 取平均后共享；typed_user 保留每个用户的 logits。
merged_matched 采用相同门控和参数量，但四分支图相同。

    L = BPR(s)
        + lambda_CL * mean_r CL_REARM(z^r)
        + lambda_diff * mean_r Orthogonal_REARM(z^r)

CL/正交保持官方公式与各自模态中间表示；逐分支求均值，避免把损失权重扩大四倍。
AdamW 正则、默认学习率及 epoch/早停与官方基线一致。不增加熵正则。

重要限制：分支化还会改变非线性处理顺序和 dropout 路径。merged_matched 用于控制这些
效应；评估模式下相同图四分支与 merged_single 分数应相同，本地测试已覆盖。
四分支计算和激活内存明显增加，本轮不宣称简化或提速。

## 协议与诊断

- 六组统一 TRAIN-only 负采样和 fallback 计数。不可直接对比旧 official all-split 结果。
- 默认 Baby，seed=2025，共六次；每次最多 2000 epoch，validation Recall@20 早停。
- 测试结果来自最佳 validation 轮次，不用 test 选组。
- 不保存模型/图检查点；输入转换后的 split 和特征软链接保留以便复查。
- `relation_diagnostics.jsonl` 记录各关系覆盖率、门控均值/用户间标准差/熵、最后训练
  batch 的加权分支分数标准差。它们是诊断量，不是特征贡献的因果测量。
- 每组 `manifest.json` 记录输入哈希、代码哈希、配置、参数量和图覆盖率。
- 根目录 `summary.csv` / `summary.json` 汇总各组最佳验证对应结果。多 seed 时输出均值、
  样本标准差；单 seed 标准差为空。只有 complete 的运行参与汇总。
- 失败即停止，保留日志；新运行使用唯一输出名，避免覆盖和队列锁冲突。
- `--resume` 仅允许相同计划/源码，跳过完整已完成项；失败或中断的子目录不会自动重跑。
  重试该模式请使用新输出根目录和 `--variants`，不删除已有结果。

## 服务器：第一轮六组

在项目目录执行。每个启动命令后都给出日志查看命令，外层日志包含每轮训练进度。

```bash
cd /root/autodl-tmp/MMRec-Ours
git pull --ff-only > rearm-relations-pull.log 2>&1
tail -n 30 rearm-relations-pull.log
```

```bash
RUN="night_runs/baby-rearm-relations-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p night_runs
nohup bash experiments/run_rearm_relations.sh \
  /root/autodl-tmp/MMRec-Ours/data 0 "$RUN" baby > "$RUN.log" 2>&1 &
echo "结果: $RUN；总日志: $RUN.log"
tail -n 60 -f "$RUN.log"
```

Ctrl+C 只退出 tail，不停止训练。队列本身收到 SIGTERM/SIGINT 会停止当前子进程。
不自动停止其他任务，不同时往同一输出目录启动第二条队列。

如果原 REARM 依赖没有安装：

```bash
python -m pip install -r experiments/requirements-rearm.txt > rearm-install.log 2>&1
tail -n 40 rearm-install.log
```

## 第二轮：仅复核有信号的组

下面是额外两个 seed 的示例，**等待第一轮验证结果再决定是否运行及选择哪些组**。
不要按 test 选组。Sports 可同样指定 `--dataset sports`，它有独立官方超参。

```bash
RUN="night_runs/baby-rearm-relations-confirm-$(date +%Y%m%d-%H%M%S)-$$"
nohup python -u experiments/run_rearm_relations.py \
  --data-path /root/autodl-tmp/MMRec-Ours/data --dataset baby --gpu-id 0 \
  --seeds 2026 2027 --variants original merged_matched typed_fixed typed_global typed_user \
  --output "$RUN" > "$RUN.log" 2>&1 &
tail -n 60 -f "$RUN.log"
```

## 如何判读

1. merged_single vs original：新构图本身是否改善结果？
2. typed_fixed vs merged_single/merged_matched：分开处理关系是否值得？
3. typed_global vs typed_fixed：是否只是固定 1/4 配比不合适？
4. typed_user vs typed_global：个性化选择是否比全局调权重有效？
5. typed_user vs merged_matched：排除额外 gate 参数和多分支计算的解释。

若 typed_user 的用户间门控标准差接近零，就不能解释为学到了用户差异。
单次 0.0001 的变化不形成结论；有方向性的收益后再多 seed，并检查计算开销。

## 发回审阅

先发根目录 `summary.csv`、`summary.json`、`manifest.json`、`status.json`。
完整轻量日志包可这样生成（不包含 data/、特征和模型）：

```bash
find "$RUN" -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.csv' -o -name '*.log' \) \
  -print0 | tar --null -czvf "$RUN-results.tar.gz" --files-from=- > "$RUN-pack.log" 2>&1
tail -n 30 "$RUN-pack.log"
```

## 本地检查

测试覆盖语义邻居编号置换一致性、关系交集/差集、空图、行归一化、成对初始化、
分支得分尺度、训练/全排序一致性、原损失等价、门控梯度、六组完整小数据训练评估、
无检查点、重复目录保护和 resume。小数据仅用于正确性检查，不能证明真实数据涨点。

```bash
python -m unittest discover -s tests -p 'test_rearm*.py' -v > rearm-relations-tests.log 2>&1
tail -n 60 rearm-relations-tests.log
```
