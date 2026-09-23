# REARM：以 item alignment 替换 item 同构传播

仅改变 item 同构处理。`REARMItemLoss` 继承逐字节保留的官方 REARM，设 `n_ii_layers=0`，
从而跳过拼接的 [item ID, visual, text] 在 `stre_ii_graph` 上的传播；后续归一化、
attention、user 同构传播、UI 传播、meta、官方各项 loss 和 AdamW 不变。
不使用任何编号顺序信息。

## 损失

对当前 batch 去重后的正例物品集合 B：

    L_II = mean_{i in B} sum_{j in N(i)} w_ij softplus(-e_i^T e_j)
    L_total = L_REARM + lambda_II * L_II

- e 为传播前的可训练 item ID embedding，与用户提出的公式一致。
- N 和原始权重来自完整基线的 `stre_ii_graph`（共现与模态相似混合图），不重选 top-k。
- 去掉自环、非正权重后逐行归一化；空邻居行贡献零。
- 使用 softplus(-score) 等价计算 -log(sigmoid(score))，避免数值下溢。
- 原示例 `knn_sim[items][:, knn_neighbour]` 不是逐行邻居权重。
  本实现直接存储与邻居 ID 对齐的稀疏边权，避免错误广播，也不构建 N×N 稠密权重。
- 使用均值和归一化边权替代原始 sum，避免 batch 大小/邻居数改变有效正则强度。
- 只对 ID 施加约束；原传播还作用于模态特征，因此这是替换假设，不是数学等价变换。
- 构图目前保留，用于提供邻居；简化的是 item 消息传播计算，不宣称无需图预处理。

## 实验

单 seed=2025，官方 Baby 配置与 official 负采样协议。顺序运行：

| 名称 | item 传播 | alignment 权重 |
|---|---|---|
| original | 官方 1 层 | 0 |
| none | 0 层 | 0 |
| alignment_0.01 | 0 层 | 0.01 |
| alignment_0.1 | 0 层 | 0.1 |
| alignment_1 | 0 层 | 1 |

以验证 Recall@20 选权重，测试集只用于报告对应轮次。保留原始总损失日志，并额外输出
`II alignment: raw=... weighted=...`；alignment 不冒充原官方 reg/CL/diff 分量。

在服务器项目根目录运行（新输出目录；遇到失败自动停止，不覆盖旧结果）：

```bash
nohup bash experiments/run_rearm_item.sh \
  /root/autodl-tmp/MMRec-Ours/data 0 night_runs/baby-rearm-item-1 \
  > baby-rearm-item-1.log 2>&1 &
tail -n 50 -f baby-rearm-item-1.log
```

不保存模型。五组各自有 result.json、summary.csv、manifest.json、status.json 和 log/。
运行后可打包所有轻量结果（不含 data/）给我审阅：

```bash
find night_runs/baby-rearm-item-1 -type f \
  \( -name '*.json' -o -name '*.csv' -o -name '*.log' \) \
  -print0 | tar --null -czvf baby-rearm-item-results.tar.gz --files-from=- \
  > baby-rearm-item-pack.log 2>&1
tail -n 30 baby-rearm-item-pack.log
```

本地验证：损失公式及梯度、官方源码哈希、图构建对照、original/none/alignment 三种模式
完整小数据训练/验证/结果导出、无模型保存。尚无真实 Baby 增益结论。
