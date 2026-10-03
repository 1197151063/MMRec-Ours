# 冻结原始特征 × 语义评分残差：四组交叉实验

主模型为 FreedomAlignSemantic，继承原 FreedomAlign，未接入任何新增 u-u 模块。
这轮只检查冻结和直接语义评分两个因素，不同时修改 SSM 负样本或 item alignment。

## 四组与固定参数

| 名称 | 原始图像/文本特征表 | 语义评分系数 beta |
|---|---|---|
| baseline | 可训练 | 0 |
| freeze | 冻结 | 0 |
| score | 可训练 | 0.1 |
| freeze_score | 冻结 | 0.1 |

四组保持 seed=999、两层 u-i LightGCN、64 维 ID/单层模态投影、默认 Xavier gain=2 初始化。
冻结组只是将原始特征表 requires_grad 设为 False；两个 Linear 投影仍然训练。
没有新增可训练参数。所有组初始化参数数值、全局训练随机流保持一致。

令 U、I 为 LightGCN 的用户/物品表示，Z_v、Z_t 为原始特征的单层投影，组合评分为：

`s(u,i) = dot(U_u,I_i) + beta*[0.5*cos(U_u,Z_v_i) + 0.5*cos(U_u,Z_t_i)]`。

图像/文本各自先归一化再混合，混合结果不再次归一化。这是两项余弦的加权和。
语义评分的图像系数 0.5 与原 item alignment 的图像系数 0.1 是不同参数。
训练 BPR 为 `mean(softplus(s(u,negative)-s(u,positive)))`；推理用完全相同的组合评分。
beta=0 直接调用父类损失和预测，保持基线精确一致。

总 loss：

`BPR(combined_score) + 0.01*(SSM_visual+SSM_text) + 0.0005*(0.1*II_visual+0.9*II_text)`。

SSM 仍为原来的 cosine/temperature=0.1、batch negatives-only denominator；II 保留原始特征初始 KNN、self 邻居、raw cosine 权重、重复正物品锚点和 sum reduction。

训练只投影当前批次需要的物品行；评估缓存全物品投影和 CF 表示。train()/calculate_loss() 清理缓存。
不保存模型文件。GPU 用于训练，最大 epoch=1000，原有验证早停和最佳验证轮次对应测试结果保持不变。

## 日志与解释

- 每组启动记录可训练参数数、freeze 开关、beta 和图像系数。
- 语义评分组每 200 batch 记录 CF 分数标准差、beta 加权语义分数标准差，以及两部分正负分数差的平均绝对值。它们是当前采样批次的统计，不是全物品分布。
- 固定随机抽样至多 256 个物品，在第 1 个已完成 epoch 和此后每 10 个 epoch 记录原始特征相对初值的余弦与相对 L2 变化。抽样使用独立 RNG，不影响训练；不保存完整初始特征副本。
- 漂移统计只说明原始特征变化程度，不能单独证明“语义被破坏”；必须结合验证指标。冻结组非零原始特征的初始余弦应接近 1，相对 L2 应为 0。
- beta=0.1 是固定的小规模初筛。点积与余弦量级不同，失败后应先看分数日志，不应直接断言语义无效。
- 若组合组提高，分别对比 baseline、freeze 和 score，才能判断两个因素是否互补。

## 服务器运行和日志

```bash
cd /root/autodl-tmp/MMRec-Ours
git pull --ff-only
SEMANTIC_RUN="night_runs/baby-semantic-$(date +%Y%m%d-%H%M%S)"
mkdir -p night_runs
echo "$SEMANTIC_RUN"
nohup bash experiments/run_freedom_semantic.sh \
  /root/autodl-tmp/MMRec-Ours/data 0 "$SEMANTIC_RUN" baby \
  > "${SEMANTIC_RUN}.log" 2>&1 &

# 队列日志
tail -n 60 -f "${SEMANTIC_RUN}.log"
```

基线启动后查看训练日志：

```bash
tail -n 60 -f "$SEMANTIC_RUN/freedom_semantic_baseline_s999/console.log"
```

score 组启动后查看评分量级和特征漂移：

```bash
tail -n 80 -f "$SEMANTIC_RUN/freedom_semantic_score_s999/console.log"
```

其他子目录为 `freedom_semantic_freeze_s999` 和 `freedom_semantic_freeze_score_s999`。
新终端请用打印出的路径恢复 SEMANTIC_RUN 变量。
只有代码/数据/参数不变且旧队列已退出时，才能用同一路径续跑并跳过完整组。

## 需要回传的文件

根目录 `summary.json`、`summary.csv`、`manifest.json`，以及 `score` 和 `freeze_score` 两组的 `console.log`。
如果失败，附失败组 `status.json` 和 `console.log`。
新版 manifest 增加每个源码文件的 SHA256 清单。若服务器总哈希不同，可以定位具体差异，而不只是看到总摘要不一致。

## 本地验证和日志

```bash
python -m unittest discover -s tests -p test_freedom_semantic.py -v \
  > /tmp/mmrec-freedom-semantic-tests.log 2>&1
tail -n 80 /tmp/mmrec-freedom-semantic-tests.log
```

覆盖基线参数/损失/梯度/RNG 精确一致、BPR 与推理分数一致、原辅助损失保持一致、冻结特征不更新但投影可训练、BPR 本身能够训练语义分支、评估缓存、四组一轮端到端队列、不保存 checkpoint 和源码清单。
本地只有合成数据；真实 Baby 性能需服务器运行后判断。
