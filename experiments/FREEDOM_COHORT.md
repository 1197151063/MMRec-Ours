# FreedomAlign 用户邻居残差：四组主模型实验

目的：检查内部隐藏交互实验中的 residual_semantic 信号能否转移到主模型。
不以连接近编号用户为目标，也不宣称内部投票的约 18% 增益能转移到主模型。

## 模型与 loss

1. 原有 64 维 user/item ID，默认 Xavier gain=2 初始化。
2. 原有两层 TRAIN u-i LightGCN，平均 0/1/2 层，得到 U、I。
3. 一次用户残差 `U_new = U + 0.1 * A_uu @ U`。I 保持原有前传结果。
4. 训练与推理都使用 U_new。BPR 和两个多模态 SSM 使用同一份 U_new；item alignment 仍使用原始 item ID embedding。

总目标继承 FreedomAlign：

`BPR(U_new, I) + 0.01*(SSM_visual + SSM_text) + 0.0005*(0.1*II_visual + 0.9*II_text)`。

不继承 FreedomAlignUU，无 u-u 吸引损失，没有新增可训练参数。
这是一次显式 u-u message passing，不能称为无图传播模型。

## 四组

| cohort_kind | 用户邻居 |
|---|---|
| none | 完全等价于原 FreedomAlign |
| shared_cosine | 共享物品数 / sqrt(双方训练度数乘积) 排序 |
| shared_random | 在共享物品候选中，按独立随机用户键取邻居 |
| residual_semantic | 共享余弦 × max(0, 剩余图像/文本平均余弦) |

三种图均最多 80 邻居、去除自身边、等权行归一化、beta=0.1。孤立用户残差为零。
语义条件移除用户对的**全部共同物品**；任一模态的剩余表示缺失不建该边。
图固定使用初始内容与 TRAIN 交互构建，随后不随可训练内容 embedding 更新。
图像、文本独立按物品归一化、高斯投影到 64 维再归一化，与上一轮内部预测的语义评分一致。
图种子固定 2026；随机键用于打破平分，不包含 ID 大小、距离、group 或 PE。
精确编号置换验证需将独立随机键跟随用户一起置换。

构图在 CPU 一次完成，每次只计算一行候选，不存储 dense U×U 矩阵。
构图耗时、边数、孤立用户数和图哈希写入每组 console.log。图不写磁盘缓存，避免用错数据/划分。
训练使用稀疏图聚合，继承原模型的评估缓存；没有模型 checkpoint。
四组使用相同 seed=999，新增构图不会消耗全局 NumPy/PyTorch 训练随机流。

## 服务器运行和日志

```bash
cd /root/autodl-tmp/MMRec-Ours
git pull --ff-only
COHORT_MODEL_RUN="night_runs/baby-cohort-model-$(date +%Y%m%d-%H%M%S)"
mkdir -p night_runs
echo "$COHORT_MODEL_RUN"
nohup bash experiments/run_freedom_cohort.sh \
  /root/autodl-tmp/MMRec-Ours/data 0 "$COHORT_MODEL_RUN" baby \
  > "${COHORT_MODEL_RUN}.log" 2>&1 &

# 队列日志
tail -n 60 -f "${COHORT_MODEL_RUN}.log"
```

具体训练日志在各子目录，例如基线：

```bash
tail -n 60 -f "$COHORT_MODEL_RUN/freedom_cohort_none_s999/console.log"
```

剩余语义组启动后：

```bash
tail -n 60 -f "$COHORT_MODEL_RUN/freedom_cohort_residual_semantic_s999/console.log"
```

完成后提供根目录 `summary.json`、`summary.csv`、`manifest.json`。
若失败，加失败组的 `console.log` 和 `status.json`。
队列可在代码/数据/参数不变且旧队列已退出时用原路径续跑；跳过完整组。
更新代码后必须新建输出目录。新终端需用打印的路径恢复变量。

## 如何判断

- 先核对 none 能否复现原基线，避免把训练差异当作 u-u 收益。
- 正式验证集选最佳 epoch，测试值仅报告对应 epoch。
- residual_semantic 需要与 shared_random、shared_cosine 一起比较；否则不能归因于语义筛选。
- 固定 beta=0.1 是小规模初筛，失败不代表所有聚合强度无效；成功也需多训练种子复核。
- 本轮保留投影、损失、初始化、UI层数等设置，不同时调参。

## 本地测试和日志

```bash
python -m unittest discover -s tests -p test_freedom_cohort.py -v \
  > /tmp/mmrec-freedom-cohort-tests.log 2>&1
tail -n 80 /tmp/mmrec-freedom-cohort-tests.log
```

验证图评分与上一轮诊断实现一致、全部共同物品剔除、稀疏图行归一化/孤立用户、编号置换、训练随机流隔离、基线精确一致、前传/梯度/评估缓存、四组端到端队列与续跑、不保存 checkpoint。
