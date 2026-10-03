# 用户图候选与排序审计

目标：在不使用ID距离构图的前提下，区分相邻用户连接不足的原因。
本次只构图和诊断，不训练推荐模型，不使用验证/测试边，不宣称推荐增益。

## 审计

重新计算 shared_cosine 和 semantic_anchor 的完整候选得分，而非仅检查已有top20文件。
对所有活跃用户，检查编号精确相差1/4/16/64/256的有效有向用户对，两方向均包含。
每个距离分别报告：

- eligible_pairs：待检查用户对的数量。
- candidate_coverage：其中进入正分数候选的比例。
- conditional_rank_quantiles：进入候选后的排名分位数。
- conditional_mean_tie_count：目标边与多少候选拥有相同计算得分。
- topk：K=5/10/20/40/80的保留率，分别以全部用户对、进入候选的用户对为分母。
- tie_cutoff：多少候选用户对的同分区间跨越topK边界。
- expected_candidate_recall_random_ties：并列时均匀随机打破的期望保留率。
- optimistic/pessimistic：允许并列名次变化时的理论上下界，仅用于诊断，不能用ID决定并列名次。

这些是全量有向对，不是之前每个距离抽10000对，因此候选覆盖率可能与旧报告略有差异。
图的 `edge_fraction_within_window` 分母是全部图边；`exact_offset_pair_recall`
分母是该精确间隔的全部有效用户对。两者不可混用。

## 候选分配对照图

item_balanced：每个用户交互过的物品各形成一个候选桶（同样交互过该物品的其他用户）。
桶内按原共同邻居 cosine 排序；各桶按独立随机物品键决定轮转顺序，
每轮每桶取一个尚未选中的用户，直到选满K或候选耗尽。
不使用物品ID大小/用户ID距离决定桶次序；重复候选只选一次。
选边权重仍是原 cosine，另外保存行归一化版本。

这是一种启发式，不保证近编号保留率或推荐效果提高，尤其低度用户本来就很少有桶可分配。
不能用近编号保留率来挑seed或调参。

## 邻居来源集中度

对于已选邻居，将一条边的1份来源额度均分给它与目标用户共享的所有物品。
报告最大物品来源占比、HHI、具有其他交互用户的历史物品覆盖率。
一个邻居可能由多个物品共同解释，此处是统计定义，不是唯一或因果归因。
对照 shared_cosine 与 item_balanced 能看出轮转分配是否真正增加来源多样性。

## 运行

代码包须同时包含 `audit_user_candidates.py` 和其依赖 `build_semantic_user_graphs.py`。
CPU即可，稀疏全候选统计会比上一轮更慢；日志定期打印已处理用户数。
只保存3张图的原始及行归一化版本，共6个npz文件，无模型保存。

在服务器项目目录执行：

```bash
nohup python -u experiments/audit_user_candidates.py \
  --data-path /root/autodl-tmp/MMRec-Ours/data --dataset baby \
  --output night_runs/baby-user-rank-1 \
  > baby-user-rank-1.log 2>&1 &
tail -n 50 -f baby-user-rank-1.log
```

输出目录必须不存在。完成后日志显示 Finished，返回 report.json 和日志即可。

## 解释顺序

先看候选覆盖率：缺失的对不可能靠重排恢复。
再看条件排名：相似度分数导致的落后与同分截断需要区别处理。
最后看分桶图能否降低来源集中度；若成立，再在同一推荐模型/训练协议内做验证集对照。
本轮没有把“近编号”作为边的监督标签。
