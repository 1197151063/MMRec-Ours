# 无编号先验的 user-user 图：四个预先固定的假设

所有构图只用训练边和原始 item 图像/文本特征。ID 距离仅用于输出诊断，不能作为选图或调参目标。
输入目录为已有 MMRec 数据格式；不读旧模型，不读取验证/测试边用于构图。

1. shared_cosine：二值交互矩阵 R 的行 cosine，top-20。
2. overlap：共享物品数 / min(d_u,d_v)，强调较小历史被另一用户覆盖；可能过度偏向低度用户，作为对照。
3. semantic_anchor：令 f_i 为分别归一化后拼接并再次归一化的图像/文本特征。
   h_{u,-i}=normalize(sum_{j in N(u),j!=i} f_j)，
   a_ui=(1+cos(h_{u,-i},f_i))/2，X_ui=R_ui*a_ui；用 X 行 cosine 构图。
   计算语义一致性时排除当前锚点物品，避免它自我证明相关；只有一次交互时取中性0.5。
   无热门物品惩罚，不保证会保留更多近编号边。
4. semantic_bridge：以 cosine(f_i,f_j) 构建正相似度 item top-10（无自环），行归一化为 S。
   T=.5I+.5S，X=RT，再按 X 行 cosine 构图；无语义邻居物品的 T 行取单位行。
   允许用户通过不同但语义接近的物品建立联系。
   XX^T=RTT^TR^T（归一化前），因此含共享端点的语义路径，并非单纯 RSR^T。

四种都去除用户自环，保留正分数，top-20 用随机实体键打破并列而非 ID 顺序。
同一轮保持这些键一致；严格重编号等价需要同步置换键，已做小数据测试。
孤立用户可没有邻居，不用编号邻居填补。

## 输出

每种图保存 raw `.npz` 和逐行归一化 `_rownorm.npz`（scipy CSR）。
报告含输入哈希、参数、边数、有邻居用户数、共享训练物品的边占比、ID邻近边占比。
ID占比用于审计/分析，不证明推荐增益、真实群组或因果关系。

这组是机制探索，不承诺恢复编号顺序。相同可观测历史/内容不能唯一确定原始编号关系。
若用 ID近邻连接率挑选模型，仍会重新依赖编号，所以最终应固定协议用验证推荐指标选型。
目前不修改推荐模型，不加入新 message passing，也不保存 checkpoint。

## 服务器运行

CPU即可；全对相似度和稀疏乘法会花时间。分块限制中间矩阵大小，仍有二次计算成本。
新输出目录必须不存在，以避免混合图文件。

```bash
nohup python -u experiments/build_semantic_user_graphs.py \
  --data-path /root/autodl-tmp/MMRec-Ours/data --dataset baby \
  --output night_runs/baby-semantic-uu-1 \
  > baby-semantic-uu-1.log 2>&1 &
tail -n 50 -f baby-semantic-uu-1.log
```

运行结束应出现 Finished 日志，report.json 应有全部4种图。
返回 report.json 和日志即可先审阅；本地测试仅覆盖公式、无自环、语义桥、同步实体重编号。
