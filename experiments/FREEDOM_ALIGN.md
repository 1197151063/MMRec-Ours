# FreedomAlign：用户提供的 FREEDOM 变体

独立模型 `src/models/freedomalign.py`，不覆盖 FREEDOM 或 REARM。
本实现复现所提供代码的有效计算路径，修复 item 权重索引；不是错误广播版本的数值复现。

## 模型和目标

- 用户和物品 ID 为64维，Xavier uniform gain=2。
- 训练集二分 UI 图归一化，2层 LightGCN，对0/1/2层求均值。
- 图像/文本特征表可训练，两侧各一个 Linear 投影到64维。
- 原始特征各自建立静态 cosine top-20 邻居；保持自环、原始有符号 cosine 值，不行归一化。
- 不进行 item-item 消息传播；预测只使用 UI LightGCN 的 user/item 内积。

    L = BPR(U_UI, I_UI)
        + .01 * [SSM(U_UI, X_visual) + SSM(U_UI, X_text)]
        + .0005 * [.1 * Align(I_raw, KNN_visual) + .9 * Align(I_raw, KNN_text)]

SSM 使用归一化后的向量和温度0.1：

    mean_b[logsumexp_c(cos(u_b, x_neg_c)/tau) - cos(u_b, x_pos_b)/tau]

分母是本 batch 的负物品池，保留重复负例；没有加入正例项，因此不是标准 InfoNCE，
其值可以为负。主 BPR 用每个用户自己的一个采样负例，均值归约。

Align 使用未传播的 item ID：

    sum_{b,k} cosine_raw(i_b, neighbor_bk) * softplus(-e_i_b dot e_neighbor_bk)

保留正例物品重复出现的次数；使用 sum 而非 mean。默认无显式 L2、Adam weight_decay=0，
与提供代码里未调用 norm_loss/L2_regloss 的事实一致。

## 修复与删除

1. 原 `knn_sim[items][:, knn_neighbour]` 是 [B,B,K]，会把不同锚点的相似度混入每条边。
   正确稠密写法为 `knn_sim[items[:,None], knn_neighbour]`，本实现直接存 [N,K] 边权。
2. 使用 softplus 和 logsumexp，保留目标公式，避免 log(sigmoid)/exp 的下溢或溢出。
3. KNN 按块计算，永久保存量从每模态 N×N 降为 N×K；计算时间仍是全对相似度量级。
4. 删除未生效的 degree-sensitive pruning：原 calculate_loss 调用 norm_adj，不使用 masked_adj。
   这也移除了无效采样的 RNG 消耗，不能承诺与旧脚本逐步轨迹完全一致。
5. 删除未使用的损失和配置，移除硬编码 .cuda()，支持 CPU/GPU；兼容新版 SciPy。
6. 投影仅计算 batch 正负例的去重行，线性投影无 BN/dropout，与全表投影数学一致。
7. 评估缓存 UI 表示，切换训练或计算训练损失时清除。

## 改进含义与限制

相对原 FREEDOM：保留 UI 传播，取消 item 传播分支，用训练时的邻居约束传递内容关系；
模态 BPR 换成 batch 负例池的 SSM；同时该提供版本已经不使用剪枝图。
因此不能把与原 FREEDOM 的性能差异全部归因于 item alignment。

相对 REARM：无 user 同构传播、注意力或 meta 网络，但这是一条独立模型路线，
不是在 REARM 上只做 item 单因素替换；此前 REARM 对照应独立保存。

合理的假设是：推荐目标约束排序，模态目标引导用户偏好，静态语义邻居约束物品表示。
此约束不等价于传播。只拉近正邻居可能增加范数或压缩区分性，自环项也会推动范数；
原始 cosine 中的负值会产生排斥项。默认忠实保留这些行为，先看实测。

`.0005` 的作用不可直接与此前 mean 版本比较：此处累加 B×K 项。
当 B=2048,K=20 时，权重前后量级可能显著大于 BPR，日志同时给出原始值和加权值。
另外，修正 [B,B,K] 广播后有效损失尺度改变，旧错误版本的增益与系数不能直接迁移。

## 配置与验证协议

默认 Adam lr=.001、seed=999、batch=2048、64维、最多1000epoch、验证早停；
这些训练默认来自当前 MMRec 仓库，用户代码未提供完整外部配置。
负采样只排除训练正例；对其他用户共享的 batch 负例池不额外过滤，因此可能有假负例。
与 REARM official 排除所有 split 正例的协议不同，不能据此进行严格单因素性能比较。

四组匹配消融：full / no_ii / no_modal / ui_only。
各组保持相同初始化方式、图、优化器和评估流程；不自动跑额外超参网格。

本地已通过：修正后稠密公式与实现对照、所有训练参数有限梯度、UI多项式验证、
分块KNN验证、重复anchor求和、四组CPU单epoch全链路、无checkpoint。
真实数据效果由服务器运行确认。

## 运行与日志

服务器项目根目录执行，选择新输出目录：

```bash
nohup bash experiments/run_freedom_align.sh \
  /root/autodl-tmp/MMRec-Ours/data 0 night_runs/baby-freedom-align-1 baby \
  > baby-freedom-align-1.log 2>&1 &
tail -n 50 -f baby-freedom-align-1.log
```

总预算12小时，每组上限180分钟。不保存模型。外层日志显示队列状态；训练细节看：

```bash
tail -n 50 -f night_runs/baby-freedom-align-1/freedom_align_full_s999/console.log
```

结束后提供根输出目录 summary.csv、summary.json、manifest.json，以及各组 console.log。
遇到 `timeout` 不是正常早停，需要增加单组预算重新运行；不能作为完整结果比较。
