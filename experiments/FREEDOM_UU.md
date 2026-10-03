# FreedomAlign + 完整共同物品/弱语义桥 UU constraint

主模型是已经实测 Baby test R@20=0.0962 的 FreedomAlign，不是 REARM。
实现类 FreedomAlignUU 继承 FreedomAlign，不改前传或已有BPR、模态SSM、II项。
不进行user-user消息传播，不读取ID距离；保留原训练负采样协议。

## 结构与目标

设 R 为二值训练交互，D_U、D_I 为两侧度数。
共同物品转移 P_c=D_U^{-1} R D_I^{-1} R^T。
语义转移 P_s=D_U^{-1} R S D_I^{-1} R^T。
S 从原始图像/文本特征分别归一化、拼接、再次归一化后计算正余弦top-5，
去除物品自环，再用双向最大权重并集无向化，最后按行概率采样。
原始邻接对称，行归一化后的转移通常不对称。

新增损失只作用于传播前的 user ID embedding：

    L_c = mean_{u in interaction batch} sum_{v != u} P_c[u,v] softplus(-e_u dot e_v)
    L_s = mean_{u in interaction batch} sum_{v != u} P_s[u,v] softplus(-e_u dot e_v)
    L_total = L_FreedomAlign + lambda_uu * (L_c + beta * L_s)

user在interaction batch重复出现时保留其重复次数，与主训练交互分布一致。
P删除自环后的行和可以小于1，语义无邻居或终点物品无训练用户时也产生零贡献。
这是明确的sub-stochastic转移目标，不是删掉自环后重新归一化的图；不静默替换二者。

## 不存稠密UU图：路径采样估计

- 共同物品分支：均匀选用户一个训练物品，再均匀选该物品的交互用户。
- 语义分支：先均匀选历史物品，再按S边权选相似物品，再均匀选其交互用户。
- 每个用户每个启用分支采16条路径，可重复；16是Monte Carlo样本数，不是user top-K。
- 回到自身/无可用终点的路径置零，不重采；固定分母B×16，使估计对上述目标无偏。
- 所有有正路径概率的用户关系都保留；每个batch只计算抽中的边，跨batch可访问不同邻居。
- 两个分支使用独立局部随机数流，不改变主模型初始化和训练负采样的全局随机状态。
- 不用之前诊断的ID近邻标签，不加载外部graph cache，也不保存模型/图checkpoint。

完整共同物品结构无需展开U×U矩阵；仅保存训练邻接和稀疏item语义图。
和此前Boolean连通性研究不同，这里使用随机游走概率赋予边权，语义边还用余弦强度。
UU为正关系吸引，不等价于传播，也不保证涨点；主BPR仍负责区分正负物品。

## 首轮7组

固定原有II=5e-4(sum)、模态SSM=.01、seed=999、16条路径、item top-5：

| 分组 | lambda_uu | beta |
|---|---|---|
| baseline | 0 | 0 |
| shared_w0.01 / 0.1 / 1.0 | .01 / .1 / 1 | 0 |
| bridge_w0.01 / 0.1 / 1.0 | .01 / .1 / 1 | .1 |

选择配置只用验证Recall@20，测试只报告对应轮次。此轮是单seed机制筛选，不能据此宣称统计显著。
先检查baseline是否复现之前的量级，再判断共享关系是否有效、弱语义桥是否带来增量。
日志记录UU原始值、加权贡献、自环/死路径过滤后的有效路径比例及总损失。

## 运行和日志

服务器项目根目录，新输出目录：

```bash
nohup bash experiments/run_freedom_uu.sh \
  /root/autodl-tmp/MMRec-Ours/data 0 night_runs/baby-freedom-uu-1 baby \
  > baby-freedom-uu-1.log 2>&1 &
tail -n 50 -f baby-freedom-uu-1.log
```

外层日志显示队列状态，当前组训练细节看其console.log，例如：

```bash
tail -n 50 -f night_runs/baby-freedom-uu-1/freedom_uu_baseline_s999/console.log
```

脚本最多12小时，每组最多90分钟、1000epoch，正常情况按验证早停。无模型保存。
使用相同源码和参数可重新执行同一命令，跳过已完成组；manifest发生变化需新目录。
返回根目录summary.csv、summary.json、manifest.json和各组console.log。

## 本地验证

- 共享/语义采样频率与精确矩阵转移概率对照，包含自环mask。
- 固定分母损失公式与有限梯度、空路径处理。
- 局部随机流独立、不扰动初始化；lambda=0与FreedomAlign损失一致。
- 七组小数据完整队列、结果导出、无checkpoint。

尚未运行真实Baby，收益需要服务器结果确认。
