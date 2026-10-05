# REARM + 协同过滤：72 个预先定义的实验

更新：默认跳过原始 REARM，`all` 运行 71 个新配置、`quick` 运行 13 个。
已有 Baby/2025 基线单列于汇总的 `historical_references`，保留来源和输入/设置核对结果。
需要完整重跑含基线的 72/14 组时显式加 `--include-reference`。
下面的 72 组清单仍包含历史参照，用于解释原始实验设计。

## 本轮目标与模型

回到完整 REARM，测试哪些 CF 改动确实有用。本轮不再使用上轮的四关系分支/门控，
也不使用 ID 顺序、PE、连续 ID 分组。上游固定版本源码不修改。

保留 REARM 的 item/user 同构图、图文 self/cross attention、UI LightGCN、低秩
meta transformation，以及最后的三分支拼接内积预测。默认保留原图层数、AdamW、
学习率和正交损失。图像/文本投影仍是 Linear 到 64 维。

主要起点 `ssm_base` 的损失为：

\[
L=L_{BPR}(z_u,z_i,z_j)+\lambda_s[\alpha L_{SSM}(h_u,W_vx_i)
 +(1-\alpha)L_{SSM}(h_u,W_tx_i)]+\lambda_d L_{diff}.
\]

- `z`：REARM 最终 192 维预测表示，BPR 和测试都用它。
- `h_u`：REARM 经过同构图和 UI 图后的 64 维 ID 分支用户表示。
- `W_m x_i`：可训练模态表经过原始 Linear 的 64 维输出。
- **SSM 替换的是原图文 CL；也改变了辅助任务的两端，从图文对比改为用户–模态对比。**
- 默认 λs=0.01、α=0.5、τ=0.1、32 个均匀负样本，cosine similarity。
- 正交损失继续使用 REARM 原来的公式和 dataset preset；不是新的正则。

你的 SSM 保留负样本分母：

\[
L_{SSM}(a,p)=\operatorname{logsumexp}_{j\in B^-} (\cos(a,j)/\tau)-\cos(a,p)/\tau.
\]

使用 logsumexp 避免 exp/log 数值溢出。此损失允许为负，不表示程序错误。
`ssm_positive_denominator` 则把正样本放回分母，单独比较两个定义。
默认排除 TRAIN positives；`ssm_legacy_sampling` 单独保留你原始代码的无过滤抽样。
不使用 valid/test positives 筛负样本。抽样允许重复。

## 72 组构成

每个配置默认一个 seed=2025。实验不是全参数笛卡尔积；独立配置及其假设都写在
`rearm_cf_plan.py` 和队列的 `manifest.json`。以下数量加起来为 72。

| 家族 | 数量 | 主要问题 |
|---|---:|---|
| 原始控制 | 2 | 完整 REARM；仅去掉原 CL |
| SSM 替换与校准 | 14 | projected/propagated、分母、抽样、权重、温度、负样本数、图文权重、CL 共存、去正交 |
| 主损失 | 2 | 最终表示用负分母 SSM 或正样本含分母 softmax 替代 BPR |
| UltraGCN 启发 | 12 | degree-weighted BPR、UI constraint、II constraint、K、BCE、去掉 UI 传播 |
| CAGCN 启发 | 6 | JC/LHN 两种 CIR，相对 LightGCN 的混合比例 0.25/0.5/1 |
| NT-SSM 启发 | 7 | 普通 SSM 控制、双向控制、用户/物品来源负分数的分开加权 |
| 训练过程重加权 | 5 | warmup 后截断高损失样本、边级 EMA 软降权、延后干预 |
| 负采样 | 4 | hard-4、hard-8、semi-hard-8、流行度抽样 |
| 优化与容量 | 6 | 学习率、AdamW decay、冻结模态/用户兴趣表、两层 UI |
| 保留原 CL 的桥接控制 | 3 | 原 CL + II、原 CL + CIR、原 CL + trim |
| 预定义组合 | 5 | UI+II、CIR+II、CIR+trim、II+semi-hard、较弱 SSM+lr |
| GraphDA 启发 | 6 | 重建 UI 的 K、残差图、加入 UU/II、教师训练时长 |

每个独立模块默认相对 `ssm_base` 比较；桥接控制相对 `reference` 比较。
组合相对其单独组件比较，不能把整个组合的收益归于其中一个组件。
`ultra_no_ui_propagation` 的直接参照是 `combo_ultra`。
`ultra_bce` 与 `combo_ultra` 比较，控制约束项不变，只换主损失。

## 四篇论文的具体借用与改写边界

### UltraGCN：把协同邻居用于训练监督

TRAIN 二值矩阵 R；用户/物品度 d_u,d_i：

\[
\beta_{ui}=d_u^{-1}\sqrt{(d_u+1)/(d_i+1)}.
\]

`ultra_ui_*` 对正/负 UI 点积分别施加 beta 加权的 softplus(-s_pos)/softplus(s_neg)，
以 TRAIN 正边平均 beta 作固定尺度归一化。
`degree_bpr_*` 是另一种适配：beta 的 0.5/1 次幂、batch 均值归一化后加权 BPR。
这两者不是同一个损失。

II 图使用论文 Eq.16：G=RᵀR，g_i=Σ_j G_ij（**含对角线**），

\[
\omega_{ij}=\frac{G_{ij}}{g_i-G_{ii}}\sqrt{g_i/g_j}.
\]

按 omega 选 top-K、排除自身，无邻居时权重为零。约束是 **u 对正物品 i 的邻居 j**：
`omega_ij * softplus(-dot(z_u,z_j))`，并非 `dot(z_i,z_j)`。
对邻居求和、对 batch 求均值，再除以全物品平均邻居权重和，避免 K 和 batch size
只通过扩大 loss 数值影响结果。归一化常数由 TRAIN 图计算，与梯度无关。

保留 REARM 编码器的这些配置属于 UltraGCN 启发的目标函数适配，不是完整 UltraGCN
复现。去掉 UI 层的配置仍保留 REARM 的 UU/II，不声称完全 graph-free。

### CAGCN：在原有边上改变传播强度

用一跳 CIR：对用户历史内的物品两两计算 Jaccard 或 LHN 共现相似度，取平均；
物品接收用户信息的方向同理。计入相似度对角项，两个方向独立计算，所以允许非对称。

phi_vw 仅在训练边上非零；依据论文附录 Eq.25：

    C_vw = row_sum(S)_v * phi_vw / row_sum(phi)_v
    S_new = (1-mix)*S + mix*C

S 是原始 LightGCN 归一化矩阵。mix=1 对应这里实现的一跳 CAGCN 权重；
mix<1 为保守混合适配，不是论文另一个 CAGCN* 的加法公式。
每行总传播强度保持一致，避免把简单分数放大误认为拓扑收益。
替换 REARM 的 `norm_adj`，同时作用在原有 ID 与模态 UI 分支，UU/II 前置图不变。
块式稀疏计算，不构造完整 U×U 或 I×I 稠密矩阵。

### NT-SSM：单独调节负样本中不同类型节点的贡献

你提供的 2605.24015 是 **Rethinking Contrastive Learning for Graph Collaborative
Filtering: Limitations and a Simple Remedy**，提出 NT-SSM，不是 memorization 的论文。

REARM 最终有非线性 attention/meta，不能直接宣称整模型具有论文的线性分解。
本实现只在 REARM 的**线性 UI ID 分支**上分解：

    h = h_from_users + h_from_items
    h_negative_weighted = a_U*h_from_users + a_I*h_from_items

起点是经过 REARM 同构处理/归一化的 ID 表示。正分数不变，负分数的分子按类型加权，
分母仍用原始完整表示的 norm；a_U=a_I=1 时严格退化到普通正分母 SSM。
测试已检查此性质。该辅助损失加到 BPR+模态 SSM+diff，不改变预测表示。
双向版本包含 item→negative-user，两个方向取平均以控制总权重；两方向系数共享。
这与论文用两方向之和、独立方向系数的完整搜索不同。所有负用户也仅按 TRAIN 排除。

### GraphDA：先学表示，再重建图，再重新训练

默认先训练 120 epoch 的完整 REARM 教师（原 BPR+CL+diff）；另有 40 epoch 控制。
教师只训练，**不做验证或测试**。然后：

1. 用户→物品、物品→用户分别 raw-dot top-K，取并集得到新 UI 图；不把预测边当成
   数据标签去改 BPR 的正例集合，正例仍是原 TRAIN。
2. 可选 teacher-derived UU/II，排除自身、做对称并集，放入联合传播图。
3. 学生恢复该 seed 开始时的所有参数值及随机状态，丢弃教师优化器状态后重新训练。
4. 学生才进入原有的 validation early stopping；教师轮次不会成为“最佳学生轮次”。

REARM 的原始前置同构模块仍保留，因此是 GraphDA 思路在 REARM 上的适配。
残差版本 `graphda_residual` 是归一化后 0.75 原图 + 0.25 新图；其他默认完全使用新图。
初始参数快照只放 CPU 内存，不写磁盘。额外 RAM 约等于一份模型参数（Baby 约 0.5 GB），
每组教师独立训练，不跨配置共享被挑选的教师。图构造计时、保留 TRAIN 边数和新边数
写到该组 `manifest.json` 的 `cf_graphs.graphda`。

## Memorization / denoising 的处理

参考 [Denoising Implicit Feedback for Recommendation](https://arxiv.org/abs/2006.04153)
的“先学容易样本，后拟合难样本”实验思路。该文的 ADT 在 BCE 上实例化；这里明确
是 BPR 上的启发式适配，没有真实噪声标签，也不把高损失自动解释成错标。

- `trim_*`：前 30 epoch 不改权重，随后 30 epoch 线性增加丢弃比例至 10%/20%。
  只在 batch 内按 detached 主损失排序。`trim_late` 改为 80 epoch 后开始。
- `ema_reweight_*`：按真实 TRAIN (u,i) 键追踪主损失 EMA（0.9）；batch 内按 EMA
  难度排名，逐渐把最难样本权重降至 0.5/0.2，不彻底删除。
- 权重均值归一化；作用于主损失和新增模态 SSM。保留的原 CL 和 diff 不重加权。
- `cf_diagnostics.jsonl` 每轮记录主损失、SSM、原 CL+SSM 总辅助项、diff、UI、II、NT
  的加权数值及权重最小/最大值。避免某项数值压过主损失而不被发现。

## 运行协议与成本

- 默认一张 GPU 串行，72 个配置 × 1 seed，最多 2000 epoch/组，沿用 REARM 早停。
- 这是大规模筛查，不承诺一个晚上完成。以之前每组约 19 分钟粗估，72 组约 23 小时，
  另加教师训练/构图成本；实际按服务器日志为准。快筛 14 组也包含一次教师训练。
- 原 REARM 完整参数量较大；冻结特征配置才减少可训练参数。默认不擅自削减模型。
- 不保存模型/优化器/图 checkpoint，只有结果、配置、诊断日志及输入转换文件/特征软链接。
- 所有构图、负采样、流行度、EMA 仅取 TRAIN；validation Recall@20 选轮次和配置。
- 测试结果仅为对应最佳验证轮次的报告值，队列排序仅按验证指标。
- 72 组里单 seed 的小涨点有选择偏差，不能直接写成显著提升。后续固定优胜配置，
  与 reference 做相同 3 seeds，再迁移 Sports/Clothing。
- 记录输入和代码 SHA256；同一目录恢复时要求计划、代码、输入一致。
- 队列继续非连续失败，连续 3 组失败自动停；缺少 CUDA/依赖时在预检阶段停止。
- 所有新增算子和 72 组反传在 CPU 合成数据验证；不是 Baby 完整训练的精度结果。

## 服务器：直接跑完整 72 组

先同步；查看同步日志确认成功后再启动：

```bash
cd /root/autodl-tmp/MMRec-Ours
git pull --ff-only > rearm-cf-pull.log 2>&1
tail -n 30 rearm-cf-pull.log
```

```bash
RUN="night_runs/baby-rearm-cf-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p night_runs
printf '%s\n' "$RUN" > night_runs/last-rearm-cf-run.txt
nohup python -u experiments/run_rearm_cf.py \
  --data-path /root/autodl-tmp/MMRec-Ours/data --dataset baby --gpu-id 0 \
  --suite all --output "$RUN" > "$RUN.log" 2>&1 &
tail -n 80 -f "$RUN.log"
```

Ctrl+C 只退出 tail；不要为了看日志再启动一条队列。
新终端查看当前组和日志：

```bash
RUN=$(cat night_runs/last-rearm-cf-run.txt)
cat "$RUN/status.json"
tail -n 80 -f "$RUN.log"
```

如只想先筛 14 组，在**新目录**跑：

```bash
RUN="night_runs/baby-rearm-cf-quick-$(date +%Y%m%d-%H%M%S)-$$"
nohup python -u experiments/run_rearm_cf.py \
  --data-path /root/autodl-tmp/MMRec-Ours/data --suite quick --output "$RUN" \
  > "$RUN.log" 2>&1 &
tail -n 80 -f "$RUN.log"
```

只恢复原 72 组任务，成功项跳过；失败/中断项的完整目录先移到 `attempts/`，再重跑：

```bash
RUN=$(cat night_runs/last-rearm-cf-run.txt)
nohup python -u experiments/run_rearm_cf.py \
  --data-path /root/autodl-tmp/MMRec-Ours/data --dataset baby --gpu-id 0 \
  --suite all --output "$RUN" --resume --retry-failed >> "$RUN.log" 2>&1 &
tail -n 80 -f "$RUN.log"
```

无 checkpoint，所以未完成的单组从头训练；已经成功的不会重跑。若源码改动，必须用新目录。
快筛的恢复命令须保持 `--suite quick` 和原目录；已退出的队列才能恢复。

预览全部配置和假设（不读数据/启动训练）：

```bash
python experiments/run_rearm_cf.py --data-path /root/autodl-tmp/MMRec-Ours/data \
  --output night_runs/preview --dry-run > rearm-cf-plan.json
tail -n 40 rearm-cf-plan.json
```

定向多 seed 复测支持 `--variants`，下面**仅为命令用法示例，并非已找到优胜配置**：

```bash
RUN="night_runs/baby-rearm-cf-confirm-$(date +%Y%m%d-%H%M%S)-$$"
nohup python -u experiments/run_rearm_cf.py \
  --data-path /root/autodl-tmp/MMRec-Ours/data --output "$RUN" \
  --variants reference ssm_base ultra_ii_0.1 cir_jc_0.5 --seeds 2025 2026 2027 \
  > "$RUN.log" 2>&1 &
tail -n 80 -f "$RUN.log"
```

返回根目录 `summary.csv`、`summary.json`、`manifest.json`、`status.json`。
若有失败，另发对应组 `console.log`；优胜组可一并发 `cf_diagnostics.jsonl`。
GraphDA 组再附该子目录的 `manifest.json`，用来核对重建图覆盖率。

## 文件导航与核验

- `rearm_cf.py`：损失、采样、类型分解和诊断。
- `rearm_cf_graphs.py`：UltraGCN omega、CIR、GraphDA 图。
- `rearm_cf_training.py`：训练循环、两阶段重置。
- `rearm_cf_plan.py`：72 个配置，含 14 组 QUICK 名单。
- `run_rearm_cf.py`：队列、恢复、汇总；`run_rearm.py --cf-config` 为单组入口。
- `tests/test_rearm_cf.py`：独立公式对照、原模型损失一致性、72 组有限梯度、教师参数/优化器
  重置、14 组端到端队列、恢复/归档和不保存 checkpoint。

来源：UltraGCN (arXiv:2110.15114, Eq.9/12/16/17)，CAGCN (WWW 2023,
DOI:10.1145/3543507.3583229, Eq.5/25)，GraphDA (SIGIR 2023, §3.2)，
NT-SSM (arXiv:2605.24015, §6.1 与 Appendix D)。均为用户提供的 PDF。
