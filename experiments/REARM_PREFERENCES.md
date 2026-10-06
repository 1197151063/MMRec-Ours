# REARM：把参数数量与有效容量分开检验

## 判断依据

不能由 98% 参数在模态表就推断它们贡献了 98% 精度。原用户偏好输出为
`P_u @ W.T + b`，其中 P 为 N×d_m，W 为 64×d_m；自由学习 N×64 表可覆盖
这一层原本可生成的全部输出。当 W 满行秩时，原参数化也可生成任意 N×64 输出。
因此高维 P 增大的是参数化规模，并不自动增加后续网络看到的自由维度。
初始化、梯度预条件、权重衰减和优化路径仍然不同，预测等价不等于训练等价。

原始每个实体最终得到一个 192 维表示，点积分数矩阵的秩至多 192。
给用户增加多个向量再直接相加仍可合并成一个向量；逐分支点积后求和则等价于
拼接向量的点积，必须如实报告总表示维度。增加 ID 维度本身并非违规，新的偏好表
同样是用户特定参数，需要相同数据协议、参数/成本报告和对照，不能用命名替代比较。

## 新参数化

1. **projected**：用原投影结果初始化直接学习的 64 维偏好表，用户投影换 Identity。
   同 seed 下初始前传与原 REARM 相同；这是高维表是否必要的对照。
2. **residual**：历史均值成为固定 buffer，保留可训练投影，新增零初始化 64 维残差：
   `p_u^m = encoder_m(history_u^m) + delta_u^m`。
   初始前传也与原模型相同；历史始终作为输入，而非只在初始化出现一次。
   固定原始历史仍占 buffer 内存，减少的是可训练参数和优化器状态，并非所有内存。
3. **clean**：采用 residual，去四个 attention、去 meta 变换，主 PReLU 换 GELU。
   保留原残差路径的共享 LayerNorm、UU/II/UI、192 维输出与第三分支 ID 残差。
   item 投影可为 Linear 或 `input→256/512→ReLU→64`；user 历史投影保持 Linear。
   这是几项改动的组合候选，不可把其单次性能差归因于某一个模块。

不改普通 ID 表的标准 normal 初始化；没有人为排序、分组或 PE。
上游 pinned REARM 文件不改。clean 在前传中移除了 attention、meta、可学习 PReLU；
兼容构造阶段仍先创建原模型，再删除相关模块，以保持公共参数初始化一致。

## 真正不同的打分能力：候选相关的多兴趣

保持原 REARM 分数 s0，额外对每个模态 m 引入 K 个用户 64 维可学习 code。
设图传播后的模态向量为 h：

```
q[u,m,k] = normalize(normalize(h[u,m]) + code[u,m,k])
z[i,m]   = normalize(h[i,m])
a[u,i,m] = tau * (logsumexp_k(dot(q[u,m,k], z[i,m]) / tau) - log(K))
score[u,i] = s0[u,i] + weight * (a[u,i,v] + a[u,i,t]) / 2
```

这是候选相关的平滑多兴趣匹配，不是把 K 个偏好向量先平均，也不是声称复现 MIND。
K=1 是必需对照：与 K>1 比较才能区分额外语义分支和多兴趣选择的作用。
减 log(K) 使重复相同头不会凭空抬高分数；不同 K 的分数分布仍可能不同。
code 用独立 normal(std=0.1) 初始化打破对称；共同头在不同 K 下初始化一致。
普通初始化不会被包装成创新。主 loss 始终是完整新分数上的 BPR，原 CL/diff 保留。
其余 SSM/困难负采样/UltraGCN 本轮不混用，便于解释容量效果。

每个模态/用户 K 个 code，新增 `2*N_users*K*64` 参数，K=4 时 Baby 为 9,955,840。
推理不再是单次点积检索，需计算多个头并做 logsumexp；全量评估沿 item 分块以控制中间
显存，但总计算仍随 K 增加。当前分块默认 512，可在 config 的 interest_chunk 调整。
cf_diagnostics.jsonl 记录头间平均余弦；接近 1 表示兴趣高度相似，不能据此宣称学到多样性。

多兴趣本身已有工作，例如 [MIND](https://arxiv.org/abs/1904.08030)。本轮为性能假设验证，
不声称“未被尝试过”。用户平均历史 + 残差也需要进一步文献比较才能确定论文定位。

## 默认 12 组：历史基线不重新训练

| 配置 | 目的 |
|---|---|
| pref_freeze_users | 只冻结原始用户偏好表，投影继续训练 |
| pref_freeze_items | 只冻结原始物品模态表 |
| pref_projected64 | 同初始前传，直接学习 64 维用户模态表 |
| pref_history_residual64 | 同初始前传，固定历史 + 64 维残差 |
| pref_clean_linear | 去 attention/meta，GELU，Linear 投影 |
| pref_clean_mlp256 | clean + item 两层 256 隐藏维投影 |
| pref_clean_mlp512 | clean + item 两层 512 隐藏维投影 |
| pref_interests1 | MLP256 + 每模态 1 个兴趣头，weight=1 |
| pref_interests2 | 同上，2 个兴趣头 |
| pref_interests4 | 同上，4 个兴趣头 |
| pref_interests8 | 同上，8 个兴趣头 |
| pref_interests4_w01 | 4 个兴趣头，weight=0.1 |

`preferences_extended` 增加 3 组：同时冻结两类表、residual 上仅去 attention、clean 上 GELU
换 ReLU。默认保留已有训练超参数而未逐模型调优；首次筛选不能把失败解释为该模型无潜力，
也不能从单 seed 得出显著提升。最终候选需要学习率/正则与多 seed 验证，以及同预算更大
ID embedding 对照；本轮没有声称完成公平预算比较。

主要解释路径：freeze_users 对历史 REARM 判断用户自由适配是否有用；projected64 对 REARM
判断高维参数化是否必要；history_residual64 对 projected64 判断持续的语义路径；clean 的
256 对 512 判断共享非线性容量；heads1 对 heads2/4/8 判断候选相关多兴趣容量。

默认复用历史 Baby/2025 基线，沿用数据和运行设置核验；没有匹配历史的配置不自动补跑。
固定 TRAIN-only，按 valid Recall@20 选择 epoch，测试不参与排序，不保存模型。
summary.csv 同时报告注册总参数和 requires_grad 参数数目。requires_grad 不代表一定收到
梯度：原 REARM 的 meta 压缩层输出 detach，仍保留原统计语义。

## 服务器启动与查看日志

同步代码：

```bash
cd /root/autodl-tmp/MMRec-Ours
git pull --ff-only > rearm-preferences-pull.log 2>&1
tail -n 30 rearm-preferences-pull.log
```

默认跑 12 组，用唯一目录避免与旧队列冲突：

```bash
RUN="night_runs/baby-rearm-preferences-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p night_runs
printf '%s\n' "$RUN" > night_runs/last-rearm-preferences-run.txt
nohup python -u experiments/run_rearm_cf.py \
  --data-path /root/autodl-tmp/MMRec-Ours/data --dataset baby --gpu-id 0 \
  --suite preferences --output "$RUN" > "$RUN.log" 2>&1 &
tail -n 80 -f "$RUN.log"
```

要跑 15 组，将 suite 改为 preferences_extended；不要同时启动两套队列。
新终端查看进度：

```bash
RUN=$(cat night_runs/last-rearm-preferences-run.txt)
cat "$RUN/status.json"
tail -n 80 -f "$RUN.log"
```

仅对已停止且代码/配置相同的本轮队列恢复；成功项跳过，失败项归档并从头训练：

```bash
RUN=$(cat night_runs/last-rearm-preferences-run.txt)
nohup python -u experiments/run_rearm_cf.py \
  --data-path /root/autodl-tmp/MMRec-Ours/data --dataset baby --gpu-id 0 \
  --suite preferences --output "$RUN" --resume --retry-failed >> "$RUN.log" 2>&1 &
tail -n 80 -f "$RUN.log"
```

扩展队列恢复时保持 preferences_extended。完成后提供 summary.csv、summary.json、
manifest.json、status.json；多兴趣模型再附 cf_diagnostics.jsonl。

## 验证范围

本地仅运行小型合成数据：初始前传等价、参数/冻结/梯度检查、兴趣头重复不改变分数、
配对 BPR 与分块全量预测一致、完整队列训练评估，以及旧 CF/容量实验回归。
本地没有真实 Baby 数据或 CUDA，因此没有宣称推荐精度提升。

