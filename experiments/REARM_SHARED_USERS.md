# 优先实验：让共享历史的用户持续共享表示

本轮优先级高于 preferences 容量队列。默认只跑 4 组，扩展 7 组。
没有启动或停止服务器上的任何进程；本地只有合成数据验证，真实性能由服务器队列评估。

## 机制与边界

仅使用 TRAIN 的二值交互 R，不按用户 ID 距离构图或训练，不使用 PE、编号分组或近 ID
监督。原 REARM 的历史均值随后成为独立参数表；本实验在每次前传都重新从当前物品基底
生成用户输入，保证相关用户持续共享底层参数：

    a_ui = R_ui * (degree_i + 1)^(-beta)
    A_ui = a_ui / sum_j a_uj
    shared_id_u = sum_i A_ui * item_id_i
    shared_modal_u = sum_i A_ui * user_projector_m(item_feature_i^m)

user_projector 仍是 REARM 原来的 Linear；物品模态表仍可训练。beta 默认 0，
完整保留训练历史，没有 UU Top-K 截断、构造稠密 UU 矩阵或新增多跳扩散。
这是一次 item-to-user 聚合，不能宣称完全无消息传递或本身就是新颖方法。

用户残差从零开始，使用相对范数上界：

    cap_u = gamma * sqrt(n_u / (n_u + c))
    e_u = shared_u + cap_u * stopgrad(norm(shared_u)) * delta_u / (1 + norm(delta_u))

其中 gamma 为 0 / 0.1 / 0.3，c 为 0 或 5。这个约束作用于送入 REARM 后续图传播前的
用户输入：残差范数不超过共享向量范数的 cap_u 倍。不是对最终预测表示的距离保证。
无历史用户得到零聚合；本轮不声称解决用户冷启动。

相同历史、无残差的用户，在本模块输出处一定相同；不同的后续 UU 邻域仍可能使最终表示
不同。近 ID 用户只有在 TRAIN 关系相似时才可能因此靠近，不能保证恢复编号的隐藏信息。
共享一次物品也不意味着用户整个偏好应该相同。

## 对照

| 配置 | 用户模态输入 | 用户 ID 输入 | 残差 |
|---|---|---|---|
| shared_modal | 当前物品模态聚合 | 原自由 ID embedding | 无模态残差 |
| shared_all | 当前物品模态聚合 | 当前物品 ID 聚合 | 无 |
| shared_all_res01 | 同上 | 同上 | 相对范数上限 10% |
| shared_all_res01_degree | 同上 | 同上 | 上限 10%，稀疏用户进一步收缩 |

默认保持 REARM 的 attention/meta/PReLU、II/UU/UI、BPR+CL+diff 和其他超参数，
本轮不同时做 MLP 扩容或多兴趣，避免无法解释效果来源。

扩展 shared_users_extended 追加：
- shared_all_res03：30% 残差上限；
- shared_all_res01_no_uu：取消后续 UU 传播，其他保持；
- shared_all_res01_pop05：beta=0.5，降低热门物品权重。

历史原 REARM 继续复用，不自动重跑。没有保存过基线 embedding，
因此不能声称有“训练后原 REARM 距离”的同批对照；已有基线用于精度参考。
要补这一对照需明确运行并记录基线几何指标，当前不默认增加这笔训练成本。

## 诊断与模型选择分开

cf_diagnostics.jsonl / result.json 的 user_geometry 包含：
- 模态用户输入与最终 192 维用户表示两个阶段；
- ID 偏移 1/4/16/64 的用户对；
- 同一 anchor、目标用户 TRAIN 度完全相同且 ID 距离大于 256 的远端对照；
- 随机共同物品用户对（不以 ID 距离过滤）；
- 全局随机用户对；
- 平均余弦、欧氏距离平方、表示范数与各维平均方差。

使用固定本地 NumPy 随机数生成器，缺少精确对照时跳过并记录数量，无近似回退。
小型合成数据可能没有足够远的用户，此时相关指标为 null，而非 0。
这些统计 detached，不影响训练 RNG，不参与 loss，不用于给超参数或图排序。
summary.csv 展示主要几何统计；模型排序仍只按 valid Recall@20。

判断有效性必须同时看：
1. valid 推荐性能是否改善；
2. 共享历史用户是否比随机用户更接近；
3. 近 ID 对相对于匹配对照的差异是否保留/扩大（仅作解释）；
4. 全局随机用户是否也全部接近、表示方差是否明显塌缩。

不能只追求 near-ID 平均余弦更高。ID 距离上的事后诊断不应反过来变成调参标签。

## 服务器运行和日志

代码同步成功后：

```bash
cd /root/autodl-tmp/MMRec-Ours
git pull --ff-only > rearm-shared-pull.log 2>&1
tail -n 40 rearm-shared-pull.log
```

先运行默认 4 组，不启动之前的 12/15 组容量队列：

```bash
RUN="night_runs/baby-rearm-shared-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p night_runs
printf '%s\n' "$RUN" > night_runs/last-rearm-shared-run.txt
nohup python -u experiments/run_rearm_cf.py \
  --data-path /root/autodl-tmp/MMRec-Ours/data --dataset baby --gpu-id 0 \
  --suite shared_users --output "$RUN" > "$RUN.log" 2>&1 &
tail -n 80 -f "$RUN.log"
```

新终端查看：

```bash
RUN=$(cat night_runs/last-rearm-shared-run.txt)
cat "$RUN/status.json"
tail -n 80 -f "$RUN.log"
```

如需一次跑 7 组，用 shared_users_extended，勿同时运行两套重复队列。
按验证选 epoch，不保存模型。旧输出不能用于新代码；中断恢复必须保持本次版本与参数。

运行完成发 summary.csv、summary.json、manifest.json、status.json；
为核对完整几何统计，再附每组 result.json 或 cf_diagnostics.jsonl。

## 本地检查

稀疏聚合与稠密公式一致；同时重编号 user/item 后，聚合结果只发生对应行置换；
相同历史输出相同；物品表示更新会同步影响消费者；梯度确实流入消费过的物品基底；
残差范数上界成立；移除全部 ID 距离诊断不改变预测；诊断不消耗训练 RNG；
训练 BPR 与全量预测分数一致；7 组完整合成数据训练评估；旧模型回归检查。

