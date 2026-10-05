# REARM：去 attention，把容量放到 MLP

## 首轮假设：一份历史基线 + 三组新实验

检验更大的模态投影网络能否替代 attention 的作用，同时保留 REARM 的图建模和损失。
默认只跑三个新配置，Baby，seed=2025；原始 REARM 复用上一轮结果。
下表第一行是历史参照，默认不会重新训练。

| 配置 | 四个 attention | Item 图文投影 | User 兴趣投影 | Meta |
|---|---|---|---|---|
| cap_original | 保留 | Linear(d_m,64) | 原始 Linear | 保留 |
| cap_no_attention | 移除 | Linear(d_m,64) | 原始 Linear | 保留 |
| cap_mlp_item256 | 保留 | d_m→256→64 | 原始 Linear | 保留 |
| cap_no_attention_mlp_item256 | 移除 | d_m→256→64 | 原始 Linear | 保留 |

已有原始 REARM：valid R@20=0.1040，test R@20=0.1094，test N@20=0.0473。
来源及完整配置保存在 `references/rearm_baby_s2025.json`。队列在 `manifest.json` 和
`summary.json` 的 `historical_references` 中单独记录历史观察，不加入本轮 jobs、
完成计数或验证排名。核对数据哈希、epoch 上限、worker 数、CPU/CUDA 模式，差异会
列入 `mismatches`；设置匹配也不代表不同软硬件下结果必然逐位一致。
当前只登记了 Baby/2025；其他数据集或 seed 没有历史记录时不会伪造或自动重跑基线。
确实需要重跑时显式加 `--include-reference`；指定 `--variants cap_original` 也可。

核心比较：

- 第一、二组：attention 是否必要。
- 第一、三组：仅增加投影容量是否有益。
- 第三、四组：MLP 增大后，attention 是否还有额外收益。
- 第一、四组：用户提出的整体替换方案是否有效。

本轮固定原 REARM 的 **BPR + 原 CL + 原正交损失**、AdamW、图结构、传播层数和
均匀负采样。不加 hard4/SSM/UltraGCN 等新因素。上游源码保持不变。

## 精确定义

两层投影：

```python
nn.Sequential(
    nn.Linear(actual_feature_dim, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
)
```

视觉输入为 4096 时就是 4096→256→64；文本按实际维度读取，不填充到 4096。
不新增 BatchNorm、Dropout 或可训练激活。两个 Linear 用 Xavier normal 初始化、bias=0。
默认只扩大 `image_i_trs`、`text_i_trs`，user 的图文兴趣投影保持单层。

去 attention 的方式：原 REARM 的 attention 调用返回零分支，残差输入继续经过原来的
LayerNorm/PReLU。四个 MultiheadAttention 实例被无参数模块替换，优化器也不再含它们。
这保留原来的处理位置、归一化和 192 维最终表示，避免同时改动整个融合流程。
因此它是“去掉 attention 分支”的对照，并非删除其周围的全部非线性。
attention 内部 dropout 同时消失，跨架构训练随机序列并不完全相同。

保留 attention 的所有模型直接调用原 REARM forward；没有复制一份易漂移的前传。
共同模块从相同 seed 初始化；新增 MLP 在独立 RNG 上下文内初始化，避免仅因多初始化
几层就改变之后的训练随机状态。不能据此认为不同架构在训练中有完全相同的随机路径。

## 参数量：不能把 attention 当作主要参数来源

原实现用的是四个 `MultiheadAttention(1,1)`，每个 8 参数，共 **32** 参数。
其注意力序列长度是 64 个特征坐标，主要复杂度来自中间注意力计算。
一个 4096→64 Linear 有 262,208 参数；4096→256→64 MLP 有 1,065,280 参数，
增加 **803,072**。本轮不是等参数预算比较，也不预先声称参数减少。

`manifest.json` / `result.json` 会分别统计：

- 可训练原始 item 模态表和 user 兴趣表；
- user/item ID embeddings；
- 四个投影网络；
- attention；
- meta 网络；
- 其他参数（如 LayerNorm/PReLU）。

`summary.csv` 给出总参数及各组参数数目，以及最佳验证轮次的训练秒数和 CUDA peak
allocated MB。每轮 `cf_diagnostics.jsonl` 也包含训练时间/峰值显存。
该显存值是训练段峰值的已分配显存，不是 reserved 显存或整个进程最大显存。
总运行时间会受提前停止轮次影响；不要用它单独声称每步计算更快。

## 可选扩展：七组新配置，加历史基线

`--suite capacity_extended` 包含上述三个新配置，再加：

| 配置 | 变化 |
|---|---|
| cap_mlp_all256 | 四个 item/user 模态投影都改为两层，保留 attention |
| cap_no_attention_mlp_all256 | 四个投影都增大，去 attention |
| cap_no_attention_no_meta | 去 attention 和低秩 meta，投影仍为单层 |
| cap_no_attention_no_meta_mlp_item256 | 去 attention/meta，增大 item 两个投影 |

去 meta 会真正删除四个低秩 MLP 及两个 meta 压缩层；meta 分支输出零变换，
保留该分支原有的 ID residual。因此预测仍是原来的 192 维拼接，并非减少为 128 维。
这让“去 meta”不同时改变 ID residual 的重复次数。图结构、CL/diff 继续保持。
这两组尚未触及参数量可能更大的可训练模态/用户兴趣表。

## 服务器运行：三组新实验

同步后先检查日志，成功再运行：

```bash
cd /root/autodl-tmp/MMRec-Ours
git pull --ff-only > rearm-capacity-pull.log 2>&1
tail -n 30 rearm-capacity-pull.log
```

```bash
RUN="night_runs/baby-rearm-capacity-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p night_runs
printf '%s\n' "$RUN" > night_runs/last-rearm-capacity-run.txt
nohup python -u experiments/run_rearm_cf.py \
  --data-path /root/autodl-tmp/MMRec-Ours/data --dataset baby --gpu-id 0 \
  --suite capacity --output "$RUN" > "$RUN.log" 2>&1 &
tail -n 80 -f "$RUN.log"
```

Ctrl+C 仅退出 tail。想一次跑七组，把 `--suite capacity` 改为 `--suite capacity_extended`。
不要同时运行基础和扩展队列重复占用同一 GPU。

新终端查看状态和日志：

```bash
RUN=$(cat night_runs/last-rearm-capacity-run.txt)
cat "$RUN/status.json"
tail -n 80 -f "$RUN.log"
```

恢复本版本启动后已停止的三组队列（成功项跳过，失败项归档后重跑）：

```bash
RUN=$(cat night_runs/last-rearm-capacity-run.txt)
nohup python -u experiments/run_rearm_cf.py \
  --data-path /root/autodl-tmp/MMRec-Ours/data --dataset baby --gpu-id 0 \
  --suite capacity --output "$RUN" --resume --retry-failed >> "$RUN.log" 2>&1 &
tail -n 80 -f "$RUN.log"
```

扩展队列恢复时须保持 `capacity_extended`。旧输出不要复用于新实验；代码哈希已变化。
若旧版四组队列已经启动，让它完成即可，不需要为了这次默认值调整中断重跑。
不保存模型检查点，未完成的单组从头重跑。仍按 validation Recall@20 选轮次，
最多 2000 epoch/组，沿用原 REARM 早停，测试不用于挑模型。

完成后发根目录 `summary.csv`、`summary.json`、`manifest.json`、`status.json`。
如要核对单轮计算开销，再带原模型和替换模型的 `cf_diagnostics.jsonl`。

## 本地验证

合成数据覆盖八组完整训练与评估，检查有限梯度、MLP 两层收到梯度、共同参数初始化一致、
无 attention 模型等价于将原 attention 输出置零、删除参数从优化器参数集合消失、输出维度
及参数账目一致。另运行原 REARM/旧 CF/关系分支回归检查，核对固定上游源码未变。
本地没有真实 Baby 数据及 CUDA，实际精度和 GPU 成本以服务器结果为准。
