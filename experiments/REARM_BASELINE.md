# 完整 REARM 官方基线

来源：[REARM 官方实现](https://github.com/MrShouxingMa/REARM)，固定提交
`3038f1ac3b7a957f140550bfecd34b09688a689b`。`third_party/rearm/UPSTREAM.json`
记录下载文件的 SHA256；这些官方文件逐字节保留，包括 `model.py`、helper、trainer 和 main。
本项目的适配在 `experiments/rearm_runtime.py` 和 `run_rearm.py` 中，未移植进 RelationRec。

## 这次运行什么

完全保留官方模型各模块：

1. 用户/物品 ID embedding；可训练的图像/文本原始特征表与用户兴趣表。
2. 用户共现+兴趣相似图、物品共现+内容相似图；两侧同构传播。
3. 物品模态 self/cross attention，用户模态 PReLU。
4. user-item 图传播，同时作用于 ID 和模态表示。
5. 低秩 meta 网络共享信息提取，最终拼接三个表示分支。

主目标为 **BPR + 跨模态对比损失 + 模态差异/正交约束**，正则使用 AdamW。
不使用上一轮的 SSM、WBPR、邻居约束 loss，也不使用 PE、group 或顺序初始化。
包括官方 `.detach()`、attention 的维度设定等细节都保留，暂不修订模型设计。

默认读取官方 README 中对应数据集的配置。Baby：64 维、4 层 UI、1 层 UU/II、rank=3、
lr=5e-5、AdamW decay=5e-4、batch=2048、seed=2025、最多 2000 epoch、
验证 Recall@20 早停（官方为无提升步数 >20）。只在验证改善时更新测试记录。
Sports/Clothing 自动切换到 README 的相应配置。这里未做超参搜索。

## 数据与官方协议差异

- 接受现有 `DATA_ROOT/baby/baby.inter`（userID,itemID,x_label=0/1/2），
  转成官方 [2,N] npy；保留编号、行序与划分，并沿用本仓库 cold-user 评估过滤。
  若没有 .inter 则读取官方 train/valid/test.npy。输出目录中保留转换后的数据。
- 图像/文本特征直接软链接输入文件，不修改或重编码。
- 检查 split 重复、交叉重叠、连续编号、特征行数、训练历史，异常直接失败。
- **默认 `--negative-scope official` 忠实保留官方 all-split 负采样行为**：
  排除 train+valid+test 的已知正例；官方 `dict_train_u_i` 原地更新还使
  无共现邻居时的用户流行度 fallback 使用所有 split 的计数。
  同构图共现计数、兴趣均值与 UI 图本身仍从训练边构建。
  此结果只能标成官方协议复现，不能称作严格 train-only 的无泄漏结果。
- `--negative-scope train` 同时修复上述负采样与 fallback 计数，保留完全相同模型。
  这属于独立协议对照，本次默认不自动额外运行。
- 官方精确结果还取决于数据版本和软件环境；manifest 记录输入哈希、配置、版本和源码哈希。
  在现有 MMRec 划分/现代 PyTorch 上不承诺逐位重现论文指标。

## 运行适配（不替换模型模块）

- 兼容新版 SciPy：用稀疏拼接替代已移除的 DOK `_update`，归一化仍调用官方函数。
- 共现计数用稀疏交集乘法分块构建，与集合交集计数一致，仍在完整行上用官方 torch.topk。
  top-200、softmax、邻居不足重复采样和 fallback 全部保留。
- 模态 KNN 分块计算相似度，保留自环、相似度权重和官方度归一化。
  不同设备的浮点舍入可能影响极接近的 top-k 边界；并非逐位一致保证。
- Python 3.11+ 不支持 random.sample(set)，候选集合转 tuple，均匀采一个负例不变。
- 修复官方评估迭代器以交互数终止、却按用户分批导致的空批次问题。
- 每次仅用当前输入重新构图，不读旧图缓存、不存图或模型检查点。
- 保留官方训练循环与验证选择方式，增加 result.json/summary.csv/status.json 和失败检查。

## 服务器运行

在 `/root/autodl-tmp/MMRec-Ours` 下，首次补齐官方依赖（使用当前已有 PyTorch 环境）：

```bash
python -m pip install -r experiments/requirements-rearm.txt > rearm-install.log 2>&1
tail -n 50 rearm-install.log
```

完整 Baby 基线，单个 seed；输出目录必须是新的：

```bash
nohup bash experiments/run_rearm.sh \
  /root/autodl-tmp/MMRec-Ours/data \
  0 night_runs/baby-rearm-official-1 baby \
  > baby-rearm-official-1.log 2>&1 &
tail -n 50 -f baby-rearm-official-1.log
```

`Ctrl+C` 只退出日志查看。没有实验队列时长/45 分钟截断；运行到官方早停或 epoch 上限。
不会自动停止其他已运行任务。不要对同一输出目录重复启动。

如需以后单独核对 train-only 协议，使用新的目录：

```bash
nohup python -u experiments/run_rearm.py \
  --data-path /root/autodl-tmp/MMRec-Ours/data --dataset baby --gpu-id 0 \
  --negative-scope train --output night_runs/baby-rearm-train-1 \
  > baby-rearm-train-1.log 2>&1 &
tail -n 50 -f baby-rearm-train-1.log
```

运行结束后提供：输出目录中的 `result.json`、`summary.csv`、`manifest.json`、`status.json`，
以及外部 `baby-rearm-official-1.log`。初始化构图时 status=preparing，开始训练后为 running，
结束为 complete；异常为 failed。日志应依次显示构图、模型、epoch loss、valid/test。

## 下一步替换顺序

先确认完整基线能在当前划分达到合理结果；然后固定协议与超参，逐项替换：
item 同构传播 → user 同构传播 → UI 传播。每次只替换一处，并保留完整基线对照。
在完整基线结果出来前，不再叠加新损失或启动大规模组合实验。

## 本地验证

已验证原始文件哈希、共现/KNN/UI 归一化与原式一致、小型数据完整两轮官方训练和评估、
最佳验证对应结果导出、无检查点和重复目录保护。未运行真实 Baby，性能等待服务器结果。
