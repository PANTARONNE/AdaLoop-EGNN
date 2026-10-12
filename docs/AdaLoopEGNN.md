# 单性质 AdaLoop-EGNN（M0–M3）

消息传播使用原有 `EGNNOrigStep`，完全图、固定坐标、共享参数。没有改动
`models/backbones/EGNNOrig.py`，原有 fixed / untied / tied 模型和 checkpoint 保持兼容。
本轮实现单性质模型，不包含 property embedding 或多性质联合训练。

**范围与配置**

| 配置 | 用途 |
|---|---|
| `qm9_loop_egnn_orig_diagnostic` | M0：共享参数、Kmax=12、所有深度等权 MAE 监督 |
| `qm9_adaloop_egnn_orig` | 单性质自适应模型，hidden_dim=128、Kmax=12 |
| `qm9_adaloop_egnn_orig_smoke` | 本地 PyG QM9 短程验收，hidden_dim=32、Kmax=4、3 epochs |

默认 `task.target=mu`、`task.normalization=mad`，归一化仅用训练标签拟合。
可通过 `task.normalization=standard` 或 `none` 切换；对照实验需统一该设置。
默认 `task.lambda_compute=0.03`、`model.execution.threshold=0.5`。
改变深度仍使用 `model.execution.depth`；自适应模型中该值表示 Kmax。

固定深度、Loop、M0 诊断和 AdaLoop 均直接使用 `dataset=qm9`、`dataset.root=data/qm9`。
已有 `processed/data_v3.pt` 可直接加载，无需另行准备 XYZ 或 atomref。配置名称中的 orig
表示消息传播骨架 EGNNOrig。默认划分为 110000 train / 10000 validation / remaining test，
`dataset.split_seed=42`；所有基线需使用相同划分和标签处理。
PyG 标签直接读取，不额外扣除原子参考值；zpve 为 eV，U0 是总能量目标。
`qm9_egnn_orig` 和 `qm9_loop_egnn_orig` 仍默认 final 监督。

**训练与停止语义**

Controller 输入每图的节点特征均值、总体标准差和相邻步特征变化范数均值；
不输入坐标、标签、Oracle 或人工复杂度指标。`model.halting.use_delta=false`
仅去掉 controller 的变化量输入，保留骨架的残差连接。

`q_k` 是条件停止概率，`p_k=q_k*prod(1-q_j, j<k)` 是停止质量。
最后一步强制停止并接收所有剩余质量。损失对每样本、每深度先计算 MAE，再用 p 加权：

```text
prediction_loss = mean_b sum_k p[b,k] * abs(y_k[b]-normalized_target[b])
compute_loss = mean_b sum_k p[b,k] * k/Kmax
loss = prediction_loss + lambda_compute*compute_loss
```

默认 1000 epochs 中前 200 epochs 为 warm-up，每 batch 均匀抽一个深度，只监督该深度。
后续完整展开到 Kmax，联合训练 controller。训练过程不声称节省消息传播计算。
`task.warmup_epochs` 是显式值；缩短训练需同时调整，恢复时不重新按比例计算。

推理选择第一个 `sum_{j<=k}p_j >= threshold` 的深度，输出该步预测。
已停止分子的节点和边从后续消息传播中移除；不是继续全 batch 计算后仅冻结特征。
未达到阈值的分子在 Kmax 强制退出。

| 模式 | 用途 | 计算 |
|---|---|---|
| `warmup` | 随机深度训练 | 执行到抽中的深度 |
| `full` | 自适应训练和 validation 损失 | 全部执行；同时按 CDF 选择 hard 输出计算 MAE |
| `diagnostic` | 离线轨迹分析 | 全部执行，提供逐深度预测和概率 |
| `hard` | 实际推理 | 批内压缩 active graphs，必须在 eval + no_grad/inference_mode 下运行 |

普通 `model.eval()` 推理默认 hard。完整 trace 必须显式指定 diagnostic；打开 trace 不会
悄悄改变计算协议。full/diagnostic 总是保留全部可微预测用于损失，不受 trace 深度选择影响；
trace 的 depth 选择只控制 detached 表示快照。hard 不伪造未执行的未来预测或完整停止分布。

**训练示例（以下完整训练命令未在本轮自动执行）**

在已激活项目环境的 WSL Bash 中：

```bash
python scripts/train.py experiment=qm9_loop_egnn_orig_diagnostic task.target=mu
python scripts/train.py experiment=qm9_adaloop_egnn_orig task.target=mu
python scripts/train.py experiment=qm9_adaloop_egnn_orig task.normalization=standard
```

直接读取本地 QM9 缓存的短程训练：

```bash
python scripts/train.py experiment=qm9_adaloop_egnn_orig_smoke trainer.device=cuda
```

M0 本轮仅验收代码，没有启动完整诊断实验。M1–M3 已用本地 PyG QM9 完成短程
链路验收；smoke 脚本与常规训练使用同一个正式入口。可复现该验收：

```bash
python scripts/smokeAdaLoop.py --data-root data/qm9 --device auto
```

该脚本要求已有 `processed/data_v3.pt`，不下载数据。固定使用 mu、MAD、128/32/32 个真实
分子、3 epochs（1 warm-up + 2 adaptive），记录数据来源和样本划分；用 full 与 hard
预测一致性及实际 node/edge step 计数验证执行链路。可传 `--normalization standard`。
脚本调用常规 `train()`，随后从 best.pt 重建模型、Task 和数据集进行 full/hard 对照。

**Checkpoint 与日志**

`last.pt` 每 epoch 保存 model/controller、optimizer、scheduler、task 阶段、transform 和 RNG。
warm-up 期间不选择 best.pt；此时允许从只有 last.pt 的目录恢复。进入 adaptive 阶段后，
按 CDF hard 输出的 validation MAE 选择 best.pt。训练恢复沿用原入口：

```bash
python scripts/train.py experiment=qm9_adaloop_egnn_orig \
  resume=outputs/<run>/checkpoints/last.pt
```

保持数据、模型、task（含 warm-up、lambda）、优化器及 cosine 周期配置一致。
本轮不实现从旧 Loop 权重迁移初始化 AdaLoop；resume 不是跨模型迁移。

metrics.csv 新增 prediction/compute loss、expected_depth、halt_depth、executed_depth、
首步/末步停止比例、分布熵和 warm-up 标记。按样本数加权；不适用统计留空，表头不变。
expected_depth 是 soft 期望；halt_depth 是选择结果；executed_depth 是实际执行长度。
validation 使用完整轨迹，因此其 executed_depth=Kmax，不能当作部署执行成本。

**预测、Oracle 与结构指标**

```bash
python scripts/predict.py --checkpoint outputs/<run>/checkpoints/best.pt --split test
python scripts/predict.py --checkpoint outputs/<run>/checkpoints/best.pt \
  --split validation --mode diagnostic
```

自适应默认输出 `test_best_adaptive.csv`，一行对应一个分子，含 hard 预测与停止/执行深度。
diagnostic 输出 `validation_best_diagnostic_by_depth.csv`，含 y_k、q、p、survival、soft E[K]。
所有文件保存 checkpoint 哈希、split ID、lambda、CDF 阈值和执行模式。
旧 fixed checkpoint 仍输出原 `*_by_depth.csv`；`evalDepth.py` 仅汇总 fixed 模型。

M0 固定轨迹或 adaptive diagnostic 轨迹均可离线计算 Oracle：

```bash
python -m eqcompute.analysis.oracleDepth \
  outputs/<run>/predictions/validation_best_by_depth.csv \
  --checkpoint outputs/<run>/checkpoints/best.pt \
  --lambda-oracle 0.03 --output outputs/<run>/analysis/oracle.csv

python -m eqcompute.analysis.complexity \
  --checkpoint outputs/<run>/checkpoints/best.pt --split validation \
  --output outputs/<run>/analysis/complexity.csv
```

Oracle 按 checkpoint 的训练集 scale 缩放绝对误差，输出无成本与带 k/Kmax 成本的两种
argmin；平局取最浅深度，另保存 normalization、scale 和 lambda_oracle。错误列为归一化
绝对误差，unit 列标记原目标单位。lambda_oracle 与训练 lambda_compute 相互独立。
检查哈希、性质、单位、深度覆盖和重复行；存在 split_ids.json 时还检查整套样本覆盖。
Oracle 使用标签，仅供分析，不能用于推理或 test 超参选择。

结构指标包括总原子数、重原子数、等权几何中心回转半径（坐标单位 Å）、元素种类数和
自然对数元素熵。暂不包含化学图直径、环数和可旋转键。Orig 完全图不能用于解释
化学图距离或局部传播范围。

**验收**

```bash
python -m pytest -q
python -m ruff check src tests scripts
```

测试覆盖概率/手算损失、梯度、controller 注册、共享参数、对称性、单节点图、CDF 停止、
CPU/CUDA 非连续图压缩、实际算子工作量、阶段切换、断点恢复、配置归一化切换、
预测/分析 provenance，以及原有模型和固定深度回归。结果见 [验收记录](../VALIDATION.md)。
