# AdaLoop-EGNN

在 QM9 单目标回归上比较固定深度 EGNN、共享参数 Loop-EGNN，以及学习停止循环次数的 AdaLoop-EGNN。
当前 AdaLoop 使用两个独立训练阶段：先按 batch 随机深度训练预测器，再冻结预测器、只训练停止器。

所有配置使用本地 PyG QM9 数据协议。默认训练入口仍为固定深度 `qm9_egnn`；
AdaLoop 需要显式选择 `qm9_adaloop_egnn_predictor` 和 `qm9_adaloop_egnn_halting`。
详细模型与损失定义见 [EGNN 说明](docs/EGNN.md) 和 [AdaLoop 说明](docs/AdaLoopEGNN.md)，
验证结果见 [VALIDATION.md](VALIDATION.md)。

## 环境与数据

从仓库根目录运行命令。已有环境直接激活：

```bash
conda activate AdaLoopEGNN
```

重建环境：

```bash
conda env create -f environment.yml
conda activate AdaLoopEGNN
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[dev]'
```

已验证环境为 Python 3.11、PyTorch 2.3.1+cu121、PyG 2.7.0、NumPy 1.26.4。
[requirements-wsl.lock.txt](requirements-wsl.lock.txt) 保存已记录的环境依赖版本。
训练设备默认为 `auto`，可通过 `trainer.device=cpu` 或 `trainer.device=cuda:0` 指定。

默认数据目录为 `data/qm9`，现有 `processed/data_v3.pt` 可直接使用。
常规训练在缺少数据时由 PyG 下载、准备 QM9；独立两阶段 smoke 脚本要求已有缓存。
覆盖数据位置使用 `dataset.root=/data/qm9`，smoke 脚本使用 `--data-root /data/qm9`。
数据路径会在保存配置中转为绝对路径，后续预测和分析需能访问该位置。

数据与目标处理约定：

- 默认划分为 110000 train / 10000 validation / 剩余 test，`dataset.split_seed=42`；训练 `seed` 独立设置。
- 样本 ID 使用原始 QM9 idx，完整划分写入 `split_ids.json`。
- 默认 `task.target=mu`，`task.normalization=mad`。MAD 使用训练集均值和平均绝对偏差；
  也支持 `standard` 和 `none`。归一化仅拟合训练标签，MAE/RMSE 和预测 CSV 使用原属性单位。
- 直接使用 PyG 标签，不额外扣除原子参考能，也不重复转换 Hartree。
  `U0`（列 7）和 `U0_atom`（列 12）是不同目标。
- 适配器保留 19 个目标；默认学习率表覆盖 `mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv`。
  选择其他目标时需显式设置 `optimizer.learning_rate`。

## 训练配置

配置位于 [configs/experiment](configs/experiment)。`K` 表示执行深度，AdaLoop 中表示最大循环次数。

| `experiment` | 训练内容 | 默认 K | 默认轮数 |
|---|---|---:|---:|
| `qm9_egnn` | 各层独立参数，末层监督；默认入口 | 7 | 1000 |
| `qm9_loop_egnn` | 共享 EGNN 参数，末层监督 | 7 | 1000 |
| `qm9_loop_egnn_diagnostic` | 共享参数，所有深度等权监督 | 12 | 1000 |
| `qm9_adaloop_egnn_predictor` | 冻结停止器，随机深度训练预测器 | 24 | 1000 |
| `qm9_adaloop_egnn_halting` | 冻结预测器，仅训练停止器 | 24 | 500 |
| `qm9_smoke` | 固定深度基线短程训练 | 3 | 3 |
| `qm9_adaloop_egnn_smoke` | 仅预测器阶段短程训练 | 4 | 3 |

完整配置默认 hidden_dim=128、batch size=96、Adam、MAE loss 和每 epoch 一次 cosine 调度。
`qm9_smoke` 使用 hidden_dim=32、batch size=32；AdaLoop smoke 使用 hidden_dim=32、batch size=16。
两种 smoke 均取 192 个样本，划分为 128/32/32。短程误差用于检查流程。

### AdaLoop 两阶段训练

先训练预测器：

```bash
python scripts/train.py experiment=qm9_adaloop_egnn_predictor task.target=mu
```

从日志中的 `output=...` 获取第一阶段运行目录，替换以下 `PREDICTOR_RUN` 后训练停止器：

```bash
python scripts/train.py experiment=qm9_adaloop_egnn_halting task.target=mu \
  init_from=outputs/PREDICTOR_RUN/checkpoints/best.pt
```

两个阶段分别生成运行目录，不会根据 epoch 自动切换。最大深度、模型、目标属性、归一化设置和数据划分需匹配；
若第一阶段覆盖了这些参数，第二阶段需使用相同设置。

第一阶段训练 initializer、共享 EGNN、readout 和预测 head，停止器被冻结且不参与前向计算。
每个 batch 均匀采样一个深度 `d ∈ [1, K]`，整个 batch 使用该深度的归一化预测误差训练。
验证计算全部深度，不调用停止器：

| 第一阶段验证字段 | 含义 |
|---|---|
| `validation_mae` | 每个分子选取绝对误差最小的深度，再对分子平均；用于选择 `best.pt` |
| `validation_mean_depth_mae` | 所有分子、所有深度的平均绝对误差 |
| `validation_final_depth_mae` | 全部使用第 K 层预测的 MAE |
| `validation_oracle_depth` | 每个分子最优深度的平均值，并列时选择较浅层 |
| `validation_loss` | 所有深度归一化误差的平均值，无熵项 |

第一阶段的 `validation_mae` 是 **oracle MAE**，使用真实标签选层，是理想选层的误差下限。

第二阶段固定整个预测器及其状态，只更新停止器，训练完整展开到 K。
`qm9_adaloop_egnn_halting` 默认使用 `task.halting_loss=soft_cdf`，
以平滑 CDF 跨越推理阈值的位置生成权重 `w`，再加权每层预测误差：

```text
F[k] = sum_{j<=k} p[j]
S[0] = 0; S[K] = 1
S[k] = sigmoid((F[k] - threshold) / temperature)  # 1 <= k < K
w[k] = S[k] - S[k-1]
loss = mean_i sum_k w[i,k] * abs(z_pred[i,k] - z_target[i])
       - entropy_beta * mean_i H(p[i])
```

`z` 为归一化目标或预测，`p` 为原停止概率质量；熵仍对 `p` 计算，不对 `w` 计算。
`task.soft_cdf_temperature` 默认 `0.1`，是固定的正温度，不自动退火。
`threshold` 复用 `model.execution.threshold`。`task.entropy_beta` 保持默认 `0.01`，可设为 `0`。
若要对照原来的概率期望损失，在停止器训练命令中追加 `task.halting_loss=expected`，此时权重为 `p`。
两种模式都先计算各层误差再加权，不通过混合预测抵消误差。

`prediction_loss` 记录所选训练目标，`expected_prediction_loss` 始终记录原 `p` 下的归一化期望误差。
验证每个分子取停止概率 CDF 首次达到 `model.execution.threshold=0.5` 的深度，
将该层预测还原到原单位后统计 `validation_mae`，并据此选模。
该指标不含熵项。两阶段 `validation_mae` 的选层方式不同，比较结果时需注明阶段。

`init_from` 可指定第一阶段的 `best.pt` 或 `last.pt`，只复制预测器权重和归一化参数；
停止器重新初始化，优化器、cosine 周期、epoch 和历史最佳指标均重新开始。
来源路径和 SHA256 记录在 metadata 中。停止器阶段必须提供 `init_from` 或同阶段 `resume`。
Checkpoint 保存损失类型与温度，恢复时须保持一致。缺少新字段的旧 checkpoint 按 `expected` 读取，
不能通过 `resume` 将旧停止器训练切换成 soft-CDF；可以用原 predictor checkpoint 初始化新的 soft-CDF 运行。

### 基线与逐深度诊断

```bash
python scripts/train.py experiment=qm9_egnn task.target=mu
python scripts/train.py experiment=qm9_loop_egnn task.target=gap model.execution.depth=8
python scripts/train.py experiment=qm9_loop_egnn_diagnostic task.target=mu
```

`task.supervision=final` 只监督最终深度；`all_steps` 等权监督所有深度，共用预测 head。
基线两种监督方式都按最终深度的 validation MAE 选模。
`final` 模型的中间层未获直接监督，逐层结果应注明训练协议。

### 短程验收

基线短程训练：

```bash
python scripts/train.py experiment=qm9_smoke
```

AdaLoop 完整两阶段链路使用独立脚本：

```bash
python scripts/smokeAdaLoop.py --data-root data/qm9 --device auto
```

脚本调用正式 `train()`，分别训练预测器 1 轮、停止器 2 轮，
检查预测器状态完全不变、完整展开与实际提前停止的预测和停止深度一致，
并导出测试集预测及 `smoke_summary.json`。可用 `--normalization standard` 或 `none` 切换归一化，
用 `--output-root` 指定输出位置。直接运行 `experiment=qm9_adaloop_egnn_smoke` 只覆盖预测器阶段。

## 输出与恢复训练

每次训练在 `output_root`（默认 `outputs`）下创建唯一目录：

```text
outputs/EXPERIMENT_NAME__UTC_TIMESTAMP_RUN_ID/
  config.yaml
  metadata.json
  split_ids.json
  checkpoints/best.pt
  checkpoints/last.pt
  metrics.csv
  predictions/
  logs/train.log
```

AdaLoop 目录名包含目标、阶段、最大深度和 seed；固定深度目录名包含目标、参数共享设置、深度、监督方式和 seed。
两个阶段均从第一轮起保存最佳模型。`metrics.csv` 按样本数累计指标，避免小尾批次获得过高权重。

同阶段恢复使用 `last.pt`，并重复原运行的配置覆盖参数：

```bash
python scripts/train.py experiment=qm9_adaloop_egnn_predictor task.target=mu \
  resume=outputs/PREDICTOR_RUN/checkpoints/last.pt
python scripts/train.py experiment=qm9_adaloop_egnn_halting task.target=mu \
  resume=outputs/HALTING_RUN/checkpoints/last.pt
```

`resume` 与 `init_from` 互斥；恢复停止器时不需要再访问第一阶段文件。
恢复会创建新目录，载入模型、优化器、scheduler、目标变换和 Python/NumPy/Torch/CUDA、DataLoader RNG，
保留已有最佳 checkpoint；新 `metrics.csv` 只记录新增轮次。支持 epoch 边界恢复。

模型、task、数据、优化器、loader、seed、梯度裁剪和 scheduler 配置需与原运行一致。
默认 `trainer.scheduler.t_max=${trainer.epochs}`，因此直接增加 `trainer.epochs` 会同时改变 cosine 周期并被拒绝。
若要续跑至更大的轮数，必须保持原 `t_max`，且总轮数不能超过该周期；
正式训练中断后恢复通常只需使用原来的轮数设置。
旧自动 warm-up / 联合训练配置及其 adaptive checkpoint 已不再支持。

## 预测导出

将 `FIXED_RUN` 和 `HALTING_RUN` 替换为对应训练目录。

固定深度模型一次展开导出所有深度：

```bash
python scripts/predict.py --checkpoint outputs/FIXED_RUN/checkpoints/best.pt --split test
```

输出 `predictions/test_best_by_depth.csv`，每行对应样本、性质与读出深度。
包含真实值、预测值、单位、checkpoint SHA256、split ID、seed 和训练协议。
可用已有逐层 CSV 绘制误差曲线：

```bash
python -m eqcompute.analysis.depthCurves \
  outputs/FIXED_RUN/predictions/test_best_by_depth.csv \
  --output outputs/FIXED_RUN/predictions/depth_curve.png
```

AdaLoop 的停止推理使用**停止器阶段 checkpoint**：

```bash
python scripts/predict.py --checkpoint outputs/HALTING_RUN/checkpoints/best.pt --split test
python scripts/predict.py --checkpoint outputs/HALTING_RUN/checkpoints/best.pt \
  --split validation --mode diagnostic
```

默认 `hard` 模式实际提前退出，已停止分子的节点和边从后续计算中移除，
输出 `test_best_adaptive.csv`。`diagnostic` 完整展开所有深度，
输出 `validation_best_diagnostic_by_depth.csv`，含逐层预测、停止概率及深度统计。
预测器阶段的停止器尚未训练，其停止结果不作为推理性能报告。

预测支持 `--device auto|cpu|cuda:0`、`--batch-size` 和 `--overwrite`；已有输出默认拒绝覆盖。
测试集用于最终报告，模型和超参数选择依据验证集。

## 深度扫描与汇总

固定深度配置可通过 Hydra 串行扫描多个组合：

```bash
python scripts/train.py -m experiment=qm9_egnn \
  'model.execution.depth=1,2,3,4,6,8,10,12' \
  'model.weight_tying=false,true' 'seed=0,1,2'
```

单机多 GPU 调度使用 `sweepDepth.py`。每个深度 × seed 独立训练，每张 GPU 同时运行一个任务：

```bash
python scripts/sweepDepth.py --gpus 0,1 --depths 2,4,8,12 --seeds 0,1,2 \
  --weight-tying true -- experiment=qm9_loop_egnn task.target=mu
```

`--weight-tying true/false` 必填。其他设置通过 `--` 后的 Hydra 参数传入；
深度、seed、参数共享设置、设备、resume 和 Hydra 扫描参数由调度器管理。
追加 `--dry-run`（放在 `--` 之前）可只检查配置与任务列表，不启动训练或准备数据。
GPU 编号按当前 CUDA 可见设备顺序解释，子进程使用分配到的单张 GPU。
调度前串行准备一次数据，再启动训练；CPU 数据加载开销随并发数增加。

输出结构为：

```text
outputs/_depth_sweeps/BATCH_ID/
  jobs.json
  logs/kDEPTH_sSEED.log
  runs/kDEPTH_sSEED/TRAINING_RUN/
```

`jobs.json` 记录参数、状态、退出码、日志和运行目录。失败后继续其他任务，失败任务可缩小深度/seed 列表重跑；
中断会终止本批次子进程。调度器不自动续训，单任务恢复使用 `train.py resume=...`。

汇总脚本只支持固定深度模型，通过 `sweepDepth.py` 生成的 `jobs.json` 查找各运行：

```bash
python scripts/evalDepth.py --sweep-dir outputs/_depth_sweeps/BATCH_ID --device auto
```

按深度输出各 seed 的 validation/test MAE 均值和样本标准差，test 使用最终训练深度的预测。
有效测试预测缓存会复用；缺失时重新评估。失败或不完整任务会说明并跳过，有跳过任务时返回非零退出码。
该汇总入口不处理 AdaLoop 的 oracle/CDF 指标。

## 分子深度分析与绘图

对停止器阶段运行目录生成逐层 diagnostic 预测和分析；默认使用 validation，可用 `--split test` 切换：

```bash
python scripts/depthComplexity.py --run-dir outputs/HALTING_RUN
python scripts/depthWorsening.py --run-dir outputs/HALTING_RUN
```

两者需要 `checkpoints/best.pt` 和保存的数据划分，支持 `--device`、`--batch-size`、`--overwrite`。
默认复用有效缓存，补齐缺失产物；来源不符或内容不完整时报错。

| 输出目录 | 内容 |
|---|---|
| `analysis/depth_complexity/validation/` | `complexity.csv`、`oracle.csv`、`depth_complexity.csv`、`summary.json` |
| `analysis/depth_worsening/validation/` | `depth_worsening.csv`、`summary.json` |

复杂度包括总原子数、重原子数、等权回转半径、元素种类数和元素熵。
`oracle_depth` 按逐分子最小预测误差选择，平局取较浅层；`halt_depth` 按 CDF 阈值选择。
Oracle 使用真实标签，仅作为理想选层参照。相邻深度分析统计所有分子从 k 到 k+1 的绝对误差严格增加比例，
持平不计入变差，分母包括所有分子。

绘图只读取分析 CSV，不重新运行模型：

```bash
python plot/depthComplexity.py \
  --input outputs/HALTING_RUN/analysis/depth_complexity/validation/depth_complexity.csv \
  --complexity all --output-dir outputs/HALTING_RUN/analysis/depth_complexity/validation/figures
python plot/depthDistribution.py \
  --input outputs/HALTING_RUN/analysis/depth_complexity/validation/depth_complexity.csv \
  --output-dir outputs/HALTING_RUN/analysis/depth_complexity/validation/figures
python plot/depthWorsening.py \
  --input outputs/HALTING_RUN/analysis/depth_worsening/validation/depth_worsening.csv \
  --output-dir outputs/HALTING_RUN/analysis/depth_worsening/validation/figures
```

分别生成五种复杂度与深度的散点图、Oracle/停止深度比例直方图、相邻深度误差变差比例图，
输出 SVG 和 300 DPI PNG。绘图要求 Matplotlib 可找到真实 Arial 字体；
Windows/WSL 可读取 Windows 字体，Linux 需准备 Arial。
本机已验证的字体环境配置与原始结果路径见 [验证记录](VALIDATION.md)。

## 代码结构与 Trace

```text
Dataset → collate → SupervisedBatch
Task → Model → ModelOutput → loss
Trainer → backward / optimizer / metrics / checkpoint
Checkpoint → predict → CSV → analysis / plot
```

- `src/eqcompute/data/`：PyG QM9 适配、样本身份、划分和 batch。
- `src/eqcompute/models/`：EGNN、参数共享、固定/随机/自适应执行和停止器；当前坐标固定，使用分子内完全有向图。
- `src/eqcompute/tasks/`：目标归一化、训练阶段、冻结规则、监督与 loss。
- `src/eqcompute/training/`：优化器、scheduler、checkpoint 初始化与恢复、通用训练循环。
- `src/eqcompute/evaluation/`：固定深度和 adaptive 预测导出。
- `src/eqcompute/analysis/` 与 `plot/`：结果分析和绘图。

显式请求逐层预测或状态：

```python
from eqcompute.models.trace import TraceSpec

# 固定深度模型：逐层预测；保留梯度，可用于多深度监督。
out = model(batch.graph, task.context, trace=TraceSpec(predictions=True))

# AdaLoop：完整展开，收集逐层预测及指定深度的状态。
out = model(batch.graph, task.context, execution_mode="diagnostic", trace=TraceSpec(
    node_features=True, graph_features=True, positions=True, depths=(1, 3)
))
```

`predictions_by_depth` 按执行模式和监督需要保存预测；`trace` 表示为 detached CPU copies。
AdaLoop 的普通 eval 推理默认实际提前停止；完整轨迹需显式选择 `diagnostic`。

## 验证

```bash
QM9_ROOT=data/qm9 python -m pytest -q
python -m ruff check src tests scripts plot
git diff --check
```

测试不会下载 QM9。未设置有效 `QM9_ROOT` 时跳过真实数据集成测试；CUDA 不可用时跳过 GPU 测试。
绘图测试也要求 Arial 字体。

2026-10-22 已在 `AdaLoopEGNN` 环境、RTX 3060 上完成 **225 passed、0 failed、0 skipped**；
MAD 和 standard 两种归一化均完成两阶段 CUDA 短程验收，预测器状态保持不变，停止深度一致率为 100%。
结果与复现条件见 [VALIDATION.md](VALIDATION.md)。完整 1000 + 500 轮训练需另行启动。
