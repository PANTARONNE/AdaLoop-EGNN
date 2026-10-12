# Adaptive Equivariant Computation — Experiment 0

当前仅实现 QM9、untied EGNN、weight-tied Loop-EGNN、fixed-depth sweep 与逐深度预测。
不包含 adaptive halting、AdaLoop、计算惩罚、Oracle、QMugs、PaiNN 或晶体模型。

新增独立的官方 QM9 EGNN 复现入口：`experiment=qm9_egnn_orig`。
模型为 `EGNNOrig.py`，离线 XYZ 预处理为 `QM9Orig.py`；采用每 epoch 结束后一次 cosine
调度。数据准备、配置、单位和与官方脚本的差异见 [EGNNOrig 使用说明](docs/EGNNOrig.md)。
共享参数循环实验使用 `experiment=qm9_loop_egnn_orig`；多卡扫描指定 `--weight-tying true`。

## WSL 环境

仓库在 Windows 的 `D:\prjs\AdaLoopEGNN`，对应 WSL 的 `/mnt/d/prjs/AdaLoopEGNN`。
以下命令在 **WSL Bash** 中运行：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate AdaLoopEGNN
cd /mnt/d/prjs/AdaLoopEGNN
```

在另一台机器重建环境：

```bash
conda env create -f environment.yml
conda activate AdaLoopEGNN
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[dev]'
```

本机使用 Python 3.11、CUDA 12.1 版 PyTorch 2.3.1 与 PyG 2.7.0。
选择这一组合是为了复用已有完整安装缓存；实现不依赖新版本专属 API。
NumPy 限制在 1.x，兼容该 PyTorch wheel。实际安装版本记录于 `requirements-wsl.lock.txt`。
精确重建 Python 包时，可在新 Python 3.11 环境中运行
`python -m pip install -r requirements-wsl.lock.txt`。
CUDA 安装命令来源：[PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/#v231)。
不需要安装 torch-scatter、torch-cluster 或完整 CUDA toolkit。

## 最小运行

```bash
pytest -q
python scripts/train.py experiment=qm9_smoke trainer.device=cuda
```

首次 QM9 运行会下载 PyG 官方数据。没有 RDKit 时使用其预处理版本。
普通测试不下载数据；真实数据测试需显式指定已有数据目录：

```bash
QM9_ROOT=data/qm9 pytest -q -m integration
```

Smoke 配置随机选出 192 个真实样本，128 train / 32 validation / 32 test，
hidden_dim=32、K=3、3 epochs。其误差只用于检查流程，不是基准结果。

完整单次训练和不同配置：

```bash
python scripts/train.py experiment=qm9_egnn_depth
python scripts/train.py experiment=qm9_loop_egnn_depth task.target=gap
python scripts/train.py experiment=qm9_loop_all_steps
```

固定深度 sweep（会产生多个独立训练，不在 smoke 验证中自动启动）：

```bash
python scripts/train.py -m experiment=qm9_egnn_depth \
  'model.execution.depth=1,2,3,4,6,8,10,12' \
  'model.weight_tying=false,true' 'seed=0,1,2'
```

`experiment=qm9_egnn_depth` 对应 `configs/experiment/qm9_egnn_depth.yaml`。
默认深度为 4；单次训练可用 `model.execution.depth=8` 覆盖，
多卡扫描由 `sweepDepth.py --depths ...` 为每个任务设置深度。

Hydra preset 只覆盖差异项，没有重复的 train_depthN.py。
Hydra 的 sweep 记录单独保存在 `outputs/_sweeps/`；每个训练任务仍有独立运行目录。
从仓库根目录启动；相对数据路径按当前目录解释，并在保存配置中解析为绝对路径。

## 多 GPU 深度扫描（单机 Linux）

新增入口 `scripts/sweepDepth.py` 扫描 `depth × seed`。每个组合是独立单卡训练，
每张指定 GPU 同时只运行一个任务，任务结束后立即领取下一个组合。
`--weight-tying true/false` 必填，整批固定；该参数优先于 experiment preset 中的设置。

例如在 8 张 GPU 上执行 8 个深度 × 3 个 seed，共 24 次 untied EGNN 训练：

```bash
python scripts/sweepDepth.py \
  --gpus 0,1,2,3,4,5,6,7 \
  --depths 1,2,3,4,6,8,10,12 \
  --seeds 0,1,2 \
  --weight-tying false -- \
  experiment=qm9_egnn_depth task.target=mu
```

Loop-EGNN 使用 `--weight-tying true`。其他训练设置继续通过 `--` 后的 Hydra overrides
传入，例如 `dataset.root=/data/qm9`、`trainer.epochs=200`、`loader.batch_size=128`。
一次只扫描深度和 seed；不要再传 `-m`、逗号扫描、`seed=...`、`model.execution.depth=...`、
`model.weight_tying=...`、`trainer.device=...`、`resume=...` 或 Hydra 运行控制参数。

先检查任务列表而不准备数据、不创建输出、不启动训练：

```bash
python scripts/sweepDepth.py --gpus 0,1 --depths 2,4 --seeds 0,1 \
  --weight-tying true --dry-run -- experiment=qm9_smoke
```

去掉 `--dry-run` 即为 4 个短训练的服务器验收命令。dry-run 只验证参数与配置，
不验证 GPU 是否存在、可用或显存充足。

GPU 编号按当前 CUDA 可见设备顺序解释。没有设置 `CUDA_VISIBLE_DEVICES` 时，
`--gpus 0,1` 选择 CUDA 设备 0、1；若已有 `CUDA_VISIBLE_DEVICES=3,7`，
`--gpus 0,1` 选择该列表中的两张卡。子进程只看见分配给它的单张卡，使用逻辑设备 `cuda:0`。
指定列表视为已分配给本批次的 GPU，不检测其他程序是否占用这些卡。

调度前通过现有 dataset factory 串行准备一次数据，完成后才并行启动训练，避免本批次
首次下载/预处理冲突。服务器不能联网时需提前准备好完整 QM9 缓存并指定 `dataset.root`。
各任务保持相同 dataset 配置和 `split_seed`，只改变训练 seed 和深度；模型、Task、Trainer
及 checkpoint 格式均复用单次训练实现。每个进程各自加载数据，CPU 内存与
`loader.num_workers` 的总开销会随并发任务数增加。

相对数据和输出路径统一按仓库根目录解释；子进程使用启动脚本的同一个 Python 环境。
输出保存在 `output_root` 下的独立批次目录：

```text
outputs/_depth_sweeps/<UTC_timestamp>_<id>/
  jobs.json
  logs/k2_s0.log
  runs/k2_s0/<原有唯一运行目录>/
    config.yaml
    metadata.json
    split_ids.json
    checkpoints/{best.pt,last.pt}
    metrics.csv
    predictions/
    logs/train.log
```

`jobs.json` 持续记录各任务参数、GPU、PID、状态、退出码、完整启动参数、日志和运行目录。
任务日志还包含训练目录建立之前的启动错误。任一任务失败后继续派发其他任务，
全部完成后有失败则退出码为 1，全成功为 0。Ctrl+C 或 SIGTERM 会停止本批次的训练进程
及其 DataLoader 子进程，未启动任务标为 cancelled，调度器退出码为 130。

失败任务手动重跑，例如只重跑 depth=4、seed=1，并保持其余训练参数一致：

```bash
python scripts/sweepDepth.py --gpus 0 --depths 4 --seeds 1 \
  --weight-tying false -- experiment=qm9_egnn_depth task.target=mu
```

重跑创建新的批次与运行目录。需要从某个 `last.pt` 继续时，使用下文的单次训练恢复入口；
本脚本不自动重试或恢复整批任务。原有 `scripts/train.py` 单次训练和 Hydra 串行 sweep 用法保留。

## 实验协议

`task.supervision=final`：仅最终深度预测参与 loss。用于分别训练不同 K 的基线。
中间预测可以导出，但没有得到直接监督，不能把它们当作充分训练的停止候选。

`task.supervision=all_steps`：所有执行深度的标量 loss 等权平均，所有深度共用同一 head。
用于单个 checkpoint 的预测随深度变化实验。这是另一种训练协议，不能与 final 混报。
两者均无 halting、随机深度 warm-up 或计算惩罚。

默认目标为 mu。支持通过配置选择 PyG QM9 的 19 个目标；推荐先比较 mu、alpha、gap、U0。
单目标 checkpoint 拒绝其他 property context；预留参数不意味着已经实现多目标模型。

默认数据划分为 110000 / 10000 / remaining，split_seed=42；训练 seed 与划分 seed 分开。
按原始 QM9 idx 生成 sample_id。数据集目标单位沿用 PyG，不能再次做 Hartree 到 eV 转换。
U0（列 7）与 U0_atom（列 12）不是同一目标。
标准化只拟合训练集，预测导出和 MAE/RMSE 都恢复为原始物理单位。不使用原子参考能修正。
划分是本项目声明的固定随机划分，不声称与某篇论文逐样本一致。

## 模型与数据流

Python 文件名使用 lower camel case，专有简称保留大写，例如 `depthCurves.py`、
`GPUQueue.py`、`EGNN.py`、`QM9.py`、`testCUDA.py`。`__init__.py` 与 `conftest.py`
保留工具约定名称；pytest 通过 `test*.py` 发现测试文件。

```text
Dataset -> collate -> SupervisedBatch
Task -> Model(GraphInputs, PropertyContext) -> ModelOutput -> loss
Trainer -> Task.run -> backward/optimizer/checkpoint/Metric
Checkpoint -> 独立 predict -> CSV -> 离线 analysis
```

- `contracts.py`：GraphInputs 与带标签的 SupervisedBatch 分开。
- `data/`：QM9 adapter、稳定身份与持久化划分；不计算 loss。
- `models/backbones/EGNN.py`：initializer 与一次 EGNN 更新。
- `models/operators.py`：参数所有权；tied 仅注册一个 step，untied 使用独立 ModuleList。
- `models/execution.py`：固定执行次数；不定义 loss。
- `models/heads.py`：读出与带 property context 的共享预测头。
- `tasks/`：目标变换、监督协议与 loss；不管理 optimizer。
- `training/`：通用训练与 checkpoint；不读取 depth 或具体模型类型。
- `metrics/`：按样本累计 MAE/RMSE，避免小尾批次被过度加权。
- `evaluation/`：逐深度预测流式写盘。
- `analysis/`：仅消费结果文件，不参与 Trainer。

原有 `model.name=egnn` 实现是记录了变体细节的 EGNN，不声称严格复现原论文的 QM9 配置：
原子序数 embedding；Gaussian RBF；有向 radius edges；mean message aggregation；
特征残差和 LayerNorm；归一化相对向量、tanh 标量与 coordinate_scale=0.1 的坐标更新；
默认 sum pooling 与共享标量 MLP head。

radius graph 在 initialize 时由原始输入坐标构建，执行过程中拓扑不变。
每步从当前 state.positions 重算距离和 RBF；cutoff 仅决定初始边集合，不在后续动态裁剪边。
输入 positions 不被原地修改。化学键 edge_index/edge_attr 保留在输入中，但不直接用于 radius edges。
按分子计算 cdist，没有邻居数截断，不依赖可选 CUDA 扩展；复杂度为每个分子 O(N²)，当前面向 QM9。
未来大分子应替换构图函数；未来晶体还需专门的 cell/PBC/周期位移支持。

直接操作单步：

```python
state = model.initialize(batch.graph)
for k in range(1, 5):
    state = model.step(state, batch.graph, depth=k)
    graph_repr = model.readout(state, batch.graph)
    prediction = model.head(graph_repr, task.context)
```

`depth=k` 仅为 untied 模式选择参数；EGNNStep 自身不知道总深度。
tied 模式所有 k 都返回同一算子。untied 超出已注册层数会报错。

## Trace

```python
from eqcompute.models.trace import TraceSpec

out = model(batch.graph, task.context)  # 不创建逐步状态列表
out = model(batch.graph, task.context, trace=TraceSpec(predictions=True))
out = model(batch.graph, task.context, trace=TraceSpec(
    node_features=True, graph_features=True, positions=True, depths=(1, 3)
))
```

`predictions_by_depth` 保留梯度，供多深度监督使用。
`trace` 中的表示是 detached CPU copies，仅供分析。实际递归 state 不 detach。
关闭 trace 不消除反向传播必需的激活；它消除的是额外保存的逐步输出。
图表示和节点表示只在请求时保存。进行大批量表示分析时应按 batch 消费，不能累积整套数据。

## 输出与预测

每次运行产生唯一目录，名称包含 target、tying、depth、监督方式、seed 和运行后缀：

```text
outputs/<experiment_name>__<UTC_timestamp>_<run_id>/
  config.yaml
  metadata.json
  split_ids.json
  checkpoints/{best.pt,last.pt}
  metrics.csv
  predictions/
  logs/train.log
```

`best.pt` 按 validation 的最终输出 MAE 选择；all_steps 的 loss 是各步平均，但 checkpoint
仍按最终 MAE 选取。测试集不用于选深度、checkpoint 或训练超参数。

```bash
python scripts/predict.py --checkpoint outputs/<run>/checkpoints/best.pt --split test
python scripts/predict.py --checkpoint outputs/<run>/checkpoints/best.pt --split validation
```

导出文件为 `predictions/test_best_by_depth.csv` 等。
每行对应 sample_id × property × readout_depth；含 checkpoint SHA256、split ID、训练深度、
监督方式、seed、真实值、预测值和单位。同一个 batch 的所有深度来自一次执行轨迹。
已有文件默认拒绝覆盖，可显式传入 `--overwrite`。

```bash
python -m eqcompute.analysis.depthCurves \
  outputs/<run>/predictions/test_best_by_depth.csv \
  --output outputs/<run>/predictions/depth_curve.png
```

可追加 `--sample-id qm9:123`，样本必须实际存在于该 CSV。
本模块只画误差曲线，不计算 Oracle 或 Minimum Sufficient Depth。

## 汇总一次深度扫描

`scripts/evalDepth.py` 读取当次扫描的 `jobs.json`，在终端按训练 Depth 汇总各 seed 的
validation MAE 与 test MAE：

```bash
python scripts/evalDepth.py \
  --sweep-dir outputs/_depth_sweeps/<批次目录> \
  --device cuda:0
```

默认 `--device auto`；可指定 `cpu`。可选 `--batch-size 128` 控制缺失预测的评估批大小，
`--precision 6` 控制小数位数。命令应在已安装项目的 Python 环境中运行。

终端表格示例（以下数字仅作格式示意）：

```text
Target: mu | Unit: D | Model: egnn | weight_tying: false | supervision: final
Mean ± sample SD (ddof=1); Seeds = evaluated/planned; checkpoint = best.pt.
Depth  Seeds  Val MAE              Test MAE
-----  -----  -------------------  -------------------
1      3/3    0.125000 ± 0.008000  0.132000 ± 0.010000
2      3/3    0.110000 ± 0.006000  0.118000 ± 0.007000
4      1/3    0.098000 ± N/A       0.105000 ± N/A
```

每个 seed 使用按 validation MAE 选出的 `best.pt`：Val MAE 读取 checkpoint 中保存的
`best_value`，Test MAE 使用该 checkpoint 在**最终训练深度**的预测计算，不混入中间读出。
`±` 为各 seed 的样本标准差（ddof=1），只有一个有效 seed 时显示 `± N/A`，
某个深度没有有效 seed 时两项指标显示 `N/A`。指标沿用目标属性的原始物理单位。

脚本复用 `predictions/test_best_by_depth.csv`；缺失、checkpoint 哈希不匹配、
实验信息不匹配或最终深度测试样本不完整时，调用现有预测模块重新评估并替换该 CSV。
首次评估需访问 checkpoint 中记录的数据路径，多个任务按顺序在指定设备上评估。
复制扫描目录后优先使用批次内的 `runs/k<depth>_s<seed>/` 寻找运行目录，
数据集仍需在 checkpoint 记录的位置可访问。

失败、未完成、缺少 checkpoint 或测试评估失败的任务会跳过并在 stderr 说明原因；
Val/Test 只汇总同一组成功评估的 seed，`Seeds` 显示有效数/计划数。
拒绝混合不同目标、单位、模型、数据划分、参数共享设置或监督协议，也拒绝重复 depth/seed。
表格写入 stdout，可用 `> summary.txt` 保存；全部成功退出码为 0，有跳过任务为 1，
输入配置错误为 2。测试集结果用于报告，深度与超参数选择仍依据验证集。

## 恢复训练

```bash
python scripts/train.py experiment=qm9_smoke trainer.epochs=5 \
  resume=outputs/<previous_run>/checkpoints/last.pt
```

恢复到新运行目录，metadata 保留来源。除增加 epochs 或改变设备外，需保持实验配置一致。
恢复模型、optimizer、目标变换、Python/NumPy/Torch/CUDA RNG 和 loader generator；
保留来源运行已有的 best checkpoint。新 metrics.csv 只记录新增 epochs。
支持 epoch 边界恢复，不支持中途 batch 恢复。CPU 测试检查逐位一致；CUDA reduction
可能存在浮点非确定性，不能保证跨设备/软件版本逐位复现。
checkpoint 含 Python RNG 对象，只加载本项目生成且可信的文件。

## 验收范围

pytest 覆盖：离线 batch/ID、split、目标变换、单步与 wrapper 一致性、参数共享、梯度、
trace、旋转/平移/反射/置换、批间隔离、空邻域、小图过拟合、checkpoint 恢复、逐深度导出。
真实 QM9 子集的 GPU 验证记录另见 `VALIDATION.md`。

后续 AdaLoop 可以增加 ExecutionPolicy 和相应 Task；PaiNN 可以增加自己的 state、step、readout。
这些扩展无需将模型循环塞入 Trainer，但并不意味着当前已经实现这些方法。
