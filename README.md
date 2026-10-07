# Adaptive Equivariant Computation — Experiment 0

当前仅实现 QM9、untied EGNN、weight-tied Loop-EGNN、fixed-depth sweep 与逐深度预测。
不包含 adaptive halting、AdaLoop、计算惩罚、Oracle、QMugs、PaiNN 或晶体模型。

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
python scripts/train.py experiment=qm9_egnn_depth4
python scripts/train.py experiment=qm9_loop_egnn_depth4 task.target=gap
python scripts/train.py experiment=qm9_loop_all_steps
```

固定深度 sweep（会产生多个独立训练，不在 smoke 验证中自动启动）：

```bash
python scripts/train.py -m experiment=qm9_egnn_depth4 \
  'model.execution.depth=1,2,3,4,6,8,10,12' \
  'model.weight_tying=false,true' 'seed=0,1,2'
```

Hydra preset 只覆盖差异项，没有重复的 train_depthN.py。
Hydra 的 sweep 记录单独保存在 `outputs/_sweeps/`；每个训练任务仍有独立运行目录。
从仓库根目录启动；相对数据路径按当前目录解释，并在保存配置中解析为绝对路径。

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

```text
Dataset -> collate -> SupervisedBatch
Task -> Model(GraphInputs, PropertyContext) -> ModelOutput -> loss
Trainer -> Task.run -> backward/optimizer/checkpoint/Metric
Checkpoint -> 独立 predict -> CSV -> 离线 analysis
```

- `contracts.py`：GraphInputs 与带标签的 SupervisedBatch 分开。
- `data/`：QM9 adapter、稳定身份与持久化划分；不计算 loss。
- `models/backbones/egnn.py`：initializer 与一次 EGNN 更新。
- `models/operators.py`：参数所有权；tied 仅注册一个 step，untied 使用独立 ModuleList。
- `models/execution.py`：固定执行次数；不定义 loss。
- `models/heads.py`：读出与带 property context 的共享预测头。
- `tasks/`：目标变换、监督协议与 loss；不管理 optimizer。
- `training/`：通用训练与 checkpoint；不读取 depth 或具体模型类型。
- `metrics/`：按样本累计 MAE/RMSE，避免小尾批次被过度加权。
- `evaluation/`：逐深度预测流式写盘。
- `analysis/`：仅消费结果文件，不参与 Trainer。

该实现是记录了变体细节的 EGNN，不声称严格复现原论文的 QM9 配置：
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
python -m eqcompute.analysis.depth_curves \
  outputs/<run>/predictions/test_best_by_depth.csv \
  --output outputs/<run>/predictions/depth_curve.png
```

可追加 `--sample-id qm9:123`，样本必须实际存在于该 CSV。
本模块只画误差曲线，不计算 Oracle 或 Minimum Sufficient Depth。

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
