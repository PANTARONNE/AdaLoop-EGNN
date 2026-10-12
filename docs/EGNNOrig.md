# QM9 EGNNOrig

模型：`src/eqcompute/models/backbones/EGNNOrig.py`。
数据：`src/eqcompute/data/QM9Orig.py`。
组合配置：`configs/experiment/qm9_egnn_orig.yaml`。
共享参数循环配置：`configs/experiment/qm9_loop_egnn_orig.yaml`。
原有 `egnn` / `qm9` 配置继续使用原来的模型和 PyG 预处理。

## 复现依据和范围

固定参考 [vgsatorras/egnn commit e9ca6c0c](https://github.com/vgsatorras/egnn/tree/e9ca6c0c3e1d30a7598efbd66034121b4af8dccc)
的 `qm9/models.py`、`models/gcl.py` 和 `qm9/data/`，以及
[论文补充材料 C.3](https://proceedings.mlr.press/v139/satorras21a/satorras21a-supp.pdf)。
相关源码使用 MIT 许可证，副本在 `third_party/EGNN-LICENSE.txt`。

- 15 维输入：H/C/N/O/F one-hot 与 `(Z / 9) ** [0, 1, 2]` 的乘积。
- 分子内完整有向图，不含自环；使用距离平方，无 RBF、cutoff 或化学键属性。
- Linear + Sigmoid 消息门控、sum 聚合、残差 MLP、SiLU；不含 LayerNorm。
- QM9 中坐标固定，`node_attr=0`，7 个独立层、128 隐藏维度。
- 节点 MLP → sum pooling → 图 MLP；仅最终深度参与 MAE loss。
- Adam，weight decay `1e-16`，1000 epochs，batch size 96，无梯度裁剪。
- homo/lumo/gap 的初始学习率为 `1e-3`，其他 9 个性质为 `5e-4`；随 target 自动选择。
- 训练集均值和 MAD 归一化，验证和测试 MAE 使用反归一化后的单位。

按本项目约定，cosine scheduler **在每个 epoch 的优化和验证完成后调用一次**。
第一个 epoch 使用初始学习率，最后一个 epoch 完成后降至 0。验证和预测不会推进调度器。
这是明确选择的训练协议：官方公开脚本在训练/验证/测试共用函数开头调用 scheduler，
默认一轮会调用三次；本实现不复制这一调用方式。

批处理复用现有 packed graph 接口，移除 padding 节点及被 mask 的边。
在真实节点上与官方 padded 模型等价，并通过相同参数下的预测、梯度和 Adam 更新对照验证。
训练 seed 由项目现有机制统一设置；不保证与原脚本随机初始化、shuffle 顺序或跨硬件逐位相同。
这些实现级验证不等于已经完成 1000 epochs 的论文精度验收。

## 离线准备数据

使用 QM9 原始 XYZ 数据，不复用 `data/qm9/processed/data_v3.pt`。
在有网络的机器准备下面三个原始文件，再复制到训练机器：

| 文件 | 来源 |
|---|---|
| `dsgdb9nsd.xyz.tar.bz2` | [QM9 XYZ archive](https://springernature.figshare.com/ndownloader/files/3195389) |
| `uncharacterized.txt` | [排除名单](https://springernature.figshare.com/ndownloader/files/3195404) |
| `atomref.txt` | [原子参考值](https://springernature.figshare.com/ndownloader/files/3195395) |

目录结构：

```text
data/qm9_orig/
  raw/
    dsgdb9nsd.xyz.tar.bz2
    uncharacterized.txt
    atomref.txt
```

首次训练会自动生成 `processed/qm9Orig.pt` 与 `processed/qm9Orig.json`。
整个新数据流程不发起网络请求。也可以提前准备：

```bash
python -c 'from eqcompute.data.QM9Orig import prepare_qm9_orig; prepare_qm9_orig("data/qm9_orig")'
```

完成后可以只复制两个 processed 文件到另一台机器，保留相同目录结构；若 raw 文件仍存在，
加载时也会校验其 SHA256。缓存校验失败会报错，不会静默换用其他数据。
多卡扫描在启动训练进程前串行准备一次；避免同时用多个独立命令初始化同一个新缓存。

预处理保留 H 原子和 XYZ 中的原始坐标；排除官方名单中的 3054 个分子，剩余 130831 个。
采用官方 NumPy legacy `RandomState(0)` 划分算法，100000 train / 17748 validation / 13083 test。
集合内保持 archive 顺序，与官方读取方式一致。训练 seed 不改变这个划分。
`sample_id` 使用 `qm9orig:<原始1起始分子ID>`，与旧 PyG 数据身份区分。

原始标签先按官方代码转为 float32；对 `zpve/U0/U/H/G/Cv` 扣除对应原子参考值总和，
随后使用官方常数进行单位转换。缓存保存原始标签和参考值，避免多次加载重复扣除或重复转换。
训练配置、原始文件哈希、缓存哈希和目标处理方式一起写入运行记录。

## 训练、扫描和分析

单次论文结构基线：

```bash
python scripts/train.py experiment=qm9_egnn_orig task.target=mu trainer.device=cuda:0
```

修改数据位置：在命令后添加 `dataset.root=/path/to/qm9_orig`。
快速验证本地真实数据上的完整流程（结果不作为论文基准）：

```bash
python scripts/train.py experiment=qm9_egnn_orig trainer.device=cuda:0 \
  trainer.epochs=3 dataset.train_limit=128 dataset.validation_limit=32 dataset.test_limit=32
```

沿用现有多卡入口：

```bash
python scripts/sweepDepth.py \
  --gpus 0,1,2,3,4,5,6,7 --depths 2,4,6,7,8,10,12 --seeds 0,1,2 \
  --weight-tying false -- experiment=qm9_egnn_orig task.target=mu

python scripts/evalDepth.py --sweep-dir outputs/_depth_sweeps/<批次> --device cuda:0
```

`--depths` 会覆盖默认 7 层，其他设置继续沿用原版配方；不同深度属于深度消融实验。
论文基线使用 `weight_tying=false`。`EGNNOrig` 支持共享参数循环，两种模式均使用最终层监督。
可用 `--dry-run` 检查扫描计划。

共享参数循环实验使用新配置，继承原版的数据、优化器和每 epoch 一次学习率调度：

```bash
python scripts/train.py experiment=qm9_loop_egnn_orig task.target=mu \
  model.execution.depth=8 trainer.device=cuda:0

python scripts/sweepDepth.py \
  --gpus 0,1,2,3,4,5,6,7 --depths 2,4,6,7,8,10,12 --seeds 0,1,2 \
  --weight-tying true -- experiment=qm9_loop_egnn_orig task.target=mu
```

循环模式只注册一个 `EGNNOrigStep`，每个深度重复使用这组边 MLP、节点 MLP 和消息门控参数。
梯度沿整个循环反向传播并累积到共享参数；输入编码器、节点读出和图预测头也各只有一份。
默认执行深度为 7，可由 CLI 覆盖。增加循环深度不增加参数量，仍会增加计算量和训练激活内存。
该配置属于基于 EGNNOrig 的共享参数实验，不标作论文原版基线。

也可直接用 `experiment=qm9_egnn_orig model.weight_tying=true` 启用循环。
扫描脚本以 `--weight-tying` 为准；分析继续使用 `evalDepth.py`，表头及 CSV 会记录 tying 模式。
共享与独立参数的 checkpoint 具有不同参数布局，恢复训练需保持原来的 tying 模式。

## 结果口径

`best.pt` 按最低 validation MAE 保存，使用该模型计算最终深度的 test MAE。
仍可导出中间深度预测，但这些读出未经独立监督。

`homo/lumo/gap/U0/U/H/G` 使用 **eV**；与论文表格的 meV 比较时乘以 1000。
`zpve` 按官方数据代码直接使用 **meV**，不要再次乘以 1000。
mu 为 D，alpha 为 a0³，r2 为 a0²，Cv 为 cal/(mol K)。终端表格和预测 CSV 都标明单位。

对扣除了原子参考值的性质，CSV 中的 `y_true/y_pred` 表示修正后的目标及其预测，
不是分子总量。对预测和标签加回相同参考值不改变 MAE。
`split_ids.json` 记录 `subtract_thermo` 和缓存来源；更改预处理会改变数据指纹，避免评估混用。

## 学习率记录与恢复

`metrics.csv` 的 `learning_rate` 是该 epoch 训练实际使用的学习率。
checkpoint 中同时保存 optimizer 和 scheduler 状态。

```bash
python scripts/train.py experiment=qm9_egnn_orig task.target=mu \
  resume=outputs/<原运行>/checkpoints/last.pt
```

恢复时保持原来的配置，包括 cosine 周期 `trainer.scheduler.t_max`。
默认周期随 `trainer.epochs` 解析，因此不能通过改 epochs 静默重设已经运行的学习率曲线。
允许在原周期内延长原本提前结束的训练，但必须显式保留原来的 `t_max`；不允许超过周期后回升。
原有未使用 scheduler 的配置及 checkpoint 仍按原有方式恢复。

## 验证

```bash
pytest -q tests/testEGNNOrig.py
pytest -q
ruff check src tests scripts
```

测试自带固定版本的官方模型源码快照以及合成的 XYZ/archive/atomref 文件，不访问网络。
覆盖官方前向/梯度/Adam 一步对照、对称性、坐标固定、数据处理和缓存完整性、
共享参数身份及梯度累积、每 epoch 调度、恢复等价性，以及训练 → 预测 → 扫描结果汇总。
