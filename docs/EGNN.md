# QM9 EGNN

模型：`src/eqcompute/models/backbones/EGNN.py`。
当前数据：`src/eqcompute/data/QM9.py`，直接读取本地 PyG QM9 缓存。
入口：`configs/experiment/qm9_egnn.yaml`。
不传 `experiment` 时默认使用同一基线。
共享参数入口：`qm9_loop_egnn`；自适应入口：`qm9_adaloop_egnn_predictor` / `qm9_adaloop_egnn_halting`。

**模型来源与范围**

模型固定参考 [vgsatorras/egnn commit e9ca6c0c](https://github.com/vgsatorras/egnn/tree/e9ca6c0c3e1d30a7598efbd66034121b4af8dccc)
中的 `qm9/models.py` 和 `models/gcl.py`。源码许可见 `third_party/EGNN-LICENSE.txt`。

- 15 维输入：H/C/N/O/F one-hot 与 `(Z / 9) ** [0, 1, 2]` 的乘积。
- 分子内完全有向图，无自环，使用距离平方，无 RBF、cutoff 或化学键属性。
- 消息门控、sum 聚合、残差 MLP、SiLU，坐标固定。
- 节点 MLP、sum pooling、共享图预测头；基线默认 final MAE 监督。
- 基线默认 7 层、128 隐藏维度；Loop 只注册一个共享 step。
- AdaLoop 默认 Kmax=24，单性质训练与停止机制见 [AdaLoop 说明](AdaLoopEGNN.md)。

packed graph 在真实节点上的预测、梯度和 Adam 更新已与固定版本的官方 padded 模型
对照验收。当前数据和训练调度采用本项目协议，不能将结果标作原论文数据流程的逐项复现。

**直接使用本地数据**

已有缓存目录：

```text
data/qm9/
  processed/
    data_v3.pt
```

固定深度、Loop、诊断和 AdaLoop 的配置继承 `dataset=qm9`，默认根目录 `data/qm9`。
数据适配器仅使用 PyG QM9 的数据接口与缓存。
如数据位于其他目录，设置 `dataset.root=/path/to/qm9`。

标签沿用 PyG 的 19 个目标定义，训练默认 mu。PyG 输入特征 x 和化学键不替代模型自己的
15 维原子编码及完全传播图；EGNN 从 z、pos 构造输入。

默认划分为 110000 train / 10000 validation / remaining test，split_seed=42。
训练 seed 不改变划分。sample ID 为 `qm9:<PyG 原始 idx>`，完整分组记录在 split_ids.json。
小样本验收使用 `dataset.subset_size`、`train_size`、`val_size`、`test_size`。

所有性质直接读取 PyG 标签，不额外扣除原子参考值，不重复做 Hartree 转换：

| 性质 | 单位 / 目标 |
|---|---|
| mu | D |
| alpha | a0^3 |
| homo/lumo/gap | eV |
| zpve | eV，转换为 meV 展示时乘以 1000 |
| U0/U/H/G | eV，总能量类目标 |
| r2 | a0^2 |
| Cv | cal/(mol K) |

U0 与 PyG 的 U0_atom 是不同目标，不将二者隐式互换。
默认训练集均值 / MAD 归一化，支持 `task.normalization=standard` 或 `none`。
验证和预测输出恢复到物理单位。数据来源和 target_convention=raw 写入 metadata。

**训练和 GPU**

在已激活项目环境的 WSL Bash 中：

```bash
python scripts/train.py experiment=qm9_egnn task.target=mu trainer.device=cuda:0
python scripts/train.py experiment=qm9_loop_egnn task.target=mu \
  model.execution.depth=8 trainer.device=cuda:0
python scripts/train.py experiment=qm9_adaloop_egnn_predictor task.target=mu trainer.device=cuda:0
```

AdaLoop 短程验收：

```bash
python scripts/train.py experiment=qm9_adaloop_egnn_smoke trainer.device=cuda:0
python scripts/smokeAdaLoop.py --data-root data/qm9 --device cuda:0
```

固定基线短程验收：

```bash
python scripts/train.py experiment=qm9_egnn trainer.device=cuda:0 \
  trainer.epochs=3 dataset.subset_size=192 \
  dataset.train_size=128 dataset.val_size=32 dataset.test_size=32
```

Adam、weight decay=1e-16，homo/lumo/gap 默认学习率 1e-3，其余原有性质 5e-4。
cosine 在每 epoch 训练和验证完成后调用一次；metrics.csv 记录该 epoch 实际学习率。
恢复时保持原 scheduler.t_max、数据配置及 task 配置，使用原 last.pt 恢复入口。

**固定深度扫描和预测**

```bash
python scripts/sweepDepth.py \
  --gpus 0,1,2,3 --depths 2,4,6,7,8,10,12 --seeds 0,1,2 \
  --weight-tying true -- experiment=qm9_loop_egnn task.target=mu
python scripts/evalDepth.py --sweep-dir outputs/_depth_sweeps/<batch> --device cuda:0
```

增加 tied 模型深度不增加参数量，仍增加执行成本。untied 参数则随层数增长。
固定模型 best.pt 按 validation 最终深度 MAE 选择；中间读出未受直接监督。
M0 诊断配置使用 all_steps，可用于单个 checkpoint 的逐深度 Oracle 分析。
自适应的 hard/diagnostic 导出、两阶段训练与恢复和概率损失见 AdaLoop 说明。

**验收**

```bash
python -m pytest -q
python -m ruff check src tests scripts
```

测试覆盖官方模型计算对照、当前 PyG 配置到训练/预测的链路、MAD/standard 切换、
权重共享、对称性、调度、恢复和结果汇总。当前源码仅保留 PyG QM9 数据适配器，
训练和预测测试通过小规模 PyG 测试数据验证，不下载数据或改写已有缓存。
