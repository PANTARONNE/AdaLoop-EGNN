# 单性质 AdaLoop-EGNN（M0–M3）

消息传播使用 `models/backbones/EGNN.py` 中的 `EGNNStep`，完全图、固定坐标、共享参数。
固定深度基线支持独立参数和共享参数；AdaLoop 复用同一消息传播计算。
本轮实现单性质模型，不包含 property embedding 或多性质联合训练。

**范围与配置**

| 配置 | 用途 |
|---|---|
| `qm9_loop_egnn_diagnostic` | M0：共享参数、Kmax=12、所有深度等权 MAE 监督 |
| `qm9_adaloop_egnn` | 单性质自适应模型，hidden_dim=128、Kmax=24 |
| `qm9_adaloop_egnn_smoke` | 本地 PyG QM9 短程验收，hidden_dim=32、Kmax=4、3 epochs |

默认 `task.target=mu`、`task.normalization=mad`，归一化仅用训练标签拟合。
可通过 `task.normalization=standard` 或 `none` 切换；对照实验需统一该设置。
默认 `model.execution.threshold=0.5`。
改变深度仍使用 `model.execution.depth`；自适应模型中该值表示 Kmax。

固定深度、Loop、M0 诊断和 AdaLoop 均直接使用 `dataset=qm9`、`dataset.root=data/qm9`。
已有 `processed/data_v3.pt` 可直接加载，无需另行准备 XYZ 或 atomref。
默认划分为 110000 train / 10000 validation / remaining test，
`dataset.split_seed=42`；所有基线需使用相同划分和标签处理。
PyG 标签直接读取，不额外扣除原子参考值；zpve 为 eV，U0 是总能量目标。
`qm9_egnn` 和 `qm9_loop_egnn` 仍默认 final 监督。

**Halting Controller 设计**

设分子图 $b$ 有 $N_b$ 个节点，最大展开深度为 $K=K_{\max}$，第 $k$ 步节点特征为
$h_{b,i}^{(k)}\in\mathbb{R}^{d}$。$h^{(0)}$ 是 initializer 输出；每一步先执行共享的
`EGNNStep`，再由共享 readout/head 得到该深度的归一化预测 $\hat y_{b,k}$，
同时由 controller 判断是否停止。消息传播、readout/head 和 controller 的参数均跨深度共享，
controller 不显式输入深度 $k$。

Controller 在每个图内计算以下统计量：

$$
\mu_{b,k}=\frac{1}{N_b}\sum_i h_{b,i}^{(k)},\qquad
\sigma_{b,k}=\sqrt{\frac{1}{N_b}\sum_i
\left(h_{b,i}^{(k)}-\mu_{b,k}\right)^{\odot 2}+\epsilon},
$$

$$
\delta_{b,k}=\frac{1}{N_b}\sum_i
\left\|h_{b,i}^{(k)}-h_{b,i}^{(k-1)}\right\|_2,\qquad
z_{b,k}=[\mu_{b,k};\sigma_{b,k};\delta_{b,k}].
$$

均值和标准差均为 $d$ 维向量；$\odot 2$ 表示逐特征维平方。标准差使用总体方差
（分母为 $N_b$，无无偏修正），$\epsilon=10^{-8}$，因此单节点图也能得到有限值。
变化量 $\delta_{b,k}$ 是“各节点变化向量的 L2 范数再取均值”，为一个标量，
用于提供相邻步更新幅度信息，并不直接证明预测已收敛。

图级统计量输入两层 MLP，输出条件停止概率：

$$
q_{b,k}=\operatorname{sigmoid}
\left(W_2\,\operatorname{SiLU}(W_1z_{b,k}+a_1)+a_2\right).
$$

| 项目 | 当前设计 |
|---|---|
| 输入维度 | 启用 delta 时为 $2d+1$，关闭时为 $2d$ |
| 网络结构 | `Linear(input_dim, controller_dim) → SiLU → Linear(controller_dim, 1) → Sigmoid` |
| 默认维度 | `model.hidden_dim=128`，`model.halting.hidden_dim=64`，即 `257 → 64 → 1` |
| 输出 | 每个图一个标量 $q_{b,k}$；batch 中各图分别计算 |
| delta 消融 | `model.halting.use_delta=false` 仅移除变化量输入，保留骨架残差连接 |

Controller 不输入原始坐标、标签、Oracle、性质 embedding 或人工复杂度指标。
这些统计量对节点排列不变；由于当前骨架以原子特征和相对距离构造标量隐藏特征，
停止决策也保持对整体平移、旋转和反射的不变性。几何信息可经骨架隐藏特征间接影响决策。

$q_{b,k}$ 表示“已执行到第 $k$ 步且此前未停止”条件下的停止概率，
不是直接对所有深度做 softmax。设 $S_{b,k}$ 为到达第 $k$ 步的剩余概率质量，则：

$$
S_{b,1}=1,\qquad
p_{b,k}=S_{b,k}q_{b,k},\quad
S_{b,k+1}=S_{b,k}(1-q_{b,k})\quad (k<K),
$$

$$
p_{b,K}=S_{b,K}=\prod_{j=1}^{K-1}(1-q_{b,j}).
$$

因此 $p_{b,k}$ 是恰好在第 $k$ 步停止的概率，满足 $p_{b,k}\geq 0$、
$\sum_{k=1}^{K}p_{b,k}=1$。末步直接设 $q_{b,K}=1$ 并接收全部剩余质量，
不调用 controller 学习末步停止概率；$K=1$ 时退化为 $p_{b,1}=1$。
例如 $q=(0.2,0.5,1)$ 对应 $p=(0.2,0.4,0.4)$。

实现见 [HaltingController 与停止分布](../src/eqcompute/models/halting.py)、
[AdaptiveExecution](../src/eqcompute/models/adaptiveExecution.py)。

**Loss 设计**

训练目标是停止分布下的期望预测误差。标签先转换为
$\tilde y_b=(y_b-m)/s$，其中 $m$、$s$ 仅在训练集拟合。默认 `mad` 使用训练均值 $m$
和平均绝对偏差 $s=\operatorname{mean}_{\mathrm{train}}|y-m|$
（这里不是中位数绝对偏差）；`standard` 使用总体标准差，二者的 scale 均下限截断为
$10^{-8}$；`none` 使用 $m=0,s=1$。所有深度的 $\hat y_{b,k}$ 都在同一标签尺度上监督，
报告 MAE 时再将所选预测还原到原目标单位。

对于含 $B$ 个分子的 batch，默认 `task.loss=mae`：

$$
\ell_{b,k}=|\hat y_{b,k}-\tilde y_b|,\qquad
\mathcal{L}_{\mathrm{prediction}}
=\frac{1}{B}\sum_{b=1}^{B}\sum_{k=1}^{K}p_{b,k}\ell_{b,k},
$$

$$
\mathcal{L}
=\mathcal{L}_{\mathrm{prediction}}.
$$

`task.loss=mse` 时仅将 $\ell_{b,k}$ 换为平方误差，停止分布保持相同定义。
单性质预测张量形状为 `[B,K,1]`，标签为 `[B,1]`，停止质量为 `[B,K]`；
实现先计算每个样本、每个深度的误差，再沿深度按 $p$ 加权，最后沿 batch 求均值：

```text
prediction_loss = mean_b sum_k p[b,k] * abs(y_k[b]-normalized_target[b])
loss = prediction_loss
```

这里优化的是停止分布下的**期望误差**，不是先混合预测再计算误差
$|\sum_k p_{b,k}\hat y_{b,k}-\tilde y_b|$。例如目标为 2、两步预测为 0 和 4、
停止质量各为 0.5 时，期望 MAE 为 2，而混合预测的 MAE 为 0；
后者可能通过不同深度误差抵消，无法反映实际选择某一步输出时的误差。

各深度误差相同时，调整停止质量不会改变损失；停止深度由各步误差和模型训练结果决定。
比较不同实验时需统一 normalization。

自适应阶段完整展开到 $K$，不采样离散停止深度，也不通过 CDF 阈值选择反向传播。
$\hat y_{b,k}$ 与 $p_{b,k}$ 都保留计算图：预测项训练 backbone/readout/head，
并通过停止质量训练 controller。Controller 根据各深度的预测误差分配停止质量，
无需 Oracle 停止标签或 REINFORCE。当前目标没有额外的熵正则或
Oracle 监督；`halting_entropy` 仅作为日志指标。

Warm-up 阶段每个 batch 均匀抽取一个深度 $k^\star\sim\mathrm{Uniform}\{1,\ldots,K\}$，
整个 batch 执行到该深度，使用
$\mathcal{L}_{\mathrm{warmup}}=\frac{1}{B}\sum_b\ell_{b,k^\star}$。
此时不调用 controller，仅训练共享 backbone/readout/head，
使其在不同深度都能输出预测；进入自适应阶段后再联合学习停止策略。
Validation 始终使用 `full` 模式计算上述自适应 loss，包括 warm-up 期间的 validation。

实现见 [adaptive_loss 与训练阶段](../src/eqcompute/tasks/adaptiveRegression.py)、
[标签变换](../src/eqcompute/tasks/targetTransform.py)；
[手算损失与梯度测试](../tests/testHalting.py) 验证停止质量、逐样本加权和梯度。

**训练与停止语义**

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
python scripts/train.py experiment=qm9_loop_egnn_diagnostic task.target=mu
python scripts/train.py experiment=qm9_adaloop_egnn task.target=mu
python scripts/train.py experiment=qm9_adaloop_egnn task.normalization=standard
```

直接读取本地 QM9 缓存的短程训练：

```bash
python scripts/train.py experiment=qm9_adaloop_egnn_smoke trainer.device=cuda
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
python scripts/train.py experiment=qm9_adaloop_egnn \
  resume=outputs/<run>/checkpoints/last.pt
```

保持数据、模型、task（含 warm-up）、优化器及 cosine 周期配置一致。
本轮不实现从旧 Loop 权重迁移初始化 AdaLoop；resume 不是跨模型迁移。

metrics.csv 保存 prediction_loss、expected_depth、halt_depth、executed_depth、
首步/末步停止比例、分布熵和 warm-up 标记。按样本数加权；不适用统计留空，表头不变。
expected_depth 是 soft 期望；halt_depth 是选择结果；executed_depth 是实际执行长度。
validation 使用完整轨迹，因此其 executed_depth=Kmax。

**预测、Oracle 与结构指标**

```bash
python scripts/predict.py --checkpoint outputs/<run>/checkpoints/best.pt --split test
python scripts/predict.py --checkpoint outputs/<run>/checkpoints/best.pt \
  --split validation --mode diagnostic
```

自适应默认输出 `test_best_adaptive.csv`，一行对应一个分子，含 hard 预测与停止/执行深度。
diagnostic 输出 `validation_best_diagnostic_by_depth.csv`，含 y_k、q、p、survival、soft E[K]。
所有文件保存 checkpoint 哈希、split ID、CDF 阈值和执行模式。
旧 fixed checkpoint 仍输出原 `*_by_depth.csv`；`evalDepth.py` 仅汇总 fixed 模型。

M0 固定轨迹或 adaptive diagnostic 轨迹均可离线计算 Oracle：

```bash
python -m eqcompute.analysis.oracleDepth \
  outputs/<run>/predictions/validation_best_by_depth.csv \
  --checkpoint outputs/<run>/checkpoints/best.pt \
  --output outputs/<run>/analysis/oracle.csv

python -m eqcompute.analysis.complexity \
  --checkpoint outputs/<run>/checkpoints/best.pt --split validation \
  --output outputs/<run>/analysis/complexity.csv
```

Oracle 按 checkpoint 的训练集 scale 缩放绝对误差，仅输出误差最小的深度；
平局取最浅深度，另保存 normalization 和 scale。错误列为归一化绝对误差，
unit 列标记原目标单位。
检查哈希、性质、单位、深度覆盖和重复行；存在 split_ids.json 时还检查整套样本覆盖。
Oracle 使用标签，仅供分析，不能用于推理或 test 超参选择。

结构指标包括总原子数、重原子数、等权几何中心回转半径（坐标单位 Å）、元素种类数和
自然对数元素熵。暂不包含化学图直径、环数和可旋转键。EGNN 完全图不能用于解释
化学图距离或局部传播范围。

**验收**

```bash
python -m pytest -q
python -m ruff check src tests scripts
```

测试覆盖概率/手算损失、梯度、controller 注册、共享参数、对称性、单节点图、CDF 停止、
CPU/CUDA 非连续图压缩、实际算子工作量、阶段切换、断点恢复、配置归一化切换、
预测/分析 provenance，以及原有模型和固定深度回归。结果见 [验收记录](../VALIDATION.md)。
