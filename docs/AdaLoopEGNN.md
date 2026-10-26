# 单性质 AdaLoop-EGNN（M0–M3）

消息传播使用 `models/backbones/EGNN.py` 中的 `EGNNStep`，完全图、固定坐标、共享参数。
固定深度基线支持独立参数和共享参数；AdaLoop 复用同一消息传播计算。
本轮实现单性质模型，不包含 property embedding 或多性质联合训练。

**范围与配置**

| 配置 | 用途 |
|---|---|
| `qm9_loop_egnn_diagnostic` | M0：共享参数、Kmax=12、所有深度等权 MAE 监督 |
| `qm9_adaloop_egnn_predictor` | 预测器训练，1000 epochs，hidden_dim=128、Kmax=24 |
| `qm9_adaloop_egnn_halting` | 冻结预测器，仅训练停止器，500 epochs |
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

停止器阶段通过 `task.halting_loss` 选择 `expected` 或 `soft_cdf`，再加原停止分布的熵正则。
停止器预设 `qm9_adaloop_egnn_halting` 默认使用 `soft_cdf`；预测器预设保留 `expected`，
但预测器阶段不使用这项设置。缺少新字段的旧配置按 `expected` 解释。
标签先转换为
$\tilde y_b=(y_b-m)/s$，其中 $m$、$s$ 仅在训练集拟合。默认 `mad` 使用训练均值 $m$
和平均绝对偏差 $s=\operatorname{mean}_{\mathrm{train}}|y-m|$
（这里不是中位数绝对偏差）；`standard` 使用总体标准差，二者的 scale 均下限截断为
$10^{-8}$；`none` 使用 $m=0,s=1$。所有深度的 $\hat y_{b,k}$ 都在同一标签尺度上监督，
报告 MAE 时再将所选预测还原到原目标单位。

对于含 $B$ 个分子的 batch，默认 `task.loss=mae`，每层误差为
$\ell_{b,k}=|\hat y_{b,k}-\tilde y_b|$。`expected` 模式保留原概率期望目标：

$$
\mathcal{L}_{\mathrm{expected}}
=\frac{1}{B}\sum_{b=1}^{B}\sum_{k=1}^{K}p_{b,k}\ell_{b,k},
$$

`soft_cdf` 模式对 CDF 首次越过阈值的位置做可微近似。设
$c=\texttt{model.execution.threshold}$、$\tau=\texttt{task.soft_cdf_temperature}$，则：

$$
F_{b,k}=\sum_{j=1}^{k}p_{b,j},\qquad
A_{b,0}=0,\quad A_{b,K}=1,\quad
A_{b,k}=\sigma\!\left(\frac{F_{b,k}-c}{\tau}\right)\quad (1\le k<K),
$$

$$
w_{b,k}=A_{b,k}-A_{b,k-1},\qquad
\mathcal{L}_{\mathrm{softCDF}}
=\frac{1}{B}\sum_{b=1}^{B}\sum_{k=1}^{K}w_{b,k}\ell_{b,k}.
$$

CDF 单调，因此 $w_{b,k}\ge0$ 且 $\sum_kw_{b,k}=1$。$K=1$ 时 $w_{b,1}=1$。
温度默认为固定值 `0.1`，必须有限且严格为正，不能是布尔值；当前不自动退火。
当没有 $F_{b,k}=c$ 的精确平局时，$\tau\to0$ 会趋近硬 CDF 所选层的一热权重。
精确平局时 sigmoid 仍为 $0.5$，这一平滑近似不等于硬推理 `>=` 所采用的最浅层规则。
温度过小可能导致梯度饱和，`0.1` 仅是可配置的起始值，不保证最优。

`prediction_loss` 使用上述选定的目标 $\mathcal{L}_{\mathrm{prediction}}$；总损失为：

$$
\mathcal{L}
=\mathcal{L}_{\mathrm{prediction}}-\beta H,\qquad
H=-\frac{1}{B}\sum_{b=1}^{B}\sum_{k=1}^{K}p_{b,k}\log p_{b,k}.
$$

默认固定系数 `task.entropy_beta=0.01`，可通过 `task.entropy_beta=0` 关闭。
系数必须为有限、非负数，不能是布尔值。熵针对每个样本的完整停止分布 $p$ 计算，
再按 batch 平均；不是条件停止概率 $q$ 的熵、soft-CDF 权重 $w$ 的熵，
也不是 batch 平均分布的熵。
使用自然对数，不除以 $\log K$；计算采用 `p * log(p.clamp_min(1e-12))`，
零概率项贡献为零。熵项保持梯度，日志统计才 detach。
`0.01` 是待验证的实验起始值，不保证消除停止分布坍缩。

`task.loss=mse` 时仅将 $\ell_{b,k}$ 换为平方误差，停止分布保持相同定义。
单性质预测张量形状为 `[B,K,1]`，标签为 `[B,1]`，停止质量为 `[B,K]`；
实现先计算每个样本、每个深度的误差，再沿深度按所选权重加权，最后沿 batch 求均值：

```text
weights = p if halting_loss == "expected" else soft_cdf_weights(p, threshold, temperature)
prediction_loss = mean_b sum_k weights[b,k] * abs(y_k[b]-normalized_target[b])
expected_prediction_loss = mean_b sum_k p[b,k] * abs(y_k[b]-normalized_target[b])
halting_entropy = -mean_b sum_k p[b,k] * log(max(p[b,k], 1e-12))
entropy_loss = -entropy_beta * halting_entropy
loss = prediction_loss + entropy_loss
```

两种模式都对**每层误差加权**，不是先混合预测再计算误差
$|\sum_k w_{b,k}\hat y_{b,k}-\tilde y_b|$。例如目标为 2、两步预测为 0 和 4、
权重各为 0.5 时，加权 MAE 为 2，而混合预测的 MAE 为 0；
后者可能通过不同深度误差抵消，无法反映实际选择某一步输出时的误差。

各深度误差相同时，调整停止质量不会改变 prediction_loss，但正系数的熵正则偏好
更高熵的停止分布。总损失可能为负，不代表预测误差为负；比较预测质量应看 MAE
和 prediction_loss；跨损失模式比较时还可查看 expected_prediction_loss。
比较不同实验时需注明损失模式、温度、normalization 和熵正则系数。

停止器阶段完整展开到 $K$，不采样离散停止深度，也不通过硬 CDF 的布尔比较或整数选层反向传播。
预测器（initializer、共享 backbone、readout/head）全部冻结，使用 eval 模式；
`expected` 通过 $p_{b,k}$ 反传，`soft_cdf` 通过 $p\to F\to A\to w$ 的平滑路径反传到 controller。
Controller 根据固定预测器的各深度误差分配停止质量，
无需 Oracle 停止标签或 REINFORCE。熵正则也仅更新 controller，当前目标不包含计算成本。

预测器阶段每个 batch 均匀抽取一个深度 $k^\star\sim\mathrm{Uniform}\{1,\ldots,K\}$，
整个 batch 执行到该深度，使用
$\mathcal{L}_{\mathrm{predictor}}=\frac{1}{B}\sum_b\ell_{b,k^\star}$。
此阶段不调用 controller，不加入熵正则，仅训练预测器。
验证遍历全部深度，也不调用 controller；validation loss 为全部深度的归一化误差平均值。

预测器的 `validation_mae` 为原单位下的逐分子最优深度误差：

$$
\mathrm{oracle\ MAE}=\frac1N\sum_b\min_{1\le k\le K}|s\hat y_{b,k}+m-y_b|.
$$

并列时取较浅深度。该值用于保存预测器 `best.pt`，它使用真实标签选层，
是理想选层误差下限，不是实际可部署的推理性能。
同时记录 `validation_mean_depth_mae`（所有分子、所有深度平均 MAE）、
`validation_final_depth_mae`（末层 MAE）及 `validation_oracle_depth`。
停止器阶段 validation 使用 `full` 模式和上述带熵正则的 loss，
MAE 则使用 CDF 首次达到阈值的单层预测，不含熵项，也不是概率加权 MAE。

实现见 [adaptive_loss 与训练阶段](../src/eqcompute/tasks/adaptiveRegression.py)、
[标签变换](../src/eqcompute/tasks/targetTransform.py)；
[手算损失与梯度测试](../tests/testHalting.py) 验证停止质量、逐样本加权和梯度。

**训练与停止语义**

使用 `task.phase=predictor` / `halting` 的两个独立训练配置，分别默认 1000 / 500 epochs。
每阶段使用新的优化器及独立 cosine 周期，起始学习率沿用对应属性的现有设置。
不再自动切换阶段，不保留旧联合训练模式或 `task.warmup_epochs`。
停止器训练与验证完整展开，训练过程不声称节省消息传播计算。

推理选择第一个 `sum_{j<=k}p_j >= threshold` 的深度，输出该步预测。
已停止分子的节点和边从后续消息传播中移除；不是继续全 batch 计算后仅冻结特征。
未达到阈值的分子在 Kmax 强制退出。

| 模式 | 用途 | 计算 |
|---|---|---|
| `random_depth` | 预测器随机深度训练 | 执行到抽中的深度 |
| `predictor_full` | 预测器验证 | 全部执行，用预测 trace 收集各层输出，不调用停止器 |
| `full` | 停止器训练和 validation 损失 | 全部执行；同时按 CDF 选择 hard 输出计算 MAE |
| `diagnostic` | 离线轨迹分析 | 全部执行，提供逐深度预测和概率 |
| `hard` | 实际推理 | 批内压缩 active graphs，必须在 eval + no_grad/inference_mode 下运行 |

普通 `model.eval()` 推理默认 hard。完整 trace 必须显式指定 diagnostic；打开 trace 不会
悄悄改变计算协议。full/diagnostic 总是保留全部逐层预测用于损失，不受 trace 深度选择影响；
trace 的 depth 选择只控制 detached 表示快照。hard 不伪造未执行的未来预测或完整停止分布。

**训练示例**

```bash
python scripts/train.py experiment=qm9_adaloop_egnn_predictor task.target=mu
python scripts/train.py experiment=qm9_adaloop_egnn_halting task.target=mu \
  init_from=outputs/<predictor_run>/checkpoints/best.pt
```

`init_from` 必须指向预测器阶段的 checkpoint，可指定 `best.pt` 或 `last.pt`。
只迁移预测器权重及归一化参数，不复制停止器、优化器、scheduler、epoch 或历史最佳指标。
来源阶段、模型配置、目标属性、归一化设置和数据划分必须匹配；
输出 metadata 记录来源路径与 SHA256。停止器无 `init_from` 或 `resume` 时直接报错。
例如关闭停止器熵正则可在第二条命令追加 `task.entropy_beta=0`。
停止器默认使用 soft-CDF，可追加 `task.soft_cdf_temperature=0.05` 指定固定温度。
对照原期望损失时，使用：

```bash
python scripts/train.py experiment=qm9_adaloop_egnn_halting task.target=mu \
  task.halting_loss=expected \
  init_from=outputs/<predictor_run>/checkpoints/best.pt
```

本地真实 QM9 短程验收：

```bash
python scripts/smokeAdaLoop.py --data-root data/qm9 --device auto
```

脚本要求已有 `processed/data_v3.pt`，不下载数据。使用 mu、MAD、128/32/32 个分子，
分别训练预测器 1 轮、停止器 2 轮，产生两个独立运行目录。
检查冻结的预测器状态完全一致，并从停止器 `best.pt` 重建模型执行 full/hard 对照。
汇总记录预测器目录及 oracle MAE、停止器 MAE 和预测一致性。
可传 `--normalization standard`。单独使用 `qm9_adaloop_egnn_smoke` 配置默认只训练预测器。

**Checkpoint 与日志**

`last.pt` 每 epoch 保存 model/controller、optimizer、scheduler、task 阶段、transform 和 RNG。
两个阶段均从第一轮起选择 `best.pt`：预测器按 oracle validation MAE，停止器按 CDF MAE。
同阶段恢复使用对应配置，保留原运行的配置覆盖参数，例如：

```bash
python scripts/train.py experiment=qm9_adaloop_egnn_predictor \
  resume=outputs/<predictor_run>/checkpoints/last.pt
python scripts/train.py experiment=qm9_adaloop_egnn_halting \
  resume=outputs/<halting_run>/checkpoints/last.pt
```

`resume` 与 `init_from` 互斥。Checkpoint 保存损失类型和温度。
恢复时保持模型、数据、task（含 phase、entropy_beta、halting_loss 和 soft_cdf_temperature）、
优化器及 cosine 周期一致；恢复停止器时不需要重新提供第一阶段文件。
缺少新字段的旧 checkpoint 按 `halting_loss=expected`、`soft_cdf_temperature=0.1` 读取。
不能将旧停止器 checkpoint 通过 `resume` 静默切换为 soft-CDF；
从 predictor checkpoint 通过 `init_from` 启动新的 soft-CDF 停止器训练则允许。
旧联合训练 checkpoint 不兼容，不实现从旧 Loop 权重迁移。

`metrics.csv` 的指标按样本数加权。两个阶段分别记录在各自运行目录中。
总损失满足 `loss = prediction_loss + entropy_loss`，预测器阶段 entropy_loss 恒为 0。
停止器的 `prediction_loss` 随损失模式变化；`expected_prediction_loss` 始终是原停止质量 $p$
下的归一化期望误差，不等于实际硬 CDF MAE。后者仍用于选择停止器 `best.pt`。
停止器额外记录 expected_depth、halt_depth、executed_depth、首步/末步停止比例与分布熵。
expected_depth 是 soft 期望，halt_depth 是 CDF 选择结果，executed_depth 是实际执行长度。
两个阶段的 validation 都完整展开，因此 executed_depth=Kmax。

**预测、Oracle 与结构指标**

```bash
python scripts/predict.py --checkpoint outputs/<run>/checkpoints/best.pt --split test
python scripts/predict.py --checkpoint outputs/<run>/checkpoints/best.pt \
  --split validation --mode diagnostic
```

上述停止推理示例应使用停止器阶段 checkpoint；预测器阶段的停止器尚未训练。
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
