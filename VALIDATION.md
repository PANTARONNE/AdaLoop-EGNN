# 验证记录

**2026-10-16：移除 computation cost**

训练仅保留停止概率加权预测误差；删除成本权重配置、日志和导出字段，以及成本最优
Oracle 和 CLI 参数。保留全部停止深度统计和分子结构复杂度；散点图保留两类深度，
深度分布图改为两行、8.5 × 6 英寸。smoke 保留 full/hard 一致性检查，删除成本汇总。

非 integration 测试：**182 passed、4 skipped、1 deselected**；4 项跳过均因 CUDA
不可用，1 项真实 QM9 integration 测试未选择。Ruff（src/tests/scripts/plot）和
`git diff --check` 通过。测试覆盖 MAE/MSE 手算损失、等误差深度的概率重分配、
controller 梯度与阶段切换、新格式恢复、导出、缓存、CLI 及 SVG/PNG 绘图。

本次测试通过绝对路径 `PYTHONPATH` 导入源码，并以 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
隔离环境中不相关的 pytest 插件。Arial 解压到 `/tmp/adaloop-font-data/fonts`，
测试设置 `XDG_DATA_HOME=/tmp/adaloop-font-data`、`MPLCONFIGDIR=/tmp/adaloop-matplotlib`；
保留原有 Arial 字体要求。未启动完整训练或修改既有实验结果。

> 以下为各记录日期的历史验收结果。自 2026-10-16 起，当前实现已移除训练计算成本项、
> 成本最优 Oracle 及 smoke 成本汇总；历史结果、路径和指标保留原样。
> 当前行为见 [AdaLoopEGNN.md](docs/AdaLoopEGNN.md)。

**2026-10-12：训练默认切换到本地 PyG QM9**

按用户要求，直接修改原配置的继承链；固定 EGNNOrig、Loop、M0 诊断和 AdaLoop
统一默认 dataset=qm9、root=data/qm9。保留 MAD 和配置切换能力，不建立 qm9_orig 备份配置。
正式 train() 解除骨架与 XYZ 数据协议的绑定；smoke 使用同一入口，不再单独组装 Trainer。
PyG 标签保持原始目标语义，不额外扣原子参考值；metadata 记录 dataset=qm9_pyg、raw 目标和单位。

验收：116 passed、1 skipped，Ruff 检查通过。新增配置检查覆盖全部五个入口，
小型端到端测试覆盖 PyG 数据分组、MAD/standard、checkpoint 加载及 hard/diagnostic 导出。
旧 XYZ 适配器的离线正确性测试仍通过；跳过项仍为未设置 QM9_ROOT 的已有 integration 测试。

再次通过正式 train() 完成 CUDA 本地真实数据 smoke：128/32/32 个分子，mu、MAD、
Kmax=4，3 epochs（1 warm-up + 2 adaptive）。Validation MAE=1.2337818108 D，
Test MAE=0.9804102145 D，RMSE=1.2473613851 D。full/hard 停止深度一致率 100%，
最大标准化预测差异 2.3841857910e-7；node-steps=2276/569，edge-steps=39632/9908。
短程模型全部在第 1 步停止，结果用于链路验收，不作正式性能结论。

运行目录：`outputs/qm9_mu_adaloop_orig_k4_lambda0.03_s0__20261012T090115_32b0a866/`。
该目录包含正式入口产生的 config/metadata/split/checkpoint、预测及 smoke_summary.json。

以下是切换前的历史验收记录。

**2026-10-12：单性质 AdaLoop（M0–M3）**

范围：M0 只做代码和验收测试，不启动完整诊断实验；M1–M3 完成实现、测试及一次本地
真实数据短程 smoke。EGNNOrig 消息传播代码不变，默认 MAD，配置切换 standard 已测试。
完整用法见 [AdaLoopEGNN.md](docs/AdaLoopEGNN.md)。

最终执行：

```bash
python -m pytest -q
python -m ruff check src tests scripts
python scripts/smokeAdaLoop.py --device auto
```

结果：**111 passed、1 skipped、0 failed**；Ruff 全部通过，`git diff --check` 无空白错误。
跳过项是已有真实 QM9 数据检查，因为本次 pytest 未设置 `QM9_ROOT`；独立真实数据 smoke
已实际完成。PyG 的 distributed 弃用提示仍存在。新增验收覆盖：

- Orig all_steps 诊断及 MAD / standard 配置，从训练到 checkpoint 再到导出。
- 停止分布边界、末步剩余质量、手算逐样本期望损失、梯度与 controller 参数注册。
- warm-up 不更新 controller、随机深度覆盖、切换到 adaptive 后更新。
- warm-up 无 best.pt 时的恢复，以及 adaptive 阶段恢复，与连续 CPU 训练逐位一致。
- 旋转、平移、反射、节点置换及批间隔离。
- CPU/CUDA 上非连续图提前退出、节点/边重编号、原始顺序回填。
- 构造分子分别在第 1/2/4 步停止时，后续实际进入 step 的节点/边数量减少。
- full 与 hard 输出一致、模式不会相互污染、单节点与 Kmax=1 边界。
- Oracle 的尺度、平局、惩罚、哈希、缺失/重复深度及整套 split 样本覆盖。
- 原有固定深度结果可继续汇总，自适应 checkpoint 不会被误报为固定深度结果。

真实数据 smoke 使用用户确认的本地 **PyG QM9** 缓存；它不是 Orig XYZ/原子参考值/划分
协议的研究实验。`mu`，训练/验证/测试 128/32/32 个分子，hidden_dim=32，Kmax=4，
3 epochs（1 warm-up + 2 adaptive），lambda=0.03，CDF threshold=0.5，MAD，CUDA。

| 指标 | 结果 |
|---|---:|
| Validation MAE | 1.2364565842 D |
| Test MAE | 0.9851845838 D |
| Test RMSE | 1.2444310733 D |
| full / hard 停止深度一致率 | 100% |
| full / hard 最大标准化预测差异 | 1.1920928955e-7 |
| 平均 hard 深度 | 1.0 |
| full / hard node-steps | 2276 / 569 |
| full / hard edge-steps | 39632 / 9908 |

该短程模型全部在首步停止，仅证明链路、数值一致性和实际计算跳过行为。
不据此声称已学到样本相关深度、达到基准精度或获得实测延迟提升。
批内不同停止时间的压缩由专门的 CPU/CUDA 构造测试覆盖。

运行目录：`outputs/adaloop_pyg_mu_smoke__20261012T045526_25e4826f/`，包含 config、metadata、
split IDs、metrics、best/last checkpoint、hard/diagnostic 预测及 `smoke_summary.json`。

**以下为原 Experiment 0 历史验收记录。**

验证日期：2026-10-06。范围为软件正确性与真实 QM9 小规模端到端运行。
没有执行完整数据集训练或正式多 seed depth sweep；下面的误差不能用于研究结论。

## 环境

- WSL2 Ubuntu，Miniconda 环境：`/home/wang_zy/miniconda3/envs/AdaLoopEGNN`
- Python 3.11.17
- PyTorch 2.3.1+cu121，CUDA runtime 12.1
- PyTorch Geometric 2.7.0，Hydra 1.3.2，NumPy 1.26.4
- NVIDIA GeForce RTX 4070，12 GB；CUDA 张量运算和模型反传均实际执行通过
- `pip check`：No broken requirements found
- 完整 Python 依赖见 `requirements-wsl.lock.txt`

原计划安装的 CUDA 12.8 PyTorch 下载速度较慢，改用本机已有完整缓存中的官方
CUDA 12.1 PyTorch wheel，安装到全新环境，没有修改已有环境。
其他依赖在官方 PyPI 连接出现 TLS EOF 后使用清华 PyPI 镜像安装。

QM9 来自 PyG 官方地址 `https://data.pyg.org/datasets/qm9_v3.zip`，没有 RDKit，使用
官方预处理版本。单连接传输较慢，改用同一 HTTPS URL 的分段下载；校验每段范围、大小及 ZIP CRC
后交给 PyG 解包处理。没有使用生成数据替代 QM9。

## 自动测试

```bash
QM9_ROOT=data/qm9 python -m pytest -q
ruff check src scripts tests
```

结果：**27 passed，0 failed，0 skipped**，耗时约 15 秒；Ruff 全部通过。
有一条 PyG 自身的 distributed 模块弃用提示，与本项目单设备训练无关。

已验证：

- Batch 形状、稳定 sample_id、划分互斥与可复现。
- QM9 真实数据加载、目标索引/单位、仅训练集拟合目标变换。
- tied/untied forward，手动逐步执行与 wrapper 输出一致。
- tied 参数对象共享、参数量不随执行次数增长、重复调用梯度回到同一算子。
- untied 参数独立，超出已有层数报错。
- trace 默认关闭，分析快照 detached，trace 不改变最终结果。
- CPU 双精度旋转、平移、反射、节点置换、批间隔离与空邻域。
- CUDA float32 对称性、多深度反向传播与有限梯度。
- final/all_steps 小图过拟合，all_steps 每个读出都有直接梯度。
- CPU epoch 边界恢复与不中断训练的模型状态逐位一致。
- 配置不兼容时拒绝恢复；样本加权指标；预测导出不重复且不静默覆盖。

## 真实 QM9 GPU smoke runs

公共配置：mu，seed=0，split_seed=42，128 train / 32 validation / 32 test，
hidden_dim=32，num_rbf=8，cutoff=5 Å，batch_size=32，AdamW lr=0.001。
目标标准化仅使用 128 个训练样本。下表 MAE 为最终深度输出，单位 D。

| 模式 | K | epochs | 参数量 | best validation MAE | 对应 test MAE |
|---|---:|---:|---:|---:|---:|
| tied / final | 3 | 3 | 11,521 | 1.128686 | 0.985946 |
| untied / final | 3 | 3 | 24,769 | 1.243128 | 0.995120 |
| tied / all_steps | 3 | 3 | 11,521 | 1.080957 | 1.002815 |

这只是短训练流程检查，不能据此比较方法优劣。
best checkpoint 仅按 validation 最终深度 MAE 选择；all_steps 的 best 来自 epoch 2。

对应输出：

- [tied / final](outputs/qm9_mu_egnn_tiedTrue_k3_final_s0__20261006T104641_eb159870/)
- [untied / final](outputs/qm9_mu_egnn_tiedFalse_k3_final_s0__20261006T104653_4b55aafa/)
- [tied / all_steps](outputs/qm9_mu_egnn_tiedTrue_k3_all_steps_s0__20261006T104725_856e5dba/)

另实际运行了 Hydra multirun：

```bash
python scripts/train.py -m experiment=qm9_smoke trainer.device=cuda \
  'model.weight_tying=true,false'
python scripts/train.py -m experiment=qm9_smoke trainer.device=cuda trainer.epochs=1 \
  'model.execution.depth=1,2'
```

K=1、2 的 tied 模型参数量均为 11,521，所有任务使用同一 split ID，并产生独立输出目录。
这些运行验证了 sweep 入口，不是正式 fixed-depth 对比实验。

## 预测与恢复验收

已从磁盘 checkpoint 重新构造模型，独立执行 `scripts/predict.py`，导出：

- tied/final：validation 和 test 各一份。
- untied/final：test 一份。
- tied/all_steps：test 一份。

四份 CSV 均通过以下检查：每份 96 行；每个深度恰好包含相同 32 个样本；深度集合为
{1, 2, 3}；无重复 sample/depth；sample_id 与保存的 split manifest 一致；checkpoint SHA256
与实际文件一致；预测有限且单位为 D。
验证集导出 MAE 为 1.1286861595，与 checkpoint 的验证指标在 1e-5 内一致。

[all_steps 逐深度 CSV](outputs/qm9_mu_egnn_tiedTrue_k3_all_steps_s0__20261006T104725_856e5dba/predictions/test_best_by_depth.csv)

[离线绘制的深度误差曲线](outputs/qm9_mu_egnn_tiedTrue_k3_all_steps_s0__20261006T104725_856e5dba/predictions/depth_curve.png)

已从 tied/final 的 `last.pt` 在 CUDA 上恢复并训练到 epoch 4，新 `metrics.csv` 仅包含 epoch 4。
epoch 4 未改善验证误差，原 best.pt 被完整保留，SHA256 与来源文件完全一致。
[恢复运行](outputs/qm9_mu_egnn_tiedTrue_k3_final_s0__20261006T104830_3792dbe2/)

首个 smoke run 发现 Hydra 禁用了提前创建的训练模块 logger：指标与 checkpoint 正常，但 epoch
日志缺失。已修复 logger 初始化；上面列出的后续运行已确认每个 epoch 同时写入日志和 CSV。
初始诊断运行目录保留，不纳入上表。

## 当前边界

- 只支持单设备标量回归；不包含 force loss 或多目标联合训练。
- radius graph 的逐分子 O(N²) 实现面向 QM9；大分子需替换构图实现。
- 仅支持 epoch 边界恢复；CUDA reduction 不保证跨设备/版本逐位一致。
- 没有实现 adaptive halting、AdaLoop、Oracle、复杂度分析、计算惩罚、QMugs、PaiNN 或 Matbench。

## 多 GPU 深度调度验收

新增 `scripts/sweepDepth.py` 与独立的 `eqcompute.launching.GPUQueue` 模块。
现有训练入口、模型、Task、Trainer 和 checkpoint 实现没有修改，也没有新增依赖。

`python -m pytest -q`：**45 passed，1 skipped**；Ruff 全部通过。
跳过项为未设置 `QM9_ROOT` 的可选真实数据集集成测试；另单独完成了下述真实数据训练。
新增 19 项测试覆盖 depth × seed 展开、必填 tying、配置覆盖、重复与越界参数、
已有 CUDA 可见范围、无副作用 dry-run，以及真实子进程执行中的 GPU 槽位复用、
失败后继续派发、启动失败记录和 SIGTERM 对训练进程及其子进程的清理。
并发调度测试使用普通 Python 子进程模拟 GPU 任务，不等于真实多 GPU 验证。

本机 WSL / RTX 4070 使用已有 QM9 缓存，实际运行：

```bash
python scripts/sweepDepth.py --gpus 0 --depths 1,2 --seeds 0,1 \
  --weight-tying false -- experiment=qm9_smoke trainer.epochs=1 \
  output_root=outputs/sweep_validation
```

结果：4 个任务全部成功，调度器退出码 0；均使用 `cuda:0`，每个任务拥有独立日志、
单 epoch 指标和非空 `best.pt` / `last.pt`。四个任务 split ID 相同；
K=1 的参数量为 11,521，K=2 为 18,145。这里只验证流程，不比较模型效果。

[该批次任务清单](outputs/sweep_validation/_depth_sweeps/20261007T161910_f4d78d87/jobs.json)。
目标单机 Linux / 8 × RTX 4090 当前无法访问，尚未执行真实 8 卡并行训练。
可在目标服务器先运行 README 中的双卡 smoke 命令，再启动正式 sweep。
