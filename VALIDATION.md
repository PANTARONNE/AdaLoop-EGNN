# Experiment 0 验证记录

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
