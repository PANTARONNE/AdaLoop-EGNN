# AdaLoop-EGNN 实现计划：复用 EGNNOrig

> 2026-10-19 清理说明：第一步已移除旧半径图 EGNN 与 QM9Orig XYZ 预处理，
> 第二步已将保留的 EGNNOrig 骨架统一更名为 EGNN，数据使用 PyG QM9。
> 下文保留当时的名称与设计，不作为现有文件清单。当前入口见 [EGNN.md](EGNN.md)。

> 历史设计记录（2026-10-12）。自 2026-10-16 起，当前实现已移除训练计算成本项、
> 成本权重配置及成本最优 Oracle；下文相关公式与参数仅记录原设计。
> 当前行为以 [AdaLoopEGNN.md](AdaLoopEGNN.md) 为准。

编写日期：2026-10-12。依据：仓库实现，以及 `Experiment plan.pdf`，重点为第 3、7–10、14–16、19–20、23 页。本文保留完整路线图；当前实施范围已收敛为 M0–M3 单性质模型，使用说明见 [AdaLoopEGNN.md](AdaLoopEGNN.md)。

用户确认的执行范围：M0 只做代码和验收测试，不启动完整实验，不以科研结果作为 M1–M3 的实现关卡；M1–M3 加一次本地真实数据短程 smoke。默认沿用 Orig 的 MAD，保留配置切换。用户随后确认正式训练也直接使用本地 PyG QM9，不新增 qm9_orig 备份配置；固定基线、Loop、M0 诊断和 AdaLoop 默认共用 PyG 数据协议。M4–M5 留待后续。

**1. 总体方案与边界**

保留 `EGNNOrigInitializer → 共享 EGNNOrigStep → EGNNOrigReadout → prediction head`，在现有执行策略层增加 AdaLoop，在 Task 层增加概率加权预测损失与计算惩罚。Trainer 继续只管理 epoch、优化、统计和 checkpoint，不进入逐深度循环。

按以下顺序交付：

1. 补齐 EGNNOrig 的逐深度诊断训练和 Oracle 分析，确认实验动机。
2. 实现单性质 AdaLoop，跑通可微概率、两阶段训练和停止预测。
3. 实现真正跳过已停止图的推理，并验证与完整轨迹模拟一致。
4. 在同一个 checkpoint 中实现多性质条件化，完成 PDF 主模型及 conditioning 消融。
5. 补齐预算匹配基线、对称性、平滑度和效率评估。

第 2 步是最小可用版本，第 4 步才具备 PDF 中完整的 property-conditioned 含义。QMugs、PaiNN、Matbench、size OOD 和测试时增加深度独立排在后续。

**2. PDF 与当前代码的对应及取舍**

| 项目 | PDF 建议 | 当前代码 | 本计划 |
|---|---|---|---|
| 消息传播 | 共享 EGNN operator | `EGNNOrigStep` 已支持权重共享 | 完整复用 |
| 最大深度 / 隐藏维度 | 12 / 128 | 默认 7 / 128 | 新实验设置 12 / 128，旧 preset 不变 |
| 图与几何 | 32 RBF、5 Å cutoff；公式含坐标状态 | 完全有向图、无自环、距离平方、坐标固定 | 以用户指定 EGNNOrig 为准，不引入 RBF、cutoff 或坐标更新 |
| 每步读出 | 共享、可预测每个深度 | 已有共享 readout/head 和 predictions trace | 复用读出，补训练协议 |
| 性质条件 | head 与 controller 使用 property embedding | `PropertyContext.name` 只是单目标名称校验 | 新增真正的条件 head / embedding，多性质共享训练 |
| 停止概率 | hazard → 停止分布，末步强制停止 | 无 | 新增 controller + execution |
| 标签缩放 | 训练集均值 / 标准差 | Orig preset 使用均值 / MAD | 本轮按用户要求默认 `mad`，可配置切换 `standard` |
| 数据划分 | 110000 / 10000 / remaining | 原有 PyG 适配器支持该划分 | 使用本地 PyG 缓存及 split_seed=42，不额外做原子参考值扣除 |
| 推理决策 | 没有明确具体规则 | 无 | 明确新增 CDF 阈值规则，默认 0.5，属于工程选择 |
| warm-up | 前 20% 随机深度，后 80% 自适应损失 | 无 epoch 通知接口 | 增加通用 task epoch hook |

新比较组的 fixed / loop / adaptive 必须统一数据、标签处理、归一化、优化器及训练预算。若切换为 standard，需要统一相关对照设置。当前默认采用 PyG 的 110k/10k/remaining 划分；样本 ID 与 split_seed 写入 manifest，相关基线统一该协议。

**特别限制：EGNNOrig 的传播图是完全图，图直径对多原子图恒为 1。** 增加循环主要增加迭代特征变换能力，不能直接解释为扩大 k-hop 可达范围。PDF 的 underreaching 与长程合成任务不能照搬成这一骨架下的因果证据；本轮只研究误差、结构复杂度与迭代次数的关系。

**3. 已有能力与明确需要改动的位置**

| 文件 / 组件 | 已有能力 | 计划 |
|---|---|---|
| `models/backbones/EGNNOrig.py` | 初始化、完整图、消息门控、sum 聚合、残差、固定坐标、节点解码 | 不修改核心计算 |
| `models/operators.py` | tied 模式只注册一个 step | 直接复用；adaptive 必须 tying=true |
| `models/iterative.py` | 注入 execution 的统一模型包装 | 扩展 `build_egnn_orig`；forward 仅追加可选关键字，旧调用不变 |
| `models/execution.py` | 固定深度执行 | 保留原实现；新策略放新文件 |
| `models/trace.py` | 可微逐深度预测、CPU detached 分析快照 | 在 ModelOutput 尾部添加可选 adaptive 信息；保持原三参数构造兼容 |
| `tasks/regression.py` | final / all_steps、标准化、物理单位恢复 | 旧任务保持不变；新增 adaptive task |
| `application.py` | 数据 / 模型 / task 组装 | 替换 Orig 必须 final 的硬限制，按 execution/task 组合校验 |
| `training/trainer.py` | epoch、优化、恢复、MAE 选模 | 只加通用 epoch hook、额外指标聚合和可配置监控 |
| `evaluation/predictionWriter.py` | 一个完整轨迹导出所有深度 | 保持原格式；自适应结果使用独立导出文件 |
| `evaluation/depthSweep.py` | 固定深度汇总，只取最终训练深度预测 | 保留用于固定基线；拒绝将 adaptive 结果按此语义汇总 |
| `data/QM9Orig.py` | 12 性质标签、严格缓存、单位与原子参考处理 | 单性质阶段不改数据；多性质仅补元数据适配，不改缓存 |

注意 `QM9OrigDataset` 本身没有提供化学键。不能从 `state.edge_index` 推出化学图直径、环数或可旋转键数。

**4. 主模型的计算定义**

输入沿用 `GraphInputs`，模型不接收标签或事后复杂度指标。每次 forward 按 batch 的图独立维护停止状态。

对深度 k=1,…,Kmax：

```text
state_0 = EGNNOrigInitializer(graph)
state_k = shared_EGNNOrigStep(state_{k-1}, graph)
z_k = EGNNOrigReadout(state_k, graph)
y_k = head(z_k, property_context)
```

预测端保留 EGNNOrig 的 node MLP + sum pooling。Controller 另从原始 `state.node_features` 计算每图统计，不能直接拿经过 node decoder 的 sum readout 代替 PDF 的 mean/std。

```text
mean_k[g]  = mean(h_k[i] for i in graph g)
std_k[g]   = sqrt(mean((h_k[i]-mean_k[g])**2) + eps)
delta_k[g] = mean(norm(h_k[i]-h_{k-1}[i], 2) for i in graph g)
s_k[g]    = concat(mean_k[g], std_k[g], delta_k[g], property_embedding)
q_k[g]    = sigmoid(halt_mlp(s_k[g]))
```

使用 population variance，即 `unbiased=False` 的语义；方差在开方前保护，单节点图也应有限。第 1 步的 delta 对比 embedding 后的 h0。所有统计只在各自图内计算。输入维度为 `2*hidden_dim + 1 + property_embedding_dim`；128 维特征、16 维性质嵌入时为 273。性质嵌入维度 16、controller 隐藏维度 64 可作为首轮工程默认值，不是 PDF 已指定的超参。

不输入原始坐标、graph diameter、atom count、ring count、标签、真实误差或 Oracle depth。不额外输入深度位置编码，以保持 PDF 机制并避免后续测试时扩深的额外限制。

停止分布按 survival 递推：

```text
S_1 = ones(B)
for k in 1..Kmax-1:
    p_k = S_k * q_k
    S_{k+1} = S_k * (1 - q_k)
p_Kmax = S_Kmax
q_Kmax = ones(B)
```

这样末步接收全部剩余质量；不要将末步也简单乘可学习 q 后再整体归一化。Kmax=1 时 p1=1，controller 不具备可学习的提前停止选择，这是正确边界行为。

梯度沿预测、controller 统计及所有共享循环传播；主实验不 detach h、delta 或 p。仅写分析快照、日志和指标时 detach。半精度优化后置；概率计算首先使用 FP32。

**5. 模块所有权与输出接口**

新增 `models/halting.py` 和 `models/adaptiveExecution.py`：

- `HaltingController(nn.Module)`：图级统计和停止 logit，不定义训练损失。
- `AdaptiveExecution(nn.Module)`：共享循环、概率、训练 warm-up 和推理停止。

`AdaptiveExecution` 必须是注册的 `nn.Module`，并拥有 controller；否则沿用当前普通 Python execution 对象的写法，会使 controller 参数漏出 `model.parameters()`、设备迁移和 checkpoint。

Controller 在调用 `readout/head` 时只使用传入引用，不把同一 head 再注册成自己的子模块。性质 embedding 由新的条件 head 单独持有，并提供获取 embedding 的方法，controller 使用同一个 e_p，避免重复参数所有权。

建议在 `ModelOutput` 增加 `adaptive: AdaptiveInfo | None = None`。以下是拟定字段，不是现有 API：

| 字段 | 形状 / 含义 | 是否保留梯度 |
|---|---|---|
| `halting_probabilities` | [B,K]，条件概率 q | 训练完整轨迹保留 |
| `stop_probabilities` | [B,K]，停止分布 p | 是 |
| `expected_depth` | [B]，sum(k*p_k) | 是 |
| `halt_depth` | [B]，离散停止深度 | 否 |
| `executed_depth` | [B]，该次 forward 实际执行步数 | 否 |
| `valid_depth_mask` | [B,K]，哪些位置真实执行过 | 否 |
| `execution_mode` | warmup / full / hard / diagnostic | 元数据 |

训练 full 模式需要的 y_k 始终可微，不能依赖用户是否打开分析 trace。可复用现有 `predictions_by_depth` 字典提供它们；hard 模式只提供已执行位置并配有效掩码，未执行位置不能填成看似真实的预测或停止概率。

`ModelOutput.prediction` 的含义固定：warm-up 为抽中深度的预测；完整自适应执行为按指定 hard rule 从轨迹选择的预测；真实 hard 推理为停止时输出。概率混合预测仅作为可选分析项，不偷偷代替停止预测。

执行模式不能由 `TraceSpec(predictions=True)` 暗中决定。建议给 `IterativeModel.forward` 追加可选关键字 `execution_mode=None`，只在显式提供时转发给支持它的新 execution；固定策略收到不支持的模式时清楚报错，普通旧调用完全不变。adaptive 的默认模式按 `self.training` 和 warm-up 阶段选择：训练为 warmup/full，eval 为 hard；离线分析显式传 `execution_mode="diagnostic"`。AdaptiveTask 的 predict 同步接受该可选参数，旧 Task 无需更改。

validation 需要期望损失与 hard 指标时，由 AdaptiveTask 显式请求 full，一次展开后既算训练目标又选择 hard 输出；独立性能评估显式请求 hard，不计算未执行深度。诊断模式强制完整展开但不改变模型参数或永久推理配置。测试中验证这些模式切换不会污染后续调用。

**6. 自适应损失与 warm-up**

新增 `tasks/adaptiveRegression.py`，尽量继承 `RegressionTask` 的 targets、predict、transform 和保存逻辑。注册新 task 名 `adaptive_regression`，不要给旧任务塞多个互相冲突的监督开关。

对每个样本先得到 unreduced 误差，再按 p 加权，最后做 batch mean：

```text
y_normalized = transform.encode(y_true)
error[b,k] = abs(y_pred[b,k] - y_normalized[b])
L_pred = mean_b(sum_k(p[b,k] * error[b,k]))
L_compute = mean_b(sum_k(p[b,k] * k / Kmax))
L_total = L_pred + lambda_compute * L_compute
```

默认 MAE，沿用现有损失约定。不能写成“先对 batch 求 MAE，再乘平均 p”，因为那会丢掉逐样本分配深度的意义；也不能写成 `MAE(sum(p*y_k), y)`，它与 PDF 的期望损失不同。

所有 transform 只用训练标签拟合。`lambda_compute ∈ {0,0.01,0.03,0.1}`。计算惩罚中的 k/Kmax 是深度代理，并不等于真实 FLOPs；默认先保持 PDF 定义，按图规模加权 FLOPs 惩罚留作额外消融。

训练总阶段边界由显式 `warmup_epochs` 固定，默认在新实验中设为总训练预算的 20%。不要在恢复时根据新 `trainer.epochs` 重新计算边界。

| 阶段 | 执行 | 损失 | Controller |
|---|---|---|---|
| warm-up | 每个 batch 抽一个均匀 K∈[1,Kmax]，只运行到 K | 只监督 y_K | 不调用，参数无梯度 |
| adaptive | 运行到 Kmax | 概率加权 MAE + 计算惩罚 | 联合训练 |
| validation | 固定评估规则，无随机 warm-up | 可报告期望损失 | hard 输出用于部署精度评估 |

每 batch 共享随机 K 是减少改动的选择，在期望意义上仍覆盖全部深度；无需第一版就做每样本不同随机训练深度。warm-up 后训练仍完整展开，不宣称节省训练 FLOPs。

Trainer 新增可选 `task.on_epoch_start(model, epoch)` hook；task 在该 hook 内设置自身阶段与 execution 阶段。旧任务没有 hook 时行为不变。随机深度使用已保存状态的 Torch RNG，阶段状态写入 task checkpoint，并在恢复时验证。

选模首版继续以 hard validation MAE 为主，lambda 控制形成不同候选；同时完整记录平均 hard K 和期望 K。实际 Pareto 评估以实测计算量为准。如果需要预算约束选模，作为显式新 monitor 协议，不能隐含改变原有 `best.pt` 语义。

warm-up 的随机/未训练 controller 输出不参与最终 best 选择。warm-up 只保证 last checkpoint 可恢复；允许尚无 best.pt 的合法恢复状态。进入 adaptive 阶段后第一次有效验证建立 best.pt，之后正常比较。这需要小幅修改当前 `restore()` 无条件复制 best.pt 的行为，并增加对应测试。

单性质训练保持 Adam、按性质学习率、现有 epoch cosine 及无默认 gradient clipping；先观察数值稳定性，有证据再增加裁剪等变体。

**7. 推理停止规则与真正减少计算**

PDF 定义了分布，却没有定义确定性测试时选择。建议主协议采用：

```text
F_k = sum_{j<=k} p_j = 1 - S_{k+1}
K_halt = first k such that F_k >= tau
if none: K_halt = Kmax
y_output = y_{K_halt}
```

tau 默认 0.5，确定后写入配置和预测元数据；调参只能用 validation。它是停止分布的分位数规则，不是 q_k≥tau，也不是 argmax p_k。argmax 需要未来概率，不宜作为能在线停止的默认规则。

CDF hard 决策与训练的概率期望不完全一致，必须分别报告 soft E[K]、hard 平均 K、hard MAE。可增加按 p 采样的评估作为期望目标一致性检查，但不替换确定性主结果。不假设改变 lambda 一定单调改变深度，也不预先声称 lambda=0 必然全部到 Kmax。

推理分两次实现，先验证数学语义，再优化实际执行：

1. 完整展开到 Kmax，根据 CDF 选择每图输出。此时 halt_depth 可以较浅，但 executed_depth 仍全为 Kmax，只能用于 correctness reference。
2. 维护 active graph/node 集，每一步只对尚未停止的图运行 EGNNOrigStep，已停止图回填原 batch 位置并移出后续计算。

第二步放在新增的 `models/activeGraphs.py`，不改 `EGNNOrigStep`：

- 初始化时构图一次，持有原始 graph ID / node ID 映射。
- 裁剪时筛选整个分子，保留当前 h，筛选 positions、atomic_numbers 和图归属。
- 使用旧 node→新 node 映射重编号 edge_index 和 batch_index，设置 active num_graphs。
- 同步筛选上一时刻 h 或 delta 需要的状态，不能错配分子。
- 对可选输入 edge_index/edge_attr 同步筛选，或明确只支持 Orig 并拒绝不受支持输入；不静默保留错位边。
- 每次将停止分子的输出、K、sample 映射写回原顺序。
- active 集为空时立即结束；Kmax 时强制回填剩余图。

不要重跑 initializer 或重新从原始 embedding 开始。不要只在全 batch 上计算后用 mask 冻结特征，这种做法不会省掉主要 MLP 计算。

在线 hard 模式不为已停止图伪造剩余概率质量；完整 p / E[K] 仅从 full diagnostic 得到。benchmark 禁用完整 trace、Oracle 和未来预测。batch=1 的简单 break 可用于中间验收，但不是最终 batched efficiency 实现。

**8. 性质条件化：如何在最小改动下补齐主模型**

现有 `PredictionHead` 只允许一个 property name，分别训练 mu / alpha / gap / U0 只能说明不同任务的深度分布有差异。固定任务里 e_p 是常数，因此不能用这种配置论证“输入性质条件有效”。

主模型完成版需新增 `PropertyConditionedHead`：

```text
property_names = [mu, alpha, gap, U0]
e_p = embedding(property_index)
y_k = shared_mlp(concat(EGNNOrigReadout(h_k), e_p))
q_k = halt_mlp(concat(mean_k, std_k, delta_k, e_p))
```

Initializer、step、readout、条件 head 和 controller 跨性质共享，同一个 checkpoint 接受不同 `PropertyContext(name)`。输入图与 h_k 的演化不依赖性质；不同性质通过 head 与停止规则产生不同输出和深度。

为避免改动 GraphInputs 和 B×1 标量接口，首版多性质训练每个 batch 均匀抽一个性质，全 batch 使用同一个 PropertyContext，并读取已有 `batch.target` 中对应列。四个性质覆盖率、训练更新次数及每性质样本暴露量写入日志。与单性质模型比较时明确一个共享模型的总训练预算，避免把少训练或多训练当成 conditioning 收益。

数据层的最低必要扩展：给 DatasetBundle 提供可选 property→(index,unit) 元数据，以及 `training_targets(property_name)` 访问；默认不传参数维持现有行为。QM9Orig 使用自己的 PROPERTIES / UNITS，不能误用 PyG 的 zpve 单位表。原始缓存和 sample.y 结构不变。

新增多性质 task，维护每性质独立 TargetTransform、单位和 property 索引；`state_dict()` 保存全部 transform 与名称顺序。训练抽性质，验证确定性遍历每个性质，不随机选一个代替全性质验证。

指标按性质分别累计样本数和误差，报告物理单位 MAE / RMSE；不能把 D、eV、a0^3 直接求平均。共同选模分数建议为各性质标准化 MAE 的等权平均，名字明确为 normalized macro MAE。Trainer 的 monitor 由配置声明，固定单性质任务仍默认 `mae`。

可通过 TaskResult 追加可选分组指标和一个通用聚合路径支持多性质；不让 Trainer 硬编码 QM9 性质列表。多性质统一学习率是新的训练选择，先用 5e-4 并在 validation 检查，不能继续隐式使用单个 task.target 对应的学习率插值。

关键消融只移除 controller 的 e_p，保留 prediction head 的 e_p，且参数预算尽量对齐。这样每个性质仍能被正确预测，比较才聚焦于“停止是否需要知道目标性质”。去掉 delta 也只作用于 controller，**不是移除 EGNNOrigStep 的残差连接**。

多性质推理首版一次请求一个性质，效率按 property query 计。一次同时请求全部性质时，共享骨架需要执行到最大停止深度，成本不能简单等于各性质平均 K；这种联合推理单独实现和报告。

**9. 先完成的诊断与 Oracle 分析**

第一批实验保留两个不同问题：

- 不同 K 分别训练的 untied / tied final 模型：使用已有 sweep，比较容量与执行预算。
- 同一个 tied checkpoint 的逐深度预测：使用 all_steps 或随机深度训练，保证各深度得到监督。

先将 application 中 Orig 的全局 final 限制改为明确校验规则，允许新的诊断 preset 使用现有 all_steps。原 `qm9_egnn_orig` 和 `qm9_loop_egnn_orig` 的 final 默认值保持不变。all_steps 是已有代码上的低成本诊断协议；另记录随机深度版本，避免将其与 PDF warm-up 等同。

新增离线 `analysis/oracleDepth.py`，消费已有 sample×property×depth CSV：

```text
K_error* = earliest argmin_k e_i(k)
K_cost*  = earliest argmin_k [e_i(k) + lambda_oracle*k/Kmax]
```

跨性质比较时 e_i(k) 使用该 checkpoint 训练集 transform 缩放的误差；物理单位误差同时保留。lambda_oracle 与训练 lambda_compute 是不同参数，分别记录。PDF 将后一式称为 Minimum Sufficient Depth，报告时同时保留公式，避免将它误读成误差容忍阈值定义。

如需补充容忍式深度，可单独定义 `min{k:e(k)<=min_j e(j)+epsilon}` 并另命名，不能替换 PDF 原定义。

Oracle 的来源必须可追溯：同一 checkpoint 轨迹 / 分别训练的模型不能混成一类。主 controller 对照使用该模型自己的完整轨迹，并提供共同 warm-up 轨迹作为辅助参考，避免只报告因 controller 训练而形成的自证结果。

Oracle 使用真实标签，只在离线分析出现，不进入 controller、部署推理或 test 超参选择。用 test 标签事后展示 Oracle 可以，不能据此挑 lambda、tau、checkpoint 或 size heuristic 参数。

复杂度第一版可直接计算：总原子数、重原子数、回转半径、元素多样性。回转半径默认几何中心等权定义并记录；没有质量数据时不暗称质量加权。

化学图直径、环数、可旋转键后置：从可靠的原始化学连接信息构建独立分析侧表，按 `qm9orig:<原始分子ID>` 对齐并验证元素、节点和样本覆盖。不要将 PyG 的样本顺序当成 Orig ID，也不要用全连接传播边来计算这些化学指标。

输出包含 Oracle 直方图、逐样本 error-depth、性质×复杂度分组表、Spearman、learned/oracle depth MAE 与 ±1 步命中率。复杂度分位点只从 train/validation 定义；相关系数遇到常量输入报告 N/A，不填 0。

第一阶段是否进入大规模 AdaLoop 实验，以分布差异、跨 seed 稳定性和误差改善幅度共同判断；不把 PDF 中示意数字当成待复现的既定结果。

**10. 日志、预测文件和恢复协议**

TaskResult 在尾部增加带默认值的额外统计字段，至少提供按样本加权的 `prediction_loss`、`compute_loss`、`expected_depth`、`halt_depth`、`halt_at_1_fraction`、`halt_at_max_fraction`。可补 p 的熵帮助诊断塌缩。warm-up 不适用项明确为空或带有效计数，CSV 列集合固定，不能跨阶段换表头。

保留旧 `*_by_depth.csv` 的语义；新导出使用：

```text
predictions/<split>_<checkpoint>_adaptive.csv
    一行 = sample_id × property
    y_true, y_pred, unit, halt_depth, executed_depth
    checkpoint_id, split_id, seed, lambda_compute
    Kmax, inference_rule, threshold, execution_mode

predictions/<split>_<checkpoint>_halting_by_depth.csv
    一行 = sample_id × property × depth
    y_k, q_k, p_k, survival, valid_depth
    仅 full diagnostic 导出完整未来轨迹

analysis/oracle_depth.csv
analysis/complexity.csv
analysis/representation_by_depth.csv
evaluation/efficiency.json
```

复用现有文件哈希、sample ID 唯一性、临时文件原子替换与覆盖保护。补齐 checkpoint_id、property、split、mode、tau、lambda 等 provenance，防止错误复用不同推理规则的 CSV。

旧 `predict.py` 可以增加模式分流，默认 legacy 仍走原 exporter；新 adaptive exporter 不冒用旧的最终深度列。`evalDepth.py` 明确只处理 fixed execution，adaptive 用新 `evalAdaptive.py`。

Checkpoint 继续复用 model/task/optimizer/scheduler/RNG/metadata 容器。新增参数由注册子模块自动保存；task 保存阶段、warm-up 边界、性质变换。resume 保持结构与协议一致，不将 warm-up、lambda、tau、property 列表或归一化静默改变。

从旧 Loop checkpoint 初始化 AdaLoop 应是独立 `init_from` 语义：只拷贝形状与名称匹配的 initializer/operators/readout/head，并列出新增 controller 或条件 head 的初始化项；不伪装成 resume，不恢复不兼容 optimizer。条件 head 输入维度变化时默认重新初始化该 head，避免无说明地忽略权重。

**11. 配置设计与校验**

新增 experiment preset，继承 Orig 的数据和训练配置，仅覆盖差异。建议名称：

```text
qm9_loop_egnn_orig_diagnostic
qm9_adaloop_egnn_orig
qm9_adaloop_egnn_orig_smoke
qm9_adaloop_egnn_orig_conditional
```

单性质配置草案如下，仅用于表达待实现接口，当前不可直接执行：

```yaml
model:
  name: egnn_orig
  hidden_dim: 128
  weight_tying: true
  execution:
    name: adaptive
    depth: 12                # 沿用现有字段；adaptive 下明确表示 Kmax
    inference_rule: cdf
    threshold: 0.5
    compact_inference: true
  halting:
    hidden_dim: 64
    use_delta: true
    property_conditioning: false  # 单性质最小版；完成版为 true
task:
  name: adaptive_regression
  target: mu
  supervision: adaptive
  normalization: mad
  loss: mae
  lambda_compute: 0.03
  warmup_epochs: 200
trainer:
  epochs: 1000
  monitor: mae
```

`supervision: adaptive` 是 provenance 标记，由新 Task 处理，不送入旧 RegressionTask 构造器。主条件化版本另增加 property 列表、条件 head、embedding 维度和多性质 Task，使用 normalized macro MAE monitor。

校验至少包括：adaptive 只允许当前已支持的 Orig 骨架、tying=true、正整数 Kmax、非负有限 lambda、0<tau<1、0<=warmup_epochs<epochs、合法且不重复的性质名、execution 与 task 配对、输入数据的原子序数与 EGNNOrig 支持的 H/C/N/O/F 元素范围相符。

第一版保留 `execution.depth`，避免同时引入 depth / max_depth 两个来源；报表将它显示为 Kmax。新 experiment_name 包含 adaptive、lambda 和性质信息；旧名称模板不改。

lambda × seed × property 可先复用 Hydra multirun。现有 `sweepDepth.py` 继续只负责固定深度实验；需要多 GPU 自适应扫描时再新增小入口复用 `GPUQueue`，不要重写队列或把 depth sweep 悄悄变成多用途脚本。

**12. 对照实验、消融和效率**

| 实验 | 最小设置 | 关键约束 |
|---|---|---|
| Untied Orig | K=2,4,6,8,12 | 每个 K 独立训练，final 监督 |
| Tied Orig | K=4,8,12 | 相同数据与训练预算 |
| 诊断 Loop | Kmax=12，所有深度能预测 | 标注 all_steps / random_depth 训练协议 |
| AdaLoop 单性质 | mu 起步，再 alpha/gap/U0 | lambda=0,0.01,0.03,0.1 |
| AdaLoop 条件化 | 一个模型包含四性质 | 对应 fixed 条件化模型也共享四性质训练 |
| Random halt | 与样本内容无关的深度 | 固定随机种子，匹配验证预算，报告随机波动 |
| Size-based halt | clip(ceil(Natom/c),1,Kmax) | c=10 可作原始示例；调参仅 validation |
| Fixed budget | 邻近整数 K 与预算匹配随机混合 | 不能把 4.7 与 5 声称为精确相同成本 |
| Oracle | 同一 checkpoint 完整轨迹 | 仅事后参考，不能部署 |

随机与 size 策略首先用共同、已充分训练各深度的 Loop 模型，隔离停止策略的贡献；另报告端到端训练的 adaptive 模型。不能直接拿只监督最后一层的 baseline 中间输出来比较早退。

预算匹配以实际 cost 优先。Orig 完全图每图边数为 N(N-1)，同样平均 K 对不同图大小分配深度会产生不同 FLOPs；因此除 Avg.K 外，记录总 node-steps、edge-steps 和 controller/readout 成本。预测 head 每步执行的额外开销也应计入，而 fixed-final 可能只执行最后一次 head。

PDF 的 AMP-style baseline 后置为单独复现任务，需要明确原方法、公式与训练协议后实现；一个任意 q-threshold 变体不能直接命名为 AMP。当前计划不依赖对外部论文的未核实描述。

消融至少包括 lambda=0、controller 无 delta、controller 无 property conditioning、无 warm-up，以及 fixed/tied/adaptive 递进对比。无 warm-up 是稳定性诊断，不作为必需的性能改进。除被消融项外保持训练协议一致。

效率测量分别报告：参数量、soft E[K]、hard Avg.K、实际执行深度、FLOPs/sample、batch=1 latency、固定 batch throughput、峰值显存。GPU 计时有预热、同步/事件测量和重复统计；同一硬件、dtype、batch、数据样本，不含读盘与 CSV。统计 compaction 和 controller 开销；报告训练与推理成本分开。

首轮以操作计数和可支持的 profiler 为依据，明确 MAC/FLOP 约定及未覆盖算子；不能把参数量×深度当精确 FLOPs。主图为 hard MAE–实际 FLOPs / latency，同时提供 MAE–Avg.K 辅助图。

**13. 表示分析与对称性**

完整诊断轨迹逐 batch 计算并写标量摘要，不在内存累计全部节点特征：

```text
S_k = mean_{i!=j in same graph} cosine(h_i^k, h_j^k)
E_k = mean_{(i,j) in E} ||h_i^k-h_j^k||^2
```

注明 E 是 Orig 完全传播图还是独立化学键图。默认主报告采用 Orig 传播图；若另用化学图需单列。补充特征范数 / 归一化能量，区分整体缩放和表示趋同；单节点或无边图的未定义指标带有效标记排除。

将 MAE-depth、能量-depth、相似度-depth 和停止分布联合展示。只有观察到相应现象才能作出 oversmoothing 结论，不能因为深度加大误差升高就直接归因为平滑。

对称性测试包含旋转、平移、反射和节点置换；既测 y_k / q_k / p_k 数值一致，也测最终 hard K 和预测。PDF 建议抽 1000 个测试结构；用固定 seed 生成变换，分性质报告。

浮点下阈值附近可能发生离散翻转：记录 max prediction/probability deviation、halting consistency、CDF 到 tau 的 margin。不能为了凑 100% 删除临界样本。原始物理单位与标准化误差的容差分别声明，PDF 的 1e-5 可作目标而不是所有性质的无条件保证。

**14. 测试与验收清单**

| 类别 | 必须验证的行为 |
|---|---|
| 原有行为 | 原 EGNNOrig 官方对照、梯度、Adam 更新、固定坐标、tied 共享、旧 checkpoint / CSV 测试继续通过 |
| 参数注册 | Controller 在 parameters/state_dict 中，跟随 device；optimizer 确实更新；Kmax 变化不增加共享模型参数量 |
| 概率 | p>=0、每图 sum(p)=1、Kmax=1、末步剩余质量、极端 logits、无 NaN |
| 手算 loss | 小 batch 手算逐样本期望误差和计算成本，验证 reduction 次序与梯度 |
| 梯度 | 非退化 K>1 时 controller 与共享 step 获得梯度；delta 路径不断开；末步强制停止无伪梯度 |
| warm-up | 深度分布覆盖合法范围、仅 y_K 监督、controller 无更新、进入 adaptive 时切换正确 |
| 恢复 | 跨 warm-up 边界和 adaptive 阶段 epoch 恢复等价；warm-up 无 best.pt 时仍可恢复 |
| 停止规则 | CDF 与 hazard 的差异用构造例验证；首步/末步停止；硬输出不是 mixture |
| active graphs | 非连续图退出、重新编号、原顺序回填、不同大小图、单节点图、全停、批间隔离 |
| 数值一致 | 完整轨迹选择与 compact 推理输出/停止深度一致，阈值边界记录 margin |
| 真实省计算 | 构造已知停止分布，统计进入 step 的节点和边确实减少；仅冻结状态不能通过 |
| 条件化 | 一个 checkpoint 可预测四性质；名称顺序与 embedding 映射一致；transform/单位正确；无 conditioning 消融只屏蔽 controller 输入 |
| 对称性 | 预测、q/p、K 对旋转/平移/反射/置换的检查 |
| 导出 | 样本覆盖、ID 唯一、有效深度、规则 provenance、旧 fixed 导出不变 |

新增测试建议拆为 `testHalting.py`、`testAdaptiveExecution.py`、`testAdaptiveTraining.py`、`testAdaptiveEvaluation.py`，多性质新增 `testPropertyConditioning.py`。合成小图测试不下载数据；真实 QM9 smoke 只用已准备的本地数据。

实现后运行针对性测试，再完整 `pytest -q` 与 `ruff check src tests scripts`。CPU 确认数学行为，GPU smoke 确认训练 / compact inference / checkpoint 链路。只有计划文档变更时不启动这些模型测试。

**15. 建议的提交与里程碑**

| 顺序 | 交付 | 主要涉及文件 | 完成判据 |
|---|---|---|---|
| M0 | 诊断协议、四性质 Loop 轨迹、Oracle | application 校验、诊断 preset、新 oracleDepth | 旧 preset 不变；每深度有监督；能报告深度分布及差异 |
| M1 | Controller、概率与完整 adaptive execution | 新 halting/adaptiveExecution，trace，iterative | 概率 / 注册 / 梯度 / 对称性单测通过 |
| M2 | 单性质两阶段训练、日志、恢复 | 新 adaptive task，factories，Trainer 通用 hook，preset | 小数据过拟合；阶段切换和恢复一致；hard MAE 可选模 |
| M3 | 在线停止与图压缩、独立导出 | 新 activeGraphs、自适应评估，predict 分流 | 与 full reference 一致；已停止图不再进入 step |
| M4 | 共享多性质模型 | 条件 head、多性质 task、bundle 元数据、指标 monitor | 一 checkpoint 四性质；conditioning 消融有效 |
| M5 | 对照、预算、分析与正式结果 | 新 evalAdaptive / benchmark / analysis、小型配置集 | 3 seeds 结果、MAE–compute、Oracle、对称性与表示诊断可追溯 |

先做 mu 的小样本 smoke，不立刻启动 4 properties×4 lambdas×3 seeds 的 48 次完整单性质训练。M0 的科研假设验证通过后再展开规模；完整条件化版本的 lambda×seed 是另一组实验，不能和单性质 48 次混报。

最终最低交付包含：可训练的 Orig AdaLoop、真实 early-exit、共享性质条件化、完整恢复、可复查的输出与核心基线。PDF 更广范围的 QMugs / 晶体实验不作为这一轮的隐含依赖。

**16. 后续工作与不应提前承诺的结论**

QMugs 接入需要新数据适配和按 molecule 分组的 conformer 划分；还需解决 Orig 输入编码仅支持 H/C/N/O/F 的限制。扩元素与更大完全图成本属于显式骨架适配，不能仅切换 dataset 配置。size OOD 放在这之后。

测试时 Kmax=8→12/16 需要同时移除原第 8 步的强制终止、将强制终止移动到新上限，并将新预算单独记录；共享权重支持执行不代表训练外深度一定有效，原末步 controller hazard 未被监督也应说明。

PDF 的长程合成与 underreaching 若要作为因果实验，需要另建局部传播图变体并清楚标注它与主 EGNNOrig 完全图的区别。PaiNN / PBC / Matbench 属于独立 backbone 和数据协议扩展。

当前没有证据保证自适应一定更准、更快、产生更深的复杂样本停止或缓解平滑。本计划的验收首先保证实现与比较成立，再由实验决定这些结论是否得到支持。
