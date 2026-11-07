# Architecture Gap Analysis and Target API — EdgeFault-Bench 2.0

- Status: **Accepted（设计已冻结；见下方批次状态注记）**
- Date: 2026-11-06
- 批次状态（2026-11-06 第二批完成后更新）：G1、G2、G3、G5、G7 已在 1.2.0 开发线落地并附测试；
  G8 的版本收敛部分完成（pyproject=1.2.0，CHANGELOG 归段）；G4 的机制层就绪，
  独立环境 + 真实数据的端到端验证留待 T12；G6、G9、G10 未动。
- Scope: 以 2026-11-06 主分支代码为基线的接口差距分析（G1–G10）、目标 v2 契约设计、
  内核/插件/策略三方边界、迁移与兼容策略、验收标准。
- 原则：差距编号与建设任务（T04–T20）一一对应；每条差距给出验收标准；**本文只定义接口，
  不改代码**。

---

## 1. 现状接口清单（基线锚点）

| 接口/机制 | 现状定义 | 能力边界 |
|---|---|---|
| DatasetAdapter | `contracts.py:143`（`metadata` + `recordings()`） | 只暴露元数据与 recording 清单，**无信号读取** |
| Recording / DatasetMetadata | `contracts.py:62,96` | 不可变、构造期校验；含 domain 字典、采样率、样本数、SHA-256 |
| TaskSpec / WindowSpec | `contracts.py:153,176` | 分区互斥构造期强制；窗口规则只是 `(length, stride, normalization)` 字符串 |
| 泄漏审计 | `audit.py:73`（7 项检查） | recording 级、训练前；`audit.py:198` 的 legacy CLI 入口仍按 dataset_id 硬编码分发 |
| 划分构造 | `data/splits.py:42` | recording 跨区即抛错；窗口不重叠由 `stride ≥ length` 保证 |
| Prepared Feature Table | `prepared.py:40`、JSON schema v1 | 便携但仅 JSON；行 = `(recording_id, label, domains, features)`，**无窗口身份列** |
| 模型插件 | `plugins.py:56`（backend ∈ {sklearn, pytorch}） | 可发现、可构建；契约检查 = 方法存在性 |
| 统一执行 | `runner.py:73`（`run_prepared_benchmark`） | **仅 sklearn**；`runner.py:93` 显式拒绝 pytorch |
| PyTorch 训练 | `experiments/neural.py`（脚本） | 存在于统一 runner 之外，数据路径与任务耦合，策略未版本化 |
| 结果验证 | `reporting.py:17`、result-v1 schema | seed 序列精确匹配 + 必填字段；无输入件哈希 |
| CLI | `cli.py`（`edgefault schema/plugin/dataset/task/results/benchmark`） | v1.2 命令面已在 README 描述，但 `pyproject.toml` 仍为 1.0.0 |

## 2. 差距清单（G1–G10）

### P0（决定论文核心主张能否成立）

**G1 数据集插件不能读数据**（→ T04 / 指导 A1）
现状：`DatasetAdapter` 无信号读取接口；窗口提取与张量加载在内核侧按数据集分发
（`experiments/neural.py` 经 `torch_data.py`），新数据集无法不碰内核完成数据侧接入。
验收：新数据集只写独立 Adapter 包即可提供元数据**与信号**；内核无数据集分支代码。

**G2 无预处理契约**（→ T05 / A2）
现状：归一化藏在 `TaskSpec.window.normalization` 字符串与各脚本内部；特征表由外部
"documented extractor" 生成，拟合范围无人校验（INV-3 的 t3 guard 无从谈起）。
验收：`Preprocessor` 契约落地——`fit` 只接收 train(+val) 视图；`transform` 无状态；
拟合状态摘要（哈希）进入结果包；在含 test 行上调用 `fit` 被架构拒绝。

**G3 PyTorch 无统一执行**（→ T06 / A3+A4）
现状：统一 runner 拒绝 pytorch 后端；训练策略（优化器、早停、确定性）散落在
`experiments/neural.py`，未版本化、不可被外部插件复用。
验收：外部 PyTorch 模型插件通过 `edgefault benchmark run` 完成训练/预测/结果包，
与 sklearn 同一命令面、同一 result schema；训练策略为版本化 `TrainingPolicy` 清单。

**G4 无真实数据的端到端外部插件路径**（→ T12 / A5）
现状：`examples/edgefault-example-plugin` 是作者自维护的合成数据演示（其审计文档已如实
声明）；HUST/Mehran 的接入均修改内核。
验收：独立 Python 环境 + 独立插件包（非本仓库）在**真实数据集**上完成 t1→t8 全流程；
过程留痕为审计文档；理想情况下由非核心开发者执行（RQ2 证据）。

### P1（决定实验体系与工程可信度）

**G5 存储不可扩展 + 窗口身份缺失**（→ T07 / 指导 2.3）
现状：特征表仅 JSON（文档自认"适合小型审计示例"）；无 `window_id`/`recording_id`/
`start_sample`/`end_sample` 列；`docs/prepared-feature-tables.md` 示例把
`device-a-load-0-window-0` 写作 recording_id，窗口级泄漏审计在此命名下退化为空检查。
验收：元数据（JSON）与特征矩阵（Parquet/Arrow）分离；大信号流式读取（SignalStream）；
窗口表四列齐全且 t3 的窗口继承检查（M-LEAK-02）可执行。

**G6 统计评价停留在 3-seed 均值±标准差**（→ T15/T16 / 指导 3.2）
现状：`evaluation-protocol.md` 承诺逐 run 报告，已做到；但同 recording 的多窗口非独立
样本的问题未处理。
验收：以 recording/device 为重采样单元的 group bootstrap + 95% CI；窗口级与
recording 级双口径；模型比较的配对 group-level 检验；window-level 与 recording-level
标签稀缺、synthetic noise 与 sensor perturbation 实验分离成轨。

**G7 内核自身不一致：legacy 审计入口硬编码数据集**（→ 随 T04 顺带）
现状：`audit.py:198-208` 按 dataset_id 字符串分发适配器，与"扩展不碰内核"的主张矛盾
（统一 CLI 已走 `load_dataset_adapter`，legacy 入口未清理）。
验收：legacy 入口退役或改走插件注册表；全仓 grep 无 dataset_id 分发残留。

**G8 版本漂移**（→ T20 前收敛）
现状：`pyproject.toml=1.0.0`，README 描述 v1.2 CLI，CHANGELOG 大量 Unreleased。
验收：代码/文档/包/Schema 版本在下一个稳定发布统一，并声明 Schema 兼容策略。

### P2（表述与证据边界，投稿前必须处理但不构成架构工作）

**G9 硬件表述**：现有测量为 Apple M1 CPU。要么补 Jetson Orin Nano / Raspberry Pi 实测，
要么全部表述收敛为 "CPU inference cost"，删除工业边缘部署暗示。
**G10 框架自身开销未测量**：RQ4 需要数据读取/校验/manifest 验证/序列化耗时、峰值内存、
以及相对"不经框架的裸模型脚本"的额外开销比例。

## 3. 目标 v2 契约设计（接口草案，供审查）

设计准则：内核拥有不变量与状态机；插件拥有领域知识；策略（Policy）拥有超参与训练法。
三者以版本化清单为界面，运行期由内核按 §formal-spec 的 guard 逐转换放行。

### 3.1 数据侧：DatasetAdapter v2 + SignalStream（修 G1、G5）

```python
class DatasetAdapter(Protocol):
    @property
    def metadata(self) -> DatasetMetadata: ...
    def recordings(self) -> Sequence[Recording]: ...
    def open(self, recording_id: str) -> SignalStream: ...   # 新增：内核拉取数据的唯一通道

class SignalStream(Protocol):
    @property
    def info(self) -> SignalInfo   # channels, sample_rate_hz, dtype, units, total_samples
    def read(self, start: int, count: int) -> NDArray: ...   # 分块读取，不要求整文件载入
```

- 多模态数据集（KAIST）通过 `SignalInfo.channels` 表达异构通道；缺传感器/采样率不一致
  在 `SignalInfo` 层显式声明并由 t3 guard 拒绝静默填充。
- 原始信号不入库、不分发：SignalStream 直接读校验过的源文件（保持现有许可纪律）。

### 3.2 预处理：Preprocessor 契约（修 G2）

```python
class Preprocessor(Protocol):
    plugin_id: str; version: str
    def fit(self, view: TrainView) -> None: ...          # TrainView 只含 train(+val) 窗口
    def transform(self, windows: WindowBatch) -> NDArray: ...   # 无状态、确定性
    def fitted_state_digest(self) -> str: ...            # 拟合状态哈希 → 结果包（INV-3 证据）
```

### 3.3 模型侧：统一生命周期 + 版本化训练策略（修 G3）

```python
class Estimator(Protocol):                 # 后端中立（sklearn / PyTorch 插件同构）
    def fit(self, train: DataView, validation: DataView,
            policy: TrainingPolicy) -> TrainingRecord: ...
    def predict(self, data: DataView) -> NDArray: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "Estimator": ...

class TrainingPolicy(TypedDict, total=True):   # 版本化清单，内核持有，插件不可覆盖语义
    policy_version: str; optimizer: str; learning_rate: float; max_epochs: int
    early_stopping: Literal["validation_macro_f1"]; batch_size: int
    deterministic: bool                          # 对应 _set_seed 的确定性要求
```

- 训练插件**只能**接触 `train`/`validation` 两个 `DataView`——测试集隔离由函数签名保证
  （INV-3 的架构化）。
- 早停、模型选择只读 `validation` 指标；CORAL 等域目标由 `TrainingPolicy` 显式开关。

### 3.4 执行侧：BackendRunner 统一协议（修 G3、G4）

```python
class BackendRunner(Protocol):
    backend: Literal["sklearn", "pytorch"]
    def run(self, task: TaskSpec, data: PartitionedData,
            plugin: ModelPlugin, policy: TrainingPolicy) -> ResultBundle: ...
```

- `edgefault benchmark run --task T --model M [--backend auto]`：后端由插件声明，
  命令面单一（A4）。
- `PartitionedData` 由内核构造：`train_view()`/`validation_view()` 对插件开放，
  `test_view()` 仅评估阶段（t6）物化。

### 3.5 存储：三层分离（修 G5）

| 层 | 格式 | 内容 |
|---|---|---|
| 元数据/清单 | JSON（沿用 schema 机制，升 v2） | 来源、许可、校验值、字段定义、文件位置 |
| 特征/窗口表 | Parquet/Arrow（列式） | `window_id, recording_id, start_sample, end_sample, label, domains…, features` |
| 原始信号 | 不落地，SignalStream 流式 | 源文件由注册表哈希锁定 |

### 3.6 结果包 v2（修 INV-5 缺口）

在现有 result-v1 字段之上新增：`input_artifacts: {task_manifest_sha256,
prepared_table_sha256, policy_sha256}`、逐 seed 的 `status: ok|failed`（失败占位）、
`framework_overhead` 计时块；`results validate` 升级为重算哈希 + seed 双射 + 溯源字段
三合一（t7 guard）。

## 4. 内核 / 插件 / 策略 边界表（冻结）

| 关注点 | 内核（kernel） | 插件（plugin） | 策略（policy manifest） |
|---|---|---|---|
| 数据身份与校验 | 注册表、SHA-256、审计 | 提供元数据与信号流 | — |
| 窗口/划分 | 划分函数、互斥与继承检查 | — | 任务清单声明的条件留出 |
| 预处理 | fit 范围强制、状态哈希入册 | 实现 fit/transform | 窗口规则引用 |
| 训练 | DataView 可见性、seed 播种、确定性开关 | 实现 fit/predict/save/load | 优化器/早停/批量 |
| 评估 | test 物化、恰好一次、指标 | — | 指标集合 |
| 结果 | schema、哈希重算、seed 双射 | — | 引用哈希 |

## 5. 迁移与兼容策略

1. **冻结证据不动**：v1.0/v1.1 已发布结果与 schema v1 永久有效；新机制只作用于新工件。
2. **Schema 版本化**：task/prepared/result schema 升 v2 时保留 v1 读取能力；
   `edgefault schema validate` 双版本可用；不兼容变更须走决策记录。
3. **命令兼容**：`edgefault-audit-task` 等 legacy 入口在 G7 清理前保留，标注 deprecated。
4. **接口演进**：`DatasetAdapter` v2 以新增方法的方式扩展，旧适配器在新 runner 下报
   "capability missing" 而非静默降级（fail closed 一贯性）。
5. **内置即插件**：HUST/Mehran 适配器与四个内置模型迁入插件机制本身，消除"内置特权路径"
   （neural.py 脚本退役为薄封装），使内核零数据集/模型分支成为可测试断言（RQ2 的
   "核心代码修改量 = 0" 由此可证）。

## 6. 验收标准汇总（对齐指导 A/B/C 任务表）

| 任务 | 验收标准（可测试表述） | 对应差距 |
|---|---|---|
| A1/T04 | 新数据集接入零内核修改（grep 断言 + 独立包演示） | G1 |
| A2/T05 | 在含 test 行的视图上 `fit` 抛错；`fitted_state_digest` 出现在结果包 | G2 |
| A3/T06 | 外部 PyTorch 插件经统一命令产出 schema-v2 合法结果包 | G3 |
| A4/T06 | sklearn/PyTorch 同一 CLI 命令、同一 schema、同一统计口径 | G3 |
| A5/T12 | 独立环境 + 外部插件 + 真实数据 t1→t8 全流程审计文档 | G4 |
| B1/B2/T13/T14 | Gearbox、KAIST 适配器接入；KAIST 多通道元数据正确表达 | G1、G5 |
| B3 | 采样率/单位/通道元数据统一描述（SignalInfo） | G5 |
| B4/B5/T15 | 组合条件留出 + leave-one-device-out 外层任务清单 | G6 |
| C4/T08 | 结果包 v2 通过重算哈希与 seed 双射校验 | INV-5 缺口 |
| G7 | legacy 分发入口退役 | — |
| G8 | 版本三处一致 + Schema 兼容声明 | — |
| G9/G10 | 硬件表述收敛或补设备；框架开销基准入 `results/` | RQ4 |
