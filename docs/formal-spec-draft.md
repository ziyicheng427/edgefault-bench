# Formal Specification Draft — Experiment Lifecycle and Integrity Invariants

- Status: **Accepted（不变量与转换已冻结；T09/T10/T11 已于 2026-11-06 落地：
  TLC 模型检查无错误（`formal/tlc-report-2026-11-06.txt`）、Hypothesis 属性测试、
  `edgefault mutation audit` 12/12 检出）**
- Date: 2026-11-06
- Scope: EdgeFault-Bench 2.0 实验生命周期的形式化规范：实体词汇表、生命周期状态机、
  状态转换 guard、六条核心不变量（INV-1…INV-6）、验证技术路线与 mutation catalogue。
- 形式化对象：**抽象实验生命周期**（一个 (task, model-plugin) 组合从数据获取到结果发布的过程），
  不是 Python 实现本身。实现层通过运行时契约验证、属性测试与错误注入与之对应（§5、§6）。

---

## 1. 目的与范围

把"一次有效的可复现实验"从文档承诺（`docs/evaluation-protocol.md`）升级为可检验的数学对象：

1. 给出实验生命周期的**有限状态机**（FSM），每个状态转换附 **guard**（必须成立的条件）；
2. 给出在任何可达状态都必须成立的**六条不变量**；
3. 定义每个不变量由哪个运行时机制强制（现状：部分强制），以及用什么形式化工具验证。

对应论文 §3（Requirements and Formal Specification）与 RQ1 的证据基础。

## 2. 实体词汇表（Vocabulary）

| 符号 | 含义 | 现有实现锚点 |
|---|---|---|
| `D = (dataset_id, version, files, license)` | 数据集：身份、版本、带校验值的文件集、许可 | `DatasetMetadata`（`contracts.py:62`）、`registry/*.json` |
| `R` | recording 集合：一次独立采集 | `Recording`（`contracts.py:96`） |
| `W` | window 集合：从 recording 切出的窗口 | `WindowSpec`（`contracts.py:153`）、`data/splits.py` |
| `rec : W → R` | 窗口到原始 recording 的归属函数 | **当前缺失**（prepared 表无此列，见 G5） |
| `T = (task_id, domain_field, eval_group, Π, Λ, seeds, ω)` | 任务：留出域、评估分组、划分、标签集、种子集、窗口规则 | `TaskSpec`（`contracts.py:176`）、`registry/tasks/*.json` |
| `Π : R → {tr, va, te}` | 划分函数（由 `T.partitions` 按 domain 值诱导） | `audit.py` / `prepared.py` |
| `P_d, P_p, P_m` | 数据集/预处理/模型插件 | `plugins.py:26,56` |
| `σ_fit` | 预处理器拟合状态（只能由 train(+val) 数据导出） | **当前缺失**（无 Preprocessing 契约，见 G2） |
| `B` | 结果包（result bundle） | `runner.py:134`、`schema_specs/result-v1.schema.json` |
| `S` | 已执行 seed 集合 | `runs[].seed` |

约定：`Π` 诱导到窗口级为 `Π_W(w) := Π(rec(w))`。

## 3. 六条核心不变量（冻结）

### INV-1 Source Integrity（来源完整性）

```
∀ r ∈ R: file(r) 存在 ∧ size(file(r)) = declared_size(r) ∧ sha256(file(r)) = declared_sha256(r)
∀ B: B.dataset = (dataset_id, version) ∧ 引用的任务清单、prepared 表、训练策略均有内容哈希
```

**含义**：进入实验的数据文件必须匹配注册版本；结果包引用的每个输入件可被哈希复核。

### INV-2 Partition Exclusivity（划分互斥）

```
Π 是全函数 ∧ dom(Π) = R
∀ w ∈ W: Π_W(w) = Π(rec(w))                    （窗口继承其 recording 的划分）
distinct recordings ⇒ 不存在 r 同时映射到两个分区  （recording 不跨 train/validation/test）
```

**含义**：同一原始 recording（及其全部窗口）不得跨分区；每个 recording 恰好属于一个分区。

### INV-3 Test Isolation（测试集隔离）

```
test 数据仅在被标记为 EVALUATE 的转换中被读取：
  σ_fit ∉ Span(fit 调用涉及 te 数据)
  模型选择 / 早停 / 超参仅使用 Π⁻¹(va)
  P_m 在 TRAIN 阶段对 Π⁻¹(te) 无访问能力（架构性，非约定性）
```

**含义**：测试集不参与训练、预处理拟合、早停与模型选择；且这是**接口签名层面的不可达**，
不是文档要求。

### INV-4 Seed Completeness（种子完整性）

```
S = T.seeds ∧ |runs| = |T.seeds| ∧ runs 中 seed 序列 = T.seeds（保持冻结顺序）
失败也占位：某 seed 的失败以 failed-run 记录保留，不得静默丢弃
```

### INV-5 Provenance Completeness（溯源完整性）

```
∀ B: {task_manifest_hash, dataset(id,version), plugin(source,version), code_commit,
      environment, policy_ref, per-seed metrics, input_hashes} ⊆ B
      ∧ recompute(B.input_hashes) 一致
```

**含义**：任何结果都能追溯到数据、任务、模型、代码与环境；重算哈希必须吻合。

### INV-6 Plugin Contract（插件契约）

```
∀ P ∈ {P_d, P_p, P_m}: P 满足接口签名 ∧ entry-point 名 = 声明 id ∧ 版本兼容
                     ∧ 输入输出符合对应 schema ∧ 冲突即拒绝（fail closed）
```

## 4. 生命周期状态机

### 4.1 状态

```
EMPTY → SOURCES_VERIFIED → TASK_AUDITED → DATA_PREPARED → MODEL_CONSTRUCTED
      → TRAINED(per seed) → EVALUATED(per seed) → RESULT_VALIDATED → PUBLISHED
任意状态 guard 失败 → REJECTED（终态，附机器可读诊断报告）
```

```
            ┌────────────────────────────────────────────────────────┐
            │                        (guard 失败)                     │
            ▼                        任何状态                         │
  EMPTY ──t1──► SOURCES_VERIFIED ──t2──► TASK_AUDITED ──t3──► DATA_PREPARED
                                                                   │ t4
                                                                   ▼
  PUBLISHED ◄──t8── RESULT_VALIDATED ◄──t7── EVALUATED ◄──t6── TRAINED ◄── MODEL_CONSTRUCTED
                    (每个 seed 依次经过 t5/t6；全部 seed 完成后 t7)         ▲
                                                                            │ t5
                                                              MODEL_CONSTRUCTED
```

### 4.2 转换与 guard（冻结）

| # | 转换 | Guard（必须全部成立，否则 → REJECTED） | 强制不变量 |
|---|---|---|---|
| t1 | EMPTY → SOURCES_VERIFIED | 下载/校验完成：INV-1 第一式（逐文件 size+SHA-256）；适配器通过 `validate_dataset_adapter` | INV-1, INV-6 |
| t2 | → TASK_AUDITED | 泄漏审计通过：`audit_recordings` 全部检查通过（身份、域完整、标签支持、划分指派完全、分区非空、recording 唯一且不跨区） | INV-2 |
| t3 | → DATA_PREPARED | 窗口表含 `(window_id, recording_id, start_sample, end_sample)`；窗口划分继承检查 `Π_W(w)=Π(rec(w))`；schema 校验通过；`σ_fit` 仅由 train(+val) 行拟合且其哈希入册 | INV-2, INV-3 |
| t4 | → MODEL_CONSTRUCTED | 插件解析成功、契约校验通过、身份匹配、版本兼容登记 | INV-6 |
| t5 | → TRAINED（每个 seed s） | 训练视图只含 train+validation（test 不可达）；seed 播种确定性开启；早停仅用 va 指标；训练策略为版本化对象 | INV-3, INV-4 |
| t6 | → EVALUATED（每个 seed s） | 对 s 的 test 预测**恰好发生一次**；此转换是 test 数据唯一读取点 | INV-3, INV-4 |
| t7 | → RESULT_VALIDATED | 结果包通过 schema；INV-4 的 seed 双射检查；INV-5 字段齐全且重算哈希一致 | INV-4, INV-5 |
| t8 | → PUBLISHED | 不可变标签 + 全部工件 SHA-256 清单 | INV-1, INV-5 |

**Fail-closed 语义**：任一 guard 失败，系统进入 REJECTED 并输出结构化错误（沿用
`LeakageReport.require_pass` 的先例），**不得部分继续**。RQ1 的 mutation 实验即验证该语义。

## 5. 验证技术路线（四层，冻结）

| 层 | 工具 | 验证对象 | 产出 |
|---|---|---|---|
| L1 抽象模型检查 | **TLA+ / TLC**（首选；Alloy 作为 INV-2 划分结构的小范围实例反查备选） | §4 FSM + §3 不变量：所有可达状态满足 INV-1…INV-6；无死锁（除 REJECTED/PUBLISHED）；错误路径只能进入 REJECTED | `formal/ExperimentLifecycle.tla` + TLC 报告（论文 §5、RQ1 证据） |
| L2 属性测试 | **Hypothesis**（Python） | 生成有效/无效 Task Manifest、recording 集（注入跨区 recording）、prepared 表（注入 window↔recording 归属矛盾）、插件描述符（身份不匹配）；性质 = "审计通过 ⇔ 生成实例满足不变量"（双实现 oracle） | `tests/properties/` + CI |
| L3 错误注入 | **Mutation Testing**（§6 catalogue） | 主动植入泄漏/坏哈希/缺 seed 等，验证每项被 ≥1 机制拒绝且错误信息指向正确不变量 | 检出矩阵 JSON（RQ1 主证据） |
| L4 运行时契约 | 现有 + 扩展的 validators（audit、schema、results、plugin） | 真实执行路径上每个 guard 的 fail-closed 行为 | CI 强制 + 审计文档 |

对应关系（论文表述）：**formal specification + model checking of the abstract lifecycle +
runtime contract verification + property-based testing**。不声称实现级证明。

### TLA+ 规范骨架（T09 实现底稿）

```tla
CONSTANTS Recordings, Windows, Seeds, DomainValues
VARIABLES phase,            (* 状态机当前状态 *)
          part,             (* Π : recording → {tr,va,te} *)
          recOf,            (* rec : window → recording *)
          fitOn,            (* σ_fit 的数据来源 ⊆ {tr,va} *)
          trainedSeeds, evaluatedSeeds,
          bundleHashes, rejected

Init == phase = "EMPTY" ∧ ... 
Acquire == phase="EMPTY" ∧ FilesVerified ⇒ phase'="SOURCES_VERIFIED"   (* INV-1 guard *)
Audit   == phase="SOURCES_VERIFIED" ∧ PartitionTotalAndExclusive(part) ⇒ phase'="TASK_AUDITED"
Prepare == phase="TASK_AUDITED" ∧ WindowsInheritPartition ∧ fitOn' ⊆ {tr,va} ⇒ ...
Construct == PluginContractHolds ⇒ ...
Train(s) == phase="MODEL_CONSTRUCTED" ∧ s ∉ trainedSeeds ∧ CanSeeTestOnlyViaKernel=FALSE ⇒ ...
Evaluate(s) == phase="TRAINED" ∧ s ∉ evaluatedSeeds ⇒ ...               (* test 唯一读取点 *)
Validate == SeedsBiject(runs, Seeds) ∧ ProvenanceComplete ⇒ ...
Reject(g) == ¬GuardHolds ⇒ phase'="REJECTED" ∧ rejected' = g

Next ≜ Acquire ∨ Audit ∨ Prepare ∨ Construct ∨ ∃ s∈Seeds : Train(s) ∨ Evaluate(s) ∨ Validate ∨ Reject
Spec ≜ Init ∧ □[Next]_vars

Inv1SourceIntegrity ≜ ...   Inv2PartitionExclusivity ≜ ...
Inv3TestIsolation ≜ ...     Inv4SeedCompleteness ≜ ...
Inv5Provenance ≜ ...        Inv6PluginContract ≜ ...
```

TLC 检查项：`Spec ⊨ Inv1..Inv6`、无意外死锁、`Reject` 之外不存在到达 REJECTED 的路径。

## 6. Mutation Catalogue（RQ1 反例清单，冻结）

每项 mutation 必须：被 ≥1 层机制**拒绝**、退出码非零、错误信息指向被违反的不变量；
任何"漏检"（实验继续并产出结果包）记为设计缺陷并回流架构修订。

| ID | 植入缺陷 | 针对不变量 | 期望检出层 |
|---|---|---|---|
| M-LEAK-01 | 同一 recording 出现在两个分区 | INV-2 | L4（audit）/ L1 |
| M-LEAK-02 | 测试 recording 的窗口被重标为训练 | INV-2 | L4（窗口继承检查，需 t3 新增）/ L2 |
| M-LEAK-03 | 预处理器在含 test 行的表上拟合 | INV-3 | L4（fitOn 记录）/ L1 |
| M-LEAK-04 | 早停指标改用 test | INV-3 | L1 / L4（训练视图不可达→类型级失败） |
| M-HASH-01 | 注册表中某源文件 SHA-256 被改 | INV-1 | L4（下载校验） |
| M-SEED-01 | 结果包丢弃一个 seed 的 run | INV-4 | L4（`load_result` 已具备） |
| M-SEED-02 | 复制某 seed 冒充另一 seed | INV-4 | L4 + L2 |
| M-PROV-01 | 剥离结果包 environment/git_commit | INV-5 | L4（`load_result` 部分具备） |
| M-PROV-02 | 篡改输入件后重放结果哈希 | INV-5 | L4（重算哈希，需 v2 结果包） |
| M-PLUGIN-01 | 插件 entry-point 名与声明 id 不符 | INV-6 | L4（`discover_*` 已具备） |
| M-PLUGIN-02 | 插件缺少契约方法 | INV-6 | L4（`validate_model_implementation` 已具备） |
| M-PLUGIN-03 | 与内置插件 id 冲突 | INV-6 | L4（`discover_*` 已具备） |

已具备 = 当前代码已强制（见 §7 现状表）；需新增的机制在 architecture-gap-analysis §2 对应任务。

## 7. 不变量 → 现状 → 目标机制（现状盘点，2026-11-06 代码为准）

| 不变量 | 现状（已强制部分） | 缺口（待建） |
|---|---|---|
| INV-1 | 下载器逐文件 size+SHA-256 校验（`download.py`、`registry/*.json`）；下载 `--verify-only` 模式 | 结果包不含输入件**内容哈希**（只有路径，`runner.py:142`）；t7 重算哈希缺失 |
| INV-2 | `TaskSpec` 构造即拒绝分区重叠（`contracts.py:38`）；`audit.py` 七项检查（含 recording_exclusivity）；`splits.py:84` 拒绝 recording 跨区 | prepared 表无 `recording_id`/`window_id` 区分列，示例文档甚至示范了 `device-a-load-0-window-0` 作为 recording_id——窗口级继承检查退化为命名自律（`prepared.py:123` 的一致性检查在逐窗口唯一 id 下失效）；t3 的窗口继承 guard 缺失 |
| INV-3 | 协议文档承诺（`evaluation-protocol.md`）；sklearn runner 结构上 fit 仅 train、va/te 分开预测（`runner.py:104-105`）；神经脚本验证集早停 | **纯约定，无架构强制**：无 Preprocessing 契约（fit/transform 分离）；训练方拿到全量特征矩阵（`prepare_task_data` 返回含 test 的数组）；PyTorch 无统一 runner，训练策略未版本化 |
| INV-4 | runner 按 `task.seeds` 循环（`runner.py:99`）；`load_result` 强制 seed 序列精确匹配（`reporting.py:25`） | 失败 seed 不占位（异常直接中止）；seed 双向检查只在汇总读取时，无 t7 独立校验命令对插件结果包全覆盖 |
| INV-5 | result-v1 schema + `load_result` 必填字段（task_id/model_id/git_commit/runs/environment） | 无输入件哈希、无策略引用、无重算校验（M-PROV-02 不可检出） |
| INV-6 | `validate_dataset_adapter`（身份/域完整/重复 id）；`validate_model_implementation`（方法存在性）；发现冲突即拒绝（`plugins.py:148,168`） | 无版本兼容策略；PyTorch 插件可构建不可执行（统一 runner 拒绝，`runner.py:93`）→"构建通过"被明确不等于"可执行"，契约执行段缺失 |

## 8. 与论文和任务的对应

- 论文 §3 直接取材于本文 §3–§4；§5（Implementation）取材于 §5–§6。
- T09 = 落地 §5-L1；T10 = 落地 §5-L2；T11 = 落地 §6；RQ1 数据 = §6 检出矩阵 + TLC 报告。
- 本文通过审查后，六条不变量的编号与措辞**冻结**；后续修改须走决策记录并更新 CHANGELOG。
