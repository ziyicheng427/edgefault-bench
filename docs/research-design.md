# Research Design — EdgeFault-Bench → Software Engineering Nexus (SEN)

- Status: **Frozen（2026-11-06 维护者裁决通过，Decision 0006 已 Accepted；第二批 T04–T08 已实施）**
- Date: 2026-11-06
- Owner: Ziyi Cheng
- Scope: 第一批任务 T01–T03 的主设计文件。冻结研究定位、研究问题、贡献、模块边界、
  形式化验证模型入口、目标 API 方向与实验验证矩阵。
- Companion documents:
  - [`formal-spec-draft.md`](formal-spec-draft.md) — T02 形式化规范草案（状态机、六条不变量、验证计划）
  - [`architecture-gap-analysis.md`](architecture-gap-analysis.md) — T03 架构差距分析与目标 API
- Governing decision record: [`decisions/0006-sen-research-repositioning.md`](decisions/0006-sen-research-repositioning.md)（Proposed）

本文档通过审查后，第二批（T04–T08 架构改造）才允许开工。在此之前**不修改任何核心代码**。

---

## 1. 投稿定位变更：JOSS → SEN

当前仓库依据 Decision 0005（2026-08-05）定位为 JOSS 类型的研究软件论文。本设计提出将投稿目标
改为 **Software Engineering Nexus（SEN，Elsevier 开放获取期刊）**。SEN 的公开范围明确包含
软件工程与 **形式化方法与软件验证**（见其
[Guide for Authors](https://www.sciencedirect.com/journal/software-engineering-nexus/publish/guide-for-authors)），
与本项目计划引入的形式化规范 + 模型检查 + 运行时契约验证的技术路线吻合。

这不是换个投稿渠道，而是**换一篇论文要回答的问题**：

| 维度 | 当前（Decision 0005 / JOSS） | 目标（SEN） |
|---|---|---|
| 论文对象 | 可安装、可审计的研究软件产品 | 实验完整性问题的软件工程解法（框架 + 形式化规范 + 实证） |
| 研究问题 | 模型能否跨工况泛化？哪种模型更轻量？ | 软件机制如何**保证**评估本身有效、可扩展、可复现？ |
| 主要证据 | 软件成熟度：安装、文档、测试、复现 | RQ1–RQ4 实验矩阵：错误注入、独立插件集成、多环境复现、开销测量 |
| 领域对话 | JOSS 软件生态 | 软件工程（插件架构、契约式设计、形式化验证、实证 SE） |

JOSS 路线下的工程任务（打包发布、文档站、DOI 归档、外部用户）**仍然有效**，但降级为支撑性工作；
论文的主证据链改为下文第 6 节的实验矩阵。

**定位准确性声明**：SEN 是新刊，目前没有获得其完整的官方投稿评价细则；本定位是基于其公开范围
（软件工程 + 形式化方法）作出的投稿策略，不构成录用保证。论文写作阶段需复核其 Guide for Authors。

## 2. 冻结的论文定位

### 2.1 题目（冻结，附声明）

> **EdgeFault-Bench: A Formally Specified and Extensible Software Framework for
> Auditable Industrial Fault-Diagnosis Experiments**

中文：面向工业故障诊断实验的形式化规范与可扩展审计软件框架。

**Claim 纪律**：在 T09（TLA+/Alloy 模型检查）与 T11（Mutation Testing）完成之前，"Formally
Specified" 只是目标定位。写作期间的中间表述为 *"a software framework with a formal
specification of its experiment lifecycle"*，且论文只声称：

> formal specification + model checking of the abstract lifecycle + runtime contract
> verification + property-based testing

**不声称**对整个 Python 实现完成数学证明。

### 2.2 一句话定位（Key Idea）

> Treat the benchmark itself as a formally specified software system: an extensible
> plugin architecture whose kernel enforces a small set of experiment-integrity
> invariants at explicit lifecycle state transitions, verified by model checking,
> property-based testing, and fault injection.

把 Benchmark 本身当作被形式化规范的软件系统：一个可扩展插件架构，其内核在实验生命周期的
显式状态转换处强制执行一小组实验完整性不变量，并用模型检查、属性测试与错误注入加以验证。

### 2.3 论文类型

**Technique / System 论文**（非 Benchmark 论文，非 New Problem 论文）。叙事由 Key Idea 承担；
四个模块是实现该想法的方法学，RQ1–RQ4 是对系统的实证检验。论文**不与**跨工况故障诊断
Benchmark 比拼数据集与模型覆盖面（该赛道已有 RESS 2024 跨域研究、IEEE Access 2025 标准化
Benchmark 等强先行工作），而是回答"如何让 Benchmark 成为可靠的软件系统"。

## 3. 论文逻辑骨架（Thinking Template，冻结版）

| 阶段 | 内容 |
|---|---|
| Research background | 工业故障诊断 ML 实验普遍以跨工况（condition-shift）泛化能力为核心主张，但实验有效性依赖研究者自律：recording 级泄漏、测试集参与预处理拟合/早停/模型选择、种子缺失、来源不可追溯等问题在论文配套脚本中反复出现，导致结果不可比、不可复现。领域内标准化工作已指出 ad hoc 数据划分是结果不可比的根源之一。 |
| Limitation 1 | 现有故障诊断 / 域泛化 Benchmark（领域 DG benchmark、DomainBed、IEEE Access 2025 标准化框架）固化了**评估什么**（数据集、划分、指标），但没有固化**如何保证评估有效**：泄漏控制、测试隔离与溯源是脚本约定，不是被强制的软件契约。 |
| Limitation 2 | 通用实验管理基础设施（MLflow 等）记录用户主动上报的内容，但不强制 recording 互斥、seed 完整、插件契约；完整性核查是事后、自愿的。 |
| Limitation 3 | 现有框架的扩展方式是改内核脚本（数据解析、训练循环、划分逻辑），每个新数据集/模型重新实现——也就重新破坏——完整性机制；不存在让第三方扩展**被约束地**保持完整性的架构。 |
| Key Idea / Our Goal | 见 §2.2（一句话，可被审稿人直接引用）。 |
| Challenge 1 | **异构扩展与完整性的冲突**：任意数据集（模态、采样率、文件格式）与模型后端（scikit-learn、PyTorch）必须能以插件形式接入，同时插件不能获得破坏不变量的能力（尤其是测试集访问与预处理拟合泄漏）。 |
| Challenge 2 | **强制必须可验证而非约定**：不变量要落到显式状态机转换的 guard 上，能被模型检查器、属性测试与错误注入检验，且 fail closed；同时要诚实区分"抽象模型已验证"与"Python 实现已证明"。 |
| Challenge 3 | **有效性要在规模与分发下成立**：哈希校验、流式读取、结果包等机制必须超出玩具 JSON 规模仍可用，框架自身开销必须可测量、可接受，复现必须跨环境成立。 |
| Methodology topic sentence | 我们提出 EdgeFault-Bench 2.0，一个由形式化生命周期规范驱动、四模块分层的可扩展实验框架。 |
| Module A — Data & Protocol Management（对应 Challenge 1，数据侧） | 统一 recording/window/device/condition 实体与元数据，版本化 Task Manifest，条件留出（含组合条件与 leave-one-device-out 外层留出），来源/许可/校验值自动登记。 |
| Module B — Extensible Execution Architecture（对应 Challenge 1，执行侧） | Dataset Plugin（含真实信号读取）、Preprocessing Plugin（fit/transform 严格分离）、Model Plugin（fit/predict/save/load 生命周期）、Backend Runner（sklearn/PyTorch 同一执行协议）；训练插件只见 train+validation，测试集由独立评估阶段提供。 |
| Module C — Formal Integrity & Verification（对应 Challenge 2） | 实验生命周期有限状态机 + 六条核心不变量（INV-1…INV-6）；TLA+ 模型检查 + Hypothesis 属性测试 + Mutation Testing 验证审计机制本身有效；运行时契约验证 fail closed。 |
| Module D — Reproducibility & Evaluation（对应 Challenge 3） | 溯源完整的结果包（含输入件哈希、逐 seed 记录、失败保留）、以 recording/device 为重采样单元的分组统计（group bootstrap + 95% CI）、框架自身开销测量、跨环境复现审计。 |
| Contribution 1 | A formalized specification of reproducible industrial ML experiments（生命周期状态机 + 六条不变量）。→ 论文 §3 |
| Contribution 2 | An end-to-end extensible architecture for heterogeneous datasets and model implementations that enforces the specification at its kernel boundary。→ 论文 §4–§5 |
| Contribution 3 | An automated integrity-verification and provenance-management mechanism（模型检查 + 属性测试 + 错误注入 + 结果包验证）。→ 论文 §5–§6（RQ1、RQ3） |
| Contribution 4 | An empirical evaluation across heterogeneous datasets, independently packaged plugins, and execution environments。→ 论文 §6（RQ1–RQ4） |

### 3.1 逻辑链自检（四项一致性检查）

1. **Limitations → Key Idea**：通过。Key Idea 直接回应 L1（约定→内核强制的契约）、
   L2（自愿记录→fail-closed 验证）、L3（改内核→受限插件扩展）。
2. **Key Idea → Challenges**：通过。三个挑战分别来自"插件不能破坏不变量"（C1）、
   "强制本身要可被验证"（C2）、"机制要跨规模/跨环境成立"（C3），均由 Key Idea 自然派生。
3. **Challenges → Methodology**：通过（附说明）。C1 由 Module A（数据侧异构）与 Module B
   （执行侧异构）共同承担——两者是同一挑战在两类插件上的投影；C2→Module C；C3→Module D。
   无悬空模块，无未覆盖挑战。
4. **Methodology → Contributions**：通过。Module A+B → Contribution 2；Module C →
   Contribution 1+3；Module D → Contribution 3（溯源部分）+ Contribution 4。

结论：骨架自洽，可进入 intro-drafter / 论文写作阶段（在实验完成后）。

## 4. 四个模块的职责与边界（冻结）

### Module 1 — Data and Protocol Management

- **职责**：数据来源/格式/版本登记；统一实体定义（`recording` / `window` / `device` /
  `condition`）；任务定义独立于数据集代码；单一条件留出与组合条件留出；
  leave-one-device-out 外层验证；多模态（缺失传感器、采样率不一致、通道错配）的显式策略。
- **边界**：不读取信号内容（读取在 Module 2 的 Dataset Plugin）、不定义指标、不做统计。

### Module 2 — Extensible Execution Architecture

- **职责**：Dataset Plugin（metadata + recordings + **信号读取**）、Preprocessing Plugin
  （fit 只见训练侧）、Model Plugin（统一生命周期 fit/predict/save/load）、Backend Runner
  （版本化 Training Policy：优化器、早停、确定性标志；sklearn 与 PyTorch 同一
  `edgefault benchmark run` 命令）。
- **边界**：训练插件**只能**接触 train+validation `DataView`；test 索引由评估阶段独立持有并
  物化——测试集隔离（INV-3）是架构属性而非文档约定。Runner 不定义指标语义。

### Module 3 — Formal Integrity and Verification

- **职责**：生命周期状态机与转换 guard 的形式化规范；INV-1…INV-6；TLA+/Alloy 模型检查；
  Hypothesis 属性测试（生成有效/无效 Task Manifest 与 prepared table）；Mutation Testing
  （植入泄漏/坏哈希/缺 seed 等，检验审计机制检出能力）；运行时契约验证（fail closed）。
- **边界**：形式化对象是**抽象实验生命周期**，不是 Python 实现本身；论文表述遵守 §2.1 的
  claim 纪律。

### Module 4 — Reproducibility and Evaluation

- **职责**：统一结果 Schema（含输入件哈希、逐 seed 记录、失败状态、环境）；完整性校验与
  从结果包到数据/代码/配置的完整溯源；以 recording/device 为单元的 group bootstrap、
  95% CI、窗口级与 recording 级双口径指标；模型比较的配对 group-level 统计；
  框架自身开销（读取、校验、验证、序列化、峰值内存、相对裸脚本的开销）测量。
- **边界**：不追求 leaderboard；不混合 test-time adaptation 与主结果；硬件表述收敛为
  "CPU inference cost" 除非增加真实边缘设备（Jetson Orin Nano / Raspberry Pi）。

模块间依赖：M1 → M2 → (M3 在每个转换点作为 guard) → M4。M3 是横切的：它的不变量定义
M1/M2 的接口约束，它的运行时检查是 M4 结果包的验证内容。

## 5. 冻结的研究问题（RQ1–RQ4）与实验矩阵

| RQ | 研究问题 | 主要实验 | 评价指标 | 主要缺口依赖 |
|---|---|---|---|---|
| **RQ1** | 形式化约束能否保证实验完整性？（Can the formal constraints guarantee experiment integrity?） | 错误注入（mutation catalogue，见 formal-spec-draft §6）、Hypothesis 属性测试、TLC 模型检查 | 违规检出率、漏检率（false-negative rate）、误报率 | T09/T10/T11 |
| **RQ2** | 插件架构能否真正支持独立扩展？（Does the plugin architecture support independent extension?） | 外部数据集适配器（Gearbox、KAIST）与外部模型插件在独立环境中接入并完整执行；与传统方式（改脚本）的对照记录 | 核心代码修改量（LOC=0 为目标）、接入时间、独立环境端到端成功率 | T04–T08、T12–T14 |
| **RQ3** | 实验能否保证可追溯与可复现？（Are experiments traceable and reproducible?） | 多环境（≥2 OS/Python 版本）重复执行；从结果包出发的干净环境重建 | 复现成功率、结果一致性（数值容差内）、溯源字段完整率 | T08、T16 |
| **RQ4** | 框架能否支撑不同规模与类型的数据与模型？（Does the framework sustain heterogeneous scale and type?） | 多数据集（轴承振动 ×2 + 齿轮箱 + 多模态）、规模扩展（行数×10/×100）、双后端执行 | 框架自身开销（相对裸脚本）、峰值内存、兼容性/失败模式 | T07、T13、T14 |

**RQ2 对照实验设计**（建议做成论文的差异化实验）：

| 操作 | 传统方式 | EdgeFault-Bench |
|---|---|---|
| 增加一个数据集 | 修改多个数据处理脚本 | 独立 Dataset Plugin 包 |
| 增加一个模型 | 修改训练与评价脚本 | 独立 Model Plugin 包 |
| 更换数据划分 | 修改实验代码 | 修改 Task Manifest |
| 检查数据泄漏 | 人工检查代码 | 自动审计（fail closed） |
| 复现实验 | 手动配置环境与参数 | 读取版本化任务与结果包 |

尽量由第三方（非核心开发者）完成至少一次独立集成并记录过程；作者自维护的示例插件只能作为
机制演示，不作为外部采用证据（延续仓库现有 evidence-boundary 纪律）。

## 6. 贡献 ↔ 实验 ↔ 证据 映射（每项创新都要有实验证明）

| 贡献 | 证明它的实验 | 产出证据（入库物） |
|---|---|---|
| C1 形式化规范 | RQ1：TLC 不变量检查报告 + mutation 检出矩阵 | `formal/ExperimentLifecycle.tla`、TLC 输出、mutation 报告 JSON |
| C2 可扩展架构 | RQ2：Gearbox/KAIST 适配器 + 外部模型插件独立环境端到端 | 独立插件仓库 ×2、集成审计文档（沿用 `docs/*-audit-*.md` 体例） |
| C3 完整性验证与溯源 | RQ1+RQ3：错误注入全部被拒绝且错误原因明确；结果包→干净环境重建 | mutation 检出矩阵、复现审计文档、结果包 v2 |
| C4 实证评估 | RQ1–RQ4 全矩阵 + 框架开销基准 | `results/` 各版本目录 + 论文表格生成脚本 |

**RQ1 反例清单（mutation catalogue 摘要，全文见 formal-spec-draft §6）**：跨分区共享
recording、测试窗重标为训练、预处理在测试集上拟合、损坏源文件哈希、丢弃/重复 seed、剥离
环境溯源字段、插件身份不匹配、训练插件触碰测试视图。每项必须被 ≥1 机制检出，否则设计失败。

## 7. 论文结构（冻结）

| 章节 | 主要内容 |
|---|---|
| 1. Introduction | 工业 ML 实验可复现性与软件完整性问题；running example（一次被拦截的泄漏实验）；四贡献 |
| 2. Related Work | 故障诊断 Benchmark 与标准化（DG benchmark、IEEE Access 2025）、DomainBed、aeon、MLflow、契约式设计/形式化验证在实验系统中的应用 |
| 3. Requirements and Formal Specification | 软件需求（三类用户 → 六条不变量）、实验状态机、转换 guard |
| 4. System Architecture | 四模块设计；插件边界与测试集访问控制 |
| 5. Implementation | 插件系统、CLI、数据结构（分层存储）、验证器与 TLA+/Hypothesis/Mutation 工具链 |
| 6. Evaluation | RQ1–RQ4；含框架自身开销与威胁效度（threats to validity） |
| 7. Discussion | 适用范围（非安全关键、非流式监控）、局限（抽象层验证 ≠ 实现证明）、未来工作 |
| 8. Conclusion | 软件工程贡献总结 |

## 8. 冻结的范围边界（继承 Decision 0005 中仍有效的部分）

**继续有效的 Non-goals**：非实时监测/报警/控制系统；不做安全认证声明；不做跨数据集绝对分数
可比的统一本体；不做 AutoML/leaderboard/测试域调参；不为堆规模而积累数据集。

**新增边界**：

- 论文不声称"首个跨工况故障诊断 Benchmark"（先行工作明确存在）；差异化点是软件机制。
- 形式化层级 = 抽象生命周期模型（TLA+），实现层用运行时契约 + PBT + mutation 覆盖。
- 硬件结论表述为 CPU inference cost，除非后续批次补真实边缘设备。
- 低标签实验须拆分 window-level 与 recording-level 两种口径；高斯噪声实验表述为 synthetic
  noise robustness，与 sensor/channel perturbation 分开。

## 9. 与现有仓库文档的关系

- Decision 0005 的**产品定位**（可安装、可审计的评估框架）保留；其 **JOSS 投稿目标**由
  Decision 0006（Proposed）修订为 SEN。
- `docs/evaluation-protocol.md` 的协议原则（recording 互斥、验证集选择、测试集一次性使用）
  升级为 INV-1…INV-6 的运行时强制，而不是文档承诺。
- `docs/state-of-the-field.md` 在 T18 更新 2024–2026 相关工作（含 SEN/形式化方法应用于
  实验系统的近作）。
- `ROADMAP.md` 在本设计冻结后重写为 SEN 批次计划（T04–T20），此前不动。
- v1.0/v1.1 已发布的 HUST/Mehran 结果**继续有效**，降级为框架的验证性案例数据。

## 10. 审查通过判据（本文件的 DoD）

- [ ] 四个 RQ、四个贡献、四模块边界无异议；
- [ ] 六条不变量（formal-spec-draft §3）覆盖了全部三类用户的核心完整性诉求；
- [ ] mutation catalogue（formal-spec-draft §6）被认可为 RQ1 的主证据形式；
- [ ] 差距分析（architecture-gap-analysis §2）的 P0 清单与验收标准被接受；
- [ ] Claim 纪律（§2.1）与 Non-goals（§8）被明确接受。
