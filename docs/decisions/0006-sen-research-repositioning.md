# Decision 0006: SEN Research Repositioning

- Status: **Accepted**
- Proposed: 2026-11-06
- Accepted: 2026-11-06
- Scope: 论文投稿目标、研究问题与证据体系；不改变 v1.x 已冻结结果
- Supersedes: Decision 0005 的 JOSS 投稿目标（其产品定位与 non-goals 保留）

## Decision

将 EdgeFault-Bench 的论文目标从 JOSS 研究软件论文改为投往 Software Engineering Nexus
（SEN），论文对象从"研究软件产品"升级为"实验完整性问题的软件工程解法"：

> A formally specified and extensible software framework that enforces the integrity,
> reproducibility, and provenance of industrial fault-diagnosis experiments.

研究问题由"模型能否跨工况泛化/哪种模型更轻量"改为四个关于框架自身的可检验问题
（RQ1 完整性保证、RQ2 独立扩展、RQ3 可追溯复现、RQ4 异构规模支撑）。

## Motivation

1. SEN 的公开范围包含软件工程与形式化方法/软件验证，与计划引入的形式化规范 +
   模型检查 + 运行时契约验证路线吻合。
2. 跨工况故障诊断 Benchmark 赛道已有强先行工作（RESS 2024 跨域研究、IEEE Access 2025
   标准化框架）；以 Benchmark 覆盖面竞争不构成差异化贡献。
3. 仓库现有资产（fail-closed 审计、插件注册表、schema 体系、证据链纪律）距离"实验完整性的
   软件机制"这一主张更近，增量工作有明确的学术对应物。

## Research design documents

- `docs/research-design.md` — T01 研究设计冻结（定位、贡献、RQ、模块、实验矩阵）
- `docs/formal-spec-draft.md` — T02 形式化规范草案（状态机、INV-1…INV-6、验证计划）
- `docs/architecture-gap-analysis.md` — T03 差距分析与目标 v2 契约

## Consequences

- 第一批（T01–T03）完成前不启动架构改造；本文档与三份设计文档通过审查后，第二批
  （T04–T08）开工，ROADMAP.md 届时重写为 SEN 批次计划。
- 论文表述遵守 claim 纪律：在 T09/T11 完成前，"Formally Specified" 为目标定位；
  最终表述为 formal specification + model checking + runtime contract verification +
  property-based testing，不声称实现级证明。
- JOSS 路线下的工程任务（打包、文档站、DOI、外部使用）保留为支撑工作。
- SEN 为新刊且暂无完整公开评审细则，本定位是投稿策略而非录用保证；写作阶段复核其
  Guide for Authors。

## Ratification record

2026-11-06：基于外部顾问的定位审查（阶段一至八）起草。维护者于同日审查三份设计文档
（`docs/research-design.md` §10 的 DoD）并指示继续，本记录改为 Accepted；第二批
（T04–T08）随之开工，完成情况见 `ROADMAP.md` 与 `CHANGELOG.md` 的 1.2.0 段。
