# 论文要求与当前实现差异清单

更新日期：2026-08-25
论文基准：`D:\Work\2608月订单\Spark论文修改\报告\Spark论文v1.docx`
代码基准：`social_pdm_system`，当前本地 HEAD `a73d814`

## 1. 使用原则

本清单采用以下判定规则：

1. 核心算法的 Precision、Recall、F1、PR-AUC、KS、阈值和 SHAP 等结论，必须使用实际运行产物，不能使用论文中的预设目标值代替。
2. 不改变核心算法结果、只影响生产化程度或系统部署完整性的内容，可以保留为架构设计，并记录在本文档的工程 Todo 中。
3. 如果某项差异会改变论文实验对象、评价周期、指标定义或增强模块的有效性结论，则不能只写 Todo；必须选择“补实验/补实现”或“修改论文表述”。
4. AutoDL 当前结果统一描述为：单节点、无 Docker/Kubernetes 的 PySpark Structured Streaming 文件回放实验。不得将其描述为多节点、Kafka/K8s 横向扩缩容实验。

## 2. 已完成且可直接用于论文的核心实验

| 项目 | 当前证据 | 可支持的结论 |
|---|---|---|
| C-MAPSS FD001 分类比较 | Logistic Regression、Random Forest、Gradient Boosting 使用相同训练/测试划分、`RUL <= 30` 标签和统一阈值协议 | 三种算法在 FD001 官方测试集上的 Precision、Recall、F1、PR-AUC 和 ROC-AUC 比较 |
| 模拟数据概念漂移 | `concept_drift_v2` 实际改变特征与故障标签之间的关系，并统一采用 30 分钟前兆窗口和标签窗口 | 模拟数据包含可识别的概念漂移，而不只是协变量漂移 |
| 自适应模型消融 | 静态 RF、无权重重训练、近期加权重训练、完整自适应模型 | 重训练、近期加权和阈值校准各自对性能的影响 |
| 漂移检测 | 两样本 KS 检验，同时报告 D 值和 p 值，并设置效应量与显著性双门槛 | 当前监控窗口存在特征分布漂移，并触发重训练流程 |
| TreeSHAP | H2O 对实际预测告警生成贡献值 CSV | 全局特征贡献和单条告警局部解释 |
| Spark 单节点性能 | 5万至50万行的实际处理耗时、吞吐量和显式 Schema/运行时推断对比 | 单节点文件读取和批处理规模证据；不能解释为端到端事件延迟或 K8s 扩缩容 |
| 数据质量隔离 | Spark 实际记录 accepted/quarantined 数量 | 当前已实现质量规则下的接纳率和隔离率 |
| 可重复图表 | 12 张 PNG/PDF 图和输入清单 `figure_manifest.json` | 图表数值可追溯到 JSON/CSV 实验产物 |

当前应写入论文的主要实测值：

- C-MAPSS FD001：Logistic Regression F1 = 0.6758，Random Forest F1 = 0.7087，Gradient Boosting F1 = 0.7136。
- 模拟数据静态 RF：Precision = 0.0688，Recall = 0.0253，F1 = 0.0370，PR-AUC = 0.0434。
- 模拟数据完整自适应模型：Precision = 0.8088，Recall = 0.6646，F1 = 0.7297，PR-AUC = 0.7421。
- 最大 KS D = 0.2218，判定检测到漂移。
- 单节点50万行吞吐量 = 112,439.88 rows/s，整批处理耗时 = 4.447 s。
- 显式 Schema 相对运行时推断的本次实测读取加速比 = 6.01x。

## 3. 可直接列入工程 Todo 的非核心差异

这些项目不会改变已经取得的核心模型指标，可以暂不实现，但论文应将其称为“架构设计”“部署建议”或“未来工作”，不能称为已完成并实测。

### TODO-E01：Kubernetes 自动扩缩容

- 论文要求：通过 Kubernetes API 增加或删除 Spark executor Pod，并在流量激增时验证弹性扩容。
- 当前实现：只有 `k8s/streaming-sparkapplication.yaml` 模板，AutoDL 未运行 Kubernetes，也没有 executor Pod 数量的真实实验记录。
- 对核心算法影响：无。不会改变当前 H2O 模型的 Precision、Recall、F1 或 PR-AUC。
- 后续验收：在至少两个 worker 节点的 K8s 集群中保存 executor 数量、扩容时间、Spark event log 和 Pod 指标。
- 论文当前处理：移入架构设计/未来工作；删除“已从4个扩展到18个 executor、12秒完成”等未经本次实验验证的结果性描述。

### TODO-E02：逐事件端到端延迟

- 论文定义：`L = t_output - t_ingestion`。
- 当前实现：`system_benchmark.json` 的 `duration_seconds/latency_ms` 是整个批次的墙钟处理耗时。
- 对核心算法影响：无。不会改变模型分类结果。
- 后续验收：为每条记录保存 `ingest_time`、`prediction_output_time` 和 `event_latency_ms`，报告 P50/P95/P99。
- 论文当前处理：只能报告“whole-batch processing duration”，不能写成逐事件端到端延迟小于500 ms。

### TODO-E03：NiFi 自动化数据流

- 论文要求：NiFi 承担采集、ETL 和 NiFi→MongoDB/Spark 接入。
- 当前实现：Docker Compose 可以启动 NiFi，但仓库没有可自动导入并运行的 NiFi flow definition。
- 对核心算法影响：无。AutoDL 文件回放绕过 NiFi 后仍能完成算法实验。
- 后续验收：提交版本化 NiFi flow、参数上下文、错误路由和运行日志。
- 论文当前处理：NiFi 保留为完整架构组件，原型实验注明使用文件回放替代接入层。

### TODO-E04：MongoDB 分片集群

- 论文要求：MongoDB sharded cluster、config servers 和 mongos。
- 当前实现：Docker Compose 只有单个 MongoDB 服务，AutoDL 实验未使用 MongoDB。
- 对核心算法影响：无。当前训练和评估使用 NDJSON、Parquet 和 CSV。
- 后续验收：部署 config server replica set、至少两个 shard 和 mongos，并记录写入/查询性能。
- 论文当前处理：分片集群写为目标架构，不写成当前 AutoDL 已部署组件。

### TODO-E05：Kafka 真实积压和流量突发实验

- 论文要求：消息中间件、队列深度、backpressure 和突发流量下吞吐量。
- 当前实现：完整 Docker 路径有 Kafka 配置，但本次正式 AutoDL 结果来自加速文件回放，不包含 Kafka lag。
- 对核心算法影响：无。
- 后续验收：保存 Kafka consumer lag、每个 micro-batch 输入行数、处理耗时和积压恢复时间。
- 论文当前处理：当前吞吐量写成“单节点文件回放/批处理吞吐量”，不写成 Kafka 流吞吐量。

### TODO-E06：H2O 模型在线热更新

- 论文要求：漂移触发重训练后，验证并部署新模型到在线推理路径。
- 当前实现：KS 能控制离线 H2O 重训练实验，但 `spark/streaming_job.py` 的实时评分仍使用透明的确定性风险公式，没有自动加载新版 H2O/MOJO 模型。
- 对核心算法影响：不影响当前离线 H2O 对比指标，但影响“在线闭环已部署”的系统声明。
- 后续验收：模型注册、版本切换、原子热更新、回滚和在线预测一致性测试。
- 论文当前处理：可以写“原型实现了漂移触发的离线自适应重训练验证”；在线自动部署列为未来工作。

### TODO-E07：Prometheus/Grafana 完整监控

- 论文定位：建议性辅助组件。
- 当前实现：有 Prometheus 配置和业务 Dashboard，但没有完整 Grafana dashboard provisioning 与所有组件指标采集。
- 对核心算法影响：无。
- 论文当前处理：明确标为可选监控增强，不作为核心实验完成条件。

## 4. 不能只放进 Todo、需要修改论文或补实验的差异

以下项目涉及实验范围或增强模块结论。若不补实现，必须同步缩小论文表述。

### DECISION-D01：7天/50台逻辑服务器与12个月/500台服务器

- 论文描述：500台服务器、12个月、约2.62亿条数据，并评价六个月模型稳定性。
- 当前实测：50台逻辑服务器、7天；第1–3天训练，第5天反馈，第6–7天留出评估。
- 为什么不能只写 Todo：时间跨度直接决定“六个月稳定性”和“退化率”是否成立。
- 推荐选择：论文改为本次实际规模，并将12个月/500台标为扩展实验或未来工作。
- 替代选择：扩展生成器和训练协议，重新运行至少六个月的周期性评估和多次重训练。

### DECISION-D02：论文预写目标值与实际模型结果

状态：**已确认（2026-08-25）——论文采用实际运行结果。**

- 论文预写：增强 F1 = 0.867、六个月 F1 = 0.849、稳定性退化2.1%。
- 当前实测：完整自适应 F1 = 0.7297；仅有第6–7天逐日稳定性。
- 为什么不能只写 Todo：这些是核心算法结论。
- 已确认处理：第5章表格、正文和结论全部替换为实际结果；目标值只可保留在“Expected Outcomes”中，并与“Observed Results”分列。

### DECISION-D03：Su et al. 基线复现声明

状态：**已确认（2026-08-25）——论文采用当前 C-MAPSS 实际结果，不再要求贴合0.776。**

- 论文预写：C-MAPSS 基线 F1 约0.776，与 Su et al. 结果偏差小于5%。
- 当前实测：统一协议下 FD001 Random Forest F1 = 0.7087，Gradient Boosting F1 = 0.7136。
- 为什么不能只写 Todo：当前实验复现的是 C-MAPSS 算法基线，不是 Su et al. 整个平台的同条件复现。
- 已确认处理：改称“C-MAPSS benchmark algorithm comparison”，删除“faithfully reproduces Su et al. within 5%”。
- 替代选择：获得 Su et al. 完整数据划分、特征、标签和阈值协议后重新复现。

### DECISION-D04：仅 FD001 与 FD001–FD004

- 论文数据部分介绍 FD001–FD004 四个子集。
- 当前正式分类比较只运行 FD001。
- 为什么需要决定：如果论文声称覆盖完整 C-MAPSS，则证据不足；如果明确是 FD001 实验，则当前数据足够。
- 推荐选择：全文统一写为“NASA C-MAPSS FD001”；FD002–FD004 列入扩展实验。
- 替代选择：对四个子集使用相同协议重新运行并汇总均值/分组结果。

### DECISION-D05：五维质量控制是否完整实现

- 论文要求：Schema、范围、时序、完整性和跨字段五类规则，包含批次缺失率、采样间隔、记录数、CPU–内存相关性等阈值。
- 当前实现：已有必填字段、Schema 版本、部分范围、时间先后、重复隔离等规则，但并未逐项实现表4.1全部规则。
- 为什么需要决定：不影响 H2O 核心指标，但会影响 Enhancement 1“完整五维质量控制已验证”的结论。
- 推荐选择：论文改为“implemented quality gates covering five rule categories at prototype level”，并列出实际启用规则；不要宣称表4.1每条均已验证。
- 替代选择：补齐所有批次级、时序和跨字段规则及测试。

### DECISION-D06：三种社交数据类型与实际遥测数据

- 论文要求：结构化性能指标、半结构化应用日志、非结构化错误报告。
- 当前实现：正式模拟和模型实验以结构化遥测字段为主，只模拟了 Schema v1/v2 演进。
- 为什么需要决定：影响“multi-structured data”结论，但不改变当前结构化遥测上的模型指标。
- 推荐选择：将本次实验限定为“multi-version structured telemetry with schema evolution”；半结构化/非结构化接入写入未来工作。
- 替代选择：增加 nested JSON 日志和错误堆栈数据，并实现独立 Schema 与质量规则。

### DECISION-D07：任务委派增强是否算已实现

- 论文要求：四条规则将高优先级数据送入 stream path，将低优先级分析送入 batch path。
- 当前实现：数据含流量活动等场景字段，但没有独立规则引擎和两条实际执行路径。
- 为什么需要决定：这是 Enhancement 2 的核心组成，不应以一个配置字段代替已实现路由。
- 推荐选择：将其保留为架构设计和工程 Todo，不在实验结果中报告“94.3% routing accuracy”。
- 替代选择：实现规则引擎、路由审计日志和路由准确率测试。

### DECISION-D08：漂移检测的辅助指标

- 论文方法部分除 KS 外，还写入性能监控、特征重要性变化和集成模型分歧。
- 当前实现：核心触发器只有 KS D + p 值；F1/PR-AUC 在训练后报告，未实现特征重要性漂移和集成分歧触发。
- 为什么需要决定：影响方法定义，但当前 KS→重训练实验本身有效。
- 推荐选择：将论文算法明确为“KS-based drift trigger with post-hoc performance monitoring”，把另外两个指标列入未来工作。
- 替代选择：实现特征重要性距离和多模型分歧指标，设计组合触发规则。

### DECISION-D09：CMAPSS 非回归的含义

- 论文描述：增强架构重新运行 C-MAPSS 后，在延迟、吞吐量、F1 和数据质量上均不退化。
- 当前实现：非回归检查复用同一份干净数据和 RF 预测产物，证明显式 Schema/质量契约不改变模型指标；没有重新验证全栈延迟和吞吐量。
- 推荐选择：改称“prediction non-regression check”，不要称为完整平台非回归测试。

## 5. 建议保留的论文图表

| 图表 | 数据来源 | 论文用途 |
|---|---|---|
| C-MAPSS 标签/RUL 分布 | C-MAPSS comparison JSON | 说明标签定义和类别分布 |
| C-MAPSS 四指标算法比较 | Logistic/RF/GBT 实测结果 | 科学严谨性和算法基准比较 |
| C-MAPSS PR 曲线与 RF 混淆矩阵 | 三模型预测 CSV | 排名能力和分类错误结构 |
| C-MAPSS prediction non-regression | non-regression JSON | 证明质量契约不改变干净数据预测结果 |
| KS 特征漂移 | drift report | 漂移触发依据 |
| 模拟数据四模型消融 | adaptive comparison JSON | 静态、重训练、加权、阈值校准对比 |
| 阈值校准曲线 | H2O threshold curve | 说明高 Precision/Recall 权衡来源 |
| 第6–7天逐日稳定性 | daily hold-out metrics | 短期漂移后稳定性；不得标为六个月 |
| 单节点 Spark 规模图 | system benchmark | 吞吐量和整批处理耗时 |
| Schema/质量证据 | schema comparison + quality counts | E1 原型证据 |
| TreeSHAP | H2O contributions CSV | 全局与局部可解释性 |

## 6. 建议删除或改写的原论文结果声明

在没有新增实验产物前，不应继续作为“实际结果”使用：

- K8s executor 在12秒内从4扩展到18。
- 端到端延迟为387 ms或463 ms。
- Kafka 突发吞吐量为112,400 records/s；当前112,439.88是单节点文件/批处理吞吐量，二者口径不同。
- 任务委派准确率为94.3%。
- 六个月触发4次自动重训练。
- 六个月模型退化率为2.1%。
- SHAP 在91.2%的预测中“正确识别”前三特征；当前只能证明生成了真实贡献值，缺少人工真值评审协议。
- 数据质量由88.4%提升到96.7%，除非先定义质量分数公式并按实际规则重新计算。
- C-MAPSS 完整平台延迟降低36.6%、吞吐量提高9.6%。

## 7. 推荐的当前论文定位

在不继续实现生产化组件的情况下，建议将系统定位为：

> A single-node PySpark proof-of-concept that evaluates explicit schema processing, prototype-level in-stream quality gates, KS-triggered adaptive H2O Random Forest retraining, calibrated failure alerts, and TreeSHAP explanations using NASA C-MAPSS FD001 and a seven-day concept-drift telemetry simulation.

对应中文表述：

> 本研究实现了一个基于单节点 PySpark 的概念验证原型，利用 NASA C-MAPSS FD001 和包含概念漂移的7天模拟后端遥测数据，对显式 Schema 处理、原型级流内质量规则、KS 触发的 H2O 随机森林自适应重训练、告警阈值校准及 TreeSHAP 可解释性进行了实际验证。Kubernetes 弹性扩缩容、完整 NiFi/MongoDB 分片部署和逐事件端到端延迟测量作为后续工程工作。

该定位与现有代码和实际图表一致，同时保留论文的核心算法贡献。
