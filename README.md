# Social PDM Spark prototype

本目录是论文架构的可部署原型

## 两种部署 Profile

| Profile | 环境 | 服务链路 | 适用结论 |
|---|---|---|---|
| Docker Compose | 普通 Linux VM / 裸金属 | Kafka、MongoDB、NiFi、Spark、H2O 等完整原型 | 单节点服务集成实验；多节点后可继续验证扩缩容 |
| AutoDL 无 Docker | AutoDL 普通容器 | JSON 文件流、PySpark `local[*]`、Parquet、KS/H2O | E1、E3、单节点加速回放；不证明 Kafka/K8s 性能 |

## 论文完整实验

E1/E2 新增代码已经实现五类质量真值与检测、accepted/quarantine 路由、无损 Stream/Batch 任务路由、路由审计，以及固定单节点容量下的负载控制对照。正式运行前建议先执行：

```bash
bash autodl/run_e1_e2_smoke.sh
```

独立命令、产物说明和验收口径见 [E1_E2实验运行命令.md](E1_E2实验运行命令.md)。

在AutoDL准备好Java 17、Python依赖和`data/CMAPSSData`后，分别运行C-MAPSS基准实验和30天模拟社交遥测实验：

```bash
bash autodl/run_cmapss_all_subsets.sh \
  data/CMAPSSData \
  data/autodl_runtime/cmapss

bash autodl/run_30d_v4_60gb.sh
```

实验输出按数据集和场景分别保存：

- C-MAPSS FD001-FD004结果：`data/autodl_runtime/cmapss`
- 30天模拟遥测数据：`data/autodl_runtime/scenarios/concept_drift_30d_v4/telemetry`
- E1、E2、E3及系统性能结果：`data/autodl_runtime/scenarios/concept_drift_30d_v4/results`
- 模拟场景PNG图表：`reports/scenarios/concept_drift_30d_v4`

所有图表只读取实际实验产物。独立场景配置位于`configs/scenarios/`。

## Web Dashboard访问

Dashboard包含四个页面：总览、`model-derived server health status`、
`threshold-based failure-warning alerts`和`TreeSHAP explanations`。完整E3实验默认只对
独立测试期最后2天生成Dashboard告警历史，并为每台逻辑服务器保留最新预测；页面请求本身
不会训练模型。

新增的Dashboard数据位于：

- `adaptive_experiment/server_health_snapshot.csv`：每台逻辑服务器的最新原始遥测和预测概率；
- `adaptive_experiment/threshold_alerts.csv`：最近2天内超过校准阈值的告警，默认最多2,000条；
- `adaptive_experiment/shap_alert_explanations.csv`：告警的TreeSHAP贡献值。

可在运行E3命令时通过`--dashboard-days 1`或`--dashboard-days 2`选择推理范围。实验完成后启动：

```bash
PDM_RUNTIME=data/autodl_runtime/scenarios/concept_drift_30d_v4/results \
python -m flask --app apps.dashboard run --host 0.0.0.0 --port 6008
```

访问路径：`/`、`/server-health`、`/failure-alerts`和`/explanations`。健康页可在完整最新快照上
交互调整阈值；告警历史页可从校准阈值向上调高阈值。页面操作不会覆盖实验校准阈值或修改结果文件。
