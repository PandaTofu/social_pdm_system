# Social PDM Spark prototype

本目录是论文架构的可部署原型，不替代论文中对真实集群实验的说明。最新代码结构、指标契约、图表来源和完整实验命令见 [系统代码结构与实验设计.md](系统代码结构与实验设计.md)；早期组件设计见 [详细系统设计.md](详细系统设计.md)。

## 两种部署 Profile

| Profile | 环境 | 服务链路 | 适用结论 |
|---|---|---|---|
| Docker Compose | 普通 Linux VM / 裸金属 | Kafka、MongoDB、NiFi、Spark、H2O 等完整原型 | 单节点服务集成实验；多节点后可继续验证扩缩容 |
| AutoDL 无 Docker | AutoDL 普通容器 | JSON 文件流、PySpark `local[*]`、Parquet、KS/H2O | E1、E3、单节点加速回放；不证明 Kafka/K8s 性能 |

AutoDL 用户请直接阅读 [AutoDL部署指南.md](AutoDL部署指南.md)，不要在 AutoDL 容器内部安装 Docker。

## 论文完整实验

在AutoDL准备好Java 17、Python依赖、`data/development.ndjson`和`data/CMAPSSData`后运行：

```bash
bash autodl/run_complete_paper_experiment.sh
```

脚本依次完成漂移检测、C-MAPSS Logistic/RF/GBT同协议比较与非回归检查、自适应/消融实验、单节点Spark性能实验和统一绘图。输出位于`data/autodl_runtime/paper_run`及`reports/generated`。所有图只读取真实实验产物。

独立场景配置位于`configs/scenarios/`。运行真正改变特征—故障关系的概念漂移场景且不覆盖历史结果：

```bash
bash autodl/run_telemetry_scenario.sh configs/scenarios/concept_drift_v2.json
```

实验完成后可启动最小业务Dashboard：

```bash
PDM_RUNTIME=data/autodl_runtime/paper_run \
python -m flask --app apps.dashboard run --host 0.0.0.0 --port 8090
```

## 快速启动（Linux）

1. 安装 Docker Engine 与 Docker Compose plugin。
2. `cp .env.example .env`，修改密码与对外端口。
3. `docker compose up -d`。
4. `docker compose exec generator python /app/generate_telemetry.py --config /app/configs/experiment.yaml --output /data/development.ndjson` 生成开发数据。
5. 运行 `bash scripts/create_topics.sh`，再将生成器以 `--kafka-bootstrap kafka:29092` 运行，把事件推送至 `telemetry-raw`。
6. 运行 `bash scripts/submit_spark.sh`；作业会把有效事件写入 Parquet，并将无效事件写到 `telemetry-quarantine`。

首次运行前，请按云端实际路径设置 Spark 的 `S3_ENDPOINT`、access key、secret 和 bucket。任何吞吐/延迟结论都必须采集 Spark event log、Kafka lag 和原始运行 CSV。

## Web UI

- Spark master: `http://<host>:8080`
- NiFi: `http://<host>:8443/nifi`
- H2O Flow: `http://<host>:54321`
- MinIO: `http://<host>:9001`
- Grafana: `http://<host>:3000`

默认密码只用于本地开发，部署前务必更改。
