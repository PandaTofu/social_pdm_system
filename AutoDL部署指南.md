# AutoDL 无 Docker 部署指南

本方案适用于 AutoDL 普通 `autodl-container-*` 实例。它不在容器内安装 Docker，也不运行 Kubernetes。AutoDL 官方说明普通容器内部不支持 Docker；完整 Docker/Kafka/NiFi/MongoDB 方案需使用裸金属或普通 VM。

本 profile 的真实运行链路如下：

```text
Python generator -> NDJSON batch files -> PySpark Structured Streaming local[*]
                 -> validated Parquet / quarantine JSON / predictions Parquet / metrics JSON
                 -> KS drift monitor and optional local H2O training
```

它可以验证 E1（schema 与质量控制）、E3（KS 漂移检测与重训）和 E2 的**单机加速文件回放压力**。它不能用于报告 Kafka lag、NiFi 性能、MongoDB 吞吐、Kubernetes executor 扩缩容或多节点横向扩展。

## 1. 获取代码与创建环境

在 AutoDL 的 JupyterLab Terminal 或 SSH 终端执行：

```bash
cd /root/autodl-tmp  # 优先使用 AutoDL 数据盘；按实际挂载路径调整
git clone https://github.com/PandaTofu/social_pdm_system.git
cd social_pdm_system

conda create -n social-pdm python=3.11 openjdk=17 -y
conda activate social-pdm
export JAVA_HOME="$CONDA_PREFIX"
export PATH="$JAVA_HOME/bin:$PATH"

python -m pip install --upgrade pip
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r autodl/requirements-autodl.txt
java -version
spark-submit --version
```

如果 `conda` 不存在，使用现有 Python 虚拟环境；但仍需要 Java 17。不要使用 `sudo apt install docker`，当前容器并没有运行 Docker daemon 所需权限。

## 2. 启动开发实验

```bash
cd /root/autodl-tmp/social_pdm_system
conda activate social-pdm
export JAVA_HOME="$CONDA_PREFIX"
export PATH="$JAVA_HOME/bin:$PATH"
bash autodl/run_autodl_experiment.sh
```

该脚本会生成 50 个逻辑服务器、7 天、每分钟一条主遥测的开发数据，随后以 10,000 条/文件、每秒一个文件的速度写入 Spark 文件流 inbox。50 个逻辑服务器只存在于数据中，不代表需要 50 台云服务器。

实时查看日志：

```bash
tail -f logs/autodl-spark.log
```

实验结束后停止 Spark：

```bash
ps -ef | grep '[s]ocial-pdm-autodl-file-stream'
kill <Spark进程PID>
```

## 3. 查看结果与 Spark UI

```bash
# 每个 micro-batch 的 received/accepted/quarantined 统计
find data/autodl_runtime/output/metrics -type f | head

# 有效、隔离和预测输出
find data/autodl_runtime/output/validated -type f | head
find data/autodl_runtime/output/quarantine -type f | head
find data/autodl_runtime/output/predictions -type f | head
```

本地 Spark UI 默认端口为 `6006`。AutoDL 的端口映射/SSH 隧道策略因账号和实例而异；如果无法从浏览器直接访问，请在 AutoDL 控制台配置自定义服务或使用 SSH 隧道。不要尝试开放 Kafka/MongoDB 端口，因为本方案不运行这些服务。

## 4. 漂移检测与 H2O

`ml/drift_monitor.py` 接受两个同字段的 NPZ 特征窗口，计算每个关键特征的 KS D 并输出 JSON。例如：

```bash
python ml/drift_monitor.py \
  --reference /path/to/reference.npz \
  --current /path/to/current.npz \
  --threshold 0.20 \
  --out data/autodl_runtime/drift_report.json
```

H2O 是可选的离线模型训练组件，仍主要使用 CPU。先将 validated Parquet 导出为 CSV，再运行 `ml/train_h2o.py`；在 AutoDL 模式中应使用本地 H2O 进程，而不是 Docker 地址 `http://h2o:54321`。

## 5. 论文表述边界

推荐在论文中描述为：

> The AutoDL deployment is a single-node, Docker-free PySpark Structured Streaming prototype using accelerated file replay. It evaluates schema validation, data-quality isolation, prediction and concept-drift monitoring. It does not constitute a benchmark of Kafka, NiFi, MongoDB, Kubernetes, or horizontal autoscaling.

CMAPSS 仍用于方法正确性；模拟后端遥测仍用于领域实用性。E2 在此环境只报告 batch duration、文件摄入率、CPU/memory 与 P95/P99 processing latency，不报告 Kafka lag 或 executor 扩缩容。
