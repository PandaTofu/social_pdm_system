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

# Java 使用系统包，避免 Conda 默认 channel 中 OpenJDK 包不可用的问题
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-17-jre-headless

# AutoDL 常见的 Miniconda 路径；创建到数据盘，避免占用容器层
/root/miniconda3/bin/conda create -p /root/autodl-tmp/envs/social-pdm python=3.11 -y \
  --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
export PDM_ENV=/root/autodl-tmp/envs/social-pdm
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$PDM_ENV/bin:$JAVA_HOME/bin:$PATH"

python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r autodl/requirements-autodl.txt
java -version
spark-submit --version
```

### GitHub 无法访问时：使用 Git bundle（推荐备用方案）

若服务器可以 SSH 登录、但 `git clone` 无法连接 GitHub，可在本地已同步代码的电脑执行：

```powershell
git -C D:\Work\2608月订单\Spark论文修改\social_pdm_system bundle create social_pdm_system.bundle main
scp -P <SSH端口> social_pdm_system.bundle root@<服务器地址>:/root/
```

然后在服务器执行（不要直接 `git clone bundle`，以免默认分支未被检出）：

```bash
cd /root/autodl-tmp
mkdir -p social_pdm_system && cd social_pdm_system
git init
git fetch /root/social_pdm_system.bundle main
git checkout -B main FETCH_HEAD
git remote add origin https://github.com/PandaTofu/social_pdm_system.git
```

后续更新时，在本地重新生成 bundle 并上传，再在服务器仓库运行
`git fetch /root/social_pdm_system.bundle main && git merge --ff-only FETCH_HEAD`。这样不依赖服务器能够访问 GitHub。

若你的 AutoDL 镜像中 Conda 不在 `/root/miniconda3/bin/conda`，先执行 `which conda`，再将上述 Conda 可执行文件路径替换为实际路径。

如果 `conda` 不存在，使用现有 Python 虚拟环境；但仍需要 Java 17。不要使用 `sudo apt install docker`，当前容器并没有运行 Docker daemon 所需权限。

## 2. 启动开发实验

```bash
cd /root/autodl-tmp/social_pdm_system
export PDM_ENV=/root/autodl-tmp/envs/social-pdm
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$PDM_ENV/bin:$JAVA_HOME/bin:$PATH"
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

## 6. 运行完整论文实验并生成全部图表

准备好`data/development.ndjson`与`data/CMAPSSData`后执行：

```bash
cd /root/autodl-tmp/social_pdm_system
export PDM_ENV=/root/autodl-tmp/envs/social-pdm
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$PDM_ENV/bin:$JAVA_HOME/bin:$PATH"

git pull --ff-only
python -m pip install -r autodl/requirements-autodl.txt
bash autodl/run_complete_paper_experiment.sh
```

完整入口依次运行C-MAPSS Logistic/RF/GBT同协议分类比较与非回归检查、KS漂移检测、静态/重训/加权/完整方法消融、逐日稳定性、单节点Spark规模实验、TreeSHAP解释和统一绘图。实验数据保存在`data/autodl_runtime/paper_run`，图片保存在`reports/generated`。

如果服务器无法连接GitHub，继续使用第1节的Git bundle更新方法。

### 运行独立概念漂移场景

以下命令生成并运行`concept_drift_v2`，不会覆盖旧的`paper_run`或`covariate_shift_v1`结果：

```bash
bash autodl/run_telemetry_scenario.sh \
  configs/scenarios/concept_drift_v2.json
```

默认产物位置：

- 原始模拟数据：`data/autodl_runtime/scenarios/concept_drift_v2/telemetry.ndjson`
- Spark/H2O结果：`data/autodl_runtime/scenarios/concept_drift_v2/results`
- 图表：`reports/scenarios/concept_drift_v2`
- 日志：`logs/concept_drift_v2-*.log`

脚本包含生成数据、契约验证、KS漂移检测、静态/重训练/加权/阈值校准消融、系统基准、结果诊断和绘图。若目标路径已存在，脚本会拒绝覆盖；重跑时应指定新的源数据、结果和图表路径。

场景生成时间和Spark训练/反馈/测试切分统一使用UTC。不要依赖AutoDL宿主机时区解释第1–7天。

## 7. 启动最小业务Dashboard

完整实验结束后执行：

```bash
export PDM_RUNTIME=/root/autodl-tmp/social_pdm_system/data/autodl_runtime/paper_run
python -m flask --app apps.dashboard run --host 0.0.0.0 --port 8090
```

Dashboard只读取实际Spark、漂移、模型和SHAP文件。没有生成的指标会显示为不可用，不会用演示数字代替。通过AutoDL自定义服务或SSH端口转发访问8090端口。
