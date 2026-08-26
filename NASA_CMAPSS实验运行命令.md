# NASA C-MAPSS FD001–FD004 实验运行命令

本流程按 FD001、FD002、FD003、FD004 顺序独立运行。每个子集均采用官方训练/测试文件、`RUL <= 30` 二分类标签、相同特征、固定请求阈值 0.50，以及 Logistic Regression、Random Forest、Gradient Boosting 三种算法。

当前低配服务器使用 Spark `local[1]`、512 MB Driver、H2O 768 MB、单线程训练。仍保留论文要求的 10 折交叉验证和树模型 100 棵树，但不保留 10 个交叉验证子模型，以降低内存占用；这不会改变最终主模型在官方测试集上的评价协议。

## 1. 激活服务器环境

```bash
cd /opt/social_pdm_system
source .venv/bin/activate
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
export SPARK_LOCAL_IP=127.0.0.1
```

## 2. 检查四个子集

```bash
for subset in FD001 FD002 FD003 FD004; do
  for prefix in train test RUL; do
    test -f "data/CMAPSSData/${prefix}_${subset}.txt" \
      && echo "OK: ${prefix}_${subset}.txt" \
      || echo "MISSING: ${prefix}_${subset}.txt"
  done
done
```

## 3. 逐个运行

四个子集必须使用同一套参数独立训练和测试。默认的随机森林和 GBT 都完整训练 100 棵树，GBT 不启用提前停止，以保证子集间的参数口径一致。Spark shuffle 分区由运行脚本传入，不在 Python 代码中固定。

```bash
bash autodl/run_cmapss_subset.sh FD001
bash autodl/run_cmapss_subset.sh FD002
bash autodl/run_cmapss_subset.sh FD003
bash autodl/run_cmapss_subset.sh FD004
```

运行日志分别保存为：

```text
logs/cmapss-FD001-comparison.log
logs/cmapss-FD002-comparison.log
logs/cmapss-FD003-comparison.log
logs/cmapss-FD004-comparison.log
```

单个子集的结果位于：

```text
data/autodl_runtime/paper_run/cmapss/FD00x/classification_comparison/comparison.json
```

## 4. 一次性顺序运行

```bash
bash autodl/run_cmapss_all_subsets.sh
```

该脚本不是并行运行；只有前一个子集成功，才会运行下一个。

## 5. 汇总已有结果

```bash
python scripts/aggregate_cmapss_results.py \
  --results-root data/autodl_runtime/paper_run/cmapss \
  --out-dir data/autodl_runtime/paper_run/cmapss/summary
```

汇总输出：

```text
data/autodl_runtime/paper_run/cmapss/summary/cmapss_all_subsets.json
data/autodl_runtime/paper_run/cmapss/summary/cmapss_all_subsets.md
```

## 6. 查看模型指标

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("data/autodl_runtime/paper_run/cmapss")
for subset in ("FD001", "FD002", "FD003", "FD004"):
    path = root / subset / "classification_comparison" / "comparison.json"
    report = json.loads(path.read_text())
    for name, model in report["models"].items():
        metric = model["operating_metrics"]
        print(subset, name, "F1=", round(metric["f1"], 4),
              "Recall=", round(metric["recall"], 4),
              "PR-AUC=", round(metric["pr_auc"], 4))
PY
```

## 7. 资源覆盖参数

在更高配置服务器上可以覆盖默认值：

```bash
CMAPSS_MASTER='local[4]' \
CMAPSS_DRIVER_MEMORY=4g \
CMAPSS_H2O_MEMORY=6G \
CMAPSS_H2O_THREADS=4 \
CMAPSS_SHUFFLE_PARTITIONS=16 \
CMAPSS_TREES=100 \
CMAPSS_FOLDS=10 \
bash autodl/run_cmapss_subset.sh FD001
```

不同子集的指标必须分别报告，不应把 FD001–FD004 的所有轨迹行合并计算一个总 F1。
