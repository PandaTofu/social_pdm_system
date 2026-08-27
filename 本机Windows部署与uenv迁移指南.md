# 本机 Windows 部署与可迁移环境指南

## 结论

本机硬件（i7-12800HX、16 GB 内存、NVMe SSD）可以运行正式的 30 天、100 台逻辑服务器实验。GPU 不参与 Spark/H2O 随机森林训练。运行前建议关闭大型应用，并让 D 盘至少保留 15 GB 临时空间；更稳妥的余量是 50 GB。

项目不修改系统 Java 25。部署脚本会把 JDK 17 和 Python 3.11 放到项目的 `.runtime`，所有实验仅在当前 PowerShell 进程中使用它们。

## 一键部署

在仓库根目录运行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1 -RebuildVenv
```

旧 `.venv` 会重命名保留，不直接删除。脚本执行后可验证：

```powershell
.\.runtime\jdk-17\bin\java.exe -version
.\.venv\Scripts\python.exe -c "import pyspark,h2o; print(pyspark.__version__, h2o.__version__)"
```

## 运行正式模拟数据实验

```powershell
.\scripts\run_30d_paper_experiment_windows.ps1
```

执行顺序是：生成数据、质量契约检查、漂移窗口与 KS 检测、Spark 数据准备、独立 H2O 进程训练与消融、Spark 系统性能实验、生成论文图表。Spark 与 H2O 已拆成两个进程，不再同时占用 JVM 内存。

中断后若 Spark 准备已经完成，可只恢复 H2O 阶段：

```powershell
.\.venv\Scripts\python.exe ml\train_adaptive_h2o.py `
  --prepared-manifest data\local_runtime\scenarios\concept_drift_30d_v3\results\adaptive_experiment\prepared_manifest.json `
  --trees 80 --explain-rows 500 --h2o-memory 5G --h2o-threads 6
```

## “uenv 镜像”能否替代 Docker

不能把 Windows `.venv` 目录直接当作镜像复制到任意 PC：其中包含绝对路径、启动脚本和本机二进制依赖，换目录或换系统后可能失效。`uenv` 也不是 Windows/Spark 的通用镜像格式。

可移植方案分三层：

1. 首选：保留 Docker 镜像，用于支持 Docker 的 Linux/Windows 主机。
2. Windows 同架构 PC：复制 Git 仓库，通过本指南的一键脚本重建 JDK 17 与 Python 环境；固定依赖版本保证结果一致。
3. 无外网 Windows PC：制作“离线运行包”，其中包含项目源码、便携 JDK 17、uv、Python 3.11 安装包及 Python wheelhouse；目标 PC 解压后重新创建 `.venv`。这类似离线镜像，但仍是重建环境，不是直接搬运 `.venv`。

因此，本项目将 `.runtime` 和 `.venv` 视为本机运行状态，不提交 Git。Git 中保存配置、固定依赖和部署脚本，兼顾可重复性与体积。
