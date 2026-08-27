[CmdletBinding()]
param(
    [string]$Config = "configs/scenarios/concept_drift_30d_v3.json",
    [string]$Source = "data/local_runtime/scenarios/concept_drift_30d_v3/telemetry",
    [string]$RunRoot = "data/local_runtime/scenarios/concept_drift_30d_v3/results",
    [string]$Figures = "reports/scenarios/concept_drift_30d_v3_local",
    [int]$Trees = 80,
    [switch]$SkipSystemBenchmark
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
$Venv = Join-Path $ProjectRoot ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
$SparkSubmit = Join-Path $Venv "Scripts\spark-submit.cmd"
$JdkRoot = Join-Path $ProjectRoot ".runtime\jdk-17"

foreach ($required in @($Python, $SparkSubmit, (Join-Path $JdkRoot "bin\java.exe"), $Config)) {
    if (-not (Test-Path $required)) { throw "Missing prerequisite: $required. Run scripts\setup_windows.ps1 first." }
}
foreach ($target in @($Source, $RunRoot, $Figures)) {
    if (Test-Path $target) { throw "Refusing to overwrite existing experiment artifact: $target" }
}

$env:JAVA_HOME = $JdkRoot
$env:PATH = "$JdkRoot\bin;$Venv\Scripts;$env:PATH"
$env:PYSPARK_PYTHON = $Python
$env:PYSPARK_DRIVER_PYTHON = $Python
$env:SPARK_LOCAL_IP = "127.0.0.1"
$env:SPARK_LOG_LEVEL = "WARN"
$env:SPARK_LOCAL_DIRS = Join-Path $ProjectRoot ".runtime\spark-tmp"
New-Item -ItemType Directory -Force -Path (Split-Path $Source), $RunRoot, $Figures, "logs", $env:SPARK_LOCAL_DIRS | Out-Null

function Invoke-Checked {
    param([string]$File, [string[]]$Arguments, [string]$Log)
    & $File @Arguments 2>&1 | Tee-Object -FilePath $Log
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE. See $Log" }
}

Invoke-Checked $Python @("apps/generate_telemetry.py", "--config", $Config, "--output-dir", $Source) "logs/local-30d-generate.log"
Invoke-Checked $Python @("tests/validate_contract.py", $Source) "logs/local-30d-contract.log"
Invoke-Checked $Python @("autodl/make_drift_windows.py", "--source", $Source, "--config", $Config, "--out-dir", "$RunRoot/drift_windows") "logs/local-30d-drift-windows.log"
Invoke-Checked $Python @("ml/drift_monitor.py", "--reference", "$RunRoot/drift_windows/reference.npz", "--current", "$RunRoot/drift_windows/current.npz", "--threshold", "0.10", "--alpha", "0.05", "--out", "$RunRoot/drift_report.json") "logs/local-30d-drift.log"

Invoke-Checked $SparkSubmit @(
    "--master", "local[6]", "--driver-memory", "3g",
    "--conf", "spark.sql.shuffle.partitions=24",
    "--conf", "spark.sql.adaptive.enabled=true",
    "ml/prepare_adaptive_telemetry.py",
    "--source", $Source, "--output-dir", "$RunRoot/adaptive_experiment",
    "--drift-report", "$RunRoot/drift_report.json",
    "--scenario-name", "concept_drift_30d_v3", "--generator-config", $Config
) "logs/local-30d-spark-prepare.log"

# Spark has fully exited before this new H2O process starts.
Invoke-Checked $Python @(
    "ml/train_adaptive_h2o.py",
    "--prepared-manifest", "$RunRoot/adaptive_experiment/prepared_manifest.json",
    "--trees", "$Trees", "--explain-rows", "500",
    "--h2o-memory", "5G", "--h2o-threads", "6"
) "logs/local-30d-h2o.log"

if (-not $SkipSystemBenchmark) {
    Invoke-Checked $SparkSubmit @(
        "--master", "local[6]", "--driver-memory", "3g",
        "--conf", "spark.sql.shuffle.partitions=24",
        "spark/system_benchmark.py", "--source", $Source,
        "--out", "$RunRoot/system_benchmark.json",
        "--sizes", "50000", "100000", "250000", "500000", "1000000", "2000000", "4000000",
        "--compare-schema-read"
    ) "logs/local-30d-system-benchmark.log"
}

$figureArgs = @(
    "scripts/generate_complete_paper_figures.py",
    "--result", "$RunRoot/adaptive_experiment/adaptive_comparison.json",
    "--drift", "$RunRoot/drift_report.json",
    "--shap-contributions", "$RunRoot/adaptive_experiment/shap_alert_explanations.csv",
    "--out-dir", $Figures
)
if (-not $SkipSystemBenchmark) { $figureArgs += @("--system-benchmark", "$RunRoot/system_benchmark.json") }
Invoke-Checked $Python $figureArgs "logs/local-30d-figures.log"
Write-Host "Experiment complete. Results: $RunRoot"
Write-Host "Figures: $Figures"
