[CmdletBinding()]
param(
    [string]$PythonVersion = "3.11",
    [switch]$RebuildVenv
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RuntimeRoot = Join-Path $ProjectRoot ".runtime"
$UvCache = Join-Path $RuntimeRoot "uv-cache"
$PythonInstall = Join-Path $RuntimeRoot "python"
$JdkRoot = Join-Path $RuntimeRoot "jdk-17"
$Venv = Join-Path $ProjectRoot ".venv"

New-Item -ItemType Directory -Force -Path $RuntimeRoot, $UvCache, $PythonInstall | Out-Null
$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    throw "uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/ and rerun."
}

if (-not (Test-Path (Join-Path $JdkRoot "bin\java.exe"))) {
    $archive = Join-Path $RuntimeRoot "OpenJDK17.zip"
    $extractRoot = Join-Path $RuntimeRoot "jdk-extract"
    $jdkUri = "https://api.adoptium.net/v3/binary/latest/17/ga/windows/x64/jdk/hotspot/normal/eclipse?project=jdk"
    Write-Host "Downloading project-local JDK 17..."
    Invoke-WebRequest -Uri $jdkUri -OutFile $archive
    if (Test-Path $extractRoot) { Remove-Item -LiteralPath $extractRoot -Recurse -Force }
    New-Item -ItemType Directory -Path $extractRoot | Out-Null
    Expand-Archive -LiteralPath $archive -DestinationPath $extractRoot
    $java = Get-ChildItem -Path $extractRoot -Filter java.exe -Recurse |
        Where-Object { $_.FullName -match '[\\/]bin[\\/]java\.exe$' } |
        Select-Object -First 1
    if (-not $java) { throw "Downloaded JDK archive did not contain bin\java.exe" }
    $extractedJdk = Split-Path (Split-Path $java.FullName -Parent) -Parent
    Move-Item -LiteralPath $extractedJdk -Destination $JdkRoot
    Remove-Item -LiteralPath $extractRoot -Recurse -Force
    Remove-Item -LiteralPath $archive -Force
}

$env:UV_CACHE_DIR = $UvCache
$env:UV_PYTHON_INSTALL_DIR = $PythonInstall
& $uv.Source python install $PythonVersion
if ($LASTEXITCODE -ne 0) { throw "uv could not install Python $PythonVersion" }

if ((Test-Path $Venv) -and $RebuildVenv) {
    $backup = Join-Path $ProjectRoot (".venv_backup_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
    Move-Item -LiteralPath $Venv -Destination $backup
    Write-Host "Previous environment preserved at $backup"
}
if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
    & $uv.Source venv $Venv --python $PythonVersion --managed-python
    if ($LASTEXITCODE -ne 0) { throw "uv could not create .venv" }
}

$python = Join-Path $Venv "Scripts\python.exe"
& $uv.Source pip install --requirements (Join-Path $ProjectRoot "autodl\requirements-autodl.txt") --python $python
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }

$env:JAVA_HOME = $JdkRoot
$env:PATH = "$JdkRoot\bin;$Venv\Scripts;$env:PATH"
$env:PYSPARK_PYTHON = $python
$env:PYSPARK_DRIVER_PYTHON = $python

& (Join-Path $JdkRoot "bin\java.exe") -version
& $python -c "import h2o, numpy, pyspark; print('Python environment OK:', pyspark.__version__, h2o.__version__, numpy.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Python environment validation failed" }
Write-Host "Windows environment is ready. System Java was not changed."
