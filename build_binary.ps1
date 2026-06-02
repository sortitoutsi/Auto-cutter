# Build a standalone image-cropper.exe using PyInstaller (Windows).
# PyInstaller cannot cross-compile — run this on Windows for a Windows build.
#
# Usage (PowerShell):
#   .\build_binary.ps1                    # uses .\.venv\
#   .\build_binary.ps1 -VenvDir C:\venv   # custom venv

param([string]$VenvDir = ".venv")

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

if (-not (Test-Path $VenvDir)) {
    Write-Host "ERROR: venv $VenvDir not found. Run .\install.ps1 first." -ForegroundColor Red
    exit 1
}

$Py = Join-Path $VenvDir "Scripts\python.exe"
& $Py -m pip install --quiet "pyinstaller>=6.0"

Write-Host "Building standalone binary (this is slow and the result is ~3 GB)..."
& $Py -m PyInstaller --clean --noconfirm image_cropper.spec

if (Test-Path "dist\image-cropper") {
    $size = (Get-ChildItem -Recurse "dist\image-cropper" | Measure-Object -Property Length -Sum).Sum
    $sizeGB = [Math]::Round($size / 1GB, 2)
    Write-Host ""
    Write-Host "Done. Bundle at: dist\image-cropper\ ($sizeGB GB)"
}
