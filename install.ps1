# Install image-cropper on Windows (PowerShell).
# Creates a venv, installs the package and all dependencies.
#
# Usage (from a PowerShell prompt in this folder):
#   .\install.ps1                      # default: .\.venv\
#   .\install.ps1 -VenvDir C:\imgcrop  # custom venv location
#   .\install.ps1 -UserInstall         # install into user site-packages (no venv)

[CmdletBinding()]
param(
    [string]$VenvDir = ".venv",
    [switch]$UserInstall
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

# --- find a compatible Python (3.11–3.13) ---
function Find-Python {
    $candidates = @("python3.13", "python3.12", "python3.11", "py -3.13", "py -3.12", "py -3.11")
    foreach ($cmd in $candidates) {
        try {
            $exe = $cmd.Split(" ")[0]
            $rest = $cmd.Substring($exe.Length).Trim()
            if ($rest) {
                $ver = & $exe $rest.Split(" ") --version 2>$null
            } else {
                $ver = & $exe --version 2>$null
            }
            if ($LASTEXITCODE -eq 0 -and $ver) {
                return $cmd
            }
        } catch {}
    }
    # Fallback: python on PATH if its version is 3.11-3.13
    try {
        $ver = & python --version 2>$null
        if ($LASTEXITCODE -eq 0 -and $ver -match "Python 3\.(11|12|13)\.") {
            return "python"
        }
    } catch {}
    return $null
}

$PythonCmd = Find-Python
if (-not $PythonCmd) {
    Write-Host "ERROR: Need Python 3.11, 3.12, or 3.13 on PATH (PyTorch has no 3.14 wheels yet)." -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}
Write-Host "Using Python: $PythonCmd"

# Helper to invoke the chosen python (handles "py -3.12")
function Invoke-Py($args) {
    $parts = $PythonCmd.Split(" ")
    $exe = $parts[0]
    $prefix = $parts[1..($parts.Length - 1)]
    & $exe @prefix @args
}

if ($UserInstall) {
    Write-Host "Installing image-cropper into user site-packages..."
    Invoke-Py @("-m", "pip", "install", "--user", "--upgrade", "pip")
    Invoke-Py @("-m", "pip", "install", "--user", $ScriptDir)
    Write-Host ""
    Write-Host "Done. Launch with:"
    Write-Host "    python -m image_cropper"
} else {
    if (-not (Test-Path $VenvDir)) {
        Write-Host "Creating virtual environment at $VenvDir ..."
        Invoke-Py @("-m", "venv", $VenvDir)
    } else {
        Write-Host "Reusing virtual environment at $VenvDir"
    }

    $VenvPy  = Join-Path $VenvDir "Scripts\python.exe"
    $VenvPip = Join-Path $VenvDir "Scripts\pip.exe"
    $VenvGui = Join-Path $VenvDir "Scripts\image-cropper.exe"

    & $VenvPy -m pip install --upgrade pip --quiet
    Write-Host "Installing image-cropper and dependencies (this can take several minutes — pytorch is large)..."
    & $VenvPy -m pip install $ScriptDir

    Write-Host ""
    Write-Host "Done. Launch the GUI with:"
    Write-Host "    $VenvGui"
    Write-Host "or:"
    Write-Host "    $VenvPy -m image_cropper"
}
