@echo off
rem Launch the Image Cropper GUI on Windows.
rem Prefers .venv, then a `image-cropper` console script on PATH.
setlocal
set "SCRIPT_DIR=%~dp0"

if exist "%SCRIPT_DIR%.venv\Scripts\image-cropper.exe" (
    "%SCRIPT_DIR%.venv\Scripts\image-cropper.exe" %*
    goto :eof
)
if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    "%SCRIPT_DIR%.venv\Scripts\python.exe" -m image_cropper %*
    goto :eof
)

where image-cropper >nul 2>&1
if %ERRORLEVEL%==0 (
    image-cropper %*
    goto :eof
)

echo ERROR: image-cropper not installed. Run install.ps1 first.
exit /b 1
