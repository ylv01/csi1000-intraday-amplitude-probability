@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "CSI_RUNTIME_PYTHON="

if exist "%~dp0runtime_env\python.exe" set "CSI_RUNTIME_PYTHON=%~dp0runtime_env\python.exe"
if exist "%~dp0runtime_env\Scripts\python.exe" set "CSI_RUNTIME_PYTHON=%~dp0runtime_env\Scripts\python.exe"
if defined CSI_RUNTIME_PYTHON goto install_packages

where conda >nul 2>nul
if not errorlevel 1 (
  echo Creating a local Python 3.9 Conda environment...
  call conda create --yes --prefix "%~dp0runtime_env" python=3.9.25 pip
  if errorlevel 1 goto setup_failed
  set "CSI_RUNTIME_PYTHON=%~dp0runtime_env\python.exe"
  goto install_packages
)

where py >nul 2>nul
if not errorlevel 1 (
  py -3.9 -c "import sys" >nul 2>nul
  if not errorlevel 1 (
    echo Creating a local Python 3.9 virtual environment...
    py -3.9 -m venv "%~dp0runtime_env"
    if errorlevel 1 goto setup_failed
    set "CSI_RUNTIME_PYTHON=%~dp0runtime_env\Scripts\python.exe"
    goto install_packages
  )
)

where python >nul 2>nul
if not errorlevel 1 (
  python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 9) else 1)" >nul 2>nul
  if not errorlevel 1 (
    echo Creating a local Python 3.9 virtual environment...
    python -m venv "%~dp0runtime_env"
    if errorlevel 1 goto setup_failed
    set "CSI_RUNTIME_PYTHON=%~dp0runtime_env\Scripts\python.exe"
    goto install_packages
  )
)

echo Miniconda or Python 3.9 is required.
echo Install one of them and run this file again.
pause
endlocal & exit /b 3

:install_packages
echo Installing required packages...
"%CSI_RUNTIME_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto setup_failed
"%CSI_RUNTIME_PYTHON%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto setup_failed
"%CSI_RUNTIME_PYTHON%" -c "import certifi, joblib, lightgbm, numpy, pandas, plotly, sklearn, streamlit"
if errorlevel 1 goto setup_failed

echo.
echo Environment is ready. Double-click start_dashboard.cmd.
pause
endlocal & exit /b 0

:setup_failed
echo.
echo Environment setup failed. Check the network connection and messages above.
pause
endlocal & exit /b 1
