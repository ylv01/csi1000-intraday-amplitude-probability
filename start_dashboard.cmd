@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "CSI_PYTHON_EXE="
set "CSI_PYTHON_ARGS="

if defined CSI1000_PYTHON (
  if exist "%CSI1000_PYTHON%" set "CSI_PYTHON_EXE=%CSI1000_PYTHON%"
)

if not defined CSI_PYTHON_EXE (
  if exist "%~dp0runtime_env\python.exe" set "CSI_PYTHON_EXE=%~dp0runtime_env\python.exe"
  if exist "%~dp0runtime_env\Scripts\python.exe" set "CSI_PYTHON_EXE=%~dp0runtime_env\Scripts\python.exe"
)

if not defined CSI_PYTHON_EXE (
  if exist "%~dp0.venv\Scripts\python.exe" set "CSI_PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
)

if not defined CSI_PYTHON_EXE (
  if defined CONDA_PREFIX (
    if exist "%CONDA_PREFIX%\python.exe" set "CSI_PYTHON_EXE=%CONDA_PREFIX%\python.exe"
  )
)

if not defined CSI_PYTHON_EXE (
  where conda >nul 2>nul
  if not errorlevel 1 (
    for /f "delims=" %%I in ('call conda info --base 2^>nul') do set "CSI_CONDA_BASE=%%I"
    if exist "!CSI_CONDA_BASE!\envs\py39\python.exe" set "CSI_PYTHON_EXE=!CSI_CONDA_BASE!\envs\py39\python.exe"
    if not defined CSI_PYTHON_EXE (
      if exist "!CSI_CONDA_BASE!\envs\csi1000-v4\python.exe" set "CSI_PYTHON_EXE=!CSI_CONDA_BASE!\envs\csi1000-v4\python.exe"
    )
  )
)

if not defined CSI_PYTHON_EXE (
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3.9 -c "import sys" >nul 2>nul
    if not errorlevel 1 (
      set "CSI_PYTHON_EXE=py"
      set "CSI_PYTHON_ARGS=-3.9"
    )
  )
)

if not defined CSI_PYTHON_EXE (
  where python >nul 2>nul
  if not errorlevel 1 set "CSI_PYTHON_EXE=python"
)

if not defined CSI_PYTHON_EXE goto no_python

echo Starting CSI 1000 amplitude dashboard...
echo Python: "%CSI_PYTHON_EXE%" %CSI_PYTHON_ARGS%
echo Project: "%CD%"
echo.

"%CSI_PYTHON_EXE%" %CSI_PYTHON_ARGS% -c "import certifi, joblib, lightgbm, numpy, pandas, plotly, sklearn, streamlit" >nul 2>nul
if errorlevel 1 goto missing_dependencies

"%CSI_PYTHON_EXE%" %CSI_PYTHON_ARGS% "%~dp0scripts\run_dashboard.py" %*
set "CSI_EXIT_CODE=!ERRORLEVEL!"

if not "!CSI_EXIT_CODE!"=="0" (
  echo.
  echo Dashboard stopped with exit code !CSI_EXIT_CODE!.
  pause
)

endlocal & exit /b %CSI_EXIT_CODE%

:missing_dependencies
echo Required Python packages are missing from this interpreter.
echo Run setup_environment.cmd once, then start the dashboard again.
pause
endlocal & exit /b 2

:no_python
echo Python was not found.
echo Run setup_environment.cmd, or install Miniconda/Python 3.9 first.
pause
endlocal & exit /b 3
