@echo off
REM ============================================================================
REM Full System Launcher
REM
REM This batch file orchestrates the startup of the entire system:
REM 1. Launches the Main Application (`run_main.py`) in its own window/environment.
REM 2. Activates the GPU environment (`FIQ_gpu_env`) for Quality Assessment.
REM 3. Launches the Quality Assessment module (`run_quality.py`).
REM
REM Usage: Double-click this file to start the system.
REM ============================================================================

echo Starting Face Processing Pipeline...
echo.

echo ===================================
echo Running Main Script...
echo ===================================
python run_main.py
echo Main script completed.
echo.

echo ===================================
echo Running Quality Assessment...
echo ===================================
cd "Quality Assessment"
python run_quality.py
cd ..
echo Quality assessment completed.
echo.

echo ===================================
echo All scripts completed successfully!
echo ===================================