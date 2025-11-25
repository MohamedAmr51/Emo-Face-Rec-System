@echo off
echo Starting Face Processing Pipeline...
echo.

echo ===================================
echo Running Main Script...
echo ===================================
python run_main.py
echo Main script completed.
echo.

call conda activate FIQ_gpu_env

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
pause