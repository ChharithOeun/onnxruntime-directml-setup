@echo off
REM onnxruntime-directml-setup — Quick Start

echo ============================================
echo  ONNX Runtime DirectML — AMD GPU Windows
echo ============================================
echo.

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

python -c "import onnxruntime" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 ( pause & exit /b 1 )
)

echo Verifying GPU...
python scripts\verify_gpu.py
if errorlevel 1 (
    echo.
    echo WARNING: DirectML EP not found. Ensure onnxruntime-directml is installed:
    echo   pip uninstall onnxruntime -y
    echo   pip install onnxruntime-directml
    pause
    exit /b 1
)

echo.
echo Running benchmark (DML vs CPU)...
python scripts\benchmark.py --device both

echo.
echo Done! Try image inference:
echo   python scripts\infer_image.py --hf-model microsoft/resnet-50 --image your_photo.jpg
echo.
pause
