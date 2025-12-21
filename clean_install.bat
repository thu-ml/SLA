@echo off
REM Clean installation script for Windows
echo Cleaning old installation artifacts...

REM Remove egg-info and build directories
if exist "sparse_linear_attention.egg-info" rd /s /q "sparse_linear_attention.egg-info"
if exist "build" rd /s /q "build"
if exist "dist" rd /s /q "dist"

REM Uninstall old package
echo Uninstalling old package...
pip uninstall -y sparse-linear-attention

REM Reinstall with triton-windows
echo Installing with triton-windows support...
pip install -e .[triton-windows]

echo.
echo Installation complete!
echo.
pause
