@echo off
setlocal enabledelayedexpansion
rem Build the venv for make_seg_control.py: SAM 2 plus a CUDA torch.
rem
rem Python 3.11 because 3.12 deadlocks in thread pools on this machine, and
rem torch comes from the cu130 index first so nothing later resolves a CPU
rem wheel over the top of it.

set "HERE=%~dp0"
set "VENV=%HERE%.venv"

where uv >nul 2>&1
if errorlevel 1 (
  echo ERROR: uv is not on PATH. Install it from https://astral.sh/uv
  exit /b 1
)
for /f "delims=" %%i in ('where uv') do set "UV_EXE=%%i"

echo === Creating the venv on Python 3.11 ===
"!UV_EXE!" venv "%VENV%" --python 3.11 --seed --managed-python --allow-existing
if errorlevel 1 exit /b 1

echo === Installing torch from the cu130 index ===
"!UV_EXE!" pip install --python "%VENV%\Scripts\python.exe" --index-url https://download.pytorch.org/whl/cu130 torch torchvision
if errorlevel 1 exit /b 1

echo === Installing SAM 2 and the media dependencies ===
rem SAM2_BUILD_CUDA=0 skips the optional mask-postprocessing extension, which
rem needs MSVC and nvcc to build and is not required for video propagation.
set SAM2_BUILD_CUDA=0
rem Bare git URL, not "sam2 @ git+...": upstream declares the distribution as
rem SAM-2 while the import package is sam2, and a named direct reference is
rem rejected for the mismatch.
rem huggingface_hub is listed explicitly: sam2 imports it inside from_pretrained
rem but does not declare it as a dependency, so the failure only appears at the
rem first checkpoint download rather than at install time.
"!UV_EXE!" pip install --python "%VENV%\Scripts\python.exe" "git+https://github.com/facebookresearch/sam2.git" huggingface_hub hydra-core imageio imageio-ffmpeg iopath numpy opencv-python
if errorlevel 1 exit /b 1

echo === Verifying ===
"%VENV%\Scripts\python.exe" -c "import torch, sam2; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'built_for', torch.version.cuda); print('sam2 ok')"
if errorlevel 1 exit /b 1

echo.
echo === Done ===
echo Run:  "%VENV%\Scripts\python.exe" "%HERE%make_seg_control.py" ^<video^> -o control_seg.mp4
endlocal
