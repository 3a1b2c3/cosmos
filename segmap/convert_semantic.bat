@echo off
setlocal enabledelayedexpansion
rem Turn the checked-in custom source clip into a Cityscapes-class segmentation
rem control video next to it, ready for specs\seg_custom.json.
rem
rem The counterpart of convert.bat, which uses SAM 2. This one labels every
rem pixel by class, so a colour means the same thing in every frame of every
rem video rather than "the third-largest region in this shot".
rem
rem   convert_semantic.bat                      full source
rem   set FRAMES=300 ^& convert_semantic.bat    a shorter test
rem   set BATCH=8 ^& convert_semantic.bat       more frames per forward pass
rem
rem Run setup.bat first.

set "HERE=%~dp0"
set "TRANSFER=%HERE%..\cookbooks\cosmos3\generator\transfer"
set "PY=%HERE%.venv\Scripts\python.exe"

if not defined SOURCE set "SOURCE=%TRANSFER%\assets\custom\ePi8tDdKuWw.mp4"
if not defined OUT set "OUT=%TRANSFER%\assets\custom\control_seg.mp4"
rem 0 means the whole source. The spec's own num_frames has to match whatever
rem this produces, so changing one without the other desynchronises them.
if not defined FRAMES set "FRAMES=0"
if not defined BATCH set "BATCH=4"

if not exist "%PY%" (
  echo ERROR: %PY% is missing. Run setup.bat first.
  exit /b 1
)
if not exist "%SOURCE%" (
  echo ERROR: %SOURCE% is missing.
  exit /b 1
)

echo ==========================================
echo source: %SOURCE%
echo out:    %OUT%
echo frames: %FRAMES%  ^(0 = whole source^)
echo mode:   Cityscapes semantic classes
echo ==========================================
echo.

"%PY%" "%HERE%make_seg_control_semantic.py" "%SOURCE%" -o "%OUT%" --frames %FRAMES% --batch %BATCH%
if errorlevel 1 exit /b 1
rem Checked explicitly: a crash inside the loop has been seen to leave the exit
rem code at zero while writing no video at all.
if not exist "%OUT%" (
  echo ERROR: the run reported success but %OUT% was not written.
  exit /b 1
)

echo.
echo ==========================================
echo Next: set num_frames in specs\seg_custom.json to the frame count above,
echo       then run the transfer with that spec.
echo ==========================================
endlocal
