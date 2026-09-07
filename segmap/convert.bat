@echo off
setlocal enabledelayedexpansion
rem Turn the checked-in custom source clip into a segmentation control video
rem next to it, ready for specs/seg_custom.json.
rem
rem   convert.bat                          full source, automatic prompting
rem   set FRAMES=121 ^& convert.bat        one spec-length clip instead
rem   set POINTS=960,700 ^& convert.bat    prompt one object explicitly
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
if not defined MAXOBJ set "MAXOBJ=8"

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
echo ==========================================
echo.

if defined POINTS (
  "%PY%" "%HERE%make_seg_control.py" "%SOURCE%" -o "%OUT%" --frames %FRAMES% --max-objects %MAXOBJ% --points "%POINTS%"
) else (
  "%PY%" "%HERE%make_seg_control.py" "%SOURCE%" -o "%OUT%" --frames %FRAMES% --max-objects %MAXOBJ%
)
if errorlevel 1 exit /b 1
rem Checked explicitly: a crash inside the propagation loop has been seen to
rem leave the exit code at zero while writing no video at all.
if not exist "%OUT%" (
  echo ERROR: the run reported success but %OUT% was not written.
  exit /b 1
)

echo.
echo ==========================================
echo Next: set num_frames in specs\seg_custom.json to the frame count above,
echo       then run the transfer notebook with that spec.
echo ==========================================
endlocal
