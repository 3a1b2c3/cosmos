@echo off
setlocal enabledelayedexpansion
rem Turn the checked-in custom source clip into an edge or depth control video.
rem
rem Geometric controls describe shape, not meaning, so there are no classes to
rem misidentify. That is why they suit stylised or synthetic footage, where
rem semantic segmentation fails.
rem
rem   convert_geo.bat                          edge, full source
rem   set CONTROL=depth ^& convert_geo.bat     depth instead
rem   set FRAMES=300 ^& convert_geo.bat        a shorter test
rem
rem Run setup.bat first.

set "HERE=%~dp0"
set "TRANSFER=%HERE%..\cookbooks\cosmos3\generator\transfer"
set "PY=%HERE%.venv\Scripts\python.exe"

if not defined CONTROL set "CONTROL=edge"
if not defined SOURCE set "SOURCE=%TRANSFER%\assets\custom\ePi8tDdKuWw.mp4"
rem Named per control so edge and depth do not overwrite one another.
if not defined OUT set "OUT=%TRANSFER%\assets\custom\control_%CONTROL%.mp4"
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
echo source:  %SOURCE%
echo out:     %OUT%
echo control: %CONTROL%
echo frames:  %FRAMES%  ^(0 = whole source^)
echo ==========================================
echo.

"%PY%" "%HERE%make_geo_control.py" "%SOURCE%" -o "%OUT%" --control %CONTROL% --frames %FRAMES% --batch %BATCH%
if errorlevel 1 exit /b 1
rem Checked explicitly: a crash inside the loop has been seen to leave the exit
rem code at zero while writing no video at all.
if not exist "%OUT%" (
  echo ERROR: the run reported success but %OUT% was not written.
  exit /b 1
)

echo.
echo ==========================================
echo Next: point a spec at %OUT% under the "%CONTROL%" key, set its num_frames
echo       to the count above, then run the transfer with that spec.
echo ==========================================
endlocal
