@echo off
setlocal enabledelayedexpansion
rem Build a control video from the taxi gameplay capture.
rem
rem Defaults to depth: chase-cam gameplay puts large uniform road and sky areas
rem in frame, which SAM 2 fragments into many small regions of which only the
rem top few survive -- measured around 21% coverage on this footage against 73%
rem on a lateral tracking shot. Depth labels every pixel by construction, so
rem coverage stops being a variable.
rem
rem   convert_taxi.bat                        depth, 200 frames
rem   set CONTROL=seg ^& convert_taxi.bat     SAM 2 instead
rem   set FRAMES=0 ^& convert_taxi.bat        the whole clip
rem   set START=10 ^& convert_taxi.bat        skip 10 seconds in
rem
rem Run setup.bat first.

set "HERE=%~dp0"
set "PY=%HERE%.venv\Scripts\python.exe"
set "DATA=C:\workspace\data\taxi"

if not defined CONTROL set "CONTROL=depth"
if not defined SOURCE set "SOURCE=%DATA%\taxi_87_132.mp4"
if not defined OUT set "OUT=%DATA%\control_%CONTROL%_taxi.mp4"
rem 200 is the top of the range the framework calls acceptable, so a control of
rem this length can be generated in a single pass with no chunk chaining.
if not defined FRAMES set "FRAMES=200"
if not defined START set "START=0"
if not defined MAXOBJ set "MAXOBJ=14"

if not exist "%PY%" (
  echo ERROR: %PY% is missing. Run setup.bat first.
  exit /b 1
)
if not exist "%SOURCE%" (
  echo ERROR: %SOURCE% is missing.
  exit /b 1
)

rem Two of these on one GPU has repeatedly killed a run outright, with no
rem traceback and a zero exit code, so it is checked rather than risked.
for /f %%c in ('powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name like '%%python%%'\" ^| Where-Object { $_.CommandLine -like '*make_seg_control*' -or $_.CommandLine -like '*make_geo_control*' }).Count"') do set "BUSY=%%c"
if not "!BUSY!"=="0" (
  echo ERROR: another control-building run is already on the GPU. Wait for it to finish.
  exit /b 1
)

echo ==========================================
echo source:  %SOURCE%
echo out:     %OUT%
echo control: %CONTROL%
echo frames:  %FRAMES%  ^(0 = whole clip^)  start: %START%s
echo ==========================================
echo.

if /i "%CONTROL%"=="depth" (
  "%PY%" "%HERE%make_geo_control.py" "%SOURCE%" -o "%OUT%" --control depth --frames %FRAMES% --start %START% --batch 4
) else if /i "%CONTROL%"=="edge" (
  "%PY%" "%HERE%make_geo_control.py" "%SOURCE%" -o "%OUT%" --control edge --frames %FRAMES% --start %START%
) else if /i "%CONTROL%"=="seg" (
  rem Gameplay sits between a static portrait and a fast trailer pan, so the cut
  rem threshold is mid-range; re-prompting keeps tracking from decaying.
  "%PY%" "%HERE%make_seg_control.py" "%SOURCE%" -o "%OUT%" --frames %FRAMES% --start %START% --max-objects %MAXOBJ% --detect "car,truck,bus,person" --cut-threshold 40 --reprompt 30
) else (
  echo ERROR: CONTROL must be depth, edge or seg. Got "%CONTROL%".
  exit /b 1
)
if errorlevel 1 exit /b 1
rem Checked explicitly: a crash inside the loop has been seen to leave the exit
rem code at zero while writing no video at all.
if not exist "%OUT%" (
  echo ERROR: the run reported success but %OUT% was not written.
  exit /b 1
)

echo.
echo ==========================================
echo Next: point a spec at %OUT% under the "%CONTROL%" key, and set its
echo       num_frames to the count above.
echo ==========================================
endlocal
