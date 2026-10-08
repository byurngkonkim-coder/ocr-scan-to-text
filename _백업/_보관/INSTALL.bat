@echo off
rem Runs install.bat inside the 04_* engine folder (creates venv, installs packages).
for /d %%D in ("%~dp004_*") do (
    cd /d "%%~fD"
    call install.bat
    exit /b
)
echo [ERROR] Engine folder (04_...) not found next to this file.
pause
