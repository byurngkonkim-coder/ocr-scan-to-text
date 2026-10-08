@echo off
cd /d "%~dp0"

if exist "%~dp0engine\START_OCR.bat" (
    call "%~dp0engine\START_OCR.bat"
    exit /b
)

for /d %%D in ("%~dp004_*") do (
    call "%%~fD\START_OCR.bat"
    exit /b
)

echo [ERROR] Engine folder (engine or 04_...) not found next to this file.
pause
