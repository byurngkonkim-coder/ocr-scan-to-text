@echo off
title Stop OCR Server

rem Stop only the OCR server: the process on port 7860 whose PID is in engine\ocr_server.pid
rem (written by the server itself). Any other program on 7860 is left alone.
set "_mine="
if exist "%~dp0engine\ocr_server.pid" set /p _mine=<"%~dp0engine\ocr_server.pid"
set "_found="
set "_other="
for /f "tokens=5" %%p in ('netstat -ano 2^>nul ^| findstr ":7860 " ^| findstr "LISTENING"') do (
    if "%%p"=="%_mine%" (
        taskkill /f /pid %%p > nul 2>&1
        set "_found=1"
    ) else (
        set "_other=%%p"
    )
)

echo.
if defined _found (
    echo  [DONE] OCR server stopped.
) else if defined _other (
    echo  Port 7860 is used by another program ^(PID %_other%^) - not stopped.
    echo  If it is this OCR server started another way, close it in Task Manager.
) else (
    echo  OCR server is not running.
)
echo.
timeout /t 3 > nul
exit
