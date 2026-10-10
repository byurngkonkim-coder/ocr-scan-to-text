@echo off
title Stop OCR Server

set "_found="
for /f "tokens=5" %%p in ('netstat -ano 2^>nul ^| findstr ":7860 " ^| findstr "LISTENING"') do (
    taskkill /f /pid %%p > nul 2>&1
    set "_found=1"
)

echo.
if defined _found (
    echo  [DONE] OCR server stopped.
) else (
    echo  OCR server is not running.
)
echo.
timeout /t 3 > nul
exit
