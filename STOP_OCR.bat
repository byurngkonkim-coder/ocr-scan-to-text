@echo off
title Stop OCR Server

rem Stop only the OCR server: processes running ocr_gradio.py. Any other program on 7860 is left alone.
rem netstat / ocr_server.pid are not used: a running server was not shown as LISTENING and the PID file
rem was overwritten by a crashed start, so STOP said "not running" (2026-11-06). See engine\ocr_port.ps1.
rem Relative path only: no non-ASCII path is passed to PowerShell.
cd /d "%~dp0engine"
set "_found="
set "_other="
powershell -NoProfile -ExecutionPolicy Bypass -File "ocr_port.ps1" stop
if not errorlevel 1 set "_found=1"
if not defined _found (
    powershell -NoProfile -ExecutionPolicy Bypass -File "ocr_port.ps1" check
    if errorlevel 3 set "_other=1"
)

echo.
if defined _found (
    echo  [DONE] OCR server stopped.
) else if defined _other (
    echo  Port 7860 is used by another program - not stopped.
) else (
    echo  OCR server is not running.
)
echo.
timeout /t 3 > nul
exit
