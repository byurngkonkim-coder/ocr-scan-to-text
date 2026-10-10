@echo off
title OCR Text Extraction Tool
cd /d "%~dp0"

set "PATH=%PATH%;C:\poppler\Library\bin"
set "PYDIR=%~dp0venv\Scripts"
set "PATH=%PYDIR%;%PATH%"

netstat -an 2>nul | findstr ":7860 " | findstr "LISTENING" > nul
if %errorlevel% == 0 (
    echo.
    echo  Already running. Opening http://localhost:7860
    start http://localhost:7860
    timeout /t 3 > nul
    exit
)

echo.
echo  ==================================================
echo    OCR Scan-to-Text Tool (RapidOCR v5)
echo    PDF / PNG / JPG / TIFF  to  Text
echo  ==================================================
echo.
echo  Starting server... (first run may take about 10s)
echo.

if not exist "%PYDIR%\python.exe" (
    echo  [ERROR] Python venv not found.
    echo  Please run install.bat first.
    pause
    exit /b 1
)

echo  [OK] Local Engine: RapidOCR (PP-OCRv5 Korean ONNX) Ready.

rem Server runs with no console window. Output goes to ocr_server.log / ocr_server_err.log.
rem Relative paths only: the current folder is inherited, so no non-ASCII path is passed to PowerShell.
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath 'venv\Scripts\python.exe' -ArgumentList '-u','ocr_gradio.py' -WorkingDirectory (Get-Location).Path -WindowStyle Hidden -RedirectStandardOutput 'ocr_server.log' -RedirectStandardError 'ocr_server_err.log'"

timeout /t 8 /nobreak > nul
set /a _tries=0
:wait_loop
netstat -an 2>nul | findstr ":7860 " | findstr "LISTENING" > nul
if %errorlevel% == 0 goto open_browser
set /a _tries+=1
if %_tries% geq 30 goto start_failed
timeout /t 2 /nobreak > nul
goto wait_loop

:start_failed
echo.
echo  [ERROR] Server did not start within ~60s.
echo  Open these logs to see why:
echo     "%~dp0ocr_server.log"
echo     "%~dp0ocr_server_err.log"
echo   - If packages are missing, run install.bat
echo   - Port 7860 may be in use by another program
echo.
pause
exit /b 1

:open_browser
rem Browser tab is opened by ocr_gradio.py (launch inbrowser=True).
rem Do NOT open it here too - that produced two tabs.
echo.
echo  [DONE] Open http://localhost:7860 in your browser.
echo  The server keeps running in the background.
echo  To stop: double-click STOP_OCR.bat
echo.
timeout /t 5 > nul
exit
