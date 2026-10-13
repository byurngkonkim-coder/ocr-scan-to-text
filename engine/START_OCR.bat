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
rem The server PID is saved so the wait loop can tell "still loading" from "crashed".
if exist "ocr_server.pid" del "ocr_server.pid"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = Start-Process -FilePath 'venv\Scripts\python.exe' -ArgumentList '-u','ocr_gradio.py' -WorkingDirectory (Get-Location).Path -WindowStyle Hidden -RedirectStandardOutput 'ocr_server.log' -RedirectStandardError 'ocr_server_err.log' -PassThru; Set-Content -Path 'ocr_server.pid' -Value $p.Id -Encoding Ascii"

set "_pid="
if exist "ocr_server.pid" set /p _pid=<"ocr_server.pid"
if not defined _pid goto server_died

set /a _tries=0
:wait_loop
netstat -an 2>nul | findstr ":7860 " | findstr "LISTENING" > nul
if %errorlevel% == 0 goto open_browser
rem Process gone before the port opened = crash. Stop waiting and show the error log now.
tasklist /fi "PID eq %_pid%" /nh 2>nul | find "%_pid%" > nul
if errorlevel 1 goto server_died
set /a _tries+=1
if %_tries% geq 35 goto still_loading
timeout /t 2 /nobreak > nul
goto wait_loop

:still_loading
rem Port not open yet but the process is alive (slow first run). It is not a failure:
rem ocr_gradio.py opens the browser by itself once the server is ready.
echo.
echo  [WAIT] Server is still loading (over ~70s). It is running in the background.
echo  The browser will open by itself when it is ready.
echo  If nothing opens in a few minutes, check "%~dp0ocr_server_err.log"
echo.
pause
exit /b 0

:server_died
echo.
echo  [ERROR] Server stopped right after starting. Last lines of the error log:
echo  --------------------------------------------------
if exist "ocr_server_err.log" powershell -NoProfile -Command "Get-Content -Path 'ocr_server_err.log' -Tail 15"
echo  --------------------------------------------------

:start_failed
echo.
echo  [ERROR] Server did not start.
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
