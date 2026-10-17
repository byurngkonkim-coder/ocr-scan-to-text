@echo off
chcp 65001 > nul
title Make OCR tool portable (USB)
cd /d "%~dp0"
set "PYTHONUTF8=1"

echo ==================================================
echo   USB 복사용 준비 - 이 PC 에서 한 번만 실행
echo   python\  : Python + 패키지 전부 (설치 없이 실행)
echo   poppler\ : PDF 를 그림으로 바꾸는 도구
echo ==================================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo [오류] venv 가 없습니다. install.bat 를 먼저 실행하세요.
    pause
    exit /b 1
)
venv\Scripts\python.exe --version > nul 2>&1
if errorlevel 1 (
    echo [오류] 이 PC 의 venv 가 망가졌습니다. install.bat 를 먼저 실행하세요.
    pause
    exit /b 1
)

set "BASE="
for /f "usebackq delims=" %%p in (`venv\Scripts\python.exe -c "import sys; print(sys.base_prefix)"`) do set "BASE=%%p"
if not defined BASE (
    echo [오류] 원본 Python 위치를 찾지 못했습니다.
    pause
    exit /b 1
)
echo [1/4] Python 복사: %BASE%
robocopy "%BASE%" "python" /MIR /XD "%BASE%\Lib\site-packages" "%BASE%\Scripts" "%BASE%\Doc" "%BASE%\Lib\test" "%BASE%\Lib\idlelib" /NFL /NDL /NJH /NJS /NP > nul
if errorlevel 8 goto copy_failed

echo [2/4] 패키지 복사: venv\Lib\site-packages
robocopy "venv\Lib\site-packages" "python\Lib\site-packages" /MIR /XD __pycache__ /NFL /NDL /NJH /NJS /NP > nul
if errorlevel 8 goto copy_failed

rem onnxruntime 등이 쓰는 Visual C++ 런타임 - 없는 PC 를 위해 python.exe 옆에 둔다
for %%d in (msvcp140.dll msvcp140_1.dll msvcp140_2.dll vcruntime140.dll vcruntime140_1.dll) do (
    if not exist "python\%%d" if exist "%SystemRoot%\System32\%%d" copy /y "%SystemRoot%\System32\%%d" "python\" > nul
)

echo [3/4] poppler 복사
set "POP="
if exist "C:\poppler\Library\bin\pdftoppm.exe" set "POP=C:\poppler\Library\bin"
if not defined POP for /f "delims=" %%w in ('where pdftoppm 2^>nul') do if not defined POP set "POP=%%~dpw"
if defined POP (
    robocopy "%POP%" "poppler" /MIR /NFL /NDL /NJH /NJS /NP > nul
    if errorlevel 8 goto copy_failed
) else (
    echo [경고] 이 PC 에 poppler 가 없어 복사하지 못했습니다. 스캔 PDF 처리가 안 될 수 있습니다.
)

echo [4/4] 복사본으로 실행 점검
python\python.exe -c "import rapidocr, onnxruntime, gradio, pymupdf, cv2, pdf2image; print('  [OK] 필요한 패키지 모두 불러옴')"
if errorlevel 1 (
    echo [오류] 복사한 Python 으로 패키지를 불러오지 못했습니다.
    pause
    exit /b 1
)

echo.
echo ==================================================
echo   완료. 이제 프로그램 폴더 전체를 USB 로 복사하면
echo   다른 PC 에서 START_OCR.bat 만 눌러 실행됩니다.
echo ==================================================
pause
exit /b 0

:copy_failed
echo [오류] 파일 복사에 실패했습니다. 서버(START_OCR)가 켜져 있으면 STOP_OCR.bat 로 끄고 다시 실행하세요.
pause
exit /b 1
