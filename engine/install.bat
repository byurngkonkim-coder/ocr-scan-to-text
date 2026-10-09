@echo off
chcp 65001 > nul
echo ================================================
echo  OCR 스캔파일 텍스트추출 도구 - Windows 설치
echo ================================================
echo.

:: Python 버전 확인
python --version 2>nul
if errorlevel 1 (
    echo [오류] Python이 설치되어 있지 않습니다.
    echo https://www.python.org 에서 Python 3.10 이상을 설치해 주세요.
    pause
    exit /b 1
)

:: Python 버전 3.10 이상 확인
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo Python 버전: %PYVER%

:: OCR 엔진 안내
echo [안내] 로컬 엔진으로 고성능 RapidOCR(PP-OCRv5 Korean ONNX)이 내장 가상환경에 자동 설치됩니다.

:: Poppler 설치 확인
where pdftoppm >nul 2>&1
if errorlevel 1 (
    echo.
    echo [경고] Poppler가 설치되어 있지 않습니다.
    echo PDF 파일 처리를 위해 아래 링크에서 설치 후 PATH에 추가해 주세요:
    echo   https://github.com/oschwartz10612/poppler-windows/releases
    echo.
    echo PDF 없이 이미지 파일(PNG, JPG)만 처리하려면 설치 불필요.
) else (
    echo [OK] Poppler 감지됨
)

echo.
echo ------------------------------------------------
echo  Python 가상환경 생성 및 의존성 설치 중...
echo ------------------------------------------------

:: A venv copied from another PC points to a Python that does not exist here -> move it aside and rebuild
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe --version >nul 2>&1
    if errorlevel 1 (
        echo [WARN] Existing venv is broken on this PC. Moving it to venv_broken and rebuilding.
        if exist "venv_broken" rmdir /s /q "venv_broken"
        move "venv" "venv_broken" >nul
    )
)

:: 가상환경 생성
if not exist "venv" (
    python -m venv venv
    echo [OK] 가상환경 생성: venv\
) else (
    echo [OK] 기존 가상환경 사용: venv\
)

:: 가상환경 활성화 후 패키지 설치
venv\Scripts\python.exe -m pip install --upgrade pip --quiet
venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo ================================================
echo  설치 완료!
echo ================================================
echo.
echo 사용 방법:
echo   venv\Scripts\activate
echo   python ocr_tool.py --input 파일경로.pdf --output 출력폴더\
echo.
echo 환경 검증:
echo   python verify_env.py
echo.
pause
