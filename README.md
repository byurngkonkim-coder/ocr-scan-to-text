# OCR 스캔파일 텍스트추출 도구 (OCR Scan-to-Text)

스캔된 **PDF 및 이미지(PNG/JPG/TIFF)** 에서 텍스트를 추출하여 파일(`.txt`, `.md`)로 저장하는 고성능 OCR 도구입니다.

- **디지털 PDF**: 텍스트 레이어가 있는 문서는 무손실 고속 다이렉트 추출
- **스캔 PDF / 이미지**:
  - **RapidOCR (ONNX)**: 독립 실행 가능한 고속 로컬 신경망 엔진
  - **Vision LLM**: 복잡한 레이아웃 분석 및 서식 복원 (Gemini / Claude / 로컬 Ollama 연동 지원)
- **한글 맞춤법 및 띄어쓰기 정제**: 도서 출판물, 세로쓰기, 머리말/바닥글 노이즈 필터링 내장
- **웹 대시보드(Gradio)**: 간편한 브라우저 인터페이스 (`localhost:7860`)

---

## 폴더 구조 (엔드유저 2계층 구조)

```
01_OCR_스캔파일텍스트추출프로그램/
├── START_OCR.bat           # 대시보드 서버 및 브라우저 실행 (더블클릭)
├── OCR_대시보드_열기.url    # 이미 실행 중인 대시보드 브라우저 바로 열기
├── 01_입력/                # 처리할 PDF / 이미지 파일 투입 폴더 — ② 탭에서 다른 폴더 지정 가능 (User Layer)
├── 02_결과/                # 추출된 텍스트 결과 저장 폴더 (User Layer)
│
├── engine/                 # 핵심 엔진 및 백엔드 로직 (Internal Layer)
│   ├── core/               # OCR 파이프라인, 전/후처리, 정제 모듈
│   ├── storage/            # 런타임 캐시 및 작업 히스토리
│   ├── tests/              # 320+ 회귀 테스트 스위트
│   ├── ocr_gradio.py       # Gradio 웹 대시보드 애플리케이션
│   ├── ocr_tool.py         # 단일 파일/배치 처리 CLI 도구
│   ├── install.bat         # 가상환경 구성 및 패키지 설치
│   ├── make_portable.bat   # USB 복사용 준비 — python\ · poppler\ 를 폴더 안에 만듦
│   └── requirements.txt    # 파이썬 의존성 목록
│
├── docs/                   # 상세 기술 문서 및 사용자 가이드 (Internal Layer)
│   ├── OCR도구_사용자매뉴얼.md
│   ├── OCR도구_설계가이드.md
│   ├── OCR도구_정제규칙서.md
│   ├── document_restoration_rules.md
│   └── 인수인계서.md
│
└── _백업/                  # 과거 버전 및 이동 로그 보관
```

---

## 빠른 시작

1. **사전 준비**: Python 3.10 이상 설치 (`Add Python to PATH` 체크)
2. **패키지 설치**: `engine\install.bat` 실행
3. **프로그램 시작**: 루트의 `START_OCR.bat` 더블클릭
4. 브라우저에서 `http://localhost:7860` 자동 실행:
   - `01_입력`(또는 ② 탭에서 지정한) 폴더의 파일 선택(② 일괄 처리) 또는 웹 UI로 직접 업로드
   - 추출 모드(기본 RapidOCR / Vision) 선택 후 실행
   - `02_결과` 폴더에서 추출된 텍스트 확인

---

## USB 로 다른 PC 에 옮겨 쓰기

설치가 끝난 PC 에서 한 번만 준비하면, 받는 PC 에는 Python·poppler 를 설치하지 않아도 됩니다.

1. 설치된 PC 에서 `engine\make_portable.bat` 실행 → `engine\python\`(Python + 패키지)·`engine\poppler\` 가 생김 (약 1GB)
2. 프로그램 폴더 전체를 USB 로 복사
3. 받는 PC 에서 `START_OCR.bat` 더블클릭

- `engine\venv\` 는 복사하지 않아도 됩니다(원래 PC 전용). `python\` 이 있으면 그쪽을 먼저 씁니다.
- 폴더는 **드라이브 바로 아래처럼 짧은 경로**에 두세요(예: `E:\OCR\`). 경로가 너무 길면 Windows 260자 제한에 걸려 OCR 이 시작되지 않습니다.
- 처리 기록(`engine\storage\history.json`)은 프로그램 폴더 기준 경로로 저장돼, 옮긴 뒤에도 결과 받기가 됩니다.
- 패키지를 새로 설치·업데이트했으면 `make_portable.bat` 를 다시 실행하세요.

---

## 라이선스 및 엔진 정보
- **RapidOCR**: Apache-2.0 License (ONNX Runtime 기반 초경량·고속 한글 모델)
- **PyMuPDF / pdfplumber**: PDF 텍스트 레이어 및 래스터라이징 처리
