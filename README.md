# OCR 스캔파일 텍스트추출 도구 (OCR Scan-to-Text)

스캔된 **PDF 및 이미지(PNG/JPG/TIFF)** 에서 텍스트를 추출하여 파일(`.txt`, `.md`)로 저장하는 고성능 OCR 도구입니다.

- **디지털 PDF**: 텍스트 레이어가 있는 문서는 무손실 고속 다이렉트 추출
- **스캔 PDF / 이미지**:
  - **RapidOCR (ONNX)**: 독립 실행 가능한 고속 로컬 신경망 엔진
  - **Tesseract OCR**: 로컬 무설치/설치형 기본 OCR
  - **Vision LLM**: 복잡한 레이아웃 분석 및 서식 복원 (Gemini / Claude / 로컬 Ollama 연동 지원)
- **한글 맞춤법 및 띄어쓰기 정제**: 도서 출판물, 세로쓰기, 머리말/바닥글 노이즈 필터링 내장
- **웹 대시보드(Gradio)**: 간편한 브라우저 인터페이스 (`localhost:7860`)

---

## 폴더 구조 (엔드유저 2계층 구조)

```
01_OCR_스캔파일텍스트추출프로그램/
├── START_OCR.bat           # 대시보드 서버 및 브라우저 실행 (더블클릭)
├── OCR_대시보드_열기.url    # 이미 실행 중인 대시보드 브라우저 바로 열기
├── (../입력/)              # 처리할 PDF / 이미지 투입 폴더 = OCR프로젝트/입력 (User Layer, 프로그램 폴더 밖)
├── 02_결과/                # 추출된 텍스트 결과 저장 폴더 (User Layer)
│
├── engine/                 # 핵심 엔진 및 백엔드 로직 (Internal Layer)
│   ├── core/               # OCR 파이프라인, 전/후처리, 정제 모듈
│   ├── storage/            # 런타임 캐시 및 작업 히스토리
│   ├── tests/              # 320+ 회귀 테스트 스위트
│   ├── ocr_gradio.py       # Gradio 웹 대시보드 애플리케이션
│   ├── ocr_tool.py         # 단일 파일/배치 처리 CLI 도구
│   ├── install.bat         # 가상환경 구성 및 패키지 설치
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
   - `OCR프로젝트\입력` 폴더의 파일 선택(④ 일괄 처리) 또는 웹 UI로 직접 업로드
   - 추출 모드(기본 RapidOCR / Vision) 선택 후 실행
   - `02_결과` 폴더에서 추출된 텍스트 확인

---

## 라이선스 및 엔진 정보
- **RapidOCR**: Apache-2.0 License (ONNX Runtime 기반 초경량·고속 한글 모델)
- **PyMuPDF / pdfplumber**: PDF 텍스트 레이어 및 래스터라이징 처리
