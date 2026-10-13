---
title: OCR 스캔파일 텍스트추출 도구 — 설계가이드
version: v3.1 (Gemini 기본 + 문서 정제 단계)
updated: 2026-07-01
target: 04_처리엔진 (구 ocr-tool-Claude — 2026-09-26 폴더 정비로 이름 변경)
---

# 🏗 OCR 스캔파일 텍스트추출 도구 — 설계가이드

> **이 문서의 기본 증거등급: E2** — 구조·설정·동작은 `04_처리엔진/` 소스와 `tests/`(pytest)로 재현·확인한다.

> [!NOTE]
> **핵심 3줄 요약**
> 1. 처리 흐름: **입력감지 → 페이지분해 → 품질진단 → (그림마스킹) → 전처리(조건부) → OCR(provider) → 정규화(L1~L3) → 후처리 → [문서 정제] → 저장**.
> 2. OCR 엔진은 **`core/vision.py` 파사드**가 `config.json`의 `vision_provider` 값에 따라 **Gemini/Claude로 라우팅** — 코드 수정 없이 전환.
> 3. 이 도구는 **충실 추출 + 문서 정제(교정)** 까지 담당 — OCR 원문을 클린 플레인텍스트로 복원(`text_cleaner`, 규칙은 [정제규칙서](OCR도구_정제규칙서.md) `C-xx`). 반면 **명언/저자 추출**은 하위 파이프라인(`02_01/04_처리엔진/normalize_pipeline.py`)의 몫(관심사 분리).

---

## 1. 목적 및 설계 원칙

- **목적**: 스캔 PDF·이미지에서 한국어/영어 텍스트를 정확히 추출.
- **원칙**
  - **원문 충실(faithful extraction)**: OCR 단계는 내용을 요약·변형하지 않는다. 변환·정제는 하위 파이프라인 몫.
  - **2단 엔진 + 자동 폴백**: 로컬 Tesseract(무료) ↔ 클라우드 Vision(고정확)을 품질에 따라 전환.
  - **provider 추상화**: 클라우드 엔진(Gemini/Claude)을 설정으로 교체 가능.
  - **우아한 강등(graceful degradation)**: 키 없음·호출 실패 시 멈추지 않고 검토 필요 표시로 계속.

---

## 2. 디렉터리 / 모듈 구조

```
01_OCR_스캔파일텍스트추출프로그램/     (2026-09-26 폴더 정비 이후)
├── START_OCR.bat         ← 맨 위 실행 래퍼 → 04_처리엔진\START_OCR.bat 호출 (ASCII 전용)
├── INSTALL.bat           ← 맨 위 설치 래퍼 → 04_처리엔진\install.bat 호출
├── OCR_대시보드_열기.url  ← http://localhost:7860 바로가기
├── 입력/                ← 처리할 원본 (batch_runner.INPUT_DIR)
├── 02_결과/              ← 사용자 결과(ocr_gradio.RESULTS_DIR)
├── 03_문서/              ← 사용자매뉴얼·설계가이드·정제규칙서·인수인계서
└── 04_처리엔진/          ← (구 ocr-tool-Claude)
    ├── START_OCR.bat     ← 런처 본체(웹 서버 기동 + 브라우저). ASCII 파일명 사용
    ├── install.bat       ← 사전 점검 + venv 의존성 설치
    ├── verify_env.py     ← 환경 검증
    ├── ocr_gradio.py     ← 웹 UI(Gradio) + 처리 오케스트레이션
    ├── ocr_tool.py       ← CLI 진입점(click)
    ├── config.json       ← 설정(엔진·기본값). API 키는 넣지 않음(환경변수 전용)
    ├── requirements.txt
    ├── core/             ← 처리 엔진 모듈
    └── storage/          ← 내부 보관소: 결과 사본/업로드/이력(런타임 생성)
```

### core/ 모듈 역할
| 모듈 | 역할 |
|---|---|
| `detector.py` | 입력(파일/폴더) 수집, 파일 형식 감지 |
| `pdf_splitter.py` | PDF→페이지 이미지 변환(Poppler, 기본 300 DPI), 이미지 로드 |
| `quality_diagnoser.py` | 흐림(Laplacian)·노출·기울기·세로쓰기·해상도 진단 |
| `blank_detector.py` | 빈 페이지 감지 → OCR 건너뜀(자원 절약) |
| `figure_detector.py` | 그림·표 영역 감지·마스킹·`[그림/표]` 플레이스홀더. 본문 보호 2중: ① 후보 안에서 확신도 60+ 글자 4자 이상이면 본문으로 보고 마스킹 안 함 ② 그림·표 위 단어는 줄 띠로 합쳐 마스킹 후 원본 픽셀로 되살림 (2026-09-27, RETRO 오류 52) |
| `preprocessor.py` | 전처리: 업스케일→그레이→노출보정→**이진화(조건부)**→디스큐→패딩 |
| `tesseract_engine.py` | Tesseract OCR(`smart_run`: 언어 자동판별, `preserve_interword_spaces`) |
| **`vision.py`** | **provider 파사드** — `vision_provider`에 따라 Gemini/Claude 라우팅 |
| `gemini_vision.py` | Google Gemini 엔진(`google-genai`) |
| `claude_vision.py` | Anthropic Claude 엔진(`anthropic`) |
| `char_normalizer.py` | **L1** 규칙 보정(NFKC·비가시문자·합자·기호·혼동문자·줄끝하이픈) |
| `spellcheck.py` | **L2** 한국어 맞춤법(py-hanspell, 설치 시) |
| `llm_corrector.py` | **L3** 페이지별 문맥 교정(Claude Haiku, 별도 Anthropic 키 필요·기본 OFF) |
| **`text_cleaner.py`** | **문서 정제** — OCR 원문 전체를 Gemini로 교정(furniture 제거·페이지넘김 어절복원·띄어쓰기/맞춤법). 규칙 `C-xx`/`CK-xx`([정제규칙서](OCR도구_정제규칙서.md)) |
| `postprocessor.py` | 후처리: 마진노이즈 제거·단락 복원·공백 정규화 |
| `writer.py` | 결과 파일 생성(.txt/.md). `write_plain()` = 마커 없는 정제본 저장 |
| `storage_manager.py` | 결과 영구 저장(`storage/results/`)·업로드·이력 |
| `usage_tracker.py` | 클라우드 호출 토큰/비용 추정 기록 |
| `config.py` | 설정 로드/저장, API 키 관리(키별 환경변수 우선) |
| `types.py` | 공통 타입: `OCREngine`, `QualityReport`, `OCRResult`, `PageResult`, `FileResult` |

---

## 3. 처리 파이프라인

### 3-1. 전체 흐름
```
입력(detector) → 페이지 분해(pdf_splitter)
   └─ 페이지별:
        0) 빈 페이지 감지(blank_detector) → 건너뜀
        [Vision 우선 모드]  Gemini 즉시 → 실패 시 Tesseract
        [Tesseract 우선 모드]
           1) 품질 진단(quality_diagnoser)
           2) 그림·표 감지·마스킹(figure_detector)
           3) 품질 불량 + 키 있음 → Vision 즉시 폴백
           4) 전처리(preprocessor, 조건부)
           5) Tesseract OCR(tesseract_engine.smart_run)
           6) 저신뢰도/자모분리 + 키 있음 → Vision 폴백
              키 없음 → '검토 필요' 표시
   → 정규화/후처리(_finalize): L1 → L2 → postprocessor → L3
   → [문서 정제(text_cleaner, 기본 ON·전체 처리 시)]:
        전 페이지를 문장 끝 기준 청크로 묶어 Gemini 교정
        (furniture 제거·페이지넘김 어절복원·띄어쓰기/맞춤법; 부속물 제거는 옵션)
   → 파일 출력(writer; 정제본은 write_plain = 마커 없음) → 영구 저장(storage_manager)
```

> **문서 정제(text_cleaner)**: 페이지별 `_finalize`(L1~L3)가 '페이지 단위'라면, 이 단계는 **문서 전체 맥락**을 본다 — 그래서 페이지 넘김으로 잘린 어절 결합·러닝헤더 제거 같은 교차 페이지 처리가 가능하다. 규칙·청크 전략은 [정제규칙서](OCR도구_정제규칙서.md)(`C-01~C-11`, `CK-01~CK-02`) 참조. 미리보기는 속도 위해 제외.

### 3-2. 엔진 우선순위(모드)
- **Vision 우선**: 모든 페이지를 클라우드 엔진으로(레이아웃 이해·표→마크다운·고정확). 키 있을 때 기본.
- **Tesseract 우선**: 로컬 우선, **저품질·자모분리·저신뢰도** 페이지만 클라우드 폴백.
- 폴백 트리거: `confidence < CONFIDENCE_THRESHOLD(70)` 또는 자모 분리 감지, 또는 품질진단 `needs_fallback`(흐림<100·기울기>45°·세로쓰기·단변<500px).

### 3-3. 전처리 조건부 적용(품질 기반)
`preprocessor._is_clean_scan(report)` = `blur_score ≥ 150 AND avg_brightness ≥ 60`
- **깨끗한 스캔** → **Otsu 전역 이진화 + 잡음제거 생략**(얇은 한글 획 보존)
- **저품질/불균일** → **적응형(Adaptive Gaussian) 이진화 + Median 잡음제거**(구제)
- (흰 배경 문서는 평균 밝기가 높은 게 정상 → 상한 없음. 선명도가 핵심 판별자)

---

## 4. 엔진 provider 추상화

```
호출부(ocr_gradio.py / ocr_tool.py)
        │  from core import vision as claude_vision   # 별칭(기존 호환)
        ▼
   core/vision.py  ──(config.vision_provider)──┐
        ├─ "gemini" → core/gemini_vision.py      │  공통 인터페이스:
        └─ "claude" → core/claude_vision.py      │   run(image, lang_hint)->OCRResult
                                                 │   is_available() -> bool
```
- 파사드 함수: `run`, `run_text`(텍스트→텍스트, 정제용), `is_available`, `reset_client`, `set_session_key`, `provider_name`, `provider_label`.
- **엔진 전환**: `config.json`의 `"vision_provider"` 를 `"gemini"`↔`"claude"` 로만 변경(코드 수정 불필요).
- **모델 변경**: `"gemini_model"`(기본 `gemini-2.5-flash`) 값만 교체.

---

## 5. 품질 보정 체계

| 단계 | 모듈 | 내용 | 조건 |
|---|---|---|---|
| 전처리 | `preprocessor` | 이진화/디스큐/노출보정(조건부) | Tesseract 경로 |
| **L1** | `char_normalizer.normalize_all` | NFKC·비가시문자 제거(N-06)·합자·기호·숫자혼동·줄끝하이픈 | 항상 |
| **L2** | `spellcheck.correct_safe` | 한국어 맞춤법 | py-hanspell 설치 시 |
| 후처리 | `postprocessor.process_full` | (Tesseract)마진노이즈 제거 → 단락복원 → 공백정규화 | 엔진별 분기 |
| **L3** | `llm_corrector.correct` | Claude Haiku 페이지별 문맥 교정 | 옵션 활성 + Anthropic 키(기본 OFF) |
| **문서 정제** | `text_cleaner.iter_clean` | 문서 전체 Gemini 교정: furniture 제거·페이지넘김 어절복원·띄어쓰기/맞춤법 (+옵션 부속물 제거) | `enable_cleanup`(기본 ON) + Gemini 키 · 전체 처리 시 |

> **문서 정제 vs L3**: L3(`llm_corrector`)은 *페이지 단위·Claude Haiku·기본 OFF*의 레거시 보정이고, **문서 정제(`text_cleaner`)는 문서 전체·Gemini·기본 ON**의 주 교정 경로다. 교차 페이지 처리(어절 결합·러닝헤더 제거)는 문서 정제만 가능. 규칙 → [정제규칙서](OCR도구_정제규칙서.md).

> **설계 메모(원문 보존)**: L1은 고정밀 교정만 유지(예: 숫자 사이 `O→0`). 어두 `l→I`, 단음절 공백 제거, `...→…`·`--→—` 같은 **저정밀/스타일 변환은 제거**했다. 말줄임표·따옴표 정규화 등은 하위 정제 파이프라인이 담당.
> `postprocessor`의 마진노이즈 제거는 "선두 노이즈 토큰 2개 이상 연속"만 제거하고, 영어·표·숫자·제목 줄은 **보존**한다.

---

## 6. 설정 (`config.json`)

| 키 | 의미 | 기본값 |
|---|---|---|
| `anthropic_api_key` | Claude 키(선택: claude provider/L3용) | "" |
| `google_api_key` | Gemini 키 | "" |
| `vision_provider` | 클라우드 엔진 선택 `gemini`\|`claude` | `gemini` |
| `gemini_model` | Gemini 모델 ID(코드 수정 없이 변경) | `gemini-2.5-flash` |
| `default_lang` | OCR 언어 | `kor+eng` |
| `default_dpi` | PDF 래스터 해상도(Vision 내부 축소 → 200이면 충분) | 200 |
| `default_format` | 출력 포맷 | `txt` |
| `confidence_threshold` | 폴백/검토 임계값(%) | 70.0 |
| `auto_fallback` | 저신뢰도 자동 폴백 | true |
| `auto_spellcheck` | L2 자동 적용 | true |
| `save_figures` | 그림 크롭 저장 | false |
| `enable_cleanup` | **문서 정제** 단계 실행(Vision 기본 ON) | true |
| `cleanup_chunk_chars` | 정제 청크 목표 문자 수(soft, hard=2배) | 4000 |
| `remove_front_matter` | 앞뒤 **부속물 제거**(C-10/C-11) | false |

**키 우선순위**: 환경변수(`GEMINI_API_KEY`/`GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`) → `config.json`.
**키 가드**: 같은 키가 중복 연결된 손상값(`AIza`/`sk-ant` 2회 이상)은 무효 처리 → 401 크래시 방지.

---

## 7. 출력 / 저장

- `writer.write(result, dir, fmt)` → `.txt`/`.md` 생성(페이지 구분 `[페이지 N]`, 검토필요·그림 플레이스홀더 포함). **정제 OFF** 시 사용.
- `writer.write_plain(src, text, dir, fmt)` → **마커 없는 정제본** 저장(단락=빈 줄). **정제 ON**(기본) 시 사용.
- 사용자 결과 → 맨 위 **`02_결과/`**(`ocr_gradio.RESULTS_DIR`, 정제 중간본 `_정제중.txt` 포함).
- `storage_manager` → `04_처리엔진/storage/results/`(결과 사본·`_원문.txt`), `storage/uploads/`(원본), `storage/history.json`(이력). 웹 **③ 처리 기록** 탭에서 재다운로드. 저장 경로(`storage_path`)는 비워 두고 기본값을 쓴다(절대경로 저장 시 보관소가 갈라짐 — RETRO 오류 7).

---

## 8. 확장 방법

- **새 클라우드 엔진 추가**: `core/새엔진_vision.py`에 `run()`/`is_available()` 구현 → `core/vision.py._impl()`에 분기 추가 → `config` 값 추가.
- **모델 교체**: `config.json`의 `gemini_model` 변경.
- **새 L1 규칙**: `char_normalizer.normalize_all` 파이프라인에 함수 추가(고정밀 우선, 원문 보존).
- **전처리 튜닝**: `preprocessor._is_clean_scan` 임계값/이진화 분기 조정.
- **CLI 사용**: `python ocr_tool.py ocr -i 파일 -o 출력폴더 [-f md] [-l kor] [--llm-correct]`.

---

## 9. 테스트

```bat
python -m pytest tests/ -q      # 현재 157개 통과
```
주요 커버리지: 자모 감지, L1 보정(원문 보존 회귀 테스트 포함), 후처리 마진 제거/내용 보존, 전처리 이진화(Otsu/적응형), provider 라우팅, 키 없음 우아한 강등, **문서 정제**(`test_text_cleaner.py`: 청크 분할·문장끝 스냅·센티넬 처리·부속물 규칙·강등·클린 저장).

> 참고: 웹 UI 빌드(`build_ui`)는 import가 아닌 **실행 시점**에만 동작하므로, UI 변경은 `python ocr_gradio.py` 직접 실행으로 점검할 것.

---

## 10. 하위 파이프라인 연계 (관심사 분리)

```
[01] OCR 도구(본 도구)            →  충실 추출 + 문서 정제(교정) → 클린 플레인텍스트(.txt/.md)
        │                             (text_cleaner, C-xx/CK-xx)
        ▼
[02_01] 04_처리엔진/normalize_pipeline.py  →  명언/저자 추출·정규화(N-01~N-13/S-01~S-07)·중복제거
        (author_table.csv, 규칙서 .md 참조)
```
- **책→클린 본문 정제**(furniture 제거·어절 결합·띄어쓰기/맞춤법)는 **이 도구의 문서 정제 단계**가 담당 → [정제규칙서](OCR도구_정제규칙서.md).
- **명언 모음집→"명언_저자" 추출**은 하위 `normalize_pipeline.py`(N/S 규칙)가 담당. **산문 책엔 쓰지 않는다**(문장을 조각냄).
- 두 규칙 세트를 섞지 않는다 — 이중처리·원문 훼손 방지(관심사 분리).

---

## 11. 알려진 제약 / 주의

- **런처는 전용 `venv` Python을 사용**: `04_처리엔진\START_OCR.bat`(맨 위 `START_OCR.bat`이 호출)이 `%~dp0venv\Scripts`를 PATH 선두에 두어 **Windows 스토어 Python 스텁(0바이트, 실행 즉시 종료)·다중 Python 충돌을 회피**. → `install.bat`(또는 `python -m venv --system-site-packages venv`)으로 venv 생성 필요.
- **`gradio`는 requirements에 포함**(웹 UI 필수).
- **런처는 ASCII 파일명(`START_OCR.bat`)**: 한글 파일명 `.bat`은 일부 실행 경로에서 깨져 서버 시작 전 종료될 수 있어, 런처 파일명은 ASCII로 유지한다(경로의 한글 폴더명은 무관).
- **포트 7860 고정**: 다른 프로세스가 점유 시 기동 실패 → 런처가 7860 점유 프로세스 정리(유령 서버 방지) + 실패 시 `ocr_server.log` 안내.
- **Poppler 필요**(PDF만): 미설치 시 PDF 변환 실패(이미지 처리는 가능).
- **Vision은 LLM 기반**(Gemini/Claude): 드물게 환각 가능 → 프롬프트로 "추출만/요약 금지" 강제.
- **키 보안**: 키는 Windows 환경변수에만 둔다(2026-09-26). 웹 UI에는 키 입력칸이 없고 연결 상태·출처 변수 이름만 표시한다(`config.key_source()`). `config.json`은 동기화 폴더 내 평문이라 키를 넣지 않는다.

---

## 12. 변경 이력

### 2026-07-01 — 문서 정제 + 부속물 제거 + UI/CLI 현행화
| 항목 | 내용 |
|---|---|
| **문서 정제 신설** | `core/text_cleaner.py` — OCR 원문 전체를 Gemini로 교정(furniture 제거·페이지넘김 어절복원·띄어쓰기/맞춤법). `gemini_vision.run_text()`·`vision.run_text`·`writer.write_plain()` 추가. 규칙 = [정제규칙서](OCR도구_정제규칙서.md) `C-01~C-11`·`CK-01~CK-02` |
| **청크 문장끝 스냅** | `_chunk_pages` — soft/hard + hard 도달 시 마지막 문장 종결 페이지로 스냅(청크 경계 어절 안 잘림) |
| **✂ 부속물 제거 옵션** | `remove_front_matter`(기본 OFF) — 표제지·판권·차례·헌사·간기 제거, 본문만. 웹 토글 + config |
| **웹 UI 현행화** | Anthropic/Claude 잔존 표기 → 활성 엔진(Gemini) 동적화. 탭"크레딧 관리"→"사용량", 결제링크 aistudio, 사용량 라벨·요율(Gemini) 정정 |
| **CLI 현행화 + 버그수정** | `ocr_tool.py` provider-aware(setup가 활성 엔진 키 저장). `ocr`의 `llm_budget` NameError 수정(`--llm-budget` 신설) |
| 테스트 | 136 → **157** 통과(`test_text_cleaner.py` 16개 추가) |

### 2026-06-28 — Gemini 전환 / 품질·런처 개선
| 항목 | 내용 |
|---|---|
| 엔진 전환 | Claude Vision → **Google Gemini** 기본, `vision.py` provider 파사드 도입(무코드 전환) |
| 1순위 | 손상 API 키 제거 + 중복키 무효화 가드(401 크래시 방지) |
| 2순위 | `postprocessor.extract_korean_text` 완화 — 영어·표·숫자·제목 보존 |
| 3순위 | L1 과교정 제거(어두 l→I, 단음절 공백) + 비가시문자 제거(N-06) + `...→…`/`--→—` 제거(A안) |
| 4순위 | 전처리 조건부 이진화(Otsu/적응형) + Tesseract `preserve_interword_spaces` |
| 런처 | `START_OCR.bat` 보강 — 로그 기록(`ocr_server.log`)·시작 실패 안내·포트 기반 종료 |
| 런처(Python) | **전용 `venv` Python 사용**(PATH 선두) — Windows 스토어 스텁/다중 Python 충돌로 "콘솔 열렸다 닫힘" 문제 해결. `gradio`를 requirements에 포함 |
| 테스트 | 131 → **136** 통과(회귀 테스트 추가) |
