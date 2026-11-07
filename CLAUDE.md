# 01_OCR_스캔파일텍스트추출프로그램

- **목적·기술 스택**: 스캔 파일 텍스트 추출(OCR) 프로그램 — `START_OCR.bat`, 처리엔진은 `engine/`
- **테스트 명령**: `python -m pytest engine/tests`
- **엔드유저 2계층 폴더 구조**:
  - `START_OCR.bat`, `OCR_대시보드_열기.url`: 사용자 실행 진입점 (User Layer)
  - `01_입력/`: 원본 PDF 투입 폴더 — ② 일괄 처리 기본 목록(`batch_runner.INPUT_DIR`). ② 탭에서 다른 폴더를 지정하면 `engine/config.json` 의 `batch_input_dir` 에 저장 (User Layer, git 제외)
  - `02_결과/`: 텍스트 추출 결과물 저장 폴더 (User Layer)
  - `engine/`: 핵심 처리 엔진 및 가상환경 (`core/`, `storage/`, `tests/`, `venv/` 등 Internal Layer)
  - `docs/`: 사용자 매뉴얼, 설계 가이드, 정제 규칙서 (Internal Layer)
  - `_백업/`: 이동 기록(`move_log_*.tsv`) 및 과거 보관물 (Internal Layer)
- **보호 경로**: `.env`, `config/**`, `legacy/**`, `_보관/**` — 사전 승인 없이 수정하지 않는다(`.claude/settings.json` 에서 수정 전 승인 요청).
- 공통 규칙: `~/.claude/rules/file-layout.md` · 상위 규약: `Raw/CLAUDE.md`(L0)

## 반복 실수와 막는 장치

사용자에게 같은 지적을 다시 받으면: 고치고, 아래 표에 규칙을 더한다. 규칙이 이미 있는데 막는 장치가 없었다면 재발이다 — 같은 변경에서 장치(검사·테스트·도구)를 만든다.

| 규칙 | 막는 장치 | 근거 |
|---|---|---|
| 백슬래시(`\n`·정규식·경로)가 든 코드·데이터는 Bash heredoc·sed 로 쓰지 않는다 | `engine/tests/test_source_hygiene.py` — `core/`·`tools/`·`tests/`·최상위 `.py` 를 SyntaxWarning 까지 오류로 컴파일, 실패 메시지가 파일:줄과 고칠 방법을 알려 준다 | lessons 2026-10-06(3회), 2026-11-07 세션(6회) |
| 결과 재처리는 `engine/tools/reprocess_book.py` 로만 한다 — 손으로 돌린 점검 스크립트를 믿지 않는다 | 도구가 끝에 점검을 출력하고 불합격이면 종료 코드 2: 제목 순서 어긋남, 본문 한글 3% 넘게 감소, 책별 최고 제목 수(`02_결과/_재처리/_점검기준.json`)보다 감소 | 2026-11-07: 21세기지식경영 48→44, 12년차기획맨 40→39 를 표 대조로만 찾음 |
| 지우는 규칙(차례 조각·광고·표 교체)을 바꾸면 순차 개선한 책들을 모두 다시 돌린다 | 위 점검 기준값(회귀면 실패) — 책 목록은 `docs/순차개선_기록.md` 회차 | 커밋 `39dab04`·`1177afd`·`8cc8f76`·`b2ae6fe`·`8828a46`, 2026-11-07 MBTI 표 |
| 시간에 기대는 테스트는 '끝나면 바로 빠져나오는' 대기에 상한을 넉넉히 둔다 | `tests/test_batch_runner.py::_wait_idle`(60초 상한) | 2026-11-07 전체 테스트 2회 거짓 실패 |
| `engine/core/toc_headings.py` 는 MyOCR 와 같은 파일 — 이 파일을 고치면 MyOCR 사본도 같은 변경에서 맞춘다. 번호표 같은 보조 처리는 `text_cleaner._strip_label` 처럼 바깥에 둔다 | `tests/test_toc_headings.py::test_shared_module_identical_to_myocr` — 실패를 '원래 있던 실패'로 넘기지 않는다 | 2026-11-06~11-07 매 실행 실패를 8회 넘김 → MyOCR `add3035` 로 맞춤(사용자 허용) |
| 새 정제 규칙은 '그대로 둬야 하는 예'(이름·명사·닫는 따옴표)를 테스트에 함께 넣는다 | 판단 사항 — 강제 장치 없음 | 2026-11-07: '최은경은'·'평가했다'·''조직 '에' 오교정 |
