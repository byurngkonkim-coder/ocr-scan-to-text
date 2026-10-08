# 01_OCR_스캔파일텍스트추출프로그램

- **목적·기술 스택**: 스캔 파일 텍스트 추출(OCR) 프로그램 — `START_OCR.bat`, 처리엔진은 `engine/`
- **테스트 명령**: `python -m pytest engine/tests`
- **엔드유저 2계층 폴더 구조**:
  - `START_OCR.bat`, `OCR_대시보드_열기.url`: 사용자 실행 진입점 (User Layer)
  - `01_입력/`: 원본 PDF 투입 폴더 (User Layer)
  - `02_결과/`: 텍스트 추출 결과물 저장 폴더 (User Layer)
  - `engine/`: 핵심 처리 엔진 및 가상환경 (`core/`, `storage/`, `tests/`, `venv/` 등 Internal Layer)
  - `docs/`: 사용자 매뉴얼, 설계 가이드, 정제 규칙서 (Internal Layer)
  - `_백업/`: 이동 기록(`move_log_*.tsv`) 및 과거 보관물 (Internal Layer)
- **보호 경로**: `.env`, `config/**`, `legacy/**`, `_보관/**` — 사전 승인 없이 수정하지 않는다(`.claude/settings.json` 에서 수정 전 승인 요청).
- 공통 규칙: `~/.claude/rules/file-layout.md` · 상위 규약: `Raw/CLAUDE.md`(L0)
