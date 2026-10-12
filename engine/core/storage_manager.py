"""
파일 저장 관리 — 원본·결과 파일 보관 및 처리 이력 관리

디렉터리 구조:
  storage/
  ├── uploads/          원본 파일 (save_original=True 시)
  ├── results/          추출 텍스트 (.txt / .md)
  └── history.json      처리 이력 (최대 500건)
"""
from __future__ import annotations
import json
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path

# 기본 저장 경로 (도구 루트 기준)
_BASE = Path(__file__).parent.parent / "storage"
_UPLOADS = _BASE / "uploads"
_RESULTS = _BASE / "results"
_HISTORY = _BASE / "history.json"

MAX_HISTORY = 500   # 이력 최대 보관 건수


# ══════════════════════════════════════════════════════════════════════════════
# 초기화
# ══════════════════════════════════════════════════════════════════════════════

def init() -> None:
    """저장 디렉터리 생성 (최초 1회)."""
    _UPLOADS.mkdir(parents=True, exist_ok=True)
    _RESULTS.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 파일 저장
# ══════════════════════════════════════════════════════════════════════════════

def save_result(result_path: Path | str, original_name: str) -> Path:
    """
    추출 결과 파일을 storage/results/ 에 영구 복사.

    반환: storage/results/에 저장된 경로
    """
    init()
    src = Path(result_path)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = _safe_stem(original_name)
    dst  = _RESULTS / f"{ts}_{stem}{src.suffix}"

    shutil.copy2(src, dst)
    return dst


def raw_copy_path(result_path: Path | str) -> Path:
    """결과 파일의 '정제 전 원문' 사본 경로 — 저장과 삭제가 같은 규칙을 쓴다."""
    p = Path(result_path)
    return p.with_name(f"{p.stem}_원문{p.suffix}")


def save_raw_copy(raw_path: Path | str, stored_result: Path | str) -> Path:
    """정제 전 원문을 보관소 결과 파일 옆에 '결과이름_원문' 으로 복사."""
    dst = raw_copy_path(stored_result)
    shutil.copy2(raw_path, dst)
    return dst


def save_original(src_path: Path | str, original_name: str) -> Path:
    """
    원본 파일을 storage/uploads/ 에 복사.

    반환: storage/uploads/에 저장된 경로
    """
    init()
    src = Path(src_path)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = _safe_stem(original_name)
    dst  = _UPLOADS / f"{ts}_{stem}{src.suffix}"

    shutil.copy2(src, dst)
    return dst


# ══════════════════════════════════════════════════════════════════════════════
# 이력 관리
# ══════════════════════════════════════════════════════════════════════════════

_lock = threading.Lock()   # 처리 기록 파일 읽기-고치기-쓰기 보호


def record(
    original_name: str,
    result_path: Path | None,
    upload_path: Path | None,
    pages_total: int,
    pages_ocr: int,
    pages_blank: int,
    vision_pages: int,
    engine_mode: str,
    file_size_bytes: int,
    output_path: Path | None = None,
) -> str:
    """처리 이력 1건 추가. 생성된 entry_id 반환. output_path = 02_결과/ 확정본."""
    entry_id = str(uuid.uuid4())[:8]

    entry = {
        "id":            entry_id,
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original_name": original_name,
        "result_path":   str(result_path) if result_path else "",
        "upload_path":   str(upload_path) if upload_path else "",
        "output_path":   str(output_path) if output_path else "",
        "pages_total":   pages_total,
        "pages_ocr":     pages_ocr,
        "pages_blank":   pages_blank,
        "vision_pages":  vision_pages,
        "engine_mode":   engine_mode,       # 'vision_first' or 'tesseract_first'
        "file_size_kb":  round(file_size_bytes / 1024, 1),
        "result_size_kb": round(Path(result_path).stat().st_size / 1024, 1)
                          if result_path and Path(result_path).exists() else 0,
    }

    with _lock:     # 여러 파일 동시 처리 — 읽기-더하기-쓰기가 겹치면 기록이 빠진다
        history = load_history()
        history.insert(0, entry)              # 최신순 정렬
        _save_history(history[:MAX_HISTORY])
    return entry_id


def load_history() -> list[dict]:
    """이력 전체 로드."""
    if not _HISTORY.exists():
        return []
    try:
        return json.loads(_HISTORY.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        from core.config import backup_corrupt
        backup_corrupt(_HISTORY)   # 다음 저장이 깨진 이력을 빈 목록으로 덮어쓰지 않게 보존
        return []
    except Exception:
        return []


def delete_entry(entry_id: str, delete_files: bool = False) -> bool:
    """이력 1건 삭제. delete_files=True면 저장 파일도 삭제."""
    with _lock:
        history = load_history()
        target = next((e for e in history if e["id"] == entry_id), None)
        if not target:
            return False
        if delete_files:
            _delete_entry_files(target)
        _save_history([e for e in history if e["id"] != entry_id])
    return True


def clear_history(delete_files: bool = False) -> int:
    """이력 전체 삭제. 삭제된 건수 반환."""
    with _lock:
        history = load_history()
        if delete_files:
            for e in history:
                _delete_entry_files(e)
        _save_history([])
    return len(history)


# ══════════════════════════════════════════════════════════════════════════════
# 통계
# ══════════════════════════════════════════════════════════════════════════════

def get_stats() -> dict:
    """저장 현황 통계."""
    history = load_history()

    result_files  = list(_RESULTS.glob("*")) if _RESULTS.exists() else []
    upload_files  = list(_UPLOADS.glob("*")) if _UPLOADS.exists() else []

    result_size  = sum(f.stat().st_size for f in result_files if f.is_file())
    upload_size  = sum(f.stat().st_size for f in upload_files if f.is_file())

    return {
        "history_count":   len(history),
        "result_count":    len(result_files),
        "upload_count":    len(upload_files),
        "result_size_mb":  round(result_size / 1_048_576, 2),
        "upload_size_mb":  round(upload_size / 1_048_576, 2),
        "total_size_mb":   round((result_size + upload_size) / 1_048_576, 2),
        "storage_path":    str(_BASE.resolve()),
    }


def _engine_cell(e: dict) -> str:
    """엔진 모드 + OCR 경로면 Vision 처리 쪽 수 — Tesseract 로 대거 강등된 실행을 드러낸다."""
    mode = e.get("engine_mode", "")
    if mode in ("vision_first", "tesseract_first") and e.get("pages_total"):
        return f"{mode} · Vision {e.get('vision_pages', 0)}/{e['pages_total']}쪽"
    return mode


def get_history_table() -> list[list]:
    """Gradio Dataframe 용 테이블 데이터 반환."""
    rows = []
    for e in load_history():
        has_result = bool(e.get("result_path")) and Path(e["result_path"]).exists()
        has_upload = bool(e.get("upload_path")) and Path(e["upload_path"]).exists()
        rows.append([
            e.get("timestamp", ""),
            e.get("original_name", ""),
            e.get("pages_total", 0),
            e.get("pages_blank", 0),
            _engine_cell(e),
            f"{e.get('result_size_kb', 0):.0f} KB",
            "✅" if has_result else "❌",
            "✅" if has_upload else "—",
            e.get("id", ""),
        ])
    return rows if rows else [["—"] * 9]


# ══════════════════════════════════════════════════════════════════════════════
# 내부 유틸
# ══════════════════════════════════════════════════════════════════════════════

def _delete_entry_files(entry: dict) -> None:
    """이력 1건에 딸린 저장 파일 삭제 — 결과·정제 전 원문 사본·업로드 원본·02_결과 확정본."""
    result, upload, output = (entry.get(k, "").strip()
                              for k in ("result_path", "upload_path", "output_path"))
    paths = [Path(p) for p in (result, upload, output) if p]   # 빈 문자열 → 건너뜀 (Path("") == "." 방지)
    if result:
        paths.append(raw_copy_path(result))
    for p in paths:
        if p.is_file():      # is_file()로 디렉터리·현재경로 오삭제 방지
            try:
                p.unlink()
            except Exception:
                pass


def _safe_stem(name: str) -> str:
    """파일명에 사용할 수 없는 문자 제거."""
    import re
    stem = Path(name).stem
    return re.sub(r'[\\/:*?"<>|]', "_", stem)[:50]


def _save_history(history: list[dict]) -> None:
    from core.config import atomic_write_json
    init()
    atomic_write_json(_HISTORY, history)
