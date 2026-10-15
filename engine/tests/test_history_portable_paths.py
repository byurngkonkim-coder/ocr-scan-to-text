"""처리 기록의 파일 경로가 폴더를 옮겨도(USB 복사·다른 PC) 살아 있어야 한다.

Regression: history.json 이 C:\\Users\\...\\02_결과\\x.txt 같은 절대경로를 저장해,
프로그램 폴더를 다른 위치로 복사하면 기록 표가 전부 ❌ 가 되고 결과 받기가 실패했다.
"""
import json
from pathlib import Path

from core import storage_manager as sm


def _setup(tmp_path, monkeypatch, entries):
    root = tmp_path / "새위치" / "01_OCR"
    (root / "02_결과").mkdir(parents=True)
    (root / "engine" / "storage" / "uploads").mkdir(parents=True)
    h = root / "engine" / "storage" / "history.json"
    h.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(sm, "_ROOT", root)
    monkeypatch.setattr(sm, "_BASE", root / "engine" / "storage")
    monkeypatch.setattr(sm, "_UPLOADS", root / "engine" / "storage" / "uploads")
    monkeypatch.setattr(sm, "_RESULTS", root / "engine" / "storage" / "results")
    monkeypatch.setattr(sm, "_HISTORY", h)
    return root, h


def test_old_absolute_paths_follow_moved_folder(tmp_path, monkeypatch):
    old = r"C:\Users\other\옛폴더\01_OCR"
    root, _ = _setup(tmp_path, monkeypatch, [{
        "id": "a1",
        "result_path": old + r"\02_결과\x.txt",
        "upload_path": r"C:\Users\other\텍스트 정규화\01_OCR\04_처리엔진\storage\uploads\x.pdf",
        "output_path": old + r"\02_결과\x.txt",
    }])
    (root / "02_결과" / "x.txt").write_text("결과", encoding="utf-8")
    (root / "engine" / "storage" / "uploads" / "x.pdf").write_bytes(b"%PDF")

    e = sm.load_history()[0]
    assert Path(e["result_path"]) == root / "02_결과" / "x.txt"
    assert Path(e["upload_path"]) == root / "engine" / "storage" / "uploads" / "x.pdf"
    row = sm.get_history_table()[0]
    assert row[6] == "✅" and row[7] == "✅"


def test_saved_history_has_no_absolute_paths(tmp_path, monkeypatch):
    root, h = _setup(tmp_path, monkeypatch, [])
    out = root / "02_결과" / "y.txt"
    out.write_text("결과", encoding="utf-8")
    sm.record("y.pdf", out, None, 1, 1, 0, 0, "rapidocr_first", 10, output_path=out)

    saved = json.loads(h.read_text(encoding="utf-8"))[0]
    assert saved["result_path"] == "02_결과/y.txt"
    assert saved["output_path"] == "02_결과/y.txt"
    assert saved["upload_path"] == ""
    assert Path(sm.load_history()[0]["result_path"]) == out


def test_path_outside_project_kept_as_is(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, [])
    ext = r"D:\다른곳\z.txt"
    assert sm._to_stored(ext) == ext
