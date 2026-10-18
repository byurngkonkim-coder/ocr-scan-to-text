"""CLI 폴더 처리에서 이름이 같은 파일('상권/본문.png'·'하권/본문.png')의 결과가 덮어써지던 것(감사 28번)."""
from pathlib import Path

import pytest

ocr_tool = pytest.importorskip("ocr_tool")
from core.types import OutputFormat


def test_same_stem_files_get_separate_results(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr_tool, "_WRITTEN", {})
    out = str(tmp_path / "out")
    a, b = tmp_path / "상권" / "본문.png", tmp_path / "하권" / "본문.png"
    assert ocr_tool._unique_stem(a, out, OutputFormat.TXT) == "본문"
    assert ocr_tool._unique_stem(b, out, OutputFormat.TXT) == "본문_2"
    assert ocr_tool._unique_stem(a, out, OutputFormat.TXT) == "본문"     # 같은 파일 다시 쓰기는 같은 이름
