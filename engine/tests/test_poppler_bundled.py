"""프로그램 폴더 안의 poppler(engine/poppler)를 C:\\poppler 보다 먼저 쓴다.

Regression: poppler 위치가 C:\\poppler 로 고정돼, USB 로 복사한 다른 PC 에서는 PDF 처리가 안 됐다.
"""
from core import pdf_splitter, pdf_text_layer


def test_bundled_poppler_preferred(tmp_path, monkeypatch):
    (tmp_path / "pdftoppm.exe").write_bytes(b"")
    (tmp_path / "pdftotext.exe").write_bytes(b"")
    monkeypatch.setattr(pdf_splitter, "BUNDLED_POPPLER", tmp_path)
    assert pdf_splitter._poppler_kwargs(None) == {"poppler_path": str(tmp_path)}
    assert pdf_text_layer._find_pdftotext() == str(tmp_path / "pdftotext.exe")


def test_explicit_path_still_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_splitter, "BUNDLED_POPPLER", tmp_path)
    assert pdf_splitter._poppler_kwargs("X:/p") == {"poppler_path": "X:/p"}
