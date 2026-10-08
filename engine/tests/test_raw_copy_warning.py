"""정제 전 원문 사본(안전망) 저장 실패가 화면에 경고로 드러나는지 고정한다.

배경: 원문 사본 작성·보관 실패를 except: pass 로 삼켜, 안전망이 없는데도
화면엔 '완료'만 떴다(RETRO_텍스트정규화 오류 19).
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

import ocr_gradio
from core import storage_manager, text_cleaner
from core.types import OCREngine, PageResult


@pytest.fixture
def digital_pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    p = tmp_path / "digital.pdf"
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), ("bold leadership moment of truth " * 12) + f" page{i + 1}.",
                         fontsize=11)
    doc.save(str(p))
    doc.close()
    return p


@pytest.fixture
def scanned_image(monkeypatch):
    """OCR 경로: PNG 입력 + 페이지 판독만 고정 문장으로 대체(Tesseract 불필요).
    정제·원문 사본 작성·보관은 실제 코드를 탄다."""
    text = "스캔본 본문입니다. " * 60 + "끝."

    def fake_page(image, page_num, *args, **kwargs):
        return PageResult(page_num=page_num, raw_text=text, corrected_text=text,
                          engine=OCREngine.TESSERACT, confidence=95.0), []

    monkeypatch.setattr(ocr_gradio, "_process_page", fake_page)
    return Path(__file__).parent / "sample_docs" / "sample_kor_eng.png"


@pytest.fixture(params=["digital_pdf", "scanned_image"])
def source(request):
    """두 처리 경로(디지털 PDF 직접 추출 / OCR) 모두에서 같은 동작을 확인한다."""
    return request.param, request.getfixturevalue(request.param)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """결과 폴더·내부 보관소를 임시 폴더로, 정제는 입력을 그대로 돌려주게."""
    monkeypatch.setattr(ocr_gradio, "RESULTS_DIR", tmp_path / "results")
    base = tmp_path / "storage"
    monkeypatch.setattr(storage_manager, "_BASE", base)
    monkeypatch.setattr(storage_manager, "_UPLOADS", base / "uploads")
    monkeypatch.setattr(storage_manager, "_RESULTS", base / "results")
    monkeypatch.setattr(storage_manager, "_HISTORY", base / "history.json")
    monkeypatch.setattr(text_cleaner, "is_available", lambda: True)
    monkeypatch.setattr(text_cleaner._gemini, "run_text", lambda prompt: prompt)


def _final_status(pdf) -> str:
    last = None
    for last in ocr_gradio.process_full(
        SimpleNamespace(name=str(pdf)), "kor+eng", "txt",
        False, False, True, False, 0, "", "Tesseract 우선",
        cleanup=True, front_matter=False, progress=lambda *a, **k: None,
    ):
        pass
    return str(last[1])


def test_warns_when_raw_copy_cannot_be_written(source, sandbox, monkeypatch):
    _, src = source
    real_write = ocr_gradio.writer.write

    def failing_write(result, output_dir, fmt):
        if "raw" in str(output_dir):
            raise OSError("디스크 가득 참")
        return real_write(result, output_dir, fmt)

    monkeypatch.setattr(ocr_gradio.writer, "write", failing_write)
    status = _final_status(src)
    assert "✅ 완료" in status          # 정제본 자체는 정상 저장
    assert "원문 사본" in status and "디스크 가득 참" in status


def test_warns_when_raw_copy_cannot_be_stored(source, sandbox, monkeypatch):
    _, src = source
    def failing_save(raw_path, stored_result):
        raise OSError("권한 없음")

    monkeypatch.setattr(storage_manager, "save_raw_copy", failing_save)
    status = _final_status(src)
    assert "✅ 완료" in status
    assert "원문 사본" in status and "권한 없음" in status


def test_no_warning_when_raw_copy_succeeds(source, sandbox):
    kind, src = source
    status = _final_status(src)
    # 의도한 경로를 실제로 탔는지 — 디지털은 텍스트 레이어, 이미지는 OCR 요약
    assert ("텍스트 레이어" in status) == (kind == "digital_pdf")
    assert "원문 사본" not in status
    saved = list(storage_manager._RESULTS.glob("*_원문*"))
    assert len(saved) == 1
    # 이력에 02_결과 확정본 경로가 남아야 '파일도 함께 삭제' 때 지워진다
    output = Path(storage_manager.load_history()[0]["output_path"])
    assert output.parent == ocr_gradio.RESULTS_DIR and output.is_file()
