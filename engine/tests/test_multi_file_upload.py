"""① 탭 여러 파일 끌어다 놓기 — 올린 순서대로 하나씩 처리, 한 파일 실패해도 계속."""
from types import SimpleNamespace

import pytest

import batch_runner as br

ocr_gradio = pytest.importorskip("ocr_gradio")


def _fake_full(order):
    def fake(file_obj, *rest, **kw):
        name = file_obj.name
        order.append(name)
        if name == "bad.pdf":
            raise RuntimeError("깨짐")
        yield ("bar", "진행", f"본문 {name}", f"out_{name}.txt", None, None)
    return fake


def _run(files):
    br.direct_running.clear()
    return list(ocr_gradio.process_full_or_queue(files, "kor", "txt", False, False, False,
                                                 False, 0, "", "Tesseract 우선",
                                                 progress=lambda *a, **k: None))


def test_multiple_files_processed_in_order(monkeypatch):
    order = []
    monkeypatch.setattr(ocr_gradio, "process_full", _fake_full(order))
    monkeypatch.setattr(br, "busy", lambda: False)
    files = [SimpleNamespace(name=n) for n in ("a.pdf", "bad.pdf", "c.pdf")]
    outs = _run(files)
    assert order == ["a.pdf", "bad.pdf", "c.pdf"]
    final = outs[-1]
    assert final[3] == ["out_a.pdf.txt", "out_c.pdf.txt"]
    assert "성공 2개 / 실패 1개" in final[1]
    assert not br.direct_running.is_set()


def test_single_file_list_keeps_single_path(monkeypatch):
    order = []
    monkeypatch.setattr(ocr_gradio, "process_full", _fake_full(order))
    monkeypatch.setattr(br, "busy", lambda: False)
    outs = _run([SimpleNamespace(name="a.pdf")])
    assert order == ["a.pdf"]
    assert outs[-1][3] == "out_a.pdf.txt"


def test_multiple_files_go_to_queue_when_busy(monkeypatch):
    sent = []
    monkeypatch.setattr(br, "busy", lambda: True)
    monkeypatch.setattr(br, "submit", lambda paths, *a: sent.extend(paths))
    _run([SimpleNamespace(name="a.pdf"), SimpleNamespace(name="b.pdf")])
    assert sent == ["a.pdf", "b.pdf"]
