"""파일 안 병렬 처리 — 정제 구간·Vision 쪽을 동시에 보내도 결과·기록 순서가 유지되는지."""
import random
import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import config, text_cleaner
from core.types import OCREngine, PageResult


def _pages(n):
    # 쪽마다 문장으로 끝나고 4000자를 넘겨 쪽 하나 = 구간 하나가 되게
    return [f"{k}번째 쪽의 본문 문장이 여기 있다. " * 240 for k in range(n)]


def test_cleanup_chunks_merge_in_order_even_if_answers_arrive_shuffled(monkeypatch):
    monkeypatch.setattr(config, "get_parallel_workers", lambda: 4)

    def run_text(prompt):
        time.sleep(random.uniform(0, 0.05))          # 끝나는 순서를 뒤섞는다
        return prompt.split("---\n")[1].split("\n---")[0].replace(text_cleaner.PAGEBREAK, "")
    monkeypatch.setattr(text_cleaner._gemini, "run_text", run_text)
    written = []
    gen = text_cleaner.iter_clean(_pages(12), on_chunk=written.append)
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        out = stop.value
    firsts = [int(re.match(r"(\d+)번째", w).group(1)) for w in written]
    assert firsts == list(range(12))                 # 체크포인트도 구간 순서대로
    assert out.index("0번째") < out.index("5번째") < out.index("11번째")
    assert text_cleaner.get_last_report()["chunks"] == 12


def test_parallel_cleanup_is_faster_than_sequential(monkeypatch):
    monkeypatch.setattr(text_cleaner._gemini, "run_text",
                        lambda p: (time.sleep(0.2), p.split("---\n")[1].split("\n---")[0])[1])
    timings = {}
    for workers in (1, 4):
        monkeypatch.setattr(config, "get_parallel_workers", lambda w=workers: w)
        t0 = time.time()
        text_cleaner.clean_pages(_pages(8))
        timings[workers] = time.time() - t0
    assert timings[4] < timings[1] / 2


def test_concurrent_history_records_are_not_lost(monkeypatch, tmp_path):
    # 여러 파일 동시 처리 — 읽기-더하기-쓰기가 겹치면 기록이 빠지던 것(잠금으로 방지)
    import threading
    from core import storage_manager
    for name, sub in (("_BASE", ""), ("_UPLOADS", "uploads"), ("_RESULTS", "results"),
                      ("_HISTORY", "history.json")):
        monkeypatch.setattr(storage_manager, name, tmp_path / "storage" / sub)
    ts = [threading.Thread(target=storage_manager.record,
                           args=(f"책{k}.pdf", None, None, 1, 1, 0, 0, "vision_first", 100))
          for k in range(20)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(storage_manager.load_history()) == 20


def test_cleanup_report_is_per_run(monkeypatch):
    # 동시 실행 시 전역 get_last_report() 는 마지막 실행 것 — report_out 은 각자 것
    monkeypatch.setattr(text_cleaner._gemini, "run_text", lambda p: "")
    mine: dict = {}
    gen = text_cleaner.iter_clean(["본문이 있다."], report_out=mine)
    for _ in gen:
        pass
    text_cleaner.clean_pages(["다른 실행.", "또 다른 쪽."])
    assert mine["chunks"] == 1 and mine["fallback_guard"] == 1


def test_ocr_pages_saved_in_page_order(monkeypatch, tmp_path):
    ocr_gradio = pytest.importorskip("ocr_gradio")
    from core import storage_manager
    monkeypatch.setattr(config, "get_parallel_workers", lambda: 4)
    monkeypatch.setattr(ocr_gradio, "RESULTS_DIR", tmp_path / "results")
    for name, sub in (("_BASE", ""), ("_UPLOADS", "uploads"), ("_RESULTS", "results"),
                      ("_HISTORY", "history.json")):
        monkeypatch.setattr(storage_manager, name, tmp_path / "storage" / sub)
    monkeypatch.setattr(ocr_gradio, "_page_plan",
                        lambda path, start_page=1, **k: (10, ((n, None) for n in range(1, 11))))

    def process_page(img, n, *a, **k):
        time.sleep(random.uniform(0, 0.05))
        text = f"{n}쪽 본문입니다."
        return PageResult(page_num=n, raw_text=text, corrected_text=text,
                          engine=OCREngine.RAPIDOCR, confidence=95.0), []
    monkeypatch.setattr(ocr_gradio, "_process_page", process_page)
    png = Path(__file__).parent / "sample_docs" / "sample_kor_eng.png"
    last = None
    for last in ocr_gradio.process_full(
            SimpleNamespace(name=str(png)), "kor+eng", "txt", False, False, True, False, 0, "",
            "RapidOCR 우선", cleanup=False, front_matter=False, progress=lambda *a, **k: None):
        pass
    text = Path(last[3]).read_text(encoding="utf-8")
    nums = [int(x) for x in re.findall(r"(\d+)쪽 본문", text)]
    assert nums == list(range(1, 11))
