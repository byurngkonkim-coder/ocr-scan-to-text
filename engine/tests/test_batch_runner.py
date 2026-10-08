"""④ 일괄 처리 대기열 — 차례 처리·중단·① 탭 처리와 겹치지 않기·01_입력 목록."""
import time

import pytest

import batch_runner as br
from core import config

ocr_gradio = pytest.importorskip("ocr_gradio")


def _join_workers():
    for w in list(br._workers):
        w.join(timeout=5)


@pytest.fixture(autouse=True)
def _clean_queue(monkeypatch):
    br.stop()
    _join_workers()
    br._jobs.clear()
    br._stop.clear()
    br.direct_running.clear()
    monkeypatch.setattr(config, "get_batch_workers", lambda: 1)
    yield
    br.stop()
    _join_workers()
    br._jobs.clear()


def _fake_process(order, delay=0.0):
    def process_full(file_obj, *args, progress=None, **kwargs):
        order.append(file_obj.name)
        yield ("", "정제 1/2 구간", "", None, None, None)
        time.sleep(delay)
        yield ("", "완료", "", f"{file_obj.name}.txt", None, None)
    return process_full


def _wait_idle(timeout=5):
    end = time.time() + timeout
    while br.busy() and time.time() < end:
        time.sleep(0.05)


def test_files_processed_in_order_and_marked_done(monkeypatch, tmp_path):
    order = []
    monkeypatch.setattr(ocr_gradio, "process_full", _fake_process(order))
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    assert br.submit([a, b], ("kor+eng",), {"cleanup": True}) == 2
    _wait_idle()
    assert order == [str(a), str(b)]
    assert [j.state for j in br._jobs] == ["완료", "완료"]
    assert br._jobs[0].result == f"{a}.txt"


def test_same_file_not_queued_twice(monkeypatch, tmp_path):
    monkeypatch.setattr(ocr_gradio, "process_full", _fake_process([], delay=0.3))
    a = tmp_path / "a.pdf"
    br.submit([a], (), {})
    assert br.submit([a], (), {}) == 0


def test_stop_cancels_waiting_jobs(monkeypatch, tmp_path):
    monkeypatch.setattr(ocr_gradio, "process_full", _fake_process([], delay=0.5))
    br.submit([tmp_path / "a.pdf", tmp_path / "b.pdf"], (), {})
    time.sleep(0.1)
    br.stop()
    _join_workers()
    assert [j.state for j in br._jobs] == ["중단", "취소"]


def test_waits_while_tab1_is_processing(monkeypatch, tmp_path):
    order = []
    monkeypatch.setattr(ocr_gradio, "process_full", _fake_process(order))
    br.direct_running.set()
    br.submit([tmp_path / "a.pdf"], (), {})
    time.sleep(0.3)
    assert order == []                      # ① 탭 처리가 끝나기 전엔 시작하지 않는다
    br.direct_running.clear()
    _join_workers()
    assert order == [str(tmp_path / "a.pdf")]


def test_two_files_processed_at_the_same_time(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "get_batch_workers", lambda: 2)
    order = []
    monkeypatch.setattr(ocr_gradio, "process_full", _fake_process(order, delay=0.5))
    t0 = time.time()
    br.submit([tmp_path / "a.pdf", tmp_path / "b.pdf", tmp_path / "c.pdf"], (), {})
    _wait_idle()
    took = time.time() - t0
    assert sorted(order) == sorted(str(tmp_path / n) for n in ("a.pdf", "b.pdf", "c.pdf"))
    assert all(j.state == "완료" for j in br._jobs)
    assert took < 1.4          # 차례로면 1.5초+, 둘씩이면 약 1.0초


def test_list_inputs_marks_done_by_result_name(monkeypatch, tmp_path):
    inp, out = tmp_path / "in", tmp_path / "out"
    inp.mkdir()
    out.mkdir()
    (inp / "책A.png").write_bytes(b"x")
    (inp / "책B.png").write_bytes(b"x")
    (inp / "메모.txt").write_text("무시")
    (out / "20261005_120000_책A.txt").write_text("결과")
    monkeypatch.setattr(br, "INPUT_DIR", inp)
    monkeypatch.setattr(br, "_done_names", lambda d: (set(), [p.name for p in d.iterdir()]))
    rows = br.list_inputs(out)
    assert [(r["name"], r["status"]) for r in rows] == [("책A.png", "완료"), ("책B.png", "미처리")]
    assert "미처리" in br.choice_label(rows[1])
