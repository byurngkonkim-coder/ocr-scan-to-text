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
    (inp / "메모.docx").write_text("무시")
    (out / "20261005_120000_책A.txt").write_text("결과")
    monkeypatch.setattr(br, "INPUT_DIR", inp)
    monkeypatch.setattr(br, "_done_names", lambda d: (set(), [p.name for p in d.iterdir()]))
    rows = br.list_inputs(out)
    assert [(r["name"], r["status"]) for r in rows] == [("책A.png", "완료"), ("책B.png", "미처리")]
    assert "미처리" in br.choice_label(rows[1])


def test_partial_checkpoint_is_not_done(monkeypatch, tmp_path):
    # 실패한 실행이 남긴 빈 '_정제중' 복구본은 완료가 아니다(2026-10-17 일괄처리 실패 6건이 '완료'로 보임)
    inp, out = tmp_path / "in", tmp_path / "out"
    inp.mkdir()
    out.mkdir()
    for n in ("책A", "책B", "책C"):
        (inp / f"{n}.png").write_bytes(b"x")
    (out / "책A_정제중.txt").write_text("")
    (out / "책B_OCR중_20261017_010203.txt").write_text("")
    (out / "20261017_010203_책C.txt").write_text("결과")
    monkeypatch.setattr(br, "INPUT_DIR", inp)
    monkeypatch.setattr("core.storage_manager.load_history", lambda: [])
    rows = br.list_inputs(out)
    assert [(r["name"], r["status"]) for r in rows] == [
        ("책A.png", "미처리"), ("책B.png", "미처리"), ("책C.png", "완료")]


def test_list_inputs_includes_txt_for_cleanup_only(monkeypatch, tmp_path):
    inp, out = tmp_path / "in", tmp_path / "out"
    inp.mkdir()
    out.mkdir()
    (inp / "책C.txt").write_text("본문", encoding="utf-8")
    monkeypatch.setattr(br, "INPUT_DIR", inp)
    monkeypatch.setattr(br, "_done_names", lambda d: (set(), []))
    rows = br.list_inputs(out)
    assert [(r["name"], r["kind"]) for r in rows] == [("책C.txt", "텍스트(정제만)")]


def test_list_inputs_uses_chosen_folder(monkeypatch, tmp_path):
    default, chosen = tmp_path / "default", tmp_path / "chosen"
    default.mkdir()
    chosen.mkdir()
    (default / "기본.png").write_bytes(b"x")
    (chosen / "지정.png").write_bytes(b"x")
    monkeypatch.setattr(br, "INPUT_DIR", default)
    monkeypatch.setattr(br, "_done_names", lambda d: (set(), []))
    assert [r["name"] for r in br.list_inputs(tmp_path, chosen)] == ["지정.png"]
    assert [r["name"] for r in br.list_inputs(tmp_path)] == ["기본.png"]
    assert br.list_inputs(tmp_path, tmp_path / "없는폴더") == []


def test_default_input_dir_is_01_input():
    assert br.INPUT_DIR.name == "01_입력"


def test_text_pages_split_at_paragraphs():
    from ocr_gradio import _text_pages
    paras = [f"문단{i} " + "가" * 900 + "." for i in range(5)]
    pages = _text_pages("\n\n".join(paras), size=2000)
    assert len(pages) == 3                                   # 2+2+1 문단
    assert "\n\n".join(pages) == "\n\n".join(paras)          # 글자 손실 없음
    assert all(p.endswith(".") for p in pages)               # 문단 중간에서 자르지 않음
