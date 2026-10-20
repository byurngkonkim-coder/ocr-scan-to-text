"""② 일괄 처리 대기열 — 차례 처리·중단·① 탭 처리와 겹치지 않기·01_입력 목록."""
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


def test_suspect_list_saved_next_to_result(monkeypatch, tmp_path):
    # 결과 txt 를 저장하면 의심 목록도 02_결과/_의심목록/ 에 남는다(intent 2026-10-17)
    from core import typo_suspects
    result = tmp_path / "책A.txt"
    result.write_text("걷모습만 보고", encoding="utf-8")

    def process_full(file_obj, *args, progress=None, **kwargs):
        yield ("", "완료", "", str(result), None, None)

    saved = []
    monkeypatch.setattr(ocr_gradio, "process_full", process_full)
    monkeypatch.setattr(typo_suspects, "save", lambda p: (saved.append(p), (p.parent / "x.tsv", 3))[1])
    br.submit([tmp_path / "책A.pdf"], ("kor+eng",), {"cleanup": True})
    _wait_idle()
    assert saved == [result]
    assert br._jobs[0].state == "완료"
    assert "의심 3곳" in br._jobs[0].detail


def test_suspect_list_failure_does_not_fail_job(monkeypatch, tmp_path):
    from core import typo_suspects
    result = tmp_path / "책A.txt"
    result.write_text("본문", encoding="utf-8")

    def process_full(file_obj, *args, progress=None, **kwargs):
        yield ("", "완료", "", str(result), None, None)

    def boom(p):
        raise RuntimeError("kiwi")
    monkeypatch.setattr(ocr_gradio, "process_full", process_full)
    monkeypatch.setattr(typo_suspects, "save", boom)
    br.submit([tmp_path / "책A.pdf"], ("kor+eng",), {"cleanup": True})
    _wait_idle()
    assert br._jobs[0].state == "완료"


def test_batch_never_uses_vision_fallback(monkeypatch, tmp_path):
    # 일괄 처리는 로컬 전용 — 화면에서 폴백을 켜 둬도 API 다시 읽기를 끈다(사용자 설계원칙 2026-10-18)
    seen = []

    def process_full(file_obj, *args, progress=None, **kwargs):
        seen.append(args)
        yield ("", "완료", "", None, None, None)
    monkeypatch.setattr(ocr_gradio, "process_full", process_full)
    args = ("kor+eng", "txt", False, False, True, False, 0, "", "RapidOCR 우선")
    br.submit([tmp_path / "책A.pdf"], args, {"cleanup": True})
    _wait_idle()
    assert seen[0][br.NO_FALLBACK_ARG] is True
    assert seen[0][:3] == args[:3] and seen[0][4:] == args[4:]


def test_review_log_noted_in_detail(monkeypatch, tmp_path):
    # 처리 중 남긴 검수기록이 있으면 ② 상세에 위치를 알린다(intent 2026-10-18 정제 순서 5단계)
    from core import typo_suspects
    result = tmp_path / "책A.txt"
    result.write_text("본문", encoding="utf-8")
    (tmp_path / "_검수기록").mkdir()
    (tmp_path / "_검수기록" / "책A.txt").write_text("기록", encoding="utf-8")

    def process_full(file_obj, *args, progress=None, **kwargs):
        yield ("", "완료", "", str(result), None, None)
    monkeypatch.setattr(ocr_gradio, "process_full", process_full)
    monkeypatch.setattr(typo_suspects, "save", lambda p: (p, 0))
    br.submit([tmp_path / "책A.pdf"], ("kor+eng",), {"cleanup": True})
    _wait_idle()
    assert br._jobs[0].state == "완료"
    assert "검수기록: _검수기록/책A.txt" in br._jobs[0].detail


def test_suspect_subfolder_is_not_a_result(monkeypatch, tmp_path):
    inp, out = tmp_path / "in", tmp_path / "out"
    inp.mkdir(); out.mkdir()
    (inp / "책A.pdf").write_bytes(b"%PDF")
    (out / "_의심목록").mkdir()
    (out / "_의심목록" / "책A.tsv").write_text("")
    monkeypatch.setattr("core.storage_manager.load_history", lambda: [])
    monkeypatch.setattr(br, "_file_kind", lambda p: (1, "스캔"))
    assert br.list_inputs(out, inp)[0]["status"] == "미처리"


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


def test_partial_render_result_is_failure_not_done(monkeypatch, tmp_path):
    # 쪽 그리기 실패로 앞부분만 저장한 결과를 '완료'로 표시하던 것
    def process_full(file_obj, *args, progress=None, **kwargs):
        yield ("", "⚠ 부분 저장  |  전체 10p 중 3p까지만 처리 — 다시 처리해 주세요  |  OCR 처리 3p",
               "", f"{file_obj.name}_미완.txt", None, None)
    monkeypatch.setattr(ocr_gradio, "process_full", process_full)
    br.submit([tmp_path / "a.pdf"], ("kor+eng",), {"cleanup": True})
    _wait_idle()
    job = br._jobs[0]
    assert job.state == "실패" and "3p까지만" in job.detail


def test_unfinished_result_is_not_done(monkeypatch, tmp_path):
    inp, out = tmp_path / "in", tmp_path / "out"
    inp.mkdir()
    out.mkdir()
    (inp / "책A.png").write_bytes(b"x")
    (out / "20261017_010203_책A_미완.txt").write_text("앞부분")
    monkeypatch.setattr(br, "INPUT_DIR", inp)
    monkeypatch.setattr("core.storage_manager.load_history", lambda: [])
    assert br.list_inputs(out)[0]["status"] == "미처리"


def test_batch_never_uses_paid_l3_correction(monkeypatch, tmp_path):
    # ① 고급 설정의 'L3 쪽마다 교정(Claude 유료)'을 켜 둔 채 일괄을 돌리면 쪽마다 유료 호출이 나갔다
    seen = []

    def process_full(file_obj, *args, progress=None, **kwargs):
        seen.append(args)
        yield ("", "완료", "", None, None, None)
    monkeypatch.setattr(ocr_gradio, "process_full", process_full)
    args = ("kor+eng", "txt", False, False, True, True, 0, "", "RapidOCR 우선")
    br.submit([tmp_path / "책A.pdf"], args, {"cleanup": True})
    _wait_idle()
    assert seen[0][br.LLM_CORRECT_ARG] is False


def test_list_inputs_does_not_match_partial_names(monkeypatch, tmp_path):
    # '경영.pdf' 가 '…_경영전략.txt' 결과 때문에 '완료'로 보여 처리 대상에서 빠지던 것
    inp, out = tmp_path / "in", tmp_path / "out"
    inp.mkdir()
    out.mkdir()
    (inp / "경영.png").write_bytes(b"x")
    (inp / "경영전략.png").write_bytes(b"x")
    (out / "20261005_120000_경영전략.txt").write_text("결과")
    monkeypatch.setattr(br, "INPUT_DIR", inp)
    monkeypatch.setattr("core.storage_manager.load_history", lambda: [])
    monkeypatch.setattr(br, "_file_kind", lambda p: (1, "스캔"))
    rows = br.list_inputs(out)
    assert [(r["name"], r["status"]) for r in rows] == [("경영.png", "미처리"), ("경영전략.png", "완료")]


def test_margin_setting_changed_mid_batch_does_not_leak(monkeypatch, tmp_path):
    # ① 탭에서 처리 버튼을 누르면 여백 설정이 저장되는데, 돌고 있는 일괄 작업의 다음 쪽부터 바로 적용됐다
    seen = []
    state = {"on": False}
    monkeypatch.setattr(config, "load", lambda: {"enable_margin_filter": state["on"]})

    def process_full(file_obj, *args, progress=None, **kwargs):
        seen.append(config.get_enable_margin_filter())
        state["on"] = True                      # 작업 도중 ① 탭이 설정을 바꿈
        from concurrent.futures import ThreadPoolExecutor
        import contextvars
        with ThreadPoolExecutor(1) as ex:       # 쪽 읽기는 작업 스레드에서 돈다
            seen.append(ex.submit(contextvars.copy_context().run, config.get_enable_margin_filter).result())
        yield ("", "완료", "", None, None, None)
    monkeypatch.setattr(ocr_gradio, "process_full", process_full)
    br.submit([tmp_path / "a.pdf"], ("kor+eng",), {})
    _wait_idle()
    assert seen == [False, False]
    assert config.get_enable_margin_filter() is True       # 일괄 밖(① 탭)은 바뀐 설정을 그대로 쓴다


def test_stop_then_immediate_resubmit_still_stops_running_job(monkeypatch, tmp_path):
    # 중단 직후 다시 제출하면 submit 이 전체 중단 표시를 지워, 처리 중이던 책이 끝까지 돌았다(감사 23번)
    monkeypatch.setattr(ocr_gradio, "process_full", _fake_process([], delay=0.0))

    def slow(file_obj, *args, progress=None, **kwargs):
        for k in range(30):
            time.sleep(0.05)
            yield ("", f"{k}", "", None, None, None)
        yield ("", "완료", "", f"{file_obj.name}.txt", None, None)
    monkeypatch.setattr(ocr_gradio, "process_full", slow)
    a = tmp_path / "a.pdf"
    br.submit([a], ("kor+eng",), {})
    time.sleep(0.2)
    br.stop()
    assert br.submit([a], ("kor+eng",), {}) == 1          # 다시 넣은 것은 대기열에 들어간다
    _wait_idle(10)
    assert br._jobs[0].state == "중단"


def test_submit_while_last_worker_is_exiting_does_not_stall(monkeypatch, tmp_path):
    # 마지막 작업자가 '할 일 없음'으로 끝나는 틈에 submit 하면 새 작업자를 안 띄워 영원히 '대기'였다
    monkeypatch.setattr(ocr_gradio, "process_full", _fake_process([], delay=0.1))
    real = br._claim_job

    def slow_claim():
        job = real()
        if job is None:
            time.sleep(0.5)            # 끝나기 직전의 틈을 넓힌다
        return job
    monkeypatch.setattr(br, "_claim_job", slow_claim)
    br.submit([tmp_path / "a.pdf"], ("kor+eng",), {})
    time.sleep(0.3)                    # a 완료 → 작업자가 '할 일 없음' 틈에 있음
    br.submit([tmp_path / "b.pdf"], ("kor+eng",), {})
    _wait_idle(5)
    assert [j.state for j in br._jobs] == ["완료", "완료"]
