"""④ 일괄 처리 — 입력 폴더 목록과 백그라운드 대기열.

화면(브라우저)과 분리된 작업 스레드들이 대기열을 처리한다. 브라우저를 닫아도
처리는 계속되고, 화면은 주기적으로 status_html() 을 읽어 보여 준다.
처리 자체는 ① 탭의 '전체 처리'(ocr_gradio.process_full)를 그대로 재사용하므로 디지털 판별,
깨진 쪽 다시 읽기, 정제, 부속물 삭제, 02_결과 저장, 처리 기록이 모두 같다.

동시에 config batch_workers(기본 2)개 파일을 처리한다. 그래서 정제 보고서는 실행마다 따로
받고(iter_clean report_out), 처리 기록·사용량 파일은 잠금으로, PyMuPDF 추출은 한 번에 하나로
보호한다. ① 탭에서 직접 처리 중이면(direct_running) 그 처리가 끝날 때까지 새 파일을 시작하지 않는다.
"""
from __future__ import annotations

import html
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

INPUT_DIR = Path(__file__).resolve().parent.parent / "01_입력"   # 기본값 — ④ 탭에서 다른 폴더로 바꿀 수 있다
_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".txt"}   # .txt 는 OCR 없이 문장 정제만
_PARTIAL_RE = re.compile(r"_(?:정제중|OCR중)(?:_\d{8}_\d{6})?$|_미완$")   # 중간 저장분·앞부분만 읽은 결과
# process_full 위치 인자(lang, fmt, no_preprocess, no_fallback, …) 중 no_fallback 자리.
# 일괄 처리는 로컬 전용 — API 요금은 최소화(사용자 설계원칙 2026-10-18)
NO_FALLBACK_ARG = 3


@dataclass
class Job:
    path: Path
    args: tuple          # process_full 의 파일 다음 위치 인자들(lang … vision_first_str)
    kwargs: dict         # cleanup, front_matter
    state: str = "대기"   # 대기 | 처리 중 | 완료 | 실패 | 중단 | 취소
    detail: str = ""
    result: str = ""
    started: float = 0.0
    ended: float = 0.0


_lock = threading.Lock()
_jobs: list[Job] = []
_stop = threading.Event()
_workers: list[threading.Thread] = []
direct_running = threading.Event()   # ① 탭에서 직접 처리 중


# ── 대기열 ─────────────────────────────────────────────────────────────────────

def busy() -> bool:
    with _lock:
        return any(j.state in ("대기", "처리 중") for j in _jobs)


def submit(paths, args: tuple, kwargs: dict) -> int:
    """파일들을 대기열 끝에 넣고 작업 스레드를 띄운다. 이미 대기·처리 중인 파일은 건너뛴다."""
    from core import config
    args = tuple(args)
    if len(args) > NO_FALLBACK_ARG:   # 일괄은 로컬 전용 — 저신뢰 쪽도 API 로 다시 읽지 않고 '검토 필요'로 남긴다
        args = (*args[:NO_FALLBACK_ARG], True, *args[NO_FALLBACK_ARG + 1:])
    added = 0
    with _lock:
        active = {j.path for j in _jobs if j.state in ("대기", "처리 중")}
        for p in map(Path, paths):
            if p not in active:
                _jobs.append(Job(p, tuple(args), dict(kwargs)))
                added += 1
        _stop.clear()
        _workers[:] = [w for w in _workers if w.is_alive()]
        waiting = sum(j.state == "대기" for j in _jobs)
        while len(_workers) < min(config.get_batch_workers(), waiting):
            w = threading.Thread(target=_run, daemon=True, name=f"batch-runner-{len(_workers) + 1}")
            _workers.append(w)
            w.start()
    return added


def stop() -> None:
    """처리 중인 파일을 멈추고 남은 대기를 취소한다(이미 정제된 구간은 02_결과 *_정제중 에 남음)."""
    _stop.set()
    with _lock:
        for j in _jobs:
            if j.state == "대기":
                j.state = "취소"


def clear_finished() -> None:
    with _lock:
        _jobs[:] = [j for j in _jobs if j.state in ("대기", "처리 중")]


def _claim_job() -> Job | None:
    """다음 대기 파일을 '처리 중'으로 바꿔 가져온다 — 작업자 둘이 같은 파일을 잡지 않게 잠금 안에서."""
    with _lock:
        job = next((j for j in _jobs if j.state == "대기"), None)
        if job is not None:
            job.state, job.started = "처리 중", time.time()
        return job


def _run() -> None:
    from ocr_gradio import process_full     # 순환 import 방지 — 실행 시점에 가져온다
    while not _stop.is_set():
        while direct_running.is_set() and not _stop.is_set():
            time.sleep(2)                   # ① 탭 처리가 끝나기를 기다린다
        job = None if _stop.is_set() else _claim_job()
        if job is None:
            break
        last = None
        try:
            gen = process_full(SimpleNamespace(name=str(job.path)), *job.args,
                               progress=lambda *a, **k: None, **job.kwargs)
            for last in gen:
                if _stop.is_set():
                    gen.close()
                    job.state = "중단"
                    break
                job.detail = _last_line(last[1])
            else:
                job.result = str(last[3]) if last and last[3] else ""
                head = str(last[1]).splitlines()[0] if last and last[1] else ""
                if job.result and head.startswith("⚠ 부분 저장"):   # 쪽 그리기 실패로 앞부분만 저장
                    job.state, job.detail = "실패", "부분 저장 — " + head.split("  |  ")[1]
                    job.ended = time.time()
                    continue
                if not job.result:
                    job.detail = _last_line(last[1]) if last else "결과 없음"
                else:   # 정제 강등 안내만 남긴다 — '완료'만 보이면 정제가 안 된 것을 모른다(2026-10-13)
                    job.detail = next((ln.strip(" ⚠️") for ln in str(last[1]).splitlines()
                                       if "문장 정제 미적용" in ln), "")
                    _save_suspects(job)   # '완료' 전에 — 완료로 보이는데 목록이 아직 없는 틈을 없앤다
                    _note_review(job)
                job.state = "완료" if job.result else "실패"
        except Exception as e:   # 한 파일의 실패가 대기열 전체를 멈추지 않게
            job.state, job.detail = "실패", f"{type(e).__name__}: {e}"
        job.ended = time.time()


def _save_suspects(job: Job) -> None:
    """결과 txt 의 의심 목록(Kiwi 오타 후보)을 02_결과/_의심목록/ 에 남긴다 — 본문은 바꾸지 않는다.
    목록을 못 만들어도 결과는 이미 저장됐으므로 작업은 '완료' 그대로 둔다."""
    from core import typo_suspects
    path = Path(job.result)
    if path.suffix.lower() != ".txt" or not path.is_file():
        return
    try:
        _, n = typo_suspects.save(path)
    except Exception as e:
        job.detail = " · ".join(filter(None, [job.detail, f"의심 목록 실패: {type(e).__name__}"]))
        return
    if n:
        job.detail = " · ".join(filter(None, [job.detail, f"의심 {n}곳 — ④ 의심 목록에서 확인"]))


def _note_review(job: Job) -> None:
    """처리 중 남긴 검수기록(02_결과/_검수기록/)이 있으면 상세에 알린다."""
    from core import review_log
    if job.result and review_log.path_for(Path(job.result)).is_file():
        job.detail = " · ".join(filter(None, [job.detail, f"검수기록: {review_log.SUBDIR}/{Path(job.result).stem}.txt"]))


def _last_line(text) -> str:
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


# ── 입력 폴더 목록 ─────────────────────────────────────────────────────────────

_info_cache: dict[tuple, tuple[int | None, str]] = {}


def _file_kind(path: Path) -> tuple[int | None, str]:
    """(쪽수, 종류). 파일당 약 1초 — (경로, 수정시각, 크기)로 캐시한다."""
    st = path.stat()
    key = (str(path), st.st_mtime, st.st_size)
    if key in _info_cache:
        return _info_cache[key]
    from core import pdf_text_layer
    if path.suffix.lower() == ".txt":
        res = (None, "텍스트(정제만)")
    elif path.suffix.lower() != ".pdf":
        res = (1, "이미지")
    else:
        info = pdf_text_layer.analyze(path) if pdf_text_layer.is_available() else None
        if info is None:
            res = (None, "스캔")
        elif info.is_digital:
            bad = len(info.garbled_pages)
            res = (info.total_pages, f"디지털(깨진 쪽 {bad})" if bad else "디지털")
        elif info.garbled_pages:
            res = (info.total_pages, "글자 깨짐 → 전체 OCR")
        else:
            res = (info.total_pages, "스캔")
    _info_cache[key] = res
    return res


def _done_names(results_dir: Path) -> tuple[set[str], list[str]]:
    from core import storage_manager
    hist = {e.get("original_name", "") for e in storage_manager.load_history()}
    # 끊기거나 실패한 실행의 중간 저장분('…_정제중.txt'·'…_OCR중_시각.txt')은 결과가 아니다 —
    # 실패 6건이 빈 '_정제중' 파일 때문에 '완료'로 보였다(2026-10-17)
    outs = [p.name for p in results_dir.iterdir()
            if p.is_file() and not _PARTIAL_RE.search(p.stem)] if results_dir.exists() else []
    return hist, outs


def list_inputs(results_dir: Path, input_dir: Path | None = None) -> list[dict]:
    """입력 폴더 파일 목록 — 이름순. 처리 여부는 처리 기록 원본명 또는 02_결과 파일명으로 판정."""
    input_dir = input_dir or INPUT_DIR
    if not input_dir.is_dir():
        return []
    hist, outs = _done_names(results_dir)
    with _lock:
        queued = {j.path.name: j.state for j in _jobs if j.state in ("대기", "처리 중")}
    rows = []
    for p in sorted(input_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in _EXTS:
            continue
        pages, kind = _file_kind(p)
        done = p.name in hist or any(p.stem in o for o in outs)
        rows.append({"name": p.name, "path": p, "pages": pages, "kind": kind,
                     "status": queued.get(p.name) or ("완료" if done else "미처리")})
    return rows


def choice_label(r: dict) -> str:
    pages = f"{r['pages']}쪽" if r["pages"] else "쪽수 ?"
    mark = {"완료": "✅ 완료", "미처리": "미처리"}.get(r["status"], f"⏳ {r['status']}")
    return f"{r['name']}  ·  {pages}  ·  {r['kind']}  ·  {mark}"


# ── 화면용 상태 ────────────────────────────────────────────────────────────────

def _fmt_sec(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}시간 {m}분" if h else f"{m}분 {s}초"


def status_html() -> str:
    with _lock:
        jobs = list(_jobs)
    if not jobs:
        return ('<div class="note info">대기열이 비어 있습니다. 위 목록에서 파일을 고르고'
                ' <b>선택 처리</b>를 누르세요. 처리는 ① 탭의 처리 방식 설정을 따릅니다.</div>')
    done = sum(j.state in ("완료", "실패", "중단", "취소") for j in jobs)
    now = time.time()
    icon = {"대기": "·", "처리 중": "⏳", "완료": "✅", "실패": "❌", "중단": "■", "취소": "–"}
    rows = []
    for j in jobs:
        took = ""
        if j.started:
            took = _fmt_sec((j.ended or now) - j.started)
        note = j.result if j.state == "완료" else j.detail
        if j.state == "완료" and j.detail:
            note = f"⚠ {j.detail} · {j.result}"
        rows.append(f"<tr><td>{icon.get(j.state, '')} {j.state}</td><td>{html.escape(j.path.name)}</td>"
                    f"<td>{took}</td><td style='color:#6f6962'>{html.escape(note)[:140]}</td></tr>")
    from core import config
    waiting = any(j.state == "대기" for j in jobs)
    hold = " · ① 탭 처리가 끝나면 다음 파일을 시작합니다" if direct_running.is_set() and waiting else ""
    return (f"<div><b>대기열 {done}/{len(jobs)}</b> 끝남 · 동시 {config.get_batch_workers()}건"
            f" — 브라우저를 닫아도 처리는 계속됩니다.{hold}</div>"
            "<table style='width:100%;font-size:0.9rem'><tr><th>상태</th><th>파일</th><th>걸린 시간</th>"
            "<th>진행·결과</th></tr>" + "".join(rows) + "</table>")
