"""기존 결과 재변환 — NotebookLM용 MD 다시 만들기(intent 2026-10-24 NotebookLM용 MD 구조개선, 작업 6).

02_결과 의 정제 결과에 새 구조(장·절 제목 다시 찾기·제목 띄어쓰기·차례 없는 책 장 추정·쪽 표지 [p.N]·문서 정보
머리말·큰 책 나누기)를 다시 입힌다. 본문은 기존 정제 결과 그대로 — OCR·정제·API 를 부르지 않는다.
원문 사본(교정 후 글)으로 정제를 다시 하면 차례 인식이 처음보다 약했다(NextSociety 차례 49→46, NLP·MBA 차례 못 찾음,
2026-10-24 실측) — 그래서 원문 사본은 쪽 대조에만 쓴다. 사본이 없으면 쪽 표지만 빠진다.
순서: _백업/재변환_{시각}/ 에 원본 보관 → 임시 파일에 쓴 뒤 교체(reclean 과 같은 방식).
"""
from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path

from core import storage_manager, text_cleaner, toc_headings

BACKUP_ROOT = Path(__file__).resolve().parent.parent.parent / "_백업"
_PARTIAL_RE = re.compile(r"_(?:정제중|OCR중)(?:_\d{8}_\d{6})?$")
_STAMP_RE = re.compile(r'^\d{8}_\d{6}_')
_MARK_LINE_RE = re.compile(r'^\[p\.\d+\]\n', re.M)
_SEP_TXT = "─" * 60


def list_targets(results_dir: Path) -> list[dict]:
    """다시 만들 수 있는 결과 — 책마다 하나(.md 가 있으면 .md, 없으면 .txt). 중간 저장분 제외.
    {path, raw: 원문 사본 있음, skip: 건너뛸 사유('' 이면 대상)}"""
    if not results_dir.is_dir():
        return []
    by_stem: dict[str, Path] = {}
    for p in sorted(results_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in (".md", ".txt") and not _PARTIAL_RE.search(p.stem):
            if p.stem not in by_stem or p.suffix.lower() == ".md":
                by_stem[p.stem] = p
    rows = []
    for p in by_stem.values():
        rows.append({"path": p, "raw": _raw_copy(p) is not None,
                     "skip": "정제 안 한 결과(쪽 단위 서식)" if _unrefined(p.read_text(encoding="utf-8")) else ""})
    return rows


def parse_result(text: str) -> tuple[list[dict], str]:
    """결과 글 → (차례 항목, 본문). 앞서 다시 만든 머리말·쪽 표지는 뗀다(여러 번 돌려도 같게)."""
    lines = _MARK_LINE_RE.sub("", text).split("\n")
    i = 0
    if lines and (lines[0].startswith("> **문서 정보**") or lines[0].startswith("문서 정보 — ")):
        while i < len(lines) and lines[i].strip():
            i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    entries: list[dict] = []
    if i < len(lines) and lines[i].strip() in ("# 차례", "차례"):
        md = lines[i].startswith("#")
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        while i < len(lines) and lines[i].strip() and (not md or lines[i].lstrip().startswith("- ")):
            l = lines[i]
            title = l.strip()[2:] if md else l.strip()
            entries.append({"level": (len(l) - len(l.lstrip())) // 2 + 1, "title": title, "page": None})
            i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
    return entries, "\n".join(lines[i:])


def read_raw_pages(path: Path) -> tuple[list[tuple[int, str]], str, int | None]:
    """원문 사본(writer 서식 — MD '## 페이지 N' / TXT '[페이지 N]') → ([(쪽번호, 쪽 글)], 원본 파일명, PDF 쪽 수)."""
    raw = Path(path).read_text(encoding="utf-8")
    md = bool(re.search(r'^## 페이지 \d+\s*$', raw, re.M))
    parts = re.split(r'^## 페이지 (\d+)\s*$' if md else r'^\[페이지 (\d+)\].*$', raw, flags=re.M)
    pages = []
    for k in range(1, len(parts) - 1, 2):
        body = parts[k + 1]
        if md:
            body = re.split(r'\n-{3,}\n|\n> \*\*비텍스트', body)[0]
            body = "\n".join(l for l in body.split("\n") if not l.startswith(">"))
        else:
            body = body.split("── 비텍스트 영역 ──")[0]
            body = "\n".join(l for l in body.split("\n") if l.strip() != _SEP_TXT)
        pages.append((int(parts[k]), body.strip()))
    name = m.group(1).strip() if md and (m := re.match(r'# (.+)', raw)) else ""
    m = re.search(r'총 페이지: (\d+)', raw) or re.search(r'전체 (\d+)페이지 중', raw)
    total = int(m.group(1)) if m else (max(n for n, _ in pages) if pages else None)
    return pages, name, total


def rebuild(text: str, toc: list[dict] | None = None) -> tuple[list[dict], str, str]:
    """결과 글 → (차례 항목, 새 구조 본문, 장·절 보고). toc 를 주면(책정보 파일 — 원본 PDF·인터넷에서 모은 차례)
    결과에 든 차례 대신 쓰고, 제목 글자는 그 차례 그대로 둔다(respace 안 함)."""
    entries, body = parse_result(text)
    body, _ = text_cleaner.strip_spaced_running_header(body)
    if toc is not None:
        entries = toc
        # 옛 결과의 제목 줄은 OCR 차례 글자('# 감사의삶')라 확인된 차례와 다르다 — 보통 줄로 되돌려 다시 맞춘다
        # (rematch 는 이미 제목인 줄을 건너뛴다. 1001가지기독교명언 40곳)
        body = re.sub(r'(?m)^#{1,6} ', '', body)
    stats: dict = {"inferred": 0}
    if entries:
        # 흩어진 제목부터 — rematch 가 제목 앞부분만 같은 줄('…변하듯 04')을 먼저 잡아 차례 제목을 그 줄로 바꿨다
        body, _ = text_cleaner.drop_toc_remnant(body, entries)
        body, _ = text_cleaner.drop_toc_tables(body, entries)
        body, _ = text_cleaner.drop_toc_dense(body, entries)
        # 글자가 같은 따로 선 줄부터(경계를 세우고) → 흩어진 제목은 그 경계 안에서 → 다시 맞춤
        if toc is not None:
            body = text_cleaner.rematch_headings(body, entries, stats, respace=False, keep_titles=True,
                                                 exact=True)
        body, _ = text_cleaner.place_split_titles(body, entries)
        body = text_cleaner.rematch_headings(body, entries, stats, respace=toc is None,
                                             keep_titles=toc is not None)
        # 장 여는 쪽 절 목록에 선 제목을 진짜 자리로 옮긴 뒤, 그 사이에 생긴 자리에서 못 찾은 항목을 다시 맞춘다
        body, moved = text_cleaner.relocate_list_headings(body, entries)
        if moved:
            body = text_cleaner.rematch_headings(body, entries, stats, respace=False, keep_titles=toc is not None)
        # 한 번 더 — rematch 가 세운 제목들이 앞뒤 경계가 되어 문단 끝에 붙은 짧은 제목을 좁은 범위에서 찾는다
        body, more = text_cleaner.place_split_titles(body, entries, tail=True)
        if more or moved:
            heads = set(body.split("\n"))
            stats["missing"] = [e["title"] for e in entries if toc_headings._heading(e) not in heads]
            stats["found"] = len(entries) - len(stats["missing"]) - stats["inferred"]
        note = toc_headings.report(entries, stats)
    else:
        body, entries = text_cleaner.infer_chapters(body)
        if entries:
            body = text_cleaner.rematch_headings(body, entries, stats, respace=True)
        note = (f"차례를 찾지 못해 본문에서 장 제목 {len(entries)}개를 추정(제N장·CHAPTER N 꼴 줄)" if entries
                else "차례를 찾지 못해 장·절 정비 없이 정제")
    body, _ = text_cleaner.attach_item_numbers(body)   # 제목을 다 세운 뒤 — 문단 끝 제목 찾기가 번호 문단을 본다
    body, _ = text_cleaner.space_list_numbers(body)         # '1오늘부터' → '1 오늘부터'(CreatYourOwnFuture)
    body, _ = text_cleaner.drop_cover_ad(body)
    body, _ = text_cleaner.drop_heading_echo(body)          # 장 표지 큰 글자 '서막'(4차산업혁명시대)
    body, _ = text_cleaner.drop_subtitle_echo(body)         # '제목 _ 부제 원제' 아래 다시 읽은 부제·원제(HowToWin…)
    body, _ = text_cleaner.repair_open_brackets(body)
    body, _ = text_cleaner.join_josa_paragraphs(body)      # '…‘큰생각'' ⏎⏎ '을 만나게' (2천년의강의)
    body, _ = text_cleaner.space_glued_marks(body)          # '당신은2000년'·'진정한‘큰생각''
    # 붙어 쓴 한글 띄어쓰기(R-17) — 옛 결과는 이 단계 전에 만들어졌다('책을읽기'). 제목 줄은 차례 글자 그대로
    body = text_cleaner.keep_headings(lambda t: text_cleaner.spacing.fix_spacing(t)[0], body)
    body = (toc_headings.space_headings(body) if entries else body.strip()) + "\n"
    return entries, body, note


def remake(path: Path, backup_dir: Path | None = None) -> dict:
    """결과 한 권(.md·.txt 짝)을 새 구조로 다시 만든다. → {name, path, raw, marks, toc, status, logs}"""
    import ocr_gradio                          # 순환 import 방지 — 실행 시점에(batch_runner 와 같은 방식)
    from core.types import OutputFormat
    path = Path(path)
    md_path, txt_path = path.with_suffix(".md"), path.with_suffix(".txt")
    src = md_path if md_path.exists() else txt_path
    text = src.read_text(encoding="utf-8")
    row = {"name": path.stem, "path": path, "raw": False, "marks": 0, "toc": "", "status": "", "logs": []}
    if _unrefined(text):
        row["status"] = "건너뜀 — 정제 안 한 결과(쪽 단위 서식)"
        return row
    entries, body, note = rebuild(text)
    raw = _raw_copy(path)
    pages, name, total = read_raw_pages(raw) if raw else (None, "", None)
    info = {"name": name or _STAMP_RE.sub("", path.stem), "pages": total}
    txt, md = ocr_gradio._structured_texts(body, {"toc_entries": entries, "toc_note": note}, OutputFormat.TXT,
                                           row["logs"], pages, info)
    backup_dir = backup_dir or BACKUP_ROOT / f"재변환_{time.strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for p in (md_path, txt_path):
        if p.exists():
            shutil.copy2(p, backup_dir / p.name)
    _replace(md_path, md)
    if txt_path.exists():
        _replace(txt_path, txt)
    ocr_gradio._save_notebooklm_parts(md_path, md, row["logs"])
    row.update(raw=raw is not None, marks=len(re.findall(r'^\[p\.\d+\]$', md, re.M)), toc=note, status="완료")
    return row


def _raw_copy(path: Path) -> Path | None:
    for suf in (".md", ".txt"):
        p = storage_manager.raw_copy_path(Path(path).with_suffix(suf))
        if p.exists():
            return p
    return None


def _unrefined(text: str) -> bool:
    """정제를 끈 쪽 단위 서식(writer — '## 페이지 N' / '[페이지 N]')인가."""
    head = text[:5000]
    return bool(re.search(r'^## 페이지 \d+\s*$', head, re.M) or re.search(r'^\[페이지 \d+\]', head, re.M))


def _replace(dst: Path, text: str) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, dst)
