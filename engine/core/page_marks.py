"""쪽 경계 표지 [p.N] — 정제본에 원본 PDF 쪽번호를 되살린다(intent 2026-10-24 NotebookLM용 MD 구조개선).

정제(줄 잇기·문단 합치기·잡음 제거)는 쪽 경계를 이어 붙여 쪽 정보를 잃는다. 정제 과정에 표지를 실어 보내는 대신,
끝난 정제본을 쪽별 원문과 글자(한글·영숫자)로 대조해 쪽이 시작하는 자리를 찾는다 — 정제 엔진(규칙·Ollama·Gemini)과
무관하다. 표지는 문장을 끊지 않게 그 문장이 끝난 뒤 줄에 두고, 장 제목·Markdown 표 안에는 넣지 않는다.
정제가 지운 쪽(표지·광고)은 찾지 못하므로 건너뛴다. 실측: 5권 쪽의 96~98% 위치 확인(intent §10).
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

_KEEP_RE = re.compile(r'[0-9A-Za-z가-힣]')
_ANCHOR = 24          # 쪽 앞에서 잡는 글자 수
_ANCHOR_MIN = 12      # 이보다 글자가 적은 쪽(장 표제지 한 줄 등)은 대조하지 않는다
_ANCHOR_STEP = 20     # 첫 줄이 머리글·잡음으로 지워졌을 수 있어 이만큼씩 뒤로 옮겨 다시 잡는다
_ANCHOR_SPAN = 200
_WINDOW = 8000        # 앞 쪽 위치 뒤 이만큼(대조 글자 수) 안에서 찾는다 — 먼 곳의 같은 문구에 붙지 않게
_FUZZY_WINDOW = 20000
_SENT_END_RE = re.compile(r'[.!?…。！？]["”’\'」』)\]》]*')
_HEADING_RE = re.compile(r'^#{1,6} \S')


def mark_pages(text: str, pages: list[tuple[int, str]]) -> tuple[str, dict]:
    """pages: [(PDF 쪽번호, 그 쪽 원문)] 쪽 순서대로. → (표지 넣은 글, {found, missing})."""
    idx = [i for i, c in enumerate(text) if _KEEP_RE.match(c)]
    cn = "".join(text[i] for i in idx)
    pos, skipped, found, missing = 0, 0, 0, []
    marks: list[tuple[int, int]] = []          # (넣을 자리, 쪽번호) — 자리 오름차순
    for no, page in pages:
        t = "".join(_KEEP_RE.findall(page))
        if len(t) < _ANCHOR_MIN:
            continue
        hit = _find(cn, t, pos, _WINDOW + skipped)
        if hit is None:
            missing.append(no)
            skipped += len(t)       # 못 찾은 쪽만큼 다음 쪽을 더 멀리까지 찾는다
            continue
        found += 1
        skipped = 0
        pos = hit + 1
        at = _insert_at(text, idx[hit])
        if at is None:
            continue
        if marks and at <= marks[-1][0]:     # 같은 문장 뒤로 모이면 뒤 쪽이 이긴다
            marks[-1] = (marks[-1][0], no)
        else:
            marks.append((at, no))
    out = text
    for at, no in reversed(marks):
        if at < len(out) and out[at] != "\n" and (at == 0 or out[at - 1] == "\n"):
            out = out[:at] + f"[p.{no}]\n" + out[at:]                 # 줄 머리
        else:                                                            # 문장 뒤 — 띄어쓰기를 줄바꿈으로
            rest = out[at:].lstrip(" \t")
            out = out[:at] + f"\n[p.{no}]" + ("\n" + rest if rest and rest[0] != "\n" else rest)
    return out, {"found": found, "missing": missing}


def _find(cn: str, t: str, pos: int, window: int) -> int | None:
    """정제본 대조 글자열 cn 에서 쪽 글자 t 가 시작하는 자리."""
    k = min(_ANCHOR, len(t))
    for off in range(0, min(len(t) - k, _ANCHOR_SPAN) + 1, _ANCHOR_STEP):
        i = cn.find(t[off:off + k], pos, pos + window + k)
        if i >= 0:
            return max(pos, i - off)
    # 띄어쓰기·오타 교정으로 글자가 조금 바뀐 쪽 — 가장 길게 겹치는 곳
    win = cn[pos:pos + _FUZZY_WINDOW]
    m = SequenceMatcher(None, win, t[:k], autojunk=False).find_longest_match(0, len(win), 0, k)
    return pos + max(0, m.a - m.b) if m.size >= _ANCHOR_MIN else None


def _insert_at(text: str, o: int) -> int | None:
    """쪽이 o 에서 시작할 때 표지를 넣을 자리. 줄 머리면 그 줄 앞(제목·표·빈 줄은 건너 다음 본문 줄),
    줄 중간이면 그 문장이 끝난 자리(없으면 줄 끝). 넣을 본문이 뒤에 없으면 None."""
    ls = text.rfind("\n", 0, o) + 1
    if not text[ls:o].strip("# \t") or text[ls:].startswith("|"):
        return _next_body_line(text, ls)
    le = text.find("\n", o)
    le = len(text) if le < 0 else le
    if _HEADING_RE.match(text[ls:le]):     # 제목 줄은 끊지 않는다 — 제목 속 '?' 는 문장 끝이 아니다(StrategicalThinking)
        return _next_body_line(text, le + 1)
    m = _SENT_END_RE.search(text, o, le)
    while m and m.end() < le and not text[m.end()].isspace():    # '3.5'·'U.S.' 같은 문장 속 마침표
        m = _SENT_END_RE.search(text, m.end(), le)
    end = m.end() if m else le
    if not text[end:].strip():
        return None
    return end if end < le else _next_body_line(text, le + 1)


def _next_body_line(text: str, ls: int) -> int | None:
    while ls < len(text):
        le = text.find("\n", ls)
        le = len(text) if le < 0 else le
        line = text[ls:le]
        if line.strip() and not _HEADING_RE.match(line) and not line.startswith("|"):
            return ls
        ls = le + 1
    return None
