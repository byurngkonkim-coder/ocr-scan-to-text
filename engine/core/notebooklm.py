"""NotebookLM 소스 한도를 넘는 큰 책 나누기(intent 2026-10-24 NotebookLM용 MD 구조개선, 작업 5).

NotebookLM 은 소스 하나에 50만 단어까지 읽는다 ⟦E3: 업체 안내 기준, 공식 문서 미확인⟧ — 넘으면 뒷부분을 못 읽을 수
있다. 여유를 두고 45만 단어를 넘는 MD 만 장(#) 경계에서 나누고, 장 하나가 넘치면 절(##·###) → 문단(빈 줄) 경계로
내려간다(제목 없는 장절 책 — 성경). 조각마다 맨 앞에 '나눔' 줄과 문서 정보·전체 차례를 다시 붙인다. 본문 글자는 그대로.
"""
from __future__ import annotations

import re

MAX_WORDS = 450_000
_LEVELS = [re.compile(r'(?m)^(?=# )'), re.compile(r'(?m)^(?=## )'), re.compile(r'(?m)^(?=### )'),
           re.compile(r'(?<=\n\n)')]


def split_parts(md: str, max_words: int = MAX_WORDS) -> list[str]:
    """한도 안이면 []. 넘으면 조각 글 목록(각각 max_words 이하 — 문단 하나가 넘치는 경우만 예외)."""
    if len(md.split()) <= max_words:
        return []
    pre, body = _preamble(md)
    cap = max_words - len(pre.split()) - 10          # '나눔' 줄 몫
    blocks = _blocks(body, cap, 0)
    words = len(body.split())
    target = words / -(-words // cap)                # 같은 개수로 고르게(45만+15만 대신 30만+30만)
    parts: list[str] = []
    cur, n = "", 0
    for block in blocks:
        k = len(block.split())
        if cur and (n + k > cap or n >= target):
            parts.append(cur)
            cur, n = "", 0
        cur += block
        n += k
    if cur:
        parts.append(cur)
    total = len(parts)
    return [f"> 나눔: {total}개 중 {i}번째 파일\n\n{pre}{p}" for i, p in enumerate(parts, 1)]


def _preamble(md: str) -> tuple[str, str]:
    """(맨 앞 문서 정보·차례 목록, 본문)."""
    lines = md.splitlines(keepends=True)
    i = 0
    while i < len(lines) and (lines[i].startswith(">") or not lines[i].strip()):
        i += 1
    if i < len(lines) and lines[i].strip() == "# 차례":
        i += 1
        while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith("- ")):
            i += 1
    return "".join(lines[:i]), "".join(lines[i:])


def _blocks(text: str, cap: int, level: int) -> list[str]:
    if len(text.split()) <= cap or level >= len(_LEVELS):
        return [text]
    out: list[str] = []
    for piece in (p for p in _LEVELS[level].split(text) if p):
        out += _blocks(piece, cap, level + 1) if len(piece.split()) > cap else [piece]
    return out
