"""붙어 쓴 한글 띄어쓰기 교정 R-17 — Kiwi(kiwipiepy) 형태소 분석 기반, 2026-10-17.

실측(보관 원문 24권 + 대한민국CEO명언집): korean_corrector 는 정해진 꼴('할것이다')만 고쳐
OCR 이 붙여 읽은 일반 낱말('인류에게하나밖에없는', '정의가없는 용기는나약하기짝이')을 못 풀었다.
Kiwi.space() 를 그대로 쓰면 붙은 덩어리를 98% 풀지만, 이미 있는 띄어쓰기를 지우고
('배워야 한다'→'배워야한다' 1,720곳) 짧은 고유명사를 쪼갰다('최은경은'→'최은 경은').

안전장치:
  - 넣기만 한다 — 원문의 띄어쓰기는 지우지 않는다.
  - 한글이 5자 이상 붙은 덩어리 안에서만 띄운다(바른 책 6권에서 새로 넣은 곳 1.9%, 넣은 것 정확도 89.6%).
  - 영문에 바로 붙은 한글 덩어리('엔드루카네기Andrew')는 OCR 이 잘못 읽은 저자 이름이라 건드리지 않는다.
  - 글자는 바꾸지 않는다(띄어쓰기 외 글자가 달라지면 그 줄은 원문 그대로).
kiwipiepy 가 없으면 아무것도 하지 않는다.
"""
from __future__ import annotations

import re
import threading

_MIN_GLUED = 5
_GLUED_RE = re.compile(r"(?<![A-Za-z])[가-힣]{%d,}(?![A-Za-z])" % _MIN_GLUED)
_lock = threading.Lock()        # 일괄 처리의 여러 작업이 Kiwi 하나를 같이 쓴다
_kiwi = None


def _get():
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi()
    return _kiwi


def is_available() -> bool:
    try:
        import kiwipiepy  # noqa: F401
        return True
    except ImportError:
        return False


def _space_pos(s: str) -> set[int]:
    """띄어쓰기 자리 — 공백 앞 글자의 (공백을 뺀) 위치."""
    pos, k = set(), 0
    for ch in s:
        if ch.isspace():
            pos.add(k)
        else:
            k += 1
    return pos


def _glued_spans(line: str) -> list[tuple[int, int]]:
    """한글 5자↑ 붙은 덩어리의 (공백 뺀 위치 기준) 범위. 영문에 붙은 덩어리는 뺀다."""
    spans, k = [], 0
    for tok in line.split():
        spans += [(k + m.start(), k + m.end()) for m in _GLUED_RE.finditer(tok)]
        k += len(tok)
    return spans


def _merge(orig: str, spaced: str) -> tuple[str, int]:
    """orig 에 spaced 가 '덩어리 안에 넣은' 띄어쓰기만 더한다. (줄, 넣은 수)"""
    if re.sub(r"\s", "", orig) != re.sub(r"\s", "", spaced):
        return orig, 0
    spans = _glued_spans(orig)
    keep = {p for p in _space_pos(spaced) - _space_pos(orig) if any(a < p < b for a, b in spans)}
    if not keep:
        return orig, 0
    out, k = [], 0
    for ch in orig:
        if not ch.isspace():
            if k in keep:
                out.append(" ")
            k += 1
        out.append(ch)
    return "".join(out), len(keep)


def fix_spacing(text: str) -> tuple[str, int]:
    """붙은 한글 덩어리가 있는 줄만 Kiwi 로 띄운다. (본문, 넣은 띄어쓰기 수)"""
    if not text or not is_available():
        return text, 0
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if _glued_spans(l)]
    if not idx:
        return text, 0
    with _lock:
        spaced = list(_get().space([lines[i] for i in idx]))
    n = 0
    for i, s in zip(idx, spaced):
        lines[i], c = _merge(lines[i], s)
        n += c
    return "\n".join(lines), n
