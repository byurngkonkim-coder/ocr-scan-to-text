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


_NOUN_TAGS = {"NNG", "NNP"}


def _word_joints(line: str) -> set[int]:
    """띄워도 되는 자리(공백 뺀 위치) — 형태소 경계 중 명사와 명사가 맞닿은 곳을 뺀 것.
    명사+명사는 책이 붙여 쓴 복합명사다. 실측(현직자를위한무역실무 100쪽, 2026-11-11): 넣은 띄어쓰기 3,140곳 중
    1,490곳(47.5%)이 명사+명사였고 원문은 붙여 쓴 말('개설은행'·'운송계약')이었다. 그중 '운송장'→'운송 장'·
    '송하인'→'송 하인'·'기한부'→'기한 부' 는 낱말을 깨뜨렸다. 형태소 경계가 아닌 자리(Kiwi.space 가 낱말 분석과
    다르게 가른 '특송'→'특 송'·'파손화물보상장'→'파손 화물 보상 장')도 띄우지 않는다."""
    nos, k = [], 0
    for ch in line:
        nos.append(k)
        if not ch.isspace():
            k += 1
    toks = _get().tokenize(line)
    return {nos[b.start] for a, b in zip(toks, toks[1:])
            if not (a.start + a.len == b.start and a.tag in _NOUN_TAGS and b.tag in _NOUN_TAGS)}


def _merge(orig: str, spaced: str, allowed: set[int] | None = None) -> tuple[str, int]:
    """orig 에 spaced 가 '덩어리 안에 넣은' 띄어쓰기만 더한다. allowed 가 있으면 그 자리만. (줄, 넣은 수)"""
    if re.sub(r"\s", "", orig) != re.sub(r"\s", "", spaced):
        return orig, 0
    spans = _glued_spans(orig)
    added = _space_pos(spaced) - _space_pos(orig)
    if allowed is not None:
        added &= allowed
    keep = {p for p in added if any(a < p < b for a, b in spans)}
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


_JOIN_MARGIN = 3.0


def is_split_word(before: str, after: str) -> bool:
    """줄·쪽 경계에서 끊긴 두 어절이 한 낱말인가 — 붙인 꼴의 Kiwi 점수가 띄운 꼴보다 확실히 높으면.
    실측(마이클포터_경쟁전략, 2026-11-06): '훌륭한전'+'략일' +19, '도구로'+'서의' +13, '불러오'+'는가' +5,
    바른 경계('기업의'+'노력은'·'시장'+'점유율을')는 0."""
    if not (before and after and is_available()):
        return False
    with _lock:
        k = _get()
        joined = k.analyze(before + after)[0][1]
        spaced = k.analyze(before + " " + after)[0][1]
    return joined - spaced >= _JOIN_MARGIN


def fix_spacing(text: str) -> tuple[str, int]:
    """붙은 한글 덩어리가 있는 줄만 Kiwi 로 띄운다. (본문, 넣은 띄어쓰기 수)"""
    if not text or not is_available():
        return text, 0
    text, n = _split_short_josa_glue(text)
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if _glued_spans(l)]
    if not idx:
        return "\n".join(lines), n
    with _lock:
        spaced = list(_get().space([lines[i] for i in idx]))
        joints = [_word_joints(lines[i]) for i in idx]
    for i, s, allowed in zip(idx, spaced, joints):
        lines[i], c = _merge(lines[i], s, allowed)
        n += c
    return "\n".join(lines), n


# 4~5자 덩어리는 위(5자↑) 띄어쓰기가 보지 않는다 — '이 책을읽기 시작하는'(2천년의강의). 덩어리 안 조사 자리에서
# 띄운 꼴의 Kiwi 점수가 붙인 꼴보다 낮지 않을 때만 띄운다. 실측: 갈라 써야 할 '책을읽기'·'밥을먹고'·'눈을감고' 0.0,
# 한 낱말인 '사이좋게' -1.3·'이를테면' -2.1·'마을에서' -5.0·'사람이다' -3.3
_SHORT_GLUE_RE = re.compile(r'(?<![가-힣])[가-힣]{4,5}(?![가-힣])')
_INNER_JOSA = set("을를은는이가에의")
_SHORT_GLUE_MARGIN = -0.05
_VERB_HEAD = set("했였하해되됐돼라로이오입다려며면")   # '들어가려는' — 뒷말이 어미면 한 낱말


def _split_short_josa_glue(text: str) -> tuple[str, int]:
    cands = {}
    for m in _SHORT_GLUE_RE.finditer(text):
        w = m.group()
        if w in cands:
            continue
        cands[w] = None
        for i in range(2, len(w) - 1):          # 앞뒤 모두 2자 이상
            # 뒷말이 어미·서술격으로 시작하면 앞말은 '가·의·이'로 끝나는 명사다('평가했다'·'정의로운'·'심장이라면')
            if w[i - 1] in _INNER_JOSA and w[i] not in _VERB_HEAD:
                with _lock:
                    k = _get()
                    toks, joined = k.analyze(w)[0]
                    spaced = k.analyze(w[:i] + " " + w[i:])[0][1]
                # 그 자리가 붙인 꼴 분석에서도 조사가 끝나는 형태소 경계여야 — '최은경은'(최은경/NNP)은 아니다
                josa_end = any(t.start + t.len == i and t.tag.startswith("J") for t in toks)
                if josa_end and spaced - joined >= _SHORT_GLUE_MARGIN:
                    cands[w] = w[:i] + " " + w[i:]
                break
    fix = {w: s for w, s in cands.items() if s}
    if not fix:
        return text, 0
    out, n = _SHORT_GLUE_RE.subn(lambda m: fix.get(m.group(), m.group()), text)
    return out, sum(1 for m in _SHORT_GLUE_RE.finditer(text) if m.group() in fix)
