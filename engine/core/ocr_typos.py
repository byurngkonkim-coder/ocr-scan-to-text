"""OCR 혼동 글자 후보 — 글자 모양이 비슷해 잘못 읽힌 한 글자(구메→구매·힘올→힘을)를 찾는다.

Kiwi 기본 오타 모드는 타자 실수용이라 OCR 오류를 거의 못 잡는다(『악당의 명언』 1권 의심 17개,
일휼·힘올·양형 누락). 그래서 OCR 이 헷갈리는 자모 짝을 TypoTransformer 로 따로 주고,
교체 전후 문맥의 Kiwi 점수 차가 MARGIN_MIN 이상인 것만 후보로 낸다. 본문은 바꾸지 않는다 —
typo_suspects 의심 목록에 올려 사람이 확인한 것만 user_typos 로 고친다
(intent 2026-11-06-로컬오탈자-후처리교정, 로컬 LLM 은 느리고 뜻을 바꿔 버림).
예외: 조사 계열(AUTO_FIX)만 auto_fix() 로 본문에 바로 반영한다.
"""
from __future__ import annotations

import re

MARGIN_MIN = 8.0      # 교체문 점수 - 원문 점수(문맥 창 기준). 표본 정답 8.5~33, 1차값 — tools/ocr_typo_eval.py 로 조정
_COST = 2.5           # 오타 교정 비용 상한(자모 1개 교체 = 1)
_WIN = 20             # 점수 비교 문맥 창(앞뒤 글자 수)
_HANGUL = re.compile("[가-힣]+")

# OCR 글자 모양 혼동 — 실측 사례: 마옴·힘올(ㅗ/ㅡ), 하눌(ㅜ/ㅡ), 구메(ㅔ/ㅐ), 한게(ㅔ/ㅖ), 직게(ㅣ/ㅓ),
# 제안플(ㅍ/ㅇ), 일음(ㅁ/ㄹ), 경쟀(ㅆ/ㅇ), 일휼(ㅎ/ㅇ)
_PAIRS = [
    ["ㅗ", "ㅜ", "ㅡ", "ㅠ"],
    ["ㅐ", "ㅔ"],
    ["ㅔ", "ㅖ"],
    ["ㅣ", "ㅓ"],
    ["ㅎ", "ㅇ"],
    ["ㅍ", "ㅇ"],
    ["ㅁ", "ㄹ"],
    ["ㅆ", "ㅇ"],
    ["ㅏ", "ㅣ"],      # 텍스트 레이어 PDF 3권(2026-11-06): 분노히지·살이가는·자구촌·디르게·수 았는
]
# 자동 반영하는 (틀린 글자, 바른 글자) — 조사 '을·은·들·를·음' 이 ㅗ·ㅠ 로 읽힌 것(2단계, 사용자 결정 2026-11-06).
# 텍스트 레이어 PDF 3권 후보 231행·399곳 전수 확인에서 틀린 제안 0건(사람 판정, 추정). 나머지 짝은 목록만
AUTO_FIX = {("올", "을"), ("율", "을"), ("온", "은"), ("옴", "음"), ("듈", "들"), ("률", "를")}
_MAX_WRONG = 12       # 틀린 말을 넓힐 최대 글자 수 — 이 안에서 오류 자리만 가리키게 못 만들면 버린다
_transformer = None
# Kiwi 를 따로 쓴다 — 한 인스턴스에 기본 오타 모드(typo_suspects)와 이 TypoTransformer 를 번갈아 주면
# kiwipiepy 0.24 에서 접근 위반으로 프로세스가 죽었다(실측 12회 중 4회, spacing 의 같은 문제와 같은 꼴)
_kiwi = None


def _get():
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi()
    return _kiwi


def _typos():
    global _transformer
    if _transformer is None:
        from kiwipiepy import TypoDefinition, TypoTransformer
        _transformer = TypoTransformer([TypoDefinition(p, p, 1.0) for p in _PAIRS])
    return _transformer


def _score(kiwi, s: str) -> float:
    return kiwi.analyze(s, top_n=1)[0][1]


def _widen(text: str, pos: int, flagged: set[int]) -> tuple[int, int] | None:
    """오류 글자(pos) 둘레를 넓혀 틀린 말 구간을 정한다. 사용자 사전은 모든 책에 글자 그대로 치환하므로
    (user_typos), 틀린 말이 책 안에서 오류 자리(flagged)에만 나올 때까지 넓힌다 — 2자로만 넓히면
    '리고→라고'(그리고)·'았을→있을'(받았을) 처럼 바른 낱말을 망치는 후보가 나왔다(2026-11-06 3권).
    같은 어절 한글을 먼저 붙이고, 모자라면 옆 어절까지 넘는다. 기호·영문에서 시작·끝나지 않는다."""
    def run_ok(i):
        return 0 <= i < len(text) and _HANGUL.fullmatch(text[i])

    def unique(a, b):
        frag, off = text[a:b], pos - a
        return all(m.start() + off in flagged for m in re.finditer(re.escape(frag), text))

    a, b = pos, pos + 1
    while b - a < _MAX_WRONG:
        if b - a >= 2 and unique(a, b):
            return a, b
        if run_ok(a - 1):
            a -= 1
        elif run_ok(b):
            b += 1
        else:   # 어절 끝 — 앞 어절을 띄어쓰기째 붙인다(없으면 뒤 어절)
            if a > 1 and text[a - 1] == " " and run_ok(a - 2):
                a -= 2
            elif b + 1 < len(text) and text[b] == " " and run_ok(b + 1):
                b += 2
            else:
                return None
    return None


def candidates(lines: list[str]) -> list[tuple[str, str, float]]:
    """[(틀린 말, 제안, 점수 차)] — 줄마다 OCR 혼동 교정 후보. 여러 작업이 동시에 부르면 호출자가 잠근다."""
    hits = _hits(lines)
    text = "\n".join(lines)
    flagged = {p for p, _, _ in hits}
    out = []
    for p, ch, margin in hits:
        span = _widen(text, p, flagged)
        if span:
            a, b = span
            out.append((text[a:b], text[a:p] + ch + text[p + 1:b], margin))
    return out


def auto_fix(text: str) -> tuple[str, int]:
    """(교정본, 바꾼 곳 수) — AUTO_FIX 짝만 그 자리에서 한 글자씩 바꾼다. 한글 한 글자를 한글 한 글자로
    바꾸므로 글자 수·줄 수는 그대로다. 여러 작업이 동시에 부르면 호출자가 잠근다(typo_suspects._lock)."""
    if not text or not _HANGUL.search(text):
        return text, 0
    chars = list(text)
    n = 0
    for p, ch, _ in _hits(text.split("\n")):
        if (text[p], ch) in AUTO_FIX:
            chars[p] = ch
            n += 1
    return "".join(chars), n


def _hits(lines: list[str]) -> list[tuple[int, str, float]]:
    """[(오류 글자 위치(줄들을 줄바꿈으로 이은 글 기준), 바른 글자, 점수 차)]."""
    kiwi = _get()
    hits, base = [], 0
    for line, toks in zip(lines, kiwi.tokenize(lines, typos=_typos(), typo_cost_threshold=_COST)):
        hits += _line_hits(kiwi, line, toks, base)
        base += len(line) + 1
    return hits


def _line_hits(kiwi, line: str, toks, base: int) -> list[tuple[int, str, float]]:
    hits = []
    for t in toks:
        if not getattr(t, "typo_cost", 0):
            continue
        a, b = t.start, t.start + t.len
        span = line[a:b]
        # OCR 오인식은 한 글자 바뀜 — 용언 원형처럼 두 글자 이상 달라지면(혼들린→흔들리) 버린다
        if (len(t.form) != len(span) or not _HANGUL.fullmatch(t.form) or not _HANGUL.fullmatch(span)
                or sum(x != y for x, y in zip(t.form, span)) != 1):
            continue
        wa, wb = max(0, a - _WIN), min(len(line), b + _WIN)
        margin = _score(kiwi, line[wa:a] + t.form + line[b:wb]) - _score(kiwi, line[wa:wb])
        if margin < MARGIN_MIN:
            continue
        k = next(i for i, (x, y) in enumerate(zip(span, t.form)) if x != y)
        hits.append((base + a + k, t.form[k], round(margin, 1)))
    return hits
