"""OCR 혼동 글자 후보 — 글자 모양이 비슷해 잘못 읽힌 한 글자(구메→구매·힘올→힘을)를 찾는다.

Kiwi 기본 오타 모드는 타자 실수용이라 OCR 오류를 거의 못 잡는다(『악당의 명언』 1권 의심 17개,
일휼·힘올·양형 누락). 그래서 OCR 이 헷갈리는 자모 짝을 TypoTransformer 로 따로 주고,
교체 전후 문맥의 Kiwi 점수 차가 MARGIN_MIN 이상인 것만 후보로 낸다. 본문은 바꾸지 않는다 —
typo_suspects 의심 목록에 올려 사람이 확인한 것만 user_typos 로 고친다
(intent 2026-11-06-로컬오탈자-후처리교정, 로컬 LLM 은 느리고 뜻을 바꿔 버림).
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
]
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


def _widen(line: str, a: int, b: int) -> tuple[int, int]:
    """1자면 같은 어절 안 옆 한글을 붙인다 — 사용자 사전은 2자 이상만 받는다(user_typos).
    기호·영문은 붙이지 않는다(`n혔`·`,스` 같은 잡음 후보가 나왔다)."""
    while b - a < 2 and a > 0 and _HANGUL.fullmatch(line[a - 1]):
        a -= 1
    while b - a < 2 and b < len(line) and _HANGUL.fullmatch(line[b]):
        b += 1
    return a, b


def candidates(lines: list[str]) -> list[tuple[str, str, float]]:
    """[(틀린 말, 제안, 점수 차)] — 줄마다 OCR 혼동 교정 후보. 여러 작업이 동시에 부르면 호출자가 잠근다."""
    kiwi = _get()
    out = []
    for line, toks in zip(lines, kiwi.tokenize(lines, typos=_typos(), typo_cost_threshold=_COST)):
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
            ca, cb = _widen(line, a, b)
            wrong = line[ca:cb]
            if len(wrong) >= 2 and _HANGUL.search(wrong):
                out.append((wrong, line[ca:a] + t.form + line[b:cb], round(margin, 1)))
    return out
