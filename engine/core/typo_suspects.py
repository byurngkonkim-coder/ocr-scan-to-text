"""의심 목록 — Kiwi 오타 모드가 고치자고 하는 곳을 본문은 바꾸지 않고 목록으로만 낸다.

실측(2026-10-17, 『지혜의 기술』 문장 2,007개): 오타 모드가 바꾼 42곳 중 맞게 고친 것은 약 10곳
(지해→지혜·걷모습→겉모습), 나머지는 틀렸다(빛을→빚을·흘륨한→흙륨한). Kiwi 오타 모델은 타자
실수용이라 글자 모양이 비슷한 OCR 오류와 잘 맞지 않는다 → 자동 적용 금지, 사람이 확인해
user_typos 에 등록한 것만 다음부터 고친다(intent 2026-10-17-로컬-맞춤법-자동교정).

목록은 02_결과/_의심목록/{책이름}.tsv — 하위 폴더라 ④ 탭의 '처리 완료' 판정(파일만 봄)을 흐리지 않는다.
"""
from __future__ import annotations

import re
from pathlib import Path

from core import spacing

SUBDIR = "_의심목록"
_MODE = "basic_with_continual_and_lengthening"
_SURFACE_TAGS = ("N", "MAG", "MAJ", "MM", "XR")   # 형태 = 겉글자인 품사 — 용언·어미는 원형이라 바꿔 끼울 수 없다
_CTX = 15
_FIELDS = ("wrong", "suggestion", "count", "context")
_HEAD = "틀린 말\t제안\t횟수\t문맥"


def _candidates(line: str, tokens) -> list[tuple[str, str]]:
    out = []
    for t in tokens:
        if not getattr(t, "typo_cost", 0):
            continue
        a, b = t.start, t.start + t.len
        span = line[a:b]
        # 조사·어미·용언(이→의·기울→귀울)과 글자 수가 달라지는 것(고칭→공)은 잡음이었다(1권 실측) —
        # OCR 오류는 글자 바뀜이라 같은 길이의 겉글자 품사만 남긴다
        if (t.form == span or len(t.form) != len(span) or not t.tag.startswith(_SURFACE_TAGS)
                or any(c.isspace() for c in t.form)):
            continue
        # 1자면 같은 어절 안에서 옆 글자를 붙인다 — 사전 등록은 2자 이상만 받는다
        while b - a < 2 and b < len(line) and not line[b].isspace():
            b += 1
        while b - a < 2 and a > 0 and not line[a - 1].isspace():
            a -= 1
        wrong = line[a:b]
        sugg = line[a:t.start] + t.form + line[t.start + t.len:b]
        if len(wrong) >= 2 and re.search("[가-힣]", wrong):
            out.append((wrong, sugg))
    return out


def find(text: str, user_dict: dict[str, str] | None = None) -> list[dict]:
    """[{wrong, suggestion, count, context}] — 많이 나온 것부터. 본문은 바꾸지 않는다."""
    if not text or not spacing.is_available():
        return []
    if user_dict is None:
        from core import user_typos
        user_dict = user_typos.load()
    lines = [ln for ln in text.split("\n") if re.search("[가-힣]", ln)]
    if not lines:
        return []
    with spacing._lock:   # 일괄 처리 작업들이 Kiwi 하나를 같이 쓴다
        token_lists = list(spacing._get().tokenize(lines, typos=_MODE))
    seen: dict[tuple[str, str], dict] = {}
    for line, toks in zip(lines, token_lists):
        for wrong, sugg in _candidates(line, toks):
            if wrong in user_dict or (wrong, sugg) in seen:
                continue
            i = text.find(wrong)
            ctx = text[max(0, i - _CTX):i + len(wrong) + _CTX].replace("\n", " ")
            seen[(wrong, sugg)] = {"wrong": wrong, "suggestion": sugg,
                                   "count": text.count(wrong), "context": ctx}
    return sorted(seen.values(), key=lambda r: -r["count"])


def path_for(result_path: Path) -> Path:
    return result_path.parent / SUBDIR / f"{result_path.stem}.tsv"


def save(result_path: Path, user_dict: dict[str, str] | None = None) -> tuple[Path, int]:
    """결과 파일의 의심 목록을 저장한다. (목록 경로, 항목 수)"""
    rows = find(result_path.read_text(encoding="utf-8"), user_dict)
    out = path_for(result_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{r['wrong']}\t{r['suggestion']}\t{r['count']}\t{r['context'].replace(chr(9), ' ')}\n"
                   for r in rows)
    out.write_text(_HEAD + "\n" + body, encoding="utf-8")
    return out, len(rows)


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) == 4:
            rows.append(dict(zip(_FIELDS, (parts[0], parts[1], int(parts[2]), parts[3]))))
    return rows
