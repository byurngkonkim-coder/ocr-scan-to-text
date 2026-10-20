"""사용자 오타 사전 — ② 탭 의심 목록에서 사람이 확인해 등록한 '틀린 말 → 고칠 말'만 적용한다.

korean_corrector.TYPO_DICT 는 MyOCR·파일형식변환기_MD 와 같은 파일로 동기화돼 있어 손대지 않고,
01_OCR 에서만 쓰는 사전을 따로 둔다(intent 2026-10-17-로컬-맞춤법-자동교정).
글자 교정 오교정 0건이 기준이라 사람이 등록한 항목 말고는 글자를 바꾸지 않는다.

안전장치(등록 시 거부):
  - 틀린 말이 2자 미만 — '이→의' 같은 것은 다른 낱말 속에서 바뀐다.
  - 고칠 말이 틀린 말을 품음 — 돌릴 때마다 다시 바뀐다(두 번 적용해도 같아야 한다).
  - 기존 항목과 연쇄 — 한 항목의 고칠 말 속에 다른 항목의 틀린 말이 들어 있으면 결과가 순서에 좌우된다.
"""
from __future__ import annotations

import re
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "user_typos.tsv"
_HEADER = "# 사용자 오타 사전 — 틀린 말<TAB>고칠 말. ② 일괄 처리 탭 '의심 목록'에서 등록한다.\n"
_MIN_LEN = 2

_cache: dict[Path, tuple[float, dict[str, str], re.Pattern | None]] = {}


def load(path: Path | None = None) -> dict[str, str]:
    return dict(_compiled(path or PATH)[1])


def _compiled(path: Path) -> tuple[float, dict[str, str], re.Pattern | None]:
    mtime = path.stat().st_mtime if path.exists() else -1.0
    hit = _cache.get(path)
    if hit and hit[0] == mtime:
        return hit
    pairs: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            wrong, _, right = line.partition("\t")
            if wrong and right:
                pairs[wrong] = right
    # 긴 것부터 — '지해를' 이 '지해' 보다 먼저 맞는다. 한 번에 치환해 연쇄가 없다.
    pat = re.compile("|".join(map(re.escape, sorted(pairs, key=len, reverse=True)))) if pairs else None
    _cache[path] = (mtime, pairs, pat)
    return _cache[path]


def apply(text: str, path: Path | None = None) -> tuple[str, int]:
    """(본문, 바꾼 곳 수)."""
    _, pairs, pat = _compiled(path or PATH)
    if not text or pat is None:
        return text, 0
    n = 0

    def _sub(m: re.Match) -> str:
        nonlocal n
        n += 1
        return pairs[m.group()]

    return pat.sub(_sub, text), n


def count(text: str, wrong: str) -> int:
    """등록 전 '본문에서 몇 곳 바뀌는지' 보여 주기용."""
    return text.count(wrong) if wrong else 0


def _reject_reason(wrong: str, right: str, pairs: dict[str, str]) -> str:
    if len(wrong) < _MIN_LEN:
        return f"틀린 말은 {_MIN_LEN}자 이상이어야 합니다"
    if not right or wrong == right:
        return "고칠 말이 비었거나 틀린 말과 같습니다"
    if wrong in right:
        return "고칠 말 안에 틀린 말이 들어 있어 돌릴 때마다 다시 바뀝니다"
    if any(wrong in r for r in pairs.values()) or any(w in right for w in pairs):
        return "이미 등록한 항목과 겹쳐 결과가 순서에 따라 달라집니다"
    return ""


def add(new_pairs, path: Path | None = None) -> tuple[int, list[tuple[str, str, str]]]:
    """(등록 수, [(틀린 말, 고칠 말, 거부 사유)]). 같은 틀린 말은 새 고칠 말로 바꾼다."""
    path = path or PATH
    pairs = load(path)
    added, rejected = 0, []
    for wrong, right in new_pairs:
        wrong, right = (wrong or "").strip(), (right or "").strip()
        others = {w: r for w, r in pairs.items() if w != wrong}
        reason = _reject_reason(wrong, right, others)
        if reason:
            rejected.append((wrong, right, reason))
            continue
        pairs[wrong] = right
        added += 1
    if added:
        path.write_text(_HEADER + "".join(f"{w}\t{r}\n" for w, r in pairs.items()), encoding="utf-8")
        _cache.pop(path, None)   # 같은 초 안에 다시 쓰면 수정시각이 같아 옛 사전을 쓴다
    return added, rejected
