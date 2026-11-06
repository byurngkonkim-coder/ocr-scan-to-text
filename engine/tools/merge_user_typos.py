"""git 밖에 있던 사용자 오타 사전(백업본)을 git 의 engine/user_typos.tsv 에 합친다 — 회사 PC 1회 현행화용.

   python tools/merge_user_typos.py <백업 사전 경로>

등록 안전장치(user_typos.add)를 그대로 거친다. 같은 틀린 말의 고칠 말이 다르면 백업본 것으로 바꾼다.
"""
from __future__ import annotations

import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from core import user_typos  # noqa: E402


def main(src: str) -> None:
    have = user_typos.load()
    theirs = user_typos.load(Path(src))
    new = [(w, r) for w, r in theirs.items() if have.get(w) != r]
    changed = [(w, have[w], r) for w, r in new if w in have]
    added, rejected = user_typos.add(new)
    print(f"백업본 {len(theirs)}건 중 이미 같음 {len(theirs) - len(new)}건, 추가·변경 {added}건, 거부 {len(rejected)}건")
    for w, old, r in changed:
        print(f"  변경: {w} → {old}  ⇒  {w} → {r}")
    for w, r, why in rejected:
        print(f"  거부: {w} → {r} ({why})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
