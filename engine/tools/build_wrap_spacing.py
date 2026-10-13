"""줄바꿈 띄어쓰기 표 만들기 — 정제된 책 txt 에서 '한글 두 글자가 붙어 쓰이는지, 띄어 쓰이는지'를 센다.

스캔 책은 줄 끝에서 낱말을 아무 데서나 자른다('운용되었 / 다'). 줄을 이을 때 늘 띄우면
낱말이 갈라지고, 늘 붙이면 낱말이 붙는다. 줄 끝 글자(c1)와 다음 줄 첫 글자(c2)가 정제본에서
붙어 쓰인 비율이 높으면 붙인다 — 그 쌍만 core/data/wrap_nospace.txt 에 남긴다.

    python engine/tools/build_wrap_spacing.py <정제본 폴더>      # 기본: 02_결과
    python engine/tools/build_wrap_spacing.py <폴더> --eval      # 표를 만들지 않고 정확도만 잰다
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "engine" / "core" / "data" / "wrap_nospace.txt"
MIN_SEEN = 5          # 이보다 적게 본 쌍은 판단하지 않는다(기존대로 띄움)
NOSPACE_RATIO = 0.6   # 붙여 쓴 비율이 이 이상이면 붙인다

_PAIR = re.compile(r"([가-힣])( ?)(?=([가-힣]))")


def count_pairs(texts) -> tuple[Counter, Counter]:
    joined, spaced = Counter(), Counter()
    for t in texts:
        for m in _PAIR.finditer(t):
            (spaced if m.group(2) else joined)[m.group(1) + m.group(3)] += 1
    return joined, spaced


def nospace_pairs(joined: Counter, spaced: Counter) -> set[str]:
    out = set()
    for pair in joined.keys() | spaced.keys():
        j, s = joined[pair], spaced[pair]
        if j + s >= MIN_SEEN and j / (j + s) >= NOSPACE_RATIO:
            out.add(pair)
    return out


def _files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.txt") if not re.search(r"_(OCR중|정제중|원문)", p.stem))


def evaluate(folder: Path) -> None:
    """책을 반씩 나눠 앞 절반으로 표를 만들고, 뒤 절반에서 '줄 끝 자리'를 흉내 내 맞힌 비율을 잰다."""
    files = _files(folder)
    train, test = files[::2], files[1::2]
    table = nospace_pairs(*count_pairs(p.read_text(encoding="utf-8-sig") for p in train))
    right = {"always_space": 0, "table": 0}
    total = 0
    for p in test:
        for m in _PAIR.finditer(p.read_text(encoding="utf-8-sig")):
            truth_space = bool(m.group(2))
            total += 1
            right["always_space"] += truth_space
            right["table"] += truth_space == ((m.group(1) + m.group(3)) not in table)
    print(f"학습 {len(train)}권 · 평가 {len(test)}권 · 글자 경계 {total:,}곳 · 표 {len(table):,}쌍")
    for k, v in right.items():
        print(f"  {k:13s} 정확도 {v / total:.1%}")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    folder = Path(args[0]) if args else ROOT / "02_결과"
    if "--eval" in sys.argv:
        evaluate(folder)
        return
    joined, spaced = count_pairs(p.read_text(encoding="utf-8-sig") for p in _files(folder))
    pairs = sorted(nospace_pairs(joined, spaced))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(pairs) + "\n", encoding="utf-8")
    print(f"{len(pairs):,}쌍 → {OUT}")


if __name__ == "__main__":
    main()
