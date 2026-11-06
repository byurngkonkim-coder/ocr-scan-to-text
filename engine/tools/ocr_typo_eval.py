"""OCR 혼동 글자 후보(core/ocr_typos) 정답률 측정 — 점수 차 기준(MARGIN_MIN)과 자동 반영 여부를 정하는 근거.

1) 후보 뽑기: 결과 txt 또는 정제 전 원문(*_원문.txt, 내장 규칙으로 정제한 뒤)에서 후보를 TSV 로 낸다.
   python tools/ocr_typo_eval.py list <파일…> --out 후보.tsv [--min 0]
   → 열: 판정(빈칸) · 틀린 말 · 제안 · 점수 차 · 횟수 · 문맥. 사람이 판정 칸에 O(맞음)/X(틀림)/?(모름)를 적는다.
2) 집계: 판정을 적은 TSV 를 점수 차 구간 × 글자 수로 집계한다.
   python tools/ocr_typo_eval.py score 후보.tsv
본문·운영 설정은 건드리지 않는다(--min 은 이 도구 안에서만 MARGIN_MIN 을 바꾼다).
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from core import config, ocr_typos, text_cleaner, typo_suspects  # noqa: E402

_PAGE_RE = re.compile(r"^\[페이지\s*\d+\]\s*$", re.M)
_HEAD = "판정\t틀린 말\t제안\t점수 차\t횟수\t문맥"
_BANDS = [(0, 8), (8, 12), (12, 16), (16, 20), (20, 1e9)]


def _text_of(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    if not path.stem.endswith("_원문"):
        return raw
    pages = [p.strip() for p in _PAGE_RE.split(raw)[1:]]
    config.get_local_cleanup_engine = lambda: "rules"    # 비용 우선 내장 규칙 경로
    return text_cleaner.clean_pages(pages, local=True)


def cmd_list(paths: list[Path], out: Path, minimum: float | None) -> None:
    if minimum is not None:
        ocr_typos.MARGIN_MIN = minimum
    rows = []
    for p in paths:
        text = _text_of(p)
        lines = [ln for ln in text.split("\n") if re.search("[가-힣]", ln)]
        best: dict[tuple[str, str], float] = {}
        with typo_suspects._lock:
            for w, s, m in ocr_typos.candidates(lines):
                best[(w, s)] = max(m, best.get((w, s), m))
        for (w, s), m in best.items():
            i = text.find(w)
            ctx = text[max(0, i - 15):i + len(w) + 15].replace("\n", " ").replace("\t", " ")
            rows.append(("", w, s, f"{m:.1f}", str(text.count(w)), ctx))
        print(f"{p.name}: 후보 {len(best)}개")
    rows.sort(key=lambda r: -float(r[3]))
    out.write_text(_HEAD + "\n" + "".join("\t".join(r) + "\n" for r in rows), encoding="utf-8")
    print(f"→ {out} ({len(rows)}행) — 판정 칸에 O/X/? 를 적은 뒤 score 로 집계")


def cmd_score(path: Path) -> None:
    tally: Counter = Counter()
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        f = line.split("\t")
        if len(f) < 4 or f[0].strip().upper() not in ("O", "X"):
            continue
        m = float(f[3])
        band = next(f"{lo:g}~{hi:g}" if hi < 1e9 else f"{lo:g}~" for lo, hi in _BANDS if lo <= m < hi)
        size = "2자" if len(f[1]) <= 2 else "3자+"
        tally[(band, size, f[0].strip().upper())] += 1
    print("| 점수 차 | 글자 수 | 맞음 | 틀림 | 정답률 |\n|---|---|---|---|---|")
    for band in [f"{lo:g}~{hi:g}" if hi < 1e9 else f"{lo:g}~" for lo, hi in _BANDS]:
        for size in ("2자", "3자+"):
            o, x = tally[(band, size, "O")], tally[(band, size, "X")]
            if o + x:
                print(f"| {band} | {size} | {o} | {x} | {o / (o + x):.0%} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("list")
    a.add_argument("paths", nargs="+", type=Path)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--min", type=float, default=None, help="이 측정에서만 쓸 점수 차 하한(기본 MARGIN_MIN)")
    b = sub.add_parser("score")
    b.add_argument("path", type=Path)
    args = ap.parse_args()
    if args.cmd == "list":
        cmd_list(args.paths, args.out, args.min)
    else:
        cmd_score(args.path)


if __name__ == "__main__":
    main()
