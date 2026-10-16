"""내장 정제규칙(비용 우선) 평가 도구 — 규칙을 고칠 때 과적합·본문 손실을 숫자로 확인한다.

storage/uploads 의 '정제 전 원문' 사본(*_원문.txt)에 iter_clean(local=True)을 돌려
noise_rules 각 단계가 무엇을 바꿨는지 규칙별로 잰다(운영 코드는 건드리지 않고 함수를 감싸 측정).

  python tools/cleanup_eval.py [원문 glob…] [--front-matter] [--samples N] [--out 보고서.md]

보고 항목(파일별·합계):
  한글 보존율(전체), 규칙별 지운 한글 글자 수, 규칙별 지운 줄 표본(본문 손실 점검용),
  번호 항목 단락 수, 빈 항목(번호만 있는 단락) 수
판정 기준(경고): 규칙 하나가 파일 한글의 3% 이상을 지우면 '검토' 표시 — 부속물 삭제(--front-matter)는 제외.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from core import config, noise_rules, text_cleaner  # noqa: E402

_HANGUL = re.compile(r"[가-힣]")
_PAGE_RE = re.compile(r"^\[페이지\s*\d+\]\s*$", re.M)
_SEP_RE = re.compile(r"^\s*[─━]{10,}\S*\s*\n")
_WARN_RATIO = 0.03


def pages_of(raw: str) -> list[str]:
    return [_SEP_RE.sub("", p, count=1).strip("\n") for p in _PAGE_RE.split(raw)[1:]]


def hangul(s: str) -> int:
    return len(_HANGUL.findall(s))


class Probe:
    """noise_rules 함수를 감싸 단계별 한글 감소량과 사라진 줄을 기록한다."""

    STAGES = ("clean_page", "strip_fuzzy_headers", "drop_book_matter_rules",
              "redistribute_number_stacks", "paragraph_entries")

    def __init__(self):
        self.removed = defaultdict(int)          # 단계 → 지운 한글 수
        self.lines = defaultdict(list)           # 단계 → 사라진 줄(한글 있는 것)
        self.orig = {}

    @staticmethod
    def _text(x):
        if isinstance(x, tuple):
            x = x[0]
        return "\n".join(x) if isinstance(x, list) else x

    def _wrap(self, name):
        fn = self.orig[name]

        def run(*a, **kw):
            before = self._text(a[0])
            res = fn(*a, **kw)
            after = self._text(res)
            self.removed[name] += max(0, hangul(before) - hangul(after))
            gone = set(after.split("\n"))
            self.lines[name] += [l.strip() for l in before.split("\n")
                                 if l.strip() and l not in gone and hangul(l) >= 2]
            return res
        return run

    def __enter__(self):
        for n in self.STAGES:
            self.orig[n] = getattr(noise_rules, n)
            setattr(noise_rules, n, self._wrap(n))
        return self

    def __exit__(self, *exc):
        for n, fn in self.orig.items():
            setattr(noise_rules, n, fn)


def clean(pages, front):
    gen = text_cleaner.iter_clean(pages, drop_front_matter=front, local=True)
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value or ""


def evaluate(path: Path, front: bool) -> dict:
    pages = pages_of(path.read_text(encoding="utf-8", errors="replace"))
    src = hangul("\n".join(pages))
    with Probe() as probe:
        out = clean(pages, front)
    paras = [p for p in out.split("\n\n") if p.strip()]
    return {
        "name": path.name.replace("_원문.txt", ""),
        "pages": len(pages),
        "src": src,
        "out": hangul(out),
        "removed": dict(probe.removed),
        "lines": probe.lines,
        "paras": len(paras),
        "empty": sum(1 for p in paras if re.fullmatch(r"\s*\d{1,4}\s*", p)),
    }


def report(rows: list[dict], samples: int, front: bool, warn_ratio: float = _WARN_RATIO) -> str:
    st = Probe.STAGES
    out = ["# 내장 정제규칙 평가", "",
           f"대상 {len(rows)}개 · 부속물 삭제 {'켬' if front else '끔'} · "
           f"경고: 규칙 하나가 한글 {warn_ratio:.0%} 이상 삭제", "",
           "| 파일 | 쪽 | 한글 보존 | " + " | ".join(st) + " | 단락 | 빈 항목 | 판정 |",
           "|---|---|---|" + "---|" * len(st) + "---|---|---|"]
    tot_src = tot_out = 0
    flagged = []
    for r in rows:
        tot_src += r["src"]
        tot_out += r["out"]
        cells, warn = [], []
        for s in st:
            n = r["removed"].get(s, 0)
            ratio = n / r["src"] if r["src"] else 0
            cells.append(f"{n:,}" + (f" ({ratio:.1%})" if ratio >= 0.005 else ""))
            if ratio >= warn_ratio and not (front and s == "drop_book_matter_rules"):
                warn.append(s)
        keep = r["out"] / r["src"] if r["src"] else 1
        out.append(f"| {r['name'][:40]} | {r['pages']} | {keep:.1%} | " + " | ".join(cells)
                   + f" | {r['paras']} | {r['empty']} | {'검토: ' + ','.join(warn) if warn else '통과'} |")
        if warn:
            flagged.append((r, warn))
    out += ["", f"**합계 한글 보존 {tot_out / tot_src:.2%}** ({tot_out:,}/{tot_src:,})", ""]
    for r, warn in flagged:
        out.append(f"## 검토: {r['name']}")
        for s in warn:
            out.append(f"- `{s}` 지운 줄 표본:")
            for l in r["lines"][s][:samples]:
                out.append(f"  - {l[:120]}")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("globs", nargs="*", default=["*_원문.txt"])
    ap.add_argument("--front-matter", action="store_true")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--min-pages", type=int, default=1)
    ap.add_argument("--warn", type=float, default=_WARN_RATIO, help="검토 표시 기준(규칙 하나의 한글 삭제 비율)")
    ap.add_argument("--out")
    a = ap.parse_args()
    config.get_local_cleanup_engine = lambda: "rules"   # 평가는 항상 내장 규칙 경로
    up = ENGINE / "storage" / "uploads"
    files = sorted({p for g in a.globs for p in (up.glob(g) if not Path(g).is_absolute() else [Path(g)])})
    rows = []
    for f in files:
        r = evaluate(f, a.front_matter)
        if r["pages"] >= a.min_pages:
            rows.append(r)
            print(f"  {r['name'][:50]}: 보존 {r['out'] / max(1, r['src']):.1%}", file=sys.stderr)
    text = report(rows, a.samples, a.front_matter, a.warn)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
