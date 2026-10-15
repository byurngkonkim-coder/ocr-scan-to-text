"""한글 교정기 두 판을 OCR 원문 샘플에 돌려 바뀐 곳을 규칙별로 세고, 정제본과 얼마나 맞는지 잰다.

샘플 폴더에는 같은 책의 `*_원문.txt`(원시 OCR)와 `*.txt`(정제본, 이름에서 `_원문` 을 뺀 것)가 짝으로 있어야 한다.
바뀐 곳마다 앞뒤 4글자 문맥이 정제본에 어느 쪽으로 있는지 보고 일치/어긋남을 판정한다.
정제본에 쪼개진 어절('수 있 는')이 1만 자당 2개 이상 남은 책은 정제가 덜 된 것이라 판정에서 뺀다.

사용 (01_OCR 루트에서):
    git show <기준커밋>:engine/core/korean_corrector.py > old_kc.py
    engine\\venv\\Scripts\\python.exe engine/tools/corrector_compare.py old_kc.py engine/core/korean_corrector.py <샘플 폴더>
"""
import difflib
import importlib.util
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

CTX = 4
SPLIT_RE = re.compile(r"(?<![가-힣])[가-힣] (?:는|을|를|이|가|은|로|다)(?![가-힣])")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rule(old, new, old_ctx, new_ctx):
    if old == "디" and new == "다":
        return "디→다"
    if old in ("-", "디-") and new in ("", "다"):
        return "디-.→다."
    if old.strip() == "" and new == " ":
        return next((f"의존명사 띄움({k})" for k in ("것", "때", "줄") if " " + k in new_ctx), "띄움(기타)")
    if old == " " and new == "" and "않 " in old_ctx:
        return "'않 ' 결합"
    if old.strip() == "" and new == "":
        return "결합(기타)"
    if old.strip() and new.strip():
        return "글자 교정"
    return "기타"


def _compare(args):
    old_py, new_py, src = args
    old, new = _load("kc_old", old_py), _load("kc_new", new_py)
    ref_path = src.with_name(src.name.replace("_원문.txt", ".txt"))
    raw = src.read_text(encoding="utf-8", errors="replace")
    ref = ref_path.read_text(encoding="utf-8", errors="replace")
    valid = len(SPLIT_RE.findall(ref)) / max(len(ref), 1) * 1e4 < 2
    a = old.refine_korean_text(raw).splitlines(keepends=True)
    b = new.refine_korean_text(raw).splitlines(keepends=True)
    rows = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        x, y = "".join(a[i1:i2]), "".join(b[j1:j2])
        for op2, p1, p2, q1, q2 in difflib.SequenceMatcher(None, x, y, autojunk=False).get_opcodes():
            if op2 == "equal":
                continue
            oc, nc = x[max(0, p1 - CTX):p2 + CTX], y[max(0, q1 - CTX):q2 + CTX]
            ro, rn = ref.count(oc), ref.count(nc)
            verdict = "일치" if rn and not ro else "어긋남" if ro and not rn else "판정불가"
            rows.append((_rule(x[p1:p2], y[q1:q2], oc, nc), verdict, oc, nc))
    return src.name, valid, rows


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    old_py, new_py, src_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    srcs = [s for s in sorted(src_dir.glob("*_원문.txt"))
            if s.with_name(s.name.replace("_원문.txt", ".txt")).exists()]
    tab, bad = defaultdict(Counter), defaultdict(Counter)
    n_valid = 0
    with ProcessPoolExecutor() as pool:
        for name, valid, rows in pool.map(_compare, [(old_py, new_py, s) for s in srcs]):
            n_valid += valid
            for rule, verdict, oc, nc in rows:
                tab[rule]["전체"] += 1
                if valid:
                    tab[rule][verdict] += 1
                    if verdict == "어긋남":
                        bad[rule][f"{oc!r} → {nc!r}"] += 1
    total = sum(c["전체"] for c in tab.values())
    print(f"샘플 {len(srcs)}권, 판정 기준 유효 {n_valid}권 — 바뀐 곳 {total:,}건\n")
    print(f"{'규칙':16s} {'전체':>6s} {'일치':>6s} {'어긋남':>6s}  일치율")
    for rule, c in sorted(tab.items(), key=lambda x: -x[1]["전체"]):
        a, d = c["일치"], c["어긋남"]
        print(f"{rule:16s} {c['전체']:6d} {a:6d} {d:6d}  {(a / (a + d) if a + d else 0):6.1%}")
    for rule, c in bad.items():
        print(f"\n[{rule}] 어긋남 상위")
        for s, n in c.most_common(10):
            print(f"  {n:3d}  {s}")


if __name__ == "__main__":
    main()
