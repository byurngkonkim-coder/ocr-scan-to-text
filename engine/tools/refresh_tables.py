"""재처리 결과의 표를 지금 엔진의 표 인식으로 바꿔 넣는다 — 표 인식이 나아지기 전에 만든 결과용.

2x2Matrix_알렉스로위 결과(2026-10-21 03:24)는 2×2 매트릭스를 축 이름 머리행·머리열 표로 복원하는 기능
(bc6275d, 같은 날 21:08)보다 먼저 만들어져, 표가 축 없는 2×2 칸뿐이고 축 글자는 표 밖 본문으로 새어 있었다.
본문 OCR 을 다시 하지 않고 표만 다시 뽑는다:
  1) extract — 원본 PDF 쪽마다 figure_detector.detect 로 표 Markdown 을 뽑아 02_결과/_재처리/_표/{이름}.json 에 저장
  2) splice  — 재처리 결과의 옛 표와 새 표를 순서·글자로 짝지어 바꾸고, 새 표의 축에 다 들어간 표 밖 축 글자 줄을 지운다

사용: venv/Scripts/python.exe tools/refresh_tables.py extract 20261021_032448_2x2Matrix_알렉스로위 <원본 PDF>
     venv/Scripts/python.exe tools/refresh_tables.py splice  20261021_032448_2x2Matrix_알렉스로위
"""
from __future__ import annotations

import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

ENGINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ENGINE))

OUT_DIR = ENGINE.parent / "02_결과" / "_재처리"
CACHE_DIR = OUT_DIR / "_표"
_DPI = 150                      # tests/sample_docs/matrix_p*.png 와 같은 해상도(정답표 기준)
_MATCH_MIN = 0.5                # 옛 표 글자가 새 표에 든 몫 하한
_LEAK_LINE_MAX = 40


def extract(stem: str, pdf: Path) -> Path:
    import pymupdf
    from PIL import Image
    from core import figure_detector as fd
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{stem}.json"
    done = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    doc = pymupdf.open(pdf)
    for i in range(doc.page_count):
        if str(i + 1) in done:                      # 끊겨도 이어서
            continue
        pm = doc[i].get_pixmap(dpi=_DPI)
        img = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
        done[str(i + 1)] = [r.table_md for r in fd.detect(img).regions if r.kind == "table" and r.table_md]
        if (i + 1) % 10 == 0:
            cache.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
            print(f"{i + 1}/{doc.page_count} 쪽, 표 {sum(map(len, done.values()))}개", flush=True)
    cache.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
    return cache


def _norm(s: str) -> str:
    return re.sub(r'[^0-9A-Za-z가-힣]', '', s)


def _table_blocks(lines: list[str]) -> list[tuple[int, int]]:
    """Markdown 표 줄 묶음의 [시작, 끝) 줄 번호."""
    out, i = [], 0
    while i < len(lines):
        if lines[i].startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _contained(old: str, new: str) -> float:
    """옛 표 글자 중 새 표에 (차례대로) 들어 있는 몫 — 새 표엔 축 글자가 더해져 단순 유사도가 낮게 나온다."""
    if not old:
        return 0.0
    return sum(b.size for b in SequenceMatcher(None, old, new, autojunk=False).get_matching_blocks()) / len(old)


_PLAIN_TABLE_KEEP = 0.9


def _bag_contained(old: str, new: str) -> float:
    """옛 표 글자 중 새 표에 든 몫 — 순서 무관(칸 배치가 바뀌어도)."""
    from collections import Counter
    if not old:
        return 0.0
    return sum((Counter(old) & Counter(new)).values()) / len(old)


def _align(olds: list[str], news: list[str]) -> list[tuple[int, int]]:
    """순서를 지키며 같음 합이 가장 큰 옛 표↔새 표 짝(DP). 같음이 _MATCH_MIN 미만인 짝은 맺지 않는다.
    앞에서부터 욕심내어 짝지으면 같은 이름 표('제품-시장 매트릭스' 1.2·6.21)에서 순서가 어긋났다."""
    n, m = len(olds), len(news)
    score = [[_contained(o, w) if o else 0.0 for w in news] for o in olds]
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            take = score[i][j] + best[i + 1][j + 1] if score[i][j] >= _MATCH_MIN else -1.0
            best[i][j] = max(take, best[i + 1][j], best[i][j + 1])
    out, i, j = [], 0, 0
    while i < n and j < m:
        if score[i][j] >= _MATCH_MIN and best[i][j] == score[i][j] + best[i + 1][j + 1]:
            out.append((i, j))
            i, j = i + 1, j + 1
        elif best[i][j] == best[i + 1][j]:
            i += 1
        else:
            j += 1
    return out


def splice(stem: str) -> dict:
    md_path = OUT_DIR / f"{stem}.md"
    pages = json.loads((CACHE_DIR / f"{stem}.json").read_text(encoding="utf-8"))
    new = [t for p in sorted(pages, key=int) for t in pages[p]]
    lines = md_path.read_text(encoding="utf-8").split("\n")
    blocks = _table_blocks(lines)
    olds = [_norm("".join(lines[a:b])) for a, b in blocks]
    # 매트릭스(축 머리 '세로축 \ 가로축')는 바꾼다. 일반 격자 표는 옛 표 글자가 거의 다(순서 무관) 들어 있을 때만
    # — MBTI 16유형 표는 새 엔진이 칸 글자를 덜 읽어('ISTP' 등 빠짐) 바꾸면 손실이었다
    plan = [(blocks[i], j) for i, j in _align(olds, [_norm(t) for t in new])
            if " \\ " in new[j].split("\n")[0] or _bag_contained(olds[i], _norm(new[j])) >= _PLAIN_TABLE_KEEP]
    leaks = 0
    for (a, b), j in reversed(plan):
        table = new[j].split("\n")
        have = _norm(new[j])
        # 표 바로 아래·위 짧은 줄 중 낱말이 모두 새 표에 든 것 = 표 밖으로 샌 축 글자('변화없음 변화 포장')
        drop = set()
        for x in (b, b + 1, a - 1, a - 2):
            if 0 <= x < len(lines) and lines[x].strip() and len(lines[x]) <= _LEAK_LINE_MAX \
                    and not lines[x].startswith(("#", "|", "그림", "[p.")) \
                    and all(_norm(w) and _norm(w) in have for w in lines[x].split()):
                drop.add(x)
        for x in sorted(drop, reverse=True):
            del lines[x]
            if x < a:
                a, b = a - 1, b - 1
        leaks += len(drop)
        lines[a:b] = table
    inserted = _insert_unmatched(lines, new, {j for _, j in plan})
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"old_tables": len(blocks), "new_tables": len(new), "replaced": len(plan), "leak_lines": leaks,
            "inserted": inserted}


_CELL_LINE_MAX = 60
_CELL_LINE_IN = 0.9      # 줄 글자의 이만큼이 새 표 안에 있으면 표 밖으로 흩어진 칸 글자


def _insert_unmatched(lines: list[str], new: list[str], used: set[int]) -> int:
    """옛 결과에 없던 표(옛 OCR 이 표로 못 알아봐 칸 글자가 본문 줄로 흩어진 그림) — 앞뒤로 이미 넣은 새 표
    사이 줄 중 글자가 거의 다 이 표 안에 있는 짧은 줄을 흩어진 칸 글자로 보고, 첫 줄 자리에 표를 넣고 그 줄들은
    지운다. 그런 줄이 없으면 넣지 않는다(자리를 모르는 표를 아무 데나 두지 않음). 같은 표를 두 번 잡은 것은 하나만."""
    placed = 0
    for j, table in enumerate(new):
        if j in used or any(_contained(_norm(table), _norm(new[u])) > 0.9 for u in used if abs(u - j) <= 2):
            continue
        have = _norm(table)
        prev = [k for k, l in enumerate(lines) if l.startswith("|") and any(l in new[u] for u in used if u < j)]
        nxt = [k for k, l in enumerate(lines) if l.startswith("|") and any(l in new[u] for u in used if u > j)]
        lo, hi = (max(prev) + 1 if prev else 0), (min(nxt) if nxt else len(lines))
        cells = [k for k in range(lo, hi) if lines[k].strip() and len(lines[k]) <= _CELL_LINE_MAX
                 and not lines[k].startswith(("#", "|", "그림", "[p.")) and len(_norm(lines[k])) >= 3
                 and _contained(_norm(lines[k]), have) >= _CELL_LINE_IN]
        if not cells:
            continue
        first = cells[0]
        for k in reversed(cells):
            del lines[k]
        lines[first:first] = [""] + table.split("\n") + [""]
        used.add(j)
        placed += 1
    return placed


if __name__ == "__main__":
    cmd, name = sys.argv[1], Path(sys.argv[2]).stem
    if cmd == "extract":
        print(extract(name, Path(sys.argv[3])))
    else:
        print(splice(name))
