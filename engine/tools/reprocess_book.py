"""결과 한 권 순차 개선 — 02_결과 의 결과를 지금 규칙으로 다시 만들어 02_결과/_재처리/ 에 쓴다(원본 결과는 그대로).

입력은 결과 MD 본문(remake_md.rebuild 와 같은 경로 — 원문 사본은 쪽 표지 대조에만 쓴다).
책정보 파일 02_결과/_재처리/_책정보/{결과 이름}.txt 가 있으면 그 책·저자·차례를 쓴다 — 원본 스캔 PDF 앞쪽과
인터넷 검색으로 사람이 확인해 적은 정보. 형식:

    책: …
    저자: …
    출처: …(어디서 확인했는지)
    바꾸기: 정규식 => 바꿀 글      ← 이 책에만 맞는 고침(여러 줄 가능, \n = 줄바꿈)

    차례: (아무 설명)
    제1편 자아편
      01. 첫 장 제목 12          ← 들여쓰기 2칸 = 한 단계 아래, 끝 숫자 = 쪽번호
    매뉴얼 30 -                  ← 끝 숫자가 제목의 일부면 ' -' (쪽번호 없음)

사용: venv/Scripts/python.exe tools/reprocess_book.py 20261019_201720_삶의지혜_자아편
     (이름 없이 실행하면 아직 _재처리 에 없는 가장 오래된 결과 하나)
     tools/reprocess_book.py --상태   → 끝난 책 수·하던 책·다음 책(다시 시작 지점)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ENGINE))

RESULTS = ENGINE.parent / "02_결과"
OUT_DIR = RESULTS / "_재처리"
INFO_DIR = OUT_DIR / "_책정보"
_PAGE_TAIL_RE = re.compile(r'\s*(?:\.{2,}|…+|·+)?\s*(\d{1,4})$')


def read_book_info(path: Path) -> dict:
    """책정보 파일 → {book, author, basis, toc: [{level, title, page}]}. 없는 칸은 빠진다."""
    info: dict = {}
    if not path.exists():
        return info
    head, _, toc = path.read_text(encoding="utf-8").partition("\n차례:")
    for line in head.splitlines():
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if val and key in ("책", "저자", "출처"):
            info[{"책": "book", "저자": "author", "출처": "basis"}[key]] = val
        elif val and key == "바꾸기" and " => " in val:   # 이 책에만 맞는 고침(정규식 => 바꿀 글, \n 줄바꿈)
            # 바꿀 글 끝의 띄어쓰기는 살린다('성경의 진리 ' — 뒤 글과 붙지 않게)
            pat, rep = line.partition(":")[2].lstrip().split(" => ", 1)
            # '\142.' 는 묶음 1 + '42.' 가 아니라 8진수 글자 'b' 가 된다(IDEAHACKS49) — 숫자가 이어지면 \g<1> 로
            if re.search(r'\\\d\d', rep):
                raise ValueError(f"{path.name} 바꾸기 '{rep}': 묶음 번호 뒤에 숫자가 오면 \\g<1> 꼴로 쓰세요")
            info.setdefault("subs", []).append((pat, rep))
    entries = []
    for line in toc.splitlines()[1:]:
        if not line.strip():
            continue
        if line.rstrip().endswith(" -"):      # 끝 숫자가 제목의 일부('매뉴얼 30 -') — 쪽번호 없음
            entries.append({"level": (len(line) - len(line.lstrip(" "))) // 2 + 1,
                            "title": line.strip()[:-1].strip(), "page": None})
            continue
        m = _PAGE_TAIL_RE.search(line.strip())
        title = line.strip()[:m.start()] if m and not re.fullmatch(r'\d+\.?', line.strip()) else line.strip()
        entries.append({"level": (len(line) - len(line.lstrip(" "))) // 2 + 1, "title": title.strip(),
                        "page": int(m.group(1)) if m else None})
    if entries:
        info["toc"] = entries
    return info


_LOSS_MAX = 0.03     # 본문 한글 글자가 이만큼 넘게 줄면 실패 — 지우는 규칙이 본문까지 지운 것
_PAGE_DRIFT = 0.35   # 제목 자리(문서 줄 비율)와 차례 쪽(마지막 쪽 비율)이 이만큼 넘게 다르면 엉뚱한 자리
_DRIFT_SHARE = 0.2   # 쪽번호 있는 찾은 제목 중 이 비율(최소 2개)을 넘게 엉뚱하면 실패


def quality(entries: list[dict], src: str, out: str) -> dict:
    """재처리 결과 점검 — 차례 항목이 본문 제목 줄로 섰는지(같은 제목은 차례 순서로 짝지음), 제목 순서가
    차례와 같은지, 본문 한글 글자가 얼마나 줄었는지. ok=False 면 규칙이 무언가를 망가뜨린 것이다.
    반복 사고: 차례 조각 규칙이 진짜 제목 4개를 지움(21세기지식경영), 같은 제목 두 개를 한 줄로 봐 순서가
    뒤엉킴(12년차기획맨), 표 교체가 일반 표 글자를 줄임(2x2Matrix) — 2026-11-07."""
    from core import remake_md, toc_headings
    lines = out.split("\n")
    heads = [(i, l) for i, l in enumerate(lines) if l.startswith("#") and l != "# 차례"]
    used, at = set(), []
    for e in entries:
        k = next((i for i, l in heads if l == toc_headings._heading(e) and i not in used), None)
        at.append(k)
        if k is not None:
            used.add(k)
    ks = [k for k in at if k is not None]
    breaks = sum(1 for a, b in zip(ks, ks[1:]) if b < a)
    # 차례 쪽번호와 본문 속 자리 — 순서는 맞아도 제목이 모두 맨 뒤 주석 속 장 이름 줄에 선 사고(Alianment정렬)
    paged = [(k, e["page"]) for e, k in zip(entries, at) if k is not None and e.get("page")]
    last = max((p for _, p in paged), default=0)
    misplaced = sum(1 for k, p in paged if abs(k / len(lines) - p / last) > _PAGE_DRIFT) if last else 0
    hangul = lambda t, pat="[가-힣]": len(re.findall(pat, remake_md.parse_result(t)[1]))
    before, after, script = hangul(src), hangul(out), "한글"
    if hangul(src, "[A-Za-z]") > before:   # 영문 책 — 한글만 세면 '0 → 0' 이라 문단이 지워져도 몰랐다(JesusCEO 첫 문단 96곳)
        before, after, script = hangul(src, "[A-Za-z]"), hangul(out, "[A-Za-z]"), "영문"
    loss = (before - after) / before if before else 0.0
    return {"found": len(ks), "total": len(entries), "order_breaks": breaks, "misplaced": misplaced, "script": script,
            "missing": [e["title"] for e, k in zip(entries, at) if k is None],
            "hangul_before": before, "hangul_after": after, "loss": round(loss, 4),
            "ok": breaks == 0 and loss <= _LOSS_MAX and misplaced <= max(2, len(paged) * _DRIFT_SHARE)}


BASELINE = OUT_DIR / "_점검기준.json"


def ratchet(stem: str, q: dict, path: Path = BASELINE) -> str:
    """책마다 지금까지 가장 많이 세운 제목 수를 기준으로 — 줄면 회귀(규칙을 바꾼 뒤 앞 책들을 다시 돌려 확인).
    반복 사고: 21세기지식경영 48→44, 12년차기획맨 40→39 를 사람이 표를 대조해서야 찾았다(2026-11-07).
    늘면 기준을 올린다. 회귀면 사유 문자열, 아니면 ''."""
    import json
    base = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    best = base.get(stem, {}).get("found", 0)
    if q["found"] < best:
        return f"제목 {best} → {q['found']} 회귀"
    if q["found"] > best or stem not in base:
        base[stem] = {"found": q["found"], "total": q["total"]}
        path.write_text(json.dumps(base, ensure_ascii=False, indent=1), encoding="utf-8")
    return ""


def next_target() -> str | None:
    done = {p.stem for p in OUT_DIR.glob("*.md")}
    for p in sorted(RESULTS.glob("*.md")):      # 이름 앞 처리 시각 순 = 오래된 것부터
        if p.stem not in done:
            return p.stem
    return None


LOG = ENGINE.parent / "docs" / "순차개선_기록.md"
_ROUND_RE = re.compile(r'^### \d+회 — (.+?) \(\d{4}-\d{2}-\d{2}\)', re.M)


def status(log: Path = LOG) -> dict:
    """어디서 다시 시작할지 — 기록 문서에 회차가 적힌 책은 끝난 책, _재처리 결과는 있는데 회차가 없으면 하던 책
    (규칙 고치기·기록·커밋 전에 멈춤), 그다음은 아직 다시 만들지 않은 가장 오래된 결과."""
    done = _ROUND_RE.findall(log.read_text(encoding="utf-8")) if log.exists() else []
    made = sorted(p.stem for p in OUT_DIR.glob("*.md"))
    return {"done": len(done), "last": done[-1] if done else None,
            "unfinished": [s for s in made if s not in done], "next": next_target(),
            "remaining": sum(1 for p in RESULTS.glob("*.md") if p.stem not in done)}


def reprocess(stem: str) -> tuple[Path, dict]:
    import ocr_gradio
    from core import remake_md
    from core.types import OutputFormat
    src = RESULTS / f"{stem}.md"
    from core import user_typos
    info = read_book_info(INFO_DIR / f"{stem}.txt")
    text = user_typos.apply(src.read_text(encoding="utf-8"))[0]   # 결과를 만든 뒤 사전에 늘어난 항목까지
    subs = 0
    for pat, rep in info.get("subs", []):
        text, k = re.subn(pat, rep, text)
        subs += k
    entries, body, note = remake_md.rebuild(text, info.get("toc"))
    raw = remake_md._raw_copy(src)
    pages, name, total = remake_md.read_raw_pages(raw) if raw else (None, "", None)
    head = {"name": name or remake_md._STAMP_RE.sub("", stem), "pages": total, "front": body[:5000],
            **{k: info[k] for k in ("book", "author", "basis") if k in info}}
    logs: list[str] = []
    _, md = ocr_gradio._structured_texts(body, {"toc_entries": entries, "toc_note": note}, OutputFormat.MD,
                                         logs, pages, head)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{stem}.md"
    out.write_text(md, encoding="utf-8")
    tables = None
    if (OUT_DIR / "_표" / f"{stem}.json").exists():   # 지금 엔진으로 다시 뽑은 표가 있으면 바꿔 넣는다(refresh_tables)
        import refresh_tables
        tables = refresh_tables.splice(stem)
        # 새로 넣은 표 칸에도 오타 사전·이 책 고침을 — 칸 글자 OCR 오인식('경행사'·'따고드는', 2x2Matrix)
        md2 = user_typos.apply(out.read_text(encoding="utf-8"))[0]
        for pat, rep in info.get("subs", []):
            md2, k = re.subn(pat, rep, md2)
            subs += k
        out.write_text(md2, encoding="utf-8")
    q = quality(entries, src.read_text(encoding="utf-8"), out.read_text(encoding="utf-8"))
    return out, {"note": note, "logs": logs, "book_info": bool(info), "subs": subs, "tables": tables, "quality": q}


if __name__ == "__main__":
    if sys.argv[1:] == ["--상태"]:      # 다시 시작 지점(ocr-reprocess 스킬이 맨 먼저 부른다)
        st = status()
        print(f"끝난 책 {st['done']}권 (마지막: {st['last']}) · 남은 책 {st['remaining']}권")
        print("하던 책(회차 기록 없음):", ", ".join(st["unfinished"]) or "없음")
        print("다음 책:", st["next"] or "없음")
        sys.exit(0)
    target = sys.argv[1] if len(sys.argv) > 1 else next_target()
    if not target:
        print("다시 만들 결과가 없습니다")
        sys.exit(0)
    path, rep = reprocess(Path(target).stem)
    print(path)
    print("책정보 파일:", "있음" if rep["book_info"] else "없음", f"· 이 책 고침 {rep['subs']}곳")
    print(rep["note"])
    if rep["tables"]:
        print("표 다시 넣기:", rep["tables"])
    for line in rep["logs"]:
        print(line)
    q = rep["quality"]
    print(f"점검: 제목 {q['found']}/{q['total']} · 순서 어긋남 {q['order_breaks']} · 쪽과 먼 제목 {q['misplaced']} · "
          f"본문 {q['script']} {q['hangul_before']:,} → {q['hangul_after']:,} ({-q['loss']:+.1%})")
    regress = ratchet(Path(target).stem, q)
    if not q["ok"] or regress:
        print("⚠ 불합격 —", regress or "제목 순서가 차례와 다르거나, 제목이 차례 쪽과 먼 자리에 섰거나, 본문 한글이 3% 넘게 줄었다",
              "· 방금 바꾼 규칙부터 의심할 것")
        sys.exit(2)
