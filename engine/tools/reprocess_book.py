"""결과 한 권 순차 개선 — 02_결과 의 결과를 지금 규칙으로 다시 만들어 02_결과/_재처리/ 에 쓴다(원본 결과는 그대로).

입력은 결과 MD 본문(remake_md.rebuild 와 같은 경로 — 원문 사본은 쪽 표지 대조에만 쓴다).
책정보 파일 02_결과/_재처리/_책정보/{결과 이름}.txt 가 있으면 그 책·저자·차례를 쓴다 — 원본 스캔 PDF 앞쪽과
인터넷 검색으로 사람이 확인해 적은 정보. 형식:

    책: …
    저자: …
    출처: …(어디서 확인했는지)

    차례: (아무 설명)
    제1편 자아편
      01. 첫 장 제목 12          ← 들여쓰기 2칸 = 한 단계 아래, 끝 숫자 = 쪽번호

사용: venv/Scripts/python.exe tools/reprocess_book.py 20261019_201720_삶의지혜_자아편
     (이름 없이 실행하면 아직 _재처리 에 없는 가장 오래된 결과 하나)
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
    entries = []
    for line in toc.splitlines()[1:]:
        if not line.strip():
            continue
        m = _PAGE_TAIL_RE.search(line.strip())
        title = line.strip()[:m.start()] if m and not re.fullmatch(r'\d+\.?', line.strip()) else line.strip()
        entries.append({"level": (len(line) - len(line.lstrip(" "))) // 2 + 1, "title": title.strip(),
                        "page": int(m.group(1)) if m else None})
    if entries:
        info["toc"] = entries
    return info


def next_target() -> str | None:
    done = {p.stem for p in OUT_DIR.glob("*.md")}
    for p in sorted(RESULTS.glob("*.md")):      # 이름 앞 처리 시각 순 = 오래된 것부터
        if p.stem not in done:
            return p.stem
    return None


def reprocess(stem: str) -> tuple[Path, dict]:
    import ocr_gradio
    from core import remake_md
    from core.types import OutputFormat
    src = RESULTS / f"{stem}.md"
    info = read_book_info(INFO_DIR / f"{stem}.txt")
    entries, body, note = remake_md.rebuild(src.read_text(encoding="utf-8"), info.get("toc"))
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
    return out, {"note": note, "logs": logs, "book_info": bool(info)}


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else next_target()
    if not target:
        print("다시 만들 결과가 없습니다")
        sys.exit(0)
    path, rep = reprocess(Path(target).stem)
    print(path)
    print("책정보 파일:", "있음" if rep["book_info"] else "없음")
    print(rep["note"])
    for line in rep["logs"]:
        print(line)
