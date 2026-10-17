"""기존 결과 재교정 — 02_결과 의 txt 에 지금의 자동 교정을 다시 적용한다(intent 2026-10-17).

R-17 띄어쓰기(2026-10-17 오전)와 사용자 오타 사전보다 먼저 만든 결과에는 이 교정이 빠져 있다.
순서: _백업/재교정_{시각}/ 에 원본 보관 → 오타 사전(korean_corrector.TYPO_DICT 단계) → 사용자 오타 사전
→ R-17 띄어쓰기(넣기만) → 임시 파일에 쓴 뒤 교체 → 의심 목록.
글자를 바꾸는 것은 두 사전뿐이고, 띄어쓰기는 R-17 규칙 그대로다. 바뀐 게 없으면 백업·쓰기 모두 하지 않는다.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

from core import spacing, typo_suspects, user_typos

BACKUP_ROOT = Path(__file__).resolve().parent.parent.parent / "_백업"
_PARTIAL_RE = re.compile(r"_(?:정제중|OCR중)(?:_\d{8}_\d{6})?$")


def list_results(results_dir: Path) -> list[Path]:
    """재교정할 수 있는 결과 txt — 이름순, 중간 저장분 제외."""
    if not results_dir.is_dir():
        return []
    return sorted(p for p in results_dir.iterdir()
                  if p.is_file() and p.suffix.lower() == ".txt" and not _PARTIAL_RE.search(p.stem))


def correct(text: str) -> tuple[str, int, int]:
    """(교정본, 사용자 사전으로 바꾼 곳, 넣은 띄어쓰기)."""
    from core.korean_corrector import fix_spelling_and_typos
    text = fix_spelling_and_typos(text)
    text, n_typo = user_typos.apply(text)
    # R-17 은 한 번에 끝나지 않는다 — 띄운 뒤 남은 5자↑ 조각을 다음 번에 또 띄운다.
    # 02_결과 25권 실측: 1회 107,216곳 → 2회 16곳 → 3회 0곳. 누를 때마다 바뀌지 않게 멈출 때까지 돈다.
    n_space = 0
    for _ in range(3):
        text, n = spacing.fix_spacing(text)
        n_space += n
        if not n:
            break
    return text, n_typo, n_space


def _backup_dir() -> Path:
    d = BACKUP_ROOT / f"재교정_{time.strftime('%Y%m%d_%H%M%S')}"
    k = 1
    while d.exists():                       # 같은 초에 또 돌리면 기존 백업을 덮지 않는다
        d = d.with_name(f"{d.name.split('~')[0]}~{k}")
        k += 1
    return d


def reclean_files(paths) -> list[dict]:
    """[{path, changed, typos, spaces, backup, suspects, n_suspects, skipped}] — 파일 하나의 실패가 나머지를 막지 않는다."""
    rows, bdir = [], None
    for path in map(Path, paths):
        row = {"path": path, "changed": False, "typos": 0, "spaces": 0, "backup": None,
               "suspects": None, "n_suspects": 0, "skipped": ""}
        rows.append(row)
        if path.suffix.lower() != ".txt":
            row["skipped"] = "txt 아님"
            continue
        if _PARTIAL_RE.search(path.stem):
            row["skipped"] = "중간 저장분"
            continue
        try:
            src = path.read_text(encoding="utf-8")
            new, row["typos"], row["spaces"] = correct(src)
            # 무손실 확인 — 줄 수가 달라지면 교정이 줄을 먹은 것이다. 원본을 그대로 둔다
            if new != src and new.count("\n") != src.count("\n"):
                row["skipped"] = "줄 수가 달라져 원본 유지"
                continue
            if new != src:
                bdir = bdir or _backup_dir()
                bdir.mkdir(parents=True, exist_ok=True)
                row["backup"] = bdir / path.name
                row["backup"].write_bytes(path.read_bytes())
                tmp = path.with_name(path.name + ".tmp")
                tmp.write_text(new, encoding="utf-8")
                os.replace(tmp, path)       # 쓰다 끊겨도 원본이 반쯤 덮이지 않는다
                row["changed"] = True
            row["suspects"], row["n_suspects"] = typo_suspects.save(path)
        except Exception as e:
            row["skipped"] = f"오류: {type(e).__name__}: {e}"
    return rows
