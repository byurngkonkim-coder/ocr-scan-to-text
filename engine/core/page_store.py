"""쪽 원문 파일 — 1차(글자 읽기)가 남기는 쪽 단위 결과(intent 2026-11-10 정제-후처리-통합).

후처리(쪽 교정·정제·재처리)가 OCR 없이 다시 돌 수 있게, 판독 원문을 줄 그대로 둔다 — 쪽 교정이 줄을 이어 붙인
원문 사본으로 정제를 다시 하면 차례 인식이 약했다(표본 19권 중 7권에서 제목 1~3개 감소, 2026-11-10 실측).

    {version, …머리(source·total_pages·options 등), pages: [{page_num, raw_text, engine, confidence, figures,
     needs_review, review_reason, is_blank, blank_reason, corrected_text?}]}

corrected_text 는 저장할 때 이미 있던 쪽에만 둔다(유료 교정을 거친 쪽·옛 복구본에서 되살린 쪽 — 다시 만들 수 없는 글자).
비어 있는 쪽은 후처리가 판독 원문에서 다시 교정한다.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from core.types import OCREngine, PageResult

VERSION = 1


def save(path: Path | str, pages: list[PageResult], meta: dict | None = None) -> Path:
    """쪽들을 파일 하나로 — 임시 파일에 쓴 뒤 바꿔치기(쓰다 끊겨도 반쪽 파일이 남지 않게)."""
    rows = []
    for p in pages:
        row = asdict(p)
        row["engine"] = p.engine.value
        if not row["corrected_text"]:
            del row["corrected_text"]
        rows.append(row)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"version": VERSION, **(meta or {}), "pages": rows}, ensure_ascii=False),
                   encoding="utf-8")
    os.replace(tmp, path)
    return path


def load(path: Path | str) -> tuple[list[PageResult], dict]:
    """→ (쪽들, 머리). 읽을 수 없으면 어느 파일인지 알려 주는 ValueError."""
    path = Path(path)
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        pages = [PageResult(**{"corrected_text": "", **row, "engine": OCREngine(row["engine"])})
                 for row in meta.pop("pages")]
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"쪽 원문 파일을 읽을 수 없습니다({path.name}): {e}") from e
    return pages, meta
