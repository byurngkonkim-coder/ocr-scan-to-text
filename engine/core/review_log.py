"""최종 검수 기록 — 책마다 OCR 원문 → 쪽 교정·잡음 제거 → 최종 정제본의 글자 수 변화와 정제 보고를 남긴다.

정제 순서를 바꿔 보는 대신 기록만 남기기로 했다(intent 2026-10-18-정제-순서-재배치 §10):
맞춤법 교정을 줄 잇기 뒤로 옮기자 표본 3권 120쪽에서 나빠진 곳 25 · 나아진 곳 4 · 한 문장 소실.
글자 수가 단계 사이에 크게 줄면 그 단계에서 본문이 빠졌다는 신호다 — 정제 전 원문 사본과 대조한다.

기록은 02_결과/_검수기록/{결과 파일 이름}.txt — 하위 폴더라 ② 탭의 '처리 완료' 판정을 흐리지 않는다.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

SUBDIR = "_검수기록"
_CLASSES = (("한글", r"[가-힣]"), ("한자", r"[一-鿿㐀-䶿]"), ("숫자", r"\d"), ("영문", r"[A-Za-z]"))


def count(text: str) -> dict[str, int]:
    c = {name: len(re.findall(pat, text)) for name, pat in _CLASSES}
    c["어절"] = len(text.split())
    return c


def path_for(result_path: Path) -> Path:
    return result_path.parent / SUBDIR / f"{result_path.stem}.txt"


def _row(name: str, c: dict[str, int]) -> str:
    return f"| {name} | " + " | ".join(str(v) for v in c.values()) + " |"


def save(result_path: Path, raw_pages: list[str], page_pages: list[str], final_text: str,
         warnings: list[str]) -> Path:
    raw, page, final = count("\n".join(raw_pages)), count("\n".join(page_pages)), count(final_text)
    lines = [
        f"검수기록 — {result_path.name}",
        f"작성: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"쪽 수: {len(raw_pages)} (빈 쪽 제외)",
        "",
        "## 단계별 글자 수",
        "| 단계 | " + " | ".join(raw) + " |",
        "|---" * (len(raw) + 1) + "|",
        _row("판독 원문", raw),
        _row("쪽 교정·잡음 제거 후", page),
        _row("최종 정제본", final),
        _row("원문→최종 변화", {k: final[k] - raw[k] for k in raw}),
        "",
        "## 정제 보고",
        *([f"- {w}" for w in warnings] or ["- 없음"]),
    ]
    path = path_for(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
