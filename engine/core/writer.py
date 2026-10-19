"""TXT / MD 출력 — UTF-8, 페이지 구분자·플레이스홀더 포함"""
import re
from pathlib import Path

from core.figure_detector import TABLE_TOKEN_RE, table_token
from core.types import FileResult, OutputFormat

PAGE_SEPARATOR_TXT = "─" * 60
PAGE_SEPARATOR_MD = "---"
REVIEW_MARKER_TXT = "⚠ [검토필요]"
REVIEW_MARKER_MD = "> ⚠ **검토 필요**"


def write(result: FileResult, output_dir: str | Path, fmt: OutputFormat = OutputFormat.TXT) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem     = result.source_path.stem
    out_path = out_dir / f"{stem}.{fmt.value}"
    content  = _build_md(result) if fmt == OutputFormat.MD else _build_txt(result)
    out_path.write_text(content, encoding="utf-8")
    result.output_path = out_path
    return out_path


def write_plain(source_path: str | Path, text: str, output_dir: str | Path,
                fmt: OutputFormat = OutputFormat.TXT) -> Path:
    """
    정제(교정) 완료 본문을 '마커 없는' 순수 플레인텍스트로 저장.

    write()가 붙이는 [페이지 N] 헤더·구분선을 넣지 않고, 정제기가 돌려준
    본문(단락=빈 줄)을 그대로 기록한다. 파일명은 write()와 동일 규칙.
    """
    src      = Path(source_path)
    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src.stem}.{fmt.value}"
    body = text if text.endswith("\n") else text + "\n"
    out_path.write_text(body, encoding="utf-8")
    return out_path


# ── TXT 빌더 ──────────────────────────────────────────────────────────────────

def _build_txt(result: FileResult) -> str:
    content_pages = [p for p in result.pages if not p.is_blank]
    blank_pages   = [p for p in result.pages if p.is_blank]
    review_pages  = [p for p in content_pages if p.needs_review]
    parts: list[str] = []

    # 빈 페이지 건너뜀 안내
    if blank_pages:
        parts += _blank_summary_txt(blank_pages, result.total_pages)
        parts.append("")

    # 검토 필요 요약
    if review_pages:
        parts += _review_summary_txt(review_pages, result.total_pages)
        parts.append("")

    for page in content_pages:
        header = f"[페이지 {page.page_num}]"
        if page.needs_review:
            header += f"  {REVIEW_MARKER_TXT} {page.review_reason}"
        parts.append(header)
        parts.append(PAGE_SEPARATOR_TXT)
        parts.append(place_tables(page.corrected_text, [page]))
        others = _non_tables(page)
        if others:
            parts.append("")
            parts.append("── 비텍스트 영역 ──")
            for ph in others:
                parts.append(ph)
        parts.append("")
    return "\n".join(parts).strip() + "\n"


def _blank_summary_txt(blank_pages, total: int) -> list[str]:
    page_nums = ", ".join(str(p.page_num) for p in blank_pages)
    return [
        "⬜ 빈 페이지 건너뜀 안내",
        PAGE_SEPARATOR_TXT,
        f"전체 {total}페이지 중 {len(blank_pages)}페이지가 빈 페이지로 감지되어 OCR을 건너뜁니다.",
        f"  건너뜀 페이지: {page_nums}",
        PAGE_SEPARATOR_TXT,
    ]


def _review_summary_txt(review_pages, total: int) -> list[str]:
    lines = [
        "★ OCR 품질 검토 필요 안내 ★",
        PAGE_SEPARATOR_TXT,
        f"전체 {total}페이지 중 {len(review_pages)}페이지의 OCR 정확도가 낮습니다.",
        "아래 페이지는 원본 문서와 대조하여 수동으로 확인해 주세요.",
        "",
    ]
    for p in review_pages:
        lines.append(f"  · 페이지 {p.page_num}: {p.review_reason}")
    lines += [
        "",
        "※ 정확도를 높이려면 Vision(고정확) 엔진 API 키를 설정하세요.",
        "   설정: python ocr_tool.py setup  (또는 웹 '🔑 API 키·사용량' 탭)",
        PAGE_SEPARATOR_TXT,
    ]
    return lines


# ── MD 빌더 ───────────────────────────────────────────────────────────────────

def _build_md(result: FileResult) -> str:
    filename      = result.source_path.name
    content_pages = [p for p in result.pages if not p.is_blank]
    blank_pages   = [p for p in result.pages if p.is_blank]
    review_pages  = [p for p in content_pages if p.needs_review]
    total_pages   = result.total_pages

    parts: list[str] = [
        f"# {filename}",
        "",
        f"- 총 페이지: {total_pages}",
        f"- OCR 처리: {len(content_pages)}p  |  빈 페이지 건너뜀: {len(blank_pages)}p",
        f"- 출처: {result.source_path}",
        "",
    ]

    if blank_pages:
        page_nums = ", ".join(f"p.{p.page_num}" for p in blank_pages)
        parts += [
            f"> ⬜ 빈 페이지 {len(blank_pages)}개 건너뜀 ({page_nums})",
            "",
        ]

    if review_pages:
        parts += _review_summary_md(review_pages, total_pages)
        parts.append("")

    parts += [PAGE_SEPARATOR_MD, ""]

    for page in content_pages:
        parts.append(f"## 페이지 {page.page_num}")
        parts.append("")

        if page.needs_review:
            parts += [
                REVIEW_MARKER_MD,
                f"> 사유: {page.review_reason}",
                "> 이 페이지는 원본 문서와 대조하여 확인하세요.",
                "",
            ]

        parts.append(place_tables(page.corrected_text, [page]))
        parts.append("")

        others = _non_tables(page)
        if others:
            parts.append("> **비텍스트 영역**")
            parts.append(">")
            for ph in others:
                parts.append(f"> {ph}")
            parts.append("")

        parts.append(PAGE_SEPARATOR_MD)
        parts.append("")

    return "\n".join(parts)


def _review_summary_md(review_pages, total: int) -> list[str]:
    page_list = ", ".join(f"p.{p.page_num}" for p in review_pages)
    return [
        "> ### ⚠ OCR 품질 검토 필요",
        f"> 전체 {total}페이지 중 **{len(review_pages)}페이지** ({page_list})의 OCR 정확도가 낮습니다.",
        "> 원본 문서와 대조하여 수동으로 확인해 주세요.",
        ">",
        "> 정확도 향상: `python ocr_tool.py setup` 으로 Vision(고정확) 폴백 엔진을 활성화하세요.",
    ]


_TABLE_PH = re.compile(r"\[표 (\d+)\]\n")


def _page_tables(page) -> dict[str, str]:
    """쪽의 표 자리표시 '[표 N]\\n| … |' → {본문 표지: Markdown 표}"""
    out = {}
    for ph in page.figures:
        m = _TABLE_PH.match(ph)
        if m:
            out[table_token(page.page_num, int(m.group(1)))] = ph[m.end():]
    return out


def _non_tables(page) -> list[str]:
    """표로 바꾼 것(본문에 들어감)을 뺀 비텍스트 자리표시 — 그림·수식·못 읽은 표"""
    return [ph for ph in page.figures if not _TABLE_PH.match(ph)]


def place_tables(text: str, pages) -> str:
    """본문의 표 자리 표지를 Markdown 표로 바꾼다. 표지가 사라진 표(Vision 으로 다시 읽은 쪽·정제가 지운 표지)는
    글자를 잃지 않게 끝에 「[표 N (p.쪽)]」로 붙인다."""
    tables: dict[str, str] = {}
    for p in pages:
        tables.update(_page_tables(p))
    used: set[str] = set()

    def sub(m: re.Match) -> str:
        md = tables.get(m.group(0))
        if md is None:
            return ""
        used.add(m.group(0))
        return f"\n\n{md}\n\n"

    text = TABLE_TOKEN_RE.sub(sub, text)
    rest = [f"[표 {TABLE_TOKEN_RE.match(t).group(2)} (p.{TABLE_TOKEN_RE.match(t).group(1)})]\n{md}"
            for t, md in tables.items() if t not in used]
    if rest:
        text = text.rstrip() + "\n\n" + "\n\n".join(rest)
    return _clean(text)


def _clean(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()
