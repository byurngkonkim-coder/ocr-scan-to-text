"""표 구조 인식 — RapidTable(SLANETPLUS) + RapidOCR 한글 v5 로 표 이미지를 2차원 격자로 만든다.

MyOCR engine/table_engine.py 에서 옮겼다(2026-10-17). 같은 표 12장 측정에서 예전 방식(글자 상자를
세로 겹침으로 행에 묶기)은 정답 9장 중 7장, 이 방식은 10장 중 10장 정확했다.
- 칸 상자 열 기준 보정(_align_cell_boxes): 머리글 첫 칸 상자가 둘째 열까지 덮어 '이름 부서' 가 합쳐지던 문제
- 맨 위 빈 행 제거(_drop_leading_empty_rows): 선 없는 표에서 머리글 위에 빈 행이 하나 더 나오던 문제
rapid_table 이 없으면 is_available() 이 False — 호출부가 예전 방식으로 대신한다.
"""
from __future__ import annotations

from html.parser import HTMLParser

import numpy as np
from PIL import Image

from core import rapidocr_engine

_PIPES: dict = {}


def is_available() -> bool:
    try:
        import rapid_table  # noqa: F401
        return True
    except Exception:
        return False


def recognize(image: Image.Image, lang: str = "kor+eng") -> list[list[str]]:
    """표 이미지 → 격자(행마다 같은 열 수). 표를 못 찾으면 빈 목록."""
    pipe = _pipe(lang)
    res = pipe(_upscale_if_small(np.array(image.convert("RGB"))))
    grids = [_drop_leading_empty_rows(html_table_to_grid(h)) for h in (res.pred_htmls or []) if h]
    grids = [g for g in grids if g]
    return max(grids, key=lambda g: sum(len(r) for r in g)) if grids else []


def _pipe(lang: str):
    if lang not in _PIPES:
        from rapid_table import RapidTable
        from rapid_table.utils import RapidTableInput

        pipe = RapidTable(cfg=RapidTableInput(ocr_params=rapidocr_engine.engine_params(lang)))
        structure = pipe.table_structure

        def aligned_structure(imgs):
            # 글자 상자를 칸에 짝짓기 전에 칸 상자를 열 기준으로 보정
            pred_structures, cell_bboxes = structure(imgs)
            logic = pipe.table_matcher.decode_logic_points(pred_structures)
            return pred_structures, [_align_cell_boxes(b, l) for b, l in zip(cell_bboxes, logic)]

        pipe.table_structure = aligned_structure
        _PIPES[lang] = pipe
    return _PIPES[lang]


def _upscale_if_small(rgb: np.ndarray, min_side: int = 1000, max_scale: float = 4.0) -> np.ndarray:
    """저해상도 표는 칸 검출이 어긋나므로 짧은 변이 min_side 가 되도록 확대."""
    h, w = rgb.shape[:2]
    short = min(h, w)
    if short >= min_side:
        return rgb
    scale = min(max_scale, min_side / short)
    return np.array(Image.fromarray(rgb).resize((int(round(w * scale)), int(round(h * scale))), Image.LANCZOS))


def _align_cell_boxes(boxes, logic):
    """칸 상자의 좌우 가장자리를 같은 논리 열 칸들의 중앙값으로 맞춘다.

    SLANETPLUS 는 칸의 행·열(logic: [행시작, 행끝, 열시작, 열끝])은 맞게 내면서 상자 좌표를
    어긋나게 낼 때가 있다. 같은 열 칸들은 가장자리가 같으므로 다수결로 바로잡는다. 세로 범위는 그대로.
    """
    boxes = np.asarray(boxes, dtype=float)
    logic = np.asarray(logic)
    if boxes.ndim != 2 or len(boxes) == 0 or logic.shape != (len(boxes), 4):
        return boxes
    x0, x1 = boxes[:, 0::2].min(1), boxes[:, 0::2].max(1)
    y0, y1 = boxes[:, 1::2].min(1), boxes[:, 1::2].max(1)
    left = {c: np.median(x0[logic[:, 2] == c]) for c in set(logic[:, 2].tolist())}
    right = {c: np.median(x1[logic[:, 3] == c]) for c in set(logic[:, 3].tolist())}
    out = boxes.copy()
    for i, (_, _, c0, c1) in enumerate(logic.tolist()):
        lx, rx = left[c0], right[c1]
        if lx >= rx:
            continue
        out[i] = [lx, y0[i], rx, y0[i], rx, y1[i], lx, y1[i]] if boxes.shape[1] == 8 else [lx, y0[i], rx, y1[i]]
    return out


def _drop_leading_empty_rows(grid: list[list[str]]) -> list[list[str]]:
    """맨 위 글자 없는 행을 지운다. 중간·맨 아래 빈 행은 빈 양식 칸일 수 있어 남긴다."""
    i = 0
    while i < len(grid) and not any(c.strip() for c in grid[i]):
        i += 1
    return grid[i:]


class _TableHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, int, int]]] = []
        self._row: list[tuple[str, int, int]] | None = None
        self._in_cell = False
        self._buf: list[str] = []
        self._span = (1, 1)

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            a = dict(attrs)
            self._in_cell = True
            self._buf = []
            self._span = (_int(a.get("colspan"), 1), _int(a.get("rowspan"), 1))
        elif tag == "br" and self._in_cell:
            self._buf.append(" ")

    def handle_data(self, data):
        if self._in_cell:
            self._buf.append(data)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ("td", "th") and self._in_cell:
            self._row.append((" ".join("".join(self._buf).split()), self._span[0], self._span[1]))
            self._in_cell = False
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def _int(value, default):
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def html_table_to_grid(html: str) -> list[list[str]]:
    """HTML <table> → 2차원 격자. colspan/rowspan 은 좌상단 칸에 글자, 나머지는 빈 문자열."""
    parser = _TableHTMLParser()
    parser.feed(html or "")
    grid: list[list[str]] = []
    occupied: set[tuple[int, int]] = set()
    for r, row in enumerate(parser.rows):
        c = 0
        for text, colspan, rowspan in row:
            while (r, c) in occupied:
                c += 1
            for dr in range(rowspan):
                for dc in range(colspan):
                    rr, cc = r + dr, c + dc
                    while len(grid) <= rr:
                        grid.append([])
                    while len(grid[rr]) <= cc:
                        grid[rr].append("")
                    if dr == 0 and dc == 0:
                        grid[rr][cc] = text
                    else:
                        occupied.add((rr, cc))
            c += colspan
    max_cols = max((len(row) for row in grid), default=0)
    for row in grid:
        row.extend([""] * (max_cols - len(row)))
    return grid
