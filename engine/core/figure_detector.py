"""그림·표·수식 감지 및 플레이스홀더 삽입 — OpenCV 기반"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core import rapidocr_engine


@dataclass
class Region:
    """감지된 비텍스트 영역"""
    kind: str          # 'figure' | 'table' | 'formula'
    index: int         # 1부터 시작하는 일련번호
    x: int
    y: int
    w: int
    h: int
    crop: Image.Image | None = field(default=None, repr=False)
    table_md: str = ""  # 표인 경우 Markdown 변환 결과
    # 영역 안에서 확실히 읽힌 단어 상자(페이지 좌표) — 마스킹 후 원본으로 되살린다
    text_boxes: list[tuple[int, int, int, int]] = field(default_factory=list, repr=False)


@dataclass
class DetectionResult:
    regions: list[Region] = field(default_factory=list)
    masked_image: Image.Image | None = None  # 비텍스트 영역을 흰색으로 마스킹한 이미지


# 영역 종류별 최소 면적 (px²)
_MIN_FIGURE_AREA = 5_000
_MIN_TABLE_AREA = 3_000
_MIN_FORMULA_AREA = 1_000
# 그림 후보의 최소 높이·너비 (쪽 크기 대비) — 높이가 작으면 글자 줄 조각, 너비가 작으면
# 쪽 가장자리 그림자 띠다(예측기계 실측: 띠 너비 10~11.5%, 진짜 그림 21~60%)
_MIN_FIGURE_H = 0.08
_MIN_FIGURE_W = 0.15

# 수식 판단용 수학 기호 패턴
_MATH_PATTERN = re.compile(r"[∑∫∂√±×÷≤≥≠∞∈∉⊂⊃∪∩αβγδεζηθλμπρστφχψω]|"
                            r"\b(sin|cos|tan|log|lim|sum|int)\b")

# 그림 후보 안에서 이 이상 글자가 확신도 높게 읽히면 본문으로 보고 마스킹하지 않는다
# (글자 판별·표 읽기는 본문과 같은 RapidOCR 한글 v5 — 예전 Tesseract 는 한글 조각을 못 읽어 본문을 지웠다)
_WORD_CHARS = re.compile(r"[가-힣A-Za-z0-9]")
_TEXT_MIN_CONF = 0.6
_TEXT_MIN_CHARS = 4
_TEXT_BOX_PAD = 4   # 되살릴 단어 상자 여유 (획 끝 잘림 방지)
_OCR_MARGIN = 20    # 영역 OCR 시 둘러 줄 흰 여백


def _to_gray_cv(image: Image.Image) -> np.ndarray:
    arr = np.array(image.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def detect(image: Image.Image, lang: str = "kor+eng") -> DetectionResult:
    """
    이미지에서 그림·표·수식 영역을 감지하고 DetectionResult 반환.

    처리 순서:
      1. 표 감지 (수평·수직 선분 격자 패턴)
      2. 그림 감지 (연속 윤곽선 + 면적)
      3. 수식 감지 (수학 기호 밀집 영역)
    """
    # 글자 판별 엔진이 없으면 본문을 그림으로 잘못 지울 수 있다 — 감지하지 않고 원본 그대로 둔다
    if not rapidocr_engine.is_available():
        return DetectionResult(regions=[], masked_image=image.copy())

    img_cv = _to_gray_cv(image)
    h, w = img_cv.shape

    regions: list[Region] = []
    fig_idx = tab_idx = form_idx = 1

    # ── 1. 표 감지 ──────────────────────────────────────────────────────────────
    found = _detect_tables(img_cv)
    table_boxes = [_expand_to_border(img_cv, b, [o for o in found if o != b]) for b in found]
    for (x, y, bw, bh) in table_boxes:
        crop = image.crop((x, y, x + bw, y + bh))
        md = _extract_table_markdown(crop, lang)
        # 행 절반 이상이 한 칸이면 표가 아니다 — 쪽 가장자리 선·도식이 격자로 잡혀 본문 조각을
        # '| 로더정확 |' 처럼 다시 썼다(예측기계 54·107·133·135쪽). 그림 감지로 넘긴다(글자 영역이면 거기서 빠짐)
        # 칸 수는 행마다 같으니(빈 칸 포함) 글자가 든 칸으로 센다
        rows = [l for l in md.splitlines() if not l.startswith("| ---")]
        if rows and sum(sum(1 for c in l.strip("|").split("|") if c.strip()) > 1 for l in rows) * 2 < len(rows):
            continue
        # 표로 바꾼 영역의 글자는 본문에 되살리지 않는다 — 되살리면 본문 OCR 이 표 글자를 한 줄씩
        # 다시 풀어 읽어 표와 이중으로 나왔다(SI운영방안 결과의 숫자 나열). 못 바꾼 표만 글자를 살린다
        boxes = [] if md else _rough_ocr(crop, lang)[2]
        regions.append(Region(
            kind="table", index=tab_idx,
            x=x, y=y, w=bw, h=bh,
            crop=crop, table_md=md, text_boxes=_offset(boxes, x, y),
        ))
        tab_idx += 1

    # ── 2. 그림 감지 ────────────────────────────────────────────────────────────
    # 표로 이미 감지된 영역은 마스킹 후 그림 탐색
    masked_cv = img_cv.copy()
    for r in regions:
        cv2.rectangle(masked_cv, (r.x, r.y), (r.x + r.w, r.y + r.h), 255, -1)

    figure_boxes = _detect_figures(masked_cv, image)
    for (x, y, bw, bh) in figure_boxes:
        # 표 영역과 겹치면 제외
        if _overlaps_any(x, y, bw, bh, regions):
            continue
        crop = image.crop((x, y, x + bw, y + bh))
        rough_text, confident_text, boxes = _rough_ocr(crop, lang)
        # 본문 행이 윤곽선 팽창으로 그림 후보가 된다 — 마스킹하면 문장이 잘려 나간다
        if _is_text_region(confident_text):
            continue
        # 수식인지 먼저 판별
        if _is_formula(rough_text) and bw * bh >= _MIN_FORMULA_AREA:
            regions.append(Region(
                kind="formula", index=form_idx,
                x=x, y=y, w=bw, h=bh, crop=crop,
            ))
            form_idx += 1
        elif bw * bh >= _MIN_FIGURE_AREA:
            regions.append(Region(
                kind="figure", index=fig_idx,
                x=x, y=y, w=bw, h=bh, crop=crop, text_boxes=_offset(boxes, x, y),
            ))
            fig_idx += 1

    # ── 마스킹 이미지 생성 ─────────────────────────────────────────────────────
    masked_pil = _mask_regions(image, regions)

    return DetectionResult(regions=regions, masked_image=masked_pil)


def save_crops(regions: list[Region], out_dir: str | Path, stem: str) -> dict[str, Path]:
    """
    감지된 영역 크롭 이미지를 out_dir/figures/ 에 저장.
    반환값: {placeholder → 저장경로}
    """
    out = Path(out_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)

    mapping: dict[str, Path] = {}
    for r in regions:
        if r.crop is None:
            continue
        label = _region_label(r)
        filename = f"{stem}_{r.kind}{r.index}.png"
        path = out / filename
        r.crop.save(str(path))
        mapping[label] = path
    return mapping


TABLE_TOKEN_RE = re.compile(r"⟨TABLE:(\d+)-(\d+)⟩")


def table_token(page_num: int, index: int) -> str:
    """본문 안 표 자리 표지 — 저장 직전 writer.place_tables 가 Markdown 표로 바꾼다."""
    return f"⟨TABLE:{page_num}-{index}⟩"


def build_placeholder(r: Region) -> str:
    """Region → 플레이스홀더 문자열"""
    if r.kind == "figure":
        return f"[그림 {r.index}: 이미지 감지됨]"
    if r.kind == "table":
        if r.table_md:
            return f"[표 {r.index}]\n{r.table_md}"
        return f"[표 {r.index}: 표 감지됨]"
    if r.kind == "formula":
        return f"[수식 {r.index}: OCR 인식 불가]"
    return f"[{r.kind} {r.index}]"


# ── 내부 감지 함수 ─────────────────────────────────────────────────────────────

def _detect_tables(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """수평·수직 선분 격자 패턴으로 표 영역 감지"""
    h, w = gray.shape

    # 이진화
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 수평선 커널
    horiz_len = max(40, w // 8)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)

    # 수직선 커널
    vert_len = max(40, h // 8)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)

    # 격자 합성
    grid = cv2.add(horiz, vert)
    dilated = cv2.dilate(grid, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw * bh >= _MIN_TABLE_AREA and bw > 50 and bh > 30:
            boxes.append((x, y, bw, bh))

    return _merge_overlapping(boxes)


def _expand_to_border(gray: np.ndarray, box: tuple[int, int, int, int],
                      others: list[tuple[int, int, int, int]] = ()) -> tuple[int, int, int, int]:
    """감지한 표 상자를 근처의 흐린 바깥 테두리까지 넓힌다.

    Otsu 이진화는 연회색 선을 놓친다 — 150dpi 렌더(대시보드 기본)에서 SI운영방안 p5 인벤토리 표가
    왼쪽 테두리 60px·합계 행을 빼고 잡혀, 프로젝트명 앞 글자와 합계 행이 본문으로 새었다(2026-10-19).
    상자 밖 15% 안에서 상자 변 길이 절반 이상 이어진 선이 있으면 가장 바깥 선까지 넓힌다.
    옆에 다른 표 상자가 있으면 그 상자 앞에서 멈춘다(p5 인벤토리 바로 위 제목 띠)."""
    x, y, w, h = box
    H, W = gray.shape
    mx, my = int(w * 0.15), int(h * 0.15)
    x0, x1, y0, y1 = max(0, x - mx), min(W, x + w + mx), max(0, y - my), min(H, y + h + my)
    for ox, oy, ow, oh in others:   # 넓힐 범위를 이웃 상자 바로 앞까지로 줄인다
        if ox < x + w and x < ox + ow:
            if oy + oh <= y:
                y0 = max(y0, oy + oh + 1)
            elif oy >= y + h:
                y1 = min(y1, oy - 1)
        if oy < y + h and y < oy + oh:
            if ox + ow <= x:
                x0 = max(x0, ox + ow + 1)
            elif ox >= x + w:
                x1 = min(x1, ox - 1)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    vert = cv2.morphologyEx(binary[y:y + h, x0:x1], cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 4))))
    horiz = cv2.morphologyEx(binary[y0:y1, x:x + w], cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 4), 1)))
    cols = np.where(vert.sum(0) / 255 >= h * 0.5)[0] + x0
    rows = np.where(horiz.sum(1) / 255 >= w * 0.5)[0] + y0
    left = min([x] + [c for c in cols if c < x])
    right = max([x + w] + [c + 1 for c in cols if c >= x + w])
    top = min([y] + [r for r in rows if r < y])
    bottom = max([y + h] + [r + 1 for r in rows if r >= y + h])
    return left, top, right - left, bottom - top


def _detect_figures(gray: np.ndarray, orig: Image.Image) -> list[tuple[int, int, int, int]]:
    """연속 윤곽선 + 컬러 분석으로 그림·사진 영역 감지"""
    # 컬러 이미지에서 채도 높은 영역 감지
    orig_cv = cv2.cvtColor(np.array(orig.convert("RGB")), cv2.COLOR_RGB2HSV)
    saturation = orig_cv[:, :, 1]
    _, sat_mask = cv2.threshold(saturation, 30, 255, cv2.THRESH_BINARY)

    # 윤곽선 감지
    edges = cv2.Canny(gray, 30, 100)
    dilated = cv2.dilate(edges, np.ones((8, 8), np.uint8), iterations=3)
    combined = cv2.bitwise_or(dilated, sat_mask)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h_img, w_img = gray.shape
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        # 너무 작거나 페이지 전체에 가까운 영역 제외
        if area < _MIN_FIGURE_AREA:
            continue
        if bw > w_img * 0.95 and bh > h_img * 0.95:
            continue
        # 텍스트 행처럼 넓고 얇은 영역 제외
        if bh < 20 or (bw / bh > 15):
            continue
        # 예측기계 스캔본 실측: 본문 쪽마다 한 줄 높이(쪽 높이의 2~5%) 조각이 40~70개 그림 후보가 돼
        # 줄이 지워졌고, 가장자리 띠는 줄 끝 글자를 지웠다. 진짜 그림은 높이 13~34%·너비 21~60%였다
        if bh < h_img * _MIN_FIGURE_H or bw < w_img * _MIN_FIGURE_W:
            continue
        boxes.append((x, y, bw, bh))

    return _merge_overlapping(boxes)


def _extract_table_markdown(crop: Image.Image, lang: str) -> str:
    """표 크롭 이미지에서 RapidOCR 로 글자 상자를 읽어 행·열 칸에 넣고 Markdown 표로.

    예전엔 「세로로 겹치는 상자 = 같은 행」으로 이어 붙였는데, 기울어진 스캔에서 겹침이 연쇄돼
    15행 표가 2행으로 뭉쳤고 열은 상자 순서라 행마다 칸 수가 달랐다(SI운영방안 p1·p5, 2026-10-19).
    스캔 격자선은 끊기거나 흐려(p5 인벤토리 내부선 0개) 선만으로 칸을 못 나눈다 → 글자 배치가 기준,
    선은 「그 사이 줄을 한 칸으로 합치지 않는다」·「열 경계」 보조로만 쓴다.
    """
    try:
        boxes = _read_table_boxes(crop, lang)
        if not boxes:
            return ""
        unit = sorted(b[3] for _t, b in boxes)[len(boxes) // 2]   # 글자 높이(중앙값)
        h_lines, v_lines = _grid_segments(_to_gray_cv(crop), unit)
        lines = _text_lines(boxes, unit)
        cols = _column_bounds(lines, v_lines, crop.width, unit)
        rows = _group_rows(lines, cols, h_lines, unit)
        if len(rows) < 2:
            return ""
        grid = [_fill_cells(row, cols) for row in rows]
        keep = [j for j in range(len(cols)) if any(r[j] for r in grid)]   # 막대·화살표만 있던 칸(간트) 빼기
        grid = [[r[j] for j in keep] for r in grid if any(r)]
        _fold_lone_columns(grid)
        if len(grid) < 2:
            return ""
        md = ["| " + " | ".join(c.replace("|", "/") for c in r) + " |" for r in grid]
        md.insert(1, "| " + " | ".join(["---"] * len(grid[0])) + " |")
        return "\n".join(md)
    except Exception:
        return ""


# 대시보드 렌더(150dpi) 표 칸 글자는 RapidOCR 이 낱말 조각으로 끊어 읽었다(SI운영방안 p2 설명 칸) — 키워서 읽는다.
# 1.5배는 p2 가 여전히 조각났고 2배에서 4행 모두 칸 단위로 읽혔다(2026-10-19)
_TABLE_OCR_SCALE = 2.0


def _read_table_boxes(crop: Image.Image, lang: str) -> list[tuple[str, tuple[int, int, int, int]]]:
    s = _TABLE_OCR_SCALE
    big = crop.resize((round(crop.width * s), round(crop.height * s)), Image.LANCZOS)
    return [(t.strip(), (round(x / s), round(y / s), round(w / s), round(h / s)))
            for t, _sc, (x, y, w, h) in _read_framed(big, lang) if t and t.strip()]


def _grid_segments(gray: np.ndarray, unit: int) -> tuple[list[tuple[int, int, int]], list[int]]:
    """표 안 가로선 조각 (y, x0, x1) 과 세로선 x 위치. 흐린 스캔 선도 잡게 적응형 이진화.
    세로선은 글자 높이의 1.3배 이상만 — 그보다 짧으면 글자 획(ㅣ·l)이다."""
    h, w = gray.shape
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 20), 1)))
    v_len = max(20, round(unit * 1.3))
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len)))
    h_segs = []
    for cnt in cv2.findContours(horiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, bw, bh = cv2.boundingRect(cnt)
        h_segs.append((y + bh // 2, x, x + bw))
    v_xs = []
    for cnt in cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, _y, bw, _bh = cv2.boundingRect(cnt)
        v_xs.append(x + bw // 2)
    return h_segs, sorted(v_xs)


def _text_lines(boxes, unit: int) -> list[list]:
    """글자 상자를 가운데 높이로 줄로 묶는다 → [[가운데 y, [(글자, 상자), ...]], ...] (위에서 아래로)."""
    lines: list[list] = []
    for t, b in sorted(boxes, key=lambda tb: tb[1][1] + tb[1][3] / 2):
        cy = b[1] + b[3] / 2
        if lines and cy - lines[-1][0] < unit * 0.5:
            lines[-1][1].append((t, b))
            lines[-1][0] = sum(bb[1] + bb[3] / 2 for _t, bb in lines[-1][1]) / len(lines[-1][1])
        else:
            lines.append([cy, [(t, b)]])
    return lines


def _group_rows(lines, cols, h_lines, unit: int) -> list[list]:
    """줄을 행으로. 촘촘한 데이터 행(SI운영방안 p5: 줄 간격이 글자 높이의 1.3배)과 한 칸 안 여러 줄을
    간격으로는 못 가른다 → 「칸 절반 이상을 채운 줄」은 새 행, 몇 칸만 채운 줄(설명 칸 이어지는 줄·
    2단 머리글의 윗줄·아랫줄)은 가로선이 사이에 없고 가까우면 이웃 행에 붙인다."""
    def filled(items) -> int:
        return sum(1 for c in _fill_cells(items, cols) if c)

    def ruled(y0, y1) -> bool:
        return any(min(y0, y1) < ly < max(y0, y1) for ly, _x0, _x1 in h_lines)

    def joins(group, cy) -> bool:
        return not ruled(group[1], cy) and cy - group[1] < unit * 2

    full = max(filled(items) for _cy, items in lines)
    rows: list[list] = []        # [위 가운데 y, 아래 가운데 y, 상자들]
    pending: list[list] = []     # 아직 칸 절반을 못 채운 줄 묶음 — 다음 행 위에 붙인다(2단 머리글 윗줄)
    for cy, items in lines:
        if filled(items) * 2 < full:
            # 대기 묶음 → 앞 행 순으로 붙이고, 못 붙으면(첫 줄·세로 병합 칸 이름 「목표 및」처럼
            # 선에 걸친 줄) 새 대기 묶음
            target = pending[-1] if pending else rows[-1] if rows else None
            if target is not None and joins(target, cy):
                target[1] = cy
                target[2].extend(items)
            else:
                pending.append([cy, cy, list(items)])
            continue
        top = cy
        if pending and joins(pending[-1], cy):
            last = pending.pop()
            top, items = last[0], last[2] + items
        rows += pending
        pending = []
        rows.append([top, cy, list(items)])
    rows += pending
    return [items for _t, _b, items in sorted(rows, key=lambda r: r[0])]


def _column_bounds(lines, v_lines, width: int, unit: int) -> list[tuple[int, int]]:
    """열 경계 — 거의 모든 줄에서 글자가 없는 세로 띠와 세로선. 여러 열에 걸친 머리글 줄은 띠를 막지 않게 뺀다."""
    cover = np.zeros(width + 1, dtype=np.int32)
    for _cy, items in lines:
        for _t, (x, _y, w, _h) in items:
            cover[max(0, x):min(width, x + w) + 1] += 1
    # 줄의 30% 이하만 걸친 띠도 경계 — 여러 열에 걸친 머리글 2줄(SI운영방안 p1 「SI수행」「Project수행팀」)이
    # 그 아래 「가득률 | 70%+」 경계를 가렸다. 단 그런 띠 안에 글자가 전혀 없는 곳이 있으면 거기서 자른다
    # (머리글에만 있는 「고객」 열이 넓은 띠에 묻혀 옆 열과 합쳐졌다, p5).
    limit = max(1, round(len(lines) * 0.35))
    cuts = list(v_lines)
    x = 0
    while x <= width:
        if cover[x] > limit:
            x += 1
            continue
        start = x
        while x <= width and cover[x] <= limit:
            x += 1
        zeros, z = [], start
        while z < x:
            if cover[z] == 0:
                z0 = z
                while z < x and cover[z] == 0:
                    z += 1
                zeros.append((z0 + z) // 2)
            z += 1
        cuts += zeros or [(start + x) // 2]
    # 가까운 경계는 하나로 — 세로선 옆 여백이 따로 잡히면 칸이 한 칸씩 어긋났다(p2). 세로선을 우선한다
    merged: list[int] = []
    for c in sorted(c for c in cuts if 0 < c < width):
        if merged and c - merged[-1] < unit:
            if c in v_lines:
                merged[-1] = c
            continue
        merged.append(c)
    bounds = [0] + merged + [width]
    return list(zip(bounds[:-1], bounds[1:]))


def _fold_lone_columns(grid: list[list[str]]) -> None:
    """한 행에만 글자가 있고 그 왼쪽 칸이 비어 있는 열은 왼쪽으로 붙인다 — 프로젝트명 칸 오른쪽에 붙은
    「합계」가 따로 열 하나를 만들었다(p5). 머리글에만 있는 열(「고객」)은 왼쪽 칸이 차 있어 그대로 둔다."""
    j = len(grid[0]) - 1
    while j > 0:
        used = [r for r in grid if r[j]]
        if len(used) == 1 and not used[0][j - 1]:
            used[0][j - 1], used[0][j] = used[0][j], ""
            for r in grid:
                del r[j]
        j -= 1


def _fill_cells(row, cols) -> list[str]:
    """상자를 칸에 넣는다 — 상자가 시작하는 칸(병합 칸·긴 프로젝트명은 첫 칸). 시작 칸에 30% 미만만
    걸쳤으면 가운데가 든 칸. 가운데 기준만 쓰면 긴 이름이 이름 열 안의 약한 경계를 넘어 옆 열로 갔다(p5)."""
    cells: list[list] = [[] for _ in cols]
    for t, (x, y, w, _h) in row:
        start = next((j for j, (a, b) in enumerate(cols) if a <= x < b), 0)
        if min(cols[start][1], x + w) - x >= w * 0.3:
            j = start
        else:
            cx = x + w / 2
            j = next((j for j, (a, b) in enumerate(cols) if a <= cx < b), len(cols) - 1)
        cells[j].append((y, x, t.strip()))
    return [" ".join(t for _y, _x, t in sorted(c)) for c in cells]


def _is_formula(text: str) -> bool:
    return bool(_MATH_PATTERN.search(text))


def _read_framed(crop: Image.Image, lang: str) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """흰 여백을 둘러 RapidOCR 로 읽고 상자를 crop 좌표로 되돌린다 — 글자 줄에 딱 맞게 잘린 영역도 읽히게."""
    m = _OCR_MARGIN
    framed = Image.new("RGB", (crop.width + 2 * m, crop.height + 2 * m), "white")
    framed.paste(crop.convert("RGB"), (m, m))
    return [(t, s, (x - m, y - m, w, h)) for t, s, (x, y, w, h) in rapidocr_engine.read_boxes(framed, lang)]


def _rough_ocr(crop: Image.Image, lang: str) -> tuple[str, str, list[tuple[int, int, int, int]]]:
    """(전체 인식 문자열, 신뢰도 0.6 이상 줄만 이은 문자열, 그 줄들의 상자)"""
    rough, confident, boxes = [], [], []
    for text, score, box in _read_framed(crop, lang):
        if not text:
            continue
        rough.append(text)
        if score >= _TEXT_MIN_CONF and _WORD_CHARS.search(text):
            confident.append(text)
            boxes.append(box)
    return " ".join(rough), " ".join(confident), boxes


def _offset(boxes: list[tuple[int, int, int, int]], dx: int, dy: int) -> list[tuple[int, int, int, int]]:
    return [(x + dx, y + dy, w, h) for x, y, w, h in boxes]


def _is_text_region(confident_text: str) -> bool:
    """그림 잡음이 아닌 본문 글자가 충분히 읽히면 텍스트 영역"""
    return len(_WORD_CHARS.findall(confident_text)) >= _TEXT_MIN_CHARS


def _region_label(r: Region) -> str:
    return build_placeholder(r).split(":")[0].strip("[]")


def _overlaps_any(x: int, y: int, w: int, h: int, regions: list[Region]) -> bool:
    for r in regions:
        if _rect_overlap(x, y, w, h, r.x, r.y, r.w, r.h):
            return True
    return False


def _rect_overlap(ax, ay, aw, ah, bx, by, bw, bh) -> bool:
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def _merge_overlapping(boxes: list[tuple]) -> list[tuple]:
    """겹치는 바운딩박스를 하나로 합침"""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[0] * 10000 + b[1])
    merged = [list(boxes[0])]
    for x, y, w, h in boxes[1:]:
        last = merged[-1]
        if _rect_overlap(x, y, w, h, last[0], last[1], last[2], last[3]):
            nx = min(last[0], x)
            ny = min(last[1], y)
            merged[-1] = [
                nx, ny,
                max(last[0] + last[2], x + w) - nx,
                max(last[1] + last[3], y + h) - ny,
            ]
        else:
            merged.append([x, y, w, h])
    return [tuple(b) for b in merged]


def _mask_regions(image: Image.Image, regions: list[Region]) -> Image.Image:
    """감지된 영역을 흰색으로 마스킹한 이미지 반환 (텍스트만 남김)

    영역 안에서 읽힌 단어 상자는 원본 픽셀로 되살린다 — 그림·표 위에 얹힌 글자 보존.
    """
    orig = np.array(image.convert("RGB"))
    arr = orig.copy()
    for r in regions:
        arr[r.y:r.y + r.h, r.x:r.x + r.w] = 255
    pad = _TEXT_BOX_PAD
    for r in regions:
        for x, y, w, h in _merge_rows(r.text_boxes):
            y0, x0 = max(0, y - pad), max(0, x - pad)
            arr[y0:y + h + pad, x0:x + w + pad] = orig[y0:y + h + pad, x0:x + w + pad]
    return Image.fromarray(arr)


def _merge_rows(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """세로로 겹치는 단어 상자를 한 줄 띠로 합친다 — 확실히 읽힌 단어 사이의 글자도 되살리기 위해"""
    rows: list[list[int]] = []
    for x, y, w, h in sorted(boxes, key=lambda b: b[1]):
        for row in rows:
            if y < row[1] + row[3] and row[1] < y + h:  # 세로 겹침 = 같은 줄
                nx, ny = min(row[0], x), min(row[1], y)
                row[2] = max(row[0] + row[2], x + w) - nx
                row[3] = max(row[1] + row[3], y + h) - ny
                row[0], row[1] = nx, ny
                break
        else:
            rows.append([x, y, w, h])
    return [tuple(r) for r in rows]
