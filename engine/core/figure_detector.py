"""그림·표·수식 감지 및 플레이스홀더 삽입 — OpenCV 기반"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image


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

# 수식 판단용 수학 기호 패턴
_MATH_PATTERN = re.compile(r"[∑∫∂√±×÷≤≥≠∞∈∉⊂⊃∪∩αβγδεζηθλμπρστφχψω]|"
                            r"\b(sin|cos|tan|log|lim|sum|int)\b")

# 그림 후보 안에서 이 이상 글자가 확신도 높게 읽히면 본문으로 보고 마스킹하지 않는다
_WORD_CHARS = re.compile(r"[가-힣A-Za-z0-9]")
_TEXT_MIN_CONF = 60
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
    img_cv = _to_gray_cv(image)
    h, w = img_cv.shape

    regions: list[Region] = []
    fig_idx = tab_idx = form_idx = 1

    # ── 1. 표 감지 ──────────────────────────────────────────────────────────────
    table_boxes = _detect_tables(img_cv)
    for (x, y, bw, bh) in table_boxes:
        crop = image.crop((x, y, x + bw, y + bh))
        md = _extract_table_markdown(crop, lang)
        _, _, boxes = _rough_ocr(crop, lang)
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
        boxes.append((x, y, bw, bh))

    return _merge_overlapping(boxes)


def _extract_table_markdown(crop: Image.Image, lang: str) -> str:
    """표 크롭 이미지에서 Tesseract TSV 모드로 셀 텍스트 추출 → Markdown"""
    try:
        data = pytesseract.image_to_data(
            crop, lang=lang,
            output_type=pytesseract.Output.DICT,
            config="--psm 6",
        )
        # 블록·단락·라인별 그룹핑으로 행 구성
        rows: dict[int, list[str]] = {}
        for i, text in enumerate(data["text"]):
            if not str(text).strip():
                continue
            line_num = data["line_num"][i]
            block_num = data["block_num"][i]
            key = block_num * 1000 + line_num
            rows.setdefault(key, []).append(str(text).strip())

        if not rows:
            return ""

        lines = [" | ".join(cells) for cells in rows.values()]
        if len(lines) < 2:
            return ""

        # 첫 행을 헤더로
        md_lines = [
            f"| {lines[0]} |",
            f"| {' | '.join(['---'] * lines[0].count('|') + ['---'])} |",
        ]
        for line in lines[1:]:
            md_lines.append(f"| {line} |")
        return "\n".join(md_lines)
    except Exception:
        return ""


def _is_formula(text: str) -> bool:
    return bool(_MATH_PATTERN.search(text))


def _rough_ocr(crop: Image.Image, lang: str) -> tuple[str, str, list[tuple[int, int, int, int]]]:
    """(전체 인식 문자열, 신뢰도 60 이상 단어만 이은 문자열, 그 단어들의 상자)

    psm 11(흩어진 글자 탐색) — 그림 위에 놓인 글자처럼 행 구조가 없는 글자도 찾는다.
    글자 줄에 딱 맞게 잘린 영역은 Tesseract가 못 읽으므로 흰 여백을 둘러 읽는다.
    """
    m = _OCR_MARGIN
    framed = Image.new("RGB", (crop.width + 2 * m, crop.height + 2 * m), "white")
    framed.paste(crop.convert("RGB"), (m, m))
    data = pytesseract.image_to_data(
        framed, lang=lang, config="--psm 11", output_type=pytesseract.Output.DICT,
    )
    rough, confident, boxes = [], [], []
    for i, t in enumerate(data["text"]):
        word = str(t).strip()
        if not word:
            continue
        rough.append(word)
        if float(data["conf"][i]) >= _TEXT_MIN_CONF and _WORD_CHARS.search(word):
            confident.append(word)
            boxes.append((data["left"][i] - m, data["top"][i] - m, data["width"][i], data["height"][i]))
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
