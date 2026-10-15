"""스캔 PDF 머리글·바닥글 여백 자동 측정 → 여백을 가리고 본문만 인식.

고정 비율(enable_margin_filter, 위·아래 6%)은 책·스캔마다 본문과 쪽 꼬리 사이가 달라
닿지 않거나 본문을 자른다(인수인계서 §8: 예측기계는 본문 끝과 쪽 꼬리 사이 3.5%).
그래서 파일마다 먼저 잰다.

  1. 표본 쪽: 앞부분(표지·그림·목차)과 끝부분(색인·광고)을 건너뛴 가운데 쪽에서 고른다.
  2. 줄 상자: RapidOCR 검출만(글자 인식 없음, 쪽당 약 0.5초).
  3. 본문 영역: 넓은 줄(본문 너비의 60% 이상)과, 그 줄에 줄 간격으로 이어진 줄(문단 끝 짧은 줄).
     본문에서 줄 간격 1.5배 넘게 떨어진 줄 = 머리글·바닥글 후보.
  4. 자름선: 여러 본문 쪽(40% 이상)에서 되풀이되는 쪽만 자르고, 모든 표본 쪽의 본문 끝에서
     반 줄 높이 이상 떨어뜨린다 → 표본 쪽의 본문은 잘리지 않는다.
  5. 가릴 때(apply): 앞부분(표본 앞 쪽) 중 띠에 머리글·바닥글 같지 않은 줄(긴 줄 또는 본문 쪽
     머리글과 가로 위치가 다른 줄 — 표지 제목·머리말 본문·저자 이름)이 걸친 쪽은 가리지 않고, 자름선이 글줄을 가로지르면 여백 쪽으로 가장 가까운 줄 사이 빈 행까지 옮겨
     그 줄을 통째로 살린다(반쪽 잘린 줄 = 깨진 글자).
     전 권 실측(2026-10-15): 예측기계·이한줄에서 가로지른 줄 11개, 앞부분 본문 줄이 이 경우였다.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from PIL import Image

log = logging.getLogger(__name__)

SAMPLE_PAGES = 12          # 표본 쪽 수
SHORT_DOC_PAGES = 20       # 이 쪽 수 이하면 앞부분을 건너뛰지 않고 전 쪽을 잰다
FRONT_SKIP = 0.15          # 앞부분(표지·그림·목차) 건너뛰는 비율
BACK_SKIP = 0.10           # 끝부분(참고문헌·색인) 건너뛰는 비율
MIN_LINES = 8              # 본문 쪽으로 볼 최소 줄 수
MIN_WIDE = 6               # 본문 쪽으로 볼 최소 넓은 줄 수
MIN_BODY_PAGES = 3         # 본문 쪽이 이보다 적으면 재지 않는다(여백 그대로)
BAND_SHARE = 0.4           # 머리글·바닥글이 본문 쪽의 이 비율 이상에서 되풀이돼야 자른다
GAP_PITCH = 1.5            # 본문에서 이 줄 간격 배수보다 떨어지면 본문이 아니다
BAND_LINE_MAX_W = 0.35     # 띠 안 줄이 쪽 너비의 이 비율 이상이면 머리글·바닥글이 아니다
                           # (실측 2026-10-15: 머리글·쪽 꼬리 0.05~0.17, 표지 제목·머리말 줄 0.5 이상)
BAND_X_TOL = 0.04          # 앞부분 쪽 띠 줄의 왼쪽·오른쪽 끝·가운데가 모두 본문 쪽 머리글·바닥글 자리에서
                           # 이보다 벗어나면 머리글 아님 (실측: 이한줄 머리글 왼쪽 끝 0.11~0.24,
                           # 앞부분 「엮은이 / 김동범」 0.48~0.57 가운데 놓임)
MEASURE_DPI = 100

Box = tuple[float, float, float, float]   # (위, 아래, 왼쪽, 오른쪽) — 쪽 대비 비율


@dataclass
class Margins:
    top: float        # 이 위(쪽 높이 비율)는 가린다. 0 = 위쪽 안 가림
    bottom: float     # 이 아래는 가린다. 1 = 아래쪽 안 가림
    body_pages: int   # 측정에 쓴 본문 쪽 수
    sampled: int      # 고른 표본 쪽 수
    line_h: float = 0.025   # 본문 줄 높이(쪽 높이 비율) — 빈 행을 찾는 범위
    skip_pages: frozenset[int] = frozenset()   # 가리지 않는 앞부분 쪽(띠 안에 머리글·바닥글 아닌 줄이 있음)
    top_x: tuple[tuple[float, float], ...] = ()      # 본문 쪽 머리글 줄의 (왼쪽, 오른쪽) — 앞부분 쪽 판별 기준
    bottom_x: tuple[tuple[float, float], ...] = ()   # 본문 쪽 바닥글 줄의 (왼쪽, 오른쪽)

    @property
    def active(self) -> bool:
        return self.top > 0.0 or self.bottom < 1.0

    def describe(self) -> str:
        parts = []
        if self.top > 0.0:
            parts.append(f"위 {self.top:.1%}")
        if self.bottom < 1.0:
            parts.append(f"아래 {1 - self.bottom:.1%}")
        where = "·".join(parts) if parts else "가릴 머리글·바닥글 없음"
        skip = f", 앞부분 {len(self.skip_pages)}쪽 제외" if parts and self.skip_pages else ""
        return f"여백 자동 측정(본문 쪽 {self.body_pages}/{self.sampled}): {where}{skip}"


def sample_pages(total: int) -> list[int]:
    """측정할 쪽 번호(1부터). 짧은 문서는 전 쪽, 긴 문서는 앞·끝을 빼고 고르게."""
    if total <= SHORT_DOC_PAGES:
        return list(range(1, total + 1))
    lo = int(total * FRONT_SKIP) + 1
    hi = int(total * (1 - BACK_SKIP))
    n = min(SAMPLE_PAGES, hi - lo + 1)
    step = (hi - lo) / max(n - 1, 1)
    return sorted({lo + round(i * step) for i in range(n)})


def _page_body(lines: list[Box]) -> tuple[set[int], float, float] | None:
    """본문 줄 번호 집합, 줄 높이, 줄 간격. 본문 쪽이 아니면 None."""
    if len(lines) < MIN_LINES:
        return None
    widths = sorted(b[3] - b[2] for b in lines)
    col = widths[int(0.9 * (len(widths) - 1))]
    wide = [i for i, b in enumerate(lines) if b[3] - b[2] >= 0.6 * col]
    if len(wide) < MIN_WIDE:
        return None
    lh = median(b[1] - b[0] for b in lines)
    steps = [lines[j][0] - lines[i][0] for i, j in zip(wide, wide[1:])]
    steps = [s for s in steps if 0 < s < 3 * lh]
    pitch = median(steps) if steps else 1.4 * lh

    body = set(wide)
    changed = True
    while changed:              # 본문 줄에 줄 간격으로 이어진 줄도 본문(문단 끝 짧은 줄·소제목)
        changed = False
        for i in range(len(lines)):
            if i in body:
                continue
            for j in (i - 1, i + 1):
                if j in body and abs(lines[i][0] - lines[j][0]) < GAP_PITCH * pitch:
                    body.add(i)
                    changed = True
                    break
    return body, lh, pitch


def _pct(values: list[float], q: float) -> float:
    v = sorted(values)
    return v[round(q * (len(v) - 1))]


def measure_from_boxes(pages: list[list[Box]], sampled: int | None = None) -> Margins | None:
    """표본 쪽들의 줄 상자로 자름선을 정한다. 본문 쪽이 모자라면 None."""
    found = []
    for raw in pages:
        lines = sorted(raw)
        res = _page_body(lines)
        if res:
            found.append((lines, *res))
    if len(found) < MIN_BODY_PAGES:
        return None

    body_top = min(min(ls[i][0] for i in body) for ls, body, _, _ in found)
    body_bottom = max(max(ls[i][1] for i in body) for ls, body, _, _ in found)
    pad = 0.5 * median(lh for _, _, lh, _ in found)
    need = max(2, BAND_SHARE * len(found))

    heads, foots, top_x, bottom_x = [], [], [], []
    for lines, body, _, _ in found:
        above = [b for i, b in enumerate(lines) if i not in body and b[1] <= body_top - pad]
        below = [b for i, b in enumerate(lines) if i not in body and b[0] >= body_bottom + pad]
        if above:
            heads.append(max(b[1] for b in above))
            top_x += [(b[2], b[3]) for b in above]
        if below:
            foots.append(min(b[0] for b in below))
            bottom_x += [(b[2], b[3]) for b in below]

    top = min(body_top - pad, _pct(heads, 0.75) + pad) if len(heads) >= need else 0.0
    bottom = max(body_bottom + pad, _pct(foots, 0.25) - pad) if len(foots) >= need else 1.0
    return Margins(top=max(top, 0.0), bottom=min(bottom, 1.0),
                   body_pages=len(found), sampled=sampled or len(pages), line_h=2 * pad,
                   top_x=tuple(top_x) if top > 0 else (), bottom_x=tuple(bottom_x) if bottom < 1 else ())


def front_page_unsafe(boxes: list[Box], m: Margins) -> bool:
    """앞부분 쪽을 가리면 안 되나 — 띠에 걸친 줄이 머리글·바닥글 같지 않으면 True.

    긴 줄(표지 제목·머리말 본문)이거나, 가로 위치가 본문 쪽 머리글·바닥글 자리와 다른 줄
    (가운데 놓인 「엮은이 / 김동범」 등)이 하나라도 있으면 그 쪽은 가리지 않는다.
    위치는 왼쪽 끝·오른쪽 끝·가운데 중 하나만 맞으면 같은 자리로 본다 — 왼쪽 맞춤 머리글,
    오른쪽 맞춤 쪽 꼬리(장 제목 길이에 따라 가운데가 움직임), 가운데 쪽번호.
    """
    def off_position(b: Box, xs: tuple[tuple[float, float], ...]) -> bool:
        l, r = b[2], b[3]
        return bool(xs) and all(abs(l - l2) > BAND_X_TOL and abs(r - r2) > BAND_X_TOL
                                and abs((l + r) - (l2 + r2)) / 2 > BAND_X_TOL for l2, r2 in xs)

    for b in boxes:
        if m.top > 0 and b[0] < m.top:
            xs = m.top_x
        elif m.bottom < 1 and b[1] > m.bottom:
            xs = m.bottom_x
        else:
            continue
        if b[3] - b[2] >= BAND_LINE_MAX_W or off_position(b, xs):
            return True
    return False


def _detect_boxes(image: Image.Image) -> list[Box]:
    import numpy as np
    from core import rapidocr_engine
    w, h = image.size
    # 검출기만 직접 부른다. engine(img, use_rec=False) 는 그 설정을 공유 엔진에 남겨
    # 뒤이은 본 OCR 이 글자를 읽지 않게 된다(rapidocr update_params).
    r = rapidocr_engine.get_engine().text_det(np.array(image.convert("RGB")))
    boxes = getattr(r, "boxes", None)
    if boxes is None:
        return []
    return [(float(b[:, 1].min()) / h, float(b[:, 1].max()) / h,
             float(b[:, 0].min()) / w, float(b[:, 0].max()) / w) for b in boxes]


_cache: dict[tuple, Margins | None] = {}
_lock = threading.Lock()


def measure(pdf_path: str | Path, total: int) -> Margins | None:
    """PDF 한 파일의 여백 측정(같은 파일은 한 번만 — 미리보기 뒤 전체 처리에서 다시 쓰지 않음)."""
    from core import pdf_splitter
    p = Path(pdf_path)
    st = p.stat()
    key = (str(p.resolve()), st.st_mtime, st.st_size)
    with _lock:
        if key in _cache:
            return _cache[key]
    def boxes_of(n: int) -> list[Box]:
        img = next(pdf_splitter.iter_pdf_pages(p, dpi=MEASURE_DPI, first_page=n, last_page=n), None)
        return _detect_boxes(img) if img is not None else []

    pages = sample_pages(total)
    sampled = {n: boxes_of(n) for n in pages}
    m = measure_from_boxes(list(sampled.values()), sampled=len(pages))
    if m is not None and m.active:
        # 앞부분(표지·속표지·머리말·목차)은 쪽마다 본다 — 띠에 긴 줄이 걸치면 그 쪽은 가리지 않는다
        m.skip_pages = frozenset(n for n in range(1, min(pages))
                                 if front_page_unsafe(boxes_of(n), m))
    with _lock:
        _cache[key] = m
    return m


def _snap_to_blank(gray, row: int, step: int, reach: int, run: int) -> int | None:
    """row 에서 step 방향(여백 쪽)으로 reach 행 안에서, 빈 행이 run 줄 이상 이어진 곳(줄 사이)의
    첫 행. 글줄 안의 짧은 빈 행(받침·윗선 사이)은 줄 사이로 보지 않는다. 없으면 None."""
    h, w = gray.shape
    ink = (gray < 160).sum(axis=1) > max(2, 0.005 * w)
    for r in range(row, row + step * reach, step):
        if not 0 <= r < h:
            return None
        ahead = range(r, r + step * run, step)
        if all(0 <= k < h and not ink[k] for k in ahead):
            return r
    return None


def apply(image: Image.Image, m: Margins, page_num: int = 0) -> Image.Image:
    """자름선 바깥(머리글·바닥글 띠)을 흰색으로 가린다.

    자름선이 글줄을 가로지르면 여백 쪽 빈 행으로 옮겨 줄을 통째로 살린다.
    빈 행이 줄 높이 2배 안에 없으면 그 쪽의 그 방향은 가리지 않는다.
    """
    import numpy as np
    from PIL import ImageDraw
    img = image.convert("RGB")
    if page_num in m.skip_pages:
        return img
    w, h = img.size
    gray = np.asarray(img.convert("L"))
    reach = max(3, int(2 * m.line_h * h))
    run = max(2, int(0.2 * m.line_h * h))
    draw = ImageDraw.Draw(img)
    if m.top > 0.0:
        r = _snap_to_blank(gray, int(h * m.top), -1, reach, run)
        if r is not None and r > 0:
            draw.rectangle([0, 0, w - 1, r], fill=(255, 255, 255))
    if m.bottom < 1.0:
        r = _snap_to_blank(gray, int(h * m.bottom), +1, reach, run)
        if r is not None and r < h - 1:
            draw.rectangle([0, r, w - 1, h - 1], fill=(255, 255, 255))
    return img
