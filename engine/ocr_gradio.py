"""Gradio 웹 UI — OCR 스캔파일 텍스트추출 도구 (localhost:7860)"""
from __future__ import annotations
import html
import os
import sys
import time
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import gradio as gr
except ImportError:
    print("Gradio가 설치되지 않았습니다.  pip install gradio")
    sys.exit(1)

from core import detector, pdf_splitter, quality_diagnoser, preprocessor
from core import rapidocr_engine, tesseract_engine, char_normalizer, postprocessor, writer
# vision 파사드 — config.json 의 vision_provider 에 따라 Gemini/Claude 라우팅.
# 기존 호출부 호환을 위해 claude_vision 이름으로 import 한다.
from core import vision as claude_vision
from core import spellcheck, figure_detector, llm_corrector
from core import config, usage_tracker, blank_detector, storage_manager
from core import text_cleaner
from core import pdf_text_layer
import batch_runner
from core.types import FileResult, OutputFormat, PageResult, OCREngine

PREVIEW_PAGES = 3      # 빠른 미리보기 처리 페이지 수
WARN_SIZE_MB = 20

# 사용자가 결과를 바로 찾는 고정 폴더(프로그램 루트/02_결과).
# storage/ 는 이력 관리용 보관소이고, 이쪽은 '작업 산출물' 자리다.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "02_결과"

_CSS = """
:root, .gradio-container {
  --ocr-accent:#0f766e; --ocr-accent-soft:#e8f3f0; --ocr-ink:#1c1917; --ocr-muted:#6f6962;
  --ocr-line:#e7e4df; --ocr-line-strong:#a8a29e; --ocr-rule:#1c1917; --ocr-card:#ffffff; --ocr-canvas:#f4f3ef;
  --ocr-field:#faf9f7;
  --ocr-ok-bg:#e9f3ec; --ocr-ok:#1f6b3a; --ocr-warn-bg:#faf2de; --ocr-warn:#7a4e0a;
  --ocr-err-bg:#faeaea; --ocr-err:#8f2323;
  --ocr-shadow:0 1px 2px rgba(41,37,36,.04), 0 10px 28px -16px rgba(41,37,36,.14);
}
.dark {
  --ocr-accent:#5fb8a8; --ocr-accent-soft:#16302b; --ocr-ink:#ece9e4; --ocr-muted:#a8a29e;
  --ocr-line:#2f2c29; --ocr-line-strong:#6f6962; --ocr-rule:#e7e5e4; --ocr-card:#1c1a18; --ocr-canvas:#131211;
  --ocr-field:#221f1d;
  --ocr-ok-bg:#15291c; --ocr-ok:#8fd1a3; --ocr-warn-bg:#2d2413; --ocr-warn:#e5c07b;
  --ocr-err-bg:#2e1616; --ocr-err:#efa3a3;
  --ocr-shadow:0 1px 2px rgba(0,0,0,.3), 0 10px 28px -16px rgba(0,0,0,.5);
}
.gradio-container { max-width:none !important; margin:0 !important;
                    padding-left:28px !important; padding-right:28px !important; }

/* ── 머리말 ─────────────────────────────────────────────── */
.app-head { display:flex; justify-content:space-between; align-items:center; gap:16px;
            flex-wrap:wrap; padding:6px 2px 6px; }
.brand { display:flex; align-items:center; gap:12px; }
.brand-mark { width:36px; height:36px; border-radius:10px; background:var(--ocr-accent);
              display:grid; place-items:center; flex:none; }
.brand-mark svg { width:18px; height:18px; }
.app-title { font-size:1.45rem; font-weight:650; letter-spacing:-0.025em; color:var(--ocr-ink);
             line-height:1.2; }
.app-sub   { color:var(--ocr-muted); font-size:0.86rem; margin-top:2px; }
.chips { display:flex; gap:6px; flex-wrap:wrap; }
.chip  { display:inline-flex; align-items:center; gap:6px; padding:5px 10px; border-radius:999px;
         font-size:0.8rem; font-weight:500; border:1px solid var(--ocr-line-strong);
         color:var(--ocr-muted); background:var(--ocr-card); }
.chip::before { content:""; width:6px; height:6px; border-radius:50%; background:currentColor; opacity:.7; }
.chip.ok   { color:var(--ocr-ok); }
.chip.warn { color:var(--ocr-warn); }
.chip.err  { color:var(--ocr-err); }

/* ── 탭 ─────────────────────────────────────────────────── */
.tab-wrapper { border-bottom:1px solid var(--ocr-rule) !important; margin-bottom:10px !important; }
[role=tab] { font-weight:500 !important; }

/* ── 카드: 흰 면 + 1px 선 1개 + 옅은 그림자 ─────────────── */
.step-card { border:1px solid var(--ocr-rule) !important; border-radius:16px !important;
             background:var(--ocr-card) !important; box-shadow:var(--ocr-shadow) !important;
             padding:16px 18px 18px !important; gap:12px !important; }
.step-card .form { background:transparent !important; border:0 !important; gap:14px !important;
                   box-shadow:none !important; }
.step-card .block { background:transparent !important; border:0 !important;
                    box-shadow:none !important; padding:0 !important; }
.step-card .html-container { padding:0 !important; }

/* 카드 제목 — 아래 1px 선 하나로 구획 */
.step-top   { padding-bottom:10px; border-bottom:1px solid var(--ocr-rule); }
.step-head  { display:flex; align-items:center; gap:10px; }
.step-no    { width:24px; height:24px; border-radius:7px; background:var(--ocr-accent); color:#fff;
              display:inline-grid; place-items:center; font-size:0.78rem; font-weight:600;
              font-variant-numeric:tabular-nums; flex:none; }
.step-title { font-size:1.05rem; font-weight:600; color:var(--ocr-ink); letter-spacing:-0.015em; }
.step-hint  { font-size:0.85rem; color:var(--ocr-muted); margin:6px 0 0; max-width:62ch;
              line-height:1.6; text-wrap:pretty; }

/* 선택지: 회색 알약 대신 1px 선 목록 */
.step-card fieldset label, .step-card .wrap > label {
  background:var(--ocr-card) !important; border:1px solid var(--ocr-line-strong) !important;
  border-radius:10px !important; box-shadow:none !important; }
.step-card fieldset label.selected, .step-card .wrap > label.selected {
  border-color:var(--ocr-accent) !important; background:var(--ocr-accent-soft) !important; }

/* 입력칸·로그 */
.step-card textarea, .step-card input[type=text], .step-card .secondary-wrap {
  background:var(--ocr-field) !important; }
.step-log textarea { font-size:0.82rem !important; line-height:1.65 !important; }
.rec-table { border:1px solid var(--ocr-line); border-radius:8px; }
.rec-table table { width:100%; border-collapse:collapse; font-size:0.8rem; color:var(--ocr-ink); }
.rec-table th { position:sticky; top:0; background:var(--ocr-field); text-align:left;
                font-weight:600; border-bottom:1px solid var(--ocr-line-strong); }
.rec-table th, .rec-table td { padding:5px 8px; border-bottom:1px solid var(--ocr-line);
                               white-space:nowrap; }

/* 파일 올리는 곳: 점선 1px */
.step-card button.boundedheight { border:1px dashed var(--ocr-line-strong) !important;
                                  border-radius:12px !important; background:var(--ocr-field) !important; }
.step-card button.boundedheight:hover { border-color:var(--ocr-accent) !important; }
/* 비어 있는 결과 파일 칸: 큰 빈 상자 대신 한 줄 높이 */
.step-card .empty.large { min-height:64px !important; height:64px !important;
                          border:1px dashed var(--ocr-line-strong) !important; border-radius:10px !important; }

/* 접이식(고급 설정 등): 카드 안에서는 위쪽 선 1개, 밖에서는 선 테두리 */
.block:has(> button.label-wrap) { border:1px solid var(--ocr-rule) !important; border-radius:12px !important;
                                  background:var(--ocr-card) !important; box-shadow:none !important; }
.step-card .block:has(> button.label-wrap) { border:0 !important; border-top:1px solid var(--ocr-rule) !important;
                                             border-radius:0 !important; padding-top:12px !important; }
button.label-wrap span { font-weight:500; }

/* ── 안내 상자 ───────────────────────────────────────────── */
.note { padding:10px 12px; border-radius:10px; font-size:0.86rem; line-height:1.6;
        background:var(--ocr-field); color:var(--ocr-ink); border:1px solid var(--ocr-line-strong); }
.note.info { background:var(--ocr-accent-soft); color:var(--ocr-accent); border-color:transparent; }
.batch-list .wrap { flex-direction:column; align-items:stretch; }   /* ④ 탭 파일 목록: 한 줄에 하나 */
.note.ok   { background:var(--ocr-ok-bg);   color:var(--ocr-ok);   border-color:transparent; }
.note.warn { background:var(--ocr-warn-bg); color:var(--ocr-warn); border-color:transparent; }
.note.err  { background:var(--ocr-err-bg);  color:var(--ocr-err);  border-color:transparent; }
.note code { font-size:0.8rem; word-break:break-all; }
.guide { font-size:0.88rem; line-height:1.7; color:var(--ocr-ink); }
.guide ol { margin:10px 0 8px 1.3em; padding:0; }
.guide li { margin:4px 0; }
.guide .muted { color:var(--ocr-muted); font-size:0.82rem; }
.guide a { color:var(--ocr-accent); font-weight:500; }

/* ── 진행바 ─────────────────────────────────────────────── */
.prog { padding:2px 0; }
.prog-top { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:8px;
            font-size:0.9rem; color:var(--ocr-ink); font-weight:500; }
.prog-top .muted { color:var(--ocr-muted); font-size:0.82rem; margin-left:8px; font-weight:400; }
.prog-num { font-variant-numeric:tabular-nums; font-weight:600; }
.prog-track { background:var(--ocr-line); border-radius:4px; height:6px; overflow:hidden; }
.prog-fill  { height:100%; background:var(--ocr-accent); transition:width .4s ease; }
.prog-fill.done { background:var(--ocr-ok); }
.prog-idle { color:var(--ocr-muted); }

/* ── 저장 현황 숫자 ─────────────────────────────────────── */
.stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
         border:1px solid var(--ocr-rule); border-radius:12px; overflow:hidden; }
.stat  { padding:14px 16px; }
.stat + .stat { border-left:1px solid var(--ocr-rule); }
.stat b    { display:block; font-size:1.4rem; font-weight:600; color:var(--ocr-ink);
             font-variant-numeric:tabular-nums; letter-spacing:-0.02em; }
.stat span { font-size:0.8rem; color:var(--ocr-muted); }

.warn-panel     { background:var(--ocr-warn-bg); border-radius:10px; padding:12px 14px; margin:4px 0; }
.continue-panel { background:var(--ocr-ok-bg);   border-radius:10px; padding:12px 14px; margin:4px 0; }

/* ── 버튼 ───────────────────────────────────────────────── */
button { transition:transform .12s ease, background-color .2s ease, border-color .2s ease,
         box-shadow .2s ease; }
button:active { transform:translateY(1px); }
footer { display:none !important; }   /* Gradio 기본 하단 문구 — 스크롤만 늘린다 */
button.primary:hover { box-shadow:0 6px 16px -8px rgba(15,118,110,.55); }
"""

_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.stone,
    radius_size=gr.themes.sizes.radius_md,
    font=["Pretendard", "Malgun Gothic", "Apple SD Gothic Neo", "system-ui", "sans-serif"],
    font_mono=["D2Coding", "Consolas", "ui-monospace", "monospace"],
).set(
    body_background_fill="#f4f3ef", body_background_fill_dark="#131211",
    block_border_width="0px", block_shadow="none", block_shadow_dark="none",
    form_gap_width="0px",
    input_background_fill="#faf9f7", input_background_fill_dark="#221f1d",
    input_border_color="#a8a29e", input_border_color_dark="#6f6962", input_border_width="1px",
    checkbox_label_background_fill="#ffffff", checkbox_label_background_fill_dark="#1c1a18",
    checkbox_label_background_fill_selected="#e8f3f0",
    checkbox_label_background_fill_selected_dark="#16302b",
    checkbox_label_border_color="#a8a29e", checkbox_label_border_color_dark="#6f6962",
    checkbox_label_border_color_selected="#0f766e", checkbox_label_border_width="1px",
    button_border_width="1px",
    button_primary_background_fill="#0f766e", button_primary_background_fill_hover="#0d6b63",
    button_primary_background_fill_dark="#2f8f82",
    button_secondary_background_fill="#ffffff", button_secondary_background_fill_dark="#1c1a18",
    button_secondary_background_fill_hover="#f4f3ef",
    button_secondary_background_fill_hover_dark="#26231f",
    button_secondary_border_color="#d7d2ca", button_secondary_border_color_dark="#403c38",
    button_secondary_border_color_hover="#b9b2a8",
)


# ══════════════════════════════════════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    return f"{s}초" if s < 60 else f"{s // 60}분 {s % 60}초"


class _CleanCheckpoint:
    """정제 청크를 02_결과/ 에 즉시 기록하는 복구용 체크포인트.

    기존에는 전 구간 정제가 끝나야 파일이 한 번 쓰였다. 그래서 중간에 끊기면
    (브라우저 종료·API 정지·네트워크 대기) 이미 정제된 구간까지 전부 소실됐다.
    여기서는 구간이 끝날 때마다 '{원본명}_정제중.{fmt}' 에 append·flush 하고,
    정상 완료 시 최종 본문으로 '{시각}_{원본명}.{fmt}' 를 확정한 뒤 중간 파일을
    지운다. 끊기면 _정제중 파일이 그대로 남아 손으로 회수할 수 있다.

    (최종 파일을 따로 쓰는 이유: 페이지 넘김 어절 결합 등 마지막 전체 후처리는
     전 구간이 모인 뒤에야 적용되므로, 중간 파일은 어디까지나 복구본이다.)
    """

    def __init__(self, stem: str, fmt: OutputFormat, suffix: str = "정제중"):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.stem = stem
        self.ext  = fmt.value
        self.path = RESULTS_DIR / f"{stem}_{suffix}.{self.ext}"
        # 앞 실행이 끊겨 남긴 복구본은 시각을 붙여 보존(재실행이 0바이트로 덮어쓰던 결함)
        if self.path.exists() and self.path.stat().st_size:
            self.path.replace(RESULTS_DIR / f"{stem}_{suffix}_{time.strftime('%Y%m%d_%H%M%S')}.{self.ext}")
        self._fh  = open(self.path, "w", encoding="utf-8")

    def append(self, chunk: str) -> None:
        self._fh.write(chunk + "\n\n")
        self._fh.flush()

    def finish(self, final_text: str) -> Path:
        """정상 완료 — 최종 본문 확정. 확정된 경로를 돌려준다."""
        self.close()
        final = RESULTS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{self.stem}.{self.ext}"
        final.write_text(
            final_text if final_text.endswith("\n") else final_text + "\n",
            encoding="utf-8")
        self.path.unlink(missing_ok=True)
        return final

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def _publish_to_results(out_path: Path, stem: str) -> Path:
    """정제를 안 한 결과물도 02_결과/ 에 사본을 남긴다. 실패는 호출부가 기록한다."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = RESULTS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{stem}{out_path.suffix}"
    dest.write_bytes(Path(out_path).read_bytes())
    return dest


def _finalize_results(ckpt: "_CleanCheckpoint | None", cleaned_text: str | None,
                      out_path: Path, stem: str,
                      logs: list[str] | None = None) -> Path | None:
    """최종 산출물을 02_결과/ 에 확정하고 그 경로를 돌려준다(실패 시 None + logs 경고)."""
    try:
        if ckpt is not None and cleaned_text:
            return ckpt.finish(cleaned_text)
        if ckpt is not None:
            ckpt.close()   # 정제 결과가 비었으면 _정제중 파일을 증거로 남겨 둔다
        return _publish_to_results(out_path, stem)
    except Exception as e:
        # 조용히 넘기면 화면의 '저장 위치'가 임시 폴더를 가리킨다 — 반드시 표면화
        if logs is not None:
            logs.append(f"  ⚠ 02_결과 폴더에 결과를 저장하지 못했습니다(아래 경로는 임시 사본): {e}")
        return None


def _progress_html(pages_done: int, total: int, label: str,
                   elapsed_str: str = "", done: bool = False,
                   unit: str = "페이지") -> str:
    pct = int(pages_done / total * 100) if total > 0 else 0
    fill_cls = "prog-fill done" if done else "prog-fill"
    idle = " prog-idle" if (pages_done == 0 and not done) else ""
    step_text = f"{pages_done}/{total} {unit} · {pct}%" if total > 0 else ""
    elapsed_html = f'<span class="muted">경과 {elapsed_str}</span>' if elapsed_str else ""
    return f"""
<div class="prog{idle}">
  <div class="prog-top"><span>{label}{elapsed_html}</span><span class="prog-num">{step_text}</span></div>
  <div class="prog-track"><div class="{fill_cls}" style="width:{pct}%"></div></div>
</div>"""


def _status_text(page_num: int, total: int, pct: int,
                 logs: list[str], elapsed_str: str = "") -> str:
    header = f"처리 중: {page_num}/{total} 페이지 ({pct}%)"
    if elapsed_str:
        header += f"  |  경과: {elapsed_str}"
    lines = [header]
    if logs:
        lines.append("")
        lines.extend(logs[-5:])
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 공통: 페이지 계획 (지연 렌더링)
# ══════════════════════════════════════════════════════════════════════════════

def _page_plan(file_path: Path, start_page: int = 1, max_pages: int | None = None):
    """
    (총페이지수, 이터레이터[(page_num, image)]) 반환.

    대용량(책 스캔) 대응: PDF는 전 페이지를 한꺼번에 메모리에 올리지 않고
    `pdf_splitter.iter_pdf_pages`로 batch 단위 지연 렌더링한다. DPI는 config의
    default_dpi(기본 200)를 따른다 → 네트워크 전송량·메모리 동시 감소.
    """
    dpi = config.get_default_dpi()
    fmt_type = detector.detect(file_path)

    if fmt_type == "pdf":
        total = pdf_splitter.count_pages(file_path)
        last = total if max_pages is None else min(total, start_page + max_pages - 1)

        def _gen():
            k = 0
            for img in pdf_splitter.iter_pdf_pages(
                file_path, dpi=dpi, first_page=start_page, last_page=last
            ):
                yield start_page + k, img
                k += 1

        return total, _gen()

    # 단일 이미지 파일
    imgs = pdf_splitter.load_image(file_path)
    total = len(imgs)
    end = total if max_pages is None else min(total, start_page - 1 + max_pages)

    def _gen_img():
        for idx in range(start_page - 1, end):
            yield idx + 1, imgs[idx]

    return total, _gen_img()


# ══════════════════════════════════════════════════════════════════════════════
# 단일 페이지 OCR
# ══════════════════════════════════════════════════════════════════════════════

def _process_page(image, page_num: int, lang: str,
                  no_preprocess: bool, no_fallback: bool, no_figure: bool,
                  llm_correct: bool, llm_budget: int,
                  quick: bool = False,
                  vision_first: bool = False) -> tuple[PageResult, list[str]]:
    """
    vision_first=True : Claude Vision 우선 모드 (API 키 있을 때)
      1순위: Claude Vision  — 레이아웃 이해, 마진 무시, 한국어 고정확도
      2순위: Tesseract       — Vision 실패/미설정 시 자동 전환

    vision_first=False (기본): Tesseract 우선 모드
      1순위: Tesseract       — 로컬·무료·빠름
      2순위: Claude Vision   — 저신뢰도·자모분리 시 폴백

    quick=True : 미리보기 전용 경량 모드
    """
    logs: list[str] = []
    placeholders: list[str] = []

    # ══════════════════════════════════════════════════════════════════════════
    # 0단계: 빈 페이지 감지 (OCR 전 차단 — API 크레딧 절약)
    # Vision 우선 모드에서는 fast 모드(픽셀·엣지만, Tesseract 생략)로 실행 —
    # 어차피 Gemini가 본문을 읽으므로 느린 Tesseract 빈페이지 확정은 불필요.
    # ══════════════════════════════════════════════════════════════════════════
    blank = blank_detector.check(image, fast=(quick or vision_first))
    if blank.is_blank:
        logs.append(
            f"  p.{page_num}: ⬜ 빈 페이지 감지 [{blank.reason}] → 건너뜀"
        )
        return PageResult(
            page_num=page_num,
            raw_text="",
            corrected_text="",
            engine=OCREngine.TESSERACT,
            confidence=0.0,
            is_blank=True,
            blank_reason=blank.reason,
        ), logs

    # ══════════════════════════════════════════════════════════════════════════
    # Vision 우선 모드 (API 키 있을 때)
    # ══════════════════════════════════════════════════════════════════════════
    if vision_first and claude_vision.is_available():
        logs.append(f"  p.{page_num}: {claude_vision.provider_label()} 처리 중...")
        try:
            _t = time.time()
            ocr_result = claude_vision.run(image, lang_hint=lang)
            logs.append(f"  p.{page_num}: ✅ Vision 완료 ({time.time() - _t:.1f}s)")
            return _finalize(ocr_result, page_num, placeholders, llm_correct, llm_budget), logs
        except Exception as e:
            logs.append(f"  p.{page_num}: Vision 오류 {time.time() - _t:.1f}s → Tesseract 폴백 ({e})")
            # Vision 실패 시 Tesseract로 전환

    elif vision_first and not claude_vision.is_available():
        logs.append(f"  p.{page_num}: ⚠ API 키 미설정 → Tesseract로 처리")

    # ══════════════════════════════════════════════════════════════════════════
    # Tesseract 처리 (우선 or Vision 실패 폴백)
    # ══════════════════════════════════════════════════════════════════════════
    report    = quality_diagnoser.diagnose(image)
    ocr_image = image

    # 그림·표 감지 (Vision 우선 모드·빠른 모드에서는 생략)
    if not no_figure and not quick and not vision_first:
        det = figure_detector.detect(image, lang=lang)
        if det.regions:
            placeholders = [figure_detector.build_placeholder(r) for r in det.regions]
            ocr_image = det.masked_image or image
            logs.append(f"  p.{page_num} 감지 영역: {len(det.regions)}개")

    # 품질 불량 즉시 Vision 폴백 (Tesseract 우선 모드일 때)
    # Vision 엔진은 실패 시 예외를 던진다 — 받지 않으면 한 쪽 실패로 파일 전체가 중단된다
    vision_err = ""
    if not vision_first and not no_fallback and report.needs_fallback and claude_vision.is_available():
        logs.append(f"  p.{page_num}: 품질 불량 → {claude_vision.provider_label()} 폴백")
        try:
            ocr_result = claude_vision.run(image, lang_hint=lang)
        except Exception as e:
            vision_err = str(e)
            logs.append(f"  p.{page_num}: 폴백 실패 → Tesseract로 계속 ({e})")
        else:
            return _finalize(ocr_result, page_num, placeholders, llm_correct, llm_budget), logs

    # 전처리
    if no_preprocess:
        processed = ocr_image
    elif quick:
        processed = preprocessor.quick_preprocess(ocr_image)
    else:
        # RapidOCR 사용 가능 시 딥러닝 피처(RGB 컬러/그라데이션) 보존 경량 전처리 적용
        if rapidocr_engine.is_available():
            processed = preprocessor.preprocess_for_rapidocr(ocr_image, report)
        else:
            processed = preprocessor.preprocess(ocr_image, report)

    ocr_result  = tesseract_engine.smart_run(processed, default_lang=lang, fast=quick)
    low_quality = ocr_result.confidence < config.get_confidence_threshold() or ocr_result.has_jamo_split

    # Tesseract/RapidOCR 저신뢰도 → Vision 폴백 (비용 우선 모드일 때만)
    if not vision_first and not no_fallback and low_quality:
        if claude_vision.is_available() and not vision_err:  # 이미 실패했으면 재시도 안 함 (Ollama 쪽당 최대 9분)
            reason = "자모분리" if ocr_result.has_jamo_split else f"신뢰도 {ocr_result.confidence:.1f}%"
            logs.append(f"  p.{page_num}: {reason} → {claude_vision.provider_label()} 폴백")
            try:
                vision_result = claude_vision.run(image, lang_hint=lang)
            except Exception as e:
                vision_err = str(e)
                logs.append(f"  p.{page_num}: 폴백 실패 → 로컬 OCR 결과 유지 ({e})")
            else:
                return _finalize(vision_result, page_num, placeholders, llm_correct, llm_budget), logs
        reason = ("자모 분리 감지" if ocr_result.has_jamo_split
                  else f"신뢰도 낮음 ({ocr_result.confidence:.1f}%)")
        if vision_err:
            reason += " · Vision 폴백 실패"
        logs.append(f"  p.{page_num}: ⚠ {reason} — 검토 필요")
        return _finalize(ocr_result, page_num, placeholders, llm_correct, llm_budget,
                         needs_review=True, review_reason=reason), logs

    engine_name = "RapidOCR 한글 v5" if rapidocr_engine.is_available() else "Tesseract"
    logs.append(f"  p.{page_num}: {engine_name} 신뢰도 {ocr_result.confidence:.1f}%")
    # 품질 불량 진단 쪽은 Tesseract 신뢰도가 높아도 폴백 실패를 결과에 남긴다
    return _finalize(ocr_result, page_num, placeholders, llm_correct, llm_budget,
                     needs_review=bool(vision_err),
                     review_reason="품질 불량 · Vision 폴백 실패" if vision_err else ""), logs


def _finalize(ocr_result, page_num: int, placeholders: list[str],
              llm_correct: bool, llm_budget: int,
              needs_review: bool = False, review_reason: str = "") -> PageResult:
    """
    OCR 결과 → 최종 텍스트 변환 (엔진별 전체 처리 체인 적용).

    처리 순서:
      1. L1  — char_normalizer  (Unicode·합자·특수기호·한글분절·영문혼동)
      2. L2  — spellcheck       (한국어 맞춤법, 설치 시 자동)
      3. 후처리 — postprocessor.process_full()
              Tesseract: 마진노이즈제거 → 단락복원 → 공백정규화
              Vision:    단락복원 → 공백정규화
      4. L3  — llm_corrector    (Claude Haiku 문맥 교정, 활성화 시)
    """
    engine_str = ocr_result.engine.value  # 'tesseract' or 'claude_vision'

    # L1 — 규칙 기반 보정 (합자·특수기호·한글분절·영문혼동)
    text = char_normalizer.normalize_all(ocr_result.text)

    # MyOCR 기반 한국어 정제 (띄어쓰기, 맞춤법, 어절 쪼개짐/결합 보정)
    try:
        from core.korean_corrector import refine_korean_text
        text = refine_korean_text(text)
    except Exception:
        pass

    # L2 — 한국어 맞춤법 검사 (설치된 경우)
    if spellcheck.is_available():
        text = spellcheck.correct_safe(text)

    # 후처리 — 엔진별 분기
    text = postprocessor.process_full(text, engine=engine_str)

    # L3 — LLM 문맥 교정 (활성화 시)
    if llm_correct and llm_corrector.is_available():
        text = llm_corrector.correct(text, budget=llm_budget)

    return PageResult(
        page_num=page_num, raw_text=ocr_result.text, corrected_text=text,
        engine=ocr_result.engine, confidence=ocr_result.confidence,
        figures=placeholders, needs_review=needs_review, review_reason=review_reason,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 디지털 PDF 텍스트 레이어 직접 추출 (OCR·Vision API 미사용)
# ══════════════════════════════════════════════════════════════════════════════

def _finalize_digital(text: str, page_num: int) -> PageResult:
    """디지털 PDF 텍스트 레이어 1페이지 → PageResult (OCR·Vision·API 없음).

    내장 텍스트는 이미 유니코드 정상이라 OCR 교정(char_normalizer)·마진 노이즈
    제거가 불필요하다. 다단 편집의 컬럼 줄바꿈만 문단으로 복원하고 공백을
    정규화한다(Vision과 동일한 경량 후처리). 텍스트가 거의 없는 페이지(빈 페이지·
    이미지 전용)는 blank 처리해 저장·정제 대상에서 제외한다.
    """
    raw = text or ""
    if pdf_text_layer.real_chars(raw) < pdf_text_layer.MIN_PAGE_CHARS:
        return PageResult(
            page_num=page_num, raw_text="", corrected_text="",
            engine=OCREngine.PDF_TEXT, confidence=100.0,
            is_blank=True, blank_reason="텍스트 없음(빈 페이지·이미지 전용)",
        )
    corrected = postprocessor.process_full(raw, engine=OCREngine.PDF_TEXT.value)
    return PageResult(
        page_num=page_num, raw_text=raw, corrected_text=corrected,
        engine=OCREngine.PDF_TEXT, confidence=100.0,
    )


def _page_reader(file_path: Path, lang: str, no_preprocess: bool, no_fallback: bool,
                 no_figure: bool, llm_correct: bool, llm_budget: int, vision_first: bool):
    """쪽 번호 → (PageResult, 로그). 디지털 PDF에서 내장 글자가 깨진 쪽만 이미지로 다시 읽는다."""
    def read(page_num: int):
        image = next(pdf_splitter.iter_pdf_pages(
            file_path, dpi=config.get_default_dpi(), first_page=page_num, last_page=page_num))
        return _process_page(image, page_num, lang, no_preprocess, no_fallback, no_figure,
                             llm_correct, llm_budget, vision_first=vision_first)
    return read


def _run_digital_pdf(file_path: Path, pages_text: list[str], out_fmt: OutputFormat,
                     cleanup: bool, front_matter: bool, start_time: float,
                     label: str = "디지털 PDF 추출",
                     garbled_pages=(), read_page=None, noisy_pages=(), repaired_lines=None):
    """디지털 PDF(텍스트 레이어 보유) 전용 초고속 경로.

    이미지 렌더링·OCR·Vision API를 모두 건너뛰고 내장 텍스트를 그대로 읽는다.
    호출부(process_preview/_run_ocr)가 pdf_text_layer.analyze()로 디지털임을
    판별한 뒤 진입시킨다. 이후 정제(교정)·저장·이력 기록은 OCR 경로와 동일
    파이프라인을 재사용한다. common_outputs 와 동일한 6-튜플을 yield 한다.
    """
    elapsed = lambda: _fmt_elapsed(time.time() - start_time)
    total = len(pages_text)

    yield (
        _progress_html(0, total, "디지털 PDF 감지 — 텍스트 레이어 직접 추출 중...",
                       elapsed_str=elapsed()),
        "📄 텍스트 레이어를 직접 추출합니다 (OCR·API 없음 · 무료·오프라인).",
        "", None, gr.update(visible=False), None,
    )

    result = FileResult(source_path=file_path, output_path=None)
    step_logs: list[str] = []
    garbled = set(garbled_pages)
    reread = 0
    for i, txt in enumerate(pages_text):
        page_num = i + 1
        if page_num in garbled and read_page is not None:
            # 내장 글자가 깨진 쪽(검색 가능 스캔본의 불량 OCR) — 이미지로 다시 읽는다
            try:
                page_result, page_logs = read_page(page_num)
                if pdf_text_layer.is_garbled(page_result.corrected_text):
                    # 실측: 다시 읽어도 깨진 쪽은 장식 그림·손글씨 쪽이었다(세상의지혜 181쪽 등)
                    page_result.needs_review = True
                    page_result.review_reason = "다시 읽어도 글자가 깨짐 — 그림·장식 쪽일 수 있음"
                    page_logs.append(f"  ⚠ p.{page_num}: 다시 읽어도 글자가 깨짐 — 그림·장식 쪽인지 확인 필요")
                result.pages.append(page_result)
                step_logs.extend(page_logs)
                reread += 1
            except Exception as e:   # 렌더·OCR 실패 시 내장 글자라도 남긴다
                step_logs.append(f"  p.{page_num}: 다시 읽기 실패 → 내장 글자 사용 ({e})")
                result.pages.append(_finalize_digital(txt, page_num))
        else:
            result.pages.append(_finalize_digital(txt, page_num))
        # 초고속이라 5페이지마다(또는 마지막)만 진행률 갱신
        if page_num == total or page_num % 5 == 0 or page_num in garbled:
            pct = int(page_num / total * 100) if total else 100
            yield (
                _progress_html(page_num, total,
                               f"{label}: 텍스트 레이어 추출 {page_num}/{total}p",
                               elapsed_str=elapsed()),
                _status_text(page_num, total, pct, step_logs, elapsed()),
                "", None, gr.update(visible=False), None,
            )

    text_pages  = sum(1 for p in result.pages if not p.is_blank)
    blank_count = total - text_pages
    step_logs.append("  ⚡ 텍스트 레이어 직접 추출 — OCR·API 미사용 (무료·오프라인)")
    if repaired_lines:
        pgs = sorted(repaired_lines)
        step_logs.append(f"  🔤 글자 정보가 깨진 글꼴(표·제목 등) {sum(repaired_lines.values())}줄은 이미지로 다시 읽음"
                         f" ({', '.join(map(str, pgs[:10]))}{'…' if len(pgs) > 10 else ''}쪽)")
    if reread:
        step_logs.append(f"  🔁 내장 글자가 깨진 {reread}쪽은 이미지로 다시 읽음"
                         f" ({', '.join(map(str, sorted(garbled)[:10]))}{'…' if len(garbled) > 10 else ''})")
    if noisy_pages and not cleanup:
        step_logs.append(f"  ⚠ 잡음·띄어쓰기 문제가 있는 {len(noisy_pages)}쪽이 교정 없이 저장됩니다"
                         " — '문장 정제'를 켜고 다시 처리하면 고쳐집니다")
    if blank_count:
        step_logs.append(
            f"  ⬜ 텍스트 없는 페이지 {blank_count}p 제외 (빈 페이지·이미지 전용)"
        )

    # ── 문서 전체 정제(교정) — OCR 경로와 동일(선택) ────────────────────────────
    # cleanup=True 이고 Gemini 키가 있으면 추출 텍스트도 문서 맥락 교정을 적용한다.
    # (완전 오프라인·무비용을 원하면 '문장 정제' 체크를 끄면 이 단계가 생략된다.)
    cleaned_text = None
    ckpt = None
    content_pages = [p for p in result.pages if not p.is_blank]
    if cleanup and content_pages and text_cleaner.is_available():
        ckpt = _CleanCheckpoint(file_path.stem, out_fmt)
        crep: dict = {}     # 이 실행의 정제 보고서(여러 파일 동시 처리 시 전역 보고서는 섞인다)
        clean_gen = text_cleaner.iter_clean(
            [p.corrected_text for p in content_pages],
            drop_front_matter=front_matter,
            on_chunk=ckpt.append,
            report_out=crep,
        )
        try:
            while True:
                done_c, total_c = next(clean_gen)
                yield (
                    # 진행률은 '구간' 기준 — 페이지 기준(=이미 100%)으로 그리면
                    # 정제가 남았는데도 화면이 완료처럼 보인다.
                    _progress_html(done_c, total_c,
                                   f"📝 문장 정제·교정 중 ({done_c}/{total_c} 구간)",
                                   elapsed_str=elapsed(), unit="구간"),
                    _status_text(total, total, 100, step_logs, elapsed())
                    + f"\n  📝 정제(교정) 진행: {done_c}/{total_c} 구간"
                    + f"\n  💾 중간 저장 중: {ckpt.path}",
                    "", None, gr.update(visible=False), None,
                )
        except StopIteration as stop:
            cleaned_text = stop.value or ""
        if cleaned_text:
            step_logs.append(f"  📝 문장 정제 완료 ({len(cleaned_text):,}자)")
        _crep = crep or None
        if _crep and (_crep["fallback_guard"] or _crep["fallback_api"]):
            step_logs.append(
                f"  ⚠️ 정제 무결성 가드 발동 {_crep['fallback_guard'] + _crep['fallback_api']}"
                f"/{_crep['chunks']} 구간 — LLM 교정 대신 원문 보존(내용 무손실)")
        if _crep:   # 부속물 제거·지어낸 문장 등 확인할 경고 — 강등 건은 위 개수 줄로 대신한다
            step_logs.extend(f"  ⚠ {w}" for w in _crep["warnings"] if "강등" not in w)

    # ── 저장 ───────────────────────────────────────────────────────────────────
    yield (
        _progress_html(total, total, "결과 파일 저장 중...", elapsed_str=elapsed()),
        _status_text(total, total, 100, step_logs, elapsed()),
        "", None, gr.update(visible=False), None,
    )
    tmp_dir = Path(tempfile.mkdtemp(prefix="ocr_output_"))
    raw_path = None
    try:
        if cleaned_text:
            out_path = writer.write_plain(file_path, cleaned_text, tmp_dir, out_fmt)
            # 정제 전 원문(페이지 마커 포함)도 별도 보존 — 정제 단계에서 소실이
            # 생겨도 항상 원문으로 복구할 수 있는 안전망 (메시지성경 사고 재발 방지)
            try:
                raw_path = writer.write(result, tmp_dir / "raw", out_fmt)
            except Exception as e:
                # 안전망이 없다는 사실을 숨기면 '완료'만 보고 원문을 믿게 된다 — 표면화
                raw_path = None
                step_logs.append(f"  ⚠ 정제 전 원문 사본을 만들지 못했습니다(복구용 원문 없음): {e}")
        else:
            out_path = writer.write(result, tmp_dir, out_fmt)
    except Exception as e:
        if ckpt is not None:
            ckpt.close()   # 중간 저장분(_정제중)은 회수용으로 남긴다
        yield (_progress_html(total, total, "저장 오류"),
               f"❌ {e}\n   💾 정제 중간 저장분: {ckpt.path if ckpt else '없음'}",
               "", None, gr.update(visible=False), None)
        return

    if cleaned_text:
        preview = (cleaned_text[:4000] + "\n\n... (이하 생략)"
                   if len(cleaned_text) > 4000 else cleaned_text)
    else:
        preview = _build_preview(result)

    # ── 02_결과/ 확정 — 이력에 경로를 남기려고 기록보다 먼저 ──────────────────────
    results_path = _finalize_results(ckpt, cleaned_text, out_path, file_path.stem, step_logs)

    # ── storage: 결과 파일 영구 저장 + 이력 기록 ───────────────────────────────
    try:
        stored_path = storage_manager.save_result(out_path, file_path.name)
        if raw_path is not None:
            try:
                storage_manager.save_raw_copy(raw_path, stored_path)
            except Exception as e:
                step_logs.append(f"  ⚠ 정제 전 원문 사본을 보관소에 저장하지 못했습니다(복구용 원문 없음): {e}")
        stored_upload = (storage_manager.save_original(file_path, file_path.name)
                         if _should_save_original() else None)
        storage_manager.record(
            original_name   = file_path.name,
            result_path     = stored_path,
            upload_path     = stored_upload,
            pages_total     = total,
            pages_ocr       = text_pages,
            pages_blank     = blank_count,
            vision_pages    = 0,
            engine_mode     = "pdf_text_layer",
            file_size_bytes = file_path.stat().st_size,
            output_path     = results_path,
        )
    except Exception as e:
        # 조용히 넘기면 화면엔 '완료'만 뜨고 파일은 없다 — 반드시 표면화한다.
        step_logs.append(f"  ⚠ 이력 보관소 저장 실패: {e}")

    total_el = elapsed()
    summary = [
        f"✅ 완료  |  디지털 PDF 텍스트 레이어 직접 추출  |  총 소요: {total_el}",
        f"   전체 {total}p → 추출 {text_pages}p  ·  OCR·API 미사용 (무료)",
    ]
    if blank_count:
        summary.append(f"   ⬜ 텍스트 없는 페이지 제외: {blank_count}p")
    if cleaned_text:
        summary.append("   📝 문장 정제 적용됨")
    summary.append(f"   📁 저장 위치: {results_path or out_path}")
    if step_logs:
        summary.append("")
        summary.extend(step_logs[-5:])

    yield (
        _progress_html(total, total, f"완료  ·  총 {total_el}",
                       elapsed_str=total_el, done=True),
        "\n".join(summary),
        preview,
        str(results_path or out_path),
        gr.update(visible=False),
        None,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ① 빠른 미리보기 (처음 N페이지)
# ══════════════════════════════════════════════════════════════════════════════

def process_preview(
    file_obj, lang, fmt, no_preprocess, no_fallback, no_figure,
    llm_correct, llm_budget, api_key_input, vision_first_str,
    cleanup=True, front_matter=False,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    vision_first = "Vision 우선" in str(vision_first_str)
    """처음 PREVIEW_PAGES 페이지만 OCR → 결과 표시 + 전체 처리 여부 안내.

    미리보기는 '속도 확인' 목적이라 정제(교정) 단계는 적용하지 않는다.
    정제는 전체 처리(🚀/✅ 계속) 시에만 실행된다(cleanup 인자는 시그니처
    호환을 위해 받되 여기서는 사용하지 않음)."""
    file_obj = _first_file(file_obj)   # 여러 개를 올렸으면 미리보기는 첫 파일만
    start_time = time.time()
    elapsed = lambda: _fmt_elapsed(time.time() - start_time)

    if file_obj is None:
        yield (_progress_html(0, 0, "대기", done=False),
               "⚠ 파일을 업로드해 주세요.", "", None,
               gr.update(visible=False), None)
        return

    if api_key_input.strip():
        claude_vision.set_session_key(api_key_input)

    file_path = Path(file_obj.name)
    out_fmt   = OutputFormat.MD if fmt == "md" else OutputFormat.TXT
    file_mb   = file_path.stat().st_size / 1_048_576

    # ── 디지털 PDF 우선 경로 ──────────────────────────────────────────────────
    # 텍스트 레이어가 있으면 렌더·OCR·API 없이 즉시 전체 추출한다(미리보기/이어서
    # 분할 불필요). 스캔 PDF는 is_digital=False → 아래 일반 미리보기 경로로 진행.
    if file_path.suffix.lower() == ".pdf" and pdf_text_layer.is_available():
        _info = pdf_text_layer.analyze(file_path, repair=True)
        if _info and _info.is_digital:
            yield from _run_digital_pdf(
                file_path, _info.pages, out_fmt, cleanup, front_matter, start_time,
                label="디지털 PDF 추출",
                garbled_pages=_info.garbled_pages, noisy_pages=_info.noisy_pages,
                repaired_lines=_info.repaired_lines,
                read_page=_page_reader(file_path, lang, no_preprocess, no_fallback, no_figure,
                                       llm_correct, llm_budget, vision_first),
            )
            return

    # 사전 경고 구성 (렌더링 없이 페이지 수만 조회 — 즉시)
    pre_warnings: list[str] = []
    if file_mb >= WARN_SIZE_MB:
        pre_warnings.append(f"⚠ 대용량 파일 ({file_mb:.1f} MB)")

    yield (
        _progress_html(0, 0, "파일 분석 중...", elapsed_str=elapsed()),
        "\n".join(pre_warnings) if pre_warnings else "📂 파일을 읽는 중...",
        "", None, gr.update(visible=False), None,
    )

    # 페이지 계획 — 미리보기는 앞 PREVIEW_PAGES만 지연 렌더(전체 책 변환 안 함)
    try:
        total, pages = _page_plan(file_path, start_page=1, max_pages=PREVIEW_PAGES)
    except Exception as load_err:
        yield (
            _progress_html(0, 0, "오류", elapsed_str=elapsed()),
            f"❌ 파일 오류: {load_err}", "", None, gr.update(visible=False), None,
        )
        return

    sample_count  = min(PREVIEW_PAGES, total)
    result        = FileResult(source_path=file_path, output_path=None)
    step_logs: list[str] = []

    # 샘플 페이지 OCR (지연 렌더링 이터레이터)
    processed = 0
    for page_num, image in pages:
        done = processed
        pct  = int(done / sample_count * 100) if sample_count else 0

        yield (
            _progress_html(done, sample_count,
                           f"미리보기 {page_num}/{sample_count}페이지 처리 중...",
                           elapsed_str=elapsed()),
            _status_text(page_num, sample_count, pct, step_logs, elapsed()),
            "", None, gr.update(visible=False), None,
        )
        progress((done / sample_count) if sample_count else 0,
                 desc=f"미리보기 {page_num}/{sample_count}")

        # 빠른 모드로 처리
        # Vision 우선 모드이면 전처리 불필요, quick=False여도 Vision이 먼저 실행
        page_result, page_logs = _process_page(
            image, page_num, lang, no_preprocess, no_fallback,
            no_figure=True,
            llm_correct=False, llm_budget=0,
            quick=(not vision_first),   # Vision 우선이면 quick 전처리 불필요
            vision_first=vision_first,
        )
        result.pages.append(page_result)
        step_logs.extend(page_logs)
        processed += 1

        pct = int(processed / sample_count * 100) if sample_count else 100
        yield (
            _progress_html(processed, sample_count,
                           f"미리보기 {page_num}/{sample_count}페이지 완료",
                           elapsed_str=elapsed()),
            _status_text(page_num, sample_count, pct, step_logs, elapsed()),
            "", None, gr.update(visible=False), None,
        )
        progress((processed / sample_count) if sample_count else 1.0)

    # 미리보기 결과 저장
    tmp_dir  = Path(tempfile.mkdtemp(prefix="ocr_preview_"))
    out_path = writer.write(result, tmp_dir, out_fmt)

    # 미리보기: corrected_text(이미 전체 체인 처리 완료)에서 직접 조합
    # → 파일 재처리 없이 처리 결과 그대로 표시 (이중 처리 방지)
    preview = _build_preview(result)

    review_count = sum(1 for p in result.pages if p.needs_review)

    # 전체 처리 안내 패널 HTML
    remain = total - sample_count
    if remain > 0:
        avg_s     = (time.time() - start_time) / max(1, sample_count)
        remain_lo = _fmt_elapsed(avg_s * remain * 0.8)
        remain_hi = _fmt_elapsed(avg_s * remain * 1.5)
        quality_warn = ""
        if review_count:
            quality_warn = (
                f'<p style="color:#b45309;margin:6px 0 0">⚠ 미리보기 {sample_count}페이지 중 '
                f'{review_count}페이지 검토 필요 — {claude_vision.provider_label()} 폴백 권장</p>'
            )
        continue_html = f"""
<div class="continue-panel">
  <p style="font-weight:600;font-size:0.95rem;color:#15803d;margin:0 0 6px">
    ✅ 미리보기 완료 — 처음 {sample_count}페이지 추출 결과를 확인하세요.
  </p>
  <p style="color:#374151;margin:4px 0">
    📄 전체 <b>{total}페이지</b> 중 <b>{remain}페이지</b> 남음
    &nbsp;·&nbsp; 추가 예상 소요: <b>{remain_lo} ~ {remain_hi}</b>
  </p>
  {quality_warn}
  <p style="color:#6b7280;font-size:0.82rem;margin:8px 0 0">
    결과가 괜찮으면 위의 <b>「전체 처리」</b> 버튼으로 처음부터 끝까지 처리하세요.
  </p>
</div>"""
    else:
        continue_html = f"""
<div class="continue-panel">
  <p style="font-weight:600;color:#15803d;margin:0">
    ✅ 전체 {total}페이지 미리보기 완료 (파일이 {total}페이지 이하)
  </p>
</div>"""

    blank_count_preview = sum(1 for p in result.pages if p.is_blank)
    status_summary = [
        f"미리보기 완료: {sample_count}/{total}페이지  |  경과: {elapsed()}",
    ]
    if blank_count_preview:
        status_summary.append(f"⬜ 빈 페이지 {blank_count_preview}p 건너뜀")
    if review_count:
        status_summary.append(f"⚠ 검토 필요 페이지: {review_count}p")
    status_summary.extend(step_logs[-4:])

    yield (
        _progress_html(sample_count, total,
                       f"미리보기 완료 ({sample_count}/{total}p)",
                       elapsed_str=elapsed()),
        "\n".join(status_summary),
        preview,
        str(out_path),
        gr.update(visible=(remain > 0), value=continue_html),
        None,   # 전 페이지 캐시 보관 안 함 — '계속'은 지연 렌더로 이어서 처리(메모리 절약)
    )
    progress((sample_count / total) if total else 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# ② 전체 처리 (처음부터 전체 or 미리보기 이후 이어서)
# ══════════════════════════════════════════════════════════════════════════════

def process_full(
    file_obj, lang, fmt, no_preprocess, no_fallback, no_figure,
    llm_correct, llm_budget, api_key_input, vision_first_str,
    cleanup=True, front_matter=False,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    """전체 페이지 처리 (처음부터)."""
    vision_first = "Vision 우선" in str(vision_first_str)
    yield from _run_ocr(
        file_obj, lang, fmt, no_preprocess, no_fallback, no_figure,
        llm_correct, llm_budget, api_key_input,
        cached_images=None, start_page=1, label="전체 처리",
        vision_first=vision_first, cleanup=cleanup, front_matter=front_matter,
        progress=progress,
    )


def process_full_or_queue(
    file_obj, lang, fmt, no_preprocess, no_fallback, no_figure,
    llm_correct, llm_budget, api_key_input, vision_first_str,
    cleanup=True, front_matter=False,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    """① 탭 '전체 처리'. ④ 일괄 처리 대기열이 돌고 있으면 겹치지 않게 대기열 끝에 넣는다.
    여러 파일을 올렸으면 올린 순서대로 하나씩 처리한다."""
    rest = (lang, fmt, no_preprocess, no_fallback, no_figure,
            llm_correct, llm_budget, api_key_input, vision_first_str)
    files = _as_file_list(file_obj)
    if files and batch_runner.busy():
        batch_runner.submit([f.name for f in files], rest,
                            {"cleanup": cleanup, "front_matter": front_matter})
        yield (_progress_html(0, 0, "대기열에 추가됨"),
               "④ 일괄 처리 대기열이 돌고 있어 끝에 추가했습니다. ④ 탭에서 진행 상황을 볼 수 있습니다.",
               "", None, gr.update(visible=False), None)
        return
    batch_runner.direct_running.set()
    try:
        if len(files) <= 1:
            yield from process_full(files[0] if files else None, *rest, cleanup=cleanup,
                                    front_matter=front_matter, progress=progress)
        else:
            yield from _process_many(files, rest, cleanup, front_matter, progress)
    finally:
        batch_runner.direct_running.clear()


def _as_file_list(files) -> list:
    """① 탭 파일 입력(여러 개 허용) → 리스트. 단일 값(④ 대기열의 SimpleNamespace 등)도 받는다."""
    if not files:
        return []
    return list(files) if isinstance(files, list) else [files]


def _first_file(files):
    lst = _as_file_list(files)
    return lst[0] if lst else None


def _process_many(files: list, rest: tuple, cleanup, front_matter, progress):
    """여러 파일을 차례로 전체 처리. 한 파일이 실패해도 다음 파일로 넘어간다."""
    n = len(files)
    saved: list[str] = []
    failed: list[str] = []
    preview = ""
    for i, f in enumerate(files, 1):
        name = Path(f.name).name
        head = f"[{i}/{n}] {name}"
        last = None
        try:
            for last in process_full(f, *rest, cleanup=cleanup, front_matter=front_matter,
                                     progress=progress):
                bar, status, preview_i, out, _cont, _imgs = last
                yield (bar, f"{head}\n{status}", preview_i or preview,
                       saved + [str(out)] if out else (saved or None),
                       gr.update(visible=False), None)
        except Exception as e:   # 한 파일의 실패가 나머지 파일을 막지 않게
            failed.append(f"{name} — {type(e).__name__}: {e}")
            continue
        out = last[3] if last else None
        if out:
            saved.append(str(out))
            preview = last[2] or preview
        else:
            failed.append(f"{name} — {_last_status_line(last[1]) if last else '결과 없음'}")
    summary = [f"여러 파일 처리 끝: 성공 {len(saved)}개 / 실패 {len(failed)}개 (전체 {n}개)"]
    summary += [f"✅ {Path(p).name}" for p in saved]
    summary += [f"❌ {x}" for x in failed]
    yield (_progress_html(n, n, f"{n}개 파일 처리 완료"), "\n".join(summary), preview,
           saved or None, gr.update(visible=False), None)


def _last_status_line(text) -> str:
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def process_continue(
    file_obj, lang, fmt, no_preprocess, no_fallback, no_figure,
    llm_correct, llm_budget, api_key_input, vision_first_str,
    cleanup, front_matter,
    images_state,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    """미리보기 이후 나머지 페이지를 이어서 처리."""
    file_obj = _first_file(file_obj)   # 미리보기와 같은 첫 파일
    vision_first = "Vision 우선" in str(vision_first_str)
    batch_runner.direct_running.set()   # 대기열은 이 처리가 끝날 때까지 다음 파일을 시작하지 않는다
    try:
        yield from _run_ocr(
            file_obj, lang, fmt, no_preprocess, no_fallback, no_figure,
            llm_correct, llm_budget, api_key_input,
            cached_images=images_state, start_page=PREVIEW_PAGES + 1,
            label="이어서 전체 처리",
            vision_first=vision_first, cleanup=cleanup, front_matter=front_matter,
            progress=progress,
        )
    finally:
        batch_runner.direct_running.clear()


def _run_ocr(
    file_obj, lang, fmt, no_preprocess, no_fallback, no_figure,
    llm_correct, llm_budget, api_key_input,
    cached_images, start_page: int, label: str,
    vision_first: bool = False,
    cleanup: bool = False,
    front_matter: bool = False,
    progress = None,
):
    """공통 OCR 처리 루프. cached_images 있으면 PDF 재변환 생략."""
    start_time = time.time()
    elapsed    = lambda: _fmt_elapsed(time.time() - start_time)

    if file_obj is None:
        yield (_progress_html(0, 0, "대기"), "", "", None, gr.update(visible=False), None)
        return

    if api_key_input.strip():
        claude_vision.set_session_key(api_key_input)

    file_path = Path(file_obj.name)
    out_fmt   = OutputFormat.MD if fmt == "md" else OutputFormat.TXT
    file_mb   = file_path.stat().st_size / 1_048_576

    # ── 디지털 PDF 우선 경로 (신규 처리 시에만; 이어서-처리는 이미지 캐시 사용) ──
    # 텍스트 레이어가 있으면 렌더·OCR·Vision API를 건너뛰고 즉시 직접 추출한다.
    # 스캔 PDF는 is_digital=False → 아래 기존 OCR 경로로 자동 전환.
    if not cached_images and file_path.suffix.lower() == ".pdf" and pdf_text_layer.is_available():
        _info = pdf_text_layer.analyze(file_path, repair=True)
        if _info and _info.is_digital:
            yield from _run_digital_pdf(
                file_path, _info.pages, out_fmt, cleanup, front_matter, start_time,
                label=label,
                garbled_pages=_info.garbled_pages, noisy_pages=_info.noisy_pages,
                repaired_lines=_info.repaired_lines,
                read_page=_page_reader(file_path, lang, no_preprocess, no_fallback, no_figure,
                                       llm_correct, llm_budget, vision_first),
            )
            return

    pre_warnings: list[str] = []
    if file_mb >= WARN_SIZE_MB and not cached_images:
        pre_warnings.append(f"⚠ 대용량 파일 ({file_mb:.1f} MB)")

    # 페이지 계획 (대용량 PDF는 지연 렌더링 — 전 페이지 일괄 변환/메모리 점유 없음)
    if cached_images:
        total = len(cached_images)

        def _src():
            for k, img in enumerate(cached_images[start_page - 1:]):
                yield start_page + k, img

        pages = _src()
    else:
        yield (
            _progress_html(0, 0, "파일 분석 중...", elapsed_str=elapsed()),
            "\n".join(pre_warnings) if pre_warnings else "📂 파일을 읽는 중...",
            "", None, gr.update(visible=False), None,
        )
        try:
            total, pages = _page_plan(file_path, start_page=start_page)
        except Exception as load_err:
            yield (
                _progress_html(0, 0, "오류", elapsed_str=elapsed()),
                f"❌ 파일 오류: {load_err}", "", None, gr.update(visible=False), None,
            )
            return

    result      = FileResult(source_path=file_path, output_path=None)
    step_logs: list[str] = []

    processed = 0
    # OCR 원문을 쪽마다 02_결과/{원본명}_OCR중.txt 에 쌓는다 — 중단·크래시에도 이미 읽은
    # 쪽(유료 Vision 포함)이 남는다. 정상 확정 후 지운다. (QA ISSUE-B, 2026-10-03)
    ocr_ckpt = _CleanCheckpoint(file_path.stem, OutputFormat.TXT, suffix="OCR중")
    def _read(image, page_num):
        try:
            return _process_page(
                image, page_num, lang, no_preprocess, no_fallback, no_figure,
                llm_correct, llm_budget, vision_first=vision_first,
            )
        except Exception as page_err:   # 한 쪽의 예외가 파일 전체를 죽이지 않게 — 검토 필요로 남김
            reason = f"처리 오류: {page_err}"
            return (PageResult(page_num=page_num, raw_text="", corrected_text="",
                               engine=OCREngine.TESSERACT, confidence=0.0,
                               needs_review=True, review_reason=reason),
                    [f"  p.{page_num}: ⚠ {reason} — 이 쪽은 비워 두고 계속"])

    # 쪽 읽기를 parallel_workers 개까지 동시에 보낸다(Vision 1쪽 약 8초 — 대부분 응답 대기).
    # 이미지는 이 스레드에서 차례로 그리고(렌더러는 스레드 안전이 아님), 결과는 쪽 순서대로
    # 저장한다. 동시에 메모리에 있는 쪽 이미지는 최대 workers 장.
    workers = config.get_parallel_workers()
    executor = ThreadPoolExecutor(max_workers=workers)
    inflight: deque = deque()
    render_err = None
    page_iter = iter(pages)
    try:
        while True:
            while len(inflight) < workers and render_err is None:
                try:
                    page_num, image = next(page_iter)
                except StopIteration:
                    break
                except Exception as e:   # 손상 페이지 렌더 실패 등 — 읽은 쪽까지는 저장
                    render_err = e
                    break
                inflight.append((page_num, executor.submit(_read, image, page_num)))
            if not inflight:
                break
            page_num, fut = inflight.popleft()
            before = start_page - 1 + processed
            pct    = int(before / total * 100) if total else 0

            yield (
                _progress_html(before, total,
                               f"{label}: {page_num}/{total}페이지 처리 중...",
                               elapsed_str=elapsed()),
                _status_text(page_num, total, pct, step_logs, elapsed()),
                "", None, gr.update(visible=False), None,
            )
            progress((before / total) if total else 0, desc=f"{page_num}/{total}")

            page_result, page_logs = fut.result()
            result.pages.append(page_result)
            step_logs.extend(page_logs)
            ocr_ckpt.append(f"[페이지 {page_num}]\n{page_result.corrected_text}")
            processed += 1

            done_total = start_page - 1 + processed
            pct = int(done_total / total * 100) if total else 100

            # 남은 시간 추정
            elapsed_s    = time.time() - start_time
            avg_per_page = elapsed_s / processed
            remain_pages = total - done_total
            remain_str   = f"\n  남은 시간 약 {_fmt_elapsed(avg_per_page * remain_pages)}" if remain_pages > 0 else ""

            yield (
                _progress_html(done_total, total,
                               f"{label}: {page_num}/{total}페이지 완료",
                               elapsed_str=elapsed()),
                _status_text(page_num, total, pct, step_logs, elapsed()) + remain_str,
                "", None, gr.update(visible=False), None,
            )
            progress((done_total / total) if total else 1.0)
    finally:
        # 중단(제너레이터 close) 시 아직 안 보낸 쪽은 취소 — 이미 보낸 쪽만 마저 끝난다
        executor.shutdown(wait=False, cancel_futures=True)
    if render_err is not None:
        step_logs.append(f"  ⚠ {start_page + processed}쪽 이후를 읽지 못해 중단 — "
                         f"처리한 {processed}쪽만 저장합니다 ({render_err})")

    # ── 문서 전체 정제(교정) ────────────────────────────────────────────────
    # OCR 원문(페이지 충실 추출)을 문서 전체 맥락에서 Gemini로 교정:
    #   쪽번호·러닝헤더 제거 · 페이지 넘김으로 잘린 단어/문단 복원 · 띄어쓰기·맞춤법.
    # 결과가 있으면 '마커 없는 클린 플레인텍스트'로 저장한다.
    cleaned_text = None
    ckpt = None
    content_pages = [p for p in result.pages if not p.is_blank]
    if cleanup and content_pages and text_cleaner.is_available():
        ckpt = _CleanCheckpoint(file_path.stem, out_fmt)
        crep: dict = {}     # 이 실행의 정제 보고서(여러 파일 동시 처리 시 전역 보고서는 섞인다)
        clean_gen = text_cleaner.iter_clean(
            [p.corrected_text for p in content_pages],
            drop_front_matter=front_matter,
            on_chunk=ckpt.append,
            report_out=crep,
        )
        try:
            while True:
                done_c, total_c = next(clean_gen)
                yield (
                    # 진행률은 '구간' 기준 — 페이지 기준(=이미 100%)으로 그리면
                    # 정제가 남았는데도 화면이 완료처럼 보인다.
                    _progress_html(done_c, total_c,
                                   f"📝 문장 정제·교정 중 ({done_c}/{total_c} 구간)",
                                   elapsed_str=elapsed(), unit="구간"),
                    _status_text(total, total, 100, step_logs, elapsed())
                    + f"\n  📝 정제(교정) 진행: {done_c}/{total_c} 구간 (페이지 잡음 제거·띄어쓰기·맞춤법)"
                    + f"\n  💾 중간 저장 중: {ckpt.path}",
                    "", None, gr.update(visible=False), None,
                )
        except StopIteration as stop:
            cleaned_text = stop.value or ""
        if cleaned_text:
            step_logs.append(f"  📝 문장 정제 완료 ({len(cleaned_text):,}자, 마커 없는 클린 텍스트)")
        _crep = crep or None
        if _crep and (_crep["fallback_guard"] or _crep["fallback_api"]):
            step_logs.append(
                f"  ⚠️ 정제 무결성 가드 발동 {_crep['fallback_guard'] + _crep['fallback_api']}"
                f"/{_crep['chunks']} 구간 — LLM 교정 대신 원문 보존(내용 무손실)")
        if _crep:   # 부속물 제거·지어낸 문장 등 확인할 경고 — 강등 건은 위 개수 줄로 대신한다
            step_logs.extend(f"  ⚠ {w}" for w in _crep["warnings"] if "강등" not in w)
    elif cleanup and not text_cleaner.is_available():
        step_logs.append("  ⚠ 정제 건너뜀 — Gemini 키 필요(현재 Vision=Gemini 모드에서만 정제 지원)")

    # 저장
    yield (
        _progress_html(total, total, "결과 파일 저장 중...", elapsed_str=elapsed()),
        _status_text(total, total, 100, step_logs, elapsed()),
        "", None, gr.update(visible=False), None,
    )
    tmp_dir = Path(tempfile.mkdtemp(prefix="ocr_output_"))
    raw_path = None
    try:
        if cleaned_text:
            out_path = writer.write_plain(file_path, cleaned_text, tmp_dir, out_fmt)
            # 정제 전 원문(페이지 마커 포함)도 별도 보존 — 정제 단계에서 소실이
            # 생겨도 항상 원문으로 복구할 수 있는 안전망 (메시지성경 사고 재발 방지)
            try:
                raw_path = writer.write(result, tmp_dir / "raw", out_fmt)
            except Exception as e:
                # 안전망이 없다는 사실을 숨기면 '완료'만 보고 원문을 믿게 된다 — 표면화
                raw_path = None
                step_logs.append(f"  ⚠ 정제 전 원문 사본을 만들지 못했습니다(복구용 원문 없음): {e}")
        else:
            out_path = writer.write(result, tmp_dir, out_fmt)
    except Exception as e:
        if ckpt is not None:
            ckpt.close()   # 중간 저장분(_정제중)은 회수용으로 남긴다
        yield (_progress_html(total, total, "저장 오류"),
               f"❌ {e}\n   💾 정제 중간 저장분: {ckpt.path if ckpt else '없음'}",
               "", None, gr.update(visible=False), None)
        return

    # 미리보기: 정제본이 있으면 정제본을, 없으면 corrected_text 조합을 표시
    if cleaned_text:
        preview = (cleaned_text[:4000] + "\n\n... (이하 생략)"
                   if len(cleaned_text) > 4000 else cleaned_text)
    else:
        preview = _build_preview(result)
    review_count = sum(1 for p in result.pages if p.needs_review)
    vision_count = sum(1 for p in result.pages
                       if p.engine in (OCREngine.GEMINI, OCREngine.CLAUDE_VISION, OCREngine.OLLAMA))
    blank_count  = sum(1 for p in result.pages if p.is_blank)
    ocr_count    = len(result.pages) - blank_count
    total_el     = elapsed()

    # ── 02_결과/ 확정 — 이력에 경로를 남기려고 기록보다 먼저 ──────────────────────
    results_path = _finalize_results(ckpt, cleaned_text, out_path, file_path.stem, step_logs)
    ocr_ckpt.close()
    if results_path is not None:          # 확정됐을 때만 OCR 복구본을 지운다
        ocr_ckpt.path.unlink(missing_ok=True)

    # ── storage: 결과 파일 영구 저장 + 이력 기록 ─────────────────────────────
    stored_path   = None
    stored_upload = None
    try:
        stored_path = storage_manager.save_result(out_path, file_path.name)
        if raw_path is not None:
            try:
                storage_manager.save_raw_copy(raw_path, stored_path)
            except Exception as e:
                step_logs.append(f"  ⚠ 정제 전 원문 사본을 보관소에 저장하지 못했습니다(복구용 원문 없음): {e}")
        if _should_save_original():
            stored_upload = storage_manager.save_original(file_path, file_path.name)
        storage_manager.record(
            original_name   = file_path.name,
            result_path     = stored_path,
            upload_path     = stored_upload,
            pages_total     = total,
            pages_ocr       = ocr_count,
            pages_blank     = blank_count,
            vision_pages    = vision_count,
            engine_mode     = "vision_first" if vision_first else "tesseract_first",
            file_size_bytes = file_path.stat().st_size,
            output_path     = results_path,
        )
    except Exception as e:
        # 조용히 넘기면 화면엔 '완료'만 뜨고 파일은 없다 — 반드시 표면화한다.
        step_logs.append(f"  ⚠ 이력 보관소 저장 실패: {e}")

    summary = [
        f"✅ 완료  |  전체 {total}p → OCR 처리 {ocr_count}p  |  총 소요: {total_el}",
        f"   Vision 처리: {vision_count}p",
    ]
    if blank_count:
        summary.append(f"   ⬜ 빈 페이지 건너뜀: {blank_count}p (크레딧 절약)")
    if review_count:
        summary.append(f"   ⚠ 검토 필요: {review_count}p (출력 파일 상단 참조)")
    summary.append(f"   📁 저장 위치: {results_path or out_path}")
    if step_logs:
        summary.append("")
        summary.extend(step_logs[-5:])

    yield (
        _progress_html(total, total, f"완료  ·  총 {total_el}",
                       elapsed_str=total_el, done=True),
        "\n".join(summary),
        preview,
        str(results_path or out_path),
        gr.update(visible=False),
        None,
    )
    progress(1.0, desc="완료")


# ══════════════════════════════════════════════════════════════════════════════
# Gradio UI
# ══════════════════════════════════════════════════════════════════════════════

def build_ui() -> gr.Blocks:
    cfg     = config.load()
    has_key = _engine_ok(fresh=True)   # 활성 provider(Gemini/Claude/Ollama) 기준

    with gr.Blocks(title="OCR 텍스트추출 도구", fill_width=True) as demo:

        images_state = gr.State(None)
        # API 키는 화면에서 받지 않는다(환경변수 전용). 처리 함수의 인자 순서를
        # 유지하기 위해 빈 값을 넘기는 자리만 남긴다.
        api_key_input = gr.State("")

        header_html = gr.HTML(_header_html())

        with gr.Tabs():

            # ══════════════════════════════════════════════════════════════════
            # 탭 1: 처리하기 — 1 파일 → 2 방식 → 3 실행 → 4 결과
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("① 처리하기"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=320):

                        with gr.Column(elem_classes="step-card"):
                            gr.HTML(_step_head(1, "파일 선택",
                                    "처리할 PDF나 이미지를 올리세요. 여러 개를 끌어다 놓으면 "
                                    "「전체 처리」 때 올린 순서대로 하나씩 처리합니다."))
                            file_input = gr.File(
                                show_label=False,
                                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"],
                                file_count="multiple",
                                height=150,
                            )
                            file_info = gr.HTML(_file_info_html(None))

                        with gr.Column(elem_classes="step-card"):
                            gr.HTML(_step_head(3, "실행",
                                    f"처음이라면 앞 {PREVIEW_PAGES}쪽 미리보기로 품질을 먼저 확인하세요. "
                                    "디지털 PDF는 바로 전체 처리해도 됩니다."))
                            with gr.Row():
                                preview_btn = gr.Button(f"앞 {PREVIEW_PAGES}쪽 미리보기",
                                                        variant="secondary", size="lg")
                                run_btn = gr.Button("전체 처리", variant="primary", size="lg")
                            stop_btn = gr.Button("처리 중단", variant="stop", size="sm", visible=False)

                    with gr.Column(scale=4, min_width=320):
                        with gr.Column(elem_classes="step-card"):
                            gr.HTML(_step_head(2, "처리 방식",
                                    "스캔본을 읽는 순서와 교정 여부를 고릅니다. "
                                    "디지털 PDF는 OCR 없이 바로 추출하므로 읽는 방식은 쓰이지 않습니다."))
                            vision_first = gr.Radio(
                                label="읽는 방식",
                                choices=_vision_choices(),
                                value="Vision 우선" if has_key else "Tesseract 우선",
                            )
                            vision_first_note = gr.HTML(
                                _vision_mode_note("Vision 우선" if has_key else "Tesseract 우선"))
                            cleanup_chk = gr.Checkbox(
                                label="문장 정제 — 쪽번호·머리글 제거, 띄어쓰기·맞춤법 교정, 문단 복원",
                                value=bool(cfg.get("enable_cleanup", True)) and text_cleaner.is_available(),
                                info=_cleanup_info(),
                            )
                            front_matter_chk = gr.Checkbox(
                                label="표지·판권·차례 같은 앞뒤 부속물도 삭제",
                                value=bool(config.get("remove_front_matter", False)),
                                info="문장 정제가 켜져 있을 때만 동작합니다. 끄면(기본) 부속물을 그대로 둡니다.",
                            )
                            with gr.Accordion("고급 설정 — 보통은 그대로 두세요", open=False):
                                with gr.Row():
                                    lang_input = gr.Dropdown(
                                        label="OCR 언어",
                                        choices=["kor+eng", "kor", "eng", "kor+eng+chi_sim"],
                                        value=cfg.get("default_lang", "kor+eng"),
                                    )
                                    fmt_input = gr.Radio(label="저장 형식", choices=["txt", "md"],
                                                         value=cfg.get("default_format", "txt"))
                                no_preprocess = gr.Checkbox(label="이미지 전처리 끄기", value=False)
                                no_fallback   = gr.Checkbox(label="흐린 쪽 자동 재인식(폴백) 끄기",
                                                            value=False)
                                no_figure     = gr.Checkbox(label="그림·표 감지 끄기", value=False)
                                margin_filter_chk = gr.Checkbox(
                                    label="위아래 여백(머리글·바닥글·쪽번호) 잘라내고 읽기",
                                    value=bool(cfg.get("enable_margin_filter", False)),
                                )
                                with gr.Row():
                                    margin_top_sld = gr.Slider(
                                        label="위 여백 비율", minimum=0.0, maximum=0.5, step=0.01,
                                        value=float(cfg.get("margin_top_ratio", 0.06)),
                                    )
                                    margin_bottom_sld = gr.Slider(
                                        label="아래 여백 비율", minimum=0.0, maximum=0.5, step=0.01,
                                        value=float(cfg.get("margin_bottom_ratio", 0.06)),
                                    )
                                llm_correct = gr.Checkbox(
                                    label="쪽마다 한 번 더 교정 (L3 · Anthropic Claude Haiku)",
                                    value=False,   # 기본 OFF — 켜 두면 엔진과 별개로 쪽마다 Claude 유료 호출(2026-10-03)
                                    info="환경변수 ANTHROPIC_API_KEY 필요 · A4 1쪽 약 $0.0004. "
                                         "선택한 엔진과 별도로 Claude 요금이 듭니다. 결과 분량이 원문과 크게 다르면 원문을 유지합니다.",
                                )
                                llm_budget = gr.Slider(label="L3 최대 호출 횟수 (0 = 제한 없음)",
                                                       minimum=0, maximum=200, step=10, value=0)

                    with gr.Column(scale=5, min_width=380):
                        with gr.Column(elem_classes="step-card"):
                            gr.HTML(_step_head(4, "결과",
                                    "진행 상황과 추출된 본문이 여기에 나타납니다."))
                            progress_bar = gr.HTML(value=_progress_html(0, 0, "파일을 기다리는 중"))
                            status_out = gr.Textbox(
                                label="진행 기록", lines=3, interactive=False,
                                elem_classes="step-log",
                                placeholder="처리를 시작하면 단계별 기록이 표시됩니다.")
                            continue_html = gr.HTML(visible=False)
                            with gr.Row(visible=False) as continue_row:
                                continue_btn  = gr.Button("나머지 쪽 계속 처리", variant="primary")
                                save_only_btn = gr.Button("미리보기까지만 저장", variant="secondary")
                            preview_out = gr.Textbox(
                                label="본문 미리보기 (앞 4,000자)",
                                lines=8, interactive=False,
                                placeholder="처리가 끝나면 추출된 본문 앞부분이 여기에 표시됩니다.")
                            download_out = gr.File(label="결과 파일", height=90,
                                                   file_count="multiple")
                            gr.HTML(_results_location_html())
                            open_folder_btn = gr.Button("결과 폴더 열기", variant="secondary", size="sm")
                            folder_msg = gr.Markdown("")

            # ══════════════════════════════════════════════════════════════════
            # 탭 2: 엔진 · API 키 — 키는 환경변수로만 적용(화면 입력 없음)
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("② 엔진 · API 키") as engine_tab:
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=320):
                        with gr.Column(elem_classes="step-card"):
                            gr.HTML(_step_head(None, "사용할 엔진",
                                    "스캔본을 읽을 AI 엔진입니다. 바꾸면 바로 저장되고 다음 처리부터 적용됩니다."))
                            provider_radio = gr.Radio(
                                show_label=False,
                                choices=_PROVIDER_CHOICES,
                                value=claude_vision.provider_name(),
                            )
                            provider_msg = gr.Markdown("")
                        with gr.Accordion("기본 설정 (다음 실행부터 적용)", open=False):
                            cfg_lang = gr.Dropdown(
                                label="기본 OCR 언어",
                                choices=["kor+eng", "kor", "eng"],
                                value=cfg.get("default_lang", "kor+eng"),
                            )
                            cfg_fmt = gr.Radio(
                                label="기본 저장 형식",
                                choices=["txt", "md"],
                                value=cfg.get("default_format", "txt"),
                            )
                            cfg_fallback = gr.Checkbox(
                                label="흐린 쪽(신뢰도 70% 미만) 자동 재인식",
                                value=cfg.get("auto_fallback", True),
                            )
                            cfg_dpi = gr.Radio(
                                label="PDF 변환 해상도 (DPI) — 낮을수록 빠르고 가벼움",
                                choices=[150, 200, 300],
                                value=cfg.get("default_dpi", 200),
                                info="Vision은 내부에서 축소하므로 200 권장. 책 스캔은 150도 충분합니다.",
                            )
                            cfg_margin_filter = gr.Checkbox(
                                label="여백 잘라내기 기본으로 켜기",
                                value=bool(cfg.get("enable_margin_filter", False)),
                            )
                            with gr.Row():
                                cfg_margin_top = gr.Slider(
                                    label="기본 위 여백 비율", minimum=0.0, maximum=0.5, step=0.01,
                                    value=float(cfg.get("margin_top_ratio", 0.06)),
                                )
                                cfg_margin_bottom = gr.Slider(
                                    label="기본 아래 여백 비율", minimum=0.0, maximum=0.5, step=0.01,
                                    value=float(cfg.get("margin_bottom_ratio", 0.06)),
                                )
                            save_cfg_btn = gr.Button("기본 설정 저장", variant="secondary")
                            save_cfg_msg = gr.Markdown("")

                    with gr.Column(scale=4, min_width=320):
                        with gr.Column(elem_classes="step-card"):
                            gr.HTML(_step_head(None, "API 키 적용 방법",
                                    "키는 화면에 입력하지 않습니다. Windows 환경변수에 등록하면 "
                                    "프로그램이 시작할 때 자동으로 읽습니다."))
                            engine_guide = gr.HTML(_engine_guide_html())
                            recheck_btn = gr.Button("연결 상태 다시 확인", variant="secondary", size="sm")
                            billing_html = gr.HTML(_billing_html())

                    with gr.Column(scale=5, min_width=380):
                        with gr.Column(elem_classes="step-card"):
                            gr.HTML(_step_head(None, "API 사용량",
                                    "이 프로그램이 호출한 토큰으로 계산한 추정치입니다."))
                            usage_display = gr.Textbox(
                                show_label=False,
                                value=usage_tracker.format_summary(),
                                lines=8,
                                interactive=False,
                                elem_classes="step-log",
                            )
                            with gr.Row():
                                refresh_btn = gr.Button("새로고침", variant="secondary", size="sm")
                                reset_btn   = gr.Button("통계 초기화", variant="stop", size="sm")
                            reset_msg = gr.Markdown("")
                            gr.Markdown("**최근 호출 20건**")
                            history_display = gr.HTML(value=_usage_table_html())

            # ══════════════════════════════════════════════════════════════════
            # 탭 3: 처리 기록
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("③ 처리 기록") as records_tab:
                with gr.Row(equal_height=False):
                    with gr.Column(scale=7, min_width=480):
                        with gr.Column(elem_classes="step-card"):
                            gr.HTML(_step_head(None, "저장 현황",
                                    "처리한 결과는 02_결과 폴더에 저장되고, 이력과 사본은 내부 보관소에 남습니다."))
                            stats_html = gr.HTML(value=_storage_stats_html())
                            file_history = gr.HTML(value=_records_table_html())
                            refresh_files_btn = gr.Button("새로고침", variant="secondary", size="sm")

                    with gr.Column(scale=4, min_width=340):
                        with gr.Column(elem_classes="step-card"):
                            gr.HTML(_step_head(None, "기록 선택",
                                    "목록에서 항목을 고른 뒤 다시 받거나 삭제합니다."))
                            pick_entry = gr.Dropdown(
                                label="처리 기록",
                                choices=_history_choices(),
                                value=None,
                            )
                            with gr.Row():
                                dl_type = gr.Radio(
                                    label="받을 파일",
                                    choices=["결과 파일 (텍스트)", "원본 파일"],
                                    value="결과 파일 (텍스트)",
                                )
                                dl_btn = gr.Button("다운로드", variant="primary")
                            dl_out = gr.File(label="다운로드 파일")
                            dl_msg = gr.Markdown("")
                            with gr.Row():
                                del_files_chk = gr.Checkbox(label="저장된 파일도 함께 삭제 (02_결과 포함)", value=False)
                                del_entry_btn = gr.Button("선택한 기록 삭제", variant="stop", size="sm")
                            del_msg = gr.Markdown("")

                        with gr.Accordion("보관 설정 · 전체 초기화", open=False):
                            with gr.Row():
                                save_orig_chk = gr.Checkbox(
                                    label="올린 원본 파일도 내부 보관소에 함께 저장",
                                    value=bool(config.get("save_original_files", False)),
                                )
                                save_storage_btn = gr.Button("보관 설정 저장", variant="secondary", size="sm")
                            storage_cfg_msg = gr.Markdown("")
                            with gr.Row():
                                clear_all_chk = gr.Checkbox(label="저장된 파일도 함께 삭제 (02_결과 포함)", value=False)
                                clear_all_btn = gr.Button("처리 기록 전체 삭제", variant="stop", size="sm")
                            clear_msg = gr.Markdown("")

            # ══════════════════════════════════════════════════════════════════
            # 탭 4: 일괄 처리 — 01_입력 폴더에서 골라 대기열로 차례 처리
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("④ 일괄 처리") as batch_tab:
                with gr.Column(elem_classes="step-card"):
                    gr.HTML(_step_head(None, "일괄 처리 방식 설정",
                            "④ 탭 일괄 처리에서 독립적으로 적용할 읽는 방식과 정제 옵션을 지정합니다. 기본값은 비용 우선입니다."))
                    with gr.Row():
                        batch_vision_first = gr.Radio(
                            label="읽는 방식",
                            choices=_vision_choices(),
                            value="Tesseract 우선",  # 기본값: 비용 우선 (RapidOCR 한글 v5)
                        )
                    batch_vision_note = gr.HTML(_vision_mode_note("Tesseract 우선"))
                    with gr.Row():
                        batch_cleanup_chk = gr.Checkbox(
                            label="문장 정제 — 쪽번호·머리글 제거, 띄어쓰기·맞춤법 교정, 문단 복원",
                            value=bool(cfg.get("enable_cleanup", True)) and text_cleaner.is_available(),
                            info=_cleanup_info(),
                        )
                        batch_front_matter_chk = gr.Checkbox(
                            label="표지·판권·차례 같은 앞뒤 부속물도 삭제",
                            value=bool(config.get("remove_front_matter", False)),
                            info="문장 정제가 켜져 있을 때만 동작합니다.",
                        )

                with gr.Column(elem_classes="step-card"):
                    gr.HTML(_step_head(None, "01_입력 폴더",
                            "폴더에 넣은 파일을 골라 차례로 처리합니다. 위에서 지정한 처리 방식으로 대기열에서 자동 실행됩니다."))
                    batch_note = gr.HTML("")
                    batch_list = gr.CheckboxGroup(show_label=False, choices=[], value=[],
                                                  elem_classes="batch-list")
                    with gr.Row():
                        batch_refresh_btn = gr.Button("새로고침", size="sm")
                        batch_pick_btn = gr.Button("미처리 전체 선택", size="sm")
                        batch_run_btn = gr.Button("선택 파일 일괄 처리 ▶", variant="primary")
                        batch_stop_btn = gr.Button("대기열 중단 ■", variant="stop", size="sm")
                        batch_clear_btn = gr.Button("끝난 항목 지우기", size="sm")
                with gr.Column(elem_classes="step-card"):
                    gr.HTML(_step_head(None, "대기열", "결과는 02_결과 폴더에 저장되고 ③ 처리 기록에도 남습니다."))
                    batch_status = gr.HTML(batch_runner.status_html())
                batch_timer = gr.Timer(3)

        # ── 이벤트 바인딩 ─────────────────────────────────────────────────────

        # show_progress="hidden": Gradio 내장 진행률 오버레이는 이벤트의 '모든'
        # 출력 컴포넌트 위에 같은 바를 겹쳐 그린다 — 진행바·처리 상태·미리보기에
        # 동일한 바가 3개 뜨는 원인이었다. 이 도구는 progress_bar(HTML)로 직접
        # 그리므로 내장 오버레이는 끄고 각 블록이 제 내용을 보이게 한다.
        common_inputs  = [file_input, lang_input, fmt_input,
                          no_preprocess, no_fallback, no_figure,
                          llm_correct, llm_budget, api_key_input, vision_first,
                          cleanup_chk, front_matter_chk]
        common_outputs = [progress_bar, status_out, preview_out, download_out,
                          continue_html, images_state]

        # 마진 설정 실시간 업데이트 도우미
        def _update_margin_config(margin_filter, margin_top, margin_bottom):
            config.save({
                "enable_margin_filter": margin_filter,
                "margin_top_ratio": float(margin_top),
                "margin_bottom_ratio": float(margin_bottom)
            })

        # OCR 처리 이벤트
        prev_event = preview_btn.click(
            fn=_update_margin_config, inputs=[margin_filter_chk, margin_top_sld, margin_bottom_sld]
        ).then(
            fn=lambda: gr.update(visible=True), outputs=[stop_btn]
        ).then(fn=process_preview, inputs=common_inputs, outputs=common_outputs,
               show_progress="hidden"
        ).then(fn=lambda imgs: (gr.update(visible=imgs is not None), gr.update(visible=False)),
               inputs=[images_state], outputs=[continue_row, stop_btn])

        run_event = run_btn.click(
            fn=_update_margin_config, inputs=[margin_filter_chk, margin_top_sld, margin_bottom_sld]
        ).then(
            fn=lambda: gr.update(visible=True), outputs=[stop_btn]
        ).then(fn=process_full_or_queue, inputs=common_inputs, outputs=common_outputs,
               show_progress="hidden"
        ).then(fn=lambda _: (gr.update(visible=False), gr.update(visible=False)),
               inputs=[images_state], outputs=[continue_row, stop_btn])

        cont_event = continue_btn.click(
            fn=lambda: gr.update(visible=True), outputs=[stop_btn]
        ).then(fn=process_continue, inputs=common_inputs + [images_state],
               outputs=common_outputs,  # common_inputs에 이미 vision_first 포함
               show_progress="hidden"
        ).then(fn=lambda _: (gr.update(visible=False), gr.update(visible=False)),
               inputs=[images_state], outputs=[continue_row, stop_btn])

        stop_btn.click(fn=lambda: gr.update(visible=False), outputs=[stop_btn],
                       cancels=[prev_event, run_event, cont_event])
        save_only_btn.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=False)),
            outputs=[continue_html, continue_row])
        file_input.change(
            fn=lambda f: (
                _progress_html(0, 0, "준비됨" if f else "파일을 기다리는 중"),
                "", "", None, gr.update(visible=False), None,
            ),
            inputs=[file_input], outputs=common_outputs,
        ).then(fn=lambda _: (gr.update(visible=False), gr.update(visible=False)),
               inputs=[images_state], outputs=[continue_row, stop_btn]
        ).then(fn=_file_info_html, inputs=[file_input], outputs=[file_info])

        # 읽는 방식 변경 시 안내 업데이트
        vision_first.change(fn=_vision_mode_note, inputs=[vision_first],
                            outputs=[vision_first_note])

        open_folder_btn.click(fn=_open_results_folder, outputs=[folder_msg])

        # ④ 일괄 처리
        def _batch_list(select_new: bool = False):
            rows = batch_runner.list_inputs(RESULTS_DIR)
            if not rows:
                note = (f'<div class="note info">{batch_runner.INPUT_DIR} 폴더에 처리할 파일이 없습니다.'
                        ' PDF·이미지를 넣은 뒤 <b>새로고침</b>을 누르세요.</div>')
            else:
                todo = sum(r["status"] == "미처리" for r in rows)
                note = f"<div>파일 {len(rows)}개 · 미처리 {todo}개</div>"
            value = [r["name"] for r in rows if r["status"] == "미처리"] if select_new else []
            return note, gr.update(choices=[(batch_runner.choice_label(r), r["name"]) for r in rows],
                                   value=value)

        def _batch_view(select_new: bool = False):
            return _batch_list(select_new)

        def _batch_submit(names, lang, fmt, no_pre, no_fb, no_fig, llm_c, llm_b, key, vf, cleanup, fm):
            if not names:
                return "<div class='note info'>처리할 파일을 먼저 고르세요.</div>", *_batch_list()[1:]
            if key and key.strip():
                claude_vision.set_session_key(key)
            batch_runner.submit([batch_runner.INPUT_DIR / n for n in names],
                                (lang, fmt, no_pre, no_fb, no_fig, llm_c, llm_b, key, vf),
                                {"cleanup": cleanup, "front_matter": fm})
            return batch_runner.status_html(), *_batch_list()[1:]

        batch_view_io = {"inputs": [], "outputs": [batch_note, batch_list]}
        batch_tab.select(fn=_batch_view, **batch_view_io)
        batch_refresh_btn.click(fn=_batch_view, **batch_view_io)
        batch_pick_btn.click(fn=lambda: _batch_view(select_new=True), **batch_view_io)

        batch_vision_first.change(fn=_vision_mode_note, inputs=[batch_vision_first],
                                  outputs=[batch_vision_note])

        batch_inputs = [
            batch_list,
            lang_input, fmt_input, no_preprocess, no_fallback, no_figure,
            llm_correct, llm_budget, api_key_input,
            batch_vision_first,
            batch_cleanup_chk,
            batch_front_matter_chk
        ]
        batch_run_btn.click(fn=_batch_submit, inputs=batch_inputs,
                            outputs=[batch_status, batch_list])
        batch_stop_btn.click(fn=lambda: (batch_runner.stop(), batch_runner.status_html())[1],
                             outputs=[batch_status])
        batch_clear_btn.click(fn=lambda: (batch_runner.clear_finished(), batch_runner.status_html())[1],
                              outputs=[batch_status])
        batch_timer.tick(fn=batch_runner.status_html, outputs=[batch_status])

        # 엔진 변경 — config.json 의 vision_provider 에 저장하고 관련 화면을 갱신
        def _engine_views():
            mode = "Vision 우선" if _engine_ok(fresh=True) else "Tesseract 우선"
            return (_header_html(), _engine_guide_html(),
                    gr.update(choices=_vision_choices(), value=mode),
                    _vision_mode_note(mode),
                    gr.update(value=text_cleaner.is_available(), info=_cleanup_info()),
                    _billing_html())

        engine_outputs = [header_html, engine_guide, vision_first, vision_first_note,
                          cleanup_chk, billing_html]

        def _set_provider(p: str):
            config.save({"vision_provider": p})
            claude_vision.reset_client()
            return (*_engine_views(),
                    f"저장했습니다. 이제 **{claude_vision.provider_label()}** 엔진을 씁니다.")

        provider_radio.change(fn=_set_provider, inputs=[provider_radio],
                              outputs=engine_outputs + [provider_msg])
        recheck_btn.click(fn=_engine_views, outputs=engine_outputs)

        # 기본 설정 저장
        def _save_cfg(lang, fmt, fallback, dpi, margin_filter, margin_top, margin_bottom):
            config.save({"default_lang": lang, "default_format": fmt,
                         "auto_fallback": fallback, "default_dpi": int(dpi),
                         "enable_margin_filter": margin_filter,
                         "margin_top_ratio": float(margin_top),
                         "margin_bottom_ratio": float(margin_bottom)})
            return f"기본 설정을 저장했습니다. (DPI {int(dpi)})"

        save_cfg_btn.click(fn=_save_cfg,
                           inputs=[cfg_lang, cfg_fmt, cfg_fallback, cfg_dpi,
                                   cfg_margin_filter, cfg_margin_top, cfg_margin_bottom],
                           outputs=[save_cfg_msg])

        # 사용량 새로고침
        refresh_btn.click(
            fn=lambda: (usage_tracker.format_summary(), _usage_table_html()),
            outputs=[usage_display, history_display],
        )

        # 통계 초기화
        def _reset_stats():
            usage_tracker.reset()
            return (usage_tracker.format_summary(), _usage_table_html(),
                    "사용 통계를 초기화했습니다.")

        reset_btn.click(fn=_reset_stats,
                        outputs=[usage_display, history_display, reset_msg])

        # ── 처리 기록 이벤트 ──────────────────────────────────────────────────

        def _refresh_records():
            return (_records_table_html(), _storage_stats_html(),
                    gr.update(choices=_history_choices(), value=None))

        records_outputs = [file_history, stats_html, pick_entry]
        refresh_files_btn.click(fn=_refresh_records, outputs=records_outputs)

        # 화면은 가볍게 열고, 데이터는 사용자가 탭을 열 때 읽는다 — 표 값이 서버 기동 시
        # 한 번만 읽혀 오늘 처리한 기록·사용량이 보이지 않았다.
        # 페이지 열기 갱신(load·callable value)·숨은 탭 갱신은 금지 — ③ 탭 클릭 시 멈춤
        # (2026-09-27 실측). 탭 select 갱신은 표가 HTML 이라 안전(RETRO 오류 10 은 Dataframe).
        # 드롭다운 선택지는 보일 때(펼칠 때) 갱신한다
        engine_tab.select(fn=lambda: (usage_tracker.format_summary(), _usage_table_html()),
                          outputs=[usage_display, history_display])
        records_tab.select(fn=lambda: (_records_table_html(), _storage_stats_html()),
                           outputs=[file_history, stats_html])
        pick_entry.focus(fn=lambda: gr.update(choices=_history_choices()), outputs=[pick_entry])

        # 항목 삭제
        def _del_entry(entry_id, del_files: bool):
            if not entry_id:
                return (*_refresh_records(), "삭제할 기록을 먼저 고르세요.")
            ok = storage_manager.delete_entry(entry_id, delete_files=del_files)
            if ok:
                msg = "기록을 삭제했습니다" + (" (저장 파일 포함)." if del_files else ".")
            else:
                msg = "그 기록을 찾지 못했습니다. 새로고침 후 다시 고르세요."
            return (*_refresh_records(), msg)

        del_entry_btn.click(
            fn=_del_entry,
            inputs=[pick_entry, del_files_chk],
            outputs=records_outputs + [del_msg],
        )

        # 결과 파일 다운로드
        def _download_by_id(entry_id, dl_type_str: str):
            if not entry_id:
                return None, "받을 기록을 먼저 고르세요."
            history = storage_manager.load_history()
            entry = next((e for e in history if e["id"] == entry_id), None)
            if not entry:
                return None, "그 기록을 찾지 못했습니다. 새로고침 후 다시 고르세요."
            key = "result_path" if "결과" in dl_type_str else "upload_path"
            path = Path(entry.get(key, ""))
            if not entry.get(key) or not path.exists():
                kind = "결과" if "결과" in dl_type_str else "원본"
                return None, f"{kind} 파일이 남아 있지 않습니다. (삭제되었거나 보관하지 않음)"
            return str(path), f"준비됨: {path.name}"

        dl_btn.click(
            fn=_download_by_id,
            inputs=[pick_entry, dl_type],
            outputs=[dl_out, dl_msg],
        )

        # 보관 설정 저장 — 저장 경로는 화면에서 바꾸지 않는다(RETRO 오류 7)
        def _save_storage_cfg(save_orig: bool):
            config.save({"save_original_files": save_orig})
            return "보관 설정을 저장했습니다."

        save_storage_btn.click(
            fn=_save_storage_cfg,
            inputs=[save_orig_chk],
            outputs=[storage_cfg_msg],
        )

        # 전체 초기화
        def _clear_all(del_files: bool):
            count = storage_manager.clear_history(delete_files=del_files)
            msg = f"처리 기록 {count}건을 삭제했습니다" + (" (저장 파일 포함)." if del_files else ".")
            return (*_refresh_records(), msg)

        clear_all_btn.click(
            fn=_clear_all,
            inputs=[clear_all_chk],
            outputs=records_outputs + [clear_msg],
        )

    return demo


# ── UI 헬퍼 ──────────────────────────────────────────────────────────────────

def _build_preview(result: FileResult, max_chars: int = 4000) -> str:
    """
    FileResult의 corrected_text(전체 체인 처리 완료)로부터
    미리보기 텍스트를 조합.

    - 이미 process_full()이 적용된 텍스트를 재사용 (이중 처리 없음)
    - for_preview()로 구조 마커만 추가 제거
    - 페이지 구분은 빈 줄로 표현
    """
    parts: list[str] = []
    for page in result.pages:
        if page.is_blank:
            continue  # 빈 페이지는 미리보기에서 제외
        clean = postprocessor.for_preview(page.corrected_text)
        if clean.strip():
            parts.append(clean)

    text = "\n\n".join(parts)
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n... (이하 생략)"
    return text


_PROVIDER_CHOICES = [
    ("Google Gemini — 클라우드 · 유료 · 가장 빠르고 정확", "gemini"),
    ("Anthropic Claude — 클라우드 · 유료", "claude"),
    ("로컬 Ollama — 이 PC에서 실행 · 무료 · 쪽당 수 분", "ollama"),
]

# provider → (config 키, 대표 환경변수, 키 발급처, 발급 URL)
_KEY_INFO = {
    "gemini": ("google_api_key", "GEMINI_API_KEY", "Google AI Studio",
               "https://aistudio.google.com/app/apikey"),
    "claude": ("anthropic_api_key", "ANTHROPIC_API_KEY", "Anthropic 콘솔",
               "https://console.anthropic.com/settings/keys"),
}


_OK_CACHE: list = [0.0, False]   # [확인 시각, 결과]


def _engine_ok(fresh: bool = False) -> bool:
    """화면 표시용 엔진 연결 여부. Ollama 는 확인에 최대 2초가 걸리므로
    한 번의 화면 갱신 안에서는 결과를 재사용한다(3초). 처리 경로는 이 캐시를 쓰지 않는다."""
    now = time.monotonic()
    if fresh or now - _OK_CACHE[0] > 3:
        _OK_CACHE[:] = [now, claude_vision.is_available()]
    return _OK_CACHE[1]


def _step_head(no: int | None, title: str, hint: str = "") -> str:
    num = f'<span class="step-no" aria-label="{no}단계">{no}</span>' if no else ""
    hint_html = f'<p class="step-hint">{hint}</p>' if hint else ""
    return (f'<div class="step-top"><div class="step-head">{num}'
            f'<span class="step-title">{title}</span></div>{hint_html}</div>')


def _key_state(cfg_key: str) -> tuple[str, str]:
    """(note 클래스, 문구). 키 값은 다루지 않고 출처만 표시한다."""
    src = config.key_source(cfg_key)
    usable = (config.has_google_api_key() if cfg_key == "google_api_key"
              else config.has_api_key())
    if src and not usable:
        return "err", "등록된 키의 형식이 올바르지 않습니다(같은 키가 두 번 붙어 있는 등). 값을 다시 확인하세요."
    if src == "config.json":
        return "warn", ("config.json 에 평문 키가 남아 있습니다. 아래 방법으로 환경변수에 등록한 뒤 "
                        "config.json 의 값을 비우세요.")
    if src:
        return "ok", f"연결됨 — 환경변수 <code>{html.escape(src)}</code> 에서 읽었습니다."
    return "err", "키가 없습니다. 아래 방법으로 환경변수에 등록하세요."


def _header_html() -> str:
    p, label = claude_vision.provider_name(), claude_vision.provider_label()
    ok = _engine_ok()
    chips = []
    if p == "ollama":
        chips.append(("ok", f"엔진 {label} · 연결됨") if ok else ("err", f"엔진 {label} · 미연결"))
    else:
        chips.append(("ok", f"엔진 {label} · 키 연결됨") if ok else ("err", f"엔진 {label} · API 키 없음"))
        if config.key_source(_KEY_INFO[p][0]) == "config.json":
            chips.append(("warn", "키가 config.json 에 평문 저장됨"))
    chips.append(("ok", "문장 정제 사용 가능") if text_cleaner.is_available()
                 else ("", "문장 정제 사용 불가"))
    chip_html = "".join(f'<span class="chip {c}">{t}</span>' for c, t in chips)
    return f"""
<div class="app-head">
  <div class="brand">
   <div class="brand-mark"><svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
     <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/>
     <path d="M9 13h6M9 17h4"/></svg></div>
   <div>
    <div class="app-title">OCR 텍스트 추출</div>
    <div class="app-sub">스캔 PDF·이미지를 텍스트로 바꿉니다 · 결과는 02_결과 폴더에 저장</div>
   </div>
  </div>
  <div class="chips">{chip_html}</div>
</div>"""


def _vision_choices() -> list[tuple[str, str]]:
    """값("Vision 우선"/"Tesseract 우선")은 처리 함수가 판별에 쓰므로 고정, 표시 문구만 엔진명 반영."""
    label = claude_vision.provider_label()
    return [
        (f"정확도 우선 — {label}가 먼저 읽고, 실패하면 로컬 엔진(RapidOCR)", "Vision 우선"),
        (f"비용 우선 — 로컬 고성능 엔진(RapidOCR 한글 v5)이 무료로 읽고, 흐린 쪽만 {label}", "Tesseract 우선"),
    ]


def _vision_mode_note(mode) -> str:
    label = claude_vision.provider_label()
    ok = _engine_ok()
    local = claude_vision.provider_name() == "ollama"
    local_name = "로컬 엔진(RapidOCR 한글 v5)"
    if "Vision 우선" in str(mode):
        if not ok:
            return (f'<div class="note warn">{label} 연결이 없어 지금은 {local_name}으로만 읽습니다. '
                    '② 엔진 · API 키 탭에서 연결 상태를 확인하세요.</div>')
        cost = "요금은 없지만 쪽당 수 분 걸립니다." if local else "쪽마다 API 요금이 듭니다."
        return (f'<div class="note info">{label}가 쪽마다 레이아웃을 이해하며 읽습니다. '
                f'한국어 정확도가 가장 높습니다. {cost}</div>')
    tail = (f"신뢰도가 낮은 쪽만 {label}에 다시 맡깁니다." if ok
            else f"지금은 {label} 연결이 없어 재인식 없이 {local_name}만 씁니다.")
    return f'<div class="note info">{local_name}이 이 PC에서 무료로 읽고, {tail}</div>'


def _cleanup_info() -> str:
    if text_cleaner.is_available():
        return "Google Gemini 사용 · 책 1권 약 2~4분, 수십 원 · 전체 처리에만 적용됩니다."
    if claude_vision.provider_name() == "claude":
        return "Claude 엔진에서는 쓸 수 없습니다(문장 정제는 Gemini 전용)."
    return "Gemini API 키가 없어 쓸 수 없습니다. ② 엔진 · API 키 탭을 참고하세요."


def _file_info_html(f) -> str:
    if not f:
        return ('<div class="note">PDF, PNG, JPG, TIFF 파일을 올릴 수 있습니다.'
                ' 여러 개를 한꺼번에 끌어다 놓아도 됩니다.</div>')
    if isinstance(f, list) and len(f) > 1:
        names = "".join(f"<li>{html.escape(Path(getattr(x, 'name', x)).name)}</li>" for x in f)
        return (f'<div class="note info"><b>파일 {len(f)}개</b> — 「전체 처리」를 누르면 아래 순서대로'
                f' 하나씩 처리합니다. 미리보기는 첫 파일만 합니다.<ol style="margin:4px 0 0 18px">'
                f'{names}</ol></div>')
    if isinstance(f, list):
        f = f[0]
    path = Path(getattr(f, "name", f))
    name = html.escape(path.name)
    try:
        size_mb = path.stat().st_size / 1_048_576
    except OSError:
        size_mb = 0.0
    big = (f" 파일이 커서({size_mb:.0f}MB) 시간이 오래 걸릴 수 있습니다."
           if size_mb > WARN_SIZE_MB else "")
    if path.suffix.lower() != ".pdf":
        return f'<div class="note info"><b>{name}</b> — 이미지 파일이라 OCR로 읽습니다.{big}</div>'
    info = pdf_text_layer.analyze(path) if pdf_text_layer.is_available() else None
    if info and info.is_digital:
        bad = (f' 단, 글자가 깨진 {len(info.garbled_pages)}쪽은 2단계의 읽는 방식으로 다시 읽습니다.'
               if info.garbled_pages else ' 2단계의 읽는 방식은 쓰이지 않고 문장 정제만 적용됩니다.')
        if info.noisy_pages:
            bad += (f' 잡음·띄어쓰기 문제가 있는 {len(info.noisy_pages)}쪽은 문장 정제로 고쳐지니'
                    ' 정제를 켜 두세요.')
        return (f'<div class="note ok"><b>{name}</b> — 디지털 PDF · {info.total_pages}쪽<br>'
                f'글자 정보가 들어 있어 OCR·API 없이 무료로 바로 추출합니다.{bad}{big}</div>')
    if info and info.garbled_pages:
        return (f'<div class="note info"><b>{name}</b> — 글자 정보가 깨진 PDF · {info.total_pages}쪽<br>'
                f'들어 있는 글자의 {len(info.garbled_pages)}쪽이 깨져 있어 스캔 PDF처럼 OCR로 읽습니다.'
                f' 2단계에서 읽는 방식을 고르세요.{big}</div>')
    pages = f" · {info.total_pages}쪽" if info else ""
    return (f'<div class="note info"><b>{name}</b> — 스캔 PDF{pages}<br>'
            f'이미지로 된 문서라 OCR이 필요합니다. 2단계에서 읽는 방식을 고르세요.{big}</div>')


def _results_location_html() -> str:
    return (f'<div class="note">결과는 <b>02_결과</b> 폴더에 자동으로 저장됩니다.<br>'
            f'<code>{html.escape(str(RESULTS_DIR))}</code></div>')


def _open_results_folder() -> str:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.startfile(RESULTS_DIR)  # noqa: S606 — 로컬 전용 도구, 탐색기로 결과 폴더 열기
        return ""
    except (AttributeError, OSError) as e:  # Windows 외 환경 또는 권한 문제
        return f"폴더를 열지 못했습니다({e}). 직접 여세요: `{RESULTS_DIR}`"


def _engine_guide_html() -> str:
    p = claude_vision.provider_name()
    if p == "ollama":
        ok = _engine_ok()
        model = html.escape(config.get_ollama_vision_model())
        status = (f'<div class="note ok">Ollama 연결됨 — 모델 <code>{model}</code></div>' if ok else
                  '<div class="note err">Ollama에 연결하지 못했습니다. Ollama 앱이 켜져 있는지, '
                  '모델이 설치돼 있는지 확인하세요.</div>')
        body = f"""
<p><b>로컬 Ollama는 API 키가 필요 없습니다.</b></p>
<ol>
  <li>Ollama 앱을 실행합니다.</li>
  <li>처음 한 번, 명령 프롬프트에서 <code>ollama pull {model}</code> 로 모델을 받습니다.</li>
  <li>아래 <b>연결 상태 다시 확인</b>을 누릅니다.</li>
</ol>
<p class="muted">CPU로 돌아가 쪽당 수 분 걸릴 수 있습니다.</p>"""
    else:
        cfg_key, env, console, url = _KEY_INFO[p]
        cls, msg = _key_state(cfg_key)
        status = f'<div class="note {cls}">{msg}</div>'
        names = ", ".join(f"<code>{html.escape(n)}</code>" for n in config.ENV_KEY_NAMES[cfg_key])
        body = f"""
<ol>
  <li><a href="{url}" target="_blank" rel="noopener">{console}</a>에서 API 키를 발급받습니다.</li>
  <li>Windows 검색창에 <b>환경 변수</b>를 입력하고 <b>계정의 환경 변수 편집</b>을 엽니다.</li>
  <li><b>사용자 변수</b>에서 <b>새로 만들기</b> → 변수 이름 <code>{env}</code>, 값에 발급받은 키를 붙여넣고 확인합니다.</li>
  <li>START_OCR 창을 닫고 <b>START_OCR.bat</b>을 다시 실행합니다. 환경변수는 프로그램이 시작할 때만 읽힙니다.</li>
</ol>
<p class="muted">인식하는 변수 이름(앞쪽 우선): {names}</p>"""
    cleanup = ""
    if p == "claude":
        cleanup = ('<div class="note warn" style="margin-top:10px">Claude 엔진을 고르면 '
                   '문장 정제는 쓸 수 없습니다(문장 정제는 Gemini 전용).</div>')
    elif p == "ollama":
        gcls, gmsg = _key_state("google_api_key")
        cleanup = ('<p style="margin-top:10px"><b>문장 정제</b>는 엔진과 관계없이 Google Gemini를 씁니다. '
                   f'변수 이름 <code>GEMINI_API_KEY</code>.</p><div class="note {gcls}">Gemini 키: {gmsg}</div>')
    return f'<div class="guide">{status}{body}{cleanup}</div>'


def _billing_html() -> str:
    """현재 provider 에 맞는 사용량/요금 콘솔 안내 HTML."""
    if claude_vision.provider_name() == "claude":
        url, console = "https://console.anthropic.com/settings/billing", "Anthropic 콘솔"
        lead = "실제 사용량과 요금은"
    else:
        url, console = "https://aistudio.google.com/", "Google AI Studio"
        lead = ("로컬 Ollama는 요금이 없습니다. 문장 정제(Gemini) 요금은"
                if claude_vision.provider_name() == "ollama" else "실제 사용량과 요금은")
    return (f'<div class="note guide" style="margin-top:8px">{lead} '
            f'<a href="{url}" target="_blank" rel="noopener">{console}</a>에서 확인하세요.</div>')


def _history_table() -> list[list]:
    stats = usage_tracker.get_stats()
    eng   = claude_vision.provider_label()
    rows = []
    for h in reversed(stats.get("history", [])[-20:]):
        rows.append([
            h.get("time", ""),
            f"{eng} OCR" if h.get("type") == "vision" else "문장 정제/보정",
            h.get("in", 0),
            h.get("out", 0),
            f"${h.get('cost', 0):.5f}",
        ])
    return rows if rows else [["—", "—", 0, 0, "$0.00000"]]


# 기록 표는 gr.Dataframe 대신 HTML — Dataframe 은 숨겨진 탭에서 값이 바뀌면 그 탭을 여는 순간
# 크기 계산을 반복하며 브라우저가 멈춘다(RETRO 오류 10 · 2026-09-27 재현). HTML 은 그 과정이 없다.
_USAGE_HEADERS = ["시간", "유형", "입력토큰", "출력토큰", "비용(USD)"]
_RECORDS_HEADERS = ["처리일시", "원본 파일명", "전체p", "빈p", "엔진 모드",
                    "결과 크기", "결과파일", "원본파일", "ID"]


def _table_html(headers: list[str], rows: list[list], max_height: int) -> str:
    esc = lambda v: html.escape(str(v))
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return (f'<div class="rec-table" style="max-height:{max_height}px;overflow:auto">'
            f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>")


def _usage_table_html() -> str:
    return _table_html(_USAGE_HEADERS, _history_table(), 160)


def _records_table_html() -> str:
    return _table_html(_RECORDS_HEADERS, storage_manager.get_history_table(), 420)


def _should_save_original() -> bool:
    """설정에서 원본 파일 보관 여부 확인."""
    return bool(config.get("save_original_files", False))


def _storage_stats_html() -> str:
    s = storage_manager.get_stats()
    cells = [(s["history_count"], "처리 기록"), (s["result_count"], "보관된 결과"),
             (s["upload_count"], "보관된 원본"), (f'{s["total_size_mb"]} MB', "사용 용량")]
    grid = "".join(f"<div class='stat'><b>{v}</b><span>{k}</span></div>" for v, k in cells)
    return (f"<div class='stats'>{grid}</div>"
            f"<p class='step-hint' style='margin-top:8px;max-width:none'>내부 보관소: "
            f"<code>{html.escape(str(s['storage_path']))}</code></p>")


def _history_choices() -> list[tuple[str, str]]:
    """처리 기록 드롭다운 선택지 — (표시: 일시 · 파일명, 값: id)."""
    return [(f"{e.get('timestamp', '')[:16]} · {e.get('original_name', '')}", e["id"])
            for e in storage_manager.load_history() if e.get("id")]


# ══════════════════════════════════════════════════════════════════════════════
# 실행
# ══════════════════════════════════════════════════════════════════════════════

def main():
    demo = build_ui()
    print("\n✅ OCR 웹 UI 시작 (로컬 엔진: RapidOCR PP-OCRv5 한글)")
    print("   브라우저: http://localhost:7860\n")
    # allowed_paths: 다운로드 버튼이 02_결과/ 의 결과 파일을 그대로 내려줄 수 있게
    # 허용한다. Gradio는 기본적으로 작업 폴더·임시 폴더 밖의 파일 전송을 막아
    # (InvalidPathError) 산출물이 정상 생성돼도 다운로드 단계에서만 실패했다.
    # 서버는 127.0.0.1 에만 바인딩되므로 노출 범위는 이 PC로 한정된다.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    demo.launch(server_name="127.0.0.1", server_port=7860,
                share=False, inbrowser=True, css=_CSS, theme=_THEME,
                allowed_paths=[str(RESULTS_DIR)])


if __name__ == "__main__":
    main()
