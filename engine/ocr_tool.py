"""OCR 스캔파일 텍스트추출 도구 — CLI 진입점 (Phase 3 + API-key-less 지원)"""
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm

from core import detector, pdf_splitter, quality_diagnoser, preprocessor
from core import rapidocr_engine, char_normalizer, writer, postprocessor
# vision 파사드 — config.json 의 vision_provider 에 따라 Gemini/Claude 라우팅.
# 기존 호출부 호환을 위해 claude_vision 이름으로 import 한다.
from core import vision as claude_vision
from core import spellcheck, figure_detector, llm_corrector
from core import config
from core import margin_detector
from core.types import FileResult, OutputFormat, PageResult, OCREngine

console = Console()



# ── 활성 provider(Gemini/Claude) 인지 헬퍼 ────────────────────────────────────
# config.json 의 vision_provider 에 따라 키 저장 위치·발급 안내가 달라진다.

def _active_key() -> str:
    return (config.get_google_api_key() if claude_vision.provider_name() == "gemini"
            else config.get_api_key())


def _key_env_guide() -> None:
    """키는 파일에 저장하지 않는다 — config.json 은 git·USB 동기화 대상이라 평문 키가
    새어 나갈 수 있다(QA ISSUE-D, 2026-10-03). 웹 화면과 같은 환경변수 전용 안내."""
    var = "GEMINI_API_KEY" if claude_vision.provider_name() == "gemini" else "ANTHROPIC_API_KEY"
    console.print(
        "[yellow]API 키는 파일에 저장하지 않습니다.[/yellow]\n"
        f"Windows 검색 → '계정의 환경 변수 편집' → 사용자 변수 새로 만들기: 이름 [bold]{var}[/bold], 값에 키.\n"
        "등록 후 이 창(또는 START_OCR)을 닫고 다시 실행하세요.")


def _clear_active_key() -> None:
    if claude_vision.provider_name() == "ollama":
        return
    if claude_vision.provider_name() == "gemini":
        config.set_google_api_key("")
    else:
        config.clear_api_key()


def _key_issue_info() -> tuple[str, str]:
    """(키 발급 URL, 키 형식 예시) — 현재 provider 기준."""
    if claude_vision.provider_name() == "gemini":
        return ("https://aistudio.google.com/apikey", "AIza...")
    return ("https://console.anthropic.com", "sk-ant-api03-...")


# ══════════════════════════════════════════════════════════════════════════════
# CLI 그룹 — ocr (기본 처리) + setup + settings
# ══════════════════════════════════════════════════════════════════════════════

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """스캔 PDF·이미지에서 텍스트를 추출합니다."""
    if ctx.invoked_subcommand is None:
        # 하위 명령 없이 실행 시 도움말 출력
        click.echo(ctx.get_help())


# ── ocr: 메인 처리 명령 ────────────────────────────────────────────────────────

@cli.command("ocr")
@click.option("--input",  "-i", "input_path", required=True,
              help="처리할 파일 또는 폴더 경로")
@click.option("--output", "-o", "output_dir", default="./output",
              show_default=True, help="출력 폴더 경로")
@click.option("--format", "-f", "fmt",
              type=click.Choice(["txt", "md"], case_sensitive=False),
              default=None, help="출력 포맷 (기본값: 설정 파일 또는 txt)")
@click.option("--lang",   "-l", default=None,
              help="OCR 언어 코드 (기본값: 설정 파일 또는 kor+eng)")
@click.option("--dpi",    "-d", default=None, type=int,
              help="PDF 변환 DPI (기본값: 설정 파일 또는 300)")
@click.option("--no-preprocess",  "no_preprocess",  is_flag=True)
@click.option("--no-fallback",    "no_fallback",    is_flag=True,
              help="Vision 폴백 강제 비활성화")
@click.option("--no-spellcheck",  "no_spellcheck",  is_flag=True)
@click.option("--no-figure",      "no_figure",      is_flag=True)
@click.option("--save-figures",   "save_figures",   is_flag=True)
@click.option("--llm-correct",    "llm_correct",    is_flag=True)
@click.option("--llm-budget",     "llm_budget",     default=0, type=int,
              help="L3 LLM 보정 최대 호출 횟수 (0=무제한)")
@click.option("--margin-filter",  "margin_filter",  is_flag=True, help="머리글·바닥글·쪽번호 마진 제거 필터 활성화")
@click.option("--margin-top",     "margin_top",     type=float, default=None, help="상단 여백 비율 (0.0 ~ 1.0)")
@click.option("--margin-bottom",  "margin_bottom",  type=float, default=None, help="하단 여백 비율 (0.0 ~ 1.0)")
@click.option("--verbose", "-v",  is_flag=True)
def ocr_cmd(input_path, output_dir, fmt, lang, dpi,
            no_preprocess, no_fallback, no_spellcheck, no_figure, save_figures,
            llm_correct, llm_budget, margin_filter, margin_top, margin_bottom, verbose):
    """스캔 파일에서 텍스트를 추출합니다."""

    # CLI 인자로 들어온 마진 필터 옵션을 config에 업데이트
    updates = {}
    if margin_filter:
        updates["enable_margin_filter"] = True
    if margin_top is not None:
        updates["margin_top_ratio"] = margin_top
    if margin_bottom is not None:
        updates["margin_bottom_ratio"] = margin_bottom
    if updates:
        config.save(updates)

    # 설정 파일에서 기본값 로드
    cfg  = config.load()
    fmt  = fmt  or cfg.get("default_format", "txt")
    lang = lang or cfg.get("default_lang",   "kor+eng")
    dpi  = dpi  or cfg.get("default_dpi",    300)
    if not no_fallback and not cfg.get("auto_fallback", True):
        no_fallback = True
    if not no_spellcheck and not cfg.get("auto_spellcheck", True):
        no_spellcheck = True
    if not save_figures and cfg.get("save_figures", False):
        save_figures = True

    console.print(Panel(
        "[bold cyan]OCR 스캔파일 텍스트추출 도구[/bold cyan]\n"
        "PDF · PNG · JPG · TIFF → 플레인 텍스트",
        expand=False
    ))
    _print_env_status(no_fallback, no_spellcheck, no_figure, llm_correct, llm_budget)

    try:
        files = detector.collect_files(input_path)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]오류:[/red] {e}")
        sys.exit(1)

    console.print(f"\n처리 대상: [bold]{len(files)}[/bold]개 파일\n")

    out_fmt = OutputFormat.MD if fmt.lower() == "md" else OutputFormat.TXT
    results: list[FileResult] = []
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("처리 중...", total=len(files))

        for file_path in files:
            progress.update(task, description=f"[cyan]{file_path.name}[/cyan]")
            result = _process_file(
                file_path, output_dir, out_fmt, lang, dpi,
                no_preprocess, no_fallback, no_spellcheck,
                no_figure, save_figures, llm_correct, llm_budget, verbose,
            )
            results.append(result)

            if not result.success:
                failed += 1
                console.print(f"  [red]✗[/red] {file_path.name}: {result.error_message}")
            else:
                engines    = {p.engine.value for p in result.pages}
                fig_count  = sum(len(p.figures) for p in result.pages)
                rev_count  = sum(1 for p in result.pages if p.needs_review)
                engine_str = "+".join(sorted(engines))
                extra = []
                if fig_count:  extra.append(f"감지영역 {fig_count}개")
                if rev_count:  extra.append(f"[yellow]검토필요 {rev_count}p[/yellow]")
                extra_str = " · ".join(extra)
                console.print(
                    f"  [green]✓[/green] {file_path.name} "
                    f"→ {result.output_path.name} "
                    f"([dim]{result.total_pages}p · {engine_str}"
                    + (f" · {extra_str}" if extra_str else "") + "[/dim])"
                )
            progress.advance(task)

    _print_summary(results, failed)

    # API 키 없고 검토 필요 페이지가 있을 때 설정 안내
    if not claude_vision.is_available() and not no_fallback:
        total_review = sum(
            1 for r in results if r.success
            for p in r.pages if p.needs_review
        )
        if total_review:
            issue_url, _ = _key_issue_info()
            eng = claude_vision.provider_label()
            console.print()
            console.print(Panel(
                f"[yellow]검토 필요 페이지 {total_review}개가 있습니다.[/yellow]\n\n"
                f"{eng} 폴백을 활성화하면 정확도를 높일 수 있습니다.\n\n"
                "[bold]지금 설정하기:[/bold]\n"
                "  python ocr_tool.py setup\n\n"
                f"API 키 발급: [link={issue_url}]{issue_url}[/link]"
                "  (무료 플랜 사용 가능)",
                title="💡 정확도 향상 안내",
                border_style="yellow",
                expand=False,
            ))

    sys.exit(1 if failed == len(files) else 0)


# ── setup: API 키 설정 마법사 ──────────────────────────────────────────────────

@cli.command("setup")
def setup_cmd():
    """API 키 설정 마법사 (현재 활성 엔진 기준)."""
    eng = claude_vision.provider_label()
    issue_url, key_fmt = _key_issue_info()
    console.print(Panel(
        "[bold]OCR 도구 초기 설정[/bold]\n"
        f"{eng} 폴백 및 기본 옵션을 설정합니다.",
        expand=False
    ))

    # 현재 상태 표시 (활성 provider 키 기준)
    current_key = _active_key()
    if current_key:
        masked = current_key[:8] + "..." + current_key[-4:] if len(current_key) > 12 else "설정됨"
        console.print(f"\n현재 {eng} API 키: [green]{masked}[/green]")
    else:
        console.print(f"\n현재 {eng} API 키: [yellow]미설정[/yellow]")
        console.print("\n[dim]키 없이도 RapidOCR 한글 v5 엔진으로 기본 동작합니다.[/dim]")
        console.print("[dim]키가 있으면 저품질 스캔의 정확도를 크게 높일 수 있습니다.[/dim]\n")

    # API 키 입력 (활성 provider 위치에 저장)
    want_key = Confirm.ask("API 키를 설정하시겠습니까?", default=bool(current_key))
    if want_key:
        console.print(f"\n[dim]API 키는 {issue_url} 에서 발급받을 수 있습니다.[/dim]")
        console.print(f"[dim]키 형식: {key_fmt}[/dim]\n")
        _key_env_guide()
    elif current_key:
        clear = Confirm.ask("기존 API 키를 삭제하시겠습니까?", default=False)
        if clear:
            _clear_active_key()
            console.print("[yellow]API 키가 삭제되었습니다.[/yellow]")

    # 기본 설정 옵션
    console.print("\n[bold]기본 동작 설정[/bold]")

    cfg = config.load()
    updates: dict = {}

    auto_fb = Confirm.ask(
        f"저신뢰도 페이지에 {eng} 자동 폴백 사용",
        default=cfg.get("auto_fallback", True),
    )
    updates["auto_fallback"] = auto_fb

    auto_spell = Confirm.ask(
        "한국어 맞춤법 자동 검사 (py-hanspell 설치 시 활성)",
        default=cfg.get("auto_spellcheck", True),
    )
    updates["auto_spellcheck"] = auto_spell

    fmt_choice = Prompt.ask(
        "기본 출력 포맷",
        choices=["txt", "md"],
        default=cfg.get("default_format", "txt"),
    )
    updates["default_format"] = fmt_choice

    config.save(updates)
    console.print("\n[green]✓ 설정이 저장되었습니다.[/green]")
    console.print(f"[dim]설정 파일 위치: {config.config_path()}[/dim]\n")

    # 저장 후 상태 요약
    _show_settings_table()


# ── settings: 현재 설정 조회·수정 ─────────────────────────────────────────────

@cli.command("settings")
@click.option("--show",       "action", flag_value="show",  default=True)
@click.option("--clear-key",  "action", flag_value="clear_key",
              help="저장된 API 키 삭제")
@click.option("--set-key",    "set_key", default=None,
              help="API 키 직접 지정 (--set-key <키>)")
def settings_cmd(action, set_key):
    """현재 설정 조회 및 수정."""
    if set_key:
        _key_env_guide()
        sys.exit(1)
    if action == "clear_key":
        _clear_active_key()
        console.print("[yellow]API 키가 삭제되었습니다.[/yellow]")
        return
    _show_settings_table()


def _show_settings_table():
    cfg = config.show()
    eng = claude_vision.provider_label()
    key = _active_key()
    key_masked = (key[:8] + "..." + key[-4:]) if len(key) > 12 else ("설정됨" if key else "미설정")

    table = Table(title="현재 설정", show_header=True)
    table.add_column("항목",  style="dim")
    table.add_column("값",    justify="left")
    table.add_row(f"{eng} API 키", key_masked)
    labels = {
        "default_lang":         "기본 언어",
        "default_dpi":          "기본 DPI",
        "default_format":       "기본 출력 포맷",
        "confidence_threshold": "폴백 신뢰도 임계값",
        "auto_fallback":        "자동 Vision 폴백",
        "auto_spellcheck":      "자동 맞춤법 검사",
        "save_figures":         "그림 크롭 저장",
        "enable_cleanup":       "문장 정제(교정)",
        "remove_front_matter":  "부속물 제거",
        "enable_margin_filter": "여백 마진 필터",
        "margin_top_ratio":     "상단 여백 비율",
        "margin_bottom_ratio":  "하단 여백 비율",
    }
    for cfg_key, label in labels.items():
        val = cfg.get(cfg_key, "-")
        if isinstance(val, bool):
            val = "[green]ON[/green]" if val else "[dim]OFF[/dim]"
        table.add_row(label, str(val))
    console.print(table)
    console.print(f"\n[dim]설정 파일: {config.config_path()}[/dim]")


# ══════════════════════════════════════════════════════════════════════════════
# 내부 처리 함수
# ══════════════════════════════════════════════════════════════════════════════

# 이번 실행에서 쓴 결과 파일 → 원본. 폴더를 재귀로 모으면 '상권/본문.png'·'하권/본문.png' 처럼 이름이
# 같은 파일이 생기는데, 결과 이름이 stem 만 써서 뒤 파일이 앞 결과를 조용히 덮어썼다(감사 28번)
_WRITTEN: dict[Path, Path] = {}


def _unique_stem(file_path: Path, output_dir: str, fmt: OutputFormat) -> str:
    stem, k = file_path.stem, 2
    while _WRITTEN.get(Path(output_dir) / f"{stem}.{fmt.value}", file_path) != file_path:
        stem, k = f"{file_path.stem}_{k}", k + 1
    _WRITTEN[Path(output_dir) / f"{stem}.{fmt.value}"] = file_path
    return stem


def _process_file(
    file_path: Path, output_dir: str, fmt: OutputFormat,
    lang: str, dpi: int,
    no_preprocess: bool, no_fallback: bool, no_spellcheck: bool,
    no_figure: bool, save_figures: bool,
    llm_correct: bool, llm_budget: int, verbose: bool,
) -> FileResult:
    result = FileResult(source_path=file_path, output_path=None)
    try:
        fmt_type = detector.detect(file_path)
        images = (
            pdf_splitter.split_pdf(file_path, dpi=dpi)
            if fmt_type == "pdf"
            else pdf_splitter.load_image(file_path)
        )
        if fmt_type == "pdf":
            images = _mask_margins(file_path, images, verbose)
        vision_budget = claude_vision.VisionBudget(len(images))   # 문서당 Vision 상한(GUI 와 같음)
        for page_num, image in enumerate(images, start=1):
            page_result = _process_page(
                image, page_num, lang,
                no_preprocess, no_fallback, no_spellcheck,
                no_figure, save_figures, output_dir,
                file_path.stem, llm_correct, llm_budget, verbose,
                vision_budget=vision_budget,
            )
            result.pages.append(page_result)
        writer.write(result, output_dir, fmt, stem=_unique_stem(file_path, output_dir, fmt))
    except Exception as e:
        result.success = False
        result.error_message = str(e)
        if verbose:
            import traceback
            console.print(f"[red]{traceback.format_exc()}[/red]")
    return result


def _mask_margins(file_path: Path, images: list, verbose: bool) -> list:
    """본문 쪽에서 머리글·바닥글 여백을 재서 모든 쪽의 그 띠를 가린다 (웹 화면과 같은 규칙)."""
    if not config.get_auto_margin():
        return images
    try:
        margins = margin_detector.measure(file_path, len(images))
    except Exception as e:      # 측정 실패는 여백 그대로 처리 — 파일 전체를 막지 않는다
        console.print(f"  [yellow]⚠ 여백 자동 측정 실패 → 여백 그대로 인식 ({e})[/yellow]")
        return images
    if margins is None:
        return images
    if verbose:
        console.print(f"  [dim]📐 {margins.describe()}[/dim]")
    if not margins.active:
        return images
    return [margin_detector.apply(img, margins, n) for n, img in enumerate(images, start=1)]


def _process_page(
    image, page_num: int, lang: str,
    no_preprocess: bool, no_fallback: bool, no_spellcheck: bool,
    no_figure: bool, save_figures: bool, output_dir: str, stem: str,
    llm_correct: bool, llm_budget: int, verbose: bool,
    vision_budget: claude_vision.VisionBudget | None = None,
) -> PageResult:

    report = quality_diagnoser.diagnose(image)
    if verbose:
        console.print(f"\n  [dim]페이지 {page_num}:[/dim]")
        console.print(quality_diagnoser.describe(report))

    # 그림·표 감지
    placeholders: list[str] = []
    ocr_image = image
    if not no_figure:
        det_result = figure_detector.detect(image, lang=lang)
        if det_result.regions:
            placeholders = [figure_detector.build_placeholder(r) for r in det_result.regions]
            ocr_image = det_result.masked_image or image
            if save_figures:
                saved = figure_detector.save_crops(det_result.regions, output_dir, stem)
                if verbose and saved:
                    for path in saved.values():
                        console.print(f"  [dim]크롭 저장: {path.name}[/dim]")
            if verbose and det_result.regions:
                kinds = [f"{r.kind}({r.index})" for r in det_result.regions]
                console.print(f"  [dim]감지된 영역: {', '.join(kinds)}[/dim]")

    # 이미지 품질 진단만으로는 Vision 에 보내지 않는다 — RapidOCR 신뢰도로 판단(ocr_gradio 와 같음).
    # 실측(2026-10-15, B2B 301쪽): 품질불량 진단 35쪽 중 34쪽을 RapidOCR 이 99~100% 일치하게 읽었다.
    # Vision 엔진은 실패 시 예외를 던진다 — 받지 않으면 한 쪽 실패로 파일 전체가 중단된다
    vision_err = ""

    # 전처리
    if no_preprocess:
        processed = ocr_image
    else:
        processed = preprocessor.preprocess_for_rapidocr(ocr_image, report)

    # 로컬 OCR — RapidOCR 한글 v5 (실패하면 신뢰도 0 결과 → 아래 저신뢰 흐름)
    # 한글 교정기는 쪽 마무리에서 1번만 — 두 번 돌리면 띄어쓰기를 다시 붙이기도 한다
    ocr_result = rapidocr_engine.smart_run(processed, default_lang=lang, refine=False)

    if verbose:
        console.print(
            f"  [dim]엔진: rapidocr(v5) / 신뢰도: {ocr_result.confidence:.1f}% "
            f"/ 자모분리: {'감지' if ocr_result.has_jamo_split else '없음'}[/dim]"
        )

    low_quality = ocr_result.confidence < config.get_confidence_threshold() or ocr_result.has_jamo_split

    # 신뢰도 폴백 시도
    if not no_fallback and low_quality:
        # 문서당 상한 소진 — 유료 호출 없이 로컬 결과로 남긴다
        over_budget = (claude_vision.is_available() and vision_budget is not None
                       and not vision_budget.take())
        if claude_vision.is_available() and not vision_err and not over_budget:  # 이미 실패했으면 재시도 안 함 (Ollama 쪽당 최대 9분)
            reason = "자모분리" if ocr_result.has_jamo_split else f"신뢰도 {ocr_result.confidence:.1f}%"
            if verbose:
                console.print(f"  [yellow]→ {claude_vision.provider_label()} 폴백 ({reason})[/yellow]")
            try:
                vision_result = claude_vision.run(image, lang_hint=lang)
            except Exception as e:
                vision_err = str(e)
                console.print(f"  [yellow]⚠ p.{page_num} 폴백 실패 → 로컬 OCR 결과 유지 ({e})[/yellow]")
            else:
                return _finalize_page(vision_result, page_num, placeholders,
                                       no_spellcheck, llm_correct, llm_budget)
        # API 키 없음 또는 폴백 실패 → 검토 필요 페이지로 마킹
        reason = (
            "자모 분리 현상 감지 (스캔 품질 불량)"
            if ocr_result.has_jamo_split
            else f"OCR 신뢰도 낮음 ({ocr_result.confidence:.1f}% / 기준 {config.get_confidence_threshold()}%)"
        )
        if vision_err:
            reason += " · Vision 폴백 실패"
        elif over_budget:
            reason += " · 문서당 Vision 상한 도달(비용 보호)"
        console.print(
            f"  [yellow]⚠ p.{page_num} 검토 필요 — {reason}[/yellow]"
        )
        return _finalize_page(
            ocr_result, page_num, placeholders,
            no_spellcheck, llm_correct, llm_budget,
            needs_review=True, review_reason=reason,
        )

    # 품질 불량 진단 쪽은 RapidOCR 신뢰도가 높아도 폴백 실패를 결과에 남긴다
    return _finalize_page(ocr_result, page_num, placeholders,
                           no_spellcheck, llm_correct, llm_budget,
                           needs_review=bool(vision_err),
                           review_reason="품질 불량 · Vision 폴백 실패" if vision_err else "")


def _finalize_page(
    ocr_result, page_num: int, placeholders: list[str],
    no_spellcheck: bool, llm_correct: bool,
    llm_budget: int = 0,
    needs_review: bool = False, review_reason: str = "",
) -> PageResult:
    """
    OCR 결과 전체 처리 체인 (CLI용).
      L1 → L2 → process_full(엔진별) → L3
    """
    engine_str = ocr_result.engine.value  # 'rapidocr' 또는 Vision 엔진

    # L1 — 규칙 기반 보정
    text = char_normalizer.normalize_all(ocr_result.text)

    # MyOCR 기반 한국어 정제
    try:
        from core.korean_corrector import refine_korean_text
        text = refine_korean_text(text)
    except Exception:
        pass

    # L2 — 맞춤법 검사
    if not no_spellcheck and spellcheck.is_available():
        text = spellcheck.correct_safe(text)

    # 후처리 — 엔진별 분기 (Vision: 단락복원만 / RapidOCR: 쪽번호·마진제거+단락복원)
    text = postprocessor.process_full(text, engine=engine_str)

    # L3 — LLM 문맥 교정
    if llm_correct and llm_corrector.is_available():
        text = llm_corrector.correct(text, budget=llm_budget)

    return PageResult(
        page_num=page_num,
        raw_text=ocr_result.text,
        corrected_text=text,
        engine=ocr_result.engine,
        confidence=ocr_result.confidence,
        figures=placeholders,
        needs_review=needs_review,
        review_reason=review_reason,
    )


# ── 환경 상태 ──────────────────────────────────────────────────────────────────

def _print_env_status(no_fallback: bool, no_spellcheck: bool, no_figure: bool,
                      llm_correct: bool = False, llm_budget: int = 0) -> None:
    if no_fallback:
        vision_str = "[dim]비활성 (--no-fallback)[/dim]"
    elif claude_vision.is_available():
        vision_str = "[green]활성 (API 키 설정됨)[/green]"
    else:
        vision_str = (
            "[yellow]비활성 — API 키 미설정 "
            "(저신뢰도 페이지는 출력에 검토 필요 표시)[/yellow]"
        )

    spell_str = (
        "[dim]비활성[/dim]"        if no_spellcheck
        else "[green]활성[/green]"  if spellcheck.is_available()
        else "[dim]py-hanspell 미설치[/dim]"
    )
    fig_str = "[dim]비활성[/dim]" if no_figure else "[green]활성[/green]"

    if llm_correct:
        llm_str = (
            f"[green]활성 (Claude Haiku"
            + (f" · 최대 {llm_budget}회" if llm_budget else "") + ")[/green]"
            if llm_corrector.is_available()
            else "[yellow]API 키 미설정 (비활성)[/yellow]"
        )
    else:
        llm_str = "[dim]비활성 (--llm-correct 로 활성화)[/dim]"

    console.print(f"  {claude_vision.provider_label()} 폴백  : {vision_str}")
    console.print(f"  L2 맞춤법           : {spell_str}")
    console.print(f"  L3 LLM 보정         : {llm_str}")
    console.print(f"  그림·표 감지        : {fig_str}")


# ── 요약 ──────────────────────────────────────────────────────────────────────

def _print_summary(results: list[FileResult], failed: int) -> None:
    success     = len(results) - failed
    total_pages = sum(r.total_pages for r in results if r.success)
    vision_pages = sum(
        1 for r in results if r.success
        for p in r.pages if p.engine in (OCREngine.GEMINI, OCREngine.CLAUDE_VISION, OCREngine.OLLAMA)
    )
    review_pages = sum(
        1 for r in results if r.success
        for p in r.pages if p.needs_review
    )
    total_figs = sum(
        len(p.figures)
        for r in results if r.success
        for p in r.pages
    )

    table = Table(title="처리 결과 요약", show_header=True)
    table.add_column("항목",             style="dim")
    table.add_column("값",               justify="right")
    table.add_row("처리 파일",            str(len(results)))
    table.add_row("성공",                 f"[green]{success}[/green]")
    table.add_row("실패",                 f"[red]{failed}[/red]" if failed else "0")
    table.add_row("총 페이지",            str(total_pages))
    table.add_row("Vision 폴백 페이지",   str(vision_pages))
    table.add_row("검토 필요 페이지",
                  f"[yellow]{review_pages}[/yellow]" if review_pages else "0")
    table.add_row("감지된 비텍스트 영역", str(total_figs))
    console.print()
    console.print(table)


# ══════════════════════════════════════════════════════════════════════════════
# 직접 실행: python ocr_tool.py --input ... (하위 호환)
# ══════════════════════════════════════════════════════════════════════════════

# click group이지만 단독 실행 시 ocr 서브커맨드처럼 동작하도록 래핑
main = cli

if __name__ == "__main__":
    main()
