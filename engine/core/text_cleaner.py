"""문서 전체 정제(교정) — OCR 원문을 학습용 클린 플레인텍스트로 복원.

OCR(gemini_vision)이 페이지별로 '충실 추출'한 결과에는 쪽번호·러닝헤더·
반복 목차·스캔 구분선·페이지 넘김으로 잘린 단어 등 '레이아웃 노이즈'가 남는다.
이 모듈은 그 원문을 문서 전체 맥락에서 LLM(Gemini 텍스트 호출)으로 교정하여
마커 없는 순수 본문(단락 = 빈 줄)으로 되돌린다.

핵심 설계
─────────
1) 페이지 경계 신호: 페이지 텍스트를 ⟨PAGEBREAK⟩ 센티넬로 이어 붙이고,
   프롬프트에서 "여긴 스캔 페이지 경계 — 삭제하되, 단어/문장이 잘려 넘어가면
   이어 붙여라"고 명시한다. → 페이지 넘김 단어 분리(예: '…성' | '찰의…')를
   LLM이 직접 복원한다.
2) 청크 분할: 모델 출력 토큰 한계 때문에 페이지 단위로 청크를 나눈다.
   청크 경계는 '문장이 깔끔히 끝나는 페이지'에서만 끊어(soft limit), 청크와
   청크 사이에서 단어가 갈라지는 경우를 최소화한다. soft limit을 크게 넘기면
   문장 끝이 아니어도 강제로 끊는다(hard limit).
3) 실패 안전: API 오류 시 규칙 기반 최소 정제(_fallback_clean)로 강등하여
   배치 전체가 죽지 않게 한다.
"""
from __future__ import annotations
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from core import config
from core import gemini_vision as _gemini
from core import ollama_vision as _ollama
from core import noise_rules
from core import spacing, verse_rules
from core import toc_headings
from core.postprocessor import _PAGE_FOOTER_LINE, _FIGURE_CAPTION_RE

# 페이지 경계 센티넬 — 입력에 실제로 등장할 가능성이 없는 문자열
PAGEBREAK = "⟨PAGEBREAK⟩"

# 앞부속(front matter) 제거 규칙을 적용할 '선두 최대 청크 수'.
# 본문이 실질 분량 미만으로 정제돼 body_started가 늦게 켜지는 경우에도, 이 상한을
# 넘어선 청크에는 앞부속 규칙을 절대 적용하지 않아 본문 오삭제를 원천 차단한다.
# (책 부속물은 현실적으로 선두 몇 청크≈수십p 안에 몰려 있음)
_FRONT_MATTER_MAX_CHUNKS = 3

# 사용자 제공 프롬프트 기반. 규칙 4(페이지 경계 처리)만 이 도구용으로 추가.
CLEAN_PROMPT = """당신은 오염된 스캔 서적 OCR 텍스트를 학습용/타이핑용 고품질 plain text로 복원하는 전문 한국어 텍스트 에디터입니다.

서론, 결론, 설명문이나 마크다운 코드 블록(```)을 절대 쓰지 말고 오직 '정제 완료된 본문'만 출력하시오.

[입력 데이터]
다음은 원본 서적에서 오염된 상태로 추출된 한국어 OCR 텍스트의 일부입니다.
'⟨PAGEBREAK⟩' 표시는 스캔된 책의 '페이지 경계'를 뜻합니다.

---
{TEXT}
---

[수행 규칙]
0. 무손실 원칙 (최우선): 입력의 모든 본문 문장을 한 문장도 빠뜨리지 말고 순서 그대로 전부 출력하십시오. 요약·축약·생략·재배열을 절대 하지 마십시오. 내용이 반복적이거나 목록·족보처럼 단조로워 보여도 전부 출력하십시오. '[그림: …]'·'[그림 1: …]'처럼 대괄호로 된 그림 설명 줄도 내용이므로 지우거나 고치지 말고 제자리에 그대로 출력하십시오.교정이 확실하지 않은 부분은 고치지 말고 원문 그대로 두십시오. 아래 3번에 명시된 레이아웃 노이즈 외에는 어떤 것도 삭제하지 마십시오. 출력의 본문 분량은 입력과 거의 같아야 합니다.
1. 문맥 파악 및 단어 복원: 전후 문맥을 통해 오타·변환 오류를 자연스럽게 교정하십시오. (예: '님외'→'남의', '중고는'→'충고는', '가중쇼한'→'가장 중요한')
2. 정밀 띄어쓰기 및 맞춤법: 한국어 어법에 맞게 띄어쓰기와 맞춤법을 완벽히 적용하십시오. (예: '할수있다'→'할 수 있다')
3. 레이아웃 노이즈'만' 제거: 다음 항목만 삭제하십시오 — ① 쪽번호(단독 숫자), ② 러닝헤더/러닝푸터: 각 페이지의 맨 위나 맨 아래(입력 맨 앞·맨 뒤, ⟨PAGEBREAK⟩ 바로 앞뒤 포함)에 붙는 '책 제목' 단독 줄이나 '마태복음 5'처럼 '제목+쪽수' 형태의 짧은 줄, ③ 자모 노이즈(단독 'ㄱ','ㄴ'), ④ 스캔 줄('──'), ⑤ 특수 기호 불릿. 단, 본문에 속한 고유 번호 체계(성경의 장·절 번호, 법조문 번호, 항목·목록 번호 등)는 레이아웃 노이즈가 아니라 콘텐츠이므로 반드시 그대로 보존하십시오.
4. 페이지 경계 처리: '⟨PAGEBREAK⟩' 표시 자체는 삭제하십시오. 그 경계에서 한 단어나 문장이 잘려 다음 페이지로 이어지면 자연스럽게 이어 붙이십시오. (예: '...성' ⟨PAGEBREAK⟩ '찰의...' → '...성찰의...')
5. 외래어 영문 병기 완성: 불완전한 괄호쌍을 완성하십시오. (예: '제물(wealth'→'제물(wealth)', '컴퓨터 Computer)'→'컴퓨터(Computer)')
6. 출력 포맷: 단락마다 빈 줄로 구분하십시오. 정제된 본문만 출력하고 설명은 일절 배제하십시오. 원문에 없는 내용을 지어내지 마십시오.
"""

# 부속물(front matter/뒤부속) 제거 옵션이 켜졌을 때만 프롬프트에 추가되는 규칙
_RULE_FRONT_MATTER = (
    "7. 앞부속 삭제: 이 구간에 표제지(반복된 표제 포함), 판권(저작권)면, 차례(목차), 헌사, "
    "제사(에피그래프)·인용 헌사, 시리즈/총서 광고·소개, 저자·역자·감수자 약력, "
    "출판사 소개·구매 혜택 안내, 디자인·제작 크레딧 등 '책 앞부속(front matter)'이 "
    "있으면 모두 삭제하고 실제 본문만 남기십시오. 단, 감수자·역자 해설, 머리말, 서문, 주석처럼 "
    "내용이 있는 글은 삭제하지 말고 그대로 두십시오. 본문(챕터 내용)이 시작된 뒤부터도 "
    "그대로 두십시오. 삭제 후 남는 본문이 없으면 아무것도 출력하지 마십시오."
)
_MATTER_PRECEDENCE = (
    "★ 최우선 예외 — 아래 7·8번(책 부속물 삭제)은 0번 무손실 원칙보다 먼저 적용합니다. "
    "책 부속물은 '본문'이 아니므로 무손실 대상이 아닙니다."
)
_RULE_COLOPHON = (
    "8. 뒤부속 삭제: 이 구간 끝의 판권지·간기(출판사·주소·전화·팩스·이메일·ISBN·"
    "인쇄정보·CIP 등)와 광고 등 '책 뒤부속'이 있으면 삭제하십시오."
)

# 문장이 '깔끔히' 끝났는지 판단 (이 문자로 끝나면 청크 경계로 안전)
_SENT_END = re.compile(r'[.!?…。！？”’"\'\)\]》」』]\s*$')

# ── 청크 무결성 가드 (2026-07-06 신설) ────────────────────────────────────────
# 실측 사고: 메시지성경 신약 842p 처리에서 일부 청크가 통째로 소실됨
# (계시록 14~18장, 마태복음 4:1-12 등). 원인: Gemini가 안전필터 차단·조기 종료
# 등으로 빈/과소 응답을 돌려줘도 예외가 발생하지 않아(run_text는 ''를 반환)
# 그대로 수용·폐기되던 결함. LLM 정제는 레이아웃 노이즈만 걷어내므로 정상
# 출력의 비공백 길이는 입력의 90% 이상이다 — 아래 임계 미만이면 응답을 버리고
# 1회 재시도, 그래도 미달이면 규칙 기반 최소 정제(_fallback_clean)로 강등해
# 원문을 보존한다(정확성보다 무손실 우선 — 통합_정제규칙서 §14 정신).
_GUARD_RATIO = 0.75          # 일반 청크: 출력/입력 비공백 길이 최소 비율
# 마진 필터 ON: 러닝헤더·쪽번호가 OCR 이전(mask_margins)에 이미 빠져 있으므로
# LLM이 삭제할 노이즈가 거의 없다 → 정상 정제라면 손실률이 더 낮아야 한다.
# 필터 OFF 상태에서 이 값을 쓰면 LLM의 정당한 헤더 삭제까지 과다삭제로 오판되므로
# 필터 여부에 따라 갈라 쓴다.
_GUARD_RATIO_MASKED = 0.90
_GUARD_RATIO_COLOPHON = 0.30 # 뒤부속 제거 청크: 판권 삭제분 감안한 하한
_last_report: dict | None = None


def get_last_report() -> dict | None:
    """직전 clean_pages/iter_clean 실행의 무결성 리포트.

    {"chunks": 총 청크, "fallback_api": API 오류 강등 수,
     "fallback_guard": 무결성 가드 강등 수, "warnings": [사람용 메시지…]}
    """
    return _last_report


def _output_ok(cand: str, joined: str, want_front: bool, want_colophon: bool) -> bool:
    """LLM 출력이 입력 대비 소실 없이 돌아왔는지 검사.

    앞부속 제거 청크는 '전부 삭제 후 빈 출력'이 정당하므로 검사하지 않는다
    (선두 _FRONT_MATTER_MAX_CHUNKS개뿐이라 소실 노출도 제한적).
    """
    if want_front:
        return True
    src = re.sub(r"\s+", "", joined.replace(PAGEBREAK, ""))
    if not src:
        return True
    # 마진 필터 ON이면 프롬프트가 ⟨PAGEBREAK⟩ 를 출력에 남기라고 지시하므로
    # cand 쪽에서도 센티넬을 빼야 입력과 같은 기준으로 비교된다(비율 부풀림 방지).
    ratio = len(re.sub(r"\s+", "", cand.replace(PAGEBREAK, ""))) / len(src)
    if want_colophon:
        floor = _GUARD_RATIO_COLOPHON
    elif config.get_enable_margin_filter():
        floor = _GUARD_RATIO_MASKED
    else:
        floor = _GUARD_RATIO
    return ratio >= floor


# ── 공개 API ──────────────────────────────────────────────────────────────────

def is_available(local: bool = False) -> bool:
    """정제 사용 가능 여부. local=True(비용 우선)는 내장 규칙(항상 가능) 또는 설정 시 Ollama, 아니면 Gemini."""
    if local:
        return config.get_local_cleanup_engine() == "rules" or _ollama.is_text_available()
    if config.get_vision_provider() == "claude":
        return False  # 정제 경로는 아직 Gemini만 구현
    return _gemini.is_available()


def clean_pages(pages: list[str], progress_cb=None, drop_front_matter: bool = False,
                local: bool = False) -> str:
    """
    페이지 텍스트 리스트 → 정제된 전체 본문(문자열).

    progress_cb(done_chunks, total_chunks) 가 주어지면 청크 단위로 호출한다.
    drop_front_matter=True 면 표제지·판권·차례 등 앞뒤 부속물까지 제거한다.
    """
    gen = iter_clean(pages, drop_front_matter=drop_front_matter, local=local)
    try:
        while True:
            done, total = next(gen)
            if progress_cb:
                progress_cb(done, total)
    except StopIteration as stop:
        return stop.value or ""


def iter_clean(pages: list[str], drop_front_matter: bool = False, on_chunk=None,
               report_out: dict | None = None, local: bool = False,
               raw_pages: list[str] | None = None, toc: dict | None = None):
    """
    제너레이터: 청크 처리마다 (done_chunks, total_chunks) 를 yield 하고,
    마지막에 정제된 전체 본문을 return(StopIteration.value)한다.

    local=True: 비용 우선 모드 — 유료 API 호출 0. 기본은 내장 정제규칙(_rules_clean)만,
    local_cleanup_engine='ollama' 면 로컬 exaone3.5 (무결성 가드·강등은 Gemini 경로와 같다).

    drop_front_matter=True: 앞부속(표제지·판권·차례·헌사·시리즈 광고)은 본문이
    나올 때까지 앞 청크에서 제거하고, 마지막 청크의 뒤부속(판권/간기/CIP)도 제거한다.
    부속물은 책의 맨 앞·맨 뒤에 몰려 있으므로 첫 청크(들)와 끝 청크에만 규칙을 적용해
    중간 본문은 건드리지 않는다.

    on_chunk(text): 청크 하나가 정제될 때마다 즉시 호출된다. 호출부가 이를 디스크에
    기록해 두면, 전 구간이 끝나기 전에 실행이 끊겨도(브라우저 종료·API 정지) 이미
    정제된 구간을 잃지 않는다. 반환값은 지금까지처럼 전 구간 완료 후에만 나온다.

    report_out: 넘기면 이 실행의 무결성 리포트를 채워 준다. 여러 파일을 동시에 처리할 때는
    get_last_report()(전역 — 마지막에 끝난 실행 것) 대신 이것을 쓴다.

    raw_pages: 디지털 PDF 원문 쪽(줄바꿈·줄 끝 공백 그대로). 내장 규칙 모드에서 장절 책(성경 등)이면
    이것으로 장·절 단위를 세운다(R-15) — 앞부속 제거는 하지 않는다.
    """
    global _last_report
    _last_report = None
    rules_mode = local and config.get_local_cleanup_engine() == "rules"
    # 장절 책: 쪽 단위 줄 잇기·잡음 줄 판정을 거치면 장 머리·관주가 지워지고 절이 앞 절에 붙는다
    # (king_james_bible 실측) → 원문 줄로 바로 세운다
    if rules_mode and raw_pages and verse_rules.is_verse_book(raw_pages):
        text, st = verse_rules.format_book(raw_pages)
        text = deterministic_preclean(text)
        report = {"chunks": 1, "fallback_api": 0, "fallback_guard": 0, "warnings": [
            f"장절 책으로 정제 — 장 {st['chapters']}개·절 {st['verses']}개,"
            f" 쪽 머리글·쪽번호 {st['headers']}줄 제거 (정제 전 원문 사본에서 확인 가능)"]}
        _last_report = report
        if report_out is not None:
            report_out.update(report)
        if on_chunk is not None:
            on_chunk(text)
        yield 1, 1
        return (text + "\n") if text else ""
    # NFD(자모 분리) 한글은 정제 규칙의 '가-힣' 판정에 안 걸려 잡음으로 지워진다 — 먼저 합친다
    pages = [unicodedata.normalize("NFC", p).strip() for p in pages if p and p.strip()]
    if not pages:
        return ""

    # 통합_정제규칙서 §3.3: 결정론적 사전 정제(D-02) — LLM보다 먼저 적용해
    # 같은 오염이 청크마다 다르게 처리되는 것을 막는다.
    pages = [deterministic_preclean(p) for p in pages]
    # 비용 우선(내장 규칙) — LLM 이 대신 걸러 주지 않으므로 OCR 잡음을 규칙으로 먼저 걷는다
    # (2026-10-16 명언집 5권 실측: 잡음 줄·기호·전각 부호·깨진 머리글이 결과에 그대로 남음)
    header_notes: list[str] = []
    if rules_mode:   # 잡음 기호 제거가 '#' 을 지운다 — 장·절 제목 줄(toc_headings)은 피해서
        pages = [keep_headings(noise_rules.clean_page, p) for p in pages]
    # 청크만 보는 LLM은 책 전체에 반복되는 머리글을 알아보지 못한다 → 문서 단위로 선제거.
    # 장·절 제목 줄은 숨겨 둔다 — 장 표제지의 '# I.자신의 길을 찾는다' 를 머리글 'I.자신의 길을 찾는다 17' 과
    # 같은 글자로 보고 지웠다(1분의사결정 스캔본 실측)
    pages, hidden = _hide_headings(pages)
    pages = strip_running_headers(pages)
    if rules_mode:
        pages, n_hdr = noise_rules.strip_fuzzy_headers(pages)
        # 머리글·꼬리말을 뗀 뒤 드러난 쪽 끝 쪽번호를, 다음 쪽 첫 줄에 붙기 전에 뗀다('61 어떤…'·'그러 23 니', R-16)
        pages, n_pno = noise_rules.strip_page_numbers(pages)
        if n_pno:
            header_notes.append(f"쪽번호 {n_pno}곳 자동 제거 (정제 전 원문 사본에서 확인 가능)")
        # 출처 표기 명언집: 줄 잇기 전에 '명언 → 저자 출처' 그룹 경계를 단락으로 고정(번호를 못 읽은 항목 포함)
        pages, _ = noise_rules.group_attributed_quotes(pages)
        if n_hdr:
            header_notes.append(f"깨져 읽힌 머리글 {n_hdr}곳 자동 제거 (정제 전 원문 사본에서 확인 가능)")
    pages = _show_headings(pages, hidden)
    matter_notes: list[str] = []
    # 머리글을 뗀 뒤라야 잡음 쪽이 짧아져 후보가 된다. 지운 잡음 쪽은 빈 자리로 남겨
    # 부속물 보고와 같은 쪽 번호 기준(내용 있는 쪽 순서)을 유지하고, 부속물 제거 뒤에 걸러 낸다
    noise_notes: list[str] = []
    pages = drop_noise_pages(pages, noise_notes)
    # 차례 기준 장·절 정비(toc_headings — MyOCR 와 같은 파일): 머리글을 뗀 뒤, 줄 잇기 전에 차례 제목과 같은
    # 본문 줄을 '# 제목' 으로. 차례 쪽은 결과 맨 앞 목록으로 옮기므로 본문에서 뺀다 — 부속물 제거가 차례 쪽을
    # 기준으로 본문 시작을 잡으므로 차례 쪽은 그 뒤에 뺀다
    # toc: 호출부(ocr_gradio._toc_prepass)가 쪽마다 줄 잇기 전 원문으로 이미 정비했으면 {entries, stats} —
    # 쪽 교정(postprocessor)이 차례 줄·제목 줄을 앞뒤 줄에 이어 붙이기 전에 해야 찾을 수 있다
    toc_entries, toc_idx = (toc["entries"], []) if toc else toc_headings.extract_toc_text(pages)
    toc_stats: dict = toc["stats"] if toc else {}
    toc_pages_text: dict[str, str] = {}
    if toc_entries and not toc:
        pages = toc_headings.mark_headings(pages, toc_entries, [], toc_stats)
        toc_pages_text = {pages[i]: toc_headings.toc_page_leftover(pages[i], toc_entries) for i in toc_idx}
    if drop_front_matter:
        if rules_mode:   # LLM 문단 분류를 못 쓰니 표지·저자 소개·광고·저작권·차례를 규칙으로 먼저 고른다
            before = pages
            pages = noise_rules.drop_book_matter_rules(pages, noise_notes)
            # 장 표제지('# I.자신의 길을 찾는다' 한 줄)는 짧아 '표지·제목' 쪽으로 지워진다 — 제목 줄은 되살린다
            pages = [p if p.strip() or not b.strip() else
                     "\n\n".join(l for l in b.split("\n") if _MD_HEADING_RE.match(l))
                     for p, b in zip(pages, before)]
        pages, matter_notes = strip_book_matter(pages)
    # 차례 쪽은 차례 항목을 빼고 함께 실린 문장만 남긴다(toc_page_leftover)
    pages = [q for p in pages if (q := toc_pages_text.get(p, p)).strip()]
    if not pages:
        return ""
    matter_notes = header_notes + noise_notes + matter_notes

    soft = config.get_cleanup_chunk_chars()
    hard = soft * 2
    chunks = _chunk_pages(pages, soft, hard)
    total = len(chunks)

    report = {"chunks": total, "fallback_api": 0, "fallback_guard": 0, "warnings": list(matter_notes),
              # 저장 단계가 MD 맨 앞 차례 목록·TXT 변환에 쓴다(ocr_gradio._structured_texts)
              "toc_entries": toc_entries, "toc_note": toc_headings.report(toc_entries, toc_stats)}
    out_parts: list[str] = []

    def merge(res: _ChunkResult) -> None:
        """구간 결과를 순서대로 합친다(경고·체크포인트도 구간 순서 유지)."""
        report["fallback_api"] += res.fallback_api
        report["fallback_guard"] += res.fallback_guard
        report["warnings"].extend(res.warnings)
        if res.cleaned:
            out_parts.append(res.cleaned)
            if on_chunk is not None:
                on_chunk(res.cleaned)

    # 1) 앞부속 구간 — 본문이 시작됐는지(body_started)가 앞 구간 결과에 달려 있어 차례로.
    # 앞부속 제거는 본문 시작 전 + 선두 상한(_FRONT_MATTER_MAX_CHUNKS) 이내에서만 —
    # body_started 가 늦게 켜져도 문서 중반 본문에 삭제 규칙이 새지 않는다.
    i = 0
    while drop_front_matter and i < min(total, _FRONT_MATTER_MAX_CHUNKS):
        yield i, total
        res = _clean_chunk(chunks[i], i, total, front_chunk=True,
                           want_colophon=(i == total - 1), local=local)
        merge(res)
        i += 1
        if len(res.cleaned) >= 100:     # 실질 본문이 나오면 이후 구간은 그대로 둔다
            break

    # 2) 나머지 구간 — 서로 독립이라 동시에 보내고(parallel_workers), 끝나는 대로 순서대로 합친다.
    # 실측(2026-10-05): 구간 1개 정제 약 12초, 차례 처리 시 715쪽 책 약 20분.
    if i < total:
        yield i, total
        ex = ThreadPoolExecutor(max_workers=config.get_parallel_workers())
        try:
            futs = {ex.submit(_clean_chunk, chunks[k], k, total, front_chunk=False,
                              want_colophon=drop_front_matter and k == total - 1,
                              local=local): k
                    for k in range(i, total)}
            ready: dict[int, _ChunkResult] = {}
            next_k, done = i, i
            for fut in as_completed(futs):
                ready[futs[fut]] = fut.result()
                done += 1
                while next_k in ready:
                    merge(ready.pop(next_k))
                    next_k += 1
                yield done, total
        finally:
            # 중단(제너레이터 close) 시 아직 안 보낸 구간은 취소 — 이미 보낸 것만 마저 끝난다
            ex.shutdown(wait=False, cancel_futures=True)

    _last_report = report
    if report_out is not None:
        report_out.update(report)
    yield total, total

    text = "\n\n".join(out_parts)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()

    if config.get_enable_margin_filter():
        text = post_process_pagebreaks(text)
    text, _ = rejoin_split_sentences(text)
    text, _ = repair_open_brackets(text)      # 짝 없는 '[' — 낫표 오인식(2천년의강의)
    text, _ = join_josa_paragraphs(text)
    text, _ = space_glued_marks(text)
    text, n_spaced = strip_spaced_running_header(text)
    if n_spaced:
        report["warnings"].append(f"한 글자씩 띄운 쪽 머리글 {n_spaced}곳 제거 (정제 전 원문 사본에서 확인 가능)")
    if rules_mode:
        text = noise_rules.finish_groups(text)   # 출처 명언 그룹 경계 표지 → 빈 줄(줄 잇기가 끝난 뒤)
        text, _ = noise_rules.repair_quote_brackets(text)   # 출처 인용 괄호 〈〉「」『』 짝 복원
        # 한 줄에 붙은 한글 문장과 영어 문장을 줄바꿈으로 가른다(가독성 — 단락은 유지)
        text, _ = noise_rules.split_language_lines(text)
    # 번호 붙은 항목(명언집 등)마다 단락 복원 — 연속 번호 항목 책일 때만(일반 문서의 표·목록 보호)
    if rules_mode and noise_rules.is_numbered_book(text):
        text, _ = noise_rules.redistribute_number_stacks(text)   # 여백 번호가 몰려 읽힌 것부터 제자리로
        text, _ = noise_rules.paragraph_entries(text)
    if rules_mode:   # 맨 끝 — 줄이 다 자리 잡은 뒤 붙어 쓴 한글만 띄운다(R-17, 넣기만 함)
        # 장·절 제목 줄(R-18)은 빼고 — 맨 앞 차례 목록과 제목 글자가 달라지지 않게
        n_sp = 0

        def _fix(t: str) -> str:
            nonlocal n_sp
            t, k = spacing.fix_spacing(t)
            n_sp += k
            return t
        text = keep_headings(_fix, text) if toc_entries else _fix(text)
        if n_sp:
            report["warnings"].append(f"붙어 쓴 낱말 띄어쓰기 {n_sp}곳 교정 (정제 전 원문 사본에서 확인 가능)")
    text = unify_paragraph_breaks(text)
    guessed: list[dict] = []
    if not toc_entries:     # 차례를 못 찾은 책 — 본문의 장 번호표 줄로 장을 세운다
        text, guessed = infer_chapters(text)
        toc_entries.extend(guessed)       # report(_out) 의 toc_entries 와 같은 목록
    if toc_entries:
        # 흩어진 제목부터 — rematch 가 제목 앞부분만 같은 줄을 먼저 잡아 차례 제목을 그 줄로 바꾼다(삶의지혜)
        text, _ = place_split_titles(text, toc_entries)
        text = rematch_headings(text, toc_entries, toc_stats, respace=rules_mode)   # found·missing 도 여기서
        report["toc_note"] = (f"차례를 찾지 못해 본문에서 장 제목 {len(guessed)}개를 추정(제N장·CHAPTER N 꼴 줄)"
                              if guessed else toc_headings.report(toc_entries, toc_stats))
        if report_out is not None:
            report_out["toc_note"] = report["toc_note"]
        text = toc_headings.space_headings(text)   # 장·절 제목 줄 앞뒤 빈 줄(R-18)

    return (text + "\n") if text else ""


# 앞 줄에 붙여 두는 줄 — 명언 저자 줄('- 벤저민 프랭클린…')·글머리 목록
_ATTACHED_LINE = re.compile(r"[-•·▶■□○●◆※]\s")


def unify_paragraph_breaks(text: str) -> str:
    """단락 사이를 빈 줄 하나로 통일한다(정제규칙서 C-08 '단락 = 빈 줄').

    줄 잇기가 끝난 뒤 남은 줄바꿈은 단락 경계인데, 쪽 안에서는 줄바꿈 하나·쪽 경계에서는 빈 줄이라
    두 가지가 섞여 있었다(02_결과 14권 실측: 빈 줄 74~230곳, 줄바꿈 하나 1천~1만여 곳, 2026-10-18).
    글머리(- •)로 시작하는 줄은 앞 줄과 한 덩어리로 둔다 — 명언과 저자 줄이 갈라지지 않게.
    """
    lines = [l.strip() for l in text.split("\n")]
    out: list[str] = []
    for line in lines:
        if not line:
            continue
        if out:
            out.append("\n" if _ATTACHED_LINE.match(line) else "\n\n")
        out.append(line)
    return "".join(out)


# 붙이면 안 되는 1음절 단어 — 관형사·대명사·부사 + 의존명사(수·것…)
# ('할 수⟨PB⟩있다'→'할 수있다' 오결합, QA ISSUE-H 2026-10-03)
_EXCLUDE_1CHAR = {"이", "그", "저", "한", "새", "첫", "옛", "딴", "온", "몇", "늘", "더", "덜", "참",
                  "수", "것", "줄", "데", "바", "뿐", "듯", "채", "때", "등", "및", "또", "곧",
                  "잘", "못", "안", "좀", "다", "꼭", "왜"}

# 뒤 어절 '전체'가 조사일 때만 결합 — 첫 글자만 보면 '가족을'·'도시'까지 붙었다
_JOSA_WORDS = {"을", "를", "은", "는", "이", "가", "의", "에", "와", "과", "로", "도", "만", "께",
               "으로", "에서", "에게", "에는", "에도", "까지", "부터", "처럼", "보다", "께서",
               "한테", "이나", "나", "랑", "이랑", "조차", "마저", "으로는", "에서는", "와는", "과는"}


def _glue(before: str, after: str) -> str:
    """끊긴 두 어절 사이에 넣을 문자 — 한 단어가 잘린 것이면 '', 아니면 ' '."""
    is_hangul = lambda c: "가" <= c <= "힣"
    if is_hangul(before[-1]) and is_hangul(after[0]):
        # 1. 앞 단어가 1음절이고 제외 목록에 없는 경우 -> 결합 ('성'+'찰의')
        if len(before) == 1 and before not in _EXCLUDE_1CHAR:
            return ""
        # 2. 뒤 어절 전체가 조사인 경우 -> 결합 ('결정'+'을')
        if re.sub(r'[^가-힣]+$', '', after) in _JOSA_WORDS:
            return ""
        # 3. 한 낱말이 갈라진 것('훌륭한전'+'략일', '도구로'+'서의') — Kiwi 점수로 판정
        from core import spacing
        if spacing.is_split_word(re.sub(r'^[^가-힣]+', '', before), re.sub(r'[^가-힣]+$', '', after)):
            return ""
    # 4. 그 외에는 일반적인 단어 경계로 보고 띄어씀
    return " "


# ── LLM 출력 후처리 (2026-10-05 정제본 7권 대조) ──────────────────────────────
# 실측: LLM이 쪽 경계·청크 경계·쪽 안 줄바꿈을 잇지 못해 문장 중간에서 줄·문단이 끊긴 곳이
# 권당 최대 84건(성경)·83(고백록)·82(드러커). 띄어쓰기는 _glue 추정이라 '크라이슬 러'처럼 틀릴 수 있다.
_SPLIT_MIN_LEN = 60     # 이보다 짧은 줄은 제목·시행일 수 있어 잇지 않는다
_TOC_LIKE_RE = re.compile(r'\s\d{1,4}$| · ')
# 단어 첫머리에 올 수 없는 어미·조사 — 다음 줄이 이걸로 시작하면 앞 줄이 짧아도(제목일 수 없으니) 잇는다.
# 실측(MyOCR 1분의사결정): '…잘못을 발견하' ⟨쪽 경계⟩ '며, 아무도…'. '이·가·서·만·도' 는 '이 책'·'서 있다'·
# '만 원' 처럼 독립 단어로도 쓰여 제외한다
_BOUND_START = {"며", "으며", "니", "으니", "니까", "지만", "면서", "도록", "거나", "는데", "은데", "으면",
                "는", "은", "을", "를", "에", "의", "에서", "에게", "으로", "로", "께서",
                "었다", "았다", "였다", "했다", "한다", "된다", "합니다", "입니다", "습니다", "니다"}


def _starts_bound(line: str) -> bool:
    # 첫 토큰의 앞쪽 한글만 본다 — 전각 쉼표가 반각으로 바뀌며 '며,아무도' 처럼 붙어 오는 경우가 있다
    m = re.match(r'[가-힣]+', line.strip())
    return bool(m) and m.group(0) in _BOUND_START


# ── OCR 잡음 쪽 (MyOCR text_refiner 이식) ───────────────────────────────────────
# 실측(MyOCR 1분의사결정 91쪽): '…해낼 만큼' ⟨'인걸' 두 글자뿐인 쪽⟩ '의 인격을…' — 그림·얼룩을 읽은 쪽이
# 문장 한가운데 끼어든다. 글자 수만으로는 '서문'·'PART 1' 같은 제목 쪽과 구분되지 않으므로
# 문장이 그 쪽을 건너 실제로 이어질 때 — 앞 쪽은 문장이 끝나지 않았고 다음 쪽은 단어 첫머리에 올 수 없는
# 어미·조사('의 인격을')로 시작 — 만 잡음으로 본다. 앞 쪽 끝만 보면 서명('스펜서 존슨') 뒤 장 제목 쪽을 지운다.
_NOISE_PAGE_MAX = 15        # 의미 있는 글자(한글·영문·숫자)가 이보다 적은 쪽만 후보


def drop_noise_pages(pages: list[str], notes: list[str]) -> list[str]:
    """문장 중간에 끼어든 짧은 잡음 쪽을 비운다(쪽 순서 유지). 지운 쪽은 notes 에 보고."""
    out = list(pages)
    dropped = []
    for i, p in enumerate(out):
        s = p.strip()
        if not s or len(re.findall(r'[가-힣A-Za-z0-9]', s)) >= _NOISE_PAGE_MAX or _SENT_END.search(s):
            continue
        prev = next((q.rstrip() for q in reversed(out[:i]) if q.strip()), "")
        nxt = next((q for q in out[i + 1:] if q.strip()), "")
        if prev and nxt and not _SENT_END.search(prev) and _starts_bound(nxt):
            dropped.append(f"{i + 1}번째 쪽('{' '.join(s.split())[:10]}')")
            out[i] = ""
    if dropped:
        notes.append(f"OCR 잡음 쪽 자동 제거 — {', '.join(dropped)} (정제 전 원문 사본에서 확인 가능)")
    return out


# 실측(드러커 471쪽): OCR이 책과 무관한 글('디지털 전환…')을 지어낸 쪽을 LLM 정제가 조용히 지우고
# 앞뒤 쪽을 매끄럽게 이어 붙여, 쪽 하나가 사라진 것이 결과만 봐서는 보이지 않았다.
# 분량 가드(구간 75%)는 1쪽 삭제를 통과시킨다 → 원문 문장이 연달아 빠진 곳을 따로 알린다.
_DROP_RUN = 5           # 원문 문장이 이만큼 연달아 출력에 없으면 '통째로 빠짐'


def _dropped_run(cleaned: str, source: str) -> str | None:
    """원문 문장이 _DROP_RUN개 이상 연달아 출력에 없으면 그 첫 문장을, 아니면 None."""
    ns = lambda s: re.sub(r"\s", "", s)
    out = ns(cleaned)
    run = 0
    for sent in re.split(r"(?<=[.?!])\s+", source.replace(PAGEBREAK, "\n")):
        key = ns(sent)
        if len(key) < 30:
            continue
        if key[8:20] in out:
            run = 0
            continue
        run += 1
        if run == 1:
            first = sent.strip()
        if run >= _DROP_RUN:
            return first
    return None


_MD_HEADING_RE = re.compile(r'^#{1,6} \S')


_HIDDEN_HEADING_RE = re.compile(r'⟨HEAD(\d+)⟩')


def _hide_headings(pages: list[str]) -> tuple[list[str], list[str]]:
    """장·절 제목 줄을 ⟨HEADn⟩ 표지로 바꿔 둔다(머리글 제거 규칙이 제목을 머리글로 보지 않게)."""
    store: list[str] = []

    def sub(m):
        store.append(m.group(0))
        return f"⟨HEAD{len(store) - 1}⟩"
    return [re.sub(r'(?m)^#{1,6} \S.*$', sub, p) for p in pages], store


def _show_headings(pages: list[str], store: list[str]) -> list[str]:
    if not store:
        return pages
    return [_HIDDEN_HEADING_RE.sub(lambda m: store[int(m.group(1))], p) for p in pages]


_BACK_MATTER = {"주석", "미주", "후주", "참고문헌", "찾아보기", "색인", "주", "notes"}


def _back_matter_fences(lines: list[str], entries: list[dict]) -> list[int]:
    """차례 항목마다 넘으면 안 되는 줄 — 그 항목 뒤에 오는 뒷부분 항목('주석'·'찾아보기')이 본문에 따로 선 줄의
    마지막 자리. 본문에 없는 '서문' 이 맨 뒤 주석 속 '서문' 줄을 잡아 그 뒤 장 제목이 모두 주석 속 장 이름 줄에
    섰다(Alianment정렬 2026-11-07)."""
    fences = [len(lines)] * len(entries)
    fence = len(lines)
    for ei in range(len(entries) - 1, -1, -1):
        fences[ei] = fence
        t = _strip_label(entries[ei]["title"]).strip()
        if t.lower() in _BACK_MATTER:
            k = next((i for i in range(len(lines) - 1, -1, -1)
                      if lines[i].strip().lstrip("#").strip() == t), None)
            if k is not None:
                fence = min(fence, k)
    return fences


# 차례 쪽 비율 둘레 이만큼(문서 줄 비율) 안에서만 제목을 찾는다. 점검(0.35)보다 넉넉히 — 짧은 줄이 많은 찾아보기가
# 뒤에 길면 줄 비율이 쪽 비율보다 처진다(ProfitFromTheCore 5장: 쪽 0.90, 줄 0.56). 요약 쪽 오인은 0.75 차이
_TITLE_PAGE_WINDOW = 0.5
_TITLE_PAGE_MIN_LINES = 200  # 책 한 권쯤 되는 글에서만 — 짧은 글은 줄 비율이 쪽 비율을 따르지 않는다


def _best_title_lines(lines: list[str], entries: list[dict], at: list[int | None], exact: bool) -> dict[int, int]:
    """못 찾은 차례 항목을 차례 순서대로 앞뒤로 이미 선 제목 줄 사이에서 찾는다 — 뒷부분 항목 줄 너머는 보지
    않는다(_back_matter_fences). {항목 번호: 줄 번호}
    가장 많이 세우는 짝을 고르는 방식은 장 여는 쪽의 절 이름 목록 줄을 골라 진짜 절 자리를 버렸다(2x2Matrix 64→61)."""
    at = list(at)
    fences = _back_matter_fences(lines, entries)
    last_page = max((e["page"] for e in entries if e.get("page")), default=0)
    out: dict[int, int] = {}
    for ei, e in enumerate(entries):
        if at[ei] is not None:
            continue
        lo = max((k for k in at[:ei] if k is not None), default=-1) + 1
        hi = min(min((k for k in at[ei + 1:] if k is not None), default=len(lines)), fences[ei])
        if e.get("page") and last_page and len(lines) >= _TITLE_PAGE_MIN_LINES:
            # 차례 쪽번호 둘레만 — 책 끝 요약이 장 제목을 오자 없이 되풀이해 그쪽을 먼저 잡고, 뒤 장들이 모두 따라갔다
            # (SalesSuperStar). 범위는 점검의 '쪽과 먼 제목' 과 같은 비율
            mid = e["page"] / last_page * len(lines)
            lo = max(lo, int(mid - _TITLE_PAGE_WINDOW * len(lines)))
            hi = min(hi, int(mid + _TITLE_PAGE_WINDOW * len(lines)) + 1)
        k = _title_line(lines, lo, hi, e["title"], exact)
        if k is not None:
            out[ei] = at[ei] = k
    return out


def rematch_headings(text: str, entries: list[dict], stats: dict, respace: bool,
                     keep_titles: bool = False, exact: bool = False) -> str:
    """정제 뒤 한 번 더 — 줄 잇기 전 원문에서 못 찾은 차례 제목이 정제본에 따로 선 줄로 남아 있으면 제목으로
    (DrivingSalesmanPowerSkill '03 이제 세일즈를 해 볼까' 등 19곳, 2026-10-24 실측).
    무손실: 한 줄만, 앞뒤로 찾은 차례 제목 줄 사이에서만 찾고, 그 줄 글자 그대로 '#' 만 붙인다 — 차례 목록 제목을
    그 줄 글자로 맞춘다. mark_headings 를 다시 쓰면 여러 줄 잇기가 옆 본문 줄('피터 드러커')을 지우고 이미 넣은
    장 제목을 또 넣었다(NextSociety). 본문 문장 속 같은 말은 줄이 길어 걸리지 않는다.
    stats(found·inferred·missing)를 최종 결과로 고친다. respace: 제목도 본문과 같은 띄어쓰기(R-17) —
    차례 목록(entries)과 본문 제목 줄을 함께 바꿔 같은 글자로 둔다."""
    lines = text.split("\n")
    at: list[int | None] = []          # 차례 항목마다 본문 제목 줄 번호(없으면 None) — 차례 순서대로
    pos = 0
    for e in entries:
        h = toc_headings._heading(e)
        k = next((i for i in range(pos, len(lines)) if lines[i] == h), None)
        at.append(k)
        if k is not None:
            pos = k + 1
    for ei, k in _best_title_lines(lines, entries, at, exact).items():
        e = entries[ei]
        if k is not None:
            if not keep_titles:   # 사람이 확인한 차례(책정보 파일)면 제목은 차례 글자 그대로 — OCR 오타 줄로 안 바꿈
                e["title"] = lines[k].strip()
            # 한글 제목 줄 뒤에 붙은 영어 원제('시나리오 경영Scenarios', 2x2Matrix)는 제목 아래 문단으로 남긴다
            latin = _latin_tail(lines[k], e["title"]) if keep_titles else ""
            lines[k] = toc_headings._heading(e) + (f"\n\n{latin}" if latin else "")
            at[ei] = k
    text = "\n".join(lines)
    # 줄 단위로 다시 나눠 센다 — 영어 원제를 아래로 내린 줄은 '제목\n\n원제' 한 덩어리다(Helping 7장 'Teamwork')
    heads = {l for l in text.split("\n") if _MD_HEADING_RE.match(l)}
    missing = [e["title"] for e in entries if toc_headings._heading(e) not in heads]
    inferred = stats.get("inferred", 0)
    stats.update(found=len(entries) - len(missing) - inferred, inferred=inferred, missing=missing)
    if respace:
        renamed = {}
        for e in entries:
            old = toc_headings._heading(e)
            e["title"] = spacing.fix_spacing(e["title"])[0]
            renamed[old] = toc_headings._heading(e)
        text = "\n".join(renamed.get(l, l) for l in text.split("\n"))
    return text


# ── 장 번호 표지를 사이에 두고 흩어진 장 제목 (삶의지혜_자아편 실측, 2026-11-07) ─────────────
# 장 첫 쪽의 큰 제목이 '…사람'과 / '…사람'은 / 03 / 목표가 다르다' 처럼 여러 문단으로 갈리거나 본문 첫 문장에
# 붙어('…않는다고 08 생각해 보라 나태함은…') 한 줄만 보는 rematch_headings 로는 21장 중 0장을 찾았다.
# 오삭제 방지: 차례 번호와 같은 번호가 따로 선 낱말('13년'·'10만' 은 아님) 둘레 7문단 안에서만, 그 번호를 낀
# 연속 낱말 묶음이 차례 제목과 _SPLIT_TITLE_MIN 이상 같을 때만 그 낱말들을 제목 줄 하나로 바꾼다.
_SPLIT_TITLE_MIN = 0.8
_SPLIT_TITLE_REACH = 3      # 번호 문단 앞뒤로 볼 문단 수
_SPLIT_TITLE_SLACK = 4      # 제목 낱말 수보다 이만큼 더 긴 묶음까지 본다


def _split_title_span(words: list[tuple[int, str]], t: int, target: str) -> tuple[int, int] | None:
    """번호 낱말 words[t] 를 낀 [i, j] 중 차례 제목(target, 한글·숫자만)과 가장 같은 묶음."""
    norms = [toc_headings._norm_hangul(w) for _, w in words]
    reach = len(target) // 2 + _SPLIT_TITLE_SLACK
    best, span = 0.0, None
    for i in range(max(0, t - reach), t + 1):
        for j in range(t, min(len(words), t + reach + 1)):
            cand = "".join(norms[i:t]) + "".join(norms[t + 1:j + 1])
            if not cand or len(cand) > len(target) * 1.5:
                continue
            r = SequenceMatcher(None, cand, target, autojunk=False).ratio()
            if r > best:
                best, span = r, (i, j)
    return span if best >= _SPLIT_TITLE_MIN else None


def _entry_positions(paras: list[str], entries: list[dict]) -> list[int | None]:
    """차례 항목마다 본문 제목 줄 위치 — 차례 순서대로 앞에서부터 짝짓는다. 같은 제목이 두 번 있는 차례
    (12년차기획맨: PART I·II 모두 '1. 외부 환경 분석')에서 글자로만 찾으면 두 항목이 한 줄을 같이 가리켰다."""
    out: list[int | None] = []
    heads = [(i, p) for i, p in enumerate(paras) if p.startswith("#")]
    pos = 0
    for e in entries:
        h = toc_headings._heading(e)
        k = next((i for i, p in heads if i >= pos and p == h), None)
        out.append(k)
        if k is not None:
            pos = k + 1
    return out


def _bounds(paras: list[str], entries: list[dict], ei: int) -> tuple[int, int, bool]:
    """ei 번째 항목을 찾을 범위 — 앞 항목 제목 줄 뒤부터 뒤 항목 제목 줄 앞까지. (lo, hi, 이미 있음)"""
    at = _entry_positions(paras, entries)
    lo = max((k for k in at[:ei] if k is not None), default=-1) + 1
    hi = min((k for k in at[ei + 1:] if k is not None), default=len(paras))
    return lo, hi, at[ei] is not None


# 점 없는 번호표 '07 고객의 …'(MBA명강의)도 — 빈칸 뒤 한글일 때만('30억'·'2천년' 처럼 붙은 수는 그대로)
_EXTRA_LABEL_RE = re.compile(r'^\s*(?:제\s*)?\d+\s*[장부편강권]\s*|^\s*\d{1,2}\s+(?=[가-힣])')


def _strip_label(title: str) -> str:
    """차례 제목의 번호표를 뗀다 — toc_headings._LABEL_RE(MyOCR 와 같은 파일이라 손대지 않음)가 모르는
    '제1강'(2천년의강의)·'6장'(2x2Matrix) 꼴까지."""
    return _EXTRA_LABEL_RE.sub("", toc_headings._LABEL_RE.sub("", title))


def place_split_titles(text: str, entries: list[dict], tail: bool = False) -> tuple[str, int]:
    """본문에 아직 없는 번호 붙은 차례 제목('03. …')을, 번호 표지 둘레에 흩어진 제목 글자 자리에 세운다.
    (본문, 세운 수). 제목 줄 글자는 차례 제목 그대로 — 흩어진 OCR 글자 대신."""
    paras = text.split("\n\n")
    placed, start = 0, 0
    for ei, e in enumerate(entries):
        head = toc_headings._heading(e)
        at = _entry_positions(paras, entries)
        if at[ei] is not None:
            start = at[ei] + 1
            continue
        m = re.match(r'0*(\d{1,3})\.\s*(.+)', e["title"])
        if not m:
            continue
        tok = re.compile(rf'0*{m.group(1)}[.,]?')   # '2.' 도 — '2. 내부 역량 분석 내부 역량 분석 업무를…'(12년차기획맨)
        target = toc_headings._norm_hangul(m.group(2))
        # 뒤 차례 항목의 제목 줄이 이미 섰으면 그 앞까지만 — 12년차기획맨 '1. 투자 유치를 위한 IR 자료' 가 다음 절들
        # 뒤에 되풀이된 같은 글자에 섰다
        later = min((k for k in at[ei + 1:] if k is not None and k >= start), default=len(paras))
        for k in range(start, later):
            if _MD_HEADING_RE.match(paras[k]) or not any(tok.fullmatch(w) for w in paras[k].split()):
                continue
            lo = k
            while lo > max(start, k - _SPLIT_TITLE_REACH) and not _MD_HEADING_RE.match(paras[lo - 1]):
                lo -= 1
            hi = k
            while hi < min(len(paras) - 1, k + _SPLIT_TITLE_REACH) and not _MD_HEADING_RE.match(paras[hi + 1]):
                hi += 1
            words = [(p, w) for p in range(lo, hi + 1) for w in paras[p].split()]
            t = next(x for x, (p, w) in enumerate(words) if p == k and tok.fullmatch(w))
            span = _split_title_span(words, t, target)
            if span is None:
                continue
            (pi, _), (pj, _) = words[span[0]], words[span[1]]
            before = " ".join(w for x, (p, w) in enumerate(words) if p == pi and x < span[0])
            after = " ".join(w for x, (p, w) in enumerate(words) if p == pj and x > span[1])
            paras[pi:pj + 1] = [s for s in (before, head, after) if s]
            start = pi + (2 if before else 1)
            placed += 1
            break
    placed += _place_unnumbered_titles(paras, entries)
    placed += _place_latin_led_titles(paras, entries)
    if tail:   # 앞뒤 제목이 다 선 뒤에만(좁은 범위) — 넓은 범위에서는 '죄' 같은 짧은 제목이 엉뚱한 문단 끝에 걸린다
        placed += _place_tail_titles(paras, entries)
    return "\n\n".join(paras), placed


def _place_latin_led_titles(paras: list[str], entries: list[dict]) -> int:
    """문단 첫머리가 한글 차례 제목이고 바로 영어 원제가 이어지는 꼴('문제와 해결책Problems and Solutions Watts
    Wacker, Jim Taylor 미래가 나타나길…', 2x2Matrix 매트릭스 쪽) — 첫머리를 제목 줄로 세우고 나머지는 그대로
    다음 문단으로. 앞뒤 차례 제목 줄 사이에서."""
    placed = 0
    for ei, e in enumerate(entries):
        title = e["title"].strip()
        if len(re.sub(r'\s+', '', title)) < 3:   # 영어 섞인 제목도('T-그룹리더십T-Group…'·'MBTI Myers Briggs…')
            continue
        lo, hi, found = _bounds(paras, entries, ei)
        if found:
            continue
        # 번호표를 뗀 제목도('제2강 비교력' → '비교력 "지금 폐하께서는…"'), 뒤엔 영어 원제나 인용문
        bare = _strip_label(title).strip()
        leads = [re.compile(r'\s?'.join(map(re.escape, re.sub(r'\s+', '', t))) + r'\s?(?=[A-Za-z"“\'‘「『])')
                 for t in dict.fromkeys([title, bare]) if len(re.sub(r'\s+', '', t)) >= 3]
        for p in range(lo, hi):
            m = next((m for m in (ld.match(paras[p]) for ld in leads) if m), None)
            if m:
                paras[p:p + 1] = [toc_headings._heading(e), paras[p][m.end():]]
                placed += 1
                break
    return placed


_ITEM_NO_RE = re.compile(r'\d{1,4}')


def _place_tail_titles(paras: list[str], entries: list[dict]) -> int:
    """앞 항목 문단 끝에 붙은 짧은 주제 제목 — '…진리가 그 속에 없다 부족함' / '329' / 다음 명언
    (1001가지기독교명언 22곳). 문단 끝 낱말들이 제목과 같은 글자이고 바로 다음 문단이 항목 번호뿐일 때만,
    앞뒤 차례 제목 줄 사이에서."""
    placed = 0
    for ei, e in enumerate(entries):
        head = toc_headings._heading(e)
        lo, hi, found = _bounds(paras, entries, ei)
        if found:
            continue
        want = re.sub(r'\s+', '', e["title"])
        # 띄어 쓴 제목 글자 사이엔 빈칸이 있어도 되고, 앞 문장 끝('…해가된다 교회출석'·'…저항해야 한다시련')에 붙어도 된다
        tail_re = re.compile(r'(?:(?<=\s)|(?<=[다.!?"”’)\]]))' + r'\s?'.join(map(re.escape, want)) + r'\s*$')
        for p in range(lo, min(hi, len(paras) - 1)):
            m = tail_re.search(paras[p])
            if m and m.start() > 0 and re.match(r'\d{1,4}(?:\s|$)', paras[p + 1].strip()):
                paras[p:p + 1] = [paras[p][:m.start()].rstrip(), head]
                placed += 1
                break
    return placed


_ITEM_CHAIN_MIN = 20     # 이만큼 이어지는 번호 문단이 있어야 번호 항목 책(명언집)으로 본다
_ITEM_CHAIN_GAP = 5      # 번호를 OCR 이 몇 개 못 읽어도 이어진 것으로
_ITEM_CHAIN_START = 20


def attach_item_numbers(text: str) -> tuple[str, int]:
    """번호 항목 책에서 따로 선 항목 번호 문단('77')을 다음 항목 앞에 붙인다 — '77 고립된 그리스도인은…'.
    번호가 1, 2, 3 … 늘어나는 사슬(빈 번호 _ITEM_CHAIN_GAP 까지)에 든 것만 — 쪽번호·연도는 사슬을 이루지 않는다.
    (본문, 붙인 수)"""
    paras = text.split("\n\n")
    chain, last = [], 0
    for i, p in enumerate(paras):
        s = p.strip()
        # 첫 번호는 _ITEM_CHAIN_START 까지 — OCR 이 한 자리 번호(1~9)를 못 읽어 10 부터 시작했다(기독교명언)
        if _ITEM_NO_RE.fullmatch(s) and 0 < int(s) - last <= (_ITEM_CHAIN_START if not last else _ITEM_CHAIN_GAP):
            chain.append(i)
            last = int(s)
    if len(chain) < _ITEM_CHAIN_MIN:
        return text, 0
    n = 0
    for i in reversed(chain):
        if i + 1 < len(paras) and not _MD_HEADING_RE.match(paras[i + 1]) \
                and not _ITEM_NO_RE.fullmatch(paras[i + 1].strip()):
            paras[i:i + 2] = [paras[i].strip() + " " + paras[i + 1]]
            n += 1
    return "\n\n".join(paras), n


_GLUED_LIST_NO_RE = re.compile(r'(\d{1,2})(?=[가-힣])(?!(?:년|월|일|시|분|초|대|개|명|번|차|장|절|부|권|세|살|위|등|배|가지|퍼센트|'
                               r'만|천|억|조|원|달러|층|호|회|주|점|편|쪽|건|곳|종|단계|인|척|평|명|쇄|백|판|채|마리|벌|그루))')
_LIST_NO_RE = re.compile(r'(\d{1,2})(?=\s?[가-힣])')
_LIST_RUN_REACH = 4      # 이웃 번호를 찾아볼 문단 수


def space_list_numbers(text: str) -> tuple[str, int]:
    """문단 첫머리에 글과 붙은 목록 번호('1오늘부터 …' / '2자신의 …')를 띄운다(CreatYourOwnFuture, 10권 합 393곳).
    이웃 문단(앞뒤 _LIST_RUN_REACH 안)이 n±1·n±2 번호로 시작할 때만 — 외딴 숫자와 단위가 붙은 수('2년 만에'·
    '24시간'·'10대와')는 그대로. 띄운 모양은 attach_item_numbers 와 같은 '번호 글'. (본문, 띄운 수)"""
    paras = text.split("\n\n")
    nums = {}
    for i, p in enumerate(paras):
        m = _LIST_NO_RE.match(p)
        if m and not p.startswith("#"):
            nums[i] = int(m.group(1))
    n = 0
    for i, p in enumerate(paras):
        m = _GLUED_LIST_NO_RE.match(p)
        if not m or i not in nums:
            continue
        k = nums[i]
        if any(1 <= abs(nums.get(j, -99) - k) <= 2 for j in range(i - _LIST_RUN_REACH, i + _LIST_RUN_REACH + 1) if j != i):
            paras[i] = m.group(1) + " " + p[m.end():]
            n += 1
    return "\n\n".join(paras), n


_TOC_PIECE_RE = re.compile(r'(?<![\d.])(?=\d{1,2}\.\s)')
_TOC_REMNANT_MIN = 3          # 차례 항목과 같은 문단이 이만큼 이어져야 남은 차례로 본다
_TOC_REMNANT_NOPAGE = 6       # 쪽번호가 하나도 없는 덩어리는 이만큼 — 부·편·장 여는 쪽(3개)은 차례가 아니다
_TOC_REMNANT_FRONT = 0.15     # 책 앞쪽(문단 수 기준)에서만
_TOC_REMNANT_EDGE_RE = re.compile(r'^제\s*\d+\s*[편부장]|차\s?례')


def drop_toc_remnant(text: str, entries: list[dict]) -> tuple[str, int]:
    """본문 앞쪽에 남은 OCR 차례 덩어리('11. 성숙함은 … 111' / '210' / '역자후기' …)를 지운다 — 맨 앞 '# 차례'
    목록과 같은 글이라 무손실(삶의지혜_자아편: 차례 쪽 일부만 읽혀 본문 서문 뒤에 남았다). 문단마다 번호 항목으로
    나눠 모든 조각이 차례 제목(_MATCH_MIN 이상)이나 쪽번호일 때만 차례 문단, 그런 문단이 3개 이상 이어질 때만.
    (본문, 지운 문단 수)"""
    titles = [toc_headings._norm_hangul(e["title"]) for e in entries]
    titles = [t for t in titles if len(t) >= 2]
    if not titles:
        return text, 0
    paras = text.split("\n\n")

    def kind(p: str) -> int | set:  # 맞은 차례 제목 번호 집합 = 차례 제목 문단, 1 = 쪽번호뿐, 0 = 본문
        if re.fullmatch(r'[\d\s.…·]+', p):
            return 1
        if _MD_HEADING_RE.match(p):
            return 0
        pieces = [re.sub(r'[\s.…·]*\d{1,4}$', '', s.strip()) for s in _TOC_PIECE_RE.split(p) if s.strip()]
        hit = set()
        for s in pieces:
            n = toc_headings._norm_hangul(s)
            ratios = [SequenceMatcher(None, n, t, autojunk=False).ratio() for t in titles] if n else [0]
            if max(ratios) < toc_headings._MATCH_MIN:
                return 0
            hit.add(ratios.index(max(ratios)))
        return hit

    limit = max(1, int(len(paras) * _TOC_REMNANT_FRONT))
    kinds = [kind(p) for p in paras[:limit]]
    i = 0
    while i < len(kinds):
        if not kinds[i]:
            i += 1
            continue
        j = i
        while j < len(kinds) and kinds[j]:
            j += 1
        # 서로 다른 제목으로 센다 — 장 여는 쪽의 '장 제목 두 번 + 첫 절 제목' 은 차례가 아니다(Helping 2장).
        # 쪽번호(끝 숫자·숫자뿐 문단)가 없으면 제목이 아주 많을 때만 — '제1부'/'제1편 …'/'제1장 …' 여는 쪽(전쟁론)
        distinct = len(set().union(*(k for k in kinds[i:j] if isinstance(k, set))))
        paged = any(k == 1 or re.search(r'\d\s*$', paras[x]) for x, k in zip(range(i, j), kinds[i:j]))
        if distinct >= (_TOC_REMNANT_MIN if paged else _TOC_REMNANT_NOPAGE):
            a, b = i, j      # 앞뒤 '제1편 자 아 편'·'부록1. … 차례' 같은 차례 머리·꼬리 한 문단씩
            if a and len(paras[a - 1]) <= 60 and _TOC_REMNANT_EDGE_RE.search(paras[a - 1]):
                a -= 1
            if b < len(paras) and len(paras[b]) <= 60 and _TOC_REMNANT_EDGE_RE.search(paras[b]):
                b += 1
            del paras[a:b]
            return "\n\n".join(paras), b - a
        i = j
    return text, 0


_TOC_TABLE_MIN = 3      # 차례 제목이 이만큼 든 표는 부(PART) 여는 쪽의 작은 차례
# 표 글자 중 차례 제목 몫 — 작은 차례는 0.88~1.08(12년차기획맨·DrivingSalesman), 책 내용 개념도는 0.10(IDEAHACKS49)
_TOC_TABLE_SHARE = 0.5


def _contains_title(hay: str, title: str) -> bool:
    """정규화한 글(hay)에 차례 제목이 OCR 오자 몇 개를 빼고 이어져 들어 있나 — '도출합 것인가'·'전숨이'."""
    if len(title) < 4:
        return False
    m = SequenceMatcher(None, hay, title, autojunk=False).find_longest_match(0, len(hay), 0, len(title))
    if m.size == len(title):
        return True
    a = max(0, m.a - m.b)
    return SequenceMatcher(None, hay[a:a + len(title) + 2], title, autojunk=False).ratio() >= toc_headings._MATCH_MIN


def drop_toc_tables(text: str, entries: list[dict]) -> tuple[str, int]:
    """부(PART)를 여는 쪽의 작은 차례가 OCR 에서 표로 읽힌 것('| 기획의 실행 | PART III | … Chapter 7. 기획서,
    제안에서 성공까지 1.기획서의 제안과 승인 …')을 지우고, 그 표에 든 가장 높은 단계 차례 항목의 제목 줄 하나로
    바꾼다 — 맨 앞 차례 목록과 같은 글이라 무손실(12년차기획맨 3곳). 그대로 두면 제목 맞추기가 표 속 글자를 먼저
    잡아 진짜 장·절 자리를 놓치고 표를 깨뜨렸다. 차례 제목이 _TOC_TABLE_MIN 개 이상 든 표만. (본문, 지운 표 수)"""
    if not entries:
        return text, 0
    norm = toc_headings._norm_hangul
    titles = [(e, norm(_strip_label(e["title"])) or norm(e["title"])) for e in entries]
    paras = text.split("\n\n")
    out, n, i = [], 0, 0
    while i < len(paras):
        j = i
        while j < len(paras) and paras[j].strip() and all(l.lstrip().startswith("|")
                                                            for l in paras[j].split("\n") if l.strip()):
            j += 1
        if j == i:
            out.append(paras[i])
            i += 1
            continue
        hay = norm(" ".join(paras[i:j]))
        found = [(e, t) for e, t in titles if _contains_title(hay, t)]
        hits = [e for e, _ in found]
        if len(hits) >= _TOC_TABLE_MIN and sum(len(t) for _, t in found) >= len(hay) * _TOC_TABLE_SHARE:
            top = min(hits, key=lambda e: e["level"])
            head = toc_headings._heading(top)
            if head not in paras:
                out.append(head)
            n += 1
        else:
            out.extend(paras[i:j])
        i = j
    return "\n\n".join(out), n


# ── 짝 없는 '[' — 낫표 「」『』 오인식 (2천년의강의·21세기지식경영 실측, 2026-11-07) ──────────────────
# 여는 낫표가 '[' 로 읽히고 닫는 낫표는 ','·띄어쓰기로 읽히거나 사라졌다: '[사기, 에'·'[사기열전 에서'·'[사기는'·
# '[이사열전'(줄 끝). 원본이 「」인지 『』인지 구별할 수 없어 「」 로 통일한다. 닫는 자리 찾는 차례:
# 쉼표+조사 → 줄 끝(짧을 때) → 다음 낱말이 조사뿐 → '열전'·'편' 낱말 뒤 → 조사 붙은 첫 낱말의 조사 앞.
# '과·와'는 제목 안('폭력과 성스러움'·'죄와 벌')에 흔해 조사로 보지 않는다(다음 '[' 바로 앞 끝만 예외).
_JOSA_TAIL = ("이라는", "에서는", "에서", "으로", "에게", "처럼", "까지", "부터", "에는", "에도", "라는",
              "은", "는", "을", "를", "에", "의", "이", "가", "로", "도", "만")
_JOSA_ONLY = set(_JOSA_TAIL) | {"과", "와"}
_TITLE_TAIL_RE = re.compile(r'^(.*?(?:열전|편))((?:' + "|".join(_JOSA_TAIL) + r')?)$')
_BRACKET_LINE_MAX = 20


def _close_title(seg: str, stopped_by_open: bool) -> str | None:
    """'[' 뒤 글(seg) 에 닫는 낫표를 넣은 결과 — 「 는 호출부가 붙인다. 못 정하면 None."""
    # 강조한 용어 뒤 원어 괄호 — '[변화 주도자(change leader)뿐이다' → 「변화 주도자」(change leader)(21세기지식경영)
    g = re.match(r'([가-힣][^()\[\],.]{0,14}?)\s?(\((?:[A-Za-z]|[一-鿿]).*)$', seg, re.S)
    if g:
        return f"{g.group(1)}」{g.group(2)}"
    # 쉼표는 짧은 제목 바로 뒤일 때만 닫는 낫표('[사기, 에'·'[서경, 제자백가') — 긴 구절의 쉼표는 진짜 쉼표다
    m = re.match(r'([^,]{1,12}), ?(.*)$', seg, re.S)
    if m and re.search(r'[가-힣]', m.group(1)) and len(m.group(1).split()) <= 3:
        rest = m.group(2)
        j = next((x for x in sorted(_JOSA_ONLY, key=len, reverse=True) if rest.startswith(x)), None)
        return f"{m.group(1)}」{rest}" if j and (len(rest) == len(j) or not re.match(r'[가-힣]', rest[len(j)])) \
            else f"{m.group(1)}」, {rest}"
    words = seg.split(" ")
    fixed = _close_by_words(words)
    if fixed is not None:
        return fixed
    # 낱말로 못 정하면 짧은 조각 끝 — 줄 끝 '[한 장 유 열 전도' · 다음 '[' 앞 '[적과 혹[죄와 벌'
    if len(seg.strip()) <= _BRACKET_LINE_MAX and len(words) <= (3 if stopped_by_open else 6) \
            and (stopped_by_open or "\n" not in seg):
        body = seg.rstrip()
        k = next((x for x in ("과", "와") + _JOSA_TAIL if body.endswith(x) and len(body) > len(x) + 1), "")
        return f"{body[:len(body) - len(k)]}」{k}{seg[len(body):]}"
    return None


def _close_by_words(words: list[str]) -> str | None:
    for i, w in enumerate(words[:4]):
        if i + 1 < len(words) and words[i + 1] in _JOSA_ONLY:     # '[사기열전 에서' — 띄어쓰기가 닫는 낫표 자리
            return " ".join(words[:i + 1]) + "」" + " ".join(words[i + 1:])
        t = _TITLE_TAIL_RE.match(w)
        if t:                                                       # '[장의 열전 첫머리' · '[사기 열전의'
            return " ".join(words[:i] + [t.group(1)]) + "」" + t.group(2) + (" " if i + 1 < len(words) else "") \
                + " ".join(words[i + 1:])
        if w in ("등", "등의", "등을", "같은"):                         # '[죄와 벌 등의'
            return " ".join(words[:i]) + "」 " + " ".join(words[i:]) if i else None
        k = next((x for x in _JOSA_TAIL if w.endswith(x) and len(w) - len(x) >= 2), "") \
            or next((x for x in ("과", "와") if w.endswith(x) and len(w) - 1 >= 4), "")   # '[태사공자서와 짝을'
        if k:                                                       # '[사기는' · '[허사 대사전을'
            return " ".join(words[:i] + [w[:-len(k)]]) + "」" + k + (" " if i + 1 < len(words) else "") \
                + " ".join(words[i + 1:])
    return None


def repair_open_brackets(text: str) -> tuple[str, int]:
    """줄 안에 닫는 ']' 가 없는 '[' 를 「…」 로 고친다. 낱말 첫머리가 아닌 '[' ('균열[을'·'모래언덕[ 되에'·
    '['에'')는 잡음으로 지운다. (본문, 고친 곳 수)"""
    out, n = [], 0
    for line in text.split("\n"):
        if line.startswith(("[p.", "|", "#")):
            out.append(line)
            continue
        # 겹따옴표 '""사기열전, 에' — 바깥 큰따옴표 + 『 를 '"' 로 읽은 것. 안쪽 인용은 작은따옴표로('"'사기열전'에')
        line = _DOUBLE_OPEN_RE.sub('"\x01', line)
        if "[" not in line and "\x01" not in line:
            out.append(line)
            continue
        parts = re.split(r'([\[\x01])', line)
        res = parts[0]
        for k in range(1, len(parts), 2):
            mark, seg = parts[k], parts[k + 1]
            opener, closer = ("「", "」") if mark == "[" else ("'", "'")
            if mark == "[" and seg.find("]") >= 0:      # 짝이 있으면 그대로
                res += "[" + seg
                continue
            if mark == "[" and (not seg[:1] or not re.match(r'[가-힣一-鿿]', seg) or seg.split(" ")[0] in _JOSA_ONLY
                                or re.match(r'(?:' + "|".join(_JOSA_TAIL) + r')(?:\s|$)', seg)):
                res += seg                       # 잡음 '[' — 지운다
                n += 1
                continue
            fixed = _close_title(seg, stopped_by_open=k + 2 < len(parts) and not seg.endswith(" "))
            if fixed is None:
                res += ("[" if mark == "[" else "") + seg
                continue
            res += opener + fixed.replace("」", closer, 1)
            n += 1
        out.append(res)
    return "\n".join(out), n


_DOUBLE_OPEN_RE = re.compile(r'""(?=[가-힣])')


# ── 조사로 시작하는 문단·조사 뒤 숫자·둥근 따옴표 앞 (2천년의강의 실측, 2026-11-07) ────────────────
# '…진정한‘큰생각'' ⏎⏎ '을 만나게 될 것이다' — 쪽·줄 경계에서 낱말 꼬리 조사가 다음 문단으로 갈렸다.
# 문단 첫머리에 올 수 없는 조사만('이·가·도·만·로' 는 '이 책'·'가지'·'도시' 처럼 낱말 머리일 수 있어 뺀다).
_LEAD_JOSA_RE = re.compile(r'^(?:에서|으로|에게|을|를|은|는|의|에|과|와)(?=[\s,.])')
_GLUED_DIGIT_RE = re.compile(r'(?<=[가-힣])(?<=(?:은|는|을|를|에|의|와|과|도|이|가))(?=\d)'
                             # '용어로서4차'·'따라서3가지'·'하지만2016년' — 앞이 한글일 때만('1만2000' 은 그대로)
                             r'|(?<=[가-힣][서만로])(?=\d)'
                             r'|(?<=\d년)(?=\d{1,2}월)|(?<=\d월)(?=\d{1,2}일)'    # '2017년5월' (4차산업혁명시대)
                             # 문장 끝 뒤 숫자+단위 '있다.1992년'·'있다.3장과' — 각주 번호('하였다.2 첫')는 단위가 없어 그대로
                             r'|(?<=[다요]\.)(?=\d+(?:년|월|일|장|부|개|명|%|가지|세기|차|위|배))')
# 조사 뒤 대문자 약어 '제조업과ICT'·'최고의IT' — 낱말 속 약어('삼성SDI'·'한국IT업계')는 앞 글자가 조사가 아니라 그대로
_GLUED_LATIN_RE = re.compile(r'(?<=[가-힣]{2})(?<=[과와을를은는이가에의도])(?=[A-Z]{2})'
                             # 낱말 첫머리 지시어+책 '이책은'·'그책을'(CEO의책꽃이 23곳) — '정책'·'…이책임' 은 앞이 한글이라 그대로
                             r'|(?<=[이그저])(?<![가-힣][이그저])(?=책)')
_GLUED_QUOTE_RE = re.compile(r'(?<=[가-힣])(?=[‘“])')
# 보조용언 '…기도 하다' — 용언 줄기+'기도' 뒤 '했다·한다…'(2천년의강의 '고백하기도했다'). 명사 '기도했다'(기도를
# 했다)는 앞에 줄기가 없어 걸리지 않는다
_AUX_GIDO_RE = re.compile(r'((?:하|되|있|없|받|주|보|많|크|작|좋|싫|하였|되었)기도)(했|한다|하다|하고|하며|해서|하는|할)')


def join_josa_paragraphs(text: str) -> tuple[str, int]:
    """조사로 시작하는 문단을 앞 문단 끝에 공백 없이 잇는다 — 앞 문단이 한글·따옴표로 끝날 때만. (본문, 이은 수)"""
    out: list[str] = []
    n = 0
    for p in text.split("\n\n"):
        prev = out[-1] if out else ""
        if (prev and _LEAD_JOSA_RE.match(p) and not prev.startswith(("#", "|", "[p.", ">"))
                and re.search(r"[가-힣]['’”\"]?$", prev)):
            out[-1] = prev + p
            n += 1
        else:
            out.append(p)
    return "\n\n".join(out), n


_QUOTE_OPEN = {"‘": "s", "“": "d"}
_QUOTE_CLOSE = {"’": "s", "”": "d"}
_QUOTE_BOTH = {"'": "s", '"': "d"}


def _space_open_quotes(line: str) -> tuple[str, int]:
    """한 줄 안 따옴표를 여닫는 차례로 읽어, 한글 바로 뒤에 붙은 여는 따옴표 앞을 띄운다
    ('‘권력'과'조직'' → '‘권력'과 '조직'', 2천년의강의). 곧은따옴표는 짝으로만 여닫이를 알 수 있어 종류별
    개수가 홀수인 줄은 건드리지 않는다. 여는 따옴표 뒤가 한글·숫자일 때만(영어 아포스트로피 제외)."""
    counts = {"s": 0, "d": 0}
    for ch in line:
        kind = _QUOTE_OPEN.get(ch) or _QUOTE_CLOSE.get(ch) or _QUOTE_BOTH.get(ch)
        if kind:
            counts[kind] += 1
    is_open = {"s": False, "d": False}
    out, n = [], 0
    for i, ch in enumerate(line):
        kind = _QUOTE_OPEN.get(ch) or _QUOTE_CLOSE.get(ch) or _QUOTE_BOTH.get(ch)
        if kind:
            opening = ch in _QUOTE_OPEN or (ch in _QUOTE_BOTH and not is_open[kind])
            if (opening and counts[kind] % 2 == 0 and i and re.match(r'[가-힣]', line[i - 1])
                    and i + 1 < len(line) and re.match(r'[가-힣0-9]', line[i + 1])):
                out.append(" ")
                n += 1
            is_open[kind] = opening if ch not in _QUOTE_CLOSE else False
        out.append(ch)
    return "".join(out), n


def space_glued_marks(text: str) -> tuple[str, int]:
    """조사 뒤에 붙은 숫자('당신은2000년')와 한글 뒤에 붙은 여는 따옴표('진정한‘큰생각''·''권력'과'조직'')를
    띄운다. (본문, 넣은 띄어쓰기 수)"""
    text, a = _GLUED_DIGIT_RE.subn(" ", text)
    text, a2 = _AUX_GIDO_RE.subn(r"\1 \2", text)   # '고백하기도했다' → '고백하기도 했다'
    text, a3 = _GLUED_LATIN_RE.subn(" ", text)
    a += a2 + a3
    b = 0
    lines = text.split("\n")
    for k, line in enumerate(lines):
        if not line.startswith(("#", "|")) and any(q in line for q in "‘“'\""):
            lines[k], c = _space_open_quotes(line)
            # 짝 수와 상관없이 확실한 여는 따옴표 — 둥근 여는 따옴표, 닫는 따옴표+조사 바로 뒤 따옴표('권력'과'조직')
            lines[k], c2 = _SURE_OPEN_RE.subn(" ", lines[k])
            b += c + c2
    return "\n".join(lines), a + b


# 닫는 따옴표와 다음 따옴표 사이가 조사뿐일 때만 — ''조직'에' 의 '조직' 은 조사가 아니다
_J1 = "[과와를을은는이가의에도로만]"
_J2 = "(?:에서|으로|이나|에게|까지|부터|처럼|보다|이라|라는)"
_SURE_OPEN_RE = re.compile(r'(?<=[가-힣])(?=[‘“][가-힣0-9])|(?<=[가-힣])(?=["“][\'‘][가-힣])'
                           rf'|(?<=[\'’]{_J1})(?=[\'‘][가-힣])|(?<=[\'’]{_J2})(?=[\'‘][가-힣])'
                           rf'|(?<=["”]{_J1})(?=["“][가-힣])|(?<=["”]{_J2})(?=["“][가-힣])')


_TOC_DENSE_MIN = 3        # 차례 제목이 이만큼 들고
_TOC_DENSE_COVER = 0.7    # 문단 글자의 이만큼을 차례 제목이 채우면 장 여는 쪽의 작은 차례


def drop_toc_dense(text: str, entries: list[dict]) -> tuple[str, int]:
    """장 여는 쪽에 늘어놓은 그 장의 항목 이름('고객 니즈의 프레임 워크 고객 주도를 넘어서 불연속성과 …' /
    '구매 가능성' / …)을 지운다(2x2Matrix 6~8장). 맨 앞 차례 목록과 같은 글이라 무손실. 그대로 두면 제목
    맞추기가 그 속의 짧은 줄을 먼저 잡아 뒤 항목들이 설 자리가 없었다. 차례 제목이 문단 글자의 70% 이상을
    채우는 문단과, 그런 문단 바로 옆의 차례 제목과 같은 짧은 줄만. 맨 앞 차례(문서 첫 제목 줄 앞)는 건드리지 않는다.
    (본문, 지운 문단 수)"""
    norm = toc_headings._norm_hangul
    # 같은 글자는 한 번만 — '제5장에 들어가면서'·'제6장에 들어가면서' 는 번호를 떼면 같다(21세기지식경영)
    titles = sorted({t for t in (norm(_strip_label(e["title"])) for e in entries) if len(t) >= 4})
    if not titles:
        return text, 0
    paras = text.split("\n\n")

    def dense(p: str) -> bool:
        n = norm(p)
        if len(n) < 12 or p.startswith("#"):
            return False
        hits = [t for t in titles if t in n]
        return (len(hits) >= _TOC_DENSE_MIN and sum(map(len, hits)) >= len(n) * _TOC_DENSE_COVER
                and max(map(len, hits)) < len(n) * 0.6)    # 제목 하나가 문단 대부분이면 제목 줄이다

    marks = [dense(p) for p in paras]
    for i, p in enumerate(paras):     # 목록 사이에 따로 선 항목 이름 한 줄
        if not marks[i] and len(p) <= 40 and norm(p) in titles and (
                any(marks[max(0, i - 2):i]) and any(marks[i + 1:i + 3])):
            marks[i] = True
    out = [p for p, m in zip(paras, marks) if not m]
    return "\n\n".join(out), len(paras) - len(out)


_COVER_AD_START_RE = re.compile(r'ISBN|값\s?[\d,]+\s?원')
_COVER_AD_LINE_RE = re.compile(r'ISBN|값\s?[\d,]+\s?원|지음|옮김|그림\s?:|시리즈')
_COVER_AD_PARA_MAX = 400


def drop_cover_ad(text: str) -> tuple[str, int]:
    """첫 제목 줄 앞에 남은 뒤표지 광고 — ISBN·값이 든 문단부터, 뒤이어 오는 짧은 시리즈 광고·그림 크레딧
    문단('2. 1분 매니저 케네스 블랜차드 지음…'·'본문그림:손문상')까지(1분의사결정). 첫 제목 줄 뒤(본문)는 보지
    않고, 문단이 길면(본문 글) 멈춘다. (본문, 지운 문단 수)"""
    paras = text.split("\n\n")
    first = next((i for i, p in enumerate(paras) if _MD_HEADING_RE.match(p)), len(paras))
    a = next((i for i in range(first) if _COVER_AD_START_RE.search(paras[i])), None)
    if a is None or len(paras[a]) > _COVER_AD_PARA_MAX:
        return text, 0
    b = a + 1
    while b < first and len(paras[b]) <= _COVER_AD_PARA_MAX and _COVER_AD_LINE_RE.search(paras[b]) \
            or (b < first and len(paras[b]) <= 40 and b + 1 < first and _COVER_AD_LINE_RE.search(paras[b + 1])):
        b += 1
    del paras[a:b]
    return "\n\n".join(paras), b - a


_ECHO_LINE_MAX = 60
_PAGE_MARK_PARA_RE = re.compile(r'\[p\.\d+\]')
_DECOR_PARA_RE = re.compile(r'[oO0°•○◦∘·*]{1,3}')
# 'CHAP' 이 그림에 가려 남은 'TER3', 글자 사이가 띄워진 'C H A P T E R 4'(FastSecond)
_LONE_LABEL_RE = re.compile(r'^(?:제\s*)?(\d{1,2})\s*[부장편강]$'
                            r'|^(?:(?:C\s?H\s?A\s?P\s?)?T\s?E\s?R|P\s?A\s?R\s?T|Chapter|Part)\s*(\d{1,2})$')


def drop_heading_echo(text: str) -> tuple[str, int]:
    """제목 바로 아래 문단이 제목 끝 한두 낱말로만 되어 있으면 지운다 — 장 표지의 큰 글자를 OCR 이 따로 읽은
    것(4차산업혁명시대: '## 1장 혁신 전쟁의 서막' 아래 '서막'). 짧은 줄이 그 끝말로 시작하면(용어 상자 제목에
    붙음: '산업 트렌드 오픈 이노베이션(…)') 끝말만 떼어낸다. 문장(…다.)이거나 긴 문단은 본문이라 두다.
    (본문, 고친 수)"""
    paras = text.split("\n\n")
    fixed = 0
    for i in range(1, len(paras)):      # 제목 바로 위에 같은 번호의 표지 번호만 남은 줄('1부'·'CHAPTER9')
        lab = _LONE_LABEL_RE.match(paras[i - 1].strip())
        num = re.match(r'#+\s*(?:(?:제\s*)?(\d+)\s*[부장편강]|(?:CHAPTER|PART|Chapter|Part)\s*(\d+))', paras[i])
        if lab and num and int(lab.group(1) or lab.group(2)) == int(num.group(1) or num.group(2)):
            paras[i - 1] = None
            fixed += 1
    for i in range(len(paras) - 1):
        m = paras[i] is not None and _MD_HEADING_RE.match(paras[i])
        j = i + 1      # 쪽 표지 '[p.240]' 은 건너뛰고 본다(First,BreakAllTheRules 6장)
        while j < len(paras) - 1 and paras[j] and _PAGE_MARK_PARA_RE.fullmatch(paras[j].strip()):
            j += 1
        nxt = (paras[j] or "").strip()
        if not m or not nxt or _MD_HEADING_RE.match(nxt):
            continue
        if _DECOR_PARA_RE.fullmatch(nxt):     # 장 표지 장식 '•○•' 이 'o'·'o°' 로 읽힌 줄(HalfTimeShock)
            paras[j] = None
            fixed += 1
            continue
        whole = _strip_label(paras[i].lstrip("#").strip())
        # 한글만 남겨 견준다 — 영어 제목은 남는 글이 없어 아무 문단이나 같다고 보게 된다(JesusCEO 첫 문단 소실)
        norm_whole = toc_headings._norm_hangul(whole)
        if len(norm_whole) >= 4 and toc_headings._norm_hangul(nxt) == norm_whole:
            paras[j] = None      # 번호표를 뗀 제목 전체가 다음 줄에 다시('2부' 줄에 선 제목 아래 '내부 조직 간 …')
            fixed += 1
            continue
        words = paras[i].lstrip("#").split()
        # 같은 줄이면 몇 낱말이든 — 두 줄 제목의 둘째 줄('나이키가 고객을 뛰게 했다', Management by Map). 앞부분 떼기는 1~2낱말만
        for k in range(len(words) - 1, 0, -1):
            if len(words) <= k:
                continue
            tail = " ".join(words[-k:])
            if len(tail) < 2:
                continue
            if nxt == tail:
                paras[j] = None
            elif k < 3 and nxt.startswith(tail + " ") and len(nxt) <= _ECHO_LINE_MAX and "다." not in nxt \
                    and not nxt.endswith(("다", ".", "?", "!")):
                paras[j] = nxt[len(tail):].strip()
            else:
                continue
            fixed += 1
            break
    return "\n\n".join(p for p in paras if p is not None), fixed


_SUB_HEAD_RE = re.compile(r'^#+ .+? _ ([가-힣][^A-Za-z]*?)\s*([A-Za-z].*)?$')
# 부제 첫 글자가 영문자로 읽힌 것('H매한 표현')까지
_SUB_ECHO_RE = re.compile(r'^[—−–-]?\s*([A-Za-z]?[가-힣]+(?:\s[가-힣]+)*)\s*([A-Za-z][A-Za-z\' ,.:-]*[A-Za-z])\s*')
_SUB_ECHO_MIN = 0.7      # 부제 글자 닮음(OCR 오자 '통한입' 정도)
_SUB_LATIN_MIN = 0.5     # 깨진 영어 원제 줄 닮음('The gencti falacy'·'ludkd wcrcks' 0.55) — 줄 글자 중 원제와 맞는 몫도
_SUB_LATIN_LETTERS = 5   # 이보다 짧은 영어 줄('Yes')은 견주지 않는다 — 'Acent'(Accent) 는 본다
_SUB_LATIN_LINE_MAX = 40 # 원제 줄 길이 — 긴 영어 문장은 본문
_SUB_ECHO_LINES = 3      # 제목 아래 이만큼의 줄(쪽 표지 빼고)만 본다 — 잡음 줄('글 꼴 둠 없는') 하나 뒤까지


def _latin_echo(line: str, lat: str) -> bool:
    """한글 없는 줄이 제목의 영어 원제를 다시(깨져) 읽은 것인가 — 전체 닮음이나, 줄 글자가 원제에 든 몫('ad Lpidem')."""
    s = re.sub(r'[^a-z]', '', line.lower())
    if not lat or re.search(r'[가-힣]', line) or len(s) < _SUB_LATIN_LETTERS or len(line) > _SUB_LATIN_LINE_MAX:
        return False
    m = SequenceMatcher(None, s, lat, autojunk=False)
    return max(m.ratio(), sum(b.size for b in m.get_matching_blocks()) / len(s)) >= _SUB_LATIN_MIN


def drop_subtitle_echo(text: str) -> tuple[str, int]:
    """'제목 _ 부제 English' 꼴 제목(책정보 차례) 바로 아래, 장 표지의 부제·원제를 OCR 이 다시 읽은 것을 지운다
    (HowToWinEveryArgument 81곳): 원제를 깨진 글자로 읽은 영어만의 줄('The gencti falacy'), 본문 첫머리에 붙은
    '—부제 원제'('−발생학적 오류 The genetic fallacy 발생학적 오류는 …' → '발생학적 오류는 …'). 부제와 다른 글로
    시작하는 줄은 본문이라 둔다. (본문, 고친 수)"""
    paras = text.split("\n\n")
    fixed = 0
    for i, p in enumerate(paras):
        m = p is not None and _SUB_HEAD_RE.match(p)
        if not m:
            continue
        sub, lat = toc_headings._norm_hangul(m.group(1)), re.sub(r'[^a-z]', '', (m.group(2) or "").lower())
        seen, j = 0, i + 1
        while seen < _SUB_ECHO_LINES and j < len(paras) and paras[j] is not None \
                and not _MD_HEADING_RE.match(paras[j]):
            lines = paras[j].split("\n")
            for k, s in enumerate(lines):
                if _PAGE_MARK_PARA_RE.fullmatch(s.strip()) or seen >= _SUB_ECHO_LINES:
                    continue
                seen += 1
                e = _SUB_ECHO_RE.match(s)
                if _latin_echo(s, lat):
                    lines[k] = None
                elif e and SequenceMatcher(None, toc_headings._norm_hangul(e.group(1)), sub,
                                           autojunk=False).ratio() >= _SUB_ECHO_MIN \
                        and _latin_echo(e.group(2), lat):   # 영어도 원제와 닮아야 — 글쓴이 줄('피터 센게Peter Senge')은 둔다
                    lines[k] = s[e.end():] or None
                else:
                    continue
                fixed += 1
            kept = [s for s in lines if s is not None]
            paras[j] = "\n".join(kept) if kept else None
            j += 1
    return "\n\n".join(p for p in paras if p is not None), fixed


_LIST_LINE_RE = re.compile(r'^(?:\d{1,2}\s?[.,)]?\s?\S|\d\)|[-•·])')
_LIST_LINE_MAX = 160


def relocate_list_headings(text: str, entries: list[dict]) -> tuple[str, int]:
    """장 여는 쪽의 절 목록('1)거시 환경 분석' / '2)산업분석' …) 속 줄에 선 절 제목을, 다음 제목 줄 전에 같은
    글자로 다시 나오는 진짜 절 자리로 옮긴다(12년차기획맨: '1. 외부 환경 분석' 이 목록에 서서 본문 자리를 놓침,
    '5. 전략 기획서 보고' 가 장 첫머리 목록에 서서 1~4절 자리가 없어짐). 목록 속이라고 보는 것: 바로 앞이나
    뒤 문단이 짧은 번호·글머리 줄이거나 다른 제목 줄. 옮긴 자리의 글은 그 줄 글자 그대로 본문으로 돌려놓는다.
    (본문, 옮긴 수)"""
    norm = toc_headings._norm_hangul
    paras = text.split("\n\n")
    moved = 0
    last_page = max((e["page"] for e in entries if e.get("page")), default=0)

    def listy(k: int) -> bool:
        if not 0 <= k < len(paras):
            return False
        p = paras[k].strip()
        return (bool(_MD_HEADING_RE.match(p)) or (len(p) <= _LIST_LINE_MAX and bool(_LIST_LINE_RE.match(p)))
                or len(re.findall(r'\d\)', p)) >= 3       # 한 문단으로 붙어 읽힌 목록('1)거시… 2)산업… 3)…')
                or (len(p) <= 40 and not _SENT_END.search(p)))   # 장 제목 조각('현실적으로는')

    # 뒤 항목부터 — 앞 항목의 진짜 자리가 뒤 항목의 잘못 선 제목 줄 뒤에 있으면 뒤 것이 먼저 비켜야 한다(12년차기획맨 8장)
    for ei in reversed(range(len(entries))):
        e = entries[ei]
        head = toc_headings._heading(e)
        p = _entry_positions(paras, entries)[ei]
        if p is None:
            continue
        if not (listy(p - 1) or listy(p + 1)):   # 같은 제목 줄이 다음 제목 전에 또 나와야 옮기므로 한쪽만 봐도 된다
            continue
        q = next((k for k in range(p + 1, len(paras)) if _MD_HEADING_RE.match(paras[k])), len(paras))
        t = norm(e["title"])
        k = next((k for k in range(p + 2, q) if len(paras[k]) <= len(e["title"]) * 1.4 + 5 and norm(paras[k])
                  and SequenceMatcher(None, norm(paras[k]), t, autojunk=False).ratio() >= toc_headings._MATCH_MIN),
                 None)
        if k is None:
            continue
        # 차례 쪽번호가 있으면 옮길 자리가 지금 자리보다 그 쪽에 가까울 때만 — 책 끝 '요약 및 정리' 가 장 제목을
        # 늘어놓은 곳으로 21장이 모두 옮겨 갔다(SalesSuperStar, 쪽과 먼 제목 12)
        # (이 단계엔 쪽 표지가 아직 없어 문서 속 자리 비율과 차례 쪽 비율로 견준다 — 점검의 '쪽과 먼 제목' 과 같은 셈)
        if e.get("page") and last_page:
            want = e["page"] / last_page
            if abs(k / len(paras) - want) > abs(p / len(paras) - want):
                continue
        paras[p], paras[k] = e["title"], head
        moved += 1
    return "\n\n".join(paras), moved


_SPLIT_TITLE_PLAIN_MIN = 0.85   # 번호 표지가 없으면 더 엄하게


def _place_unnumbered_titles(paras: list[str], entries: list[dict]) -> int:
    """번호 표지를 OCR 이 못 읽은 장(삶의지혜 01장 — '…시안 진정 가치 있는 것은 '황금알'자체가 아니라.' /
    ''황금알'을 낳을 수 있는 닭이다 자신을 바꿔라') — 앞뒤 차례 제목 줄 사이에서 제목 첫 글자로 시작하는 낱말
    묶음 중 처음으로 _SPLIT_TITLE_PLAIN_MIN 이상 같은 곳. 본문이 제목을 되풀이하는 곳보다 제목 자리가 앞선다."""
    placed = 0
    for ei, e in enumerate(entries):
        head = toc_headings._heading(e)
        target = toc_headings._norm_hangul(re.sub(r'^0*\d{1,3}\.\s*', '', e["title"]))
        if len(target) < 8:
            continue
        lo, hi, found = _bounds(paras, entries, ei)
        if found:
            continue
        words = [(p, w) for p in range(lo, hi) if not _MD_HEADING_RE.match(paras[p]) for w in paras[p].split()]
        norms = [toc_headings._norm_hangul(w) for _, w in words]
        reach = len(target) // 2 + _SPLIT_TITLE_SLACK
        for i in range(len(words)):
            if not norms[i] or not target.startswith(norms[i][:2]):
                continue
            best, bj = 0.0, None
            for j in range(i, min(len(words), i + reach)):
                if words[j][0] != words[i][0] and _MD_HEADING_RE.match(paras[words[j][0]]):
                    break
                cand = "".join(norms[i:j + 1])
                if len(cand) > len(target) * 1.3:
                    break
                r = SequenceMatcher(None, cand, target, autojunk=False).ratio()
                if r > best:
                    best, bj = r, j
            if best < _SPLIT_TITLE_PLAIN_MIN:
                continue
            (pi, _), (pj, _) = words[i], words[bj]
            before = " ".join(w for x, (p, w) in enumerate(words) if p == pi and x < i)
            after = " ".join(w for x, (p, w) in enumerate(words) if p == pj and x > bj)
            paras[pi:pj + 1] = [s for s in (before, head, after) if s]
            placed += 1
            break
    return placed


_LATIN_TAIL_RE = re.compile(r'[A-Za-z][^가-힣]*$')


def _latin_tail(line: str, title: str) -> str:
    """한글 제목 줄 끝의 영어 부분 — 차례 제목에 없는 것만('시나리오 경영Scenarios' → 'Scenarios')."""
    if not re.search(r'[가-힣]', title):
        return ""
    m = _LATIN_TAIL_RE.search(line.strip())
    return m.group().strip() if m and m.group().strip() not in title else ""


def _title_line(lines: list[str], lo: int, hi: int, title: str, exact: bool = False) -> int | None:
    """lines[lo:hi] 에서 차례 제목과 같은 따로 선 짧은 줄(제목 줄·표 줄 제외). 번호표를 뗀 제목으로도 본다.
    차례 OCR 이 두 항목·저자를 한 줄로 붙여 읽은 항목('다이렉트경영 마이클델', CEO의책꽃이 실측)은 그 앞부분과
    같은 줄도 — 6자 이상·항목의 40% 이상일 때만."""
    han = bool(re.search(r'[가-힣]', title))
    norm = toc_headings._norm_hangul if han else toc_headings._norm
    # '주제 _ 부제' 차례 제목은 본문 제목 줄에 주제만 있기도 하다('2장 2×2의 변증법 _ 문제 상황을…' → '2×2의 변증법')
    main = title.split(" _ ")[0] if " _ " in title else ""
    # 장 표지 '큰 제목 _ 작은 제목' 은 본문에 작은 제목만 남기도 한다('… _ 역사적 고찰' → '역사적 고찰', Alianment정렬)
    # 마지막(느슨한) 패스에서만 — 앞 패스에서 쓰면 큰 제목 줄을 찾기 전에 작은 제목 줄을 잡았다(2천년의강의)
    sub = title.split(" _ ", 1)[1] if " _ " in title and not exact else ""
    targets = [t for t in dict.fromkeys([norm(title), norm(_strip_label(title)),
                                         norm(_strip_label(main)), norm(sub)]) if len(t) >= 4]
    # 짧은 주제('추천사 _ …')·번호표를 뗀 짧은 제목('제1강 관찰력' → '관찰력', 2천년의강의)
    short_main = norm(_strip_label(main)) if main else norm(_strip_label(title))
    for i in range(lo, hi):
        l = lines[i].strip()
        # 한글 제목은 줄 끝 영어 원제를 빼고 길이를 잰다('시나리오 경영Scenarios', 2x2Matrix)
        size = len(_LATIN_TAIL_RE.sub("", l)) if han and re.search(r'[가-힣]', l) else len(l)
        if not l or _MD_HEADING_RE.match(l) or l.startswith("|") or size > len(title) * 1.4 + 5:
            continue
        if _FIGURE_CAPTION_RE.match(l):     # 그림 설명('그림6.37 영향력-불확실성 매트릭스', 2x2Matrix)은 제목 아님
            continue
        # '서문'·'죄' 같은 짧은 제목 — 같은 글자 줄만(삶의지혜·1001가지기독교명언)
        if (len(norm(title)) == 1 and l == title.strip()) or (2 <= len(norm(title)) < 4 and norm(l) == norm(title)) \
                or (2 <= len(short_main) < 4 and norm(l) == short_main):    # '추천사 _ 제임스 길모어…' → '추천사'
            return i
        n = norm(l)
        # 장 표지 제목이 짧은 두 줄로 갈림('세번째 열쇠:' / '장점을 개발하라', First,BreakAllTheRules) — 이은 글자가
        # 차례 제목과 같을 때만. 남은 둘째 줄은 drop_heading_echo 가 제목 끝말 반복으로 지운다
        # 두 줄 모두 한글 2자 이상(한글 제목) — '#' 잡음 줄(DrivingSalesmanPowerSkill)·'2x2 Matrix'(2x2Matrix)는 아님
        enough = (lambda s: len(re.findall(r'[가-힣]', s)) >= 2) if han else (lambda s: len(norm(s)) >= 2)
        if enough(l) and len(l) <= _SPLIT_LINE_MAX:
            j = next((k for k in range(i + 1, min(i + 3, hi)) if lines[k].strip()), None)
            if j is not None and len(lines[j].strip()) <= _SPLIT_LINE_MAX and not _MD_HEADING_RE.match(lines[j]) \
                    and enough(lines[j]) and n + norm(lines[j]) in targets and n not in targets:
                return i
        if exact:   # 띄어쓰기·영어 원제만 다른 같은 글자 줄만 — 비슷한 줄(색인 줄)이 진짜 제목보다 먼저 잡혔다(2x2Matrix)
            if n in targets:
                return i
            continue
        if any(SequenceMatcher(None, n, t).ratio() >= toc_headings._MATCH_MIN
               or (len(n) >= _FRONT_MIN and len(n) >= len(t) * _FRONT_RATIO
                   and SequenceMatcher(None, n, t[:len(n)]).ratio() >= toc_headings._MATCH_MIN) for t in targets):
            return i
    # 마지막(느슨한) 패스에서 제목 글자 줄이 없으면 장 번호표만 따로 선 줄('제1장') — 장 표지에 큰 제목 없이 번호만
    # 있는 책(FocalPoint: 제목은 차례에만). 번호표가 정확히 같은 줄만
    lab = None if exact else _EXTRA_LABEL_RE.match(title)
    if lab and lab.group().strip() != title.strip():
        want = re.sub(r'\s', '', lab.group())
        return next((i for i in range(lo, hi) if re.sub(r'\s', '', lines[i]) == want), None)
    return None


_FRONT_MIN = 6
_FRONT_RATIO = 0.4
_SPLIT_LINE_MAX = 30     # 두 줄로 갈린 제목의 한 줄 최대 길이
_CHAPTER_LINE_MAX = 40
_CHAPTER_MIN = 3
_BOOK_PART_RE = re.compile(r'^book\s*\d', re.I)


def infer_chapters(text: str) -> tuple[str, list[dict]]:
    """차례가 없는 책 — 본문의 장 번호표 줄('CHAPTER 1'·'제1장 집중의 힘'·'1부')을 '#' 제목으로, 차례 항목도 만든다.
    (Alignment·FocalPoint·1Q84·IDEAHACKS49 실측 — 차례를 못 찾아 수십만 자가 한 덩어리였다)
    오탐 방지: 40자 이하 줄만, 번호가 1·2·3… 차례대로 이어질 때만(같은 번호 반복·앞선 차례 잔재는 건너뜀 —
    OCR 이 한 장을 못 읽은 것만큼 하나 건너뛰기는 허용, IDEAHACKS49 2장), 부 뒤에선 장 번호가 1부터 다시 시작해도 된다.
    쪽번호로 끝나는 차례 잔재('제3장 삶을 단순화하라·56')·문장('제1장을 시작하며' — toc_headings 장 꼴이 거른다) 제외,
    장·부가 모두 3개 미만이면 하지 않는다. 'BOOK 1 APRIL-JUNE' 은 부(1Q84).
    다음 줄을 제목으로 합치지 않는다 — 인용문일 때가 있다(FocalPoint). 글자는 그대로."""
    lines = text.split("\n")
    picks: list[tuple[int, int]] = []          # (줄 번호, 단계 꼴 0=부 1=장)
    expect = {0: 1, 1: 1}
    for i, raw in enumerate(lines):
        l = raw.strip()
        if not l or len(l) > _CHAPTER_LINE_MAX or _MD_HEADING_RE.match(l) or l.startswith("|"):
            continue
        kind = 0 if _BOOK_PART_RE.match(l) else toc_headings._level_kind(l)
        m = re.search(r'\d+', l)
        if kind not in (0, 1) or not m:
            continue
        rest = toc_headings._LABEL_RE.sub("", l)
        if rest and re.search(r'\d\s*$', rest):
            continue
        no = int(m.group())
        # 건너뛰기는 첫 장을 찾은 뒤에만 — 부 뒤 주석 구간의 '2장 …' 소제목이 첫 장으로 잡혔다(Alignment)
        # 장 번호가 1로 돌아오면 — 부 뒤, 또는 장을 3개 넘게 찾은 뒤(권 표시 줄 없이 권마다 1장부터: 성경)
        if (no == expect[kind] or (expect[kind] > 1 and no == expect[kind] + 1)
                or (kind == 1 and no == 1 and picks and (picks[-1][1] == 0 or expect[1] > _CHAPTER_MIN + 1))):
            picks.append((i, kind))
            expect[kind] = no + 1
    if sum(k == 1 for _, k in picks) < _CHAPTER_MIN and sum(k == 0 for _, k in picks) < _CHAPTER_MIN:
        return text, []
    has_part = any(k == 0 for _, k in picks)
    entries = []
    for i, kind in picks:
        e = {"level": 1 if kind == 0 or not has_part else 2, "title": lines[i].strip(), "page": None}
        lines[i] = toc_headings._heading(e)
        entries.append(e)
    return "\n".join(lines), entries


def keep_headings(fn, text: str) -> str:
    """text 의 장·절 제목 줄('# 제목')은 그대로 두고, 그 사이 본문 덩어리에만 fn 을 적용한다.
    쪽 교정·잡음 제거(postprocessor·noise_rules.clean_page)는 '#' 을 기호로 보고 지운다."""
    if "#" not in text:
        return fn(text)
    blocks, body = [], []

    def flush():
        if any(b.strip() for b in body):
            fixed = fn("\n".join(body)).strip("\n")
            if fixed.strip():
                blocks.append(fixed)
        body.clear()

    for line in text.split("\n"):
        if _MD_HEADING_RE.match(line):
            flush()
            blocks.append(line)
        else:
            body.append(line)
    flush()
    return "\n\n".join(blocks)


def rejoin_split_sentences(text: str) -> tuple[str, int]:
    """문장이 끝나지 않은 긴 줄과 이어지는 한글 줄을 한 문장으로 잇는다. (텍스트, 이은 수)."""
    lines = text.split("\n")
    out: list[str] = []
    joined = 0
    i = 0
    while i < len(lines):
        cur = lines[i]
        while True:
            j = i + 1
            if j < len(lines) and not lines[j].strip():
                j += 1                      # 빈 줄 하나(문단 끊김)까지는 건너 본다
            if j >= len(lines):
                break
            s, nxt = cur.rstrip(), lines[j].strip()
            if _MD_HEADING_RE.match(s) or _MD_HEADING_RE.match(nxt):   # 장·절 제목 줄(toc_headings)
                break
            bound = bool(s) and re.search(r'[가-힣]$', s) is not None and _starts_bound(nxt)
            if bound:   # 끊긴 어미·조사 — 한 단어가 잘린 것이니 공백 없이 잇는다
                cur = s + nxt
                joined += 1
                i = j
                continue
            if (len(s) < _SPLIT_MIN_LEN or _SENT_END.search(s) or _TOC_LIKE_RE.search(s)
                    or not re.match(r'[가-힣]', nxt)
                    or (len(nxt) < 30 and not _SENT_END.search(nxt))):   # 다음 줄이 제목 꼴
                break
            cur = s + _glue(s.split()[-1], nxt.split()[0]) + nxt
            joined += 1
            i = j
        out.append(cur)
        i += 1
    return "\n".join(out), joined


@dataclass
class _ChunkResult:
    cleaned: str = ""
    warnings: list[str] = field(default_factory=list)
    fallback_api: int = 0
    fallback_guard: int = 0


def _clean_chunk(chunk: list[str], i: int, total: int, *, front_chunk: bool,
                 want_colophon: bool, local: bool = False) -> _ChunkResult:
    """구간 하나를 정제한다. 공유 상태를 건드리지 않아 여러 구간을 동시에 돌릴 수 있다."""
    res = _ChunkResult()
    want_front = front_chunk
    if front_chunk and not local:   # 문단 분류는 Gemini 호출 — 비용 우선에선 쓰지 않는다
        # 문단 분류로 먼저 걷어낸다 — 실패하면 기존 규칙 7(정제 중 삭제)로 돌아간다
        cls = drop_front_paragraphs(chunk)
        if cls is not None:
            chunk, removed = cls
            want_front = False      # 삭제는 끝남 → 정제는 일반 구간처럼(무결성 가드 적용)
            if removed:
                res.warnings.append(
                    f"구간 {i + 1}/{total}: 앞부속 문단 {len(removed)}개 삭제"
                    f" ('{' '.join(removed[0].split())[:30]}…' 등)")
    if not chunk:                   # 구간 전체가 앞부속
        return res
    joined = ("\n" + PAGEBREAK + "\n").join(chunk)
    if local and config.get_local_cleanup_engine() == "rules":
        res.cleaned = _rules_clean(joined)   # 비용 우선 기본 — LLM 없이 정제규칙만
        return res
    run_text = _ollama.run_text if local else _gemini.run_text   # 비용 우선(ollama 설정) → 로컬 exaone
    prompt = _build_prompt(joined, drop_front=want_front, drop_colophon=want_colophon)
    # 무결성 가드: 빈/과소 응답(안전필터 차단·조기 종료 — 예외 없이 옴)은
    # 수용하지 않고 1회 재시도, 그래도 미달이면 원문 보존 강등.
    cleaned = None
    api_failed = False
    for _attempt in range(2):
        try:
            cand = _strip_output(run_text(prompt))
        except Exception:
            api_failed = True
            break
        if _output_ok(cand, joined, want_front, want_colophon):
            cleaned = cand
            break
    degraded = cleaned is None
    if degraded:
        cleaned = _fallback_clean(joined)   # 강등 — 내용 소실 없는 최소 정제
        if api_failed:
            res.fallback_api = 1
            res.warnings.append(
                f"구간 {i + 1}/{total}: API 오류 — 규칙 기반 최소 정제로 강등(원문 보존)")
        else:
            res.fallback_guard = 1
            res.warnings.append(
                f"구간 {i + 1}/{total}: LLM 빈/과소 응답 감지 — 원문 보존 강등"
                " (LLM 교정 미적용, 내용 무손실)")
    # 앞부속 제거 청크는 가드 면제(전부 삭제가 정당)라, 남겼는데 크게 줄었으면 경고만 남긴다
    # — 본문 요약·축약을 사용자가 알아챌 수 있게(원문 사본은 storage 에 보존). QA ISSUE-I
    if want_front and cleaned:
        src_len = len(re.sub(r"\s+", "", joined.replace(PAGEBREAK, "")))
        ratio = len(re.sub(r"\s+", "", cleaned)) / src_len if src_len else 1.0
        if ratio < _GUARD_RATIO_COLOPHON:
            res.warnings.append(
                f"구간 {i + 1}/{total}: 앞부속 제거 후 분량 {ratio:.0%} — 본문이 줄었는지 확인"
                " (정제 전 원문 사본 있음)")
    # 부속물 삭제 구간은 빠지는 게 정상이라 보지 않는다
    dropped = None if degraded or want_front or want_colophon else _dropped_run(cleaned, joined)
    if dropped:
        preview = " ".join(dropped.split())[:40]
        res.warnings.append(
            f"구간 {i + 1}/{total}: 원문 문장 {_DROP_RUN}개 이상이 연달아 빠짐"
            f" ('{preview}…') — OCR 오인식 쪽이거나 LLM 누락,"
            " 정제 전 원문 사본과 대조 필요")
    res.cleaned = cleaned
    return res


_HEADING_MAX = 25                     # 이보다 긴 쪽 첫 줄은 앞 쪽에서 이어지는 본문 줄로 본다
_HEADING_TAIL_STRIP = '"\'”’)]」』》'


def _is_heading_line(rest: str) -> bool:
    """쪽 첫 줄이 제목·장 번호인가 — 짧고, 문장 끝(. ? ! … , 다 요 까)으로 끝나지 않고, 아래에 줄이 더 있다.
    짧은 이음 줄('고 하는 데에는 다른이유가 있다.', '져나갈궁리를 한다')은 문장 끝으로 끝나 앞 쪽에 잇는다."""
    first, _, more = rest.partition("\n")
    tail = first.strip().rstrip(_HEADING_TAIL_STRIP)
    return (0 < len(first.strip()) <= _HEADING_MAX and bool(tail) and tail[-1] not in ".!?。…,다요까"
            and bool(more.strip()))


def post_process_pagebreaks(text: str) -> str:
    """
    텍스트 내의 ⟨PAGEBREAK⟩ 경계를 분석하여 잘려나간 어절을 합치고 센티넬을 소거합니다.
    """
    if not text:
        return ""

    from core.postprocessor import is_item_start, item_letters, _FIGURE_CAPTION_RE
    letters = item_letters(text.replace(PAGEBREAK, "\n").splitlines())

    def _repl(match):
        before, ws_before, ws_after, after = match.groups()
        line_start = match.string.rfind("\n", 0, match.start()) + 1
        # 앞 쪽 끝 줄이나 다음 쪽 첫 줄이 장·절 제목(toc_headings)이면 잇지 않는다
        if re.fullmatch(r'#{1,6}', after) or _MD_HEADING_RE.match(match.string[line_start:match.end(1)]):
            return before + "\n\n" + after
        # 다음 쪽이 항목(2. ② 나. 제2조 …)으로 시작하면 앞 쪽 끝에 붙이지 않는다
        if "\n" in ws_before + ws_after and is_item_start(match.string[match.start(4):], letters):
            return before + "\n" + after
        # 다음 쪽이 그림 제목으로 시작하면 문단 경계('…커다 그림 3.8 컨텐트와 프로세스 매트릭스', 2x2Matrix)
        if _FIGURE_CAPTION_RE.match(match.string[match.start(4):]):
            return before + "\n\n" + after
        # 다음 쪽이 제목('4 / 지난 일은 지나가게 두라')이나 들여 쓴 문단으로 시작하면 문단 경계다.
        # 장 끝 요약 문구는 마침표 없이 끝나('…터득하는 것이다') 아래 규칙만으로는 제목을 앞 문단에 붙였다
        # (인생잠언 21쪽 등, 2026-10-19)
        if "\n" in ws_before + ws_after and (ws_after.endswith("  ")
                                             or _is_heading_line(match.string[match.start(4):])):
            return before + "\n\n" + after
        # 앞 문장이 끝났고 사이에 줄바꿈이 있으면 쪽 경계 = 문단 경계 → 빈 줄 유지
        if _SENT_END.search(before):
            return before + ("\n\n" if "\n" in ws_before + ws_after else " ") + after
        return before + _glue(before, after) + after

    # ⟨PAGEBREAK⟩ 앞뒤 어절 + 사이 공백(줄바꿈 여부 판정용)
    pattern = re.compile(r'(\S+)(\s*)⟨PAGEBREAK⟩(\s*)(\S+)')

    # 여러 번 적용하여 중결합 해소
    processed = pattern.sub(_repl, text)
    # 남은 PAGEBREAK 센티넬은 제거
    processed = processed.replace(PAGEBREAK, "")
    return processed


def _build_prompt(joined: str, drop_front: bool = False, drop_colophon: bool = False) -> str:
    """정제 프롬프트 생성. 부속물 제거 옵션이 켜지면 해당 규칙을 덧붙인다."""
    enable_filter = config.get_enable_margin_filter()

    prompt = CLEAN_PROMPT
    if enable_filter:
        prompt = prompt.replace(
            "3. 레이아웃 노이즈'만' 제거: 다음 항목만 삭제하십시오 — ① 쪽번호(단독 숫자), ② 러닝헤더/러닝푸터: 각 페이지의 맨 위나 맨 아래(입력 맨 앞·맨 뒤, ⟨PAGEBREAK⟩ 바로 앞뒤 포함)에 붙는 '책 제목' 단독 줄이나 '마태복음 5'처럼 '제목+쪽수' 형태의 짧은 줄, ③ 자모 노이즈(단독 'ㄱ','ㄴ'), ④ 스캔 줄('──'), ⑤ 특수 기호 불릿. 단, 본문에 속한 고유 번호 체계(성경의 장·절 번호, 법조문 번호, 항목·목록 번호 등)는 레이아웃 노이즈가 아니라 콘텐츠이므로 반드시 그대로 보존하십시오.",
            "3. 레이아웃 노이즈'만' 제거: 선행 제거 필터링 후 잔존할 수 있는 미세 노이즈(자모 노이즈 'ㄱ', 'ㄴ', 스캔 잡음줄 '──', 기호 불릿)만 제거하십시오. (러닝헤더와 쪽번호는 사전에 제거되었으므로 고려하지 않아도 됩니다.)"
        )
        prompt = prompt.replace(
            "4. 페이지 경계 처리: '⟨PAGEBREAK⟩' 표시 자체는 삭제하십시오. 그 경계에서 한 단어나 문장이 잘려 다음 페이지로 이어지면 자연스럽게 이어 붙이십시오. (예: '...성' ⟨PAGEBREAK⟩ '찰의...' → '...성찰의...')",
            "4. 페이지 경계 처리: '⟨PAGEBREAK⟩' 표시 자체는 삭제하지 말고 그대로 출력하십시오. 단, 그 경계에서 앞뒤로 쪼개진 단어가 있으면 결합하거나 띄어쓰기를 교정하여 '단어 ⟨PAGEBREAK⟩' 또는 '⟨PAGEBREAK⟩ 단어' 형태로 어휘만 올바르게 수정하십시오."
        )

    prompt = prompt.replace("{TEXT}", joined)
    extras = []
    if drop_front:
        extras.append(_RULE_FRONT_MATTER)
    if drop_colophon:
        extras.append(_RULE_COLOPHON)
    if extras:
        # 규칙 0(최우선 무손실)에 밀려 LLM이 7·8번을 무시한다 — 맨 앞에 예외로 선언해야 듣는다
        # (실측 2026-10-05: 끝에 붙이거나 0번 문구만 고치면 표제지·약력·광고가 그대로 남음)
        prompt = prompt.replace("[수행 규칙]\n", "[수행 규칙]\n" + _MATTER_PRECEDENCE + "\n", 1)
        prompt = prompt.rstrip() + "\n" + "\n".join(extras) + "\n"
    return prompt


# ── 내부 ──────────────────────────────────────────────────────────────────────

def _page_ends_cleanly(page_text: str) -> bool:
    """페이지의 마지막 의미 있는 줄이 문장 종료 부호로 끝나면 True."""
    for line in reversed(page_text.splitlines()):
        s = line.strip()
        if s:
            return bool(_SENT_END.search(s))
    return True  # 내용이 없으면 안전한 경계로 간주


def _chunk_pages(pages: list[str], soft: int, hard: int) -> list[list[str]]:
    """
    페이지들을 청크(페이지 리스트)로 묶는다. 청크 경계는 '문장이 종결되는
    페이지'에만 두어, 어떤 구간에서도 문장/어절이 청크 경계에서 잘리지 않게 한다.

    - soft 이상 누적 + 현재 페이지가 문장으로 깔끔히 끝 → 그 페이지에서 끊음
      (가장 자연스러운 경계).
    - hard 이상 누적(문장이 계속 이어져 soft 지점을 못 잡은 경우) → 강제 분할하되,
      가능하면 '마지막으로 문장이 끝난 페이지'까지만 청크에 넣고 나머지는 다음
      청크로 넘겨(carry) 경계가 문장 끝에 오도록 스냅한다. 문장 끝 페이지가
      전혀 없을 때만 현재 지점에서 끊는다.
    - 페이지는 절대 중간에서 쪼개지 않는다(청크 안에서 페이지 넘김 복원 보장).
    """
    chunks: list[list[str]] = []
    cur: list[str] = []
    cur_len = 0
    last_clean = -1   # cur 내에서 '문장이 끝난' 마지막 페이지의 인덱스
    for pg in pages:
        cur.append(pg)
        cur_len += len(pg)
        if _page_ends_cleanly(pg):
            last_clean = len(cur) - 1

        # soft: 문장이 끝나는 페이지에서 끊기 (가장 자연스러운 경계)
        if cur_len >= soft and _page_ends_cleanly(pg):
            chunks.append(cur)
            cur, cur_len, last_clean = [], 0, -1
            continue

        # hard: 강제 분할 — 마지막 '문장 끝' 페이지까지만 넣고 나머지는 이월
        if cur_len >= hard:
            if 0 <= last_clean < len(cur) - 1:
                chunks.append(cur[:last_clean + 1])
                cur = cur[last_clean + 1:]
                cur_len = sum(len(p) for p in cur)
            else:
                chunks.append(cur)
                cur, cur_len = [], 0
            # 이월된 페이지는 모두 '마지막 문장 끝 페이지' 이후라 정의상 문장이
            # 안 끝난 페이지들뿐 → last_clean=-1 리셋이 정확(이후 새 페이지가
            # 문장을 끝내면 그때 다시 갱신됨).
            last_clean = -1
    if cur:
        chunks.append(cur)
    return chunks


def _strip_output(text: str) -> str:
    """모델 출력에서 잔여 센티넬·코드펜스·과공백 제거.

    마진 필터 ON 일 때는 센티넬을 남긴다 — 그 모드의 프롬프트는 규칙 4를
    "⟨PAGEBREAK⟩ 를 지우지 말고 그대로 출력"으로 바꿔 두고, 페이지 넘김 어절 결합을
    마지막 post_process_pagebreaks 에 맡기기 때문. 여기서 지우면 그 단계가
    무동작이 되어 잘린 어절이 복원되지 않는다.
    """
    if not text:
        return ""
    if not config.get_enable_margin_filter():
        text = text.replace(PAGEBREAK, " ")
    # 혹시 모델이 ```로 감쌌으면 제거
    text = re.sub(r'^\s*```[a-zA-Z]*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    # 줄 내부 과공백 정리(줄바꿈은 보존)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


_SCAN_LINE_RE = re.compile(r'[─━═]{5,}')                  # C-01·C-07 스캔·구분선
_LONE_JAMO_RE = re.compile(r'^[ \t]*[ㄱ-ㅎㅏ-ㅣ][ \t]*$\n?', re.M)  # C-07 단독 자모 줄


def _rules_clean(joined: str) -> str:
    """비용 우선 정제(LLM 없음): 구분선·단독 자모 제거(C-01·C-07) + 쪽 경계 잘린 어절 결합(C-03)
    + 단락 복원(C-08). C-02·C-10/11 은 iter_clean 앞단, C-04·C-05(사전 범위)는 korean_corrector 가 맡는다.
    실측(2026-10-15, DBR 디지털 PDF): 구분선 제거가 없으면 '────' 195개가 결과에 남았다."""
    from core import postprocessor
    text = _SCAN_LINE_RE.sub("\n", joined)
    text = _LONE_JAMO_RE.sub("", text)
    text = post_process_pagebreaks(text)
    text = postprocessor.join_broken_lines(text)
    return _strip_output(text)


def _fallback_clean(joined: str) -> str:
    """
    API 미가용/실패 시 규칙 기반 최소 정제(강등 경로).
    센티넬 제거 + 페이지 내 단락 복원(postprocessor 재사용)만 수행.
    """
    from core import postprocessor
    text = joined.replace(PAGEBREAK, "\n")
    text = postprocessor.join_broken_lines(text)
    return _strip_output(text)


# ── 통합_정제규칙서 §3.3 D-02: 단일 음절 반복 스페이싱 결합 (결정론, LLM 불필요) ──
# 판권면 라벨 등에서 나타나는 `번 역 :`, `발 행 인 :`, `정 동 수` 같은
# 한 글자씩 띄어쓴 표기를 정규식으로 결합한다. 오탐 방지를 위해 문장 중간은
# 건드리지 않고, (a) 콜론 라벨 앞, (b) 줄 전체가 단음절 나열인 경우만 결합한다.
# LLM 정제 전에 적용해 청크마다 다르게 처리되는 비일관성을 없애고,
# API 실패 강등 경로에서도 동일하게 작동한다.
# 콜론 라벨: 앞에 한글이 붙어있지 않은(단어 경계) 2~4음절 나열 + 콜론.
# 부정 후방탐색으로 '그리스도 예수안에 발 행 인 :'의 '에'가 끌려들지 않게 한다.
_SYLLABLE_LABEL_RE = re.compile(r'(?<![가-힣])(?:[가-힣]\s){1,3}[가-힣](?=\s*:)')
_SYLLABLE_LINE_RE = re.compile(r'^(?:[가-힣]\s){2,}[가-힣]\s*$')


# ── 러닝헤더 결정론 제거 (QA 2026-10-03) ─────────────────────────────────────
# 실측: LLM은 청크(약 4000자)만 보므로 'Chapter 1 새벽, …'·'하루 명화 하루 명언' 같은
# 책 전체 반복 머리글/꼬리말을 본문으로 오인해 문단 사이에 남겼다(5권 중 3권).
# 오삭제 방지: 페이지 맨 첫/끝 줄만, 같은 줄이 _HEADER_MIN_PAGES쪽 이상 반복될 때만,
# 첫 등장(표제·장 시작)은 보존. 쪽번호는 떼고 남는 게 여러 단어일 때만 뗀다 —
# 'Chapter 1'·'누가복음 5'가 하나로 묶여 차례·장 머리글이 지워지는 것을 막는다.
_HEADER_MIN_PAGES = 5
_HEADER_MAX_LEN = 40
_PAGE_NO_RE = re.compile(r'^\d{1,4}\s+|\s+\d{1,4}$')
# '머리말 | 029'·'028 | 피터 드러커의 매니지먼트' 꼴 쪽 꼬리말 — 장마다 제목이 달라(짧은 장·OCR 오타
# '미리말 | 037') 반복 횟수 규칙에 안 걸리던 것. 모양이 뚜렷해 페이지 가장자리 줄이면 바로 뗀다.
_FOOTER_RE = re.compile(r'^(?:[^|\d][^|]{0,39}\|\s*\d{1,4}|\d{1,4}\s*\|[^|]{1,40})$')


# OCR 이 같은 머리글을 쪽마다 다르게 읽는 경우(실측: MyOCR 1분의사결정 165쪽) —
#   장 번호가 깨짐('IV.'·'n.'·'in,'), 쪽번호가 깨지거나 붙음('6?'·'6i'·'찾는다23'), 띄어쓰기가 다름('머리를써라')
_SECTION_PREFIX_RE = re.compile(r'^(?:[IVXivxnl]{1,4}|\d{1,3})?\s*[.,]\s*')     # 'IV.'·'I .'·'n.'·'in,'
# 쪽번호: 숫자가 섞인 꼬리('6?'·'6i'·'찾는다23'), 또는 한글 뒤 숫자 닮은 글자뿐인 꼬리('51'→'si')
_GARBLED_PAGE_NO_RE = re.compile(r'\s*[\dIlOoi?!|]*\d[\dIlOoi?!|]*$|(?<=[가-힣])\s+[IlOoiSsZz?!|]{1,3}$')


def _header_key(line: str) -> str | None:
    k = re.sub(r'\s+', ' ', line.strip())
    # 한글 꼬리말 + 쪽번호는 띄어 썼든 붙었든 같은 꼬리말('정의에 관한 명언 15'·'충고에관한명언143'·'156세계명언집'
    # — 대한민국CEO명언집 실측: Vision 쪽은 띄어 쓰고 OCR 쪽은 붙어 서로 다른 머리글로 셌다).
    # 한글 5자 이상만 — '누가복음 5'·'창세기 3' 같은 장 머리글은 묶지 않는다. 쪽번호가 붙어 있으니 첫 등장도 꼬리말('#')
    if (re.fullmatch(r'\d{1,4}\s?[가-힣][가-힣 ]+|[가-힣][가-힣 ]+?\s?\d{1,4}', k)
            and len(re.findall(r'[가-힣]', k)) >= 5):
        return '#' + re.sub(r'[\d\s]', '', k)
    stripped = _PAGE_NO_RE.sub('', k).strip()
    if stripped != k and ' ' in stripped:
        k = stripped
    # 장 번호 표시가 앞에 있으면 깨진·붙은 쪽번호까지 떼고 공백 없이 비교한다.
    # 장 번호 표시가 없는 '누가복음 5'·'Chapter 1' 은 위 규칙대로 묶지 않는다(장 머리글 보호)
    m = _SECTION_PREFIX_RE.match(k)
    if m and m.end() > 0:
        core = _GARBLED_PAGE_NO_RE.sub('', k[m.end():]).strip()
        if len(re.findall(r'[가-힣A-Za-z]', core)) >= 4:
            k = '§' + re.sub(r'\s+', '', core)
    # '4장왜지능이라고 하는가?55' — 장·부 번호로 시작하는 쪽 꼬리는 붙은 쪽번호까지 떼고 비교한다(예측기계 실측)
    elif re.match(r'\d{1,2}\s?[장부]', k):
        core = re.sub(r'\s*\d{1,4}$', '', k)
        if len(re.findall(r'[가-힣A-Za-z]', core)) >= 4:
            k = '§' + re.sub(r'\s+', '', core)
    if not (2 <= len(k) <= _HEADER_MAX_LEN) or k.startswith('['):
        return None
    if not re.search(r'[가-힣A-Za-z]{2}', k):
        return None
    return k


def _edge_indices(lines: list[str]) -> list[int]:
    idx = [i for i, l in enumerate(lines) if l.strip()]
    return sorted({idx[0], idx[-1]}) if idx else []


def strip_running_headers(pages: list[str]) -> list[str]:
    """페이지 맨 첫/끝 줄에 _HEADER_MIN_PAGES쪽 이상 반복되는 줄(러닝헤더)과 '제목 | 쪽' 꼬리말을 제거.

    '장·부 제목+쪽번호' 쪽 꼬리(postprocessor._PAGE_FOOTER_LINE)도 모양만으로 뗀다 — 짧은 장은 5번 미만이라
    반복 규칙에 안 걸리고, 짝수 쪽 꼬리('1582부의사결정')는 쪽마다 달라 묶이지 않았다(예측기계 표본 40쪽 중
    28쪽에 남음, 2026-10-14). 쪽번호가 붙어 있어 장 제목(번호 없음)과 헷갈리지 않으므로 첫 등장도 뗀다."""
    split = [p.splitlines() for p in pages]
    seen: dict[str, set[int]] = {}
    for n, lines in enumerate(split):
        for i in _edge_indices(lines):
            k = _header_key(lines[i])
            if k:
                seen.setdefault(k, set()).add(n)
    headers = {k: min(ns) for k, ns in seen.items() if len(ns) >= _HEADER_MIN_PAGES}
    out = []
    for n, lines in enumerate(split):
        drop = {i for i in _edge_indices(lines)
                if ((k := _header_key(lines[i])) in headers and (headers[k] != n or k.startswith('#')))
                or _FOOTER_RE.match(lines[i].strip()) or _PAGE_FOOTER_LINE.match(lines[i].strip())}
        out.append("\n".join(l for i, l in enumerate(lines) if i not in drop)
                   if drop else pages[n])
    return out


# ── 차례·판권 결정론 제거 (2026-10-05) ───────────────────────────────────────
# ── 한 글자씩 띄운 쪽 머리글 (삶의지혜_자아편 실측, 2026-11-07) ─────────────────────
# '소 중 한 사 람에 게 주 고 싶 은 삶의 지 혜' 가 문단 한가운데 섞여 들었다(15곳). OCR 이 매번 조금씩 달리 읽어
# ('사 람 예 게'·'삶의 지 혜다른'·끝 글자 빠짐) 쪽 맨 위·아래 같은 줄만 지우는 strip_running_headers 로는 못 지운다.
# 오삭제 방지: 책 안에서 같은 꼴로 3번 이상 되풀이된 8자 이상 띄운 문구만 머리글로 보고, 지울 자리도
# 글자 사이가 절반 이상 띄어진 곳만 — 보통 띄어쓰기로 쓴 같은 말('소중한 사람에게 …')은 본문이다.
_SPACED_RUN_RE = re.compile(r'(?<![가-힣])(?:[가-힣]{1,2} +){4,}[가-힣]{1,2}(?![가-힣])')
_SPACED_ANY_RE = re.compile(r'(?:[가-힣]{1,3} +){3,}[가-힣]+')
_SPACED_HEADER_REPEAT = 3
_SPACED_HEADER_MIN = 8
_SPACED_HEADER_MATCH = 0.8


def strip_spaced_running_header(text: str) -> tuple[str, int]:
    """책 전체에 되풀이된 한 글자씩 띄운 머리글을 지운다. (본문, 지운 곳 수)"""
    counts: dict[str, int] = {}
    for m in _SPACED_RUN_RE.finditer(text):
        k = m.group().replace(" ", "")
        if len(k) >= _SPACED_HEADER_MIN:
            counts[k] = counts.get(k, 0) + 1
    key = max((k for k, c in counts.items() if c >= _SPACED_HEADER_REPEAT), key=counts.get, default=None)
    if key is None:
        return text, 0
    cuts: list[tuple[int, int]] = []
    for m in _SPACED_ANY_RE.finditer(text):
        pos = [m.start() + i for i, ch in enumerate(m.group()) if ch != " "]
        compact = "".join(text[p] for p in pos)
        blocks = [b for b in SequenceMatcher(None, compact, key, autojunk=False).get_matching_blocks() if b.size]
        if sum(b.size for b in blocks) < len(key) * _SPACED_HEADER_MATCH:
            continue
        a, b = pos[blocks[0].a], pos[blocks[-1].a + blocks[-1].size - 1] + 1
        if text.count(" ", a, b) * 2 >= b - a - text.count(" ", a, b):
            cuts.append((a, b))
    for a, b in reversed(cuts):
        left, right = text[:a].rstrip(" "), text[b:].lstrip(" ")
        text = left + (" " if left[-1:] not in ("", "\n") and right[:1] not in ("", "\n") else "") + right
    return text, len(cuts)


# 실측: 피터드러커 매니지먼트(715p) — 차례(24~27쪽)가 청크 6~7에 있어, 표제지 뒤 해설이
# 먼저 나와 body_started 가 켜진 뒤라 LLM 삭제 규칙이 닿지 않았고, 판권(715쪽)도 남았다.
# 차례·판권은 형태가 뚜렷하므로 LLM 앞에서 규칙으로 걷어내고, 무엇을 걷었는지 보고한다.
# 해설·머리말·주석 같은 '내용 있는 글'은 건드리지 않는다.
_TOC_HEAD_RE = re.compile(r'^(?:상권|중권|하권|전권|상|하)?\s*(?:차\s?례|목\s?차|contents)\s*$', re.I)
_SENT_LINE_RE = re.compile(r'[.?!][”’"\')」』]*\s*$')
_COLOPHON_RE = re.compile(r'ISBN|펴낸곳|펴낸이|발행처|발행인|출판등록|책값|재사용하려면|교환해\s*드립니다|KI신서|\d\s*판\s*\d\s*쇄')
_AD_RE = re.compile(r'\d{2,4}\s*[쪽면]\s*[|｜]\s*[\d,]{4,}\s*원')   # '296쪽 | 14,000원' 도서 광고
_COLOPHON_PAGE_MIN_HITS = 3     # 한 쪽에 서로 다른 표지가 이만큼이면 판권 쪽으로 본다
_COLOPHON_MAX_PAGES = 3
_MATTER_FRONT_MAX_PAGES = 40
_MATTER_FRONT_RATIO = 0.15


def _is_sentence_line(line: str) -> bool:
    s = line.strip()
    return len(s) >= 20 and bool(_SENT_LINE_RE.search(s))


def _paragraph_bounds(lines: list[str]) -> list[tuple[int, int]]:
    """줄 리스트를 빈 줄 기준 문단 (시작, 끝+1) 목록으로."""
    out, start = [], None
    for i, l in enumerate(lines):
        if l.strip():
            if start is None:
                start = i
        elif start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(lines)))
    return out


def _drop_toc(pages: list[str], notes: list[str]) -> list[str]:
    """선두 구간에서 '차례' 머리글부터 문장이 나오기 전까지를 제거."""
    n = len(pages)
    window = _front_window(n)
    out = list(pages)
    i = 0
    while i < window:
        lines = out[i].splitlines()
        head = next((k for k, l in enumerate(lines) if _TOC_HEAD_RE.match(l.strip())), None)
        if head is None:
            i += 1
            continue
        first = lines[head].strip()
        kept_before = "\n".join(lines[:head]).rstrip()
        # 머리글 쪽: 머리글 이후 첫 문장 줄이 든 문단 앞까지 제거, 문장이 없으면 쪽 끝까지
        tail = lines[head:]
        sent = next((k for k, l in enumerate(tail) if _is_sentence_line(l)), None)
        if sent is None:
            kept_after = ""
            toc_end = None
        else:
            para = next(((a, b) for a, b in _paragraph_bounds(tail) if a <= sent < b), (sent, sent))
            kept_after = "\n".join(tail[para[0]:]).strip()
            toc_end = i
        out[i] = "\n".join(x for x in (kept_before, kept_after) if x)
        last = i
        if toc_end is None:   # 다음 쪽부터 문장 줄이 나오는 쪽 직전까지 차례로 간주
            j = i + 1
            while j < n and not any(_is_sentence_line(l) for l in out[j].splitlines()) \
                    and re.search(r'\d', out[j]):
                out[j] = ""
                last = j
                j += 1
        notes.append(f"차례 {i + 1}~{last + 1}번째 쪽('{first[:12]}…')")
        i = last + 1
    return out   # 빈 쪽은 strip_book_matter 가 걸러낸다(여기서 빼면 이후 보고 쪽 번호가 당겨진다)


# 실측(2026-10-05): '교정하면서 앞부속만 지워라'(규칙 7)는 같은 입력·temperature 0 에서도
# 지웠다 남겼다 했다(드러커 상권 첫 구간). 문단 번호만 고르게 하는 분류는 3회 모두 같은 답.
_FRONT_CLASSIFY_PROMPT = """다음은 스캔한 책의 맨 앞부분입니다. 문단마다 번호가 붙어 있습니다.
책 앞부속(표제지·반복된 표제, 판권, 차례, 헌사, 시리즈/출판사 광고·소개, 구매 혜택 안내, 저자·역자·감수자 약력과 사진, 디자인·제작 크레딧, 앞·뒤표지와 날개의 문구 — 홍보 카피·짧은 추천 서평 발췌·가격·ISBN·바코드, '이미지 감지됨'·'표 감지됨' 같은 빈 표식)에 해당하는 문단 번호만 JSON 배열로 출력하십시오.
감수자·역자 해설, 머리말, 서문, 책 안의 '추천사/추천의 글' 장, 본문, 주석은 앞부속이 아닙니다. 해당 없으면 [] 를 출력하십시오. 설명은 쓰지 마십시오.

"""


_FRONT_BODY_MIN = 100   # 이 길이 이상 문단을 남기라고 분류되면 본문 시작으로 본다


def drop_front_paragraphs(pages: list[str]) -> tuple[list[str], list[str]] | None:
    """LLM 문단 분류로 앞부속 문단을 지운다. (남은 쪽들, 지운 문단들), 분류 실패 시 None."""
    split = [[p for p in re.split(r"\n\s*\n", pg) if p.strip()] for pg in pages]
    flat = [p for ps in split for p in ps]
    if not flat:
        return pages, []
    body = "\n\n".join(f"[{k}] {p}" for k, p in enumerate(flat))
    try:
        reply = _gemini.run_text(_FRONT_CLASSIFY_PROMPT + body) or ""
    except Exception:
        return None
    m = re.search(r"\[[\d,\s]*\]", reply)
    if not m:
        return None
    drop = {int(x) for x in re.findall(r"\d+", m.group(0))}
    # 앞부속은 책 맨 앞에 몰려 있다 — 남길 긴 문단(본문)이 처음 나온 뒤로는 지우지 않는다.
    # 실측: 분류가 본문 장 제목('제1장 성장과 전략')·서문 제목까지 앞부속으로 고른 경우가 있었다.
    body_at = next((k for k, p in enumerate(flat) if k not in drop and len(p) >= _FRONT_BODY_MIN),
                   len(flat))
    drop = {k for k in drop if k < body_at}
    out, removed, k = [], [], 0
    for ps in split:
        keep = []
        for p in ps:
            (removed if k in drop else keep).append(p)
            k += 1
        if keep:
            out.append("\n\n".join(keep))
    return out, removed


def _front_window(n: int) -> int:
    return min(n, _MATTER_FRONT_MAX_PAGES, max(10, int(n * _MATTER_FRONT_RATIO)))


def _cut_colophon_block(text: str) -> str:
    """판권 표지가 든 첫 문단부터 마지막 문단까지 제거 — 같은 쪽의 일러두기·표제 등은 보존."""
    lines = text.splitlines()
    hit = [(a, b) for a, b in _paragraph_bounds(lines) if _COLOPHON_RE.search("\n".join(lines[a:b]))]
    return "\n".join(lines[:hit[0][0]] + lines[hit[-1][1]:]).strip()


def _drop_colophon(pages: list[str], notes: list[str]) -> list[str]:
    """선두·말미 쪽에서 판권(간기)과 다른 책 광고를 제거.

    실측(2026-10-05, 7권): 판권이 책 앞쪽(고백록 8쪽)에 있거나, 마지막 쪽이 아닌
    끝에서 둘째 쪽(FocalPoint)에 있는 책이 있고, 앞뒤에 '지음 | 384쪽 | 15,000원' 꼴
    도서 광고 쪽이 끼어 있었다.
    """
    out = list(pages)
    n = len(out)
    found: dict[str, list[int]] = {}
    targets = sorted(set(range(_front_window(n))) | set(range(max(0, n - _COLOPHON_MAX_PAGES), n)))
    for idx in targets:
        text = out[idx]
        if _AD_RE.search(text):
            out[idx] = ""
            found.setdefault("도서 광고", []).append(idx + 1)
        elif len({m.group(0) for m in _COLOPHON_RE.finditer(text)}) >= _COLOPHON_PAGE_MIN_HITS:
            out[idx] = _cut_colophon_block(text)
            found.setdefault("판권", []).append(idx + 1)
    # 마지막 쪽: 표지가 적어도 끝 문단들이 판권이면 제거(그 앞 주석 등은 보존)
    last = max((i for i, p in enumerate(out) if p.strip()), default=None)
    if last is not None and last >= n - _COLOPHON_MAX_PAGES:
        lines = out[last].splitlines()
        cut = len(lines)
        for a, b in reversed(_paragraph_bounds(lines)):
            if _COLOPHON_RE.search("\n".join(lines[a:b])):
                cut = a
            else:
                break
        if cut < len(lines):
            out[last] = "\n".join(lines[:cut]).rstrip()
            found.setdefault("판권", []).append(last + 1)
    for kind, idxs in found.items():
        notes.append(f"{kind} {', '.join(str(i) for i in sorted(set(idxs)))}번째 쪽")
    # 빈 쪽은 여기서 빼지 않는다 — 앞쪽 판권·광고를 지운 뒤 빼면 다음 차례 보고의 쪽 번호가 당겨진다
    return out if found else pages


def strip_book_matter(pages: list[str]) -> tuple[list[str], list[str]]:
    """차례(선두)·판권(말미)을 규칙으로 제거. (남은 쪽들, 사람용 보고 줄들) 반환."""
    notes: list[str] = []
    pages = _drop_colophon(pages, notes)   # 말미 먼저 — 쪽 번호가 입력 기준으로 유지된다
    pages = _drop_toc(pages, notes)
    return [p for p in pages if p.strip()], [f"앞부속·뒤부속 자동 제거 — {s} (정제 전 원문 사본에서 확인 가능)" for s in notes]


def deterministic_preclean(text: str) -> str:
    """D-02: 단일 음절 반복 스페이싱 결합. 페이지 텍스트에 LLM 전 적용."""
    if not text:
        return text

    def _join_label(m):
        return re.sub(r'\s+', '', m.group(0))

    text = _SYLLABLE_LABEL_RE.sub(_join_label, text)
    lines = []
    for line in text.splitlines():
        if _SYLLABLE_LINE_RE.match(line.strip()):
            lines.append(re.sub(r'\s+', '', line.strip()))
        else:
            lines.append(line)
    return "\n".join(lines)
