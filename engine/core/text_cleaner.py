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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from core import config
from core import gemini_vision as _gemini

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

def is_available() -> bool:
    """정제 사용 가능 여부. 현재는 Gemini 텍스트 호출 전용."""
    if config.get_vision_provider() == "claude":
        return False  # 정제 경로는 아직 Gemini만 구현
    return _gemini.is_available()


def clean_pages(pages: list[str], progress_cb=None, drop_front_matter: bool = False) -> str:
    """
    페이지 텍스트 리스트 → 정제된 전체 본문(문자열).

    progress_cb(done_chunks, total_chunks) 가 주어지면 청크 단위로 호출한다.
    drop_front_matter=True 면 표제지·판권·차례 등 앞뒤 부속물까지 제거한다.
    """
    gen = iter_clean(pages, drop_front_matter=drop_front_matter)
    try:
        while True:
            done, total = next(gen)
            if progress_cb:
                progress_cb(done, total)
    except StopIteration as stop:
        return stop.value or ""


def iter_clean(pages: list[str], drop_front_matter: bool = False, on_chunk=None,
               report_out: dict | None = None):
    """
    제너레이터: 청크 처리마다 (done_chunks, total_chunks) 를 yield 하고,
    마지막에 정제된 전체 본문을 return(StopIteration.value)한다.

    drop_front_matter=True: 앞부속(표제지·판권·차례·헌사·시리즈 광고)은 본문이
    나올 때까지 앞 청크에서 제거하고, 마지막 청크의 뒤부속(판권/간기/CIP)도 제거한다.
    부속물은 책의 맨 앞·맨 뒤에 몰려 있으므로 첫 청크(들)와 끝 청크에만 규칙을 적용해
    중간 본문은 건드리지 않는다.

    on_chunk(text): 청크 하나가 정제될 때마다 즉시 호출된다. 호출부가 이를 디스크에
    기록해 두면, 전 구간이 끝나기 전에 실행이 끊겨도(브라우저 종료·API 정지) 이미
    정제된 구간을 잃지 않는다. 반환값은 지금까지처럼 전 구간 완료 후에만 나온다.

    report_out: 넘기면 이 실행의 무결성 리포트를 채워 준다. 여러 파일을 동시에 처리할 때는
    get_last_report()(전역 — 마지막에 끝난 실행 것) 대신 이것을 쓴다.
    """
    global _last_report
    _last_report = None
    pages = [p.strip() for p in pages if p and p.strip()]
    if not pages:
        return ""

    # 통합_정제규칙서 §3.3: 결정론적 사전 정제(D-02) — LLM보다 먼저 적용해
    # 같은 오염이 청크마다 다르게 처리되는 것을 막는다.
    pages = [deterministic_preclean(p) for p in pages]
    # 청크만 보는 LLM은 책 전체에 반복되는 머리글을 알아보지 못한다 → 문서 단위로 선제거
    pages = strip_running_headers(pages)
    matter_notes: list[str] = []
    # 머리글을 뗀 뒤라야 잡음 쪽이 짧아져 후보가 된다. 지운 잡음 쪽은 빈 자리로 남겨
    # 부속물 보고와 같은 쪽 번호 기준(내용 있는 쪽 순서)을 유지하고, 부속물 제거 뒤에 걸러 낸다
    noise_notes: list[str] = []
    pages = drop_noise_pages(pages, noise_notes)
    if drop_front_matter:
        pages, matter_notes = strip_book_matter(pages)
    pages = [p for p in pages if p.strip()]
    if not pages:
        return ""
    matter_notes = noise_notes + matter_notes

    soft = config.get_cleanup_chunk_chars()
    hard = soft * 2
    chunks = _chunk_pages(pages, soft, hard)
    total = len(chunks)

    report = {"chunks": total, "fallback_api": 0, "fallback_guard": 0, "warnings": list(matter_notes)}
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
                           want_colophon=(i == total - 1))
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
                              want_colophon=drop_front_matter and k == total - 1): k
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

    return (text + "\n") if text else ""


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
    # 3. 그 외에는 일반적인 단어 경계로 보고 띄어씀
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
                 want_colophon: bool) -> _ChunkResult:
    """구간 하나를 정제한다. 공유 상태를 건드리지 않아 여러 구간을 동시에 돌릴 수 있다."""
    res = _ChunkResult()
    want_front = front_chunk
    if front_chunk:
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
    prompt = _build_prompt(joined, drop_front=want_front, drop_colophon=want_colophon)
    # 무결성 가드: 빈/과소 응답(안전필터 차단·조기 종료 — 예외 없이 옴)은
    # 수용하지 않고 1회 재시도, 그래도 미달이면 원문 보존 강등.
    cleaned = None
    api_failed = False
    for _attempt in range(2):
        try:
            cand = _strip_output(_gemini.run_text(prompt))
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


def post_process_pagebreaks(text: str) -> str:
    """
    텍스트 내의 ⟨PAGEBREAK⟩ 경계를 분석하여 잘려나간 어절을 합치고 센티넬을 소거합니다.
    """
    if not text:
        return ""

    def _repl(match):
        before, ws_before, ws_after, after = match.groups()
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
    if not (2 <= len(k) <= _HEADER_MAX_LEN) or k.startswith('['):
        return None
    if not re.search(r'[가-힣A-Za-z]{2}', k):
        return None
    return k


def _edge_indices(lines: list[str]) -> list[int]:
    idx = [i for i, l in enumerate(lines) if l.strip()]
    return sorted({idx[0], idx[-1]}) if idx else []


def strip_running_headers(pages: list[str]) -> list[str]:
    """페이지 맨 첫/끝 줄에 _HEADER_MIN_PAGES쪽 이상 반복되는 줄(러닝헤더)과 '제목 | 쪽' 꼬리말을 제거."""
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
                if ((k := _header_key(lines[i])) in headers and headers[k] != n)
                or _FOOTER_RE.match(lines[i].strip())}
        out.append("\n".join(l for i, l in enumerate(lines) if i not in drop)
                   if drop else pages[n])
    return out


# ── 차례·판권 결정론 제거 (2026-10-05) ───────────────────────────────────────
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
