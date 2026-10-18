"""문서 정제(text_cleaner) 테스트 — 청크 분할·센티넬 처리·강등 경로·클린 저장"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import text_cleaner, writer
from core.types import OutputFormat


# ── 페이지 경계 판단 ──────────────────────────────────────────────────────────

class TestPageEndsCleanly:
    def test_ends_with_period(self):
        assert text_cleaner._page_ends_cleanly("문장이 끝난다.")

    def test_ends_with_closing_quote(self):
        assert text_cleaner._page_ends_cleanly('그는 말했다."')

    def test_word_cut_is_not_clean(self):
        # 페이지 넘김으로 어절이 잘린 전형적 케이스
        assert not text_cleaner._page_ends_cleanly("가치가 있으며, 성")

    def test_empty_is_clean(self):
        assert text_cleaner._page_ends_cleanly("   \n  ")

    def test_uses_last_nonempty_line(self):
        assert text_cleaner._page_ends_cleanly("끝났다.\n\n")


# ── 청크 분할 ─────────────────────────────────────────────────────────────────

class TestChunkPages:
    def test_single_chunk_when_small(self):
        assert text_cleaner._chunk_pages(["짧다."], soft=4000, hard=8000) == [["짧다."]]

    def test_breaks_on_clean_end_after_soft(self):
        pages = ["가나다라마.", "바사아자차.", "카타파하가."]  # 각 6자
        chunks = text_cleaner._chunk_pages(pages, soft=10, hard=100)
        assert chunks[0] == ["가나다라마.", "바사아자차."]   # 12자≥soft & 문장끝 → 끊김
        assert chunks[-1] == ["카타파하가."]

    def test_no_break_until_clean_end(self):
        pages = ["가나다라마바", "사아자차카타", "끝문장이다."]  # 앞 둘은 문장끝 아님
        chunks = text_cleaner._chunk_pages(pages, soft=6, hard=1000)
        assert chunks == [["가나다라마바", "사아자차카타", "끝문장이다."]]

    def test_hard_limit_forces_break(self):
        pages = ["가나다라마바사아자차", "카타파하가나다라마바"]  # 각 10자, 문장끝 아님
        chunks = text_cleaner._chunk_pages(pages, soft=5, hard=10)
        assert chunks[0] == ["가나다라마바사아자차"]           # hard 도달 → 강제 끊김

    def test_hard_break_snaps_to_last_clean_page(self):
        # 문장이 계속 이어져 soft를 못 잡고 hard 도달 → 경계를 '문장 끝' 페이지로 스냅.
        # p1은 문장 끝('.'), p2·p3는 문장이 안 끝남. 경계는 p1 뒤에 와야 한다.
        pages = ["짧다.", "이어지는텍스트가", "계속잘림없이길게간다"]
        chunks = text_cleaner._chunk_pages(pages, soft=8, hard=20)
        assert chunks[0] == ["짧다."]                          # 문장 끝에서 스냅
        assert chunks[1] == ["이어지는텍스트가", "계속잘림없이길게간다"]  # 나머지 이월


# ── 정제 본체 (엔진 목킹) ─────────────────────────────────────────────────────

class TestCleanPages:
    def test_calls_once_and_joins(self, monkeypatch):
        calls = []
        # 무결성 가드(입력 대비 75% 미만 출력은 거부)를 통과하도록
        # 입력과 비슷한 길이의 정제 결과를 돌려준다.
        monkeypatch.setattr(text_cleaner._gemini, "run_text",
                            lambda p: calls.append(p) or f"본문{len(calls)} 가나다라마 바사아자차.")
        out = text_cleaner.clean_pages(["가나다라마.", "바사아자차."])
        assert len(calls) == 1          # 기본 chunk_chars(큼) → 1청크
        assert "본문1" in out

    def test_strips_pagebreak_and_fences(self, monkeypatch):
        monkeypatch.setattr(
            text_cleaner._gemini, "run_text",
            lambda p: "```\n정제된 본문 " + text_cleaner.PAGEBREAK + " 이어짐.\n```",
        )
        out = text_cleaner.clean_pages(["아무 페이지."])
        assert text_cleaner.PAGEBREAK not in out
        assert "```" not in out
        assert "정제된 본문" in out

    def test_empty_pages(self):
        assert text_cleaner.clean_pages([]) == ""
        assert text_cleaner.clean_pages(["  ", ""]) == ""

    def test_fallback_on_api_error(self, monkeypatch):
        def boom(_):
            raise RuntimeError("api down")
        monkeypatch.setattr(text_cleaner._gemini, "run_text", boom)
        out = text_cleaner.clean_pages(
            ["첫 줄은 길게 이어지는 미완성 문장이고\n둘째 줄로 계속된다"]
        )
        assert text_cleaner.PAGEBREAK not in out     # 강등돼도 센티넬은 없어야
        assert out.strip() != ""

    def test_progress_callback_final_is_complete(self, monkeypatch):
        monkeypatch.setattr(text_cleaner._gemini, "run_text", lambda p: "ok.")
        seen = []
        text_cleaner.clean_pages(["문장이다."], progress_cb=lambda d, t: seen.append((d, t)))
        assert seen and seen[-1][0] == seen[-1][1]   # 마지막 진행률 done==total


# ── 청크 무결성 가드 (메시지성경 소실 사고 회귀 방지) ─────────────────────────

class TestIntegrityGuard:
    """Gemini가 예외 없이 빈/과소 응답을 돌려줘도(안전필터 차단·조기 종료)
    청크가 소실되지 않고 원문 보존으로 강등되는지 검증한다.
    실측 사고: 메시지성경 신약 842p — 계시록 14~18장·마태복음 4:1-12 등이
    빈 응답 수용으로 통째 소실(2026-07-06)."""

    def test_empty_response_falls_back_to_source(self, monkeypatch):
        calls = []
        monkeypatch.setattr(text_cleaner._gemini, "run_text",
                            lambda p: calls.append(p) or "")
        src = "요한계시록의 긴 본문이 여기 이어진다. 심판과 구원의 말씀이 계속된다."
        out = text_cleaner.clean_pages([src])
        assert len(calls) == 2                        # 빈 응답 → 1회 재시도
        assert "요한계시록의 긴 본문" in out            # 청크 소실 없음 (원문 보존)
        rep = text_cleaner.get_last_report()
        assert rep["fallback_guard"] == 1 and rep["fallback_api"] == 0

    def test_truncated_response_falls_back(self, monkeypatch):
        # 입력의 75% 미만으로 잘린 출력은 수용하지 않는다
        monkeypatch.setattr(text_cleaner._gemini, "run_text", lambda p: "짧은 조각.")
        src = ("가나다라마바사아자차카타파하 " * 30) + "끝문장이다."
        out = text_cleaner.clean_pages([src])
        assert "가나다라마바사아자차카타파하" in out
        assert text_cleaner.get_last_report()["fallback_guard"] == 1

    def test_normal_response_accepted_without_guard(self, monkeypatch):
        src = "이 문장은 정제 대상 본문이다. 오타가 조금 있다."
        monkeypatch.setattr(
            text_cleaner._gemini, "run_text",
            lambda p: "이 문장은 정제 대상 본문이다. 오타가 조금 있었다.")
        out = text_cleaner.clean_pages([src])
        rep = text_cleaner.get_last_report()
        assert rep["fallback_guard"] == 0 and rep["fallback_api"] == 0
        assert "정제 대상 본문" in out

    def test_api_error_counted_in_report(self, monkeypatch):
        def boom(_):
            raise RuntimeError("down")
        monkeypatch.setattr(text_cleaner._gemini, "run_text", boom)
        text_cleaner.clean_pages(["본문이 있다."])
        assert text_cleaner.get_last_report()["fallback_api"] == 1

    def test_front_matter_chunk_may_be_emptied(self, monkeypatch):
        # 앞부속 제거 청크는 '전부 삭제 → 빈 출력'이 정당 — 가드 미적용
        monkeypatch.setattr(text_cleaner._gemini, "run_text", lambda p: "")
        text_cleaner.clean_pages(["표제지 판권 차례."], drop_front_matter=True)
        assert text_cleaner.get_last_report()["fallback_guard"] == 0

    def test_prompt_preserves_intrinsic_numbers(self):
        # 성경 장·절 번호 등 본문 고유 번호를 노이즈로 지우지 않도록 프롬프트에 명시
        assert "장·절 번호" in text_cleaner.CLEAN_PROMPT

    def test_prompt_keeps_figure_descriptions(self):
        # 2026-10-05: [그림: …] 은 모두 남긴다(사용자 결정) — 전엔 LLM이 일부만 지워 비일관
        p = text_cleaner.CLEAN_PROMPT
        assert "[그림: …]" in p and p.index("[그림: …]") < p.index("1. 문맥 파악")

    def test_prompt_has_lossless_rule_first(self):
        # 무손실 원칙(생략·요약 금지)이 최우선 규칙(0번)으로 명시되어야 한다 —
        # 기존 프롬프트는 '지어내기 금지'(추가 금지)만 있고 생략 금지가 없어
        # LLM이 반복 구간·긴 청크를 요약/생략해도 막을 장치가 없었다.
        p = text_cleaner.CLEAN_PROMPT
        assert "무손실 원칙" in p and "생략" in p
        assert p.index("무손실 원칙") < p.index("문맥 파악")
        # 삭제 지시는 러닝헤더 등 열거 항목'만'으로 한정 (성경 '장/절' 오독 방지:
        # 절 번호를 지우라는 뜻으로 읽히던 '장/절 헤더' 표현이 없어야 한다)
        assert "장/절 헤더" not in p


# ── 부속물(front matter) 제거 ─────────────────────────────────────────────────

def _cleaning_prompts(seen):
    """앞부속 문단 분류 호출을 빼고 정제 호출만."""
    return [p for p in seen if not p.startswith(text_cleaner._FRONT_CLASSIFY_PROMPT)]


class TestFrontClassify:
    def _fake(self, monkeypatch, answer):
        seen = []

        def run(p):
            seen.append(p)
            if p.startswith(text_cleaner._FRONT_CLASSIFY_PROMPT):
                return answer
            return p.split("---\n")[1].split("\n---")[0]     # 정제는 입력을 그대로 돌려줌
        monkeypatch.setattr(text_cleaner._gemini, "run_text", run)
        return seen

    def test_classified_paragraphs_removed_and_rule7_not_used(self, monkeypatch):
        seen = self._fake(monkeypatch, "[0, 1]")
        out = text_cleaner.clean_pages(["피터 드러커의 매니지먼트\n\n지은이 피터 드러커 약력",
                                        "해설\n\n드러커 경영학을 집대성한 이 책은 경영의 모든 것을 다룬다."],
                                       drop_front_matter=True)
        assert "지은이" not in out and "해설" in out and "집대성" in out
        assert all("앞부속 삭제" not in p for p in _cleaning_prompts(seen))
        assert any("앞부속 문단 2개 삭제" in w for w in text_cleaner.get_last_report()["warnings"])

    def test_unparseable_answer_falls_back_to_rule7(self, monkeypatch):
        seen = self._fake(monkeypatch, "모르겠습니다")
        text_cleaner.clean_pages(["표제지\n\n본문 문장이다."], drop_front_matter=True)
        assert "앞부속 삭제" in _cleaning_prompts(seen)[0]


class TestFrontMatter:
    def test_build_prompt_appends_front_rule(self):
        p = text_cleaner._build_prompt("본문", drop_front=True, drop_colophon=False)
        assert "앞부속 삭제" in p
        assert "뒤부속" not in p

    def test_build_prompt_appends_colophon_rule(self):
        p = text_cleaner._build_prompt("본문", drop_front=False, drop_colophon=True)
        assert "뒤부속 삭제" in p
        assert "앞부속" not in p

    def test_front_rule_relaxes_lossless_rule_zero(self):
        # 예외 선언이 규칙 0보다 앞에 있어야 LLM이 7번을 따른다(실측)
        p = text_cleaner._build_prompt("본문", drop_front=True)
        assert "최우선 예외" in p and p.index("최우선 예외") < p.index("0. 무손실")
        assert "최우선 예외" not in text_cleaner._build_prompt("본문")

    def test_build_prompt_clean_by_default(self):
        p = text_cleaner._build_prompt("본문내용")
        assert "앞부속" not in p and "뒤부속" not in p
        assert "{TEXT}" not in p and "본문내용" in p

    def test_drop_front_matter_marks_first_and_last_chunk(self, monkeypatch):
        seen = []
        monkeypatch.setattr(text_cleaner._gemini, "run_text",
                            lambda p: seen.append(p) or ("정제된 충분히 긴 본문 " * 10))
        text_cleaner.clean_pages(["표제지 텍스트."], drop_front_matter=True)
        seen = _cleaning_prompts(seen)    # 분류 답이 배열이 아니면 → 기존 규칙 7로 돌아감
        assert "앞부속 삭제" in seen[0]   # 첫 청크 = 앞부속 제거
        assert "뒤부속 삭제" in seen[0]   # 단일 청크 → 마지막 청크이기도 함

    def test_no_front_rule_when_disabled(self, monkeypatch):
        seen = []
        monkeypatch.setattr(text_cleaner._gemini, "run_text",
                            lambda p: seen.append(p) or "본문.")
        text_cleaner.clean_pages(["표제지."], drop_front_matter=False)
        assert "앞부속" not in seen[0]

    def test_front_matter_rule_capped_to_leading_chunks(self, monkeypatch):
        # 모든 청크가 짧게(<100자) 정제돼 body_started가 끝까지 안 켜지는 상황을 강제.
        # 이때도 앞부속 규칙은 선두 _FRONT_MATTER_MAX_CHUNKS 청크에만 적용돼야
        # 문서 중반 본문에 삭제 규칙이 새지 않는다.
        seen = []
        monkeypatch.setattr(text_cleaner._gemini, "run_text",
                            lambda p: seen.append(p) or "짧음.")
        pages = ["가" * 5000 for _ in range(8)]   # 2페이지/청크 → 4청크
        text_cleaner.clean_pages(pages, drop_front_matter=True)
        seen = _cleaning_prompts(seen)
        assert len(seen) >= text_cleaner._FRONT_MATTER_MAX_CHUNKS + 1
        for i, prompt in enumerate(seen):
            if i < text_cleaner._FRONT_MATTER_MAX_CHUNKS:
                assert "앞부속 삭제" in prompt, f"청크 {i}에 앞부속 규칙이 있어야 함"
            else:
                assert "앞부속 삭제" not in prompt, f"청크 {i}엔 앞부속 규칙이 없어야 함(상한)"


class TestStripBookMatter:
    """차례·판권 결정론 제거 — 해설·머리말·주석 같은 내용 글은 보존."""
    SENT = "이것은 충분히 긴 본문 문장으로 마침표로 끝난다."

    def _book(self):
        return [
            "표제지 텍스트",
            f"해설\n\n{self.SENT}",
            "상권 차례\n\n해설 집대성 004 머리말 대안 028\n\n1 경영의 등장 043 2 경영 붐 058",
            "하권 차례\n\n29 왜 경영자가 필요한가 30 무엇이 경영자를 만드는가\n\n주석\n참고문헌",
            f"머리말\n\n전제에 대한 대안\n\n{self.SENT}",
            f"28장\n1) {self.SENT}",
            "KI신서 1546\n\n피터 드러커의 매니지먼트(상)\n\n1판 1쇄 인쇄 2008년\n\n"
            "펴낸이 김영곤 펴낸곳 북이십일\n\n책값은 뒤표지에 있습니다 ISBN 978-89-509-1605-3",
        ]

    def test_removes_toc_pages_and_colophon_keeps_content(self):
        out, notes = text_cleaner.strip_book_matter(self._book())
        joined = "\n".join(out)
        assert "차례" not in joined and "ISBN" not in joined
        assert "머리말" in joined and "28장" in joined and "해설" in joined
        assert len(out) == 4
        assert len(notes) == 2 and all("앞부속" in n for n in notes)

    def test_partial_colophon_keeps_preceding_notes(self):
        pages = [self.SENT, f"28장\n1) {self.SENT}\n\n책값은 뒤표지에 있습니다 ISBN 978-89-509-1605-3"]
        out, notes = text_cleaner.strip_book_matter(pages)
        assert "28장" in out[-1] and "ISBN" not in out[-1]
        assert notes and "판권" in notes[0]

    def test_toc_word_inside_body_sentence_is_kept(self):
        pages = [f"본문 {self.SENT}", "이 책의 차례를 보면 알 수 있다.", f"{self.SENT}"]
        out, notes = text_cleaner.strip_book_matter(pages)
        assert out == pages and notes == []

    def test_disabled_flag_leaves_pages_untouched(self, monkeypatch):
        seen = []
        monkeypatch.setattr(text_cleaner._gemini, "run_text",
                            lambda p: seen.append(p) or "정제된 충분히 긴 본문 " * 10)
        text_cleaner.clean_pages(self._book(), drop_front_matter=False)
        assert "상권 차례" in seen[0] and "ISBN" in seen[-1]

    def test_iter_clean_reports_removals(self, monkeypatch):
        monkeypatch.setattr(text_cleaner._gemini, "run_text",
                            lambda p: "정제된 충분히 긴 본문 " * 10)
        text_cleaner.clean_pages(self._book(), drop_front_matter=True)
        warns = text_cleaner.get_last_report()["warnings"]
        assert any("차례" in w for w in warns) and any("판권" in w for w in warns)

    def _long_book(self, front, back):
        body = [f"{k}장 {self.SENT}" for k in range(60)]
        return front + body + back

    def test_front_colophon_block_cut_keeps_notes_on_same_page(self):
        # 고백록 8쪽: 일러두기 + 판권이 한 쪽에
        page = (f"일러두기\n\n• {self.SENT}\n\n1판 1쇄 발행 2014 | 발행처 포이에마 | 발행인 김도완\n\n"
                "ISBN 978-89-97760-68-8")
        out, notes = text_cleaner.strip_book_matter(self._long_book([page], []))
        assert "일러두기" in out[0] and "ISBN" not in out[0] and "발행처" not in out[0]
        assert any("판권" in n for n in notes)

    def test_book_ad_pages_removed_front_and_back(self):
        ad = "고흐 그림여행\n\n최상운 지음 | 296쪽 | 14,000원"
        out, notes = text_cleaner.strip_book_matter(self._long_book([ad], [ad]))
        assert "14,000원" not in "\n".join(out) and len(out) == 60
        assert any("도서 광고" in n for n in notes)

    def test_colophon_on_second_to_last_page(self):
        colophon = "KI신서 467\n\n펴낸곳 북21 펴낸이 김영곤\n\nISBN 89-509-0533-7"
        out, _ = text_cleaner.strip_book_matter(self._long_book([], [colophon, "세미나 안내 광고"]))
        assert "ISBN" not in "\n".join(out)


class TestFooterAndRejoin:
    LONG = ("경영자는 조직의 성과를 책임지는 사람이며 그 성과는 조직 바깥에서 고객에 의해 "
            "판정되므로 내부의 노력만으로는 결코 충분하다고 말할 수 없는데 바로")

    def test_title_page_footer_removed_at_edge_only(self):
        pages = ["본문 문장.\n\n머리말 | 029", "028 | 피터 드러커의 매니지먼트\n본문.", "가 | 1 같은 본문 줄\n끝."]
        out = text_cleaner.strip_running_headers(pages)
        assert out[0] == "본문 문장.\n" and out[1] == "본문."
        assert out[2] == pages[2]   # 쪽번호로 끝나지 않는 '|' 줄은 본문

    def test_rejoins_sentence_split_across_paragraphs(self):
        text, n = text_cleaner.rejoin_split_sentences(f"{self.LONG} 그\n\n것이다. 다음 문장.")
        assert n == 1 and "\n" not in text and "바로 그 것이다." in text

    def test_glues_split_word(self):
        text, _ = text_cleaner.rejoin_split_sentences(f"{self.LONG} 상\n상력을 넓혀 주었다.")
        assert "상상력을" in text

    def test_keeps_headings_and_finished_sentences(self):
        src = f"{self.LONG}.\n다음 문장이다.\n\n{self.LONG}\n\n제2장 경영의 과업\n\n본문."
        text, n = text_cleaner.rejoin_split_sentences(src)
        assert n == 0 and text == src


class TestDroppedRunWarning:
    def test_whole_page_silently_dropped_is_warned(self, monkeypatch):
        keep = [f"남는 문장 {k}번은 정상적으로 출력에 그대로 남아 있는 긴 문장이다." for k in range(30)]
        lost = [f"사라진 문장 {k}번은 OCR이 지어낸 쪽에 있던 것이라 정제에서 빠진다." for k in range(6)]
        monkeypatch.setattr(text_cleaner._gemini, "run_text", lambda p: " ".join(keep))
        text_cleaner.clean_pages([" ".join(keep), " ".join(lost)])
        warns = text_cleaner.get_last_report()["warnings"]
        assert any("연달아 빠짐" in w and "사라진 문장 0번" in w for w in warns)

    def test_normal_cleanup_not_warned(self, monkeypatch):
        src = " ".join(f"정상 문장 {k}번은 띄어쓰기만 고쳐서 그대로 돌려받는 긴 문장이다." for k in range(8))
        monkeypatch.setattr(text_cleaner._gemini, "run_text", lambda p: src)
        text_cleaner.clean_pages([src])
        assert not text_cleaner.get_last_report()["warnings"]


# ── 클린 저장 ─────────────────────────────────────────────────────────────────

class TestWritePlain:
    def test_no_page_markers(self, tmp_path):
        src = tmp_path / "book.pdf"
        out = writer.write_plain(src, "첫 단락.\n\n둘째 단락.", tmp_path, OutputFormat.TXT)
        content = out.read_text(encoding="utf-8")
        assert "[페이지" not in content
        assert "─" * 10 not in content
        assert content.startswith("첫 단락.")
        assert content.endswith("\n")
        assert out.name == "book.txt"


# ── 결정론적 사전 정제 (통합_정제규칙서 §3.3 D-02) ─────────────────────────────

class TestDeterministicPreclean:
    def test_colon_label_joined(self):
        """판권면 라벨의 단일 음절 스페이싱을 결합한다."""
        src = "번 역 : 그리스도 예수안에 발 행 인 : 정 동 수"
        assert text_cleaner.deterministic_preclean(src) == \
            "번역 : 그리스도 예수안에 발행인 : 정 동 수"

    def test_full_line_syllables_joined(self):
        """줄 전체가 단음절 나열이면 결합한다."""
        assert text_cleaner.deterministic_preclean("판 권 소 유") == "판권소유"

    def test_normal_sentences_untouched(self):
        """정상 문장(단음절 단어의 우연한 연속 포함)은 변형하지 않는다."""
        for s in [
            "그 때 그 사람이 말했다",
            "시간 : 오후 3시",
            "처음에 하나님께서 하늘과 땅을 창조하시니라.",
        ]:
            assert text_cleaner.deterministic_preclean(s) == s

    def test_empty_input(self):
        assert text_cleaner.deterministic_preclean("") == ""


def test_paragraphs_separated_by_one_blank_line():
    # 정제규칙서 C-08 '단락 = 빈 줄' — 쪽 안은 줄바꿈 하나, 쪽 경계는 빈 줄로 섞여 있었다(2026-10-18)
    src = "첫 단락.\n둘째 단락.\n\n\n셋째 단락.\n   넷째 단락."
    assert text_cleaner.unify_paragraph_breaks(src) == "첫 단락.\n\n둘째 단락.\n\n셋째 단락.\n\n넷째 단락."


def test_attribution_line_stays_with_quote():
    # 명언 저자 줄('- 이름')·글머리 목록은 앞 줄과 한 덩어리
    src = "명언 본문이다.\n- 벤저민 프랭클린\n다음 명언.\n• 항목"
    assert text_cleaner.unify_paragraph_breaks(src) == "명언 본문이다.\n- 벤저민 프랭클린\n\n다음 명언.\n• 항목"
