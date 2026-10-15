"""디지털 PDF 텍스트 레이어 직접 추출 테스트 (core/pdf_text_layer.py)"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import pdf_text_layer as ptl


# ── 순수 함수: real_chars ─────────────────────────────────────────────────────

class TestRealChars:
    def test_counts_non_whitespace(self):
        assert ptl.real_chars("가 나\t다\n") == 3

    def test_empty(self):
        assert ptl.real_chars("") == 0
        assert ptl.real_chars("   \n\t ") == 0

    def test_none_safe(self):
        assert ptl.real_chars(None) == 0


# ── 순수 함수: looks_digital (휴리스틱) ───────────────────────────────────────

class TestLooksDigital:
    def test_empty_list(self):
        assert ptl.looks_digital([]) is False

    def test_all_blank_pages_scanned(self):
        # 스캔 PDF: 텍스트 레이어 없음(빈 문자열) → 디지털 아님
        assert ptl.looks_digital(["", "", ""]) is False

    def test_sparse_pages_not_digital(self):
        # 페이지마다 글자 몇 개뿐(쪽번호 수준) → 디지털 아님
        assert ptl.looks_digital(["12", "3", ""]) is False

    def test_real_text_is_digital(self):
        page = "가" * 200
        assert ptl.looks_digital([page, page]) is True

    def test_majority_text_pages_digital(self):
        page = "본문내용" * 30
        # 3p 중 2p가 텍스트 → 비율 0.66 ≥ 0.5 → 디지털
        assert ptl.looks_digital([page, page, ""]) is True

    def test_minority_text_pages_not_digital(self):
        page = "본문내용" * 30
        # 4p 중 1p만 텍스트 → 0.25 < 0.5 → 이미지 위주 스캔 취급 → OCR
        assert ptl.looks_digital([page, "", "", ""]) is False


# ── 백엔드 감지 ────────────────────────────────────────────────────────────────

class TestBackend:
    def test_backend_value(self):
        assert ptl.backend() in (None, "fitz", "pdftotext")

    def test_is_available_matches_backend(self):
        assert ptl.is_available() == (ptl.backend() is not None)


# ── 추출 + 판별 (실제 PDF 생성; fitz 없으면 skip) ─────────────────────────────

@pytest.fixture
def digital_pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    p = tmp_path / "digital.pdf"
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        # 기본 폰트(Helvetica)로 확실히 렌더되는 ASCII 본문 — 텍스트 레이어 보장
        page.insert_text(
            (72, 72),
            ("bold leadership moment of truth " * 12) + f" page{i + 1}",
            fontsize=11,
        )
    doc.save(str(p))
    doc.close()
    return p


@pytest.fixture
def blank_pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    p = tmp_path / "blank.pdf"
    doc = fitz.open()
    for _ in range(3):
        doc.new_page()  # 텍스트 없이 빈 페이지(스캔 PDF 모사)
    doc.save(str(p))
    doc.close()
    return p


class TestAnalyze:
    def test_digital_pdf_detected(self, digital_pdf):
        if not ptl.is_available():
            pytest.skip("추출 백엔드(PyMuPDF/poppler) 없음")
        info = ptl.analyze(digital_pdf)
        assert info is not None
        assert info.is_digital is True
        assert info.total_pages == 3
        assert info.text_pages == 3
        assert "leadership" in "".join(info.pages)

    def test_blank_pdf_not_digital(self, blank_pdf):
        if not ptl.is_available():
            pytest.skip("추출 백엔드(PyMuPDF/poppler) 없음")
        info = ptl.analyze(blank_pdf)
        assert info is not None
        assert info.is_digital is False

    def test_extract_pages_count(self, digital_pdf):
        if not ptl.is_available():
            pytest.skip("추출 백엔드(PyMuPDF/poppler) 없음")
        pages = ptl.extract_pages(digital_pdf)
        assert len(pages) == 3
        assert all(isinstance(pg, str) for pg in pages)


# ── 내장 글자 품질 (검색 가능 스캔본의 깨진 OCR) ──────────────────────────────

class TestGarbled:
    # 01_입력 '길을묻는그대에게' 80쪽 실물 발췌
    BAD = ("의미 있는 여형은 성공적안 경훈이나 성공적인 사회가 제공하는 보상핑 없이는 이루어징 수 없다. "
           "낭녀가 ^71 空로 여행한 곡때기로부터 경空이나 사회| HI 우검고 동아왓冬 때 경혼이나 그 사회는")
    GOOD = ("그리스도인은 그 앞에 거룩하고 흠이 없게 하시려고 부르심을 받는다. 주님은 거룩하신 소명으로 "
            "우리를 부르셨다. 하나님은 뛰어난 은사를 받기보다 예수님을 닮기를 원하신다. 전제(專制)")

    def test_broken_ocr_page_flagged(self):
        assert ptl.is_garbled(self.BAD)

    def test_clean_page_with_parenthesized_hanja_passes(self):
        assert not ptl.is_garbled(self.GOOD)

    def test_short_or_english_page_not_judged(self):
        assert ptl.garble_score("Chapter 1 Introduction 13") is None

    def test_noise_and_missing_spaces_flagged_as_noisy(self):
        # 01_입력 '악당의명언' 30쪽 발췌 — 글자는 맞고 잡음 기호·띄어쓰기만 문제
        noisy = ("개인 026 I 잘되는놈도 < 언젠가는자신의 I 본성 때문에 5 꼬꾸라질것이다. ■ _ ^ "
                 "성취란그맛을보기위해하는것이다 돈이 아니라 • 러너스하이 _ _ 몰입을 할 때")
        assert ptl.is_noisy(noisy) and not ptl.is_garbled(noisy)
        assert not ptl.is_noisy(self.GOOD)


# ── 띄어쓰기를 잃은 텍스트층 — 글자 간격으로 복원 ─────────────────────────────

class TestRespace:
    WORDS = "일본이 장기 불황으로 고전하는 동안 교토 지역 첨단 기업들이 세계 시장을 개척했다".split()

    def _unspaced_pdf(self, path, words=None):
        """공백 글자 없이 글자를 한 자씩 찍고 어절 사이만 띄운 PDF — SERI 교토식 경영 PDF 와 같은 구조."""
        fitz = pytest.importorskip("fitz")
        doc = fitz.open()
        page = doc.new_page()
        for row in range(4):
            x, y = 50, 100 + row * 30
            for w in words or self.WORDS:
                for ch in w:
                    page.insert_text((x, y), ch, fontname="korea", fontsize=12)
                    x += 12
                x += 6
        doc.save(str(path))
        return path

    def test_unspaced_page_detected(self):
        assert ptl.is_unspaced("".join(self.WORDS) * 3)
        assert not ptl.is_unspaced(" ".join(self.WORDS) * 3)

    def test_respace_inserts_spaces_at_wide_gaps(self):
        def ch(c, x0):
            return {"c": c, "bbox": (x0, 0, x0 + 10, 10)}
        raw = {"blocks": [{"type": 0, "bbox": (0, 0, 100, 10), "lines": [{"spans": [{"size": 10, "chars": [
            ch("교", 0), ch("토", 10), ch("식", 26), ch("경", 36), ch("영", 46)]}]}]}]}
        assert ptl.respace_rawdict(raw).strip() == "교토 식경영"

    def test_letter_spaced_title_not_split(self):
        # 실측: 아메바 경영 석사논문 표지 '지도교수:이경구'·'김찬휴' — 자간을 넓힌 제목은 한 자씩 쪼개지 않는다
        chars = [{"c": c, "bbox": (i * 20, 0, i * 20 + 10, 10)} for i, c in enumerate("김찬휴")]
        raw = {"blocks": [{"type": 0, "bbox": (0, 0, 60, 10),
                           "lines": [{"spans": [{"size": 10, "chars": chars}]}]}]}
        assert ptl.respace_rawdict(raw).strip() == "김찬휴"

    def test_extract_restores_spaces(self, tmp_path):
        if ptl.backend() != "fitz":
            pytest.skip("PyMuPDF 필요")
        pages = ptl.extract_pages(self._unspaced_pdf(tmp_path / "unspaced.pdf"))
        assert " ".join(self.WORDS) in pages[0]
        info = ptl.analyze(tmp_path / "unspaced.pdf")
        assert info.noisy_pages == []


# ── _finalize_digital (ocr_gradio) 통합 — blank/본문 분기 ─────────────────────

class TestFinalizeDigital:
    def test_blank_page_marked(self):
        # ocr_gradio import 는 gradio 의존 → 없으면 skip
        try:
            import ocr_gradio
        except Exception:
            pytest.skip("ocr_gradio(gradio) import 불가")
        pr = ocr_gradio._finalize_digital("  \n ", 1)
        assert pr.is_blank is True
        assert pr.engine.value == "pdf_text"

    def test_content_page_processed(self):
        try:
            import ocr_gradio
        except Exception:
            pytest.skip("ocr_gradio(gradio) import 불가")
        pr = ocr_gradio._finalize_digital("이것은 충분히 긴 본문 문장입니다. " * 3, 2)
        assert pr.is_blank is False
        assert pr.engine.value == "pdf_text"
        assert "본문" in pr.corrected_text
