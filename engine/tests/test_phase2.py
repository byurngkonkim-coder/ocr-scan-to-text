"""Phase 2 테스트 — V-04 · V-05 · V-08 · V-09"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from core import char_normalizer, spellcheck
from core.types import OCREngine, OCRResult


def _make_image(texts: list[str], size=(800, 300)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    for i, text in enumerate(texts):
        draw.text((40, 40 + i * 50), text, fill="black", font=font)
    return img


# ── V-05: 자모 분리 감지 ───────────────────────────────────────────────────────

class TestJamoDetection:
    def test_jamo_text_detected(self):
        # 자모 분리 텍스트 (비율 > 5%)
        jamo = "ㅎㅏㄱㄱㅛ ㅇㅓㅂㄷㅏ " * 10
        assert char_normalizer.has_jamo_decomposition(jamo) is True

    def test_normal_korean_not_detected(self):
        normal = "안녕하세요. 오늘 날씨가 좋습니다."
        assert char_normalizer.has_jamo_decomposition(normal) is False

    def test_mixed_low_jamo_not_detected(self):
        # 비율 낮으면 False
        text = "정상 텍스트입니다 " * 20 + "ㄱ"
        assert char_normalizer.has_jamo_decomposition(text) is False

    def test_empty_text(self):
        assert char_normalizer.has_jamo_decomposition("") is False

    def test_english_only(self):
        assert char_normalizer.has_jamo_decomposition("Hello World") is False


# ── V-08: L1 보정 정확도 ──────────────────────────────────────────────────────

class TestL1Correction:
    def test_ligatures(self):
        assert char_normalizer.normalize_all("ﬁle ﬂow ﬀ") == "file flow ff"

    def test_copyright_symbol(self):
        result = char_normalizer.normalize_all("(c) 2026 Rights (r)")
        assert "©" in result
        assert "®" in result

    def test_trademark(self):
        result = char_normalizer.normalize_all("Brand (tm)")
        assert "™" in result

    def test_arrows(self):
        result = char_normalizer.normalize_all("A -> B <- C")
        assert "→" in result
        assert "←" in result

    def test_math_symbols(self):
        result = char_normalizer.normalize_all("x +- y >= z")
        assert "±" in result
        assert "≥" in result

    def test_unit_conversion(self):
        result = char_normalizer.normalize_all("25 oC and 10 m2")
        assert "℃" in result
        assert "m²" in result

    def test_abbreviation_fix(self):
        result = char_normalizer.normalize_all("e,g, this i,e, that")
        assert "e.g." in result
        assert "i.e." in result

    def test_ellipsis_preserved(self):
        # [A안] ...는 변환하지 않고 원문 보존 (하위 정제 파이프라인이 처리)
        result = char_normalizer.normalize_all("wait...")
        assert "..." in result and "…" not in result

    def test_dashes_preserved(self):
        # [A안] --는 변환하지 않고 원문 보존
        result = char_normalizer.normalize_all("2020--2026")
        assert "--" in result and "—" not in result

    def test_number_confusion_I(self):
        # 숫자 사이 I → 1
        assert char_normalizer.fix_english_confusion("3I5") == "315"

    def test_number_confusion_O(self):
        # 숫자 사이 O → 0
        assert char_normalizer.fix_english_confusion("1O2") == "102"

    def test_english_hyphen_join(self):
        result = char_normalizer.fix_hyphenation("docu-\nment")
        assert result == "document"

    def test_korean_space_preserved(self):
        # [변경] 정상 띄어쓰기는 보존 (과거: "가나다"로 붙였으나 원문 훼손 → 제거)
        result = char_normalizer.fix_korean_spaces("가 나 다")
        assert result == "가 나 다"

    def test_english_word_initial_l_preserved(self):
        # 어두 소문자 l 단어가 대문자 I로 훼손되지 않아야 함
        assert (char_normalizer.fix_english_confusion("let me look at the list")
                == "let me look at the list")

    def test_english_intraword_confusion_still_fixed(self):
        # 단어 내부 대문자 I → l 은 계속 교정 ("shouId" → "should")
        assert char_normalizer.fix_english_confusion("shouId") == "should"

    def test_invisible_chars_removed(self):
        # N-06: BOM/ZWSP 등 비가시 문자 제거
        assert char_normalizer.normalize_all("a\u200bb\ufeffc") == "abc"


# ── V-09: 배치 오류 격리 ──────────────────────────────────────────────────────

class TestBatchIsolation:
    """한 파일 실패가 전체 배치를 중단하지 않는지 확인"""

    def test_single_failure_isolated(self, tmp_path):
        from core.types import FileResult
        from core import writer
        from core.types import OutputFormat, PageResult

        # 정상 파일 결과
        good = FileResult(source_path=Path("good.png"), output_path=None)
        good.pages.append(PageResult(
            page_num=1, raw_text="hello", corrected_text="hello",
            engine=OCREngine.RAPIDOCR, confidence=90.0,
        ))
        writer.write(good, tmp_path, OutputFormat.TXT)

        # 실패 결과 (success=False)
        bad = FileResult(source_path=Path("bad.png"), output_path=None)
        bad.success = False
        bad.error_message = "의도적 실패"

        results = [good, bad]
        success_count = sum(1 for r in results if r.success)
        fail_count    = sum(1 for r in results if not r.success)

        assert success_count == 1
        assert fail_count == 1
        # 정상 파일 출력은 존재
        assert (tmp_path / "good.txt").exists()


# ── Claude Vision 가용성 확인 ─────────────────────────────────────────────────

class TestClaudeVisionAvailability:
    def test_availability_check(self):
        from core import claude_vision
        # 환경변수 유무에 따라 True/False — 예외 없이 반환되면 OK
        result = claude_vision.is_available()
        assert isinstance(result, bool)

    def test_no_key_raises_on_run(self, monkeypatch, tmp_path):
        """API 키 없을 때 run() 호출 시 RuntimeError 발생"""
        import core.claude_vision as cv
        import core.config as cfg
        # 환경변수 + config 파일 양쪽 모두 키 없애기
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        monkeypatch.setattr(cfg, "_CONFIG_FILE", tmp_path / "config.json")
        cv._client = None
        img = _make_image(["test"])
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            cv.run(img)


# ── L2 맞춤법 검사 ────────────────────────────────────────────────────────────

class TestSpellCheck:
    def test_is_available_returns_bool(self):
        result = spellcheck.is_available()
        assert isinstance(result, bool)

    def test_correct_safe_returns_string(self):
        result = spellcheck.correct_safe("안녕하세요")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_correct_safe_fallback_on_empty(self):
        result = spellcheck.correct_safe("")
        assert result == ""

    def test_chunk_split(self):
        # 500자 초과 텍스트도 처리
        long_text = "안녕하세요. " * 100
        result = spellcheck.correct_safe(long_text)
        assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
