"""Phase 4 테스트 — L3 LLM 보정 · 비용 제어 · Gradio UI 구조"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import llm_corrector, config


# ── llm_corrector 단위 테스트 ──────────────────────────────────────────────────

class TestLLMCorrectorUnit:
    def setup_method(self):
        llm_corrector.reset_call_count()

    def test_empty_text_passthrough(self):
        result = llm_corrector.correct("")
        assert result == ""

    def test_whitespace_only_passthrough(self):
        result = llm_corrector.correct("   \n  ")
        assert result.strip() == ""

    def test_no_key_returns_original(self, monkeypatch, tmp_path):
        fake_cfg = tmp_path / "config.json"
        monkeypatch.setattr(config, "_CONFIG_FILE", fake_cfg)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        llm_corrector._client = None
        text = "원본 텍스트"
        result = llm_corrector.correct(text)
        assert result == text

    def test_is_available_false_without_key(self, monkeypatch, tmp_path):
        fake_cfg = tmp_path / "config.json"
        monkeypatch.setattr(config, "_CONFIG_FILE", fake_cfg)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        llm_corrector._client = None
        assert llm_corrector.is_available() is False

    def test_is_available_true_with_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        assert llm_corrector.is_available() is True

    def test_call_count_starts_zero(self):
        assert llm_corrector.get_call_count() == 0

    def test_reset_call_count(self):
        llm_corrector._call_count = 5
        llm_corrector.reset_call_count()
        assert llm_corrector.get_call_count() == 0


# ── 비용 제어 (budget) ────────────────────────────────────────────────────────

class TestBudgetControl:
    def setup_method(self):
        llm_corrector.reset_call_count()

    def test_budget_zero_means_unlimited(self, monkeypatch, tmp_path):
        """budget=0 이면 횟수 제한 없음 (API 키 없으면 원본 반환이므로 count=0)."""
        fake_cfg = tmp_path / "config.json"
        monkeypatch.setattr(config, "_CONFIG_FILE", fake_cfg)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        llm_corrector._client = None
        llm_corrector.correct("텍스트", budget=0)
        assert llm_corrector.get_call_count() == 0  # 키 없으면 호출 안 됨

    def test_budget_exceeded_returns_original(self, monkeypatch, tmp_path):
        """호출 횟수가 budget 초과 시 원본 반환."""
        fake_cfg = tmp_path / "config.json"
        monkeypatch.setattr(config, "_CONFIG_FILE", fake_cfg)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        llm_corrector._client = None
        llm_corrector._call_count = 5  # 이미 5회 사용

        # 실제 API 호출 mock
        def mock_call(chunk, api_key):
            return "CHANGED"
        monkeypatch.setattr(llm_corrector, "_call_api", mock_call)

        result = llm_corrector.correct("원본 텍스트", budget=5)
        # budget 초과 → 원본 그대로
        assert result == "원본 텍스트"


# ── 청크 분할 ────────────────────────────────────────────────────────────────

class TestChunkSplit:
    def test_short_text_single_chunk(self):
        chunks = llm_corrector._split_chunks("짧은 텍스트", 1500)
        assert len(chunks) == 1

    def test_long_text_splits(self):
        long = "안녕하세요. " * 300  # ~2400자
        chunks = llm_corrector._split_chunks(long, 1500)
        assert len(chunks) >= 2

    def test_placeholder_not_split(self):
        text = "텍스트\n[그림 1: 이미지 감지됨]\n더 많은 텍스트"
        chunks = llm_corrector._split_chunks(text, 1500)
        # 플레이스홀더 행은 독립 청크
        placeholder_chunks = [c for c in chunks if "[그림" in c]
        assert len(placeholder_chunks) == 1

    def test_empty_text_returns_one_chunk(self):
        chunks = llm_corrector._split_chunks("", 1500)
        assert len(chunks) == 1

    def test_chunks_cover_all_content(self):
        original = "줄1\n줄2\n줄3\n줄4\n줄5"
        chunks = llm_corrector._split_chunks(original, 10)
        rejoined = "".join(chunks)
        for line in ["줄1", "줄2", "줄3", "줄4", "줄5"]:
            assert line in rejoined


# ── LLM 보정 mock 통합 테스트 ────────────────────────────────────────────────

class TestLLMCorrectionMock:
    def setup_method(self):
        llm_corrector.reset_call_count()

    def test_correction_applied_when_key_present(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        llm_corrector._client = None

        def mock_call(chunk, api_key):
            return chunk.replace("오타", "교정됨")
        monkeypatch.setattr(llm_corrector, "_call_api", mock_call)

        result = llm_corrector.correct("이것은 오타 텍스트입니다.")
        assert "교정됨" in result
        assert llm_corrector.get_call_count() == 1

    def test_api_error_returns_original(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        llm_corrector._client = None

        def mock_call_fail(chunk, api_key):
            raise RuntimeError("API 오류")
        monkeypatch.setattr(llm_corrector, "_call_api", mock_call_fail)

        original = "원본 텍스트입니다."
        result = llm_corrector.correct(original)
        assert result == original  # 오류 시 원본 반환

    def test_multiple_chunks_all_processed(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        llm_corrector._client = None
        call_log = []

        def mock_call(chunk, api_key):
            call_log.append(chunk)
            return chunk + "[처리됨]"
        monkeypatch.setattr(llm_corrector, "_call_api", mock_call)

        long_text = "안녕. " * 400  # 여러 청크
        result = llm_corrector.correct(long_text, budget=0)
        assert len(call_log) >= 2
        assert "[처리됨]" in result


# ── Gradio UI 구조 테스트 ─────────────────────────────────────────────────────

class TestGradioUIStructure:
    def test_gradio_importable(self):
        try:
            import gradio  # noqa: F401
            gradio_ok = True
        except ImportError:
            gradio_ok = False
        # 미설치 시 skip
        if not gradio_ok:
            pytest.skip("gradio 미설치 — pip install gradio")

    def test_build_ui_returns_blocks(self):
        try:
            import gradio as gr
        except ImportError:
            pytest.skip("gradio 미설치")
        from ocr_gradio import build_ui
        demo = build_ui()
        assert isinstance(demo, gr.Blocks)

    def test_process_preview_no_file(self):
        """파일 없이 호출 시 오류 메시지 반환"""
        from ocr_gradio import process_preview
        # generator 이므로 첫 yield만 확인
        gen = process_preview(
            None, "kor+eng", "txt",
            False, True, True, False, 0, "",
            "⚡ Tesseract 우선 (Tesseract → Vision 폴백)",
        )
        result = next(gen)
        # (progress_html, status, preview, download, continue_html, images_state)
        status = result[1]
        assert "파일" in status

    def test_process_full_invalid_file(self, tmp_path):
        """지원 안 되는 파일 형식으로 호출 시 오류 포함"""
        from ocr_gradio import process_full

        class FakeFile:
            name = str(tmp_path / "fake.xyz")

        (tmp_path / "fake.xyz").write_text("not an image")
        gen = process_full(
            FakeFile(), "kor+eng", "txt",
            False, True, True, False, 0, "",
            "⚡ Tesseract 우선 (Tesseract → Vision 폴백)",
        )
        results = list(gen)
        # 마지막 status에 오류 포함 여부
        last_status = results[-1][1]
        assert "오류" in last_status or "❌" in last_status or last_status == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
