"""API 키 없는 사용자 처리 흐름 테스트"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import config, claude_vision, writer
from core.types import FileResult, OutputFormat, PageResult, OCREngine


def _make_page(page_num=1, confidence=85.0, needs_review=False, review_reason="") -> PageResult:
    return PageResult(
        page_num=page_num,
        raw_text="텍스트", corrected_text="텍스트",
        engine=OCREngine.TESSERACT, confidence=confidence,
        needs_review=needs_review, review_reason=review_reason,
    )


def _make_result(pages: list[PageResult]) -> FileResult:
    r = FileResult(source_path=Path("test.png"), output_path=None)
    r.pages = pages
    return r


# ── config 모듈 ───────────────────────────────────────────────────────────────

class TestConfig:
    def test_load_returns_dict(self):
        cfg = config.load()
        assert isinstance(cfg, dict)
        assert "anthropic_api_key" in cfg

    def test_get_api_key_string(self):
        key = config.get_api_key()
        assert isinstance(key, str)

    def test_has_api_key_bool(self):
        assert isinstance(config.has_api_key(), bool)

    def test_save_and_load(self, tmp_path, monkeypatch):
        fake_config = tmp_path / "config.json"
        monkeypatch.setattr(config, "_CONFIG_FILE", fake_config)
        config.save({"default_format": "md"})
        loaded = config.load()
        assert loaded["default_format"] == "md"

    def test_set_and_clear_api_key(self, tmp_path, monkeypatch):
        # 환경변수 있으면 config 파일보다 우선하므로 먼저 삭제
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        fake_config = tmp_path / "config.json"
        monkeypatch.setattr(config, "_CONFIG_FILE", fake_config)
        config.set_api_key("sk-ant-test-key-1234")
        assert config.get_api_key() == "sk-ant-test-key-1234"
        config.clear_api_key()
        assert config.get_api_key() == ""

    def test_env_var_takes_priority(self, tmp_path, monkeypatch):
        fake_config = tmp_path / "config.json"
        monkeypatch.setattr(config, "_CONFIG_FILE", fake_config)
        config.set_api_key("file-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        assert config.get_api_key() == "env-key"

    def test_show_masks_api_key(self, tmp_path, monkeypatch):
        fake_config = tmp_path / "config.json"
        monkeypatch.setattr(config, "_CONFIG_FILE", fake_config)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        test_key = "sk-ant-api03-abcdefghijklmn-1234"
        config.set_api_key(test_key)
        shown = config.show()
        masked = shown["anthropic_api_key"]
        # 앞 8자 노출, 중간 마스킹, 끝 4자 노출
        assert masked.startswith(test_key[:8])
        assert masked.endswith(test_key[-4:])
        assert "..." in masked
        # 중간 부분은 노출되지 않아야 함
        assert "abcdefghijklmn" not in masked


# ── API 키 없을 때 claude_vision 동작 ────────────────────────────────────────

class TestNoApiKey:
    def test_is_available_false_without_key(self, tmp_path, monkeypatch):
        fake_config = tmp_path / "config.json"
        monkeypatch.setattr(config, "_CONFIG_FILE", fake_config)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        claude_vision._client = None
        assert claude_vision.is_available() is False

    def test_run_raises_without_key(self, tmp_path, monkeypatch):
        from PIL import Image
        fake_config = tmp_path / "config.json"
        monkeypatch.setattr(config, "_CONFIG_FILE", fake_config)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        claude_vision._client = None
        img = Image.new("RGB", (100, 100), "white")
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            claude_vision.run(img)


# ── 검토 필요 페이지 마킹 ─────────────────────────────────────────────────────

class TestReviewMarking:
    def test_review_flag_propagates_to_output_txt(self, tmp_path):
        page = _make_page(
            confidence=45.0, needs_review=True,
            review_reason="OCR 신뢰도 낮음 (45.0% / 기준 70.0%)"
        )
        result = _make_result([page])
        out = writer.write(result, tmp_path, OutputFormat.TXT)
        content = out.read_text(encoding="utf-8")
        assert "검토필요" in content or "검토 필요" in content
        assert "45.0%" in content

    def test_review_flag_propagates_to_output_md(self, tmp_path):
        page = _make_page(
            confidence=45.0, needs_review=True,
            review_reason="OCR 신뢰도 낮음 (45.0%)"
        )
        result = _make_result([page])
        out = writer.write(result, tmp_path, OutputFormat.MD)
        content = out.read_text(encoding="utf-8")
        assert "검토 필요" in content

    def test_summary_section_in_txt(self, tmp_path):
        pages = [
            _make_page(1, 90.0),
            _make_page(2, 40.0, needs_review=True, review_reason="신뢰도 낮음"),
            _make_page(3, 35.0, needs_review=True, review_reason="자모 분리"),
        ]
        result = _make_result(pages)
        out = writer.write(result, tmp_path, OutputFormat.TXT)
        content = out.read_text(encoding="utf-8")
        # 상단 요약 섹션 존재 확인
        assert "OCR 품질 검토 필요" in content
        assert "2페이지" in content
        assert "python ocr_tool.py" in content  # 설정 안내 포함

    def test_summary_section_in_md(self, tmp_path):
        pages = [
            _make_page(1, 40.0, needs_review=True, review_reason="신뢰도 낮음"),
        ]
        result = _make_result(pages)
        out = writer.write(result, tmp_path, OutputFormat.MD)
        content = out.read_text(encoding="utf-8")
        assert "OCR 품질 검토 필요" in content
        assert "setup" in content  # 설정 명령 안내

    def test_no_review_no_summary(self, tmp_path):
        page = _make_page(1, 90.0)  # 정상 신뢰도
        result = _make_result([page])
        out = writer.write(result, tmp_path, OutputFormat.TXT)
        content = out.read_text(encoding="utf-8")
        assert "OCR 품질 검토 필요" not in content

    def test_normal_page_not_marked(self, tmp_path):
        page = _make_page(1, 90.0)
        result = _make_result([page])
        out = writer.write(result, tmp_path, OutputFormat.TXT)
        content = out.read_text(encoding="utf-8")
        assert "검토필요" not in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
