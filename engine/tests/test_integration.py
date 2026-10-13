"""
통합 테스트 — Phase 1~4 + 신규 기능 전체 파이프라인

커버 범위:
  - 빈 페이지 감지 (blank_detector)
  - 엔진별 후처리 체인 (postprocessor.process_full)
  - 저장 관리 (storage_manager)
  - 전체 OCR 파이프라인 (이미지 → PageResult)
  - writer 출력 (빈 페이지 처리 포함)
  - 설정 관리 (config)
"""
import sys
import json
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ── 테스트용 이미지 팩토리 ─────────────────────────────────────────────────────

def _font(size=22):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _make_text_image(lines: list[str], size=(800, 600)) -> Image.Image:
    img  = Image.new("RGB", size, (250, 250, 250))
    draw = ImageDraw.Draw(img)
    font = _font(22)
    for i, line in enumerate(lines):
        draw.text((60, 60 + i * 55), line, fill="black", font=font)
    return img


def _make_blank_image(size=(800, 1000)) -> Image.Image:
    return Image.new("RGB", size, "white")


def _make_noisy_blank(size=(800, 1000), dots=150) -> Image.Image:
    """스캔 잡음만 있는 실질적 빈 페이지."""
    arr = np.full((*size[::-1], 3), 255, dtype=np.uint8)
    rng = np.random.default_rng(0)
    idx = rng.choice(size[0] * size[1], dots, replace=False)
    arr.reshape(-1, 3)[idx] = [210, 210, 210]
    return Image.fromarray(arr)


# ══════════════════════════════════════════════════════════════════════════════
# 1. 빈 페이지 감지 통합 테스트
# ══════════════════════════════════════════════════════════════════════════════

class TestBlankDetectorIntegration:

    def test_pure_white_is_blank(self):
        from core.blank_detector import check
        r = check(_make_blank_image(), fast=True)
        assert r.is_blank is True
        assert r.stage == 1

    def test_noisy_blank_is_blank(self):
        from core.blank_detector import check
        r = check(_make_noisy_blank(), fast=True)
        assert r.is_blank is True

    def test_text_page_not_blank(self):
        from core.blank_detector import check
        img = _make_text_image([
            "인간은 얼마나 뛰어난 작품인가!",
            "그 이성은 얼마나 고귀한가!",
            "Hello World Test Document",
            "OCR 통합 테스트 문서입니다.",
        ])
        r = check(img, fast=True)
        assert r.is_blank is False

    def test_blank_result_fields(self):
        from core.blank_detector import check, BlankResult
        r = check(_make_blank_image())
        assert isinstance(r, BlankResult)
        assert isinstance(r.white_ratio, float)
        assert isinstance(r.reason, str)
        assert 0.0 <= r.white_ratio <= 1.0

    def test_blank_skipped_in_page_result(self, tmp_path, monkeypatch):
        """빈 페이지가 PageResult.is_blank=True로 반환되는지 확인."""
        import core.config as cfg
        monkeypatch.setattr(cfg, "_CONFIG_FILE", tmp_path / "cfg.json")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        # ocr_gradio._process_page를 직접 테스트하기 어려우므로
        # blank_detector.check 후 PageResult 직접 생성 시뮬레이션
        from core.blank_detector import check
        from core.types import PageResult, OCREngine
        img = _make_blank_image()
        blank = check(img, fast=True)
        assert blank.is_blank

        page = PageResult(
            page_num=1, raw_text="", corrected_text="",
            engine=OCREngine.TESSERACT, confidence=0.0,
            is_blank=True, blank_reason=blank.reason,
        )
        assert page.is_blank is True
        assert page.corrected_text == ""


# ══════════════════════════════════════════════════════════════════════════════
# 2. 엔진별 후처리 체인 통합 테스트
# ══════════════════════════════════════════════════════════════════════════════

class TestPostprocessorChainIntegration:

    def test_tesseract_chain_removes_margin_noise(self):
        from core.postprocessor import process_full
        raw = (
            "00 10: *인간은 얼마나 뛰어난 작품인가!\n"
            "21200 2 0 그러나 하나님을 버리고 성경을 멀리함으로써\n"
            "| 0 00000 진정으로 인간다운 삶이 무엇인지\n"
        )
        result = process_full(raw, engine="tesseract")
        # 마진 노이즈 제거됨
        assert "00 10" not in result
        assert "21200" not in result
        assert "| 0 00000" not in result
        # 실제 내용은 남아 있음
        assert "인간은" in result

    def test_vision_chain_preserves_clean_text(self):
        from core.postprocessor import process_full
        raw = (
            '"인간은 얼마나 뛰어난 작품인가! 그 이성은 얼마나 고귀한가!\n'
            '또 그 능력과 재주는 얼마나 무한한가!"\n\n'
            "햄릿의 말이 맞다. 우리 인간성이 지닌\n"
            "엄청난 영광을 당신은 깨닫고 있는가?\n"
        )
        result = process_full(raw, engine="claude_vision")
        # 깨끗한 Vision 결과는 그대로 보존
        assert "인간은" in result
        assert "햄릿의 말이 맞다" in result

    def test_vision_chain_joins_broken_lines(self):
        from core.postprocessor import process_full
        # 줄바꿈으로 잘린 단락
        raw = (
            "그러나 하나님을 버리고 성경을 멀리함으로써, 진정으로 인간다운\n"
            "삶이 무엇인지에 대한 개념을 상실한 오늘날 인간은 어떻게 될\n"
            "것인가?\n"
        )
        result = process_full(raw, engine="claude_vision")
        # 단락으로 이어짐 (줄바꿈이 줄어들어야 함)
        assert len(result.splitlines()) < len(raw.splitlines())

    def test_for_preview_strips_structural_markers(self):
        from core.postprocessor import process_full, for_preview
        raw = (
            "[페이지 1]\n"
            "────────────────────\n"
            "인간은 얼마나 뛰어난 작품인가!\n"
            "[그림 1: 이미지 감지됨]\n"
        )
        clean = process_full(raw, engine="tesseract")
        preview = for_preview(clean)
        assert "[페이지 1]" not in preview
        assert "[그림 1" not in preview
        assert "─" not in preview
        assert "인간은" in preview

    def test_l1_applied_in_tesseract_chain(self):
        """L1 보정(char_normalizer)이 한국어 본문에 올바르게 적용되는지 확인."""
        from core.char_normalizer import normalize_all
        from core.postprocessor import process_full
        # 한국어와 함께 특수기호가 포함된 실제 OCR 출력 패턴
        raw = "인간은 (c) 2026 저작권자 (r) 입니다. fi fl -> <-"
        normed = normalize_all(raw)  # L1 적용
        assert "©" in normed         # L1에서 변환됨
        assert "®" in normed
        assert "→" in normed
        # process_full은 L1 이후에 적용 (한국어 줄로 인식되어 보존됨)
        result = process_full(normed, engine="tesseract")
        assert "인간은" in result     # 한국어 본문 보존
        assert "©" in result or "저작권자" in result  # 내용 보존

    def test_tesseract_chain_preserves_non_korean_content(self):
        """마진 완화 후: 영어·숫자·표·짧은 제목이 삭제되지 않아야 한다."""
        from core.postprocessor import process_full
        # 영어 본문
        eng = process_full("This is an English document line.", engine="tesseract")
        assert "English document" in eng
        # 표·숫자 데이터
        tbl = process_full("표: 매출 1,250만원 / 비용 480만원", engine="tesseract")
        assert "1,250만원" in tbl and "480만원" in tbl
        # 연도가 붙은 숫자 토큰 보존
        yr = process_full("2026년 3분기 실적", engine="tesseract")
        assert "2026년" in yr
        # 마진 노이즈는 여전히 제거 (회귀 방지)
        mg = process_full("00 10: *인간은 얼마나 뛰어난 작품인가!", engine="tesseract")
        assert "00 10" not in mg and "인간은" in mg


# ══════════════════════════════════════════════════════════════════════════════
# 3. 저장 관리 통합 테스트
# ══════════════════════════════════════════════════════════════════════════════

class TestStorageManagerIntegration:

    def _patch_storage(self, tmp_path, monkeypatch):
        import core.storage_manager as sm
        monkeypatch.setattr(sm, "_BASE",    tmp_path / "storage")
        monkeypatch.setattr(sm, "_UPLOADS", tmp_path / "storage" / "uploads")
        monkeypatch.setattr(sm, "_RESULTS", tmp_path / "storage" / "results")
        monkeypatch.setattr(sm, "_HISTORY", tmp_path / "storage" / "history.json")
        return sm

    def test_save_result_creates_file(self, tmp_path, monkeypatch):
        sm = self._patch_storage(tmp_path, monkeypatch)
        src = tmp_path / "output.txt"
        src.write_text("추출 결과입니다.", encoding="utf-8")
        saved = sm.save_result(src, "원본파일.pdf")
        assert saved.exists()
        assert saved.read_text(encoding="utf-8") == "추출 결과입니다."

    def test_save_original_copies_file(self, tmp_path, monkeypatch):
        sm = self._patch_storage(tmp_path, monkeypatch)
        src = tmp_path / "원본.pdf"
        src.write_bytes(b"%PDF fake content")
        saved = sm.save_original(src, "원본.pdf")
        assert saved.exists()
        assert "원본" in saved.name or ".pdf" in saved.name

    def test_record_and_load_history(self, tmp_path, monkeypatch):
        sm = self._patch_storage(tmp_path, monkeypatch)
        result_file = tmp_path / "result.txt"
        result_file.write_text("결과", encoding="utf-8")

        entry_id = sm.record(
            original_name="테스트.pdf",
            result_path=result_file,
            upload_path=None,
            pages_total=10, pages_ocr=9, pages_blank=1,
            vision_pages=5, engine_mode="vision_first",
            file_size_bytes=1024 * 50,
        )
        history = sm.load_history()
        assert len(history) == 1
        assert history[0]["original_name"] == "테스트.pdf"
        assert history[0]["pages_total"] == 10
        assert history[0]["pages_blank"] == 1
        assert history[0]["engine_mode"] == "vision_first"
        assert history[0]["id"] == entry_id

    def test_delete_entry(self, tmp_path, monkeypatch):
        sm = self._patch_storage(tmp_path, monkeypatch)
        result_file = tmp_path / "result.txt"
        result_file.write_text("결과", encoding="utf-8")
        entry_id = sm.record(
            "파일.pdf", result_file, None,
            5, 5, 0, 2, "tesseract_first", 1024
        )
        assert len(sm.load_history()) == 1
        sm.delete_entry(entry_id, delete_files=False)
        assert len(sm.load_history()) == 0

    def test_delete_entry_with_file(self, tmp_path, monkeypatch):
        sm = self._patch_storage(tmp_path, monkeypatch)
        # 삭제할 결과 파일을 results 디렉터리 안에 생성
        sm.init()
        result_file = sm._RESULTS / "20260602_result.txt"
        result_file.write_text("결과", encoding="utf-8")
        entry_id = sm.record(
            "파일.pdf", result_file, None,
            3, 3, 0, 0, "tesseract_first", 512
        )
        assert result_file.exists()
        sm.delete_entry(entry_id, delete_files=True)
        assert not result_file.exists()

    def _record_with_raw_copy(self, sm, tmp_path, name):
        src = tmp_path / f"{name}.txt"
        src.write_text("정제본", encoding="utf-8")
        raw = tmp_path / f"{name}_raw.txt"
        raw.write_text("원문", encoding="utf-8")
        stored = sm.save_result(src, f"{name}.pdf")
        copy = sm.save_raw_copy(raw, stored)
        output = tmp_path / "02_결과" / f"20260927_120000_{name}.txt"   # 사용자용 확정본
        output.parent.mkdir(exist_ok=True)
        output.write_text("정제본", encoding="utf-8")
        entry_id = sm.record(f"{name}.pdf", stored, None, 1, 1, 0, 0, "test", 100,
                             output_path=output)
        return entry_id, stored, copy, output

    def test_raw_copy_stays_in_storage_when_result_is_in_results_folder(self, tmp_path, monkeypatch):
        # 결과는 02_결과 한 곳에만 — 정제 전 원문은 보관소 uploads/ 에 남는다
        sm = self._patch_storage(tmp_path, monkeypatch)
        output = tmp_path / "02_결과" / "20261013_120000_책.txt"
        output.parent.mkdir()
        output.write_text("정제본", encoding="utf-8")
        raw = tmp_path / "책_raw.txt"
        raw.write_text("원문", encoding="utf-8")
        copy = sm.save_raw_copy(raw, output)
        assert copy == sm._UPLOADS / "20261013_120000_책_원문.txt" and copy.exists()
        assert sorted(p.name for p in output.parent.iterdir()) == [output.name]
        entry_id = sm.record("책.pdf", output, None, 1, 1, 0, 0, "test", 100, output_path=output)
        sm.delete_entry(entry_id, delete_files=True)
        assert not output.exists() and not copy.exists()

    def test_delete_entry_removes_raw_copy_and_output(self, tmp_path, monkeypatch):
        sm = self._patch_storage(tmp_path, monkeypatch)
        entry_id, stored, copy, output = self._record_with_raw_copy(sm, tmp_path, "책")
        assert copy.name == f"{stored.stem}_원문{stored.suffix}" and copy.exists()
        sm.delete_entry(entry_id, delete_files=True)
        assert not stored.exists() and not copy.exists() and not output.exists()

    def test_delete_entry_keeps_files_without_delete_files(self, tmp_path, monkeypatch):
        sm = self._patch_storage(tmp_path, monkeypatch)
        entry_id, stored, copy, output = self._record_with_raw_copy(sm, tmp_path, "책")
        sm.delete_entry(entry_id, delete_files=False)
        assert stored.exists() and copy.exists() and output.exists()

    def test_clear_history_removes_raw_copies_and_outputs(self, tmp_path, monkeypatch):
        sm = self._patch_storage(tmp_path, monkeypatch)
        files = [self._record_with_raw_copy(sm, tmp_path, n)[1:] for n in ("가", "나")]
        sm.clear_history(delete_files=True)
        assert not any(p.exists() for group in files for p in group)

    def test_clear_history(self, tmp_path, monkeypatch):
        sm = self._patch_storage(tmp_path, monkeypatch)
        f = tmp_path / "r.txt"
        f.write_text("x", encoding="utf-8")
        for i in range(5):
            sm.record(f"파일{i}.pdf", f, None, 1, 1, 0, 0, "test", 100)
        assert len(sm.load_history()) == 5
        count = sm.clear_history()
        assert count == 5
        assert sm.load_history() == []

    def test_get_stats(self, tmp_path, monkeypatch):
        sm = self._patch_storage(tmp_path, monkeypatch)
        stats = sm.get_stats()
        assert "history_count" in stats
        assert "result_count" in stats
        assert "total_size_mb" in stats
        assert "storage_path" in stats

    def test_history_table_format(self, tmp_path, monkeypatch):
        sm = self._patch_storage(tmp_path, monkeypatch)
        table = sm.get_history_table()
        assert isinstance(table, list)
        assert len(table) >= 1
        assert len(table[0]) == 9  # 9컬럼

    def test_max_history_500(self, tmp_path, monkeypatch):
        sm = self._patch_storage(tmp_path, monkeypatch)
        f = tmp_path / "r.txt"
        f.write_text("x", encoding="utf-8")
        for i in range(sm.MAX_HISTORY + 10):
            sm.record(f"f{i}.pdf", f, None, 1, 1, 0, 0, "test", 100)
        assert len(sm.load_history()) == sm.MAX_HISTORY


# ══════════════════════════════════════════════════════════════════════════════
# 4. writer 빈 페이지 처리 통합 테스트
# ══════════════════════════════════════════════════════════════════════════════

class TestWriterBlankPageIntegration:

    def _make_result(self, pages_spec: list[dict]) -> object:
        from core.types import FileResult, PageResult, OCREngine
        result = FileResult(source_path=Path("test.pdf"), output_path=None)
        for spec in pages_spec:
            result.pages.append(PageResult(
                page_num=spec["num"],
                raw_text=spec.get("text", ""),
                corrected_text=spec.get("text", ""),
                engine=OCREngine.TESSERACT,
                confidence=spec.get("conf", 85.0),
                is_blank=spec.get("blank", False),
                blank_reason=spec.get("reason", ""),
            ))
        return result

    def test_blank_pages_excluded_from_txt(self, tmp_path):
        from core import writer
        from core.types import OutputFormat
        result = self._make_result([
            {"num": 1, "text": "첫 번째 페이지 내용"},
            {"num": 2, "blank": True, "reason": "공백 100%"},
            {"num": 3, "text": "세 번째 페이지 내용"},
        ])
        out = writer.write(result, tmp_path, OutputFormat.TXT)
        content = out.read_text(encoding="utf-8")
        assert "첫 번째 페이지 내용" in content
        assert "세 번째 페이지 내용" in content
        # 빈 페이지 내용은 없음
        assert "공백 100%" not in content or "건너뜀" in content

    def test_blank_summary_in_txt_header(self, tmp_path):
        from core import writer
        from core.types import OutputFormat
        result = self._make_result([
            {"num": 1, "text": "내용"},
            {"num": 2, "blank": True, "reason": "공백 100%"},
        ])
        out = writer.write(result, tmp_path, OutputFormat.TXT)
        content = out.read_text(encoding="utf-8")
        assert "빈 페이지" in content
        assert "건너뜀" in content

    def test_blank_info_in_md_header(self, tmp_path):
        from core import writer
        from core.types import OutputFormat
        result = self._make_result([
            {"num": 1, "text": "내용"},
            {"num": 2, "blank": True, "reason": "노이즈만"},
        ])
        out = writer.write(result, tmp_path, OutputFormat.MD)
        content = out.read_text(encoding="utf-8")
        assert "빈 페이지" in content

    def test_all_blank_file_outputs_gracefully(self, tmp_path):
        from core import writer
        from core.types import OutputFormat
        result = self._make_result([
            {"num": 1, "blank": True, "reason": "빈"},
            {"num": 2, "blank": True, "reason": "빈"},
        ])
        out = writer.write(result, tmp_path, OutputFormat.TXT)
        content = out.read_text(encoding="utf-8")
        assert out.exists()  # 파일 생성됨
        assert "빈 페이지" in content


# ══════════════════════════════════════════════════════════════════════════════
# 5. 전체 파이프라인 E2E (API 키 없이 Tesseract만)
# ══════════════════════════════════════════════════════════════════════════════

class TestFullPipelineE2E:

    def test_image_to_file_pipeline(self, tmp_path, monkeypatch):
        """
        이미지 → 품질진단 → 전처리 → Tesseract → L1보정 → 후처리 → 파일저장
        API 키 없이 Tesseract 전용으로 전체 파이프라인 테스트.
        """
        import core.config as cfg
        monkeypatch.setattr(cfg, "_CONFIG_FILE", tmp_path / "cfg.json")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        from core import quality_diagnoser, preprocessor, tesseract_engine
        from core import char_normalizer, postprocessor
        from core.types import OCREngine

        img = _make_text_image([
            "인간은 얼마나 뛰어난 작품인가!",
            "Hello World Test Document",
            "OCR 파이프라인 통합 테스트",
        ])

        # 품질 진단
        report = quality_diagnoser.diagnose(img)
        assert report.width_px == 800
        assert report.height_px == 600

        # 전처리
        processed = preprocessor.preprocess(img, report)
        assert isinstance(processed, Image.Image)

        # Tesseract OCR
        ocr_result = tesseract_engine.smart_run(processed, default_lang="kor+eng")
        assert ocr_result.engine == OCREngine.TESSERACT
        assert 0.0 <= ocr_result.confidence <= 100.0

        # L1 + 후처리
        text = char_normalizer.normalize_all(ocr_result.text)
        text = postprocessor.process_full(text, engine="tesseract")
        assert isinstance(text, str)

    def test_blank_page_skipped_e2e(self, tmp_path, monkeypatch):
        """빈 페이지가 전체 파이프라인에서 건너뛰어지는지 E2E 확인."""
        from core.blank_detector import check
        from core.types import PageResult, OCREngine

        blank_img = _make_blank_image()
        r = check(blank_img, fast=True)
        assert r.is_blank

        # 빈 페이지 PageResult 생성 (ocr_gradio._process_page 동작 시뮬레이션)
        page = PageResult(
            page_num=3, raw_text="", corrected_text="",
            engine=OCREngine.TESSERACT, confidence=0.0,
            is_blank=True, blank_reason=r.reason,
        )
        assert page.is_blank is True
        assert page.corrected_text == ""

    def test_mixed_pages_pipeline(self, tmp_path):
        """텍스트 페이지 + 빈 페이지 혼합 → writer 출력 확인."""
        from core.types import FileResult, PageResult, OCREngine, OutputFormat
        from core import writer

        result = FileResult(source_path=Path("mixed.pdf"), output_path=None)
        result.pages = [
            PageResult(1, "텍스트 내용", "텍스트 내용",
                       OCREngine.TESSERACT, 85.0),
            PageResult(2, "", "", OCREngine.TESSERACT, 0.0,
                       is_blank=True, blank_reason="공백 100%"),
            PageResult(3, "다른 텍스트", "다른 텍스트",
                       OCREngine.TESSERACT, 80.0),
        ]

        out = writer.write(result, tmp_path, OutputFormat.TXT)
        content = out.read_text(encoding="utf-8")

        assert "텍스트 내용" in content
        assert "다른 텍스트" in content
        assert "빈 페이지" in content
        # 실제 텍스트가 있는 페이지 1, 3은 출력됨
        assert "페이지 1" in content
        assert "페이지 3" in content


# ══════════════════════════════════════════════════════════════════════════════
# 6. 설정 관리 통합 테스트
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigIntegration:

    def test_full_config_lifecycle(self, tmp_path, monkeypatch):
        import core.config as cfg
        monkeypatch.setattr(cfg, "_CONFIG_FILE", tmp_path / "config.json")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        # 초기 상태
        assert not cfg.has_api_key()

        # API 키 저장
        cfg.set_api_key("sk-ant-test-1234567890abcd")
        assert cfg.has_api_key()
        assert cfg.get_api_key() == "sk-ant-test-1234567890abcd"

        # 마스킹 확인
        shown = cfg.show()
        assert "..." in shown["anthropic_api_key"]
        # 앞 8자 노출: "sk-ant-t" (8자)
        assert shown["anthropic_api_key"].startswith("sk-ant-t")
        assert shown["anthropic_api_key"].endswith("abcd")

        # 기본 설정 저장
        cfg.save({"default_lang": "kor", "default_format": "md",
                  "save_original_files": True})
        loaded = cfg.load()
        assert loaded["default_lang"] == "kor"
        assert loaded["default_format"] == "md"
        assert loaded["save_original_files"] is True

        # API 키 삭제
        cfg.clear_api_key()
        assert not cfg.has_api_key()

    def test_env_var_overrides_config(self, tmp_path, monkeypatch):
        import core.config as cfg
        monkeypatch.setattr(cfg, "_CONFIG_FILE", tmp_path / "config.json")
        cfg.set_api_key("file-key-value")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-value")
        assert cfg.get_api_key() == "env-key-value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
