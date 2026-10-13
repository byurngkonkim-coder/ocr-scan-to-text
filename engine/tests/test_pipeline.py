"""Phase 1 파이프라인 테스트 — V-01 ~ V-03"""
import os
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw, ImageFont
from core import detector, pdf_splitter, quality_diagnoser, preprocessor
from core import rapidocr_engine, char_normalizer, writer
from core.types import OutputFormat, FileResult, PageResult, OCREngine


# ── 공통 픽스처 ────────────────────────────────────────────────────────────────

def _make_image(texts: list[str], size=(800, 400)) -> Image.Image:
    """테스트용 이미지 생성 (한글 폰트 사용)"""
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    font_path = "C:/Windows/Fonts/malgun.ttf"
    try:
        font = ImageFont.truetype(font_path, 24)
    except Exception:
        font = ImageFont.load_default()
    for i, text in enumerate(texts):
        draw.text((40, 40 + i * 50), text, fill="black", font=font)
    return img


# ── V-01: 포맷 감지 ────────────────────────────────────────────────────────────

class TestDetector:
    def test_png_detected(self, tmp_path):
        img = _make_image(["test"])
        p = tmp_path / "test.png"
        img.save(str(p))
        assert detector.detect(p) == "png"

    def test_jpg_detected(self, tmp_path):
        img = _make_image(["test"])
        p = tmp_path / "test.jpg"
        img.save(str(p))
        assert detector.detect(p) == "jpg"

    def test_unsupported_raises(self, tmp_path):
        p = tmp_path / "test.bmp"
        p.write_bytes(b"\x00\x00")
        with pytest.raises(ValueError, match="지원하지 않는"):
            detector.detect(p)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            detector.detect("/nonexistent/file.png")

    def test_collect_folder(self, tmp_path):
        for name in ["a.png", "b.jpg", "c.txt"]:
            p = tmp_path / name
            if name.endswith(".txt"):
                p.write_text("x")
            else:
                _make_image(["x"]).save(str(p))
        files = detector.collect_files(tmp_path)
        assert len(files) == 2  # txt 제외

    def test_collect_folder_prunes_skip_dirs(self, tmp_path):
        # 하위 폴더는 찾되, 숨김·가상환경 폴더는 들어가지 않는다 (N01 — rglob 은 다 돈 뒤 걸렀다)
        for rel in ["a.png", "sub/b.png", ".git/c.png", "venv/d.png", "node_modules/e.png"]:
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            _make_image(["x"]).save(str(p))
        files = detector.collect_files(tmp_path)
        assert [f.relative_to(tmp_path).as_posix() for f in files] == ["a.png", "sub/b.png"]

    def test_collect_folder_skip_dirs_case_insensitive(self, tmp_path):
        # Windows 는 대소문자를 가리지 않는다 — 'Venv' 도 건너뛴다
        for rel in ["a.png", "Venv/b.png", "Node_Modules/c.png"]:
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            _make_image(["x"]).save(str(p))
        files = detector.collect_files(tmp_path)
        assert [f.relative_to(tmp_path).as_posix() for f in files] == ["a.png"]

    def test_collect_folder_only_in_skipped_dirs_names_them(self, tmp_path):
        # 파일이 전부 건너뛴 폴더에만 있으면, 그 폴더를 오류 메시지에 알려 준다
        p = tmp_path / ".scans" / "a.png"
        p.parent.mkdir()
        _make_image(["x"]).save(str(p))
        with pytest.raises(ValueError, match=r"\.scans"):
            detector.collect_files(tmp_path)

    def test_collect_folder_skips_non_regular_entries(self, tmp_path, monkeypatch):
        # 깨진 바로가기처럼 실제 파일이 아닌 항목은 목록에 넣지 않는다
        _make_image(["x"]).save(str(tmp_path / "a.png"))
        (tmp_path / "ghost.png").write_bytes(b"")
        real_isfile = os.path.isfile
        monkeypatch.setattr(detector.os.path, "isfile",
                            lambda f: not str(f).endswith("ghost.png") and real_isfile(f))
        files = detector.collect_files(tmp_path)
        assert [f.name for f in files] == ["a.png"]


# ── V-02: 이미지 로드 ──────────────────────────────────────────────────────────

class TestLoader:
    def test_load_png(self, tmp_path):
        img = _make_image(["hello"])
        p = tmp_path / "test.png"
        img.save(str(p))
        pages = pdf_splitter.load_image(p)
        assert len(pages) == 1
        assert isinstance(pages[0], Image.Image)


# ── 품질 진단 ──────────────────────────────────────────────────────────────────

class TestQualityDiagnoser:
    def test_normal_image(self):
        img = _make_image(["안녕하세요 테스트입니다"])
        report = quality_diagnoser.diagnose(img)
        assert report.width_px == 800
        assert report.height_px == 400
        assert report.blur_score > 0
        assert 0 <= report.avg_brightness <= 255

    def test_describe_returns_string(self):
        img = _make_image(["test"])
        report = quality_diagnoser.diagnose(img)
        desc = quality_diagnoser.describe(report)
        assert "해상도" in desc
        assert "흐림" in desc


# ── 전처리 ────────────────────────────────────────────────────────────────────

class TestPreprocessor:
    def test_output_is_pil(self):
        img = _make_image(["전처리 테스트"])
        result = preprocessor.preprocess(img)
        assert isinstance(result, Image.Image)

    def test_output_grayscale_or_binary(self):
        img = _make_image(["test"])
        result = preprocessor.preprocess(img)
        # 이진화 후 픽셀은 0 또는 255
        import numpy as np
        arr = np.array(result.convert("L"))
        unique = set(arr.flatten())
        assert unique.issubset({0, 255})

    def test_clean_scan_otsu_path_binary(self):
        # 깨끗한 스캔(Otsu 경로)도 출력은 이진(0/255)이어야 함
        from core.types import QualityReport
        import numpy as np
        img = _make_image(["깨끗한 스캔 테스트 문장입니다"])
        report = QualityReport(
            blur_score=500.0, avg_brightness=200.0, skew_angle=0.0,
            is_vertical_text=False, width_px=800, height_px=300,
        )
        arr = np.array(preprocessor.preprocess(img, report).convert("L"))
        assert set(arr.flatten()).issubset({0, 255})


# ── V-03: RapidOCR 한글 v5 ────────────────────────────────────────────────────

class TestRapidOCREngine:
    def test_english_recognized(self):
        img = _make_image(["Hello World Test"])
        result = rapidocr_engine.run(img, lang="eng")
        assert "Hello" in result.text or "hello" in result.text.lower()
        assert result.engine == OCREngine.RAPIDOCR

    def test_korean_recognized(self):
        img = _make_image(["안녕하세요 테스트입니다"])
        result = rapidocr_engine.run(img, lang="kor")
        # 일부 글자라도 인식되면 통과
        kor_chars = sum(1 for c in result.text if "가" <= c <= "힣")
        assert kor_chars > 0

    def test_confidence_range(self):
        img = _make_image(["test"])
        result = rapidocr_engine.run(img, lang="eng")
        assert 0.0 <= result.confidence <= 100.0


# ── L1 문자 보정 ───────────────────────────────────────────────────────────────

class TestCharNormalizer:
    def test_ligature_expansion(self):
        assert char_normalizer.normalize_all("ﬁle ﬂow") == "file flow"

    def test_symbol_map(self):
        result = char_normalizer.normalize_all("(c) 2026 (r)")
        assert "©" in result
        assert "®" in result

    def test_arrow_map(self):
        result = char_normalizer.normalize_all("A -> B")
        assert "→" in result

    def test_jamo_detection_true(self):
        # 자모 분리 텍스트
        jamo_text = "ㅎㅏㄴㄱㅡㄹ " * 5
        assert char_normalizer.has_jamo_decomposition(jamo_text) is True

    def test_jamo_detection_false(self):
        normal = "안녕하세요 hello world"
        assert char_normalizer.has_jamo_decomposition(normal) is False

    def test_english_confusion(self):
        # 숫자 사이 I → 1
        result = char_normalizer.fix_english_confusion("3I5")
        assert result == "315"


# ── writer ────────────────────────────────────────────────────────────────────

class TestWriter:
    def _make_result(self, text: str) -> FileResult:
        from pathlib import Path
        result = FileResult(source_path=Path("test.png"), output_path=None)
        result.pages.append(PageResult(
            page_num=1, raw_text=text, corrected_text=text,
            engine=OCREngine.RAPIDOCR, confidence=85.0,
        ))
        return result

    def test_txt_output(self, tmp_path):
        result = self._make_result("안녕하세요 테스트")
        out = writer.write(result, tmp_path, OutputFormat.TXT)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "페이지 1" in content
        assert "안녕하세요 테스트" in content

    def test_md_output(self, tmp_path):
        result = self._make_result("hello world")
        out = writer.write(result, tmp_path, OutputFormat.MD)
        assert out.suffix == ".md"
        content = out.read_text(encoding="utf-8")
        assert "## 페이지 1" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
