"""Phase 3 테스트 — V-06 · V-07 · MD 출력 · 후처리"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from core import figure_detector, postprocessor, writer
from core.types import FileResult, OutputFormat, PageResult, OCREngine


def _font(size=24):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _make_text_image(texts: list[str], size=(800, 400)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    font = _font(24)
    for i, t in enumerate(texts):
        draw.text((40, 40 + i * 50), t, fill="black", font=font)
    return img


def _make_page(text: str, figures: list[str] | None = None) -> PageResult:
    return PageResult(
        page_num=1, raw_text=text, corrected_text=text,
        engine=OCREngine.RAPIDOCR, confidence=85.0,
        figures=figures or [],
    )


def _make_result(pages: list[PageResult], name="test.png") -> FileResult:
    r = FileResult(source_path=Path(name), output_path=None)
    r.pages = pages
    return r


# ── V-06: 특수기호 플레이스홀더 (figure_detector 단위) ────────────────────────

class TestFigureDetectorUnit:
    def test_build_placeholder_figure(self):
        from core.figure_detector import Region
        r = Region(kind="figure", index=1, x=0, y=0, w=100, h=100)
        ph = figure_detector.build_placeholder(r)
        assert "그림 1" in ph
        assert "이미지 감지됨" in ph

    def test_build_placeholder_table_no_md(self):
        from core.figure_detector import Region
        r = Region(kind="table", index=2, x=0, y=0, w=200, h=100)
        ph = figure_detector.build_placeholder(r)
        assert "표 2" in ph

    def test_build_placeholder_table_with_md(self):
        from core.figure_detector import Region
        r = Region(kind="table", index=1, x=0, y=0, w=200, h=100,
                   table_md="| A | B |\n| --- | --- |\n| 1 | 2 |")
        ph = figure_detector.build_placeholder(r)
        assert "표 1" in ph
        assert "|" in ph  # Markdown 표 포함

    def test_build_placeholder_formula(self):
        from core.figure_detector import Region
        r = Region(kind="formula", index=1, x=0, y=0, w=100, h=50)
        ph = figure_detector.build_placeholder(r)
        assert "수식 1" in ph
        assert "인식 불가" in ph

    def test_detect_returns_result(self):
        img = _make_text_image(["안녕하세요", "테스트 문서입니다"])
        result = figure_detector.detect(img)
        assert result is not None
        assert isinstance(result.regions, list)
        assert result.masked_image is not None

    def test_text_lines_not_masked_as_figures(self):
        # 회귀: 본문 행이 그림으로 오인·마스킹돼 문장이 잘려 나가던 문제 (하루명화하루명언)
        img = Image.new("RGB", (1240, 800), "white")
        draw = ImageDraw.Draw(img)
        for i, t in enumerate([
            "아침마다 그래도 살아 있다는 안도감에 눈을 뜨는 햇살,",
            "계절이 바뀔 때마다 함께 호흡하며 달라지는 풍경,",
            "사람 사이의 섬에 다리가 놓이는 찰나의 순간들.",
        ]):
            draw.text((120, 200 + i * 55), t, fill="black", font=_font(30))
        result = figure_detector.detect(img)
        assert [r for r in result.regions if r.kind == "figure"] == []

    def test_line_height_blobs_are_not_figure_candidates(self):
        # 회귀: 스캔 본문 쪽에서 글자 줄 조각(한 줄 높이)이 쪽마다 40~70개 그림 후보가 되고,
        # Tesseract 가 한글 조각을 못 읽어 그대로 마스킹 → 줄 누락·깨진 글자 (예측기계 30쪽)
        gray = np.full((1750, 1240), 255, np.uint8)
        for i in range(8):                                   # 본문 줄 조각 (높이 30px)
            gray[200 + i * 60:230 + i * 60, 150:450] = 0
        gray[900:1400, 300:900] = 80                         # 진짜 그림 (500x600)
        gray[100:1650, 1120:1220] = 120                      # 쪽 가장자리 그림자 띠 (너비 8%)
        boxes = figure_detector._detect_figures(gray, Image.fromarray(gray).convert("RGB"))
        assert len(boxes) == 1
        x, y, w, h = boxes[0]
        assert y <= 900 and y + h >= 1400 and w >= 600

    def test_text_on_picture_survives_masking(self):
        # 회귀: 그림(색 영역) 위에 얹힌 글자가 영역째 흰색으로 지워지던 문제
        img = Image.new("RGB", (1240, 1100), "white")
        g = np.linspace(0, 1, 900)[None, :, None]
        paint = (np.array([200, 220, 245]) * (1 - g) + np.array([250, 215, 170]) * g).astype("uint8")
        img.paste(Image.fromarray(np.repeat(paint, 600, axis=0)), (170, 250))
        draw = ImageDraw.Draw(img)
        draw.text((230, 420), "겨울은 봄을 기다리고 봄은 겨울을 향해 걷는다", fill=(40, 30, 30), font=_font(36))
        draw.text((230, 500), "끝을 기억할 때 삶은 단단해지리라", fill=(40, 30, 30), font=_font(36))

        result = figure_detector.detect(img)
        assert result.regions, "그림 영역이 감지돼야 이 테스트가 의미 있다"

        box = (220, 410, 1010, 560)  # 두 글자 줄을 감싸는 영역
        dark = lambda im: int((np.array(im.convert("L").crop(box)) < 100).sum())
        assert dark(result.masked_image) >= dark(img) * 0.95

    def test_save_crops_creates_dir(self, tmp_path):
        from core.figure_detector import Region
        img = Image.new("RGB", (100, 100), "white")
        r = Region(kind="figure", index=1, x=0, y=0, w=50, h=50, crop=img)
        figure_detector.save_crops([r], tmp_path, "test")
        assert (tmp_path / "figures").exists()


# ── 이미지에 그림 삽입 후 감지 테스트 ─────────────────────────────────────────

class TestFigureDetection:
    def _make_image_with_figure(self) -> Image.Image:
        """텍스트 + 사각형 그림이 있는 이미지 생성"""
        img = Image.new("RGB", (800, 600), "white")
        draw = ImageDraw.Draw(img)
        font = _font(22)
        draw.text((40, 30), "이것은 텍스트입니다.", fill="black", font=font)
        draw.text((40, 80), "아래에 그림이 있습니다.", fill="black", font=font)
        # 그림 영역: 채색된 사각형
        draw.rectangle([100, 150, 500, 400], fill=(180, 200, 230), outline="black", width=2)
        draw.text((200, 260), "그림 내용", fill="navy", font=font)
        draw.text((40, 450), "그림 아래 텍스트입니다.", fill="black", font=font)
        return img

    def test_detect_runs_without_error(self):
        img = self._make_image_with_figure()
        result = figure_detector.detect(img)
        assert result is not None

    def test_masked_image_same_size(self):
        img = self._make_image_with_figure()
        result = figure_detector.detect(img)
        assert result.masked_image.size == img.size


# ── V-07: 그림 플레이스홀더 삽입 (writer 통합) ────────────────────────────────

class TestPlaceholderInOutput:
    def test_placeholder_in_txt(self, tmp_path):
        page = _make_page("텍스트 내용입니다.", ["[그림 1: 이미지 감지됨]"])
        result = _make_result([page])
        out = writer.write(result, tmp_path, OutputFormat.TXT)
        content = out.read_text(encoding="utf-8")
        assert "그림 1" in content
        assert "이미지 감지됨" in content

    def test_placeholder_in_md(self, tmp_path):
        page = _make_page("텍스트 내용입니다.", ["[그림 1: 이미지 감지됨]"])
        result = _make_result([page])
        out = writer.write(result, tmp_path, OutputFormat.MD)
        content = out.read_text(encoding="utf-8")
        assert "그림 1" in content

    def test_table_md_in_output(self, tmp_path):
        md_table = "[표 1]\n| 항목 | 값 |\n| --- | --- |\n| A | 1 |"
        page = _make_page("텍스트", [md_table])
        result = _make_result([page])
        out = writer.write(result, tmp_path, OutputFormat.MD)
        content = out.read_text(encoding="utf-8")
        assert "표 1" in content
        assert "|" in content

    def test_no_placeholder_if_no_figures(self, tmp_path):
        page = _make_page("텍스트만 있습니다.")
        result = _make_result([page])
        out = writer.write(result, tmp_path, OutputFormat.TXT)
        content = out.read_text(encoding="utf-8")
        assert "비텍스트 영역" not in content


# ── 후처리 ────────────────────────────────────────────────────────────────────

class TestPostprocessor:
    def test_normalizes_blank_lines(self):
        text = "줄1\n\n\n\n줄2"
        result = postprocessor.apply(text)
        assert "\n\n\n" not in result
        assert "줄1" in result
        assert "줄2" in result

    def test_strips_trailing_spaces(self):
        text = "줄1   \n줄2  "
        result = postprocessor.apply(text)
        for line in result.splitlines():
            assert not line.endswith(" ")

    def test_tab_to_spaces(self):
        text = "열1\t열2"
        result = postprocessor.apply(text)
        assert "\t" not in result

    def test_placeholder_appended(self):
        from core.figure_detector import Region
        r = Region(kind="figure", index=1, x=0, y=0, w=100, h=100)
        result = postprocessor.apply("텍스트", regions=[r])
        assert "그림 1" in result


# ── MD 출력 완성도 ────────────────────────────────────────────────────────────

class TestMDWriter:
    def test_md_has_title(self, tmp_path):
        result = _make_result([_make_page("내용")])
        out = writer.write(result, tmp_path, OutputFormat.MD)
        content = out.read_text(encoding="utf-8")
        assert content.startswith("# ")

    def test_md_has_page_heading(self, tmp_path):
        result = _make_result([_make_page("내용")])
        out = writer.write(result, tmp_path, OutputFormat.MD)
        content = out.read_text(encoding="utf-8")
        assert "## 페이지 1" in content

    def test_md_multiple_pages(self, tmp_path):
        pages = [_make_page(f"페이지{i} 내용") for i in range(1, 4)]
        for i, p in enumerate(pages, 1):
            p.page_num = i
        result = _make_result(pages)
        out = writer.write(result, tmp_path, OutputFormat.MD)
        content = out.read_text(encoding="utf-8")
        assert "## 페이지 1" in content
        assert "## 페이지 2" in content
        assert "## 페이지 3" in content

    def test_md_separator_between_pages(self, tmp_path):
        pages = [_make_page(f"내용{i}") for i in range(1, 3)]
        for i, p in enumerate(pages, 1):
            p.page_num = i
        result = _make_result(pages)
        out = writer.write(result, tmp_path, OutputFormat.MD)
        content = out.read_text(encoding="utf-8")
        assert "---" in content

    def test_md_utf8_encoding(self, tmp_path):
        page = _make_page("한글 영문 日本語 混合")
        result = _make_result([page])
        out = writer.write(result, tmp_path, OutputFormat.MD)
        content = out.read_text(encoding="utf-8")
        assert "한글" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
