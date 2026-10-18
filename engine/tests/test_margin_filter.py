import pytest
import numpy as np
from PIL import Image
import re
from unittest.mock import MagicMock, patch

from core import preprocessor
from core import pdf_text_layer
from core import text_cleaner
from core import config
from core import vision


def _margin_filter(enabled: bool, top=0.06, bottom=0.06):
    """마진 필터 관련 config 3종을 한 번에 패치하는 컨텍스트 매니저."""
    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(patch('core.config.get_enable_margin_filter', return_value=enabled))
    stack.enter_context(patch('core.config.get_margin_top_ratio', return_value=top))
    stack.enter_context(patch('core.config.get_margin_bottom_ratio', return_value=bottom))
    return stack


def test_image_mask_margins():
    # 1. 100x100 크기의 검은색(0, 0, 0) 이미지 생성
    img = Image.new("RGB", (100, 100), (0, 0, 0))
    
    # 2. 상하단 10%씩 마스킹 적용 (즉, y=0~9, y=90~99 영역)
    masked = preprocessor.mask_margins(img, top_ratio=0.1, bottom_ratio=0.1)
    arr = np.array(masked)
    
    # 3. 검증: 상단 10% 및 하단 10%는 흰색(255, 255, 255)이어야 함
    assert np.all(arr[0:10, :, :] == 255)
    assert np.all(arr[90:100, :, :] == 255)
    
    # 4. 검증: 중간 본문 영역(y=10~89)은 여전히 검은색(0, 0, 0)이어야 함
    assert np.all(arr[10:90, :, :] == 0)


def test_post_process_pagebreaks():
    # 1. 일반적인 1음절 결합 케이스 (예: 성찰의)
    text1 = "성 ⟨PAGEBREAK⟩ 찰의 노력 없이는"
    assert text_cleaner.post_process_pagebreaks(text1) == "성찰의 노력 없이는"
    
    # 2. 조사 결합 케이스 (예: 결정을)
    text2 = "그것은 결정 ⟨PAGEBREAK⟩ 을 뜻합니다"
    assert text_cleaner.post_process_pagebreaks(text2) == "그것은 결정을 뜻합니다"
    
    # 3. 1음절 제외 목록 케이스 (예: 관형사 '이') -> 결합 안 하고 띄어씀
    text3 = "이 ⟨PAGEBREAK⟩ 책의 내용"
    assert text_cleaner.post_process_pagebreaks(text3) == "이 책의 내용"
    
    # 4. 일반적인 2음절 이상 단어들의 결합 없음 (예: 사과 바나나)
    text4 = "사과 ⟨PAGEBREAK⟩ 바나나를 먹었다"
    assert text_cleaner.post_process_pagebreaks(text4) == "사과 바나나를 먹었다"


def test_vision_run_applies_margin_mask():
    """Vision 경로는 preprocess 를 건너뛰므로 vision.run 이 직접 마스킹해야 한다."""
    img = Image.new("RGB", (100, 100), (0, 0, 0))
    impl = MagicMock()

    # 필터 ON → impl.run 에 마스킹된 이미지가 전달되고, 호출부 원본은 보존
    with _margin_filter(True, top=0.1, bottom=0.1), \
         patch('core.vision._impl', return_value=impl):
        vision.run(img)
    passed = np.array(impl.run.call_args.args[0])
    assert np.all(passed[0:10, :, :] == 255)      # 상단 마스킹됨
    assert np.all(passed[10:90, :, :] == 0)       # 본문 보존
    assert np.all(np.array(img) == 0)             # 원본 미훼손

    # 필터 OFF → 원본 그대로 전달
    impl2 = MagicMock()
    with _margin_filter(False), patch('core.vision._impl', return_value=impl2):
        vision.run(img)
    assert impl2.run.call_args.args[0] is img


def test_guard_ratio_depends_on_margin_filter():
    """G-01 임계값: 필터 ON 0.90 / OFF 0.75."""
    joined = "가" * 100
    cand_80 = "가" * 80    # 손실 20% — OFF 통과, ON 차단
    cand_95 = "가" * 95

    with _margin_filter(False):
        assert text_cleaner._output_ok(cand_80, joined, False, False)
    with _margin_filter(True):
        assert not text_cleaner._output_ok(cand_80, joined, False, False)
        assert text_cleaner._output_ok(cand_95, joined, False, False)


def test_strip_output_keeps_sentinel_when_filter_on():
    """필터 ON 이면 센티넬을 남겨야 post_process_pagebreaks 가 어절을 결합한다."""
    raw = "성 ⟨PAGEBREAK⟩ 찰의 노력"
    with _margin_filter(True):
        kept = text_cleaner._strip_output(raw)
        assert text_cleaner.PAGEBREAK in kept
        assert text_cleaner.post_process_pagebreaks(kept) == "성찰의 노력"
    with _margin_filter(False):
        assert text_cleaner.PAGEBREAK not in text_cleaner._strip_output(raw)


@patch('fitz.open')
def test_pdf_text_layer_filter(mock_open):
    # 1. fitz page 객체 모킹
    mock_page = MagicMock()
    mock_page.rect.height = 1000
    
    # blocks 구조: (x0, y0, x1, y1, "text", block_no, block_type)
    mock_page.get_text.return_value = [
        (10, 20, 100, 40, "Header Page 1", 0, 0),         # y0=20 -> 마진 상단 영역 (top_ratio=0.06 이면 60px 이하)
        (10, 100, 500, 200, "본문 내용 첫 줄입니다.", 1, 0), # 본문
        (10, 950, 100, 980, "1", 2, 0),                    # y1=980 -> 마진 하단 영역 (bottom_ratio=0.06 이면 940px 이상)
        (10, 500, 200, 600, "이미지영역", 3, 1)             # block_type=1 (텍스트 아님)
    ]
    
    mock_doc = MagicMock()
    mock_doc.__enter__.return_value = [mock_page]
    mock_open.return_value = mock_doc
    
    # 2. config 설정 모킹 (필터 활성화)
    with patch('core.config.get_enable_margin_filter', return_value=True), \
         patch('core.config.get_margin_top_ratio', return_value=0.06), \
         patch('core.config.get_margin_bottom_ratio', return_value=0.06):
         
        # 3. 텍스트 추출 실행
        extracted = pdf_text_layer._extract_fitz("mock_pdf.pdf")
        
        # 4. 검증: "Header"와 "1" (쪽번호)이 제거되고 본문만 추출되어야 함
        assert len(extracted) == 1
        assert "Header" not in extracted[0]
        assert "1" not in extracted[0]
        assert "이미지영역" not in extracted[0]
        assert "본문 내용 첫 줄입니다." in extracted[0]


def test_numbers_joined_by_symbols_at_line_start_kept():
    # '6 · 25 전쟁'·'2026. 10. 19.'의 숫자·기호를 줄 머리 마진 잡음으로 지우던 것(감사 26번)
    from core import postprocessor
    for line in ("6 · 25 전쟁이 일어났다", "3 · 1 운동", "2026. 10. 19. 서울에서"):
        assert postprocessor._strip_margin_noise(line) == line
    assert postprocessor._strip_margin_noise("00 10: *인간은 생각한다") == "인간은 생각한다"
