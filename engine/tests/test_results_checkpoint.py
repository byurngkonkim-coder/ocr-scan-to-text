"""정제 중간 저장(체크포인트) + 02_결과/ 확정 회귀 테스트.

배경: 정제본이 전 구간 완료 후 한 번만 저장돼, 중간에 끊기면(브라우저 종료·
API 정지) 이미 정제된 구간까지 전부 소실됐다. 구간마다 디스크에 남기고
완료 시 02_결과/ 에 확정하도록 바꾼 동작을 고정한다.
"""
from pathlib import Path
from unittest.mock import patch

from core import text_cleaner
from core.types import OutputFormat

import ocr_gradio


def test_iter_clean_emits_each_chunk():
    """on_chunk 이 청크가 끝날 때마다 호출된다(마지막에 몰아서가 아니라)."""
    page = "문장입니다. " * 250 + "끝."          # 페이지당 약 1,750자
    pages = [page] * 6                            # → 2청크 (soft 4,000자)

    got: list[str] = []
    # run_text 가 프롬프트를 그대로 돌려주면 무결성 가드(_output_ok)를 통과한다
    with patch.object(text_cleaner._gemini, "run_text", side_effect=lambda p: p):
        gen = text_cleaner.iter_clean(pages, on_chunk=got.append)
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

    assert len(got) == 2
    assert all(g.strip() for g in got)


def test_checkpoint_appends_then_finalizes(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr_gradio, "RESULTS_DIR", tmp_path)

    ckpt = ocr_gradio._CleanCheckpoint("책제목", OutputFormat.TXT)
    ckpt.append("첫 구간")
    # 완료 전에도 디스크에서 읽힌다 — 중단 시 회수 가능해야 하므로
    assert "첫 구간" in ckpt.path.read_text(encoding="utf-8")
    ckpt.append("둘째 구간")

    final = ckpt.finish("첫 구간\n\n둘째 구간")
    assert final.parent == tmp_path
    assert final.name.endswith("_책제목.txt")
    assert final.read_text(encoding="utf-8") == "첫 구간\n\n둘째 구간\n"
    assert not ckpt.path.exists()          # 중간 파일은 확정 후 정리


def test_checkpoint_survives_abort(tmp_path, monkeypatch):
    """정제본이 비면(중단·실패) 중간 파일을 증거로 남긴다."""
    monkeypatch.setattr(ocr_gradio, "RESULTS_DIR", tmp_path)

    ckpt = ocr_gradio._CleanCheckpoint("중단된책", OutputFormat.TXT)
    ckpt.append("여기까지 정제됨")
    out = ocr_gradio._finalize_results(ckpt, "", Path("없는파일.txt"), "중단된책")

    assert out is None                     # 확정 산출물 없음
    assert ckpt.path.exists()              # 그러나 중간 저장분은 살아 있음
    assert "여기까지 정제됨" in ckpt.path.read_text(encoding="utf-8")
