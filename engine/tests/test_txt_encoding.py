"""② 일괄·① 탭 txt 입력 — 메모장 ANSI(cp949)·유니코드(UTF-16) 저장본이 UnicodeDecodeError 로 '실패'하던 것(감사 30번)."""
import pytest

ocr_gradio = pytest.importorskip("ocr_gradio")


@pytest.mark.parametrize("enc", ["utf-8", "utf-8-sig", "cp949", "utf-16"])
def test_txt_read_in_common_encodings(tmp_path, enc):
    p = tmp_path / "a.txt"
    p.write_bytes("경영은 사람을 통해 일을 이루는 것이다.".encode(enc))
    assert ocr_gradio._read_txt(p).lstrip("\ufeff") == "경영은 사람을 통해 일을 이루는 것이다."
