"""START/STOP bat — 7860 포트의 프로세스가 이 프로그램 서버(ocr_server.pid)인지 확인하는지, 줄바꿈 CRLF 인지.

예전엔 7860 을 연 프로그램이면 무엇이든 이 서버로 보고 STOP 이 강제 종료했다(감사 25번).
실행 확인(2026-10-19): 다른 프로그램(http.server)이 7860 을 열면 STOP 은 남기고 START 는 오류 안내.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_bats_check_server_pid_and_use_crlf():
    for bat in (ROOT / "STOP_OCR.bat", ROOT / "engine" / "START_OCR.bat"):
        raw = bat.read_bytes()
        assert raw.count(b"\n") == raw.count(b"\r\n") and not raw.startswith(b"\xef\xbb\xbf"), bat
        assert b"ocr_server.pid" in raw and b"_mine" in raw, bat


def test_server_writes_its_own_pid():
    src = (ROOT / "engine" / "ocr_gradio.py").read_text(encoding="utf-8")
    assert 'ocr_server.pid").write_text(str(os.getpid())' in src
