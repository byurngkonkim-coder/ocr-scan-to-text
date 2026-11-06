"""START/STOP bat — 7860 포트를 쓰는 것이 이 프로그램 서버인지 확인하는지, 줄바꿈 CRLF 인지.

예전엔 7860 을 연 프로그램이면 무엇이든 이 서버로 보고 STOP 이 강제 종료했다(감사 25번).
실행 확인(2026-10-19): 다른 프로그램(http.server)이 7860 을 열면 STOP 은 남기고 START 는 오류 안내.
2026-11-06: 실행 중인 서버 소켓이 netstat 에 LISTENING 으로 안 나와(FIN_WAIT_1·목록에 없음) START 가 서버를
하나 더 띄웠고, 그것이 포트 충돌로 죽으며 ocr_server.pid 를 덮어써 STOP 도 서버를 못 찾았다 →
netstat 대신 ocr_port.ps1(포트 접속 시험 + 명령줄 ocr_gradio.py 로 서버 식별), 서버는 포트가 이미 열려 있으면 안 뜬다.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BATS = (ROOT / "STOP_OCR.bat", ROOT / "engine" / "START_OCR.bat")


def _crlf_ascii(path: Path) -> bytes:
    raw = path.read_bytes()
    assert raw.count(b"\n") == raw.count(b"\r\n") and not raw.startswith(b"\xef\xbb\xbf"), path
    raw.decode("ascii")
    return raw


def test_bats_use_port_helper_not_netstat_state_and_use_crlf():
    for bat in BATS:
        raw = _crlf_ascii(bat)
        assert b"ocr_port.ps1" in raw, bat
        assert b'findstr "LISTENING"' not in raw, bat


def test_port_helper_identifies_server_by_command_line():
    raw = _crlf_ascii(ROOT / "engine" / "ocr_port.ps1")
    assert b"ocr_gradio\\.py" in raw and b"TcpClient" in raw


def test_server_writes_its_own_pid_only_when_port_is_free():
    src = (ROOT / "engine" / "ocr_gradio.py").read_text(encoding="utf-8")
    assert 'ocr_server.pid").write_text(str(os.getpid())' in src
    assert src.index("if _port_in_use():") < src.index('ocr_server.pid").write_text')
