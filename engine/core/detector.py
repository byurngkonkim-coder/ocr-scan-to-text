"""파일 포맷 감지 — 확장자 + 매직 바이트 이중 검증"""
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}

# 이미지 매직 바이트 시그니처
_MAGIC: list[tuple[bytes, str]] = [
    (b"\x89PNG",     "png"),
    (b"\xff\xd8\xff", "jpg"),
    (b"II\x2a\x00",  "tiff"),
    (b"MM\x00\x2a",  "tiff"),
    (b"GIF8",        "gif"),
]


def _detect_image_type(path: Path) -> str | None:
    with open(path, "rb") as f:
        header = f.read(8)
    for magic, fmt in _MAGIC:
        if header.startswith(magic):
            return fmt
    return None


def detect(path: str | Path) -> str:
    """
    파일 포맷을 감지하여 소문자 확장자 반환.
    지원되지 않는 포맷이면 ValueError 발생.
    반환값: 'pdf' | 'png' | 'jpg' | 'tiff'
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {p}")

    ext = p.suffix.lower()

    # PDF는 매직 바이트로 검증
    if ext == ".pdf":
        with open(p, "rb") as f:
            header = f.read(4)
        if header == b"%PDF":
            return "pdf"
        raise ValueError(f"확장자는 .pdf이지만 올바른 PDF 파일이 아닙니다: {p.name}")

    # 이미지: 매직 바이트로 실제 포맷 확인
    if ext in {".png", ".jpg", ".jpeg", ".tiff", ".tif"}:
        detected = _detect_image_type(p)
        if detected is None:
            raise ValueError(f"이미지 포맷을 인식할 수 없습니다: {p.name}")
        if detected == "jpg" or ext in {".jpg", ".jpeg"}:
            return "jpg"
        if detected == "tiff" or ext in {".tiff", ".tif"}:
            return "tiff"
        return detected

    raise ValueError(
        f"지원하지 않는 파일 형식입니다: '{ext}'\n"
        f"지원 형식: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def collect_files(path: str | Path) -> list[Path]:
    """
    파일 또는 폴더 경로를 받아 처리 가능한 파일 목록 반환.
    폴더인 경우 재귀적으로 탐색.
    """
    p = Path(path)
    if p.is_file():
        detect(p)  # 지원 여부 검증 (예외 발생 시 전파)
        return [p]

    if p.is_dir():
        files = []
        for f in sorted(p.rglob("*")):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(f)
        if not files:
            raise ValueError(f"처리 가능한 파일이 없습니다: {p}")
        return files

    raise FileNotFoundError(f"경로를 찾을 수 없습니다: {p}")
