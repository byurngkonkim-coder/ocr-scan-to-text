"""PDF → 페이지별 PIL 이미지 분해

대용량(책 스캔) 대응:
  - DEFAULT_DPI 200 — Gemini Vision은 내부적으로 축소하므로 200이면 충분.
    300 대비 픽셀 약 0.44배 → 렌더·메모리·업로드 모두 감소.
  - iter_pdf_pages() — 전 페이지를 한 번에 메모리에 올리지 않고
    batch_size 단위로 렌더링하며 하나씩 yield (수백 페이지 책 스캔 OOM 방지).
"""
from pathlib import Path
from typing import Iterator
from PIL import Image

try:
    from pdf2image import convert_from_path, pdfinfo_from_path
    from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError
    PDF2IMAGE_OK = True
except ImportError:
    PDF2IMAGE_OK = False


DEFAULT_DPI = 200


def _poppler_kwargs(poppler_path: str | None) -> dict:
    """Poppler 경로 kwargs 구성 (Windows 기본 경로 자동 탐색)."""
    if poppler_path:
        return {"poppler_path": poppler_path}
    import platform
    if platform.system() == "Windows":
        candidate = Path("C:/poppler/Library/bin")
        if candidate.exists():
            return {"poppler_path": str(candidate)}
    return {}


def _require_pdf2image(pdf_path: Path) -> None:
    if not PDF2IMAGE_OK:
        raise RuntimeError(
            "pdf2image 패키지가 설치되어 있지 않습니다.\n"
            "pip install pdf2image 를 실행해 주세요."
        )
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")


def count_pages(pdf_path: str | Path, poppler_path: str | None = None) -> int:
    """렌더링 없이 PDF 총 페이지 수만 조회 (메모리·시간 거의 0)."""
    pdf_path = Path(pdf_path)
    _require_pdf2image(pdf_path)
    try:
        info = pdfinfo_from_path(str(pdf_path), **_poppler_kwargs(poppler_path))
        return int(info.get("Pages", 0))
    except PDFInfoNotInstalledError:
        raise RuntimeError(
            "Poppler가 설치되어 있지 않거나 PATH에 없습니다.\n"
            "https://github.com/oschwartz10612/poppler-windows/releases"
        )
    except PDFPageCountError as e:
        raise ValueError(f"PDF 페이지 수를 읽을 수 없습니다: {e}")


def split_pdf(
    pdf_path: str | Path,
    dpi: int = DEFAULT_DPI,
    poppler_path: str | None = None,
) -> list[Image.Image]:
    """
    PDF를 페이지별 PIL 이미지 리스트로 변환 (전 페이지 일괄 — 소용량 전용).
    대용량은 iter_pdf_pages() 사용 권장.
    """
    pdf_path = Path(pdf_path)
    _require_pdf2image(pdf_path)
    kwargs = {"dpi": dpi, **_poppler_kwargs(poppler_path)}
    try:
        return convert_from_path(str(pdf_path), **kwargs)
    except PDFInfoNotInstalledError:
        raise RuntimeError(
            "Poppler가 설치되어 있지 않거나 PATH에 없습니다.\n"
            "https://github.com/oschwartz10612/poppler-windows/releases\n"
            "설치 후 bin 폴더를 PATH에 추가해 주세요."
        )
    except PDFPageCountError as e:
        raise ValueError(f"PDF 페이지 수를 읽을 수 없습니다: {e}")


def iter_pdf_pages(
    pdf_path: str | Path,
    dpi: int = DEFAULT_DPI,
    poppler_path: str | None = None,
    batch_size: int = 8,
    first_page: int = 1,
    last_page: int | None = None,
) -> Iterator[Image.Image]:
    """
    PDF 페이지를 batch_size 단위로 렌더링하며 하나씩 yield.

    한 번에 최대 batch_size 페이지만 메모리에 존재 → 수백 페이지 책 스캔도
    메모리 사용량이 일정하게 유지된다 (전 페이지 일괄 렌더 시의 OOM 방지).

    Args:
        first_page : 시작 페이지(1부터). 이어서 처리 시 활용.
        last_page  : 마지막 페이지(None이면 끝까지).
        batch_size : 한 번의 poppler 호출로 렌더링할 페이지 수.
    """
    pdf_path = Path(pdf_path)
    _require_pdf2image(pdf_path)
    kw = _poppler_kwargs(poppler_path)

    total = count_pages(pdf_path, poppler_path)
    last = total if last_page is None else min(last_page, total)
    batch_size = max(1, batch_size)

    p = max(1, first_page)
    while p <= last:
        end = min(p + batch_size - 1, last)
        try:
            pages = convert_from_path(
                str(pdf_path), dpi=dpi, first_page=p, last_page=end, **kw
            )
        except PDFPageCountError as e:
            raise ValueError(f"PDF 페이지 수를 읽을 수 없습니다: {e}")
        for img in pages:
            yield img
        p = end + 1


def load_image(image_path: str | Path) -> list[Image.Image]:
    """이미지 파일을 단일 페이지 리스트로 로드."""
    img = Image.open(str(image_path)).convert("RGB")
    return [img]
