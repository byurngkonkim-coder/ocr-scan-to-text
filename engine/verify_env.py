"""환경 검증 스크립트 — Phase 0 완료 기준"""
import sys
import subprocess
import importlib

REQUIRED_PYTHON = (3, 10)

def check(label: str, ok: bool, detail: str = ""):
    status = "✅" if ok else "❌"
    msg = f"  {status} {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return ok

def main():
    print("\n=== OCR 도구 환경 검증 ===\n")
    all_ok = True

    # Python 버전
    ver = sys.version_info
    ok = ver >= REQUIRED_PYTHON
    all_ok &= check(
        f"Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+",
        ok,
        f"현재 {ver.major}.{ver.minor}.{ver.micro}"
    )

    # RapidOCR (ONNX) 기본 로컬 엔진 확인
    rapid_ok = False
    try:
        import rapidocr
        import onnxruntime
        rapid_ok = True
        rapid_ver = getattr(rapidocr, "__version__", "설치됨")
    except ImportError:
        rapid_ver = "미설치"
    check("RapidOCR ONNX (기본 로컬 고속 엔진)", rapid_ok, f"v{rapid_ver}" if rapid_ok else rapid_ver)

    # Tesseract (선택/레거시 폴백)
    try:
        result = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True, text=True
        )
        ver_line = result.stdout.splitlines()[0] if result.stdout else result.stderr.splitlines()[0]
        tess_ok = result.returncode == 0
    except FileNotFoundError:
        tess_ok = False
        ver_line = "설치되지 않음 (RapidOCR이 기본 작동하므로 무방)"
    check("Tesseract OCR (보조 폴백 엔진)", tess_ok, ver_line.strip())

    if not rapid_ok and not tess_ok:
        all_ok = False

    # Tesseract 한국어팩 (Tesseract가 설치된 경우에만 검사)
    if tess_ok:
        try:
            langs = subprocess.run(
                ["tesseract", "--list-langs"],
                capture_output=True, text=True
            )
            lang_text = langs.stdout + langs.stderr
            kor_ok = "kor" in lang_text
        except Exception:
            kor_ok = False
        check("Tesseract 한국어팩(kor)", kor_ok)
        check("Tesseract 영어팩(eng)", "eng" in lang_text)

    # Poppler
    try:
        result = subprocess.run(["pdftoppm", "-v"], capture_output=True, text=True)
        pop_ok = result.returncode == 0 or "pdftoppm" in (result.stderr or "")
        ver_line = (result.stderr or result.stdout).splitlines()[0] if (result.stderr or result.stdout) else ""
    except FileNotFoundError:
        pop_ok = False
        ver_line = "설치되지 않음 (PDF 처리 불가)"
    check("Poppler (pdf2image 런타임)", pop_ok, ver_line.strip())
    # Poppler는 경고만 (이미지 파일만 쓸 경우 불필요)

    # Python 패키지
    packages = [
        ("RapidOCR",    "rapidocr"),
        ("ONNX Runtime","onnxruntime"),
        ("PyMuPDF",     "fitz"),
        ("Pillow",      "PIL"),
        ("OpenCV",      "cv2"),
        ("numpy",       "numpy"),
        ("click",       "click"),
        ("rich",        "rich"),
        ("gradio",      "gradio"),
        ("anthropic",   "anthropic"),
    ]
    print()
    for label, module in packages:
        try:
            m = importlib.import_module(module)
            ver_attr = getattr(m, "__version__", "?")
            ok = True
        except ImportError:
            ver_attr = "미설치"
            ok = False
        all_ok &= check(label, ok, f"v{ver_attr}" if ok else ver_attr)

    # 선택 패키지
    optional = [("py-hanspell (L2 맞춤법)", "hanspell")]
    print()
    print("  [선택 패키지]")
    for label, module in optional:
        try:
            importlib.import_module(module)
            ok = True
            detail = "설치됨"
        except ImportError:
            ok = True  # 선택이므로 전체 결과에 영향 없음
            detail = "미설치 (L2 보정 비활성)"
        check(label, ok, detail)

    print()
    if all_ok:
        print("✅ 모든 필수 환경 확인 완료. Phase 1 개발을 시작할 수 있습니다.\n")
    else:
        print("❌ 일부 환경이 준비되지 않았습니다. 위 항목을 확인해 주세요.\n")
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
