# -*- coding: utf-8 -*-
"""core/korean_corrector.py - 오프라인 독립형 한국어 띄어쓰기 및 맞춤법 정제기.

01_OCR_스캔파일텍스트추출프로그램 독립 배포를 위해 외부 의존성(인터넷, LLM API 등) 없이
순수 로컬 정규식·사전·형태소 규칙으로 텍스트 정제를 수행합니다.
"""
from __future__ import annotations

import re
import sys
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# 1. OCR 오인식 따옴표 및 기호 정규화
# ─────────────────────────────────────────────────────────────────────────────

def fix_ocr_quotes_and_symbols(text: str) -> str:
    """OCR 시 따옴표가 66/99로 오인식된 경우 및 기호 경계를 정규화합니다."""
    if not text:
        return text

    # (1) 어두/문장 시작 큰따옴표 66 -> "
    text = re.sub(r'(^|[\s\(\[\{])66([가-힣A-Za-z])', r'\1"\2', text)
    text = re.sub(r'(^|[\s\(\[\{])66\s+([가-힣A-Za-z])', r'\1"\2', text)

    # (2) 어미/문장 끝 닫는 큰따옴표 99 -> "
    text = re.sub(r'([가-힣A-Za-z0-9.!?…])99([\s\)\]\}\n.,!?]|$)', r'\1"\2', text)

    # (3) 괄호와 한글 사이 공백 분리
    text = re.sub(r'([가-힣0-9])(\[|\()', r'\1 \2', text)

    # (4) 한글과 숫자 경계 분리
    text = re.sub(r'([가-힣]{2,})([0-9]{1,4})(을|를|이|가|은|는|에|의|로|으로|와|과|도|만)', r'\1 \2\3', text)

    return text


# ─────────────────────────────────────────────────────────────────────────────
# 2. OCR 빈발 오탈자 및 맞춤법 교정 테이블
# ─────────────────────────────────────────────────────────────────────────────

TYPO_DICT: dict[str, str] = {
    # OCR 자모/음절 오인식
    "훌륨한": "훌륭한",
    "훌륨": "훌륭",
    "경럭": "경력",
    "띨어쓰기": "띄어쓰기",
    "띨어": "띄어",
    "밎춤법": "맞춤법",
    "있읍니다": "있습니다",
    "없읍니다": "없습니다",
    "같읍니다": "같습니다",
    "되여": "되어",
    "되였": "되었",
    "않되": "안 돼",
    "안되": "안 돼",
    "담겨있다": "담겨 있다",
    "되어있다": "되어 있다",
    "할수밖에": "할 수밖에",
    "도착할때": "도착할 때",
    "기분좋은": "기분 좋은",
    "눈을뜨자마자": "눈을 뜨자마자",
    "꽉 막혀있다": "꽉 막혀 있다",
    "몇일": "며칠",
    "어의없": "어이없",
    "역활": "역할",
    "금새": "금세",
    "일일히": "일일이",
    "틈틈히": "틈틈이",
    "깨끗히": "깨끗이",
    "곰곰히": "곰곰이",
    "더우기": "더욱이",
    "오뚜기": "오뚝이",
    "바램": "바람",
    "설레임": "설렘",
    "뒤치닥거리": "뒤치다꺼리",
    "널부러": "널브러",
    "어줍잖": "어쭙잖",
    "개발새발": "개발새발",
    "괴팍": "괴팍",
    "궁시렁": "구시렁",
    "끄나풀": "끄나풀",
    "널직": "널찍",
    "넙적": "널따랗",
    "담배꽁초": "담배꽁초",
    "댓가": "대가",
    "똬리": "똬리",
    "뜨게질": "뜨개질",
    "막냇동생": "막냇동생",
    "멋장이": "멋쟁이",
    "무등": "무동",
    "바둥바둥": "바동바동",
    "반짇고리": "반짇고리",
    "사글세": "사글세",
    "삼촌": "삼촌",
    "상판대기": "상판대기",
    "생각컨대": "생각건대",
    "서슴치": "서슴지",
    "설겆이": "설거지",
    "속앓이": "속앓이",
    "시라소니": "스라소니",
    "아지랑이": "아지랑이",
    "안성마춤": "안성맞춤",
    "애벌래": "애벌레",
    "어물쩡": "어물쩍",
    "요술장이": "요술쟁이",
    "우뢰": "우레",
    "으시대": "으스대",
    "자맥질": "자맥질",
    "주책이다": "주책이다",
    "지리하다": "지루하다",
    "짜깁기": "짜깁기",
    "풋나기": "풋내기",
    "풍지박산": "풍비박산",
    "횡그레": "휑그렁",
}

def fix_spelling_and_typos(text: str) -> str:
    """엄선된 OCR 빈발 오탈자 및 맞춤법을 규칙 기반으로 교정합니다."""
    if not text:
        return text

    for typo, correction in TYPO_DICT.items():
        if typo in text:
            text = text.replace(typo, correction)

    text = re.sub(r'않되([가-힣]*)', r'안 되\1', text)
    text = re.sub(r'않돼([가-힣]*)', r'안 돼\1', text)

    return text


# ─────────────────────────────────────────────────────────────────────────────
# 3. 과다 분리(어절 쪼개짐) 결합 (D-02, D-10, D-11)
# ─────────────────────────────────────────────────────────────────────────────

_D02_LABEL_RE = re.compile(r"(?<![가-힣])(?:[가-힣]\s){1,3}[가-힣](?=\s*:)")

_D11_BOUND_FRAGMENTS = {
    "니", "며", "히", "까", "랴", "는", "가", "를", "에", "의", "도", "로", "서",
    "니라", "리라", "리요", "로다", "도다", "노라", "소서", "이까", "께서",
    "더라", "하되", "시되", "시고", "시며", "시니", "하니", "거니와",
    "나이다", "리이다", "니이다", "이니라", "느니라", "지니라", "지어다",
    "사오니", "옵소서", "시니라", "하니라", "하시며", "리로다", "려니와",
    "즈", "본적인", "난다", "린다", "킨다", "한다",
}

_COMMON_SPLIT_PAIRS = {
    ("세일", "즈"): "세일즈",
    ("기", "본적인"): "기본적인",
    ("만", "난다"): "만난다",
    ("가", "는"): "가는",
    ("오", "는"): "오는",
    ("하", "는"): "하는",
    ("되", "는"): "되는",
    ("있", "는"): "있는",
    ("없", "는"): "없는",
    ("보", "는"): "보는",
    ("주", "는"): "주는",
    ("받", "는"): "받는",
    ("찾", "는"): "찾는",
    ("알", "는"): "아는",
    ("믿", "는"): "믿는",
    ("저", "녁"): "저녁",
    ("프로", "그램"): "프로그램",
    ("저", "장"): "저장",
    ("저", "장된"): "저장된",
    ("확", "인해"): "확인해",
    ("경", "력"): "경력",
    ("설", "득"): "설득",
    ("테크", "닉"): "테크닉",
    ("시나", "리오"): "시나리오",
}

def fix_split_tokens(text: str, document_corpus: str = "") -> str:
    """D-02, D-11, D-10 규칙을 적용하여 쪼개진 어절을 결합합니다."""
    if not text:
        return text

    text = _D02_LABEL_RE.sub(lambda m: re.sub(r"\s+", "", m.group(0)), text)

    for (w1, w2), combined in _COMMON_SPLIT_PAIRS.items():
        text = re.sub(rf"(^|[\s\[\(\"\'가-힣]){re.escape(w1)}\s+{re.escape(w2)}(?=[.,!?\s\)\"\'\]가-힣]|$)", rf"\1{combined}", text)

    corpus = document_corpus if document_corpus else text
    tokens = re.findall(r'[가-힣]{2,8}', corpus)
    tok_counts = Counter(tokens)

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        words = line.split(" ")
        res = []
        i = 0
        while i < len(words):
            w = words[i]
            if not w:
                res.append(w)
                i += 1
                continue

            if i + 1 < len(words):
                w_next = words[i + 1]
                m_next = re.match(r"^([가-힣]+)([.,!?~'\"]*)$", w_next)
                if m_next:
                    core_next = m_next.group(1)
                    punct_next = m_next.group(2)
                    joined = w + core_next

                    if core_next in _D11_BOUND_FRAGMENTS:
                        res.append(joined + punct_next)
                        i += 2
                        continue

                    if len(joined) <= 8 and tok_counts.get(joined, 0) >= 2:
                        res.append(joined + punct_next)
                        i += 2
                        continue

            res.append(w)
            i += 1

        cleaned_lines.append(" ".join(res))

    return "\n".join(cleaned_lines)


# ─────────────────────────────────────────────────────────────────────────────
# 4. 과소 분리(붙어버린 어절) 띄어쓰기 분절
# ─────────────────────────────────────────────────────────────────────────────

AUX_PATTERNS = [
    (re.compile(r"([가-힣]{1,})지\s*않(는다|았다|았고|았지만|으면|는|아|고|을|으니|은|았)"), r"\1지 않\2"),
    (re.compile(r"([가-힣]{1,})지\s*못(한다|했다|했고|하지만|하면|하는|해|하고|할|했)"), r"\1지 못\2"),
    (re.compile(r"([가-힣]{1,})수\s*(있|없)([가-힣]*)"), r"\1 수 \2\3"),
    (re.compile(r"([가-힣]{1,}(?:[ㄴ은]))\s*줄\s*(알|몰)(았다|랐다|았고|랐고|아|라|을|는)"), r"\1 줄 \2\3"),
    (re.compile(r"([가-힣]{1,}(?:[ㄴ은ㄹ을던]))\s*것\s*(같|이|은|을|에|도|으|과)"), r"\1 것 \2"),
    (re.compile(r"([가-힣]{1,}(?:[ㄹ을]))\s*때\s*(문|이|는|에|도|까|부|였)"), r"\1 때\2"),
    (re.compile(r"([가-힣]{1,})때문(에|이다|이|을|의)"), r"\1 때문\2"),
    (re.compile(r"([가-힣]{1,})위해(서|서도|는)?"), r"\1 위해\2"),
    (re.compile(r"([가-힣]{1,})통해(서|서도|는)?"), r"\1 통해\2"),
    (re.compile(r"([가-힣]{1,})대해(서|서도|는)?"), r"\1 대해\2"),
    (re.compile(r"([가-힣]{1,})인해(서|서도|는)?"), r"\1 인해\2"),
    (re.compile(r"([0-9]+(?:을|를|이|가|은|는|에|의|로|으로|와|과|도|만))([가-힣]{2,})"), r"\1 \2"),
    (re.compile(r"(통해|위해|대해|인해)([가-힣]{2,})"), r"\1 \2"),
    (re.compile(r"(해야|할지|할까|했을지)몰라"), r"\1 몰라"),
    (re.compile(r"(오늘|내일|어제|지금)\s*(내|나|우리)"), r"\1 \2"),
    (re.compile(r"([가-힣]{2,}[을를])\s*(빛내|위로하|다독이|풍요롭게|살아가|느낄|담았|만날|읽|이끌)"), r"\1 \2"),
    (re.compile(r"([가-힣]{2,}[이가])\s*(필요|부족|충분|가능|어렵|쉽)"), r"\1 \2"),
    (re.compile(r"(더|함께|다시)\s*(깊이|담았|읽|나누|보낼)"), r"\1 \2"),
]

JOSA_LIST = [
    "에서부터", "으로부터", "에게서", "에서는", "에게는", "으로는", "에서도", "에게도",
    "들은무엇을", "들은", "들이", "들을", "들에게",
    "부터", "까지", "으로", "에서", "에게", "처럼", "만큼", "마저", "조차", "이나", "이라", "이며",
    "이란", "에는", "에도", "만을", "만의", "은", "는", "이", "가", "을", "를", "에", "의", "로",
    "와", "과", "도", "만", "서", "란", "든",
]

COMMON_WORDS = {
    "세일즈", "세일즈맨", "마케팅", "입문", "매뉴얼", "연구소", "비즈", "필드", "성공", "해법",
    "고객", "정답", "내용", "초보", "선배", "도움", "결과", "분석", "경쟁력", "정보", "기업",
    "혈관", "공급", "미래", "경영", "핵심", "진단", "사람", "조직", "전문가", "체험", "확산",
    "성과", "창출", "상시", "개선", "체계", "부록", "활용", "트레이닝", "감사",
    "무엇", "어디", "언제", "누구", "어떻게", "왜", "우리", "자신", "모두", "사람들",
    "책", "글", "말", "생각", "마음", "세상", "삶", "인생", "시간", "하루", "시작",
    "끝", "처음", "마지막", "이유", "방법", "문제", "해결", "노력", "희망", "행복",
    "가장", "진짜", "정말", "매우", "너무", "아주", "꼭", "딱히", "벌써", "이미", "다시",
    "입문한", "성공한", "필요한", "해야", "할지", "몰라", "난감할", "것이다", "운이",
    "좋아서", "만난다", "해도", "담고", "있다", "없다", "바란다", "찾기", "만들고", "공급하라",
}

def _segment_long_token(token: str) -> str:
    if len(token) < 5 or not re.match(r"^[가-힣]+$", token):
        return token

    for det in ["모든", "온갖", "이", "그", "저", "새", "헌", "첫"]:
        if token.startswith(det) and len(token) > len(det) + 1:
            rest = token[len(det):]
            return det + " " + _segment_long_token(rest)

    for length in range(min(len(token) - 2, 8), 1, -1):
        prefix = token[:length]
        if prefix in COMMON_WORDS:
            rest = token[length:]
            josa_found = None
            for j in JOSA_LIST:
                if rest.startswith(j) and len(rest) > len(j):
                    rem = rest[len(j):]
                    if len(rem) >= 2:
                        josa_found = j
                        break
            if josa_found:
                split_point = length + len(josa_found)
                return token[:split_point] + " " + _segment_long_token(token[split_point:])
            elif len(rest) >= 2:
                return prefix + " " + _segment_long_token(rest)

    for josa in ["에서", "으로", "에게", "부터", "까지", "을", "를", "에", "의", "로", "와", "과"]:
        idx = token.find(josa)
        if 2 <= idx <= len(token) - len(josa) - 2:
            left = token[:idx + len(josa)]
            right = token[idx + len(josa):]
            if len(right) >= 2:
                return _segment_long_token(left) + " " + _segment_long_token(right)

    return token


def split_concatenated_korean(text: str) -> str:
    """붙어버린 어절, 보조용언, 관형사를 한국어 어법에 맞게 띄어 씁니다."""
    if not text:
        return text

    for pat, repl in AUX_PATTERNS:
        text = pat.sub(repl, text)

    lines = text.splitlines()
    out_lines = []

    for line in lines:
        tokens = line.split(" ")
        new_tokens = []
        for t in tokens:
            m = re.match(r"^([^\w가-힣]*)([가-힣]+)([^\w가-힣]*)$", t)
            if m and len(m.group(2)) >= 6:
                prefix, core, suffix = m.group(1), m.group(2), m.group(3)
                seg = _segment_long_token(core)
                new_tokens.append(f"{prefix}{seg}{suffix}")
            else:
                new_tokens.append(t)
        out_lines.append(" ".join(new_tokens))

    return "\n".join(out_lines)


# ─────────────────────────────────────────────────────────────────────────────
# 5. 올인원 로컬 한국어 정제 함수
# ─────────────────────────────────────────────────────────────────────────────

def refine_korean_text(text: str, document_corpus: str = "") -> str:
    """순수 로컬 알고리즘으로 한국어 띄어쓰기, 맞춤법, 어절 결합을 완결합니다."""
    if not text:
        return text

    text = fix_ocr_quotes_and_symbols(text)
    text = fix_spelling_and_typos(text)
    text = fix_split_tokens(text, document_corpus=document_corpus)
    text = split_concatenated_korean(text)

    # 한국어 조사 밀착 및 어휘 경계 정규화
    text = re.sub(r'([가-힣0-9])\s+(에게|에게는|에게서|에게도|에서는|에서도)', r'\1\2', text)
    text = re.sub(r'([가-힣]{2,})\s+(들은|들이|들을|들에게)', r'\1\2', text)
    text = re.sub(r'([가-힣]{1,}(?:들은|들이|들을|들에게|은|는|이|가|을|를|에|의|로|으로|와|과|도|만))\s*(초보|선배|고객|사람|세일즈|성공|해법|매뉴얼|입문|무엇|어디|누구|어떻게)', r'\1 \2', text)
    text = re.sub(r'(초보|수석|전문|선임|책임)\s*(세일즈|마케팅|개발자|엔지니어|디자이너)', r'\1 \2', text)
    text = re.sub(r'감사의글', '감사의 글', text)
    text = re.sub(r'(부록|목차|참고)([A-Za-z가-힣])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z]+(?:를|을|이|가|은|는|에|의|로|과|와))\s*([가-힣]{2,})', r'\1 \2', text)
    text = re.sub(r'([가-힣]{2,})및([가-힣]{2,})', r'\1 및 \2', text)
    text = re.sub(r'(살아있는|살아 있는|죽어있는)\s*([가-힣]{2,})', r'\1 \2', text)
    text = re.sub(r'(위한|통한|대 한|인한)\s*([가-힣]{2,})', r'\1 \2', text)
    text = re.sub(r'(무엇을|어디서|어떻게|왜)\s*(해야|할지|하는|할까)', r'\1 \2', text)
    text = re.sub(r'(해야)\s*(할지|하는|할까|한다)', r'\1 \2', text)

    text = re.sub(r"(?<=\S) {2,}", " ", text)
    return text
