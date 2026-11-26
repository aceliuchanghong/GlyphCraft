from __future__ import annotations

from pathlib import Path
from typing import List, Optional

# 94 个标准可打印 ASCII 字符（0x21 '!' ~ 0x7E '~'，不含空格 0x20）
ASCII_94: str = "".join(chr(cp) for cp in range(33, 127))

UPPER_26: str = "".join(chr(cp) for cp in range(65, 91))
LOWER_26: str = "".join(chr(cp) for cp in range(97, 123))
DIGITS_10: str = "".join(chr(cp) for cp in range(48, 58))
CORE_ALNUM_62: str = UPPER_26 + LOWER_26 + DIGITS_10
PUNCT_32: str = "".join(ch for ch in ASCII_94 if ch not in CORE_ALNUM_62)

# 用于检测大小写拓扑差异的典型非对称字母对（正常西文字体中大小写形态必然明显不同）
DISTINCT_CASE_PAIRS = (
    ("A", "a"),
    ("B", "b"),
    ("D", "d"),
    ("E", "e"),
    ("G", "g"),
    ("H", "h"),
    ("N", "n"),
    ("Q", "q"),
    ("R", "r"),
)


def glyph_stem(char: str) -> str:
    """
    将单个字符编码为跨平台绝对安全的文件名（不含扩展名）。
    94 个可打印 ASCII 统一编码为 u0021 ~ u007E，彻底避免 Windows 大小写碰撞与非法路径标点问题，
    且天然按 ASCII 码点升序排列。
    """
    if len(char) != 1:
        raise ValueError(f"glyph_stem 仅接受单个字符，收到 {char!r}")
    return f"u{ord(char):04X}"


def stem_to_char(stem: str) -> Optional[str]:
    """将 uXXXX 文件名反向解码为字符；同时兼容单字符文件名。"""
    if len(stem) == 1 and 33 <= ord(stem) <= 126:
        return stem
    if len(stem) == 5 and stem[0] in ("u", "U"):
        hex_part = stem[1:]
        if all(c in "0123456789ABCDEFabcdef" for c in hex_part):
            cp = int(hex_part, 16)
            if 33 <= cp <= 126:
                return chr(cp)
    return None


def deduplicate_ascii_chars(text: str) -> List[str]:
    """去重并保留顺序，仅保留可打印 ASCII（0x21~0x7E）。"""
    out: List[str] = []
    seen = set()
    for ch in text:
        cp = ord(ch)
        if cp < 33 or cp > 126 or ch in seen:
            continue
        seen.add(ch)
        out.append(ch)
    return out


def load_ascii_charset(
    char_file: Optional[str | Path] = None,
    chars: Optional[str] = None,
) -> List[str]:
    """加载英文字符表；若均未指定，默认返回完整 94 可打印 ASCII。"""
    if chars is not None:
        result = deduplicate_ascii_chars(chars)
    elif char_file is not None:
        result = deduplicate_ascii_chars(Path(char_file).read_text(encoding="utf-8"))
    else:
        result = list(ASCII_94)
    if not result:
        raise ValueError("有效 ASCII 字符表为空。")
    return result
