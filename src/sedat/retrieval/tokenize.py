"""Tokenizers for BM25. English: lowercase word tokens minus stopwords.
Chinese: jieba if installed, else character unigrams + bigrams."""
from __future__ import annotations

import re

_EN_STOP = set("""a an and are as at be but by for from has have he her his i if in into is it its
of on or she so than that the their them then there these they this to was we were what when where
which who whom why will with you your do does did how""".split())

_WORD = re.compile(r"\w+", re.UNICODE)
_HAN = re.compile(r"[一-鿿]")


def tokenize_en(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _EN_STOP]


def tokenize_zh(text: str) -> list[str]:
    try:
        import jieba  # type: ignore
        return [t for t in jieba.lcut(text.lower()) if _WORD.fullmatch(t)]
    except ImportError:
        pass
    toks: list[str] = []
    for chunk in _WORD.findall(text.lower()):
        if _HAN.search(chunk):
            chars = list(chunk)
            toks.extend(chars)
            toks.extend(a + b for a, b in zip(chars, chars[1:]))
        else:
            toks.append(chunk)
    return toks


def get_tokenizer(name: str):
    return {"en": tokenize_en, "zh": tokenize_zh}[name]
