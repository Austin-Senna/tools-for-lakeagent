"""
Text processing utilities — tokenisation, case conversion, stop‑word removal.

Direct port of ``dataanalysis/nlp_utils.py``.
"""

from __future__ import annotations

import re

__all__ = [
    "camelcase_to_snakecase",
    "tokenize_property",
    "curate_tokens",
    "curate_string",
]


def camelcase_to_snakecase(term: str) -> str:
    """Convert ``CamelCase`` (or ``camelCase``) to ``snake_case``.

    >>> camelcase_to_snakecase("MyFieldName")
    'my_field_name'
    """
    tmp = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", term)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", tmp).lower()


def tokenize_property(prop: str) -> list[str]:
    """Split a property name on camelCase / snake_case / kebab‑case boundaries.

    >>> tokenize_property("myField-Name")
    ['my', 'field', 'name']
    """
    snake = camelcase_to_snakecase(prop)
    snake = snake.replace("_", " ").replace("-", " ")
    return snake.split()


def curate_tokens(tokens: list[str]) -> list[str]:
    """Lowercase, deduplicate, and remove short / stop‑word tokens.

    Stop words are loaded lazily from NLTK to avoid a hard dependency
    at import time.
    """
    try:
        from nltk.corpus import stopwords
        stop = set(stopwords.words("english"))
    except (ImportError, LookupError):
        stop = set()
    tokens = list({w.lower() for w in tokens if len(w) > 1 and w.lower() not in stop})
    return tokens


def curate_string(string: str) -> str:
    """Normalise a string by converting to snake_case then replacing separators with spaces."""
    snake = camelcase_to_snakecase(string)
    return snake.replace("_", " ").replace("-", " ").lower()
