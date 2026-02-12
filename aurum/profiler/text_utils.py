"""
Text preprocessing utilities.

Ported from ``aurum/dataanalysis/nlp_utils.py`` and
``aurum/ontomatch/ss_utils.py``.  These are the cleaning functions that
feed *every* column-name / value-matching pipeline in Aurum.
"""

from __future__ import annotations

import re
from functools import lru_cache

# ---------------------------------------------------------------------------
# Stopword list — inlined to avoid an NLTK download at import time.
# This is the NLTK English stopword list (179 words, unchanged since 2010).
# ---------------------------------------------------------------------------
ENGLISH_STOPWORDS: frozenset[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "ain", "all", "am",
    "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because",
    "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does",
    "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each",
    "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn",
    "hasn't", "have", "haven", "haven't", "having", "he", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
    "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just",
    "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn",
    "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now",
    "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't",
    "she", "she's", "should", "should've", "shouldn", "shouldn't", "so",
    "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "ve", "very", "was",
    "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with", "won",
    "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves",
})

# Pre-compiled regexes (from Aurum's nlp_utils.camelcase_to_snakecase)
_RE_CAMEL_1 = re.compile(r"(.)([A-Z][a-z]+)")
_RE_CAMEL_2 = re.compile(r"([a-z0-9])([A-Z])")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def camelcase_to_snakecase(term: str) -> str:
    """Convert ``camelCase`` or ``PascalCase`` to ``snake_case``.

    Ported verbatim from ``nlp_utils.camelcase_to_snakecase``.

    >>> camelcase_to_snakecase("buildingNameLong")
    'building_name_long'
    >>> camelcase_to_snakecase("HTMLParser")
    'html_parser'
    """
    tmp = _RE_CAMEL_1.sub(r"\1_\2", term)
    return _RE_CAMEL_2.sub(r"\1_\2", tmp).lower()


def curate_string(raw: str) -> str:
    """Normalise a column / table name into space-separated lowercase tokens.

    Pipeline (from ``nlp_utils.curate_string`` + ``ss_utils.minhash`` preprocessing):
        camelCase → snake_case → replace ``_`` and ``-`` with space → lowercase

    >>> curate_string("buildingNameLong")
    'building name long'
    """
    snake = camelcase_to_snakecase(raw)
    return snake.replace("_", " ").replace("-", " ").strip()


def tokenize_name(name: str, *, min_length: int = 2) -> list[str]:
    """Tokenize and clean a column / table name.

    Steps (ported from ``ss_utils.minhash`` and ``nlp_utils.curate_tokens``):
    1. camelCase → snake_case
    2. Replace ``_`` / ``-`` with space
    3. Lowercase
    4. Split on whitespace
    5. Remove English stopwords
    6. Remove tokens shorter than *min_length*

    >>> tokenize_name("ExtGrossArea")
    ['ext', 'gross', 'area']
    >>> tokenize_name("_id")
    ['id']
    """
    curated = curate_string(name)
    return [
        t
        for t in curated.split()
        if len(t) >= min_length and t not in ENGLISH_STOPWORDS
    ]


@lru_cache(maxsize=4096)
def normalise_value(value: str) -> str:
    """Lowercase + strip a cell value for join-key comparison.

    Ported from ``data_processing_utils.join_ab_on_key_optimizer``
    where every join key goes through ``str(x).lower()``.
    """
    return str(value).lower().strip()


def is_probably_numeric_string(token: str) -> bool:
    """Return *True* if *token* looks like a number.

    Used during term-vector filtering (Aurum's
    ``filter_term_vector_by_frequency`` discards numeric tokens).
    """
    try:
        float(token)
        return True
    except ValueError:
        return False


def filter_term_vector(
    term_counts: dict[str, int],
    *,
    min_term_length: int = 4,
    min_frequency: int = 4,
) -> list[str]:
    """Filter a term→count dictionary to keep only meaningful terms.

    Ported from ``elasticstore.get_all_fields_text_signatures`` →
    ``filter_term_vector_by_frequency``.

    Rules:
    - term length > *min_term_length*
    - frequency > *min_frequency*
    - not a pure number (``float(k)`` would succeed)
    - contains no digit characters
    """
    filtered: list[str] = []
    for term, count in term_counts.items():
        if len(term) <= min_term_length:
            continue
        if count <= min_frequency:
            continue
        if is_probably_numeric_string(term):
            continue
        if re.search(r"[0-9]", term):
            continue
        filtered.append(term)
    return filtered
