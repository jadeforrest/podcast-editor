"""Word normalisation shared by matching code."""

import re

_NON_WORD = re.compile(r"[^\w']+")


def norm(token: str) -> str:
    """Case- and punctuation-insensitive form used to match words.

    Tokens that are pure punctuation keep their raw form so they still match themselves.
    """
    lowered = token.lower().replace("’", "'")
    stripped = _NON_WORD.sub("", lowered).strip("'")
    return stripped or token
