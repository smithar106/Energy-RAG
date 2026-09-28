"""Text cleaning: strip HTML, normalize whitespace, drop boilerplate."""
from __future__ import annotations

import re

from bs4 import BeautifulSoup


def clean_text(text: str) -> str:
    """Return clean, contiguous prose from arbitrary input.

    - strips HTML via BeautifulSoup (lxml parser)
    - collapses whitespace/newlines
    - removes common citation boilerplate like ``[1]``, ``[edit]``
    """
    if not text:
        return ""

    soup = BeautifulSoup(text, "lxml")
    text = soup.get_text(" ")

    text = re.sub(r"\[(edit|note|citation needed)\d*\]", " ", text, flags=re.I)
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text.strip()
