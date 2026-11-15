"""HTML exhibit -> plain text. Financial tables are removed; prose is kept."""
from __future__ import annotations

import re

import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Inline-XBRL exhibits carry an XML declaration; the HTML parser is still the right tool.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

_BLOCK = ["p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table"]


def _is_numeric_table(table) -> bool:
    """A table is treated as financial data when it has several rows and is digit-heavy."""
    if len(table.find_all("tr")) < 3:
        return False
    txt = re.sub(r"\s+", "", table.get_text(" "))
    if len(txt) < 40:
        return False
    digits = sum(ch.isdigit() for ch in txt)
    return digits / len(txt) >= 0.20


def html_to_text(html: str, drop_tables: bool = True) -> str:
    # Some exhibits are plain-text documents wrapped in <TEXT>; lxml copes with both.
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head", "title"]):
        tag.decompose()
    for tag in soup.find_all(re.compile(r"^ix:header$", re.I)):
        tag.decompose()
    if drop_tables:
        # innermost first, so a layout table wrapping prose is not judged by its nested data tables
        for table in reversed(soup.find_all("table")):
            if _is_numeric_table(table):
                table.decompose()
    for tag in soup.find_all(_BLOCK):
        tag.insert_before("\n")
        tag.insert_after("\n")
    text = soup.get_text(" ")
    text = text.replace("\xa0", " ").replace("​", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z][A-Za-z'\-]*", text))
