"""Stage 3 text preparation: paragraph rebuilding, boilerplate removal and sentence splitting.

Everything here is part of the frozen scoring specification (docs/stage3_scoring_spec.md).
It uses only the text of the release. Nothing in this module reads prices or returns.
"""
from __future__ import annotations

import re

SPEC_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Step 1: paragraphs
# ---------------------------------------------------------------------------
_EXHIBIT_HEADER = re.compile(r"^((EX-?99[\w.\-]*|EXHIBIT[\s\-]*99[\w.]*|\d{1,2}|[\w\-.]+\.html?)\s*)+$", re.I)
BULLET_CHARS = "•▪●■➢‣·∙◦▫⁃"
_BULLET_ONLY = re.compile(rf"^[{BULLET_CHARS}\-–—o*]$")
_BULLET_START = re.compile(rf"^[{BULLET_CHARS}]\s*")
_TERMINAL = re.compile(r"[.!?][\"'”’)\]]*$")
JOIN_MIN_WORDS = 12
BULLET = "• "


def paragraphs(text: str) -> list[str]:
    """Rebuild paragraphs from the lines of the table-free text.

    1. SGML/exhibit header lines at the top are dropped.
    2. A line holding only a bullet glyph marks the next line as a bullet; any other line without a letter
       (page numbers, "###", "- 30 -") is dropped.
    3. Hard-wrapped lines are joined: a line is appended to the paragraph before it when that paragraph does
       not end a sentence (. ! ?) and either has at least JOIN_MIN_WORDS words or the new line starts with a
       lower-case letter. A bullet always starts a new paragraph.
    """
    out: list[str] = []
    pending_bullet = False
    for line in text.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if len(out) < 6 and len(line.split()) <= 6 and _EXHIBIT_HEADER.match(line):
            continue
        if not re.search(r"[A-Za-z]", line):
            pending_bullet = bool(_BULLET_ONLY.match(line))
            continue
        is_bullet = pending_bullet or bool(_BULLET_START.match(line))
        pending_bullet = False
        if is_bullet:
            out.append(BULLET + _BULLET_START.sub("", line))
            continue
        if out and not _TERMINAL.search(out[-1]) and (len(out[-1].split()) >= JOIN_MIN_WORDS or line[0].islower()):
            out[-1] = out[-1] + " " + line
        else:
            out.append(line)
    return out


# ---------------------------------------------------------------------------
# Step 2: sentence splitting (rule-based, no external model, so it cannot drift)
# ---------------------------------------------------------------------------
ABBREVIATIONS = {
    "inc", "corp", "co", "ltd", "llc", "plc", "lp", "n.v", "s.a", "u.s", "u.k", "e.u", "no", "nos", "vs", "v",
    "approx", "est", "e.g", "i.e", "etc", "mr", "mrs", "ms", "dr", "jr", "sr", "st", "ft", "mt", "dept", "div",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "fig", "cf", "al",
    "u.s.a", "a.m", "p.m", "incl", "excl", "avg", "yr", "qtr", "mil", "bil", "mm", "bn", "pct", "ph.d",
}
_BOUNDARY = re.compile(r"([.!?])([\"'”’)\]]*)\s+(?=[\"'“‘(\[]*[A-Z0-9$])")
_INLINE_BULLET = re.compile(rf"\s*[{BULLET_CHARS}]\s*|\s+[–—−]\s+(?=[A-Z])")


def split_sentences(par: str) -> list[str]:
    """Split one paragraph into sentences. Bullets and spaced dashes before a capital start a new sentence."""
    out: list[str] = []
    for piece in _INLINE_BULLET.split(par):
        piece = piece.strip()
        if not piece:
            continue
        start = 0
        for m in _BOUNDARY.finditer(piece):
            if m.group(1) == ".":
                before = piece[start:m.start()]
                last = before.split()[-1] if before.split() else ""
                token = last.lower().lstrip("(\"'“")
                # no split after an abbreviation, a single initial, or dotted initials such as "U.S"
                if token in ABBREVIATIONS or re.fullmatch(r"[a-z]", token) or re.fullmatch(r"([a-z]\.)+[a-z]", token):
                    continue
            end = m.end(2)
            out.append(piece[start:end].strip())
            start = m.end()
        tail = piece[start:].strip()
        if tail:
            out.append(tail)
    return out


# ---------------------------------------------------------------------------
# Step 3: safe-harbor / forward-looking-statements removal
# ---------------------------------------------------------------------------
TRIGGER = re.compile(
    r"forward[\s\-]+looking\s+(statement|information)|private\s+securities\s+litigation\s+reform\s+act|"
    r"safe[\s\-]+harbou?r|cautionary\s+(statement|note|language|notice)",
    re.I,
)
# Wording that continues a safe-harbor passage (risk-factor lists and disclaimers).
CONTINUATION = re.compile(
    r"risks?\s+and\s+uncertainties|actual\s+results|could\s+cause|may\s+cause|differ\s+materially|"
    r"undertakes?\s+no\s+(obligation|duty)|no\s+(duty|obligation)\s+(or\s+obligation\s+)?to\s+update|"
    r"disclaims?\s+any\s+(obligation|duty|intent)|"
    r"risk\s+factors|form\s+10-[kq]|annual\s+report\s+on\s+form|quarterly\s+reports?\s+on\s+form|"
    r"securities\s+and\s+exchange\s+commission|\bSEC\b|speaks?\s+only\s+as\s+of|as\s+of\s+the\s+date|"
    r"factors\s+(that|which)\s+(could|may|might)|important\s+factors|include,?\s+(but|among)|among\s+other(s|\s+things)|"
    r"words\s+such\s+as|\b(anticipates?|believes?|expects?|intends?|estimates?|projects?),\s",
    re.I,
)
HEADING_MAX_WORDS = 15


def is_heading(par: str) -> bool:
    """Short non-bullet line with no sentence-ending punctuation: treated as a section heading."""
    return (not par.startswith(BULLET)) and len(par.split()) <= HEADING_MAX_WORDS \
        and not par.rstrip().endswith((".", ";", ",", ":"))


def _continues(par: str) -> bool:
    """Does this paragraph carry on a safe-harbor passage? Bullets, lower-case starts and disclaimer wording do."""
    return bool(par.startswith(BULLET) or par[0].islower() or CONTINUATION.search(par) or TRIGGER.search(par))


def remove_safe_harbor(pars: list[str]) -> tuple[list[str], list[str]]:
    """Return (kept paragraphs, removed text pieces).

    (a) Heading with a trigger phrase ("Forward-Looking Statements", "Cautionary Statement ..."): the heading
        and the first paragraph under it are removed, then every following paragraph for as long as it
        continues the passage (see _continues) and is not a new heading.
    (b) Body paragraph with a trigger phrase: the sentences before the first trigger sentence are kept; that
        sentence and the rest of the paragraph are removed, then following paragraphs as in (a).
    """
    kept: list[str] = []
    removed: list[str] = []
    i, n = 0, len(pars)
    while i < n:
        par = pars[i]
        if not TRIGGER.search(par):
            kept.append(par)
            i += 1
            continue
        j = i + 1
        if is_heading(par):
            removed.append(par)
            if j < n and not is_heading(pars[j]):
                removed.append(pars[j])
                j += 1
        else:
            sents = split_sentences(par)
            hits = [k for k, s in enumerate(sents) if TRIGGER.search(s)]
            first = hits[0] if hits else 0     # trigger split across sentences: remove the whole paragraph
            if first > 0:
                kept.append(" ".join(sents[:first]))
            removed.append(" ".join(sents[first:]))
        while j < n and not is_heading(pars[j]) and _continues(pars[j]):
            removed.append(pars[j])
            j += 1
        i = j
    return kept, removed


# ---------------------------------------------------------------------------
# Step 4: sentence filter
# ---------------------------------------------------------------------------
MIN_WORDS = 5
MAX_WORDS = 120
MIN_ALPHA_SHARE = 0.5
MAX_NUMERIC_TOKEN_SHARE = 0.4
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def keep_sentence(s: str) -> bool:
    """Prose test. A sentence is scored only if it has MIN_WORDS to MAX_WORDS alphabetic words, at least half
    of its non-space characters are letters, and at most 40% of its tokens contain a digit. The upper limits
    remove the debris of tables that were laid out as text rather than as HTML tables."""
    n = len(_WORD.findall(s))
    if n < MIN_WORDS or n > MAX_WORDS:
        return False
    compact = re.sub(r"\s+", "", s)
    if sum(ch.isalpha() for ch in compact) / max(len(compact), 1) < MIN_ALPHA_SHARE:
        return False
    toks = s.split()
    return sum(any(ch.isdigit() for ch in t) for t in toks) / len(toks) <= MAX_NUMERIC_TOKEN_SHARE


def count_words(text: str) -> int:
    return len(_WORD.findall(text))


def prepare(table_free_text: str, remove_safe_harbor_text: bool) -> dict:
    """Paragraphs -> (optional) safe-harbor removal -> sentences -> filter. Returns sentences and word counts."""
    pars = paragraphs(table_free_text)
    kept_pars, removed = remove_safe_harbor(pars) if remove_safe_harbor_text else (pars, [])
    sents_all = [s for p in kept_pars for s in split_sentences(p)]
    sents = [s for s in sents_all if keep_sentence(s)]
    return {
        "sentences": sents,
        "words_paragraphs": sum(count_words(p) for p in pars),
        "words_safe_harbor_removed": sum(count_words(p) for p in removed),
        "pieces_safe_harbor_removed": len(removed),
        "words_sentence_filter_removed": sum(count_words(s) for s in sents_all if not keep_sentence(s)),
        "words_scored": sum(count_words(s) for s in sents),
        "n_sentences": len(sents),
    }
