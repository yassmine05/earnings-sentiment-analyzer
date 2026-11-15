"""Small SEC EDGAR client: rate-limited, retried, cached on disk.

SEC fair-access rules: declared User-Agent, at most 10 requests/second.
We stay at 5 requests/second.
"""
from __future__ import annotations

import hashlib
import html as htmllib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from . import config

MIN_INTERVAL = 0.2  # seconds between requests


@dataclass
class FilingDoc:
    seq: str
    description: str
    href: str
    doc_type: str
    size: int

    @property
    def url(self) -> str:
        return "https://www.sec.gov" + self.href.replace("/ix?doc=", "")


class EdgarClient:
    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or config.EDGAR_CACHE
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": config.sec_user_agent(), "Accept-Encoding": "gzip, deflate"}
        )
        self._last = 0.0
        self.n_requests = 0

    # ---- low level -------------------------------------------------------
    def _cache_path(self, url: str) -> Path:
        u = urlparse(url)
        path = self.cache_dir / u.netloc / u.path.lstrip("/")
        if u.query:  # keep different queries of the same endpoint apart
            digest = hashlib.sha1(u.query.encode()).hexdigest()[:16]
            path = path.with_name(f"{path.name}__{digest}")
        return path

    def get(self, url: str, *, use_cache: bool = True) -> bytes:
        path = self._cache_path(url)
        if use_cache and path.exists():
            return path.read_bytes()
        for attempt in range(6):
            wait = MIN_INTERVAL - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            resp = self.session.get(url, timeout=90)
            self.n_requests += 1
            if resp.status_code == 200:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(resp.content)
                return resp.content
            if resp.status_code in (403, 429, 500, 502, 503, 504):
                time.sleep(2.0 * (attempt + 1) ** 2)
                continue
            resp.raise_for_status()
        raise RuntimeError(f"EDGAR request failed after retries: {url} ({resp.status_code})")

    def get_json(self, url: str, **kw) -> dict:
        return json.loads(self.get(url, **kw))

    def get_text(self, url: str, **kw) -> str:
        return self.get(url, **kw).decode("utf-8", "replace")

    # ---- endpoints -------------------------------------------------------
    def ticker_map(self) -> pd.DataFrame:
        j = self.get_json("https://www.sec.gov/files/company_tickers.json")
        return pd.DataFrame(j.values()).rename(columns={"cik_str": "cik"})

    def submissions(self, cik: int) -> tuple[dict, pd.DataFrame]:
        """Company header and the full filing list (walks the extra index files)."""
        head = self.get_json(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json")
        blocks = [head["filings"]["recent"]]
        for f in head["filings"].get("files", []):
            blocks.append(self.get_json("https://data.sec.gov/submissions/" + f["name"]))
        df = pd.concat([pd.DataFrame(b) for b in blocks], ignore_index=True)
        df = df.drop_duplicates("accessionNumber").reset_index(drop=True)
        return head, df

    def filing_base(self, cik: int, accession: str) -> str:
        return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}"

    def filing_docs(self, cik: int, accession: str) -> list[FilingDoc]:
        """Documents listed on the filing index page, with their exhibit types."""
        html = self.get_text(f"{self.filing_base(cik, accession)}/{accession}-index.htm")
        soup = BeautifulSoup(html, "lxml")
        docs: list[FilingDoc] = []
        for table in soup.find_all("table", class_="tableFile"):
            for tr in table.find_all("tr")[1:]:
                tds = tr.find_all("td")
                if len(tds) < 5:
                    continue
                a = tds[2].find("a")
                if a is None or not a.get("href"):
                    continue
                size = re.sub(r"\D", "", tds[4].get_text())
                docs.append(
                    FilingDoc(
                        seq=tds[0].get_text(strip=True),
                        description=tds[1].get_text(" ", strip=True),
                        href=a["href"],
                        doc_type=tds[3].get_text(strip=True),
                        size=int(size) if size else 0,
                    )
                )
        return docs

    def company_search(self, name: str, form: str = "10-K") -> list[dict]:
        """EDGAR company-name search (matches current and former names). Returns cik + name."""
        url = (
            "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
            f"&company={quote_plus(name)}&type={form}&dateb=&owner=include&count=40&output=atom"
        )
        try:
            xml = self.get_text(url)
        except requests.HTTPError:
            return []
        unesc = lambda s: htmllib.unescape(htmllib.unescape(s)).strip()  # names are double-escaped
        single = re.search(r"<company-info>.*?</company-info>", xml, re.S)
        if single:  # exactly one company matched
            cik = re.search(r"<cik>(\d+)</cik>", single.group(0))
            name = re.search(r"<conformed-name>(.*?)</conformed-name>", single.group(0), re.S)
            return [{"cik": int(cik.group(1)), "edgar_name": unesc(name.group(1))}] if cik and name else []
        out = []
        for entry in re.findall(r"<entry>.*?</entry>", xml, re.S):
            cik = re.search(r"<cik>(\d+)</cik>", entry)
            title = re.search(r"<title>(.*?)</title>", entry, re.S)
            if cik and title:
                out.append({"cik": int(cik.group(1)), "edgar_name": unesc(title.group(1))})
        return out

    def shares_outstanding_facts(self, cik: int) -> pd.DataFrame:
        """dei:EntityCommonStockSharesOutstanding facts (non-dimensional only)."""
        url = (
            f"https://data.sec.gov/api/xbrl/companyconcept/CIK{int(cik):010d}"
            "/dei/EntityCommonStockSharesOutstanding.json"
        )
        try:
            j = self.get_json(url)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return pd.DataFrame(columns=["end", "val", "form", "filed", "accn"])
            raise
        return pd.DataFrame(j.get("units", {}).get("shares", []))
