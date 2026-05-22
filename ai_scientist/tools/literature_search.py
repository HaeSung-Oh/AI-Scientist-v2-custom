import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import requests

from ai_scientist.tools.base_tool import BaseTool


def _trim(text: Optional[str], max_chars: int = 1200) -> str:
    if not text:
        return "No abstract/snippet available."
    text = " ".join(text.split())
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


class ArxivSearchTool(BaseTool):
    def __init__(self, max_results: int = 10):
        super().__init__(
            name="SearchArxiv",
            description=(
                "Search arXiv for recent machine learning and computer vision papers. "
                "Useful for fast-moving 2024-2026 methods."
            ),
            parameters=[
                {
                    "name": "query",
                    "type": "str",
                    "description": "The arXiv search query.",
                }
            ],
        )
        self.max_results = max_results

    def use_tool(self, query: str) -> str:
        rsp = requests.get(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": self.max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
            timeout=30,
        )
        rsp.raise_for_status()

        root = ET.fromstring(rsp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        if not entries:
            return "No arXiv papers found."

        items = []
        for i, entry in enumerate(entries, 1):
            title = _trim(entry.findtext("atom:title", namespaces=ns), 300)
            published = entry.findtext("atom:published", default="Unknown date", namespaces=ns)
            summary = _trim(entry.findtext("atom:summary", namespaces=ns))
            authors = [
                author.findtext("atom:name", default="Unknown", namespaces=ns)
                for author in entry.findall("atom:author", ns)
            ]
            link = entry.findtext("atom:id", default="", namespaces=ns)
            items.append(
                f"{i}: {title}. {', '.join(authors)}. {published[:10]}.\n"
                f"URL: {link}\nAbstract: {summary}"
            )
        return "\n\n".join(items)


class PubMedSearchTool(BaseTool):
    def __init__(self, max_results: int = 10):
        super().__init__(
            name="SearchPubMed",
            description=(
                "Search PubMed for biomedical and medical imaging literature. "
                "Useful for clinical polyp, colonoscopy, and endoscopy evidence."
            ),
            parameters=[
                {
                    "name": "query",
                    "type": "str",
                    "description": "The PubMed search query.",
                }
            ],
        )
        self.max_results = max_results
        self.email = os.getenv("NCBI_EMAIL")

    def use_tool(self, query: str) -> str:
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": self.max_results,
            "retmode": "json",
            "sort": "pub+date",
        }
        if self.email:
            params["email"] = self.email

        search_rsp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params=params,
            timeout=30,
        )
        search_rsp.raise_for_status()
        ids = search_rsp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return "No PubMed papers found."

        fetch_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml",
        }
        if self.email:
            fetch_params["email"] = self.email

        fetch_rsp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params=fetch_params,
            timeout=30,
        )
        fetch_rsp.raise_for_status()
        root = ET.fromstring(fetch_rsp.text)

        items = []
        for i, article in enumerate(root.findall(".//PubmedArticle"), 1):
            pmid = article.findtext(".//PMID", default="Unknown PMID")
            title = _trim(article.findtext(".//ArticleTitle"), 300)
            year = (
                article.findtext(".//PubDate/Year")
                or article.findtext(".//PubDate/MedlineDate")
                or "Unknown year"
            )
            journal = article.findtext(".//Journal/Title", default="Unknown journal")
            abstract_parts = [
                "".join(part.itertext())
                for part in article.findall(".//Abstract/AbstractText")
            ]
            abstract = _trim(" ".join(abstract_parts))
            items.append(
                f"{i}: {title}. {journal}, {year}.\n"
                f"PMID: {pmid}\nAbstract: {abstract}"
            )
        return "\n\n".join(items)


class OpenAlexSearchTool(BaseTool):
    def __init__(self, max_results: int = 10):
        super().__init__(
            name="SearchOpenAlex",
            description=(
                "Search OpenAlex for broad scholarly literature beyond Semantic Scholar. "
                "Results are sorted by publication date."
            ),
            parameters=[
                {
                    "name": "query",
                    "type": "str",
                    "description": "The OpenAlex search query.",
                }
            ],
        )
        self.max_results = max_results

    def use_tool(self, query: str) -> str:
        rsp = requests.get(
            "https://api.openalex.org/works",
            params={
                "search": query,
                "per-page": self.max_results,
                "sort": "publication_date:desc",
            },
            timeout=30,
        )
        rsp.raise_for_status()
        works: List[Dict] = rsp.json().get("results", [])
        if not works:
            return "No OpenAlex works found."

        items = []
        for i, work in enumerate(works, 1):
            authorships = work.get("authorships", [])
            authors = [
                a.get("author", {}).get("display_name", "Unknown")
                for a in authorships[:8]
            ]
            abstract = self._reconstruct_abstract(
                work.get("abstract_inverted_index")
            )
            venue = (
                work.get("primary_location", {})
                .get("source", {})
                .get("display_name")
                or "Unknown venue"
            )
            items.append(
                f"{i}: {work.get('title', 'Unknown title')}. "
                f"{', '.join(authors)}. {venue}, {work.get('publication_year', 'Unknown year')}.\n"
                f"URL: {work.get('id', '')}\nAbstract: {_trim(abstract)}"
            )
        return "\n\n".join(items)

    @staticmethod
    def _reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
        if not inverted_index:
            return ""
        positioned = []
        for word, positions in inverted_index.items():
            positioned.extend((pos, word) for pos in positions)
        return " ".join(word for _, word in sorted(positioned))


class SerpApiSearchTool(BaseTool):
    def __init__(self, scholarly: bool, max_results: int = 10):
        name = "SearchGoogleScholar" if scholarly else "SearchWeb"
        description = (
            "Search Google Scholar through SerpAPI. Requires SERPAPI_API_KEY."
            if scholarly
            else "Search the web through SerpAPI. Requires SERPAPI_API_KEY."
        )
        super().__init__(
            name=name,
            description=description,
            parameters=[
                {
                    "name": "query",
                    "type": "str",
                    "description": "The search query.",
                }
            ],
        )
        self.scholarly = scholarly
        self.max_results = max_results
        self.api_key = os.getenv("SERPAPI_API_KEY")

    def use_tool(self, query: str) -> str:
        if not self.api_key:
            return (
                f"{self.name} unavailable: set SERPAPI_API_KEY to enable this search."
            )

        rsp = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google_scholar" if self.scholarly else "google",
                "q": query,
                "api_key": self.api_key,
                "num": self.max_results,
            },
            timeout=30,
        )
        rsp.raise_for_status()
        data = rsp.json()
        key = "organic_results"
        results = data.get(key, [])[: self.max_results]
        if not results:
            return f"No {self.name} results found."

        items = []
        for i, item in enumerate(results, 1):
            title = item.get("title", "Unknown title")
            link = item.get("link", "")
            snippet = item.get("snippet") or item.get("publication_info", {}).get(
                "summary", ""
            )
            items.append(f"{i}: {title}\nURL: {link}\nSnippet: {_trim(snippet)}")
        return "\n\n".join(items)
