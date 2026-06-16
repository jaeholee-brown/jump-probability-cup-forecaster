from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class FirecrawlError(RuntimeError):
    pass


@dataclass(frozen=True)
class FirecrawlSearchResult:
    title: str
    url: str
    description: str = ""
    markdown: str = ""
    source: str = "web"

    def compact(self, max_chars: int = 1800) -> str:
        text = self.markdown or self.description
        text = " ".join(text.split())[:max_chars]
        return f"- {self.title} ({self.source}) {self.url}: {text}"


class FirecrawlClient:
    def __init__(self, api_key: str, *, base_url: str = "https://api.firecrawl.dev/v2") -> None:
        if not api_key:
            raise FirecrawlError("FIRECRAWL_API_KEY is required")
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=45.0,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "probability-cup-forecaster/0.1",
            },
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    @retry(
        retry=retry_if_exception_type((FirecrawlError, httpx.TimeoutException, httpx.TransportError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        sources: tuple[str, ...] = ("web", "news"),
        tbs: str | None = None,
        country: str = "US",
    ) -> tuple[list[FirecrawlSearchResult], int]:
        body: dict[str, Any] = {
            "query": query[:500],
            "limit": max(1, min(limit, 20)),
            "sources": list(sources),
            "country": country,
            "ignoreInvalidURLs": True,
            "scrapeOptions": {"formats": [{"type": "markdown"}]},
        }
        if tbs:
            body["tbs"] = tbs
        response = await self.client.post("/search", json=body)
        if response.status_code == 429:
            raise FirecrawlError(f"Firecrawl rate limited search for {query!r}")
        if response.status_code >= 400:
            raise FirecrawlError(f"Firecrawl search failed: {response.status_code} {response.text[:500]}")
        payload = response.json()
        data = payload.get("data") or {}
        results: list[FirecrawlSearchResult] = []
        for source in ("web", "news"):
            for item in data.get(source, []) or []:
                results.append(
                    FirecrawlSearchResult(
                        title=str(item.get("title") or ""),
                        url=str(item.get("url") or ""),
                        description=str(item.get("description") or item.get("snippet") or ""),
                        markdown=str(item.get("markdown") or ""),
                        source=source,
                    )
                )
        return [result for result in results if result.url], int(payload.get("creditsUsed") or 0)
