"""
NewsAPI client for energy-related news.

Fetches top headlines and articles related to energy, utilities,
renewable energy, and grid operations.

API docs: https://newsapi.org/docs
"""

from datetime import datetime, timedelta, timezone

import requests
import structlog

from config import NEWS_API_KEY, NEWS_API_BASE_URL

log = structlog.get_logger()

# Energy-related search terms
ENERGY_KEYWORDS = (
    "electricity grid OR renewable energy OR solar power OR wind power OR "
    "natural gas OR power outage OR energy prices OR utility OR ERCOT OR "
    "power grid OR electricity demand"
)


def fetch_energy_news(
    query: str | None = None,
    page_size: int = 10,
) -> list[dict]:
    """
    Fetch energy-related news articles from NewsAPI.

    News is always fetched fresh (no caching) to ensure up-to-date headlines.

    Args:
        query: Custom search query (default: energy keywords).
        page_size: Number of articles to fetch (max 100).

    Returns:
        List of article dicts with keys: title, description, url, source,
        published_at, image_url.
    """
    if not NEWS_API_KEY:
        log.warning("news_api_key_missing")
        return _get_demo_news()

    log.info("news_fetching", page_size=page_size)

    try:
        # Use everything endpoint for broader search
        url = f"{NEWS_API_BASE_URL}/everything"
        params = {
            "apiKey": NEWS_API_KEY,
            "q": query or ENERGY_KEYWORDS,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(page_size, 100),
            "from": (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d"),
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok":
            log.error("news_api_error", error=data.get("message"))
            return _get_demo_news()

        articles = _parse_articles(data.get("articles", []))
        log.info("news_fetched", articles=len(articles))

        return articles

    except requests.RequestException as e:
        log.error("news_fetch_failed", error=str(e))
        return _get_demo_news()


def _parse_articles(raw_articles: list[dict]) -> list[dict]:
    """Parse raw NewsAPI articles into a cleaner format."""
    articles = []
    for article in raw_articles:
        # Skip removed articles
        if article.get("title") == "[Removed]":
            continue

        articles.append({
            "title": article.get("title", ""),
            "description": article.get("description", ""),
            "url": article.get("url", ""),
            "source": article.get("source", {}).get("name", "Unknown"),
            "published_at": article.get("publishedAt", ""),
            "image_url": article.get("urlToImage"),
        })

    return articles


def _get_demo_news() -> list[dict]:
    """Return demo news articles when API is unavailable."""
    now = datetime.now(timezone.utc)
    return [
        {
            "title": "ERCOT Reports Record Solar Generation Amid Summer Heat",
            "description": "Texas grid operator sees solar power reach new highs as temperatures soar across the state.",
            "url": "#",
            "source": "Energy News Daily",
            "published_at": (now - timedelta(hours=2)).isoformat(),
            "image_url": None,
        },
        {
            "title": "Natural Gas Prices Rise on Increased Power Demand",
            "description": "Wholesale natural gas prices climb as utilities increase generation to meet cooling demand.",
            "url": "#",
            "source": "Market Watch",
            "published_at": (now - timedelta(hours=5)).isoformat(),
            "image_url": None,
        },
        {
            "title": "DOE Announces $2B Investment in Grid Modernization",
            "description": "Federal funding to support transmission upgrades and renewable energy integration projects.",
            "url": "#",
            "source": "Reuters Energy",
            "published_at": (now - timedelta(hours=8)).isoformat(),
            "image_url": None,
        },
        {
            "title": "Florida Utilities Prepare for Hurricane Season",
            "description": "Major utilities announce storm hardening investments and emergency response preparations.",
            "url": "#",
            "source": "Utility Dive",
            "published_at": (now - timedelta(days=1)).isoformat(),
            "image_url": None,
        },
        {
            "title": "California Grid Operator Issues Flex Alert",
            "description": "CAISO asks residents to conserve electricity during evening peak hours.",
            "url": "#",
            "source": "LA Times",
            "published_at": (now - timedelta(days=1, hours=3)).isoformat(),
            "image_url": None,
        },
    ]
