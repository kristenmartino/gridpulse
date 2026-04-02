"""
Energy news client using Google News RSS.

Fetches headlines related to energy, utilities, renewable energy,
and grid operations. No API key required.
"""

from defusedxml.ElementTree import fromstring as _safe_xml_fromstring
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime

import requests
import structlog

from data.cache import get_cache

log = structlog.get_logger()

# Google News RSS search query for energy topics
_GOOGLE_NEWS_RSS = (
    "https://news.google.com/rss/search?"
    "q=electricity+grid+OR+renewable+energy+OR+solar+power+OR+wind+power"
    "+OR+power+demand+OR+energy+prices+OR+ERCOT+OR+CAISO+OR+power+grid"
    "&hl=en-US&gl=US&ceid=US:en"
)


def fetch_energy_news(
    query: str | None = None,
    page_size: int = 10,
) -> list[dict]:
    """
    Fetch energy-related news articles from Google News RSS.

    Args:
        query: Unused (kept for API compatibility).
        page_size: Number of articles to return.

    Returns:
        List of article dicts with keys: title, description, url, source,
        published_at, image_url.
    """
    cache = get_cache()
    cache_key = f"news:google_rss:{page_size}"
    cached = cache.get(cache_key)
    if cached is not None:
        log.info("news_cache_hit", key=cache_key)
        return cached

    log.info("news_fetching_rss", page_size=page_size)

    try:
        response = requests.get(_GOOGLE_NEWS_RSS, timeout=10)
        response.raise_for_status()

        root = _safe_xml_fromstring(response.content)
        channel = root.find("channel")
        if channel is None:
            log.error("news_rss_no_channel")
            return _get_demo_news()

        articles = []
        for item in channel.findall("item"):
            if len(articles) >= page_size:
                break

            raw_title = item.findtext("title", "")
            # Google News titles end with " - Source Name"
            parts = raw_title.rsplit(" - ", 1)
            title = parts[0] if parts else raw_title
            source = parts[1] if len(parts) > 1 else "Google News"

            # Also check <source> element
            source_el = item.find("source")
            if source_el is not None and source_el.text:
                source = source_el.text

            link = item.findtext("link", "")
            pub_date = item.findtext("pubDate", "")

            # Parse RFC 2822 date
            published_at = ""
            if pub_date:
                try:
                    dt = parsedate_to_datetime(pub_date)
                    published_at = dt.isoformat()
                except (ValueError, TypeError):
                    published_at = pub_date

            articles.append(
                {
                    "title": title,
                    "description": "",
                    "url": link,
                    "source": source,
                    "published_at": published_at,
                    "image_url": None,
                }
            )

        if not articles:
            return _get_demo_news()

        log.info("news_fetched", articles=len(articles))
        cache.set(cache_key, articles, ttl=1800)
        return articles

    except (requests.RequestException, ET.ParseError) as e:
        log.error("news_fetch_failed", error=str(e))
        return _get_demo_news()


def _get_demo_news() -> list[dict]:
    """Return demo news articles when RSS feed is unavailable."""
    now = datetime.now(UTC)
    return [
        {
            "title": "ERCOT Solar Generation Hits New Records as Texas Grid Evolves",
            "description": "",
            "url": "https://www.eia.gov/todayinenergy/detail.php?id=66464",
            "source": "EIA",
            "published_at": (now - timedelta(hours=2)).isoformat(),
            "image_url": None,
        },
        {
            "title": "Natural Gas Prices Fluctuate on Shifting Power Demand",
            "description": "",
            "url": "https://www.eia.gov/outlooks/steo/report/natgas.php",
            "source": "EIA",
            "published_at": (now - timedelta(hours=5)).isoformat(),
            "image_url": None,
        },
        {
            "title": "DOE Announces $1.9B SPARK Investment in Grid Modernization",
            "description": "",
            "url": "https://www.energy.gov/articles/energy-department-announces-19b-investment-critical-grid-infrastructure-reduce-electricity",
            "source": "DOE",
            "published_at": (now - timedelta(hours=8)).isoformat(),
            "image_url": None,
        },
        {
            "title": "FPL Accelerates Grid Hardening Ahead of Hurricane Season",
            "description": "",
            "url": "https://newsroom.nexteraenergy.com/FPL-announces-plan-to-accelerate-strengthening-of-Floridas-electric-grid-during-annual-storm-drill",
            "source": "NextEra Energy",
            "published_at": (now - timedelta(days=1)).isoformat(),
            "image_url": None,
        },
        {
            "title": "Solar Power Drives U.S. Electricity Generation Growth",
            "description": "",
            "url": "https://www.eia.gov/todayinenergy/detail.php?id=67005",
            "source": "EIA",
            "published_at": (now - timedelta(days=1, hours=3)).isoformat(),
            "image_url": None,
        },
    ]
