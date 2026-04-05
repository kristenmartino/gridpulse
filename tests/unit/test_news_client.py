"""Unit tests for data/news_client.py — Google News RSS client.

Covers:
- fetch_energy_news(): cache hit, RSS success, network error fallback, demo fallback
- _get_demo_news(): structure, keys, count
- RSS XML parsing: valid XML, malformed XML, empty feed, missing channel
- Cache interactions: hit returns cached data, miss triggers fetch, result is cached
"""

from unittest.mock import MagicMock, patch
from xml.etree.ElementTree import ParseError

# ---------------------------------------------------------------------------
# Sample RSS XML fixtures
# ---------------------------------------------------------------------------

VALID_RSS_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Google News</title>
    <item>
      <title>ERCOT Demand Surges During Heatwave - Reuters</title>
      <link>https://example.com/article-1</link>
      <pubDate>Sat, 15 Jul 2024 14:00:00 GMT</pubDate>
      <source url="https://reuters.com">Reuters</source>
    </item>
    <item>
      <title>Solar Installations Hit Record High - Bloomberg</title>
      <link>https://example.com/article-2</link>
      <pubDate>Fri, 14 Jul 2024 10:30:00 GMT</pubDate>
    </item>
    <item>
      <title>No Dash Source Suffix</title>
      <link>https://example.com/article-3</link>
      <pubDate>Thu, 13 Jul 2024 08:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""

EMPTY_FEED_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Google News</title>
  </channel>
</rss>
"""

NO_CHANNEL_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
</rss>
"""

MALFORMED_XML = b"""<rss><channel><item><title>Broken"""

# Expected article keys returned by all code paths.
_ARTICLE_KEYS = {"title", "description", "url", "source", "published_at", "image_url"}


# ---------------------------------------------------------------------------
# Helper: build a fake requests.Response
# ---------------------------------------------------------------------------


def _make_response(content: bytes, status_code: int = 200) -> MagicMock:
    """Create a mock requests.Response with given content and status."""
    resp = MagicMock()
    resp.content = content
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


# ===========================================================================
# Tests for _get_demo_news()
# ===========================================================================


class TestGetDemoNews:
    """Verify the demo news fallback produces valid article dicts."""

    def test_returns_list(self):
        """_get_demo_news returns a non-empty list."""
        from data.news_client import _get_demo_news

        result = _get_demo_news()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_correct_article_structure(self):
        """Every demo article has the required keys."""
        from data.news_client import _get_demo_news

        for article in _get_demo_news():
            assert set(article.keys()) == _ARTICLE_KEYS, (
                f"Missing or extra keys: {set(article.keys()) ^ _ARTICLE_KEYS}"
            )

    def test_demo_articles_have_nonempty_titles(self):
        """Each demo article has a non-empty title string."""
        from data.news_client import _get_demo_news

        for article in _get_demo_news():
            assert isinstance(article["title"], str)
            assert len(article["title"]) > 0

    def test_demo_articles_have_valid_sources(self):
        """Each demo article has a non-empty source string."""
        from data.news_client import _get_demo_news

        for article in _get_demo_news():
            assert isinstance(article["source"], str)
            assert len(article["source"]) > 0


# ===========================================================================
# Tests for fetch_energy_news() — cache interactions
# ===========================================================================


class TestFetchEnergyNewsCacheHit:
    """When cache contains fresh data, no HTTP request should be made."""

    @patch("data.news_client.get_cache")
    @patch("data.news_client.requests.get")
    def test_cache_hit_returns_cached_data(self, mock_requests_get, mock_get_cache):
        """Cache hit returns cached list without calling requests.get."""
        from data.news_client import fetch_energy_news

        cached_articles = [
            {
                "title": "Cached Article",
                "description": "",
                "url": "https://cached.example.com",
                "source": "Cache",
                "published_at": "2024-07-15T00:00:00",
                "image_url": None,
            },
        ]
        mock_cache = MagicMock()
        mock_cache.get.return_value = cached_articles
        mock_get_cache.return_value = mock_cache

        result = fetch_energy_news()

        assert result == cached_articles
        mock_cache.get.assert_called_once()
        mock_requests_get.assert_not_called()

    @patch("data.news_client.get_cache")
    @patch("data.news_client.requests.get")
    def test_cache_miss_triggers_fetch(self, mock_requests_get, mock_get_cache):
        """Cache miss triggers an HTTP request to Google News RSS."""
        from data.news_client import fetch_energy_news

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        mock_requests_get.return_value = _make_response(VALID_RSS_XML)

        result = fetch_energy_news()

        mock_requests_get.assert_called_once()
        assert len(result) == 3

    @patch("data.news_client.get_cache")
    @patch("data.news_client.requests.get")
    def test_successful_fetch_caches_result(self, mock_requests_get, mock_get_cache):
        """After a successful RSS fetch, the result is stored in the cache."""
        from data.news_client import fetch_energy_news

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        mock_requests_get.return_value = _make_response(VALID_RSS_XML)

        fetch_energy_news()

        mock_cache.set.assert_called_once()
        cache_key, articles, *_ = mock_cache.set.call_args[0]
        assert "news:google_rss" in cache_key
        assert isinstance(articles, list)
        assert len(articles) == 3


# ===========================================================================
# Tests for fetch_energy_news() — RSS success path
# ===========================================================================


class TestFetchEnergyNewsRSSSuccess:
    """Valid RSS XML is parsed into correctly structured article dicts."""

    @patch("data.news_client.get_cache")
    @patch("data.news_client.requests.get")
    def test_valid_rss_returns_articles(self, mock_requests_get, mock_get_cache):
        """Valid RSS feed returns a list of article dicts with correct keys."""
        from data.news_client import fetch_energy_news

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache
        mock_requests_get.return_value = _make_response(VALID_RSS_XML)

        result = fetch_energy_news()

        assert isinstance(result, list)
        assert len(result) == 3
        for article in result:
            assert set(article.keys()) == _ARTICLE_KEYS

    @patch("data.news_client.get_cache")
    @patch("data.news_client.requests.get")
    def test_source_element_overrides_title_suffix(self, mock_requests_get, mock_get_cache):
        """When <source> element exists, it overrides the ' - Source' title suffix."""
        from data.news_client import fetch_energy_news

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache
        mock_requests_get.return_value = _make_response(VALID_RSS_XML)

        result = fetch_energy_news()

        # First article has <source> element with text "Reuters"
        assert result[0]["source"] == "Reuters"
        assert result[0]["title"] == "ERCOT Demand Surges During Heatwave"

    @patch("data.news_client.get_cache")
    @patch("data.news_client.requests.get")
    def test_title_without_dash_suffix_uses_google_news(self, mock_requests_get, mock_get_cache):
        """Article without ' - Source' in title and no <source> element uses 'Google News'."""
        from data.news_client import fetch_energy_news

        # Feed with a single item that has no dash in title and no <source> element
        single_item_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Google News</title>
    <item>
      <title>Plain Title Without Dash</title>
      <link>https://example.com/plain</link>
      <pubDate>Sat, 15 Jul 2024 14:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>"""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache
        mock_requests_get.return_value = _make_response(single_item_xml)

        result = fetch_energy_news()

        assert len(result) == 1
        assert result[0]["source"] == "Google News"
        assert result[0]["title"] == "Plain Title Without Dash"

    @patch("data.news_client.get_cache")
    @patch("data.news_client.requests.get")
    def test_page_size_limits_articles(self, mock_requests_get, mock_get_cache):
        """page_size parameter limits the number of returned articles."""
        from data.news_client import fetch_energy_news

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache
        mock_requests_get.return_value = _make_response(VALID_RSS_XML)

        result = fetch_energy_news(page_size=2)

        assert len(result) == 2

    @patch("data.news_client.get_cache")
    @patch("data.news_client.requests.get")
    def test_pubdate_parsed_to_iso(self, mock_requests_get, mock_get_cache):
        """RFC 2822 pubDate is converted to ISO 8601 format."""
        from data.news_client import fetch_energy_news

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache
        mock_requests_get.return_value = _make_response(VALID_RSS_XML)

        result = fetch_energy_news()

        # The first article has "Sat, 15 Jul 2024 14:00:00 GMT"
        assert "2024-07-15" in result[0]["published_at"]


# ===========================================================================
# Tests for fetch_energy_news() — error / fallback paths
# ===========================================================================


class TestFetchEnergyNewsFallback:
    """When RSS fetch fails, fetch_energy_news falls back to demo news."""

    @patch("data.news_client.get_cache")
    @patch("data.news_client.requests.get")
    def test_network_error_returns_demo_news(self, mock_requests_get, mock_get_cache):
        """RequestException triggers demo news fallback."""
        import requests as req

        from data.news_client import fetch_energy_news

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache
        mock_requests_get.side_effect = req.ConnectionError("DNS resolution failed")

        result = fetch_energy_news()

        assert isinstance(result, list)
        assert len(result) > 0
        for article in result:
            assert set(article.keys()) == _ARTICLE_KEYS

    @patch("data.news_client.get_cache")
    @patch("data.news_client.requests.get")
    def test_timeout_returns_demo_news(self, mock_requests_get, mock_get_cache):
        """Timeout triggers demo news fallback."""
        import requests as req

        from data.news_client import fetch_energy_news

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache
        mock_requests_get.side_effect = req.Timeout("Request timed out")

        result = fetch_energy_news()

        assert isinstance(result, list)
        assert len(result) > 0

    @patch("data.news_client.get_cache")
    @patch("data.news_client.requests.get")
    def test_malformed_xml_returns_demo_news(self, mock_requests_get, mock_get_cache):
        """Malformed XML triggers demo news fallback via ParseError."""
        from data.news_client import fetch_energy_news

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache
        mock_requests_get.return_value = _make_response(MALFORMED_XML)

        result = fetch_energy_news()

        assert isinstance(result, list)
        assert len(result) > 0
        for article in result:
            assert set(article.keys()) == _ARTICLE_KEYS

    @patch("data.news_client.get_cache")
    @patch("data.news_client.requests.get")
    def test_empty_feed_returns_demo_news(self, mock_requests_get, mock_get_cache):
        """RSS feed with <channel> but no <item> elements falls back to demo news."""
        from data.news_client import fetch_energy_news

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache
        mock_requests_get.return_value = _make_response(EMPTY_FEED_XML)

        result = fetch_energy_news()

        assert isinstance(result, list)
        assert len(result) > 0
        # Should be demo articles since the feed was empty
        assert result[0]["source"] in ("EIA", "DOE", "NextEra Energy")

    @patch("data.news_client.get_cache")
    @patch("data.news_client.requests.get")
    def test_no_channel_element_returns_demo_news(self, mock_requests_get, mock_get_cache):
        """RSS XML without a <channel> element returns demo news."""
        from data.news_client import fetch_energy_news

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache
        mock_requests_get.return_value = _make_response(NO_CHANNEL_XML)

        result = fetch_energy_news()

        assert isinstance(result, list)
        assert len(result) > 0

    @patch("data.news_client.get_cache")
    @patch("data.news_client.requests.get")
    def test_http_error_returns_demo_news(self, mock_requests_get, mock_get_cache):
        """HTTP 500 error triggers demo news fallback."""
        import requests as req

        from data.news_client import fetch_energy_news

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        resp = _make_response(b"Server Error", status_code=500)
        resp.raise_for_status.side_effect = req.HTTPError("500 Server Error")
        mock_requests_get.return_value = resp

        result = fetch_energy_news()

        assert isinstance(result, list)
        assert len(result) > 0


# ===========================================================================
# Tests for XML parsing via defusedxml
# ===========================================================================


class TestRSSXMLParsing:
    """Verify that defusedxml.ElementTree.fromstring is used for parsing."""

    @patch("data.news_client._safe_xml_fromstring")
    @patch("data.news_client.get_cache")
    @patch("data.news_client.requests.get")
    def test_defusedxml_fromstring_is_called(
        self, mock_requests_get, mock_get_cache, mock_fromstring
    ):
        """The RSS response content is passed to defusedxml for safe parsing."""
        from data.news_client import fetch_energy_news

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        mock_response = _make_response(VALID_RSS_XML)
        mock_requests_get.return_value = mock_response

        # Make fromstring raise so we hit the except branch (simpler than
        # building a full mock element tree).
        mock_fromstring.side_effect = ParseError("mocked parse error")

        result = fetch_energy_news()

        mock_fromstring.assert_called_once_with(VALID_RSS_XML)
        # Falls back to demo news on ParseError
        assert isinstance(result, list)
        assert len(result) > 0
