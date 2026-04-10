"""Tests for build_news_card and build_news_feed in components.cards."""

from __future__ import annotations

import unittest

from dash import html

from components.cards import build_news_card, build_news_feed


class TestBuildNewsCard(unittest.TestCase):
    """Tests for build_news_card covering all datetime parsing branches."""

    def test_valid_iso_datetime(self):
        """Valid ISO 8601 datetime is formatted correctly."""
        card = build_news_card(
            title="Grid Update",
            source="Reuters",
            published_at="2025-07-15T14:30:00+00:00",
            url="https://example.com/1",
        )
        assert isinstance(card, html.A)
        assert card.href == "https://example.com/1"
        assert card.target == "_blank"
        assert card.rel == "noopener noreferrer"
        assert card.className == "news-ribbon-card"

        inner_div = card.children
        meta_div = inner_div.children[1]
        assert "Jul 15, 14:30" in meta_div.children
        assert "Reuters" in meta_div.children

    def test_z_suffix_datetime(self):
        """ISO datetime ending with 'Z' is parsed after replacement."""
        card = build_news_card(
            title="Price Spike",
            source="Bloomberg",
            published_at="2025-01-20T09:15:00Z",
            url="https://example.com/2",
        )
        inner_div = card.children
        meta_div = inner_div.children[1]
        assert "Jan 20, 09:15" in meta_div.children

    def test_invalid_datetime_value_error(self):
        """Malformed datetime string triggers ValueError fallback to truncation."""
        card = build_news_card(
            title="Breaking",
            source="AP",
            published_at="not-a-valid-datetime-string",
            url="https://example.com/3",
        )
        inner_div = card.children
        meta_div = inner_div.children[1]
        # Fallback truncates to first 16 chars
        assert "not-a-valid-date" in meta_div.children

    def test_none_published_at_attribute_error(self):
        """None published_at triggers AttributeError fallback to empty string."""
        card = build_news_card(
            title="Alert",
            source="EIA",
            published_at=None,
            url="https://example.com/4",
        )
        inner_div = card.children
        meta_div = inner_div.children[1]
        # time_str is "" when published_at is None
        assert meta_div.children == "EIA \u00b7 "

    def test_title_and_description_passed(self):
        """Title appears in the card and description param is accepted without error."""
        card = build_news_card(
            title="Renewable Surge",
            source="NYT",
            published_at="2025-06-01T12:00:00Z",
            url="https://example.com/5",
            description="Wind generation hits record.",
        )
        inner_div = card.children
        title_div = inner_div.children[0]
        assert title_div.children == "Renewable Surge"
        assert title_div.className == "news-title"


class TestBuildNewsFeed(unittest.TestCase):
    """Tests for build_news_feed covering empty, single, and multi-article cases."""

    def test_empty_articles_list(self):
        """Empty list returns a 'No news available' placeholder."""
        feed = build_news_feed([])
        assert isinstance(feed, html.Div)
        assert feed.className == "news-ribbon"
        paragraph = feed.children
        assert isinstance(paragraph, html.P)
        assert paragraph.children == "No news available"

    def test_single_article(self):
        """Single article produces duplicated cards (1 + 1 = 2) for seamless looping."""
        articles = [
            {
                "title": "Solar Boom",
                "source": "Reuters",
                "published_at": "2025-03-10T08:00:00Z",
                "url": "https://example.com/solar",
            }
        ]
        feed = build_news_feed(articles)
        assert feed.className == "news-ribbon"

        header = feed.children[0]
        assert header.children == "Grid Signals"
        assert header.className == "news-ribbon-header"

        viewport = feed.children[1]
        assert viewport.className == "news-ticker-viewport"

        track = viewport.children
        assert track.className == "news-ticker-track"
        # 1 card duplicated -> 2 total
        assert len(track.children) == 2

    def test_multiple_articles_card_duplication(self):
        """Multiple articles produce 2N cards (N original + N duplicated)."""
        articles = [
            {
                "title": f"Article {i}",
                "source": "Wire",
                "published_at": f"2025-04-0{i + 1}T10:00:00Z",
                "url": f"https://example.com/{i}",
            }
            for i in range(3)
        ]
        feed = build_news_feed(articles)
        track = feed.children[1].children
        # 3 cards duplicated -> 6 total
        assert len(track.children) == 6

    def test_articles_capped_at_ten(self):
        """Feed uses at most 10 articles even if more are provided."""
        articles = [
            {
                "title": f"Article {i}",
                "source": "Src",
                "published_at": "2025-05-01T00:00:00Z",
                "url": f"https://example.com/{i}",
            }
            for i in range(15)
        ]
        feed = build_news_feed(articles)
        track = feed.children[1].children
        # 10 original + 10 duplicated = 20
        assert len(track.children) == 20


if __name__ == "__main__":
    unittest.main()
