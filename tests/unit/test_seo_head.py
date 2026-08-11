"""Head-markup invariants shared by all three public surfaces.

``landing.py``'s docstring commits to keeping ``/``, ``/about`` and
``/benchmark`` decoupled: each re-declares its own ``<head>`` so dashboard
chrome changes can never break a public page. That decision is kept — a
shared Python head-builder would mean templating two live static routes, and
their titles and descriptions legitimately differ.

**This file is the abstraction instead.** What actually went wrong across the
three surfaces was never editorial, it was mechanical: relative social-image
URLs, a ``summary_large_image`` card with no image, a missing ``lang``. Those
are invariants, and invariants belong in a test rather than in shared markup.

Parsed with stdlib ``html.parser`` on purpose — neither bs4 nor lxml is in
this project's environment and neither should be added for a test.
"""

from __future__ import annotations

import json
import re
import struct
from html.parser import HTMLParser
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_WEB = _ROOT / "web"
_OG_IMAGE = _ROOT / "assets" / "og-image.png"

_CANONICAL_BASE = "https://gridpulse.kristenmartino.ai"

#: Schema.org types the public content honestly supports. Anything else is a
#: claim the pages do not back — see the comment block in web/landing.html.
_ALLOWED_LD_TYPES = {"WebSite", "WebPage", "SoftwareApplication", "Person"}

#: Never permitted in structured data. `offers`/`priceCurrency` are here
#: because a machine-readable price tag is commercial framing that the
#: /about posture suite exists to exclude — and `priceCurrency` would have
#: slipped past its `pricing` substring check, which makes it worse than a
#: clean failure rather than better.
_BANNED_LD = (
    "aggregateRating",
    "review",
    "offers",
    "priceCurrency",
    "ratingValue",
    "Organization",
)


class _HeadParser(HTMLParser):
    """Collects the head facts these tests assert on."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.html_attrs: dict[str, str] = {}
        self.meta: dict[str, str] = {}
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag == "html":
            self.html_attrs = d
        elif tag == "meta":
            key = d.get("name") or d.get("property")
            if key:
                self.meta[key] = d.get("content", "")
        elif tag == "link":
            self.links.append((d.get("rel", ""), d.get("href", "")))


def _index_string() -> str:
    """The Dash shell's head as production renders it.

    Substitutes the sentinel here rather than reading ``app.index_string``,
    which is deliberate: that attribute is built from ``PUBLIC_BASE_URL``,
    which is empty off production and localhost in dev, so every absolute-URL
    assertion below would be vacuous or wrong depending on which test file
    imported ``config`` first. Test-order-dependent tests are worse than no
    tests. That the real substitution actually happens is asserted by
    ``test_redesign_smoke.py::test_index_template_is_not_an_fstring``, and
    that the value is right by ``TestHardcodedHostsStayInSyncWithConfig``.
    """
    import app as app_module

    return app_module._INDEX_TEMPLATE.replace("__BASE__", _CANONICAL_BASE)


def _surfaces() -> list[tuple[str, str]]:
    return [
        ("/", _index_string()),
        ("/about", (_WEB / "landing.html").read_text(encoding="utf-8")),
        ("/methodology", (_WEB / "methodology.html").read_text(encoding="utf-8")),
        ("/benchmark", (_WEB / "benchmark.html").read_text(encoding="utf-8")),
    ]


@pytest.fixture(scope="module", params=_surfaces(), ids=lambda s: s[0])
def surface(request):
    name, html = request.param
    parser = _HeadParser()
    parser.feed(html)
    return name, html, parser


class TestHeadInvariants:
    def test_declares_language(self, surface) -> None:
        """WCAG 3.1.1, not merely an SEO nit — a screen reader has no way to
        pick a pronunciation without it. The Dash shell shipped bare <html>."""
        name, _, p = surface
        assert p.html_attrs.get("lang") == "en", name

    def test_has_exactly_one_absolute_canonical(self, surface) -> None:
        name, _, p = surface
        hrefs = [href for rel, href in p.links if rel == "canonical"]
        assert len(hrefs) == 1, f"{name}: expected 1 canonical, got {hrefs}"
        assert hrefs[0].startswith(_CANONICAL_BASE), f"{name}: {hrefs[0]}"

    def test_og_url_matches_canonical(self, surface) -> None:
        """Two tags naming the same page must not disagree about which URL
        that page is."""
        name, _, p = surface
        canonical = next(href for rel, href in p.links if rel == "canonical")
        assert p.meta.get("og:url") == canonical, name

    def test_social_images_are_absolute(self, surface) -> None:
        """The regression pin for the live defect: the Dash shell used
        relative og:image/twitter:image paths, so every share of the
        production URL unfurled without a card. The Open Graph spec requires
        absolute URLs and no unfurler resolves a relative one."""
        name, _, p = surface
        for key in ("og:image", "twitter:image"):
            value = p.meta.get(key, "")
            assert value.startswith("https://"), f"{name}: {key}={value!r}"

    def test_large_card_actually_supplies_one(self, surface) -> None:
        """summary_large_image with no image renders as a bare link.
        /benchmark declared the type and supplied nothing at all — no image,
        no twitter:title, no twitter:description."""
        name, _, p = surface
        if p.meta.get("twitter:card") != "summary_large_image":
            pytest.skip(f"{name} does not declare a large card")
        for key in ("twitter:image", "twitter:title", "twitter:description"):
            assert p.meta.get(key), f"{name}: {key} missing"

    def test_theme_color_is_the_surface_not_the_accent(self, surface) -> None:
        """theme-color paints mobile browser chrome, so it must be --bg-base.
        The #3b82f6 accent belongs on the mask-icon; pinned so nobody
        'brands' this later."""
        name, _, p = surface
        assert p.meta.get("theme-color") == "#0a0a0b", name

    def test_declared_image_dimensions_match_the_real_file(self, surface) -> None:
        """Read straight from the PNG's IHDR so a swapped image cannot
        silently break every card by contradicting its own declaration."""
        name, _, p = surface
        header = _OG_IMAGE.read_bytes()[16:24]
        width, height = struct.unpack(">II", header)
        assert (width, height) == (1200, 630), "og-image.png is no longer 1200x630"
        if "og:image:width" not in p.meta:
            pytest.skip(f"{name} declares no image dimensions")
        assert p.meta["og:image:width"] == str(width), name
        assert p.meta["og:image:height"] == str(height), name

    def test_description_fits_a_search_result(self, surface) -> None:
        """Google truncates around 155-160 characters. Anything past that is
        written for nobody."""
        name, _, p = surface
        desc = p.meta.get("description", "")
        assert desc, f"{name} has no meta description"
        assert len(desc) <= 165, f"{name}: description is {len(desc)} chars"


class TestStructuredData:
    @staticmethod
    def _blocks(html: str) -> list[str]:
        return re.findall(r'<script type="application/ld\+json">(.*?)</script>', html, re.S)

    def test_every_block_is_valid_json(self, surface) -> None:
        name, html, _ = surface
        blocks = self._blocks(html)
        assert blocks, f"{name} carries no structured data"
        for block in blocks:
            json.loads(block)  # raises on malformed

    def test_declares_the_schema_context(self, surface) -> None:
        name, html, _ = surface
        for block in self._blocks(html):
            assert json.loads(block).get("@context") == "https://schema.org", name

    def test_only_supported_types(self, surface) -> None:
        name, html, _ = surface
        for block in self._blocks(html):
            data = json.loads(block)
            nodes = data.get("@graph", [data])
            for node in nodes:
                assert node.get("@type") in _ALLOWED_LD_TYPES, f"{name}: {node.get('@type')}"

    def test_no_fabricated_or_commercial_claims(self, surface) -> None:
        """The honesty guardrail, in the idiom of
        test_landing.py::test_posture_pins_no_commercial_language — but
        reaching the structured data that prose check cannot see."""
        name, html, _ = surface
        for block in self._blocks(html):
            for banned in _BANNED_LD:
                assert banned not in block, f"{name}: structured data contains {banned!r}"

    def test_referenced_entity_ids_are_defined_somewhere(self, surface) -> None:
        """An @id pointing at a node nobody defines is a dangling reference —
        it consolidates nothing, which is the entire reason these exist."""
        _, html, _ = surface
        defined = set()
        referenced = set()
        for _, page_html in _surfaces():
            for block in self._blocks(page_html):
                data = json.loads(block)
                for node in data.get("@graph", [data]):
                    if "@id" in node:
                        defined.add(node["@id"])
                    for value in node.values():
                        if isinstance(value, dict) and set(value) == {"@id"}:
                            referenced.add(value["@id"])
        assert referenced <= defined, f"dangling @id: {sorted(referenced - defined)}"

    def test_author_is_the_repo_owner(self, surface) -> None:
        """Pinned against the GitHub identity test_landing.py:59 already
        asserts, so the author entity cannot drift away from the one
        verifiable public account."""
        _, html, _ = surface
        for block in self._blocks(html):
            data = json.loads(block)
            for node in data.get("@graph", [data]):
                if node.get("@type") == "Person":
                    assert node["name"] == "Kristen Martino"
                    assert node["url"] == "https://github.com/kristenmartino"


class TestHardcodedHostsStayInSyncWithConfig:
    """What makes keeping the three heads duplicated safe.

    The static pages hardcode the canonical host; the Dash shell substitutes
    it from config. If those ever disagree, one surface starts declaring a
    canonical the others contradict — so assert they cannot.
    """

    def test_static_pages_use_the_configured_public_base_url(self) -> None:
        """Read the matrix directly rather than reloading config under a
        mutated ENVIRONMENT — reloading a module every other test has already
        imported is how a suite acquires order dependence."""
        import config

        assert config._ENV_DEFAULTS["production"]["public_base_url"] == _CANONICAL_BASE

    def test_non_production_tiers_are_not_indexable(self) -> None:
        """A staging box must not compete with production for production's
        own content: empty base URL, seo_indexable False."""
        import config

        for tier in ("development", "staging"):
            assert config._ENV_DEFAULTS[tier]["seo_indexable"] is False, tier
        assert config._ENV_DEFAULTS["staging"]["public_base_url"] == ""

    def test_no_surface_hardcodes_a_different_host(self) -> None:
        for name, html in _surfaces():
            hosts = set(re.findall(r"https://([a-z0-9.-]*gridpulse[a-z0-9.-]*)", html))
            assert hosts <= {"gridpulse.kristenmartino.ai"}, f"{name}: {hosts}"
