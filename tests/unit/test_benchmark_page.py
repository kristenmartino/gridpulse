"""The public forecast benchmark: ``/api/v1/benchmark`` and ``/benchmark``.

This is the surface an accuracy claim gets attacked on, so the tests pin the
things that would make it indefensible rather than merely broken:

1. **Both official arms and both verdicts publish.** Shipping only the
   favourable scoring is the exact objection the dual arm exists to pre-empt
   (``docs/BENCHMARK_METHODOLOGY.md`` §6).
2. **Exclusions publish with their reason** instead of vanishing — an
   excluded BA that simply disappears reads as a hidden loss.
3. **Drop counts survive to the trust boundary.** They are how a reader sees
   the exclusions are uneven across BAs (§4).
4. **A cold benchmark returns 503, never a fabricated row.**
5. **The page's posture** — no commercial language, no combat framing, and
   the limits section is present, because a benchmark page without its limits
   is marketing.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pytest
from flask import Flask

import api as api_module
import landing
from api import api_v1
from landing import landing_bp


@pytest.fixture(autouse=True)
def _clear_memo():
    api_module._memo.clear()
    yield
    api_module._memo.clear()


@pytest.fixture()
def client():
    app = Flask(__name__)
    app.register_blueprint(api_v1)
    app.register_blueprint(landing_bp)
    return app.test_client()


def _lead_block(**over):
    block = {
        "scoreable": True,
        "n": 640,
        "official": {"mape": 4.2, "median_ape": 3.1, "mae": 812.0, "wape": 3.9, "n": 640},
        "official_revised": {
            "mape": 4.0,
            "median_ape": 3.0,
            "mae": 795.0,
            "wape": 3.7,
            "n": 640,
        },
        "gridpulse": {"mape": 3.5, "median_ape": 3.3, "mae": 690.0, "wape": 3.2, "n": 640},
        "delta_mape": 0.7,
        "delta_wape": 0.7,
        "delta_median_ape": -0.2,
        "delta_mape_vs_revised": 0.5,
        "winner": "gridpulse",
        "winner_vs_revised": "gridpulse",
        "excluded_hours": {
            "unresolved_stub": 146,
            "first_seen_placeholder": 0,
            "unsettled": 4,
            "no_df": 71,
            "no_gridpulse": 12,
        },
        "observed_lead_h": 23.92,
        "lead_basis": "observed",
        # An internal field that must NOT cross the trust boundary.
        "_debug_scratch": {"raw": [1, 2, 3]},
    }
    block.update(over)
    return block


def _payload(region="PJM", **over):
    payload = {
        "region": region,
        "revision_class": "unknown",
        "mean_revision_pct": 0.4,
        "scoreable": True,
        "reason": None,
        "reason_detail": None,
        "df_coverage": 0.929,
        "placeholder_pct": 1.39,
        "scored_at": "2026-07-27T15:00:00+00:00",
        "leads": {"24h": _lead_block(), "48h": _lead_block()},
    }
    payload.update(over)
    return payload


def _excluded(region="LDWP"):
    return {
        "region": region,
        "revision_class": "broken",
        "scoreable": False,
        "reason": "broken-feed",
        "reason_detail": "This feed's provisional readings revise heavily before settling…",
        "df_coverage": 0.91,
        "placeholder_pct": 3.2,
        "leads": {},
    }


def _redis(payloads, fleet=None):
    """Route ``redis_get`` by key so one test can hold several regions."""
    by_key = {f"gridpulse:benchmark:{p['region']}": p for p in payloads}
    if fleet is not None:
        by_key["gridpulse:meta:benchmark_fleet"] = fleet

    def _get(key):
        return by_key.get(key)

    return _get


_FLEET = {
    "updated_at": "2026-07-27T15:00:00+00:00",
    "excluded": [{"region": "LDWP", "reason": "broken-feed", "_internal": "x"}],
    "n_scoreable": 44,
    "n_excluded": 7,
    "fleet": {
        "n": 43,
        "wins": 22,
        "losses": 21,
        "median_gridpulse_mape": 3.5,
        "median_official_mape": 3.05,
        "gridpulse_spread": {"min": 2.1, "max": 8.4, "ratio": 4.0},
        "official_spread": {"min": 1.15, "max": 47.21, "ratio": 41.1},
    },
    "isolated": {"ERCOT": dict(_lead_block(), _leak="must not publish")},
}


class TestBenchmarkEndpoint:
    @patch("api.redis_get")
    def test_publishes_both_official_arms_and_both_verdicts(self, mock_get, client) -> None:
        """Publishing only the as-issued scoring invites 'you graded their
        stale number'. Both arms and both winners must cross the boundary."""
        mock_get.side_effect = _redis([_payload()], _FLEET)
        body = client.get("/api/v1/benchmark").get_json()
        lead = body["regions"][0]["leads"]["24h"]

        assert lead["official"]["mape"] == 4.2
        assert lead["official_revised"]["mape"] == 4.0
        assert lead["winner"] == "gridpulse"
        assert lead["winner_vs_revised"] == "gridpulse"
        assert lead["delta_mape_vs_revised"] == 0.5

    @patch("api.redis_get")
    def test_every_arm_carries_median_alongside_the_mean(self, mock_get, client) -> None:
        """The verdict is a mean; the offline reports are medians. A consumer
        that can only see one of them will compare the wrong things."""
        mock_get.side_effect = _redis([_payload()], _FLEET)
        body = client.get("/api/v1/benchmark").get_json()
        lead = body["regions"][0]["leads"]["24h"]
        for arm in ("official", "official_revised", "gridpulse"):
            assert "mape" in lead[arm] and "median_ape" in lead[arm]
        assert set(body["statistics"]) >= {"mape", "median_ape", "mae", "wape"}

    @patch("api.redis_get")
    def test_drop_counts_reach_the_trust_boundary(self, mock_get, client) -> None:
        """The drops are not neutral across BAs. A reader who cannot see them
        has to take the sample on trust."""
        mock_get.side_effect = _redis([_payload()], _FLEET)
        body = client.get("/api/v1/benchmark").get_json()
        drops = body["regions"][0]["leads"]["24h"]["excluded_hours"]
        assert drops["unresolved_stub"] == 146
        assert drops["no_gridpulse"] == 12

    @patch("api.redis_get")
    def test_internal_fields_never_auto_publish(self, mock_get, client) -> None:
        """The Redis payload is a cache schema, not a contract — a future
        debug field must not appear on a public endpoint by default."""
        mock_get.side_effect = _redis([_payload()], _FLEET)
        body = client.get("/api/v1/benchmark").get_json()
        assert "_debug_scratch" not in body["regions"][0]["leads"]["24h"]

    @patch("api.redis_get")
    def test_excluded_regions_publish_with_their_reason(self, mock_get, client) -> None:
        """An excluded BA that silently vanishes reads as a hidden loss."""
        mock_get.side_effect = _redis([_payload(), _excluded()], _FLEET)
        body = client.get("/api/v1/benchmark").get_json()
        rows = {r["region"]: r for r in body["regions"]}
        assert rows["LDWP"]["scoreable"] is False
        assert rows["LDWP"]["reason"] == "broken-feed"
        assert rows["LDWP"]["reason_detail"]

    @patch("api.redis_get")
    def test_lead_basis_travels_with_the_lead(self, mock_get, client) -> None:
        """A nominal lead must be labelled nominal — the page is forbidden
        from quoting a lead figure without it."""
        mock_get.side_effect = _redis(
            [_payload(leads={"24h": _lead_block(observed_lead_h=None, lead_basis="nominal")})],
            _FLEET,
        )
        lead = client.get("/api/v1/benchmark").get_json()["regions"][0]["leads"]["24h"]
        assert lead["lead_basis"] == "nominal"
        assert lead["observed_lead_h"] is None

    @patch("api.redis_get", return_value=None)
    def test_cold_benchmark_warms_rather_than_fabricates(self, _mock_get, client) -> None:
        resp = client.get("/api/v1/benchmark")
        assert resp.status_code == 503
        assert resp.get_json()["status"] == "warming"

    @patch("api.redis_get")
    def test_per_region_route_and_unknown_region(self, mock_get, client) -> None:
        mock_get.side_effect = _redis([_payload()], _FLEET)
        assert client.get("/api/v1/benchmark/PJM").get_json()["region"] == "PJM"

        resp = client.get("/api/v1/benchmark/NOT_A_BA")
        assert resp.status_code == 404
        assert "NOT_A_BA" not in resp.get_data(as_text=True)

    @patch("api.redis_get")
    def test_notes_disclose_the_stub_rule_and_the_lead_caveat(self, mock_get, client) -> None:
        """The two things a hostile reader checks first: did you score the
        placeholder hours, and is the lead matched?"""
        mock_get.side_effect = _redis([_payload()], _FLEET)
        notes = " ".join(client.get("/api/v1/benchmark").get_json()["notes"]).lower()
        assert "as the actual" in notes
        assert "not lead-matched" in notes
        assert "mean mape" in notes

    @patch("api.redis_get")
    def test_isolated_block_goes_through_the_same_allow_list(self, mock_get, client) -> None:
        """ERCOT is reported separately, which must not make it a back door
        for fields every other region's block is filtered for."""
        mock_get.side_effect = _redis([_payload()], _FLEET)
        isolated = client.get("/api/v1/benchmark").get_json()["isolated"]
        assert isolated["ERCOT"]["n"] == 640
        assert "_leak" not in isolated["ERCOT"]
        assert "_debug_scratch" not in isolated["ERCOT"]

    @patch("api.redis_get")
    def test_exclusion_list_is_exported_so_the_count_can_be_reconciled(
        self, mock_get, client
    ) -> None:
        """A bare count asks the reader to trust it. The named list lets them
        check it against the rows — and carries nothing else."""
        mock_get.side_effect = _redis([_payload(), _excluded()], _FLEET)
        excluded = client.get("/api/v1/benchmark").get_json()["excluded"]
        assert excluded == [{"region": "LDWP", "reason": "broken-feed"}]

    @patch("api.redis_get")
    def test_per_region_row_carries_its_own_freshness(self, mock_get, client) -> None:
        """The fleet key stamps the rollup, not a row — and the per-region
        route never reads the fleet key at all."""
        mock_get.side_effect = _redis([_payload()], _FLEET)
        assert client.get("/api/v1/benchmark/PJM").get_json()["scored_at"]


class TestBenchmarkPageRoute:
    def test_serves_html_200(self, client) -> None:
        resp = client.get("/benchmark")
        assert resp.status_code == 200
        assert resp.mimetype == "text/html"

    def test_missing_file_degrades_to_404(self, client, monkeypatch) -> None:
        monkeypatch.setattr(landing, "_BENCHMARK_HTML", Path("/nonexistent/benchmark.html"))
        assert client.get("/benchmark").status_code == 404

    def test_about_page_still_served(self, client) -> None:
        """The two routes now share a helper — a refactor must not break the
        page that was already live."""
        assert client.get("/about").status_code == 200


@pytest.fixture()
def body() -> str:
    return (Path(__file__).resolve().parents[2] / "web" / "benchmark.html").read_text()


class TestBenchmarkPagePosture:
    def test_reads_from_the_public_endpoint_only(self, body) -> None:
        """The page must be renderable from what the API will admit to — no
        second, friendlier data path."""
        assert "/api/v1/benchmark" in body
        assert len(re.findall(r"fetch\(", body)) == 1

    def test_limits_section_is_present(self, body) -> None:
        """A benchmark page without its limits is marketing. These are the
        two that cut in our favour, so they are the two that must be here."""
        lowered = body.lower()
        assert "not lead-matched" in lowered
        assert "no matured" in lowered and "prediction" in lowered
        assert "methodology" in lowered

    def test_exclusions_are_a_section_not_a_footnote(self, body) -> None:
        assert 'id="excluded"' in body
        assert "excluded-h" in body

    def test_every_metric_column_names_its_statistic_and_window(self, body) -> None:
        """§8's rule as a test: no per-BA figure without metric, window, n
        and arm."""
        assert "mean MAPE %" in body
        assert "30-day window" in body
        assert "paired hours" in body

    def test_posture_pins_no_commercial_or_combat_language(self, body) -> None:
        """Same guardrail as the /about page. An accuracy comparison is the
        surface most likely to drift into a boast, so pin it here too."""
        lowered = body.lower()
        for banned in (
            "request a demo",
            "schedule a call",
            "contact sales",
            "pricing",
            "beats the",
            "outperforms",
            "best-in-class",
            "state of the art",
            "industry-leading",
        ):
            assert banned not in lowered, f"posture pin violated: {banned!r}"

    def test_external_links_open_in_a_new_tab(self, body) -> None:
        for href in re.findall(r'<a [^>]*href="(https://[^"]+)"[^>]*>', body):
            match = re.search(
                r'<a [^>]*href="' + re.escape(href) + r'"[^>]*>',
                body,
            )
            assert match and 'target="_blank"' in match.group(0), href
            assert 'rel="noopener"' in match.group(0), href

    def test_exclusions_and_still_accumulating_are_separate_sections(self, body) -> None:
        """Conflating them is dishonest in both directions: a fairness
        exclusion is permanent and reasoned, a thin sample is just young —
        and early in a deploy the second group is the larger one, which would
        make the "most of these are broken feeds" lede false."""
        assert 'id="pending-section"' in body
        assert 'id="excluded-section"' in body
        assert "still accumulating" in body.lower()
        assert "has not lost" in body.lower()
        # the split keys off the published reason, not off "has no verdict"
        assert "FAIRNESS_REASONS" in body
        assert "'broken-feed', 'df-coverage'" in body

    def test_never_calls_it_their_day_ahead_forecast(self, body) -> None:
        """§12.1 is absolute: a revision landing before our first capture is
        invisible to us, so the page may only claim the earliest value it
        observed. This is the surface that rule was written for."""
        lowered = body.lower()
        assert "earliest day-ahead we observed" in lowered
        assert "their day-ahead" not in lowered
        assert "their forecast" not in lowered

    def test_focus_indicator_is_opaque(self, body) -> None:
        """--accent-ring is 30% alpha; as a focus outline it measures ~1.5:1
        against this background and fails WCAG 1.4.11."""
        assert "outline: 2px solid var(--accent-base);" in body
        assert "outline: 2px solid var(--accent-ring);" not in body

    def test_sort_state_is_exposed_to_assistive_tech(self, body) -> None:
        """The sort glyph is invisible to a screen reader; aria-sort is the
        only signal the table reordered."""
        assert body.count('aria-sort="none"') == 6
        assert "markSortState" in body

    def test_loading_placeholder_does_not_persist_without_javascript(self, body) -> None:
        """It used to ship visible, so a JS-off reader saw "Loading…" forever
        next to a <noscript> notice saying it would never load."""
        assert re.search(r'<div id="state"[^>]*\bhidden\b', body)
        assert "state.hidden = false;" in body

    def test_spread_tile_names_its_statistic_and_population(self, body) -> None:
        """A bare ratio invites confusion with the median-APE spread in the
        scoreability report — a different statistic over different hours."""
        assert "worst ÷ best BA mean MAPE" in body
        assert "excl. ERCOT" in body
