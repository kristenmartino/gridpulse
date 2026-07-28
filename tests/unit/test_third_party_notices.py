"""The bundled-asset licence notice must match reality.

`assets/ba_polygons.geojson` was described as MIT-licensed from 2026-05-02
until 2026-07-28. It is not: the upstream repository relicensed to AGPL-3.0 in
January 2023, and the geometries we ship were edited after that date (commit
`83cfc4fe`, "Changes borders of El Paso and ERCOT" — both in our 51-BA set).

A licence notice is a factual claim about someone else's terms, so a wrong one
is worse than a missing one. These tests pin the correction so it cannot be
reverted by a copy-paste from the older docs, and so a future asset refresh
cannot silently re-import the wrong claim.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTICES = REPO_ROOT / "THIRD_PARTY_NOTICES.md"
ASSET = REPO_ROOT / "assets" / "ba_polygons.geojson"

#: Files that describe the asset's provenance in prose. Each must not assert
#: MIT as the operative licence.
PROVENANCE_FILES = (
    "THIRD_PARTY_NOTICES.md",
    "components/_callbacks_us_grid.py",
    "components/tab_us_grid.py",
    "docs/internal/NEXT_UP.md",
)


def _notices() -> str:
    return NOTICES.read_text()


class TestBundledAssetLicence:
    def test_the_asset_still_exists(self):
        """If the asset is ever replaced, these assertions must be revisited
        rather than silently passing against a file that is no longer shipped."""
        assert ASSET.exists(), "ba_polygons.geojson is gone — revisit this notice"

    def test_notice_states_agpl_as_the_operative_licence(self):
        body = _notices()
        assert "AGPL-3.0" in body
        # the operative-licence line, not merely a mention in passing
        assert re.search(r"\*\*Licen[cs]e\*\*:\s*\*\*AGPL-3\.0\*\*", body), (
            "the notice must state AGPL-3.0 as the licence, not just mention it"
        )

    def test_notice_explains_why_the_mit_grant_does_not_apply(self):
        """The MIT text is still reproduced upstream, so a reader who finds it
        needs to know why it does not govern this asset — otherwise the next
        person re-derives the original mistake."""
        body = _notices()
        assert "cb9664f" in body, "cite the relicence commit"
        assert "2023-01-30" in body, "cite the relicence date"
        assert "83cfc4fe" in body, "cite the post-relicence commit touching our BAs"

    def test_notice_records_the_network_copyleft_obligation(self):
        """AGPL §13 is the clause that matters for a served choropleth."""
        assert "§13" in _notices()

    def test_no_file_still_claims_the_asset_is_mit_licensed(self):
        """The claim appeared in four places. Fixing one and leaving three is
        not a fix — it is a contradiction a reader will resolve the wrong way."""
        offenders = []
        for rel in PROVENANCE_FILES:
            text = (REPO_ROOT / rel).read_text()
            for line in text.splitlines():
                if "MIT" not in line:
                    continue
                low = line.lower()
                # allowed: the historical/corrective framing
                if any(
                    k in low
                    for k in ("agpl", "corrected", "previously", "prior to", "pre-", "recorded as")
                ):
                    continue
                if any(
                    k in low for k in ("electricitymap", "polygon", "choropleth", "world.geojson")
                ):
                    offenders.append(f"{rel}: {line.strip()[:90]}")
        assert offenders == [], "stale MIT claim about the bundled asset:\n" + "\n".join(offenders)

    def test_the_remediation_issue_is_linked(self):
        """The notice records the licence; it does not establish compliance.
        The reader needs the pointer to where that is being handled."""
        assert "357" in _notices()
