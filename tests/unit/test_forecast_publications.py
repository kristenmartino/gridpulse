"""Publication receipts must describe actually served values and survive retries."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from google.api_core.exceptions import PreconditionFailed

from data.forecast_publications import (
    archive_publication,
    export_publication,
    prepare_publication,
)


def published_payload():
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    return {
        "region": "FPL",
        "scored_at": now.isoformat(),
        "primary_model": "xgboost",
        "served_series": "seasonal-naive",
        "served_reason": "recorded skill policy",
        "skill": {"compared_series": "ensemble"},
        "anchor": {"anchor_ts": (now - timedelta(hours=1)).isoformat()},
        "forecasts": [
            {
                "timestamp": (now + timedelta(hours=i)).isoformat(),
                "predicted_demand_mw": 100.0 + i,
                "baseline": 100.0 + i,
                "xgboost": 200.0 + i,
                "ensemble": 300.0 + i,
            }
            for i in range(48)
        ],
    }


def test_published_values_not_ensemble_and_identity_covers_times_and_values():
    payload = published_payload()
    record = prepare_publication(payload)
    assert record["forecast"][0]["demand_mw"] == 100
    assert record["series_source"] == "baseline"
    assert record["baseline_selection"]["selected_since"] is None
    assert prepare_publication(deepcopy(payload))["publication_id"] == record["publication_id"]
    regenerated = deepcopy(payload)
    regenerated["scored_at"] = (
        datetime.fromisoformat(payload["scored_at"]) + timedelta(seconds=1)
    ).isoformat()
    assert prepare_publication(regenerated)["publication_id"] != record["publication_id"]
    moved = deepcopy(payload)
    moved["forecasts"][0]["predicted_demand_mw"] += 1
    assert prepare_publication(moved)["publication_id"] != record["publication_id"]
    shifted = deepcopy(payload)
    for row in shifted["forecasts"]:
        row["timestamp"] = (
            datetime.fromisoformat(row["timestamp"]) + timedelta(hours=1)
        ).isoformat()
    assert prepare_publication(shifted)["publication_id"] != record["publication_id"]


@pytest.mark.parametrize(
    "defect", ["gap", "duplicate", "nan", "naive", "no-generation", "nonpilot"]
)
def test_bad_or_unqualified_publication_is_not_archived(defect):
    payload = published_payload()
    if defect == "gap":
        payload["forecasts"].pop(3)
    elif defect == "duplicate":
        payload["forecasts"][3]["timestamp"] = payload["forecasts"][2]["timestamp"]
    elif defect == "nan":
        payload["forecasts"][3]["predicted_demand_mw"] = float("nan")
    elif defect == "naive":
        payload["forecasts"][3]["timestamp"] = "2026-10-25T03:00:00"
    elif defect == "no-generation":
        payload.pop("scored_at")
    else:
        payload["region"] = "ERCOT"
    assert prepare_publication(payload) is None


def test_create_only_receipt_and_retry_preserves_first_observation():
    import json

    record = prepare_publication(published_payload())
    first = record["timing"]["generated_at"]
    blob = MagicMock()
    blob.upload_from_string.side_effect = [None, PreconditionFailed("already exists")]
    blob.download_as_text.return_value = json.dumps(dict(record, observed_published_at=first))
    client = MagicMock()
    client.bucket.return_value.blob.return_value = blob
    with (
        patch("config.GCS_BUCKET_NAME", "test-bucket"),
        patch("data.gcs_store._get_client", return_value=client),
        patch("data.forecast_publications._cache_receipt") as cache,
    ):
        assert archive_publication(record, observed_at=first)
        assert archive_publication(record, observed_at=datetime.now(UTC).isoformat())
    assert blob.upload_from_string.call_args.kwargs["if_generation_match"] == 0
    assert cache.call_args.args[0]["observed_published_at"] == first


def test_archive_failure_is_visible_and_does_not_publish_cache_receipt():
    record = prepare_publication(published_payload())
    client = MagicMock()
    client.bucket.return_value.blob.return_value.upload_from_string.side_effect = RuntimeError(
        "offline"
    )
    with (
        patch("config.GCS_BUCKET_NAME", "test-bucket"),
        patch("data.gcs_store._get_client", return_value=client),
        patch("data.forecast_publications._cache_receipt") as cache,
    ):
        assert archive_publication(record) is False
    cache.assert_not_called()


def test_corrupted_identity_cannot_be_archived():
    record = prepare_publication(published_payload())
    record["forecast"][0]["demand_mw"] += 1
    with patch("data.gcs_store._get_client") as client:
        assert archive_publication(record) is False
    client.assert_not_called()


def test_public_export_never_leaks_future_internal_fields():
    record = prepare_publication(published_payload())
    record["debug"] = "private"
    record["timing"]["debug"] = "private"
    record["forecast"][0]["debug"] = "private"
    record["baseline_selection"]["debug"] = "private"
    assert "private" not in str(export_publication(record))


def test_withheld_tail_is_not_presented_as_published_available_values():
    payload = published_payload()
    payload["horizon_guard"] = {"baseline": {"max_ok_horizon": 24}}
    record = prepare_publication(payload)
    assert len(record["forecast"]) == record["timing"]["target_hours"] == 24
    payload["horizon_guard"]["baseline"]["max_ok_horizon"] = 0
    assert prepare_publication(payload) is None


def test_index_watch_retry_preserves_a_concurrently_added_issue():
    import json

    from redis.exceptions import WatchError

    from data.forecast_publications import _cache_receipt, publication_summary

    first = prepare_publication(published_payload())
    first["observed_published_at"] = first["timing"]["generated_at"]
    other = deepcopy(first)
    other["publication_id"] = "a" * 64
    pipe = MagicMock()
    pipe.get.side_effect = [None, json.dumps([publication_summary(other)])]
    pipe.execute.side_effect = [WatchError("concurrent writer"), []]
    client = MagicMock()
    client.pipeline.return_value.__enter__.return_value = pipe
    with (
        patch("data.redis_client._get_redis", return_value=client),
        patch("data.redis_client.persist"),
    ):
        _cache_receipt(first)
    ids = {r["publication_id"] for r in json.loads(pipe.set.call_args.args[1])}
    assert ids == {first["publication_id"], other["publication_id"]}
