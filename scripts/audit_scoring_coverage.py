"""Reconcile a bounded scoring-state capture against matching origin logs.

Offline only. Inputs are the read-only probe's gzip JSON and Cloud Logging
JSON; no forecast is generated or rewritten. Require matching successful
region-completion and origin-resolution events before attributing any gap.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from data.forecast_evidence import utc_timestamp
from models.drift import _within_window, deserialize_records, filter_low_actuals


def _time(value):
    normalized = utc_timestamp(value)
    if normalized is None:
        raise ValueError(f"An aware UTC timestamp is required: {value!r}")
    return datetime.fromisoformat(normalized)


def reconcile(state: dict, logs: list[dict]) -> dict:
    def key(entry):
        return entry["jsonPayload"]["region"], entry["labels"]["run.googleapis.com/execution_name"]

    starts = {key(e): e for e in logs if e["jsonPayload"]["event"] == "forecast_start_resolved"}
    complete = {key(e): e for e in logs if e["jsonPayload"]["event"] == "scoring_region_complete"}
    regressed = {key(e) for e in logs if e["jsonPayload"]["event"] == "forecast_origin_regressed"}
    successful = {k for k, e in complete.items() if e["jsonPayload"].get("ok") is True}
    if successful - starts.keys():
        raise ValueError("Successful region completions are missing matching origin logs")
    if starts.keys() - complete.keys():
        raise ValueError("Origin logs are missing matching region completions")
    origins = defaultdict(set)
    completion_hours = defaultdict(set)
    for k in successful:
        region, _ = k
        completion_hours[region].add(
            _time(complete[k]["timestamp"]).replace(minute=0, second=0, microsecond=0)
        )
        if k not in regressed:
            origins[region].add(_time(starts[k]["jsonPayload"]["forecast_start"]))
    min_log = min(_time(e["timestamp"]) for e in logs)
    max_log = max(_time(e["timestamp"]) for e in logs)
    regions = {}
    for region, data in state["regions"].items():
        payload = data["horizon"]
        now = _time(payload["last_updated_at"])
        if min_log > now - timedelta(hours=168 + 72) or max_log < now:
            raise ValueError("Logs do not cover the required origins and scoring state")
        end = now.replace(minute=0, second=0, microsecond=0)
        targets = {end - timedelta(hours=k) for k in range(168)}
        leads = {}
        for horizon in (24, 48, 72):
            label = f"{horizon}h"
            block = payload["models"]["ensemble"][label]
            records = deserialize_records(block["records"])
            kept, low = filter_low_actuals(_within_window(records, 168, now_iso=now.isoformat()))
            if len(kept) != block["n_7d"]:
                raise ValueError(f"Stored scoring count does not reconcile for {region}/{label}")
            scored = {_time(r.timestamp) for r in kept}
            all_records = {_time(r.timestamp) for r in records}
            pending = {_time(s["target_ts"]) for s in payload["pending"] if s["horizon"] == label}
            counts = Counter()
            for target in targets:
                if target in scored:
                    reason = "scored"
                elif target in all_records:
                    reason = "low_actual_excluded"
                elif target in pending:
                    reason = "awaiting_usable_actual"
                elif target - timedelta(hours=horizon) in origins[region]:
                    reason = "unexplained_missing_sample"
                else:
                    reason = "no_accepted_forecast_origin"
                counts[reason] += 1
            if counts["low_actual_excluded"] != low or sum(counts.values()) != 168:
                raise ValueError("Coverage partition does not reconcile")
            leads[label] = dict(counts)
        regions[region] = {
            "as_of": now.isoformat(),
            "successful_completion_hours": len(targets & completion_hours[region]),
            "horizons": leads,
        }
    totals = {
        label: dict(sum((Counter(r["horizons"][label]) for r in regions.values()), Counter()))
        for label in ("24h", "48h", "72h")
    }
    return {
        "captured_at": state["captured_at"],
        "region_count": len(regions),
        "window_hours_per_region": 168,
        "flags": state["flags"],
        "totals": totals,
        "regions": regions,
        "limitations": [
            "Legacy forecast-start-offset samples, not a publication-lead benchmark.",
            "No accepted origin means no successful non-regressed origin in these matching logs.",
            "Pending samples await usable pipeline actuals; this does not prove the upstream never published demand.",
            "This audit does not reconstruct unavailable historical forecast values.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--logs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    state = json.loads(gzip.decompress(args.state.read_bytes()))
    report = reconcile(state, json.loads(args.logs.read_text()))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("captured_at", "region_count", "totals")}))


if __name__ == "__main__":
    main()
