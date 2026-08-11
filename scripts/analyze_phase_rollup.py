#!/usr/bin/env python3
"""Analyse ``scoring_phase_rollup`` logs — phase shares, sub-step attribution,
and the archive cache's paired miss/hit arms.

Written because this analysis was done by hand four times in one day and got
the arithmetic wrong three of them: a denominator taken from a different tick
(``forecast`` published at 91% when it was 57.7%), a "~3x" drop that was
1.81x, and an archive-leg estimate low by 4.2-5.9x. Every one of those was a
number that looked plausible and was never checked against the payload it came
from. The script exists to make the checking automatic.

Three properties it enforces, each mapping to a mistake that was actually made:

* **Never mixes ticks.** Every derived figure is computed inside one payload.
  Combining a ``phases`` block from one tick with ``fetch_substeps`` from
  another is the error that produced the bad denominator.
* **Always prints n.** The archive cache's MISS arm accrues at exactly one
  observation per UTC day, because the window only moves at 00Z. A full day of
  logs is 23 hits and *one* miss — the miss arm is the binding constraint, and
  a report that hides it invites a verdict off n=1.
* **Refuses to conclude when the arms are thin**, per
  ``docs/EVALUATION_POLICY.md``: rolling origin, never one window.

Usage::

    gcloud logging read \\
      'resource.type="cloud_run_job"
       AND resource.labels.job_name="gridpulse-scoring-job"
       AND jsonPayload.event="scoring_phase_rollup"' \\
      --project=nextera-portfolio --freshness=7d --limit 200 \\
      --format=json > rollup.json
    python scripts/analyze_phase_rollup.py rollup.json

``--freshness`` is NOT optional, and leaving it off is the trap this script
was written to catch rather than fall into. ``gcloud logging read`` defaults
to **1 day**, silently — so ``--limit 200`` without it returns ~24 ticks
containing exactly **one** 00Z miss, and the archive verdict can never come
back anything but ``INCONCLUSIVE`` no matter how large the limit. The limit
bounds the rows; the freshness bounds the *days*, and days are what the miss
arm is counted in. Seven days is ~168 ticks, so ``--limit 200`` is sized to
not truncate the window it asks for.

Also accepts the ``--format='value(...)'`` text that gcloud prints when you
ask for named fields, and reads stdin when given no path.
"""

from __future__ import annotations

import ast
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from typing import Any

#: Sub-step channels, and the phase each one must roll up against. The pairing
#: is what makes the invariant checkable: a sub-step can never exceed its own
#: phase total, and if it does, the instrumentation is wrong, not the phase.
SUBSTEP_CHANNELS: dict[str, str] = {
    "fetch_substeps": "fetch",
    "forecast_substeps": "forecast",
    "generation_substeps": "generation",
    "interchange_substeps": "interchange",
}

#: The archive window moves at 00Z, so the first tick of each UTC day is a
#: forced cache miss and the rest are hits. That is the paired contrast — same
#: code, same BAs, an hour apart.
MISS_ARM_HOUR = 0

_DICT_FIELD = re.compile(r"(\w+)=(\{[^{}]*\})")


@dataclass
class Tick:
    """One ``scoring_phase_rollup`` payload. Nothing here crosses ticks."""

    timestamp: str
    elapsed_s: float | None
    phases: dict[str, dict] = field(default_factory=dict)
    substeps: dict[str, dict[str, dict]] = field(default_factory=dict)

    @property
    def hour(self) -> int | None:
        """UTC hour, used only to identify the forced-miss arm."""
        m = re.search(r"T(\d{2}):", self.timestamp)
        return int(m.group(1)) if m else None

    @property
    def worker_s(self) -> float:
        """Summed worker time = the phases, added up in THIS tick.

        Not read from any field and never borrowed from another payload —
        that borrowing is exactly what published `forecast` at 91%.
        """
        return sum(float(v.get("total_s", 0.0)) for v in self.phases.values())

    def leg(self, channel: str, name: str) -> float | None:
        entry = self.substeps.get(channel, {}).get(name)
        return None if entry is None else float(entry.get("total_s", 0.0))

    def leg_n(self, channel: str, name: str) -> int | None:
        entry = self.substeps.get(channel, {}).get(name)
        return None if entry is None else entry.get("n")


def _coerce_blocks(payload: dict[str, Any]) -> Tick:
    tick = Tick(
        timestamp=str(payload.get("timestamp", "?")),
        elapsed_s=_maybe_float(payload.get("elapsed_s")),
        phases=payload.get("phases") or {},
    )
    for channel in SUBSTEP_CHANNELS:
        block = payload.get(channel)
        if isinstance(block, dict) and block:
            tick.substeps[channel] = block
    return tick


def _maybe_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse(text: str) -> list[Tick]:
    """Parse gcloud ``--format=json`` or ``--format='value(...)'`` output."""
    text = text.strip()
    if not text:
        return []

    if text.startswith("[") or text.startswith("{"):
        raw = json.loads(text)
        entries = raw if isinstance(raw, list) else [raw]
        out = []
        for e in entries:
            payload = e.get("jsonPayload", e)
            # gcloud puts the entry timestamp outside jsonPayload
            payload = {**payload, "timestamp": payload.get("timestamp") or e.get("timestamp")}
            out.append(_coerce_blocks(payload))
        return out

    # value() text: one tick per line, tab-separated fields, dict fields
    # printed as Python literals joined by ';'.
    ticks: list[Tick] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        payload: dict[str, Any] = {}
        cols = line.split("\t")
        for col in cols:
            col = col.strip()
            if not col:
                continue
            if _DICT_FIELD.search(col):
                blocks: dict[str, dict] = {}
                for name, literal in _DICT_FIELD.findall(col):
                    try:
                        blocks[name] = ast.literal_eval(literal)
                    except (ValueError, SyntaxError):
                        continue
                # value() loses the field NAME, so infer the channel from the
                # leg names it contains rather than from position.
                payload[_infer_channel(blocks)] = blocks
            elif re.fullmatch(r"\d{4}-\d{2}-\d{2}T[\d:.]+Z?", col):
                payload["timestamp"] = col
            elif _maybe_float(col) is not None and "elapsed_s" not in payload:
                payload["elapsed_s"] = _maybe_float(col)
        ticks.append(_coerce_blocks(payload))
    return ticks


def _infer_channel(blocks: dict[str, dict]) -> str:
    """Name a value()-printed block from its leg names.

    ``value()`` prints the dict without its field name, so position is the only
    other cue — and position silently shifts when someone reorders the format
    string. Inferring from content survives that.
    """
    keys = set(blocks)
    if keys & {"eia_demand", "weather_archive", "weather_forecast", "weather_nbm"}:
        return "fetch_substeps"
    if "eia_generation" in keys:
        return "generation_substeps"
    if "eia_interchange" in keys:
        return "interchange_substeps"
    if keys & {"predict_xgboost", "frame_climatology", "future_frame"}:
        return "forecast_substeps"
    return "phases"


# ── report ───────────────────────────────────────────────────


def _fmt_pct(part: float, whole: float) -> str:
    return f"{part / whole * 100:5.1f}%" if whole else "    ?"


def report_phases(t: Tick) -> None:
    total = t.worker_s
    print(f"\n── {t.timestamp}   elapsed {t.elapsed_s}s   summed worker {total:.1f}s")
    if t.elapsed_s:
        print(f"   effective parallelism {total / t.elapsed_s:.2f}x")
    for name, v in sorted(t.phases.items(), key=lambda kv: -float(kv[1].get("total_s", 0))):
        secs = float(v.get("total_s", 0.0))
        if secs < 1.0:
            continue
        print(
            f"   {name:22s} {secs:8.1f}s {_fmt_pct(secs, total)}"
            f"   n={v.get('n', '?')}  slowest={v.get('slowest_region', '?')}"
        )


def report_attribution(t: Tick) -> list[str]:
    """Sub-step vs its own phase. Returns invariant violations."""
    problems: list[str] = []
    for channel, phase_name in SUBSTEP_CHANNELS.items():
        block = t.substeps.get(channel)
        if not block:
            continue
        phase_total = float(t.phases.get(phase_name, {}).get("total_s", 0.0)) or None
        print(
            f"\n   {channel}  (phase `{phase_name}`"
            + (f" = {phase_total:.1f}s)" if phase_total else ")")
        )
        sub_sum = 0.0
        for leg, v in sorted(block.items(), key=lambda kv: -float(kv[1].get("total_s", 0))):
            secs = float(v.get("total_s", 0.0))
            sub_sum += secs
            share = f" {_fmt_pct(secs, phase_total)} of phase" if phase_total else ""
            print(f"     {leg:22s} {secs:8.1f}s{share}   n={v.get('n', '?')}")
            if phase_total and secs > phase_total * 1.001:
                problems.append(
                    f"{channel}.{leg} = {secs:.1f}s EXCEEDS phase `{phase_name}` "
                    f"({phase_total:.1f}s) — instrumentation bug, not a real number"
                )
        if phase_total and channel != "forecast_substeps":
            # forecast_substeps contains `future_frame`, a wrapper that
            # double-counts its own children, so its sum is not meaningful.
            print(f"     {'(remainder = our own work)':22s} {phase_total - sub_sum:8.1f}s")
    return problems


#: A tick this many times the median elapsed is an upstream event, not a
#: measurement of our code. Deliberately loose — the 2026-08-10T13:19 tick was
#: 2.8x, and anything near 2x is already unmistakable in this job.
OUTLIER_ELAPSED_FACTOR = 2.0


def flag_outlier_ticks(ticks: list[Tick], factor: float = OUTLIER_ELAPSED_FACTOR) -> set[int]:
    """Ticks whose wall clock says an upstream dependency was in trouble.

    Added after this script MISSED one. Its only confound check keyed on
    ``n != 51``, and the 2026-08-10T13:19 tick had all 51 regions while taking
    2.8x the median elapsed with `eia_generation` at 4531.5s — a real EIA
    degradation event that the check waved through.

    Excluding these is the CONSERVATIVE direction, not a convenient one: that
    tick's `weather_archive` was 6.4s, the lowest hit-arm value in the window,
    so pooling it makes the cache look BETTER. A confound that flatters the
    result is the one most worth removing.

    Needs >=3 ticks for the median to mean anything; returns an empty set
    below that rather than guessing.
    """
    vals = [t.elapsed_s for t in ticks if t.elapsed_s]
    if len(vals) < 3:
        return set()
    threshold = statistics.median(vals) * factor
    return {id(t) for t in ticks if t.elapsed_s and t.elapsed_s >= threshold}


def report_archive_arms(ticks: list[Tick], excluded: set[int] | None = None) -> None:
    """The archive cache's paired arms — the whole point of the flag flip."""
    print("\n" + "=" * 68)
    print("ARCHIVE CACHE — paired arms (window moves at 00Z)")
    print("=" * 68)

    excluded = excluded or set()
    obs = [(t, t.leg("fetch_substeps", "weather_archive")) for t in ticks if id(t) not in excluded]
    obs = [(t, v) for t, v in obs if v is not None]
    if excluded:
        print(f"  ({len(excluded)} tick(s) excluded as upstream events — see CONFOUNDS below)")
    if not obs:
        print("  no `fetch_substeps.weather_archive` in these ticks — is the")
        print("  instrumentation deployed? (it shipped in #414)")
        return

    miss = [(t, v) for t, v in obs if t.hour == MISS_ARM_HOUR]
    hit = [(t, v) for t, v in obs if t.hour != MISS_ARM_HOUR]

    for label, arm in (("MISS (00Z, forced refetch)", miss), ("HIT  (all other hours)", hit)):
        if not arm:
            print(f"  {label}: n=0")
            continue
        vals = [v for _, v in arm]
        med = statistics.median(vals)
        print(
            f"  {label}: n={len(vals):2d}  median {med:7.1f}s  "
            f"range {min(vals):.1f}-{max(vals):.1f}s"
        )

    print("\n  Uncached baseline measured 2026-08-07T12:07:58Z: 294.0s")

    if not miss or not hit:
        print("\n  VERDICT: none. Both arms are needed and one is empty.")
        return

    m_med = statistics.median([v for _, v in miss])
    h_med = statistics.median([v for _, v in hit])
    print(f"\n  miss - hit = {m_med - h_med:.1f}s per tick (summed worker time)")

    # The binding constraint, stated rather than buried: the miss arm gains
    # exactly one observation per UTC day.
    if len(miss) < 3:
        print(
            f"\n  VERDICT: INCONCLUSIVE. The MISS arm is n={len(miss)}, and it grows at"
            f"\n  one observation per UTC DAY — the window only moves at 00Z. Several"
            f"\n  days of pairs, not one (docs/EVALUATION_POLICY.md). A day of logs"
            f"\n  looks like a lot of data and contains one miss."
        )
    else:
        print(f"\n  Both arms n>=3 (miss n={len(miss)}). Report the paired difference,")
        print("  and check upstream counters before attributing it to the cache.")


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        with open(argv[1]) as fh:
            text = fh.read()
    else:
        text = sys.stdin.read()
    ticks = parse(text)
    if not ticks:
        print("no scoring_phase_rollup payloads found", file=sys.stderr)
        return 1

    print(f"parsed {len(ticks)} tick(s)")
    problems: list[str] = []
    for t in ticks:
        if t.phases:
            report_phases(t)
        if t.substeps:
            problems += report_attribution(t)

    outliers = flag_outlier_ticks(ticks)
    report_archive_arms(ticks, excluded=outliers)

    # Confounds worth seeing before anyone reads a verdict off the numbers.
    if outliers:
        med = statistics.median([t.elapsed_s for t in ticks if t.elapsed_s])
        print("\n  CONFOUNDS — upstream events, EXCLUDED from the arms above")
        print(f"  (elapsed >= {OUTLIER_ELAPSED_FACTOR}x the {med:.0f}s median):")
        for t in ticks:
            if id(t) in outliers:
                worst = max(
                    ((n, float(v.get("total_s", 0))) for n, v in t.phases.items()),
                    key=lambda kv: kv[1],
                    default=("?", 0.0),
                )
                print(
                    f"    {t.timestamp}  elapsed {t.elapsed_s}s "
                    f"({t.elapsed_s / med:.1f}x)  worst phase: {worst[0]} {worst[1]:.0f}s"
                )
    partial = [t for t in ticks if any(v.get("n", 51) not in (51, None) for v in t.phases.values())]
    if partial:
        print(
            f"\n  NOTE: {len(partial)} tick(s) have a phase with n != 51 — regions that"
            "\n  never ran that phase (early fetch fallback, or a deadline shed)."
            "\n  Those ticks are FAST for a reason unrelated to any change."
        )
    if problems:
        print("\n  INVARIANT VIOLATIONS:")
        for p in problems:
            print(f"    - {p}")
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
