"""
GridPulse scheduled jobs.

This package hosts the two Cloud Run Jobs that own the data/model pipeline
off the web request path:

- ``jobs.scoring_job``  — hourly: refreshes actuals, weather, forecasts,
  alerts, and diagnostics in Redis.
- ``jobs.training_job`` — daily: retrains XGBoost / Prophet / SARIMAX per
  region, persists artifacts to GCS, and recomputes backtests.

Both entry points are invoked via ``python -m jobs {scoring|training}``
(see :mod:`jobs.__main__`). The Dash web service never imports these
modules — it only reads from Redis.
"""

from __future__ import annotations
