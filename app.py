"""
NextEra Energy Demand Forecasting Dashboard.

Entry point for the Dash application. Sets up:
- Structured logging via observability module
- Dash app with Bootstrap DARKLY theme
- Custom CSS (assets/custom.css loaded automatically)
- Main layout from components.layout
- All callbacks from components.callbacks
- Health check + metrics endpoints for Cloud Run
- Request logging middleware

Run:
    python app.py
    gunicorn app:server --bind :8080 --workers 2 --timeout 300
"""

import os

from dotenv import load_dotenv

# Load .env file before any other imports that use environment variables
load_dotenv()

# Configure observability FIRST (before any other imports that use structlog)
from observability import add_request_logging, configure_logging  # noqa: E402

configure_logging()

import dash  # noqa: E402
import dash_bootstrap_components as dbc  # noqa: E402
import structlog  # noqa: E402
from flask import Flask, jsonify  # noqa: E402

log = structlog.get_logger()

# Flask server (needed for gunicorn and health/metrics endpoints)
server = Flask(__name__)

# Add request logging middleware
add_request_logging(server)

# Dash app
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="GridPulse",
    update_title="Loading...",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# Layout
from components.layout import build_layout  # noqa: E402

app.layout = build_layout()

# Register all callbacks
from components.callbacks import register_callbacks  # noqa: E402

register_callbacks(app)

# ── Precompute models and predictions at startup ──────────────
# With gunicorn --preload, this runs ONCE in master before workers fork.
# Workers inherit warm caches via copy-on-write.
from config import PRECOMPUTE_ENABLED  # noqa: E402

if PRECOMPUTE_ENABLED:
    from precompute import precompute_all

    precompute_all()


# ── Health check for Cloud Run ─────────────────────────────────
@server.route("/health")
def health():
    from components.callbacks import _BACKTEST_CACHE, _MODEL_CACHE, _PREDICTION_CACHE

    return jsonify(
        {
            "status": "healthy",
            "precompute": {
                "models_cached": len(_MODEL_CACHE),
                "predictions_cached": len(_PREDICTION_CACHE),
                "backtests_cached": len(_BACKTEST_CACHE),
            },
        }
    ), 200


# ── Performance metrics endpoint (internal) ────────────────────
@server.route("/metrics")
def metrics():
    from observability import perf

    return jsonify(perf.get_all_stats()), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    debug = os.getenv("DASH_DEBUG", "false").lower() == "true"
    log.info("starting_dashboard", port=port, debug=debug)
    app.run(debug=debug, host="0.0.0.0", port=port)
