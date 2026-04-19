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
server.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(32).hex())

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

# Google Analytics (GA4)
app.index_string = """<!DOCTYPE html>
<html>
  <head>
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-97LE6K3X9N"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-97LE6K3X9N');
    </script>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
  </head>
  <body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
  </body>
</html>"""

# Layout
from components.layout import build_layout  # noqa: E402

app.layout = build_layout()

# Register all callbacks
from components.callbacks import register_callbacks  # noqa: E402

register_callbacks(app)

# ── Optional in-process precompute (development only) ─────────
# In staging/production the scoring + training Cloud Run Jobs own the
# pipeline, so PRECOMPUTE_ENABLED defaults to False (see config._ENV_DEFAULTS).
# Dev keeps an in-process trigger so contributors can run the app end-to-end
# without Cloud Run Jobs or a separate Redis populator. The trigger fires on
# the first HTTP request — not at import time — to avoid Python import-lock
# deadlocks with gunicorn's fork model.
from config import PRECOMPUTE_ENABLED  # noqa: E402

if PRECOMPUTE_ENABLED:
    import threading as _threading  # noqa: E402

    _precompute_triggered = False
    _precompute_lock = _threading.Lock()

    def _run_dev_scoring() -> None:
        """Invoke the scoring job in a background thread for dev startup."""
        try:
            from jobs.scoring_job import run as scoring_run

            scoring_run()
        except Exception:  # pragma: no cover — dev convenience only
            log.exception("dev_scoring_trigger_failed")

    @server.before_request
    def _trigger_precompute():
        global _precompute_triggered  # noqa: PLW0603
        if _precompute_triggered:
            return
        with _precompute_lock:
            if _precompute_triggered:
                return
            _precompute_triggered = True
        _threading.Thread(
            target=_run_dev_scoring,
            daemon=True,
            name="dev-scoring-trigger",
        ).start()


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


# ── Performance metrics endpoint (internal only) ───────────────
@server.route("/metrics")
def metrics():
    allowed = [ip.strip() for ip in os.getenv("METRICS_ALLOWED_IPS", "127.0.0.1,::1").split(",")]
    from flask import request as flask_request

    # Use X-Forwarded-For when behind a proxy, fall back to remote_addr
    remote = flask_request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    if not remote:
        remote = flask_request.remote_addr or ""
    if remote not in allowed and os.getenv("ENVIRONMENT", "development") != "development":
        log.warning("metrics_access_denied", remote_ip=remote)
        return jsonify({"error": "forbidden"}), 403

    from observability import perf

    return jsonify(perf.get_all_stats()), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    debug = os.getenv("DASH_DEBUG", "false").lower() == "true"
    log.info("starting_dashboard", port=port, debug=debug)
    app.run(debug=debug, host="0.0.0.0", port=port)
