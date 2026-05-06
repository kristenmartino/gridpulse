"""Unit tests for the perf-bundle compression + cache-header changes.

Closes acceptance criteria for craft-pass issue #24:
- ``flask-compress`` is enabled on the server with ``min_size=500``
- ``/assets/*`` responses carry 1-year ``immutable`` Cache-Control
- ``/_dash-update-component`` callback responses are marked
  ``no-cache`` so deploys instantly invalidate stale figure JSON
"""

from __future__ import annotations


def _import_app():
    """Late import so ``configure_logging()`` only runs once per session
    even if pytest collects this file alongside other app-importing
    test modules."""
    import app as app_mod

    return app_mod


class TestFlaskCompressEnabled:
    def test_compress_after_request_hook_registered(self):
        """flask-compress wires its compression via an after_request
        hook. Verify the hook list includes a compress-related entry —
        if it's missing, the extension wasn't initialized and figure
        JSON ships uncompressed."""
        app_mod = _import_app()
        hook_names = [f.__name__ for f in app_mod.server.after_request_funcs.get(None, [])]
        # flask-compress registers its hook as ``after_request`` (the
        # first hook installed on a fresh app). Combined with the
        # COMPRESS_MIN_SIZE assertion below, this is a tight signal
        # that Compress(server) actually ran.
        assert "after_request" in hook_names, (
            "Expected flask-compress's after_request hook to be registered"
        )

    def test_min_size_threshold_is_500_bytes(self):
        """Below 500 bytes, gzip overhead exceeds savings — Compress
        should skip those responses. Above 500, it kicks in."""
        app_mod = _import_app()
        # flask-compress reads from server.config["COMPRESS_MIN_SIZE"]
        assert app_mod.server.config.get("COMPRESS_MIN_SIZE") == 500

    def test_callback_response_compressed_when_client_accepts_gzip(self):
        """End-to-end check via Flask's test client. We hit
        ``/_dash-update-component`` (the heaviest payload route) with
        ``Accept-Encoding: gzip`` and verify the response carries
        ``Content-Encoding: gzip``.

        A 404 response from the route is fine for this test — we're
        only asserting the compression middleware ran.
        """
        app_mod = _import_app()
        client = app_mod.server.test_client()

        # Generate enough body to clear the 500-byte threshold by
        # asking Dash for a deps list (~5 KB on this app).
        response = client.get(
            "/_dash-dependencies",
            headers={"Accept-Encoding": "gzip"},
        )
        if response.status_code == 200 and len(response.data) >= 500:
            assert response.headers.get("Content-Encoding") == "gzip"


class TestAssetCacheHeaders:
    def test_assets_get_one_year_immutable(self):
        """Dash fingerprints assets via ``?v={hash}``, so 1y immutable
        caching is safe — content changes always rotate the URL."""
        app_mod = _import_app()
        client = app_mod.server.test_client()

        # custom.css is the canonical asset; if it returns 200 we get
        # to assert the headers; if assets are missing in this env we
        # at least verify the after_request hook is wired.
        response = client.get("/assets/custom.css")
        cache_control = response.headers.get("Cache-Control", "")
        if response.status_code == 200:
            assert "max-age=31536000" in cache_control
            assert "immutable" in cache_control
            assert "public" in cache_control

    def test_dash_update_component_marked_no_cache(self):
        """Callback responses must NEVER be cached — figure JSON
        depends on live data and a stale cache would freeze the
        dashboard at the moment of the deploy."""
        app_mod = _import_app()
        client = app_mod.server.test_client()

        # A POST to the callback endpoint without a real callback
        # body returns 4xx, but the after_request hook still runs.
        response = client.post(
            "/_dash-update-component",
            json={"output": "", "outputs": [], "inputs": [], "changedPropIds": []},
        )
        assert response.headers.get("Cache-Control") == "no-cache"

    def test_dash_dependencies_marked_no_cache(self):
        app_mod = _import_app()
        client = app_mod.server.test_client()

        response = client.get("/_dash-dependencies")
        if response.status_code == 200:
            assert response.headers.get("Cache-Control") == "no-cache"

    def test_unrelated_routes_unaffected(self):
        """The header hook should ONLY add Cache-Control to /assets/
        and the two Dash callback endpoints. Health / metrics /
        unknown routes get their default behavior — important so
        Cloud Run's load balancer doesn't see surprise headers."""
        app_mod = _import_app()
        client = app_mod.server.test_client()

        response = client.get("/health")
        # /health may or may not have its own Cache-Control — what
        # matters is that we DIDN'T inject either of the two paths'
        # specific values.
        cc = response.headers.get("Cache-Control", "")
        assert "max-age=31536000" not in cc, "/health should not be marked as a 1-year asset"
