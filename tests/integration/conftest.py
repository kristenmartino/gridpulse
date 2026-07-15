"""
Shared fixtures for the integration suite.

Integration tests mock the HTTP layer and assert on parsed fixture data, but
every client fetch path is cache-first (``check cache -> fetch -> cache ->
return``). Without isolation those tests read and write the developer's real
``cache.db``, so the cache — not the mock — decides what they see.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Point the cache singleton at a per-test throwaway db.

    Applies to every integration test, in both directions:

    * **Reads** — a client that returns early on a cache hit never consults
      the test's ``requests.get`` mock, so a warm key silently swaps real
      cached data in for the fixture. ``_fetch_state_alerts`` bit us this way
      on 2026-07-15: running the dashboard populated ``noaa_state_TX`` and
      ``test_noaa_parse_alerts`` then saw 35 live TX alerts instead of the
      fixture's 2.
    * **Writes** — on a miss, the same path caches whatever the mock returned,
      seeding ``cache.db`` with fixture data under a real key (30-min TTL for
      NOAA state alerts). That both masks the read bug on the next run and
      leaves fabricated alerts for the dashboard to serve.

    Patching the ``data.cache._cache`` singleton rather than any one module's
    ``get_cache`` covers every client: eia, weather, noaa, news, and
    ai_briefing all resolve ``get_cache()`` per call, and nothing in the app
    constructs a ``Cache`` any other way.
    """
    from data import cache as cache_module

    monkeypatch.setattr(
        cache_module, "_cache", cache_module.Cache(db_path=str(tmp_path / "cache.db"))
    )
    yield
