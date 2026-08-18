"""Shared fixtures for the integration suite.

Cache isolation and the network guard that used to live here are now autouse
in ``tests/conftest.py``, so they cover the unit tier too — unit tests were
reading and writing the real repo-root ``cache.db`` and reaching the live EIA
and Open-Meteo APIs. This file is kept as the place for
integration-tier-specific fixtures.
"""
