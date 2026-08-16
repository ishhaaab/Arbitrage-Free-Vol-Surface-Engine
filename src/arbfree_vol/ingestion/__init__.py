"""Ingestion layer — load and clean option chain data."""

from arbfree_vol.ingestion import yahoo  # noqa: F401 — re-export so tests can patch arbfree_vol.ingestion.yahoo.*
