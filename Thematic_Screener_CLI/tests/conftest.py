"""Pytest collection hook: several tests import src.screener, which fires
cookbook execution tracing (ADS-400) at import time. Test runs must never
emit telemetry, so disable it before any test module is imported."""

import os

os.environ.setdefault("BIGDATA_DISABLE_TRACING", "1")
