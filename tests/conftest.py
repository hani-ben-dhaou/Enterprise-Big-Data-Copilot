"""
Shared test fixtures.

Tests are hermetic: they must never depend on a live stack or on .env drift
(e.g. OneDrive restores). Force the in-memory metadata backend and in-process
MCP transport before any app module is imported (settings is cached at import).
"""

import os

os.environ.setdefault("MCP_METADATA_SOURCE", "inmemory")
os.environ.setdefault("MCP_TRANSPORT", "inprocess")