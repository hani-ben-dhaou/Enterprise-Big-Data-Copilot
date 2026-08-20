"""
MCP SSE wire integration test.

Boots the *real* MCP server as a subprocess on an ephemeral port and probes
the SSE transport with raw HTTP.  We deliberately avoid the official mcp SDK
client transport here: it is flaky under Windows' Proactor event loop (see
notes in the delivery audit), while raw HTTP probing of the SSE handshake is
deterministic and validates the actual wire protocol.

Asserts the pre-init handshake of the SSE transport:
  - GET /sse  → 200, content-type text/event-stream
  - stream starts with an `endpoint` event whose data points at /messages
"""

import os
import socket
import subprocess
import sys
import time

import pytest

_WAIT_SECONDS = 30


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout: float = _WAIT_SECONDS) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"server did not open port {port} within {timeout}s")


@pytest.fixture()
def sse_server():
    port = _free_port()
    env = dict(os.environ)
    env.update(
        {
            "MCP_SERVER_HOST": "127.0.0.1",
            "MCP_SERVER_PORT": str(port),
            "MCP_METADATA_SOURCE": "inmemory",  # do not reach Trino at startup
        }
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "app.mcp.server", "--transport", "sse"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.getcwd(),
    )
    try:
        _wait_for_port("127.0.0.1", port)
        yield port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def test_sse_handshake_serves_event_stream(sse_server):
    port = sse_server
    deadline = time.time() + _WAIT_SECONDS
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
                sock.settimeout(2)
                sock.sendall(
                    f"GET /sse HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                    "Connection: close\r\n\r\n".encode("ascii")
                )
                buf = bytearray()
                while True:
                    try:
                        chunk = sock.recv(2048)
                        if not chunk:
                            break
                        buf.extend(chunk)
                    except socket.timeout:
                        # SSE keeps the stream open; whatever arrived within
                        # the window is enough to validate the handshake.
                        break
            head, _, body = bytes(buf).partition(b"\r\n\r\n")
            headers = head.decode("ascii", errors="replace")
            assert " 200 " in headers
            assert "text/event-stream" in headers
            assert "endpoint" in body.decode("utf-8", errors="replace")
            assert "/messages" in body.decode("utf-8", errors="replace")
            return
        except (OSError, AssertionError) as exc:
            last_err = exc
            time.sleep(0.5)
    raise AssertionError(f"SSE handshake never produced an endpoint event: {last_err}")