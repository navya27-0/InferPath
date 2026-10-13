"""Unit tests proving automated network isolation fixture prevents outbound access."""

import socket

import httpx
import pytest


def test_socket_connect_is_strictly_blocked():
    """Confirms that any direct TCP socket connection attempt raises RuntimeError."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match=r"Outbound network access.*is strictly blocked"):
            s.connect(("8.8.8.8", 53))
    finally:
        s.close()


def test_network_isolation_blocks_http_client():
    """Confirms that httpx client attempts are intercepted and blocked by socket guard."""
    with (
        pytest.raises(RuntimeError, match=r"Outbound network access.*is strictly blocked"),
        httpx.Client(timeout=1.0) as client,
    ):
        client.get("https://openrouter.ai/api/v1/models")
