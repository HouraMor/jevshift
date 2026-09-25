"""Keep every test offline, including CLI commands and the replay fake."""

import socket

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    for name in ("TYPESAFE_API_KEY", "TYPESAFE_DEFAULT_MODEL", "TYPESAFE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)

    def blocked(*args, **kwargs):
        pytest.fail("Tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
