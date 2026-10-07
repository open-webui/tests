"""Data an older release wrote, served by the checkout after it upgraded it.

Each data set under `integration/migrations/upgrade_data/` was filled through a release's own API
by `scripts/seed_upgrade_data.py`, and its manifest records what was made and for whom.
`upgraded_release(name, root)` unpacks one (`harness.prepared_data.release_data`), boots the
checkout on it, which runs every migration since that release, and points the embedding and chat
connections the release saved at a scripted provider of its own (`Upgraded.provider`). `settings`
go to the boot, such as `FRONTEND_BUILD_DIR` for the browser suite. `Upgraded.sign_in(who)` signs
a manifest account in with its old password, `client(who)` keeps that session and `actor(who)` is
the account as the browser suite's `page_for` takes it. `data_set_params()` gives every data set
as a pytest param, the Postgres ones marked `requires_postgres`.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import httpx
import pytest

from harness import upstream as upstream_module
from harness.actors import Actor
from harness.prepared_data import RunningBackend, release_data, serving

DATA_SETS = Path(__file__).resolve().parent.parent / "integration" / "migrations" / "upgrade_data"


def data_set_params() -> list:
    params = []
    for manifest in sorted(DATA_SETS.glob("*.json")):
        name = manifest.stem
        marks = [pytest.mark.requires_postgres] if name.endswith("-postgres") else []
        params.append(pytest.param(name, marks=marks, id=name))
    return params


@dataclass
class Upgraded:
    backend: RunningBackend
    manifest: dict
    provider: upstream_module.MockUpstream
    tokens: dict[str, str] = field(default_factory=dict)

    def sign_in(self, who: str) -> httpx.Response:
        account = self.manifest["accounts"][who]
        return httpx.post(
            f"{self.backend.base_url}/api/v1/auths/signin",
            json={"email": account["email"], "password": account["password"]},
            timeout=60.0,
        )

    def client(self, who: str) -> httpx.Client:
        if who not in self.tokens:
            signed_in = self.sign_in(who)
            assert signed_in.status_code == 200, f"{who} cannot sign in: {signed_in.text}"
            self.tokens[who] = signed_in.json()["token"]
        return self.backend.client(self.tokens[who])

    def get(self, who: str, path: str, **options) -> httpx.Response:
        with self.client(who) as client:
            return client.get(path, **options)

    def actor(self, who: str) -> Actor:
        account = self.manifest["accounts"][who]
        self.client(who).close()
        return Actor(
            id=account["id"],
            name=account["name"],
            email=account["email"],
            password=account["password"],
            role=account["role"],
            token=self.tokens[who],
            base_url=self.backend.base_url,
        )


@contextlib.contextmanager
def upgraded_release(
    name: str, root: Path, settings: dict[str, str] | None = None
) -> Iterator[Upgraded]:
    with contextlib.ExitStack() as stack:
        release = stack.enter_context(release_data(DATA_SETS / f"{name}.tar.gz", root))
        provider, shutdown = upstream_module.serve()
        stack.callback(shutdown)
        boot_settings = {
            "WEBUI_AUTH": "true",
            "RAG_EMBEDDING_ENGINE": "openai",
            "RAG_OPENAI_API_BASE_URL": provider.base_url,
            "RAG_OPENAI_API_KEY": "sk-mock",
            **release.settings,
            **(settings or {}),
        }
        backend = stack.enter_context(serving(release.data_dir, boot_settings))
        upgraded = Upgraded(backend, release.manifest, provider)
        _point_connections_at(upgraded, provider.base_url)
        yield upgraded


def _point_connections_at(upgraded: Upgraded, base_url: str) -> None:
    """The provider saved at seeding time is gone; the admin points both connections here."""
    with upgraded.client("admin") as client:
        current = client.get("/api/v1/retrieval/embedding")
        current.raise_for_status()
        form = {
            **current.json(),
            "RAG_EMBEDDING_ENGINE": "openai",
            "openai_config": {"url": base_url, "key": "sk-mock"},
        }
        client.post("/api/v1/retrieval/embedding/update", json=form).raise_for_status()
        connection = {
            "ENABLE_OPENAI_API": True,
            "OPENAI_API_BASE_URLS": [base_url],
            "OPENAI_API_KEYS": ["sk-mock"],
            "OPENAI_API_CONFIGS": {},
        }
        client.post("/openai/config/update", json=connection).raise_for_status()
