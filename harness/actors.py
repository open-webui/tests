"""Accounts on a launched instance, each with its own token and client."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import httpx

from harness.instance import ADMIN_EMAIL, ADMIN_PASSWORD, LaunchedInstance


@dataclass
class Actor:
    id: str
    name: str
    email: str
    password: str
    role: str
    token: str
    base_url: str

    def client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=120.0,
        )


def admin_of(instance: LaunchedInstance) -> Actor:
    with instance.client() as client:
        session = client.get("/api/v1/auths/")
        session.raise_for_status()
    profile = session.json()
    return Actor(
        id=profile["id"],
        name=profile["name"],
        email=ADMIN_EMAIL,
        password=ADMIN_PASSWORD,
        role="admin",
        token=instance.admin_token,
        base_url=instance.base_url,
    )


def create_user(
    instance: LaunchedInstance,
    role: str = "user",
    name: str | None = None,
    email: str | None = None,
    password: str = "userpassword123",
) -> Actor:
    """A new account added by the admin, the way the admin panel adds one."""
    suffix = uuid.uuid4().hex[:16]
    name = name or f"User {suffix}"
    email = email or f"user-{suffix}@example.com"
    with instance.client() as client:
        added = client.post(
            "/api/v1/auths/add",
            json={"name": name, "email": email, "password": password, "role": role},
        )
    if added.status_code != 200:
        raise AssertionError(f"adding {email} failed: HTTP {added.status_code} {added.text}")
    body = added.json()
    return Actor(
        id=body["id"],
        name=name,
        email=email,
        password=password,
        role=role,
        token=body["token"],
        base_url=instance.base_url,
    )


def sign_in(instance: LaunchedInstance, email: str, password: str) -> str:
    """A fresh session token, the way the sign-in form gets one."""
    response = httpx.post(
        f"{instance.base_url}/api/v1/auths/signin",
        json={"email": email, "password": password},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["token"]
