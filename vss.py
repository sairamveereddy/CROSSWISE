"""Minimal VSS retrieval API client. Credentials come from env vars; the JWT is cached in tmp/."""
import json
import os
import pathlib
import time

import requests

BASE = os.environ["INGRESS_URL"].rstrip("/") + "/api/v1"
_TOKEN_FILE = pathlib.Path(__file__).resolve().parent.parent / "tmp" / ".vss_token"
_session = requests.Session()


def login(force=False):
    if not force and _TOKEN_FILE.exists() and time.time() - _TOKEN_FILE.stat().st_mtime < 1800:
        return _TOKEN_FILE.read_text().strip()
    r = _session.post(f"{BASE}/auth/login",
                      json={"username": os.environ["USERNAME"], "password": os.environ["PASSWORD"]},
                      timeout=30)
    r.raise_for_status()
    token = r.json()["access_token"]
    _TOKEN_FILE.parent.mkdir(exist_ok=True)
    _TOKEN_FILE.write_text(token)
    _TOKEN_FILE.chmod(0o600)
    return token


def call(method, path, **kw):
    for attempt in range(2):
        headers = {"Authorization": f"Bearer {login(force=attempt > 0)}"}
        r = _session.request(method, f"{BASE}{path}", headers=headers, timeout=kw.pop("timeout", 120), **kw)
        if r.status_code != 401:
            break
    r.raise_for_status()
    return r.json()


def get(path, **params):
    return call("GET", path, params=params)


def post(path, body):
    return call("POST", path, json=body)


if __name__ == "__main__":
    login(force=True)
    print(json.dumps(get("/auth/me"), indent=2))
