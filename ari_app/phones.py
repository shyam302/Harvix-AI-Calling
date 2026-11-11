"""
Phone accounts (SIP logins for Linphone/Zoiper) managed from the web page.

Asterisk reads PJSIP accounts from two places (docker/asterisk/sorcery.conf):
  - pjsip.conf  -> the PHONES list in docker/.env (read-only here: ARI refuses to delete them)
  - astdb       -> accounts created here at runtime through ARI dynamic configuration:
                   PUT/DELETE /ari/asterisk/config/dynamic/res_pjsip/{auth|aor|endpoint}/{ext}
                   (kept on the asterisk-db volume, no Asterisk restart needed)

The PostgreSQL table ``phones`` remembers which extensions were added from the page
(so only those can be edited / deleted there). Passwords are only stored by Asterisk,
which needs them in clear for SIP digest authentication.
"""

from __future__ import annotations

import logging
import re
import secrets
from typing import Any

import httpx

from ari_app.config import load_settings

log = logging.getLogger(__name__)

EXT_RE = re.compile(r"^1\d{3}$")  # 1000-1999: the dialplan's _1XXX phone-to-phone range
DEFAULT_CODECS = "ulaw,alaw,g722"


class PhoneError(ValueError):
    """Problem shown to the admin (bad extension, already exists…)."""


def _ari() -> tuple[str, httpx.BasicAuth]:
    s = load_settings()
    return f"http://{s.ari_host}:{s.ari_port}/ari", httpx.BasicAuth(s.ari_username, s.ari_password)


async def _request(method: str, path: str, **kw: Any) -> httpx.Response:
    base, auth = _ari()
    async with httpx.AsyncClient(timeout=10.0, auth=auth) as c:
        try:
            return await c.request(method, f"{base}{path}", **kw)
        except httpx.HTTPError as exc:
            raise PhoneError(f"Asterisk (ARI) not reachable: {exc}") from exc


def _fields(d: dict[str, str]) -> dict:
    return {"fields": [{"attribute": k, "value": v} for k, v in d.items()]}


def generate_password() -> str:
    """12 characters, no look-alikes (easy to type on a phone keyboard)."""
    alphabet = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(12))


async def _codecs() -> str:
    """Use the same codecs as the phones from PHONES (CODECS in docker/.env)."""
    r = await _request("GET", "/endpoints/PJSIP")
    for ep in r.json() if r.status_code == 200 else []:
        g = await _request("GET", f"/asterisk/config/dynamic/res_pjsip/endpoint/{ep['resource']}")
        if g.status_code == 200:
            for f in g.json():
                if f.get("attribute") == "allow":
                    codecs = f.get("value", "").strip("()").replace("|", ",")
                    if codecs:
                        return codecs
    return DEFAULT_CODECS


async def list_phones(web_managed: set[str]) -> list[dict]:
    """All PJSIP accounts with online/offline state and where they come from."""
    r = await _request("GET", "/endpoints/PJSIP")
    if r.status_code != 200:
        raise PhoneError(f"ARI error {r.status_code}")
    out = []
    for ep in r.json():
        ext = ep.get("resource", "")
        out.append(
            {
                "extension": ext,
                "state": ep.get("state", "unknown"),  # online | offline | unknown
                "in_call": bool(ep.get("channel_ids")),
                "source": "web" if ext in web_managed else "env",
            }
        )
    return sorted(out, key=lambda p: p["extension"])


async def create_phone(extension: str, password: str) -> None:
    ext = (extension or "").strip()
    if not EXT_RE.match(ext):
        raise PhoneError("Extension must be 4 digits from 1000 to 1999 (e.g. 1005)")
    if len(password or "") < 6:
        raise PhoneError("Password must be at least 6 characters")
    if any(c in password for c in ' ;"\\'):
        raise PhoneError("Password can't contain spaces, ; \" or \\")
    r = await _request("GET", f"/endpoints/PJSIP/{ext}")
    if r.status_code == 200:
        raise PhoneError(f"Phone {ext} already exists")
    base = "/asterisk/config/dynamic/res_pjsip"
    steps = [
        ("auth", {"auth_type": "userpass", "username": ext, "password": password}),
        ("aor", {"max_contacts": "2", "remove_existing": "yes", "qualify_frequency": "60"}),
        (
            "endpoint",
            {
                "transport": "transport-udp",
                "context": "from-internal-app",
                "disallow": "all",
                "allow": await _codecs(),
                "auth": ext,
                "aors": ext,
                "direct_media": "no",
                "rtp_symmetric": "yes",
                "force_rport": "yes",
                "rewrite_contact": "yes",
            },
        ),
    ]
    done: list[str] = []
    for kind, fields in steps:
        r = await _request("PUT", f"{base}/{kind}/{ext}", json=_fields(fields))
        if r.status_code != 200:
            for k in reversed(done):  # roll back a half-created phone
                await _request("DELETE", f"{base}/{k}/{ext}")
            raise PhoneError(f"Asterisk refused the {kind}: {r.text[:120]}")
        done.append(kind)
    log.info("phone %s created (web page)", ext)


async def set_password(extension: str, password: str) -> None:
    if len(password or "") < 6:
        raise PhoneError("Password must be at least 6 characters")
    if any(c in password for c in ' ;"\\'):
        raise PhoneError("Password can't contain spaces, ; \" or \\")
    r = await _request(
        "PUT",
        f"/asterisk/config/dynamic/res_pjsip/auth/{extension}",
        json=_fields({"auth_type": "userpass", "username": extension, "password": password}),
    )
    if r.status_code != 200:
        raise PhoneError(f"Asterisk refused the change: {r.text[:120]}")
    log.info("phone %s password changed (web page)", extension)  # never the password


async def delete_phone(extension: str) -> None:
    base = "/asterisk/config/dynamic/res_pjsip"
    for kind in ("endpoint", "aor", "auth"):
        r = await _request("DELETE", f"{base}/{kind}/{extension}")
        if r.status_code not in (204, 404):
            raise PhoneError(f"Asterisk refused to delete the {kind}: {r.text[:120]}")
    log.info("phone %s deleted (web page)", extension)
