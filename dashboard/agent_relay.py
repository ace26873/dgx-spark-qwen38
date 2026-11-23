#!/usr/bin/env python3
"""Agent relay: opencode's web interface behind the cockpit's own login.

`opencode serve` listens on loopback with a Basic password that never leaves the
box. This relay is the only thing that reaches it. It binds one address (the
tailnet address by default), accepts a request only with a valid cockpit session
cookie, presents the Basic credentials upstream, and streams the answer back:
plain responses, server-sent events, and the WebSocket upgrade the terminal
panel uses. The browser sees one host for the cockpit and the relay (cookies
ignore ports), so the Agent tab frames the interface with no second login and
no Basic-auth prompt, which Chrome would block inside a cross-origin frame.

Stdlib only, like the cockpit. The pure helpers (header hygiene, origin check,
address resolution) are kept apart from the socket code so the tests can pin
them without a browser.
"""
from __future__ import annotations

import base64
import http.client
import importlib.util
import posixpath
import http.server
import json
import re
import select
import socket
import socketserver
import subprocess
import threading
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

RELAY_VERSION = "1.1"

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
              "te", "trailer", "transfer-encoding", "upgrade"}
# Request headers the relay never forwards: the cockpit session must not reach
# opencode, the client's own Authorization is replaced by ours, Host and Origin name
# the upstream, lengths are re-set, and the referrer says nothing about the cockpit.
# accept-encoding is not in this set: it is dropped per request (forward_request_headers),
# only on the credential routes, where the relay reads the JSON of the answer to scrub it and
# a gzip answer it cannot verify is a 502 the browser would never understand (measured 2.3.0;
# independent review of the v2 migration, finding 5: ask for identity, refuse only a server
# that lies anyway). Everywhere else the client's Accept-Encoding passes, as it did before
# the v2 migration: a box that never passes --opencode sees no change at the relay
# (blind review of 2026-11-20, D6).
REQUEST_DROP = HOP_BY_HOP | {"host", "origin", "cookie", "authorization", "content-length", "referer"}
# Response headers the relay owns (Date and Server come from this server).
RESPONSE_DROP = HOP_BY_HOP | {"content-length", "date", "server"}
SESSION_COOKIE = "cockpit"
# Fetched by browsers with no cookie (a manifest link omits credentials): the app's
# name and icons, nothing else, so it may pass without a session (GET only).
PUBLIC_PATHS = {"/site.webmanifest"}
MAX_BODY = 64 * 1024 * 1024       # request body ceiling (attachments included)
CHUNK = 64 * 1024
CONNECT_TIMEOUT = 5.0
HANDSHAKE_LIMIT = 64 * 1024
INJECT_LIMIT = 256 * 1024         # an HTML document we rewrite must stay this small (the SPA index is ~3 KB)
# OpenCode 2's /api/config, /api/provider and /api/model return the provider options after
# resolving {file:...} references: the box's API key in clear (seen on 2.3.0, 2026-11-13, and
# again in the sandbox of 2026-11-14: the canary key came back from each of those three).
# /api/credential is documented to list stored integration credentials "including its secret
# value" (empty on a box that signed into none: we scrub it the same, the answer is nobody's
# to read in a browser). The SPA only GETs these; scrub them here, before the browser sees
# one. A secret-bearing answer we cannot read (compressed, not JSON, larger than this) is
# refused, not passed.
SECRET_PATHS = ("/config", "/config/providers", "/provider", "/model",
                "/api/config", "/api/provider", "/api/model", "/api/credential",
                "/api/experimental/config", "/api/experimental/artifacts/providers",
                "/api/integration")
SECRET_KEYS = frozenset({"apikey", "api_key", "api-key", "authorization", "password",
                         "secret", "client_secret", "clientsecret", "access_token",
                         "refresh_token", "token"})
# Keys that are only secret on one path: "value" carries a credential in /api/credential's
# items, but it is a plain member everywhere else (models, costs) and must keep its shape.
SECRET_KEYS_EXTRA = {"/api/credential": frozenset({"value"})}
SCRUB_LIMIT = 16 * 1024 * 1024   # 1.x answers /provider with the whole model catalog: 7,400,464 bytes
                                 # measured on the box's 1.19.5 (2026-11-15); above this an answer is refused
# A client that sends nothing for this long, mid-request or between two, is let go. With
# no limit a request sent in part held its thread, and a descriptor, for good: 100 of
# them held 100 threads, in a process whose 1,024 descriptors its collectors and memory
# floor need too (found in review, 2026-09-24). Only waits on the client count: a stream
# this side is writing, or a tunnel both ends are quiet on, never times out here.
CLIENT_TIMEOUT = 30

# The cockpit's mobile comfort layer for opencode's interface (phone typography,
# touch targets, safe areas, theme default). opencode's own files are never
# touched: these two are the relay's, served same-origin (so its CSP 'self'
# accepts them) and linked into the served index by inject_mobile below.
INJECT_DIR = Path(__file__).resolve().parent / "static"
INJECTED_FILES = {
    "/__cockpit/agent-mobile.css": ("agent-mobile.css", "text/css; charset=utf-8"),
    "/__cockpit/agent-mobile.js": ("agent-mobile.js", "application/javascript; charset=utf-8"),
}
MOBILE_TAGS = (b'<link rel="stylesheet" href="/__cockpit/agent-mobile.css">'
               b'<script src="/__cockpit/agent-mobile.js"></script>')


# ── pure helpers ─────────────────────────────────────────────────────────────
def host_name(host_header: str | None) -> str:
    """The host part of a Host header: `a.b:30090` -> `a.b`, `[::1]:1` -> `[::1]`."""
    h = (host_header or "").strip()
    if not h:
        return ""
    if h.startswith("["):
        return h.split("]", 1)[0] + "]"
    if h.count(":") == 1:
        return h.rsplit(":", 1)[0]
    if ":" in h:
        # A bare IPv6 literal: no port can follow without brackets, and the
        # callers build http://{name}:{port} URLs, so return the bracketed
        # canonical form instead of a string that reads as host:port:port.
        return f"[{h}]"
    return h


def origin_allowed(origin: str | None, host_header: str | None) -> bool:
    """A browser names its origin on every POST and every WebSocket handshake. The
    interface is served by this relay, so the only legitimate origin is the relay's
    own host as the browser typed it. No header (curl, the cockpit's own probes) is
    fine: those requests still need the session cookie. `null` is never fine."""
    if origin is None:
        return True
    o = origin.strip().lower()
    host = (host_header or "").strip().lower()
    return bool(host) and o in (f"http://{host}", f"https://{host}")


def cookie_values(header: str | None, name: str) -> list[str]:
    """Every value the Cookie header gives `name`, read the way browsers send it (`a=b;
    c=d`) and nothing more. http.cookies.SimpleCookie gives up at the first pair it has no
    grammar for, and raises on a name such as `a/b`: another app's cookie on this host (a
    JSON value, a space in one) hid the session, or ended the connection with no answer
    (found in review, 2026-09-24)."""
    out = []
    for part in (header or "").split(";"):
        key, sep, value = part.partition("=")
        if sep and key.strip() == name:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] == '"':
                value = value[1:-1]
            out.append(value)
    return out


def basic_auth(user: str, password: str) -> str:
    return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode("ascii")


def read_credentials(path: Path) -> tuple[str, str] | None:
    """The env file the opencode unit reads (KEY=VALUE lines, optional quotes)."""
    try:
        text = Path(path).read_text()
    except OSError:
        return None
    kv = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        kv[k.strip()] = v.strip().strip('"').strip("'")
    pw = kv.get("OPENCODE_SERVER_PASSWORD", "")
    if not pw:
        return None
    return kv.get("OPENCODE_SERVER_USERNAME") or "opencode", pw


def forward_request_headers(items, upstream_host: str, auth: str,
                            websocket: bool = False, secret: bool = False,
                            page: bool = False) -> list[tuple[str, str]]:
    """page: a GET whose Accept names text/html, the document the relay links the phone
    layer into. OpenCode 2.3.2 compresses it with Brotli when asked (measured 2026-11-22),
    and the standard library cannot read Brotli to rewrite it: a page asks for identity,
    like a credential route; scripts and API calls keep the client's compression (D6)."""
    items = list(items)
    drop = REQUEST_DROP | ({"accept-encoding"} if secret or page else set())
    out = [(k, v) for k, v in items if k.lower() not in drop]
    out.append(("Host", upstream_host))
    if any(k.lower() == "origin" for k, _ in items):
        # The browser's origin was held to this relay's own host before anything was
        # forwarded (origin_allowed), so opencode is shown the origin of the page it
        # serves itself. It refuses its terminal to any origin but its own host and
        # loopback: the connect ticket in every version this repo has pinned, and the
        # WebSocket itself from 1.19.0 on, so the Agent tab opened by the tailnet
        # address had a terminal with no shell (in Chrome, 2026-10-11: the ticket
        # refused with 403 on 1.18.35 and 1.19.0, and no WebSocket opened).
        out.append(("Origin", f"http://{upstream_host}"))
    out.append(("Authorization", auth))
    if websocket:
        out.append(("Connection", "Upgrade"))
        out.append(("Upgrade", "websocket"))
    else:
        out.append(("Connection", "close"))
    return out


def forward_response_headers(items, drop=()) -> list[tuple[str, str]]:
    extra = {d.lower() for d in drop}
    out = []
    for k, v in items:
        kl = k.lower()
        if kl in RESPONSE_DROP or kl in extra:
            continue
        # a cookie named like the cockpit session would shadow the real one on this host
        if kl == "set-cookie" and v.strip().lower().startswith(SESSION_COOKIE + "="):
            continue
        out.append((k, v))
    return out


def frame_policy(host_header: str | None, cockpit_port: int) -> str:
    """Only the cockpit on the same host may frame the interface (clickjacking
    guard). The upstream policy has no frame-ancestors, so this one governs."""
    name = host_name(host_header) or "127.0.0.1"
    return f"frame-ancestors 'self' http://{name}:{cockpit_port}"


def split_upstream(url: str) -> tuple[str, int]:
    u = urllib.parse.urlsplit(url if "://" in url else "http://" + url)
    return u.hostname or "127.0.0.1", u.port or 80


def normalized_target(path: str) -> str:
    """The path opencode's router answers: /./api/config, /api//config, /foo/../api/config and
    /api/%63onfig all reach /api/config. Both the secret test and the keys to scrub read THIS path,
    never the raw one (review of v1.56.0, finding 1: the keys read the raw path, so an odd spelling
    of /api/credential was scrubbed without "value", the credential itself)."""
    p = urllib.parse.unquote(path.split("?", 1)[0])
    if not p.startswith("/"):
        p = "/" + p
    p = posixpath.normpath(p)
    return re.sub(r"/{2,}", "/", p)       # normpath keeps a leading "//" (POSIX); the router does not


def is_secret_path(path: str) -> bool:
    """A path whose JSON answer can carry provider credentials. Classified on the NORMALIZED target:
    opencode's router answers /./api/config, /api//config and /foo/../api/config as /api/config, so
    those spellings are exactly as secret (a raw match let them through unscrubbed; independent
    review of the v2 migration, finding 6)."""
    p = normalized_target(path)
    return any(p == s or p.startswith(s + "/") for s in SECRET_PATHS)


def secret_keys_for(path: str) -> frozenset:
    """The member names to replace on this path: the usual ones, plus the extras of every
    secret-bearing prefix it falls under (/api/credential and its items, measured 2026-11-14),
    on the normalized target, as is_secret_path reads it."""
    p = normalized_target(path)
    extra = frozenset()
    for prefix, keys in SECRET_KEYS_EXTRA.items():
        if p == prefix or p.startswith(prefix + "/"):
            extra |= keys
    return SECRET_KEYS | extra


def scrub_secrets(value, keys: frozenset = SECRET_KEYS):
    """value with every credential-bearing member replaced, recursively. The shape is kept:
    the SPA reads providers and models, never the key it would send."""
    if isinstance(value, dict):
        return {k: ("***" if str(k).lower() in keys else scrub_secrets(v, keys))
                for k, v in value.items()}
    if isinstance(value, list):
        return [scrub_secrets(v, keys) for v in value]
    return value


def scrub_json(raw: bytes, keys: frozenset = SECRET_KEYS) -> bytes:
    """The JSON document in raw with its credentials replaced; ValueError when it is not one."""
    return json.dumps(scrub_secrets(json.loads(raw or b"null"), keys),
                      separators=(",", ":")).encode()


def injected_asset(path: str) -> tuple[str, str] | None:
    """(filename, content-type) when the request asks for one of the cockpit's
    own injected assets, else None."""
    return INJECTED_FILES.get(path.split("?", 1)[0])


def inject_mobile(html: bytes) -> bytes:
    """Link the mobile comfort layer into a served document, just before
    </head> (the script stays before opencode's module, so its theme default
    lands before the app paints). Idempotent; appends when no </head> exists."""
    if b"/__cockpit/agent-mobile." in html:
        return html
    out, n = re.subn(rb"(?i)</head\s*>", lambda m: MOBILE_TAGS + m.group(0), html, count=1)
    return out if n else html + MOBILE_TAGS


def wants_html_rewrite(command: str, status: int, content_type: str | None,
                       content_encoding: str | None, chunked: bool,
                       length: int | None) -> bool:
    """Rewrite only a small, uncompressed 200 HTML document we were asked to GET: one
    with a length up to INJECT_LIMIT, or a chunked one, which forward reads up to that
    limit and rewrites only when the whole of it came within (opencode 1.19.4 sends its
    index chunked, 1.18.35 with a length). Everything else (SSE, images, gzip,
    redirects, a document that ends only when the connection closes) streams
    byte-exact as before."""
    return (command == "GET" and status == 200
            and "text/html" in (content_type or "").lower()
            and (content_encoding or "").strip().lower() in ("", "identity")
            and (chunked or (length is not None and 0 < length <= INJECT_LIMIT)))


def tailscale_ipv4(run=subprocess.run) -> str | None:
    """The tailnet IPv4 of this box: the interface first (kernel state, no daemon
    socket needed), the CLI as a fallback."""
    for argv in (["ip", "-4", "-o", "addr", "show", "dev", "tailscale0"], ["tailscale", "ip", "-4"]):
        try:
            out = run(argv, capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        m = re.search(r"\b(100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3})\b", out or "")
        if m:
            return m.group(1)
    return None


def resolve_bind(spec: str, run=subprocess.run) -> str | None:
    """`tailscale` resolves to the tailnet address (None while it is not up);
    anything else is taken as an address literal."""
    spec = (spec or "").strip()
    if spec.lower() == "tailscale":
        return tailscale_ipv4(run)
    return spec or None


# ── configuration ────────────────────────────────────────────────────────────
@dataclass
class RelayConfig:
    upstream: tuple[str, int]                              # opencode serve, loopback
    credentials: Callable[[], tuple[str, str] | None]      # (user, password) or None
    is_authed: Callable[[str | None], bool]                # Cookie header -> valid session?
    cockpit_port: int                                      # for frame-ancestors and the sign-in hint

    @property
    def upstream_host(self) -> str:
        return f"{self.upstream[0]}:{self.upstream[1]}"


def health(cfg: RelayConfig, timeout: float = 3.0) -> dict:
    """What the Agent panel shows as the server state. OpenCode 2 has /api/info (version,
    pid, capabilities) and no JSON /global/health; 1.x has /global/health and no /api/info.
    A 200 that is not JSON answers nothing (OpenCode 2 serves its HTML page on every path it
    does not know, issue #55704, measured on 2.3.0 the 2026-11-14): counting that as healthy
    would say a dead server is up."""
    creds = cfg.credentials()
    if not creds:
        return {"healthy": False, "error": "no credentials file"}
    for path in ("/api/info", "/global/health"):
        conn = http.client.HTTPConnection(*cfg.upstream, timeout=timeout)
        try:
            conn.request("GET", path, headers={"Authorization": basic_auth(*creds)})
            resp = conn.getresponse()
            raw = resp.read(4096)
            if resp.status == 401:
                return {"healthy": False, "error": "credentials refused (restart opencode-web.service after editing the env file)"}
            if resp.status == 200:
                if "json" not in (resp.getheader("Content-Type") or "").lower():
                    continue        # the SPA page on a path this version does not know
                data = json.loads(raw or b"{}")
                if not isinstance(data, dict):
                    return {"healthy": False, "error": f"{path} answered {type(data).__name__}, not an object"}
                if path == "/api/info":
                    return {"healthy": True, "version": data.get("version")}
                return {"healthy": bool(data.get("healthy")), "version": data.get("version")}
            if resp.status != 404:
                return {"healthy": False, "error": f"HTTP {resp.status}"}
        except (OSError, http.client.HTTPException, ValueError) as e:
            return {"healthy": False, "error": f"{type(e).__name__}: {str(e)[:80]}"}
        finally:
            conn.close()
    return {"healthy": False, "error": "no JSON health answer (HTTP 404 on every probe)"}


def upstream_json(cfg: RelayConfig, path: str, timeout: float = 3.0):
    """One authenticated GET on the upstream, parsed; None when it cannot be had."""
    creds = cfg.credentials()
    if not creds:
        return None
    conn = http.client.HTTPConnection(*cfg.upstream, timeout=timeout)
    try:
        conn.request("GET", path, headers={"Authorization": basic_auth(*creds)})
        resp = conn.getresponse()
        raw = resp.read(1024 * 1024)
        if resp.status != 200:
            return None
        return json.loads(raw or b"null")
    except (OSError, http.client.HTTPException, ValueError):
        return None
    finally:
        conn.close()


def permission_is_auto(permission) -> bool:
    """opencode's permission config, as served: "allow" or {"*": "allow"} means every
    tool call runs without a prompt (deny rules aside)."""
    if permission == "allow":
        return True
    return isinstance(permission, dict) and permission.get("*") == "allow"


_OCP_SPEC = importlib.util.spec_from_file_location(
    "oc_permissions", str(Path(__file__).resolve().parents[1] / "oc-permissions.py"))
_ocp = importlib.util.module_from_spec(_OCP_SPEC)
_OCP_SPEC.loader.exec_module(_ocp)


def _v2_wild(text: str, pattern: str) -> bool:
    """OpenCode 2's wildcard: THE server's reading, shared with oc-merge-limits.py so the guard line
    here and the installer's verification answer one question (measured on the 2.3.0 binary and read
    in its bundle the 2026-11-15: "*" any run, "?" exactly one character, brackets and dot literal,
    whole-text match; a review of the v2 migration found the two matchers had drifted, finding 7)."""
    return _ocp.server_match(text, pattern)


def _v2_rule_matches(rule, action: str, resource: str) -> bool:
    """A rule matches when both its action and its resource wildcard-match
    (measured in the sandbox of 2026-11-14: {action:"shell",resource:"*host*",effect:"deny"}
    holds a command naming that host and nothing else)."""
    a, r = rule.get("action"), rule.get("resource")
    if not isinstance(a, str) or not isinstance(r, str):
        return False
    return _v2_wild(action, a) and _v2_wild(resource, r)


GUARDED_ACTIONS = ("shell", "webfetch")   # the OpenCode 2 actions the guard's patterns are denied to


def config_docs_auto_state(entries, deny_patterns=None) -> dict | None:
    """What OpenCode 2's served /api/config says about tool calls, for the Agent line.
    The documents merge in the order the server lists them and the last matching rule
    wins (measured 2026-11-14): the unit's injected content is the document with no path.
    Returns {"auto": is a shell call allowed, "deny_ok": every guard still denies, None
    with no pattern to check} or None when the answer is not documents we can read. deny_patterns are the command
    patterns our guard denies: they must still evaluate to deny whoever allowed what after."""
    if not isinstance(entries, list) or not entries:
        return None
    rules = []
    for e in entries:
        if not isinstance(e, dict):
            return None
        if "info" not in e:
            continue        # a directory entry carries no document (measured on 2.3.0)
        info = e.get("info")
        if not isinstance(info, dict):
            return None
        perms = info.get("permissions")
        if perms is None:
            continue
        if not isinstance(perms, list) or not all(isinstance(r, dict) for r in perms):
            return None
        rules.extend(perms)

    def evaluate(action: str, resource: str):
        got = None
        for rule in rules:
            if _v2_rule_matches(rule, action, resource):
                effect = rule.get("effect")
                if effect in ("allow", "deny", "ask"):
                    got = effect
        return got     # None: no rule matched, the server's own default applies (ask)

    auto = evaluate("shell", "echo x") == "allow"
    # no pattern to check (the guard's list could not be read) proves nothing: None, never True
    # for the shell and for webfetch (v1.57.0, decision 1561: the fetches of 2026-11-21 left through webfetch,
    # which the shell rules never saw): the guard is in force only when both are denied these hosts
    deny_ok = (all(evaluate(action, p) == "deny" for action in GUARDED_ACTIONS for p in deny_patterns)
               if deny_patterns else None)
    return {"auto": auto, "deny_ok": deny_ok}


# ── the relay ────────────────────────────────────────────────────────────────
SIGN_IN_PAGE = """<!doctype html><meta charset="utf-8"><title>Alzeo cockpit</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;background:#0e1014;color:#ece9e1;display:grid;place-items:center;min-height:100vh;margin:0}}
main{{max-width:44ch;padding:32px;text-align:center}}a{{color:#d9b45e}}</style>
<main><h1 style="font-size:18px">Cockpit session required</h1>
<p>This is the agent relay of the Alzeo cockpit. Sign in to the cockpit at
<a href="{cockpit}">{cockpit}</a>, then come back to the Agent view.</p></main>"""


def make_handler(cfg: RelayConfig) -> type[http.server.BaseHTTPRequestHandler]:
    class Relay(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "AlzeoCockpitRelay/" + RELAY_VERSION
        sys_version = ""
        timeout = CLIENT_TIMEOUT

        def log_message(self, fmt, *args):   # quiet: a chatty relay hides the real journal lines
            pass

        # every method goes through the same gate and the same forwarder
        def do_GET(self):
            self.relay()

        do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_HEAD = do_GET

        # ---- gate ----
        def relay(self):
            host = self.headers.get("Host")
            if not origin_allowed(self.headers.get("Origin"), host):
                return self.refuse(403, "origin not allowed")
            # a manifest fetch carries no body: one that does is not that fetch, and went to
            # opencode with the Basic credentials and no session, up to MAX_BODY of it
            public = (self.command == "GET" and self.path.split("?", 1)[0] in PUBLIC_PATHS
                      and not self.headers.get("Transfer-Encoding")
                      and (self.headers.get("Content-Length") or "0").strip() == "0")
            if not public and not cfg.is_authed(self.headers.get("Cookie")):
                return self.refuse(401, "cockpit session required")
            asset = injected_asset(self.path) if self.command in ("GET", "HEAD") else None
            if asset:
                return self.serve_asset(asset)
            creds = cfg.credentials()
            if not creds:
                return self.refuse(503, "agent credentials missing on the box (run dashboard/install-agent.sh)")
            auth = basic_auth(*creds)
            if self.headers.get("Upgrade", "").lower() == "websocket":
                return self.tunnel(auth)
            if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
                return self.refuse(411, "length required")
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return self.refuse(400, "bad content length")
            if length < 0 or length > MAX_BODY:
                return self.refuse(413, "body too large")
            body = self.rfile.read(length) if length else b""
            self.forward(auth, body)

        def refuse(self, code: int, why: str):
            # never try to keep a connection whose body was not consumed
            self.close_connection = True
            wants_page = self.command == "GET" and "text/html" in (self.headers.get("Accept") or "")
            if wants_page and code == 401:
                cockpit = f"http://{host_name(self.headers.get('Host')) or '127.0.0.1'}:{cfg.cockpit_port}/"
                payload = SIGN_IN_PAGE.format(cockpit=cockpit).encode()
                ctype = "text/html; charset=utf-8"
            else:
                payload = json.dumps({"error": why}).encode()
                ctype = "application/json"
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.own_headers()
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        def serve_asset(self, asset: tuple[str, str]):
            """The cockpit's own injected asset, read fresh from disk each time
            (iterating on it should never need a relay restart)."""
            try:
                payload = (INJECT_DIR / asset[0]).read_bytes()
            except OSError:
                return self.refuse(404, "injected asset missing on the box: " + asset[0])
            self.send_response(200)
            self.send_header("Content-Type", asset[1])
            if self.command != "HEAD":
                self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-cache")
            self.own_headers()
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        def own_headers(self):
            self.send_header("Content-Security-Policy", frame_policy(self.headers.get("Host"), cfg.cockpit_port))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")

        # ---- plain HTTP, streamed ----
        def forward(self, auth: str, body: bytes):
            conn = http.client.HTTPConnection(*cfg.upstream, timeout=CONNECT_TIMEOUT)
            try:
                conn.connect()
                # a long generation or an idle event stream must not time out here;
                # a vanished client shows up as a write error on our side instead
                conn.sock.settimeout(None)
                conn.putrequest(self.command, self.path, skip_host=True, skip_accept_encoding=True)
                for k, v in forward_request_headers(self.headers.items(), cfg.upstream_host, auth,
                                                    secret=is_secret_path(self.path),
                                                    page=self.command == "GET"
                                                    and "text/html" in (self.headers.get("Accept") or "")):
                    conn.putheader(k, v)
                if body or self.command in ("POST", "PUT", "PATCH"):
                    conn.putheader("Content-Length", str(len(body)))
                conn.endheaders(body if body else None)
                resp = conn.getresponse()
            except (OSError, http.client.HTTPException) as e:
                conn.close()
                return self.refuse(502, f"agent server unreachable ({type(e).__name__}); is opencode-web.service running?")
            # every method but HEAD (a HEAD carries no body to leak): a script injected in
            # the tab can POST to a secret route, and the answer of a POST is not less secret
            # (blind review of 2026-11-20, D5)
            # every 2xx answer with a body, not only 200: a 201 or 206 carries the same secrets (review of
            # v1.56.0, finding 10); 204, 205 and an empty body carry none
            has_body = resp.status not in (204, 205) and resp.length != 0
            scrub_it = (self.command != "HEAD" and 200 <= resp.status < 300 and has_body
                        and is_secret_path(self.path))
            if scrub_it:
                # OpenCode 2 resolves {file:...} in the answers to these routes: the key came
                # back in clear from /api/config, /api/provider and /api/model on 2.3.0.
                # Read that answer here, scrub it, and send the length of what we send.
                ctype = (resp.getheader("Content-Type") or "").lower()
                enc = (resp.getheader("Content-Encoding") or "").strip().lower()
                if "json" not in ctype:
                    if "text/html" in ctype and enc in ("", "identity"):
                        # 1.x answers its page routes (its /model is the SPA page, measured on the box
                        # the 2026-11-15) with text/html: no key in a page, nothing to read. Any other
                        # unreadable answer is refused below: what cannot be read is not relayed.
                        scrub_it = False
                    else:
                        return self.refuse(502, "the agent server sent an unreadable answer for a "
                                               "secret-bearing path (ValueError: not plain JSON); not relayed")
            if scrub_it:
                try:
                    declared = resp.length
                    if declared is None and not resp.chunked:
                        raise ValueError("length unknown")
                    if declared is not None and declared > SCRUB_LIMIT:
                        raise ValueError("larger than the scrub limit")
                    want = declared if declared is not None else SCRUB_LIMIT + 1
                    raw = resp.read(want)
                    if declared is None and (len(raw) > SCRUB_LIMIT or resp.read(1)):
                        raise ValueError("larger than the scrub limit")
                    if declared is not None and len(raw) != want:
                        raise ValueError("cut short")
                    payload = scrub_json(raw, secret_keys_for(self.path))
                except (OSError, http.client.HTTPException, ValueError) as e:
                    # fail closed: an answer we cannot read is one we cannot scrub
                    return self.refuse(502, "the agent server sent an unreadable answer for a "
                                       f"secret-bearing path ({type(e).__name__}: {str(e)[:80]}); not relayed")
                self.send_response(resp.status, resp.reason)
                for k, v in forward_response_headers(resp.getheaders(),
                                                     drop=("cache-control", "content-encoding")):
                    self.send_header(k, v)
                self.own_headers()
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
                return
            rewrite = wants_html_rewrite(self.command, resp.status,
                                         resp.getheader("Content-Type"),
                                         resp.getheader("Content-Encoding"),
                                         resp.chunked, resp.length)
            try:
                doc = b""
                if rewrite:
                    # Read before anything is sent: a document opencode ended early (a
                    # restart mid-answer) would read as whole once injected, so the
                    # browser gets a 502 it can reload from instead. A document with a
                    # length is read to it; a chunked one up to one byte past
                    # INJECT_LIMIT, and gets the layer only when its last chunk came
                    # within that and it holds something, as a length must be above 0
                    # (past it, or empty, it streams untouched below, read part first).
                    want = resp.length if resp.length is not None else INJECT_LIMIT + 1
                    try:
                        doc = resp.read(want)
                        cut = resp.length is not None and len(doc) < want
                    except (OSError, http.client.HTTPException):
                        cut = True
                    if cut:
                        return self.refuse(502, "the agent server cut its answer short; reload the tab")
                    if not doc or (resp.length is None and not resp.isclosed()):
                        rewrite = False
                self.send_response(resp.status, resp.reason)
                for k, v in forward_response_headers(resp.getheaders(),
                                                     drop=("cache-control",) if rewrite else ()):
                    self.send_header(k, v)
                self.own_headers()
                bodyless = self.command == "HEAD" or resp.status in (204, 304) or resp.status < 200
                if bodyless:
                    if self.command == "HEAD" and resp.getheader("Content-Length") is not None:
                        self.send_header("Content-Length", resp.getheader("Content-Length"))
                    self.end_headers()
                    return
                if rewrite:
                    payload = inject_mobile(doc)
                    self.send_header("Content-Length", str(len(payload)))
                    # opencode sends no Cache-Control (Safari then serves the
                    # document from its heuristic cache and the injection - and
                    # with it the phone layer - never reaches the browser)
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if resp.length is not None and not resp.chunked:
                    self.send_header("Content-Length", str(resp.length))
                    self.end_headers()
                    remaining = resp.length
                    while remaining > 0:
                        data = resp.read(min(CHUNK, remaining))
                        if not data:
                            break
                        self.wfile.write(data)
                        remaining -= len(data)
                    self.wfile.flush()
                    if remaining > 0:
                        # opencode ended early (a restart mid-answer). The browser was
                        # promised the whole length, and kept this keep-alive connection,
                        # one of its six to this origin, waiting for the rest: closing it
                        # is how it learns the answer was cut (found in review, 2026-09-24)
                        self.close_connection = True
                        try:
                            self.connection.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                    return
                # unknown length (server-sent events, chunked answers): re-chunk for a
                # 1.1 client, close-delimit for a 1.0 one; flush every piece so an
                # event reaches the browser the moment opencode emits it
                chunked = self.request_version >= "HTTP/1.1"
                if chunked:
                    self.send_header("Transfer-Encoding", "chunked")
                else:
                    self.close_connection = True
                self.end_headers()
                while True:
                    data = doc or resp.read1(CHUNK)
                    doc = b""
                    if not data:
                        break
                    if chunked:
                        self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
                    else:
                        self.wfile.write(data)
                    self.wfile.flush()
                if chunked:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
            except (OSError, http.client.HTTPException):
                # the browser went away (tab closed, stream cancelled) or opencode
                # dropped the connection: nothing to answer to anymore
                self.close_connection = True
            finally:
                conn.close()

        # ---- WebSocket (the terminal panel) ----
        def tunnel(self, auth: str):
            try:
                up = socket.create_connection(cfg.upstream, timeout=CONNECT_TIMEOUT)
            except OSError as e:
                return self.refuse(502, f"agent server unreachable ({type(e).__name__})")
            self.close_connection = True
            try:
                up.settimeout(CONNECT_TIMEOUT)
                lines = [f"{self.command} {self.path} HTTP/1.1"]
                lines += [f"{k}: {v}" for k, v in forward_request_headers(
                    self.headers.items(), cfg.upstream_host, auth, websocket=True,
                    secret=is_secret_path(self.path))]
                up.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"))
                head = b""
                while b"\r\n\r\n" not in head:
                    piece = up.recv(4096)
                    if not piece:
                        raise OSError("upstream closed during the handshake")
                    head += piece
                    if len(head) > HANDSHAKE_LIMIT:
                        raise OSError("handshake too large")
                head, rest = head.split(b"\r\n\r\n", 1)
                # the handshake answer goes back verbatim (101, or the refusal)
                self.wfile.write(head + b"\r\n\r\n")
                self.wfile.flush()
                if not head.startswith(b"HTTP/1.1 101"):
                    if rest:
                        self.wfile.write(rest)
                        self.wfile.flush()
                    return
                up.settimeout(None)
                client = self.connection
                if rest:
                    client.sendall(rest)
                pump(client, up)
            except OSError:
                pass
            finally:
                up.close()

    return Relay


def pump(a: socket.socket, b: socket.socket):
    """Bytes both ways until either side closes."""
    peer = {a: b, b: a}
    while True:
        try:
            ready, _, broken = select.select([a, b], [], [a, b])
        except (OSError, ValueError):
            return
        if broken:
            return
        for s in ready:
            try:
                data = s.recv(CHUNK)
            except OSError:
                return
            if not data:
                return
            try:
                peer[s].sendall(data)
            except OSError:
                return


class BoundedThreadingMixIn(socketserver.ThreadingMixIn):
    """A thread per connection, as ThreadingMixIn, and a bounded number of them: past
    `max_connections` in all, or `max_per_address` from one address, a new connection is
    closed before any thread starts for it. A browser holds six per origin at most."""
    daemon_threads = True
    max_connections = 96
    max_per_address = 24

    def __init__(self, *args, **kwargs):
        self._cap_lock = threading.Lock()
        self._cap_open: dict[str, int] = {}
        super().__init__(*args, **kwargs)

    def _cap_release(self, addr: str) -> None:
        with self._cap_lock:
            n = self._cap_open.get(addr, 0) - 1
            if n > 0:
                self._cap_open[addr] = n
            else:
                self._cap_open.pop(addr, None)

    def process_request(self, request, client_address):
        addr = str(client_address[0]) if client_address else ""
        with self._cap_lock:
            full = (sum(self._cap_open.values()) >= self.max_connections
                    or self._cap_open.get(addr, 0) >= self.max_per_address)
            if not full:
                self._cap_open[addr] = self._cap_open.get(addr, 0) + 1
        if full:
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._cap_release(addr)
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._cap_release(str(client_address[0]) if client_address else "")


class RelayServer(BoundedThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True
    request_queue_size = 128


def serve(cfg: RelayConfig, bind: str, port: int) -> RelayServer:
    """Bind now (raises OSError if the address is not ours yet); the caller runs
    serve_forever in its own thread."""
    return RelayServer((bind, port), make_handler(cfg))


if __name__ == "__main__":   # dev: python3 agent_relay.py 127.0.0.1 30091 (no session check)
    import sys
    cfg = RelayConfig(upstream=split_upstream(sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:4096"),
                      credentials=lambda: read_credentials(Path.home() / ".config/qwen38/opencode-web.env"),
                      is_authed=lambda cookie: True, cockpit_port=30090)
    srv = serve(cfg, sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1", int(sys.argv[2]) if len(sys.argv) > 2 else 30091)
    print(f"relay (UNAUTHENTICATED dev mode) on {srv.server_address} -> {cfg.upstream_host}", flush=True)
    srv.serve_forever()
