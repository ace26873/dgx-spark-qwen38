#!/usr/bin/env bash
# Install the cockpit's Agent tab: opencode's web interface, behind the cockpit
# login. Idempotent, and safe to run alone. Since v1.12 install.sh runs this with
# the cockpit whenever opencode is on the PATH, and skips it with a note when it
# is not: a missing tab is not a reason to fail an install. Two pieces land:
#
#   opencode-web.service      `opencode serve` on 127.0.0.1:OPENCODE_PORT (default
#                             4096), as you, with Basic credentials generated once
#                             into ~/.config/qwen38/opencode-web.env (mode 0600).
#                             Nothing but the relay below ever reaches it.
#   qwen38-dashboard.service  re-installed with the agent relay on
#                             AGENT_BIND:AGENT_PORT (default: the tailnet address,
#                             port 30091). The relay accepts a request only with a
#                             valid cockpit session cookie and adds the credentials.
#                             The cockpit's own bind and port are kept as they are.
#
# The browser has to reach the cockpit and the relay under ONE host name (the
# session cookie is per host, ports do not matter): with the cockpit on 0.0.0.0
# or on the tailnet address, open it through the tailnet address. A cockpit bound
# to 127.0.0.1 only gets a loopback relay, usable on the box itself.
#
# Requires: the cockpit installed (dashboard/install-dashboard.sh, or any plain
# install.sh run) and opencode on your PATH: install.sh puts the pinned one there when there is none
# (docs/opencode.md), or install it from https://opencode.ai. Re-run after `opencode
# upgrade` is not needed: the unit points at the opencode command as found on
# your PATH (a symlink stays a symlink), the cockpit's Restart button serves the
# new version. Variables: OPENCODE_PORT, AGENT_PORT, AGENT_BIND (an address, or
# "tailscale"), AGENT_AUTO=1 (approve every tool call, like the oc launcher's
# --yolo; opencode serve has no such flag, so the unit sets OPENCODE_PERMISSION),
# AGENT_OUTPUT_TOKEN_MAX, AGENT_PATH (the PATH the service gets, as given, on the run
# that sets it). A re-run keeps what is installed unless you set it again: both ports,
# the relay's address, the AGENT_AUTO choice (0 or 1), and every directory of the
# service's PATH but what an editor's session put there (see below).
set -euo pipefail
# One spelling of the home, without a trailing slash, as install.sh says why: the cockpit's
# exact-argv sudoers is rendered from it and has to match the cockpit's own calls (2026-10-22).
while [ -n "${HOME:-}" ] && [ "$HOME" != / ] && [ "${HOME%/}" != "$HOME" ]; do HOME="${HOME%/}"; done; export HOME
# The PATH one slash at a time too: a login from such a home put "/home/nvidia//.local/bin" on it, which
# this script compares with the folders it builds on the home and writes into units (v1.38.1).
PATH="$(printf '%s' "$PATH" | tr -s /)"; export PATH
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT=opencode-web.service
DASH_UNIT_PATH=/etc/systemd/system/qwen38-dashboard.service
ENV_FILE="$HOME/.config/qwen38/opencode-web.env"
INSTALLED_OC="/etc/systemd/system/$UNIT"
# install.sh runs this script at every update, and an unset variable used to mean the
# default (4096, 30091, an address derived from the cockpit's), which install-dashboard.sh
# then wrote over the cockpit's own record of the relay: a relay on its own port or
# address went back to the defaults at each update, and without tailscale the Agent tab
# went dark (found in review, 2026-09-24). What is installed is read back; the
# environment still wins.
installed_dash(){ { grep -m1 -E "^Environment=$1=" "$DASH_UNIT_PATH" 2>/dev/null || true; } | cut -d= -f3-; }
if [ -z "${OPENCODE_PORT:-}" ] && [ -f "$INSTALLED_OC" ]; then
  OPENCODE_PORT="$(sed -n 's/^ExecStart=.* --port \([0-9][0-9]*\).*/\1/p' "$INSTALLED_OC" | head -1)"
fi
RELAY_PORT_INSTALLED="$(installed_dash COCKPIT_AGENT_PORT)"
if [ -n "$RELAY_PORT_INSTALLED" ] && [ "$RELAY_PORT_INSTALLED" != 0 ]; then
  AGENT_PORT="${AGENT_PORT:-$RELAY_PORT_INSTALLED}"
  AGENT_BIND="${AGENT_BIND:-$(installed_dash COCKPIT_AGENT_BIND)}"
fi
OPENCODE_PORT="${OPENCODE_PORT:-4096}"
AGENT_PORT="${AGENT_PORT:-30091}"
die(){ printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
note(){ printf 'NOTE: %s\n' "$*"; }
# As root, every path below moves to /root and what is rendered names root: the
# opencode server, and every tool call of the Agent tab, would run as root (found in
# review, 2026-09-24). install.sh and install-image.sh refuse the same way;
# ALLOW_ROOT=1 is for a box whose only login is root.
if [ "$(id -u)" = "0" ] && [ "${ALLOW_ROOT:-0}" != "1" ]; then
  die "run this as the user who will use the box, not as root: it calls sudo itself for the steps that need it${SUDO_USER:+ (your login is $SUDO_USER: drop the sudo)}."
fi

[[ "$OPENCODE_PORT" =~ ^[0-9]+$ ]] || die "OPENCODE_PORT must be a number"
[[ "$AGENT_PORT" =~ ^[0-9]+$ ]] || die "AGENT_PORT must be a number"
[ -f "$DASH_UNIT_PATH" ] || die "install the cockpit first: dashboard/install-dashboard.sh"
command -v python3 >/dev/null || die "python3 required"
command -v curl >/dev/null || die "curl required"

# ── opencode ────────────────────────────────────────────────────────────────
OPENCODE_BIN="${OPENCODE_BIN:-$(command -v opencode || true)}"
[ -n "$OPENCODE_BIN" ] && [ -x "$OPENCODE_BIN" ] || die "opencode not found on your PATH (install it from https://opencode.ai, or set OPENCODE_BIN=/path/to/opencode)"
# (captured, not piped into grep -q: under pipefail an early grep exit turns a
# perfectly good help text into a failure)
SERVE_HELP="$("$OPENCODE_BIN" serve --help 2>&1 || true)"
case "$SERVE_HELP" in *--hostname*) ;; *) die "this opencode has no 'serve --hostname' option; upgrade it (opencode upgrade)" ;; esac
OC_VERSION="$("$OPENCODE_BIN" --version 2>/dev/null || true)"; OC_VERSION="${OC_VERSION##*$'\n'}"
# (the version 2.x wraps itself in "opencode v2.3.0"; the grep takes the number out of
# either spelling, measured on the two lines 1.19.5 and 2.3.0 answer with)
OC_MAJOR="$(printf '%s' "$OC_VERSION" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 | cut -d. -f1 || true)"
# OpenCode 2 speaks the tab's language now (the relay probes /api/info, reads the ordered
# "permissions" and scrubs the answers that resolve {file:...} keys, measured on 2.3.0 the
# 2026-11-14) - but only on a box that chose it: ./install.sh --opencode 2 hands
# OPENCODE_MAJOR=2 down here. A stray 2.x found on the PATH (Homebrew Core's formula is
# one) is still refused before anything renders, so the installed unit keeps the opencode
# it runs (decision 455, 2026-10-22).
# The box's own choice, read from its marker when ./install.sh did not pass it down: a standalone
# re-run (the retry hint, docs/cockpit.md) must render the profile this box actually serves
# (independent review of the v2 migration, finding 4).
OPENCODE_MAJOR="${OPENCODE_MAJOR:-$(head -1 "$HOME/.config/qwen38/opencode.major" 2>/dev/null || true)}"
OPENCODE_MAJOR="${OPENCODE_MAJOR:-1}"
OC_PROFILE=1
if [ -n "$OC_MAJOR" ] && [ "$OC_MAJOR" -ge 2 ]; then
  if [ "${OPENCODE_MAJOR:-1}" != "2" ]; then
    die "$OPENCODE_BIN is opencode ${OC_VERSION##* }, OpenCode 2, which this box did not choose: run ./install.sh --opencode 2 (it serves it under the Agent view's own guards), or put the opencode 1.x this repo pins first on your PATH (./install.sh installs it in ~/.opencode/bin), or set OPENCODE_BIN to it (docs/opencode.md)."
  fi
  OC_PROFILE=2
fi

# ── where the cockpit listens decides where the relay may listen ───────────
COCKPIT_BIND="$({ grep -m1 -E '^Environment=COCKPIT_BIND=' "$DASH_UNIT_PATH" || true; } | cut -d= -f3-)"
COCKPIT_PORT="$({ grep -m1 -E '^Environment=COCKPIT_PORT=' "$DASH_UNIT_PATH" || true; } | cut -d= -f3-)"
COCKPIT_BIND="${COCKPIT_BIND:-127.0.0.1}"; COCKPIT_PORT="${COCKPIT_PORT:-30090}"
[ "$AGENT_PORT" != "$COCKPIT_PORT" ] || die "AGENT_PORT $AGENT_PORT is the cockpit's own port"
TS_IP="$(tailscale ip -4 2>/dev/null | head -1 || true)"
if [ -z "${AGENT_BIND:-}" ]; then
  case "$COCKPIT_BIND" in
    127.0.0.1|localhost|::1)
      AGENT_BIND=127.0.0.1
      note "the cockpit listens on loopback only, so the relay does too: the Agent view works on this box."
      note "to use it from your laptop: DASH_BIND=0.0.0.0 dashboard/install-dashboard.sh (or DASH_BIND=\$(tailscale ip -4)), then re-run this script." ;;
    0.0.0.0|tailscale|"$TS_IP")
      AGENT_BIND=tailscale ;;
    *)
      AGENT_BIND="$COCKPIT_BIND"
      note "the relay binds the cockpit's address $COCKPIT_BIND." ;;
  esac
fi
if [ "$AGENT_BIND" = "tailscale" ] && [ -z "$TS_IP" ]; then
  note "no tailnet address right now; the cockpit binds the relay as soon as tailscale is up."
fi
case "$AGENT_BIND" in *'|'*|*' '*) die "AGENT_BIND must be an address or 'tailscale'" ;; esac

# ── credentials: generated once, never printed ─────────────────────────────
mkdir -p "$(dirname "$ENV_FILE")"
if [ ! -s "$ENV_FILE" ]; then
  PW="$(python3 -c 'import secrets; print(secrets.token_urlsafe(36))')"
  ( umask 077; printf 'OPENCODE_SERVER_USERNAME=cockpit\nOPENCODE_SERVER_PASSWORD=%s\n' "$PW" > "$ENV_FILE" )
  echo "generated $ENV_FILE (mode 0600)"
fi
chmod 600 "$ENV_FILE"
grep -qE '^OPENCODE_SERVER_PASSWORD=.+' "$ENV_FILE" || die "$ENV_FILE has no OPENCODE_SERVER_PASSWORD line"
CRED_USER="$({ grep -m1 -E '^OPENCODE_SERVER_USERNAME=' "$ENV_FILE" || true; } | cut -d= -f2-)"; CRED_USER="${CRED_USER:-opencode}"
CRED_PW="$({ grep -m1 -E '^OPENCODE_SERVER_PASSWORD=' "$ENV_FILE" || true; } | cut -d= -f2-)"
if [ "$OC_PROFILE" = 2 ] && [ "$CRED_USER" != "opencode" ]; then
  # OpenCode 2 hardcodes the Basic username to "opencode" and ignores OPENCODE_SERVER_USERNAME
  # (read in packages/server/src/auth.ts of 2.3.0, and measured: the password alone opens
  # /api/*). A box installed in 1.x times with another name must present the one it accepts,
  # or every probe lands in 401. The password is kept exactly; nothing of it is printed.
  OC_ENV_FILE="$ENV_FILE" python3 - <<'PYEOF' || die "could not rewrite the username line of $ENV_FILE"
import os
path = os.environ["OC_ENV_FILE"]
lines = open(path).read().splitlines(True)
out = ["OPENCODE_SERVER_USERNAME=opencode\n" if ln.startswith("OPENCODE_SERVER_USERNAME=") else ln for ln in lines]
if not any(ln.startswith("OPENCODE_SERVER_USERNAME=") for ln in lines):
    out.insert(0, "OPENCODE_SERVER_USERNAME=opencode\n")
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as f:
    f.write("".join(out))
PYEOF
  CRED_USER=opencode
  note "$ENV_FILE now names opencode's fixed Basic user (OpenCode 2 ignores OPENCODE_SERVER_USERNAME); the password is the one it always held."
fi

# ── the output ceiling: the limits table's maximum ─────────────────────────
# opencode sends max_tokens = min(limit.output, this ceiling). limit.output is
# the per-model number, and it follows a model switch; the ceiling only has to
# stay above every value the table can produce. Reading it from the installed
# oc launcher is how the Agent tab inherited the installed target's own number:
# a box installed on the flash lane gave the tab a 64,000 ceiling, and after a
# switch to a 27B box in 1M mode (limit.output 200,000) a long turn was cut at
# 64,000 with no error and no text. AGENT_OUTPUT_TOKEN_MAX still wins.
OUT_MAX="${AGENT_OUTPUT_TOKEN_MAX:-}"
if [ -z "$OUT_MAX" ] && [ -x "$HERE/../oc-limits.sh" ]; then
  OUT_MAX="$("$HERE/../oc-limits.sh" --max-out 2>/dev/null || true)"
fi
OUT_MAX="${OUT_MAX:-200000}"
[[ "$OUT_MAX" =~ ^[0-9]+$ ]] || die "AGENT_OUTPUT_TOKEN_MAX must be a number"

# ── permissions: opencode's defaults, or every tool call approved ──────────
if [ -z "${AGENT_AUTO:-}" ] && grep -qE "^Environment='(OPENCODE_PERMISSION|OPENCODE_CONFIG_CONTENT)=" "/etc/systemd/system/$UNIT" 2>/dev/null; then
  AGENT_AUTO=1      # remembered from the installed unit, whichever OpenCode wrote it
fi
case "${AGENT_AUTO:-0}" in
  # opencode merges OPENCODE_PERMISSION after every rule of the config files and its last
  # matching rule wins, so a bare {"*":"allow"} overrode their deny rules too: an echo naming
  # an aliyuncs.com host ran under it (2026-10-28). The value repeats the rules for the hosts
  # of Cohere's report after its "*" (docs/agents-and-network.md).
  1) if [ "$OC_PROFILE" = 2 ]; then
       # OpenCode 2 ignores OPENCODE_PERMISSION entirely (absent from the 2.3.0 binary, and
       # issue #36990 lists it as "completely ignored" by design). The content it merges last
       # over every file is the way: --agent-auto-config-content puts the wildcard allow
       # first and this repo's shell denies after it - the last matching rule wins, so the
       # denies hold under the allow (measured on 2.3.0 the 2026-11-14: a shell command
       # naming a denied host did not run while an ordinary one did).
       CONTENT="$(python3 "$HERE/../oc-merge-limits.py" --agent-auto-config-content)" \
         || die "could not build AGENT_AUTO's content (oc-merge-limits.py --agent-auto-config-content)"
       AUTO_LINE="Environment='OPENCODE_CONFIG_CONTENT=$CONTENT'"
       note "AGENT_AUTO=1 on OpenCode 2: the web server approves every tool call but a shell command naming an Alibaba Cloud host or a cloud metadata address. Its allow overrides your own deny rules unless your config has one after it - the guard denies are written last, in the content and in the configs (docs/agents-and-network.md)."
     else
       PERM="$(python3 "$HERE/../oc-merge-limits.py" --agent-auto-permission)" \
         || die "could not build AGENT_AUTO's permission (oc-merge-limits.py --agent-auto-permission)"
       AUTO_LINE="Environment='OPENCODE_PERMISSION=$PERM'"
       note "AGENT_AUTO=1: the web server approves every tool call but a shell command naming an Alibaba Cloud host or a cloud metadata address. Its \"*\" overrides your own deny rules unless your config has a \"*\" of its own (docs/agents-and-network.md)."
     fi ;;
  0|"") if [ "$OC_PROFILE" = 2 ]; then
          AUTO_LINE="# (OpenCode 2's own permission defaults; re-run with AGENT_AUTO=1 to approve every tool call)"
        else
          AUTO_LINE="# (opencode's own permission defaults; re-run with AGENT_AUTO=1 to approve every tool call)"
        fi ;;
  *) die "AGENT_AUTO must be 1 or 0" ;;
esac
# The output ceiling (1.x) and the log-level case: OpenCode 2 reads neither the variable
# nor the capital (measured on 2.3.0 the 2026-11-14: --log-level INFO kills the boot,
# "info" starts it and journals at info; max_tokens follows the model's limit.output).
if [ "$OC_PROFILE" = 2 ]; then
  CAP_LINE="# OpenCode 2 sends max_completion_tokens from the model's limit.output in the config"
  CAP_LINE="$CAP_LINE"$'\n'"# (measured on 2.3.0 the 2026-11-14): there is no 32000 ceiling to lift here, and"
  CAP_LINE="$CAP_LINE"$'\n'"# OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX is read by nobody of it."
  # OpenCode 2 reads CLAUDE.md of every directory without AGENTS.md, and ~/.claude/CLAUDE.md
  # as the global fallback, into every prompt (since 2.3.1; measured on 2.3.2 in a sandbox:
  # the markers of both files land in the system message of each request). On a box that is
  # the owner's private file leaving in every Agent tab prompt, so the documented switch
  # skips every CLAUDE.md (read at server start, packages/cli/src/server-process.ts of
  # 2.3.2; docs/instructions "OPENCODE_DISABLE_CLAUDE_CODE_PROMPT=1"). The 1.x unit is not
  # touched: v1 keeps reading what it read.
  CAP_LINE="$CAP_LINE"$'\n'"Environment=OPENCODE_DISABLE_CLAUDE_CODE_PROMPT=1"
  SERVE_FLAGS="--print-logs --log-level info"
else
  CAP_LINE="# Same cap as the oc launcher: without it long thinking is cut at 32000 tokens"
  CAP_LINE="$CAP_LINE"$'\n'"# and the turn ends silently."
  CAP_LINE="$CAP_LINE"$'\n'"Environment=OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=$OUT_MAX"
  SERVE_FLAGS="--print-logs --log-level INFO"
fi

# ── PATH for the service: yours, deduplicated, the opencode dir first ──────
# (AGENT_PATH= overrides it, for an install run from a wrapper or an odd shell), and the
# installed one's directories after it, so an update run from a shell with a shorter PATH
# takes no tool away from the agent.
# Left out of both: what an editor's session puts on its terminal's PATH, known by its
# layout, which the forks of VS Code share: the remote CLI of its server (bin/remote-cli),
# its extensions' storage (User/globalStorage) and its versioned extensions
# (extensions/<publisher>.<name>-<version>). An update run from a VS Code terminal wrote
# three of them into the unit, so opencode-web restarted, and the remote CLI's directory,
# named after the editor's build, goes at the next VS Code update (the reference box,
# 2026-10-03).
# Each folder once, the installed PATH read one slash at a time as this script's own is: the unit v1.38.0
# wrote on a box whose home ends with a slash named opencode's folder in both spellings.
INSTALLED_PATH="$({ grep -m1 -E '^Environment=PATH=' "$INSTALLED_OC" 2>/dev/null || true; } | cut -d= -f3-)"
EDITOR_DIRS='/bin/remote-cli(/|$)|/User/globalStorage/|/extensions/[^/]+[.][^/]+-[0-9][^/]*(/|$)'
SVC_PATH="${AGENT_PATH:-$(printf '%s:%s:%s' "$(dirname "$OPENCODE_BIN")" "$PATH" "$INSTALLED_PATH" | tr ':' '\n' | tr -s / \
  | awk -v ed="$EDITOR_DIRS" 'NF && $0 !~ ed && !seen[$0]++' | paste -sd: -)}"
if [ -z "${AGENT_PATH:-}" ]; then
  GONE="$(printf '%s' "$INSTALLED_PATH" | tr ':' '\n' | awk -v ed="$EDITOR_DIRS" '$0 ~ ed' | paste -sd' ' -)"
  if [ -n "$GONE" ]; then note "opencode-web's PATH leaves out what an editor's session had put in it: $GONE"; fi
fi
case "$SVC_PATH" in *'|'*|*' '*) die "PATH contains a space or a | character; set PATH to something plain and re-run" ;; esac

# True (0) when a service has to be restarted to run what is on disk: it is not running, or
# it started before the last change of one of the files it reads (see install.sh).
stale_since(){
  local unit="$1" started f; shift
  [ "$(systemctl show -p ActiveState --value "$unit" 2>/dev/null)" = "active" ] || return 0
  started="$(systemctl show -p ExecMainStartTimestamp --value --timestamp=unix "$unit" 2>/dev/null | tr -dc '0-9')"
  [ -n "$started" ] || return 0
  for f in "$@"; do
    [ -e "$f" ] && [ "$(stat -L -c %Y "$f")" -gt "$started" ] && return 0
  done
  return 1
}

# ── render and install the unit ────────────────────────────────────────────
TMP_UNIT="$(mktemp)"
trap 'rm -f "$TMP_UNIT"' EXIT   # however this ends, as install-dashboard.sh does
# rendered by python, not sed: the values carry quotes, JSON and now newlines
# (CAP_LINE, AUTO_LINE), and a delimiter appearing in a path or a permission value
# would silently rewrite the wrong thing (each value goes through its own env var)
OC_TPL="$HERE/opencode-web.service.template" OC_OUT_UNIT="$TMP_UNIT" \
R_USER="$(id -un)" R_GROUP="$(id -gn "$(id -un)")" R_HOME="$HOME" R_OPENCODE_BIN="$OPENCODE_BIN" \
R_OPENCODE_PORT="$OPENCODE_PORT" R_PATH="$SVC_PATH" R_CAP_LINE="$CAP_LINE" \
R_AUTO_LINE="$AUTO_LINE" R_SERVE_FLAGS="$SERVE_FLAGS" \
python3 - <<'PYEOF' || die "could not render the unit"
import os, sys
repl = {"__USER__": "R_USER", "__GROUP__": "R_GROUP", "__HOME__": "R_HOME",
        "__OPENCODE_BIN__": "R_OPENCODE_BIN", "__OPENCODE_PORT__": "R_OPENCODE_PORT",
        "__PATH__": "R_PATH", "__CAP_LINE__": "R_CAP_LINE", "__AUTO_LINE__": "R_AUTO_LINE",
        "__SERVE_FLAGS__": "R_SERVE_FLAGS"}
text = open(os.environ["OC_TPL"]).read()
for mark, var in repl.items():
    text = text.replace(mark, os.environ[var])
open(os.environ["OC_OUT_UNIT"], "w").write(text)
PYEOF
grep -q '__[A-Z][A-Z0-9_]*__' "$TMP_UNIT" && die "unsubstituted placeholder in the unit render"
AGENT_CHANGED=0
cmp -s "$TMP_UNIT" "/etc/systemd/system/$UNIT" \
  || { sudo install -m 644 "$TMP_UNIT" "/etc/systemd/system/$UNIT"; AGENT_CHANGED=1; }
rm -f "$TMP_UNIT"
sudo systemctl daemon-reload
sudo systemctl enable --now "$UNIT"
# ── the server answers with the credentials (never on the command line) ────
health(){
  # the JSON answer of whichever OpenCode runs: 2.x answers its identity on /api/info and
  # its HTML page on every other path (issue #55704, measured on 2.3.0); 1.x answers
  # {"healthy":true,...} on /global/health. Only an answer carrying one of those keys is
  # a health answer: a page is not health, whoever served it.
  local out path
  out=""
  for path in /api/info /global/health; do
    out="$(printf 'user = "%s:%s"\nsilent\nmax-time = 3\nurl = "http://127.0.0.1:%s%s"\n' \
      "$CRED_USER" "$CRED_PW" "$OPENCODE_PORT" "$path" | curl -K - 2>/dev/null || true)"
    printf '%s' "$out" | grep -qE '"(healthy|pid)":' && { printf '%s' "$out"; return 0; }
  done
  printf '%s' "$out"
}
# A re-run with a new port, cap or binary must take effect; one that changed nothing must
# not end the turn the Agent tab is in, which it did at every ./install.sh (49 starts of
# opencode-web on the reference box on 2026-09-23; found in review, 2026-09-24). The binary's
# date alone missed an opencode updated where it lies by npm or Homebrew, which keep the date
# their archive gives the file, older than a server started since: the version the server
# answers with says it either way (found in review, 2026-10-22).
SERVED_VERSION="$(health | python3 -c 'import json, sys
try:
    print(json.load(sys.stdin).get("version") or "")
except Exception:
    print("")' 2>/dev/null || true)"
ON_DISK="${OC_VERSION##* }"
# The two configs the server reads at its start. One that the limits fit rewrote with nothing the running server
# has to read (limits it raised still fit as the server runs: oc-fit-limits.py --restart-agent-if-lowered) is no
# reason to restart it, and it was: the file being newer than the server, the fit left the Agent tab alone and
# this script ended its turn three seconds later, at every update whose boot had a larger pool than the one
# before (the reference box, updates to v1.45.2 and v1.45.3, 2026-11-09). The fit lists such a config in
# opencode-web.kept: which of the two it is, the sha256 it left it with and the start of the server that may keep
# running. Any other content, the other file with the same bytes, and any other server are judged by the dates
# as before.
KEPT_FILE="$HOME/.config/qwen38/opencode-web.kept"
STARTED="$(systemctl show -p ExecMainStartTimestamp --value --timestamp=unix "$UNIT" 2>/dev/null | tr -dc '0-9' || true)"
READ_CONFIGS=()
for which in artifact user; do
  case "$which" in artifact) f="$HOME/.config/qwen38/opencode.json" ;; *) f="$HOME/.config/opencode/opencode.json" ;; esac
  if [ -n "$STARTED" ] && [ -f "$f" ] && [ -f "$KEPT_FILE" ] \
     && grep -qxF "$which $(sha256sum < "$f" | cut -d' ' -f1) $STARTED" "$KEPT_FILE"; then
    continue
  fi
  READ_CONFIGS+=("$f")
done
if [ "$AGENT_CHANGED" -eq 1 ] || stale_since "$UNIT" "$OPENCODE_BIN" "$ENV_FILE" ${READ_CONFIGS[@]+"${READ_CONFIGS[@]}"} \
   || { [ -n "$SERVED_VERSION" ] && [ -n "$ON_DISK" ] && [ "${SERVED_VERSION#v}" != "${ON_DISK#v}" ]; }; then
  sudo systemctl try-restart "$UNIT"
fi

HEALTHY=""
for _ in $(seq 1 30); do
  case "$(health)" in *'"healthy":true'*|*'"pid":'*) HEALTHY=1; break ;; esac
  sleep 1
done
[ -n "$HEALTHY" ] || die "opencode did not answer on 127.0.0.1:$OPENCODE_PORT within 30 s (journalctl -u $UNIT -n 40)"
echo "opencode-web.service: opencode ${OC_VERSION:-?} serving on 127.0.0.1:$OPENCODE_PORT (loopback only)"

# ── the cockpit gets the relay; its own bind and port are kept ─────────────
DASH_AGENT_PORT="$AGENT_PORT" DASH_AGENT_BIND="$AGENT_BIND" DASH_QUIET=1 \
  DASH_AGENT_UPSTREAM="http://127.0.0.1:$OPENCODE_PORT" "$HERE/install-dashboard.sh"

# ── the relay refuses a request without a session (that is the proof it is up)
RELAY_HOST="$AGENT_BIND"; [ "$RELAY_HOST" = "tailscale" ] && RELAY_HOST="$TS_IP"
if [ -n "$RELAY_HOST" ]; then
  code=""
  for _ in $(seq 1 20); do
    code="$(curl -s -o /dev/null -m 2 -w '%{http_code}' "http://$RELAY_HOST:$AGENT_PORT/global/health" || true)"
    [ "$code" = "401" ] && break
    sleep 1
  done
  [ "$code" = "401" ] || die "the relay did not come up on $RELAY_HOST:$AGENT_PORT (got '$code'; journalctl -u qwen38-dashboard -n 30)"
  echo "agent relay: http://$RELAY_HOST:$AGENT_PORT refuses without a cockpit session (401), as it should"
  echo
  echo "Agent view: http://$RELAY_HOST:$COCKPIT_PORT/#agent   (log in with the API key; the view needs no second login)"
else
  echo "agent relay: binds the tailnet address when tailscale is up; then open http://<tailnet address>:$COCKPIT_PORT/#agent"
fi
echo "Remove with: sudo systemctl disable --now $UNIT; sudo rm -f /etc/systemd/system/$UNIT; DASH_AGENT_PORT=0 dashboard/install-dashboard.sh"
