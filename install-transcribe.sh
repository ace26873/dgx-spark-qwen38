#!/usr/bin/env bash
# Install the NVIDIA Nemotron Transcribe lane: Nemotron-3.5-Transcribe-30B-A4B in GGUF Q8_0 with its
# F16 projector, served by llama.cpp's own server on a port of its own (30036), answering
# /v1/audio/transcriptions and /v1/realtime, loaded on demand like the image lane and the video lane
# (decisions 1387 and 1388 of Hasan, 2026-11-15).
#
#   ./install-transcribe.sh                 first install, or update in place
#   ./install-transcribe.sh --no-smoke      skip the proof at the end
#   ./install-transcribe.sh --uninstall     remove the unit (the GGUF files and the image kept, said)
#   ./install-transcribe.sh --units-only    the unit alone: no image, no checkpoint (install.sh, on demand)
#   ./install-transcribe.sh --prepare       its image, pulled with no sudo: the cockpit's first Load
#                                           runs it (./prepare-lane.sh transcribe)
#   ./install-transcribe.sh --fetch         its GGUF files, into the cache, with no sudo:
#                                           switch-model.sh transcribe runs it before --stage
#   ./install-transcribe.sh --stage FILE    the unit for the files in the cache, written to FILE with
#                                           no sudo: switch-model.sh transcribe installs FILE
#
# WHY IT IS A LANE OF ITS OWN, LOADED ON DEMAND. Its 35 GB of weights do not hold beside the 27B (the
# box kept 81 GiB available at ready and 76 at the lowest with nothing else serving, and the 27B takes
# about 100, essai-20261114-1436): so this lane comes up when no engine serves, and its unit says that
# to systemd with Conflicts=, like every engine lane's. It is installed by default and prepared at its
# first Load, as the image lane and the video lane are; ./install.sh --no-transcribe leaves it out of
# the run that passes it. Its port is its own (30036, beside the speech lane's 30024), not the text
# lanes': what it answers is audio, and the proxy does not relay it, as it relays none of the speech
# lane's routes: programs on the box ask the engine on this port with the box's key.
#
# WHY llama.cpp. Decision 1388: the best engine by its sources, not an A/B of each. The model ships GGUF
# (mradermacher's quantization of NVIDIA's Nemotron-3.5-Transcribe-30B-A4B), and it was measured here on
# llama.cpp's pinned image (b12316, the Unsloth GGUF lane's): French WER 2.76 % over the 20-excerpt set
# against 5.6 % for the speech lane's best Qwen3-ASR 1.7B and 9.7 % for its default, English 0.40 %,
# ready in 15 s, real time factor 0.096 (~/qa-tools/etudes/nouvelles-lanes/RESULTATS.md, 2026-11-14).
#
# The pins are sha256 and size read from the cache's own blobs on the reference box on 2026-11-15
# (~/qa-tools/etudes/nouvelles-lanes/EPINGLES.md); check-pins.sh asks the Hub's LFS pointers every day.
# The engine flags are the ones that measure ran with (--reasoning off -c 65536, essai.log); the model's
# name in its answers is what llama-server derives from the file, the run's --alias of no consequence to
# a single-model server (its README: "model" is ignored in single model mode).
set -euo pipefail
# One spelling of the home, without a trailing slash, as install.sh says why: the cockpit's
# exact-argv sudoers is rendered from it and has to match the cockpit's own calls (2026-10-22).
while [ -n "${HOME:-}" ] && [ "$HOME" != / ] && [ "${HOME%/}" != "$HOME" ]; do HOME="${HOME%/}"; done; export HOME
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT=qwen38-transcribe.service
INSTALLED="${LANES_UNIT_DIR:-/etc/systemd/system}/$UNIT"
CONTAINER=qwen38-transcribe
die(){ printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
step(){ printf '\n\033[1;36m── %s\033[0m\n' "$*"; }

# Under sudo every path below moves to /root: the cache, the config and the key the unit
# reads. install.sh refuses the same way, for the same reason.
if [ "$(id -u)" = "0" ] && [ "${ALLOW_ROOT:-0}" != "1" ]; then
  die "run this as the user who will use the box, not as root (everything it installs is addressed from \$HOME)."
fi

SMOKE=1; ACTION=install; MODE=full
for a in "$@"; do case "$a" in
  --no-smoke) SMOKE=0 ;;
  --uninstall) ACTION=uninstall ;;
  # The two halves of a lane installed on demand, as for the other lanes (install-speech.sh's shape, on
  # a port of its own, and install-flash-unsloth.sh's --fetch/--stage for the switch).
  --units-only) [ "$MODE" = full ] || die "--units-only and --prepare are the two halves of an install: one at a time"; MODE=units ;;
  --prepare) [ "$MODE" = full ] || die "--units-only and --prepare are the two halves of an install: one at a time"; MODE=prepare ;;
  --fetch) [ "$MODE" = full ] || die "--fetch is a mode of its own"; MODE=fetch ;;
  --stage) [ "$MODE" = full ] || die "--stage is a mode of its own"; MODE=stage ;;
  -h|--help) awk 'NR > 1 && !/^#/ { exit } NR > 1' "$0"; exit 0 ;;
  *) if [ "$MODE" = stage ] && [ -z "${STAGE_OUT:-}" ]; then STAGE_OUT="$a"; else die "unknown option: $a"; fi ;;
esac; done
if [ "$MODE" = stage ] && [ -z "${STAGE_OUT:-}" ]; then die "--stage takes the file to write the unit to"; fi

CONFIG_DIR="$HOME/.config/qwen38"
# shellcheck source=lanes.sh
. "$HERE/lanes.sh"
# A 64 GB DGX Spark cannot hold this lane (boxmem.py): 37 GB of weights before the engine's own memory, and never
# run on one. Refused before anything is fetched; --uninstall still takes away what an older release put there.
[ "$ACTION" = uninstall ] || [ "$(box_mem_class)" != 64 ] || lane_fits transcribe || die "The NVIDIA Nemotron Transcribe lane needs a 128 GB DGX Spark, and this box is a 64 GB one."
# A terminal opened before the account joined the docker group: the run goes on with it (lanes.sh).
docker_session bash "$HERE/${BASH_SOURCE[0]##*/}" "$@"
# what the llama.cpp lanes' installers share: the fetch of the GGUF files, the check of those files, the
# service scripts and the preflight (this lane's unit goes in with lanes.sh's unit_put: it is not a text
# lane, it takes none of the boot's record, and its proof asks its own routes)
# shellcheck source=llamacpp-lane.sh
. "$HERE/llamacpp-lane.sh"
VLANE=transcribe; VLANE_SHORT="NVIDIA Nemotron Transcribe"

# The pins, at column 0 like install.sh's. switch-model.sh, check-pins.sh and the cockpit read them here.
# mradermacher's GGUF of NVIDIA's Nemotron-3.5-Transcribe-30B-A4B, the quantization the box measured
# (Q8_0, 35,004,647,040 bytes, and its F16 projector, 1,468,544,320), read from the cache's own blobs
# (~/qa-tools/etudes/nouvelles-lanes/EPINGLES.md, 2026-11-15).
TRANSCRIBE_REPO="mradermacher/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF"
TRANSCRIBE_REV="76a736c119c734a6526a37bcd6fd2c4d3b252b03"
TRANSCRIBE_FILE="NVIDIA-Nemotron-3.5-Transcribe-30B-A4B.Q8_0.gguf=4d2112abcf4a9edd14bf451696fdc647ad620163dd2e0a0fe86bd800fce55949:35004647040"
TRANSCRIBE_MMPROJ="NVIDIA-Nemotron-3.5-Transcribe-30B-A4B.mmproj-f16.gguf=0f7b0579d4d224490da9654df76cea12e53fc0135ed5c9d00e393605bc3d59d3:1468544320"
# The two values the measured engine ran with (essai.log, 2026-11-14): --reasoning off, and its window
# of 65,536 tokens. Nothing else: no -np, no cache cap, no -ngl, exactly the form that answered WER 2.76 %.
TRANSCRIBE_CTX="65536"
# llama.cpp's server image, the digest the Qwen3.8 Flash-Next Unsloth GGUF lane pins and this lane was
# measured on (b12316): one image for the llama.cpp lanes, as the two Nemotron lanes share one.
TRANSCRIBE_IMAGE="ghcr.io/ggml-org/llama.cpp@sha256:a5600a59c704208491adbb96505454a8d3d4cd3a9db48b03cf2760b660b7fd16"
LOCAL_IMAGE="$TRANSCRIBE_IMAGE"
# the two files at this revision, 36.47 GB (10^9 bytes), rounded up: the room their fetch asks.
TRANSCRIBE_GB="37"
# Its pull at its peak: the count install-flash-unsloth.sh read from the same image (4.50 GB unpacked,
# 2.70 to download, 6.7 GiB with the containerd store's copy); ceil(x 1.02) + 1 as df -BG rounds up.
IMAGE_GB=8

# The lane's port and bind, read back as install-speech.sh reads its launcher's (the unit is this lane's
# only record; the cockpit's Load and the switch re-render, they do not invent).
installed_flag(){  # $1 the flag (--port, --host): the value the installed unit gives it, its last word
  # shellcheck disable=SC2086
  { grep -hoE -- "$1 [^ \\\\']+" "$INSTALLED" 2>/dev/null || true; } | tail -1 | awk '{print $2}'
}
PORT="${TRANSCRIBE_PORT:-$(installed_flag --port)}"; PORT="${PORT:-30036}"
case "$PORT" in ''|*[!0-9]*) die "TRANSCRIBE_PORT takes a number, got: $PORT" ;; esac
ENGINE_BIND="${TRANSCRIBE_BIND:-$(installed_flag --host)}"; ENGINE_BIND="${ENGINE_BIND:-127.0.0.1}"
case "$ENGINE_BIND" in *[!0-9.]*|""|*..*) [ "$ENGINE_BIND" = localhost ] || die "TRANSCRIBE_BIND takes an IPv4 address, got: $ENGINE_BIND" ;; esac
# The Hugging Face cache the unit mounts, read the way the other lanes read it (lanes.sh).
HF_CACHE="${HF_CACHE:-$(unit_hf_cache "$INSTALLED")}"; HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

model_host(){ llane_path "$TRANSCRIBE_REPO" "$TRANSCRIBE_REV" "$1"; }
model_box(){ printf '/root/.cache/huggingface/hub/models--%s/snapshots/%s/%s\n' \
  "${TRANSCRIBE_REPO//\//--}" "$TRANSCRIBE_REV" "${1%=*}"; }

if [ "$ACTION" = uninstall ]; then
  step "Removing the NVIDIA Nemotron Transcribe lane"
  # The unit, and the boot with it: a switch to this lane makes it the one unit enabled at boot, as a switch to
  # the image lane or the video lane does (switch-model.sh transcribe), so removing it then gives the boot back to
  # the text lane it took it from (vllm-lane.sh). No opencode provider to take out: it serves no text.
  vlane_uninstall_unit
  # The image is pulled, not built: its local tag goes (lanes.sh), and the GGUF files stay, said.
  llane_uninstall_runtime "$TRANSCRIBE_REPO"
  exit 0
fi

# The unit, rendered for this box: written to $1. Its start checks the files the engine reads
# (llamacpp-files.sh): the two pinned GGUF, each where the cache keeps it.
render_unit(){
  local files f pair
  files=""
  for pair in "$TRANSCRIBE_FILE" "$TRANSCRIBE_MMPROJ"; do
    f="$(model_host "$pair")"; files="$files $f"
  done
  sed -e "s|__USER__|$(id -un)|g" -e "s|__GROUP__|$(id -gn "$(id -un)")|g" -e "s|__HOME__|$HOME|g" \
      -e "s|__IMAGE__|$TRANSCRIBE_IMAGE|g" -e "s|__HF_CACHE__|$HF_CACHE|g" -e "s|__FILES__|${files# }|g" \
      -e "s|__MODEL__|$(model_box "$TRANSCRIBE_FILE")|g" -e "s|__MMPROJ__|$(model_box "$TRANSCRIBE_MMPROJ")|g" \
      -e "s|__CTX__|$TRANSCRIBE_CTX|g" \
      -e "s|__ENGINE_BIND__|$ENGINE_BIND|g" -e "s|__TRANSCRIBE_PORT__|$PORT|g" \
      "$HERE/$UNIT.template" > "$1"
  if grep -q '__[A-Z][A-Z0-9_]*__' "$1"; then die "the unit template still holds an unsubstituted placeholder: $(grep -o '__[A-Z][A-Z0-9_]*__' "$1" | sort -u | tr '\n' ' ')"; fi
}

# The one part that needs root: the unit and its start scripts. A lane loaded on demand takes the new
# unit whenever it changes: it serves nothing between Loads, and its next Load starts what this run
# installed (the note vlane_put_unit keeps for a serving lane cannot apply to a stopped one, and this
# lane is not a text lane: no boot record, no clients of the proxy to follow).
install_service(){
  step "4/5 Service"
  llane_service_files
  RENDER="$(mktemp)"; trap 'rm -f "$RENDER"' EXIT
  render_unit "$RENDER"
  put="$(unit_put "$RENDER" "$INSTALLED")"
  case "$put" in
    same) echo "unit unchanged" ;;
    deps) sudo systemctl daemon-reload
          echo "unit updated: only the lines naming the other lanes changed, which the lane does not read" ;;
    *)    sudo systemctl daemon-reload
          echo "unit installed at $INSTALLED" ;;
  esac
}

# The room the files still missing need on the cache's disk, said before a byte is downloaded.
missing_gb(){ llane_missing_gb "$TRANSCRIBE_REPO" "$TRANSCRIBE_FILE" "$TRANSCRIBE_MMPROJ"; }
fetch_files(){ llane_fetch "$TRANSCRIBE_REPO" "$TRANSCRIBE_REV" "$TRANSCRIBE_FILE" "$TRANSCRIBE_MMPROJ"; }

# --fetch: the files, for switch-model.sh, with no sudo, the room checked first.
if [ "$MODE" = fetch ]; then
  need_gb="$(missing_gb)"
  if [ "$need_gb" -gt 0 ]; then
    need_gib=$(( (need_gb * 1000000000 + 1073741823) / 1073741824 )); free_gib="$(free_gb_at "$HF_CACHE")"
    [ "${free_gib:-0}" -ge "$need_gib" ] \
      || die "${free_gib:-?} GiB free under HF_CACHE=$HF_CACHE, the GGUF files need about $need_gib GiB more. Free some space first: nothing was downloaded."
  fi
  fetch_files
  exit 0
fi

# --stage: the unit for the files in the cache, for switch-model.sh, with no sudo.
if [ "$MODE" = stage ]; then
  [ -f "$CONFIG_DIR/llamacpp-files.sh" ] || install -m 755 "$HERE/llamacpp-files.sh" "$CONFIG_DIR/llamacpp-files.sh"
  render_unit "$STAGE_OUT"
  exit 0
fi

step "1/5 Preflight (NVIDIA Nemotron Transcribe: $TRANSCRIBE_REPO @ ${TRANSCRIBE_REV:0:12})"
vlane_preflight engine-preflight.sh llamacpp-files.sh gguf-fetch.py
WEIGHTS_GB="$(missing_gb)"
vlane_room "$LOCAL_IMAGE"
if [ "$MODE" = units ]; then
  echo "OK: $(uname -m), key present; the image and the GGUF files come with the lane's first Load"
  install_service
  step "Done: the NVIDIA Nemotron Transcribe lane is installed; its first Load prepares it"
  echo "  the cockpit: Load on NVIDIA Nemotron Transcribe in the Lanes view pulls its image (about 4.5 GB, 2.7 to"
  echo "               download, already there on a box that loaded the Qwen3.8 Flash-Next Unsloth GGUF"
  echo "               lane) and fetches its GGUF files (about $TRANSCRIBE_GB GB), then starts it"
  echo "  a terminal : ./install-transcribe.sh --prepare, then ./switch-model.sh transcribe and the commands it prints"
  exit 0
fi

step "2/5 Image (${TRANSCRIBE_IMAGE:0:40}...)"
if docker image inspect "$TRANSCRIBE_IMAGE" >/dev/null 2>&1; then
  echo "already here"
else
  docker pull "$TRANSCRIBE_IMAGE" \
    || die "docker pull of $TRANSCRIBE_IMAGE failed. Causes: no internet, a GitHub Container Registry rate limit (retry in a few minutes), or the digest removed upstream."
fi
pin_tag transcribe "$TRANSCRIBE_IMAGE"
unpin_stale transcribe "$TRANSCRIBE_IMAGE" || true

if [ "$MODE" = prepare ]; then
  step "Done: the NVIDIA Nemotron Transcribe lane's image is ready; its Load fetches the GGUF files and starts it"
  exit 0
fi

step "3/5 GGUF files ($TRANSCRIBE_REPO @ ${TRANSCRIBE_REV:0:12}, two files, resumable)"
fetch_files

install_service

if [ "$SMOKE" -eq 0 ]; then step "Done (proof skipped)"; exit 0; fi

step "5/5 Proving it transcribes (a spoken sentence heard back, then the box goes back to the lane it was serving)"
# The proof starts this lane, alone: every engine lane's proof does, and the one that served comes back
# when it ends. The request is a WAV this repository carries (tests/fixtures/transcribe-smoke.wav:
# "Bonjour, ceci est un essai de synthèse vocale sur la DGX Spark." at 24 kHz mono, 207 KB, spoken on the
# reference box the 14/11 by Liquid AI's LFM3-Audio-2B, the speech model the study ran beside this one).
# The quality claim is not made here: it is the 2.76 % French WER of the 20-excerpt
# set, a box measure the lane's page cites (docs/transcribe-lane.md).
WAS=""
for u in qwen38-sglang.service qwen38-flash.service qwen38-image.service qwen38-video.service qwen38-kolibri.service qwen38-nemotron.service qwen38-nemotron-super.service qwen38-flash-local.service qwen38-ornith-122b.service qwen38-flash-unsloth.service qwen38-image-h3.service qwen38-lfm25.service qwen38-lfm3.service; do
  systemctl is-active --quiet "$u" 2>/dev/null && WAS="$u"
done
[ -z "$WAS" ] || echo "note: $WAS is serving and will be stopped for this test, then started again."
WAS_SELF=0; systemctl is-active --quiet "$UNIT" 2>/dev/null && WAS_SELF=1
transcribe_restore(){
  [ "$WAS_SELF" -eq 1 ] || sudo systemctl stop "$UNIT" 2>/dev/null || true
  if [ -n "$WAS" ]; then
    echo "starting $WAS again"
    sudo systemctl start "$WAS" || echo "WARNING: could not restart $WAS. Do it by hand: sudo systemctl start $WAS"
  fi
  rm -rf "${SMOKE_DIR:-}" "${RENDER:-}"
}
trap transcribe_restore EXIT
SMOKE_DIR="$(mktemp -d)"
sudo systemctl start "$UNIT"
SINCE="$(date '+%Y-%m-%d %H:%M:%S')"
r0="$(systemctl show -p NRestarts --value "$UNIT" 2>/dev/null | tr -dc '0-9')"
ready=0
for i in $(seq 1 30); do
  curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ready=1; break; }
  if [ "$(systemctl is-active "$UNIT" || true)" = failed ] \
     || [ "$(systemctl show -p NRestarts --value "$UNIT" 2>/dev/null | tr -dc '0-9')" -gt "${r0:-0}" ]; then
    { sudo journalctl -u "$UNIT" --since "$SINCE" --no-pager || true; } | tail -40
    die "the lane died while starting; the journal above says why."
  fi
  [ $((i % 5)) -eq 0 ] && echo "  still loading... ($((i * 8)) s; about 15 s on the reference box once its files are on disk)"
  sleep 8
done
[ "$ready" -eq 1 ] || { { sudo journalctl -u "$UNIT" --since "$SINCE" --no-pager || true; } | tail -40; die "the lane did not answer /health within 4 min. The journal above says why."; }
echo "up after ~$(( (i - 1) * 8 )) s"
KEY="$(head -1 "$CONFIG_DIR/api-key")"
code="$(curl -sS -o "$SMOKE_DIR/heard.json" -w '%{http_code}' -m 300 "http://127.0.0.1:$PORT/v1/audio/transcriptions" \
  -H @<(printf 'Authorization: Bearer %s\n' "$KEY") \
  -F "file=@$HERE/tests/fixtures/transcribe-smoke.wav" -F language=fr || true)"
[ "$code" = 200 ] || { head -c 300 "$SMOKE_DIR/heard.json" 2>/dev/null; echo; { sudo journalctl -u "$UNIT" --since "$SINCE" --no-pager || true; } | tail -20; die "the recognizer did not answer /v1/audio/transcriptions (HTTP ${code:-000})."; }
python3 - "$SMOKE_DIR/heard.json" <<'PY' || die "what it heard back does not carry the sentence: the route answered, not the model."
import json, re, sys, unicodedata
heard = json.load(open(sys.argv[1])).get("text") or ""
norm = lambda s: re.sub(r"\W+", " ", unicodedata.normalize("NFKD", s.lower()).encode("ascii", "ignore").decode()).split()
said = norm("bonjour ceci est un essai de synthese vocale sur la dgx spark")
got = set(norm(heard))
keep = [w for w in said if w in got]
print(f"   heard: {heard[:110]!r}")
if len(keep) < 5:
    raise SystemExit(f"only {len(keep)} of {len(said)} words came back")
PY
# The key must be in none of the lane's journal lines (the same check every engine lane's proof ends with).
journal_copy="$(mktemp)"
# shellcheck disable=SC2024  # the copy is this user's own temporary file, written by this shell
if ! sudo journalctl -u "$UNIT" --since "$SINCE" --no-pager > "$journal_copy" 2>/dev/null; then
  rm -f "$journal_copy"; die "the lane's journal could not be read to check the key is not in it."
fi
if grep -q -F -f <(head -1 "$CONFIG_DIR/api-key") "$journal_copy"; then
  rm -f "$journal_copy"; die "the API key appears in the lane's journal: rotate it, and report this."
fi
echo "   the key is in none of the lane's $(wc -l < "$journal_copy") journal lines"
rm -f "$journal_copy"

step "Done: the NVIDIA Nemotron Transcribe lane is installed and proved it transcribes on $ENGINE_BIND:$PORT"
echo "  transcribe : curl http://$ENGINE_BIND:$PORT/v1/audio/transcriptions -H 'Authorization: Bearer <the box key>' \\"
echo "                 -F file=@audio.wav -F language=fr"
echo "  realtime   : the /v1/realtime websocket of the same engine, on this port (docs/transcribe-lane.md)"
echo "  load it    : Load on NVIDIA Nemotron Transcribe in the cockpit, or ./switch-model.sh transcribe"
echo "               (it starts when no other engine serves: 37 GB of weights do not hold beside one)"
