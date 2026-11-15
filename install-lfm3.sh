#!/usr/bin/env bash
# Install the Liquid AI LFM3 lane: three small Liquid AI models served by ONE llama-server in its router
# mode (decision 897 of Hasan, 2026-11-01: one shared engine, and 1383 to 1384 for this lane), on the text
# lanes' port behind the proxy and its key:
#
#   lfm3-1.7b    chat: LFM3-1.7B Q8_0, 65,536 tokens held by 8 requests at once
#   d1-2b        decisions: POST /v1/systemone, answered natively by d1-2B Q8_0 with its mmproj
#   lfm3-audio   audio:  /v1/audio/transcriptions and /v1/audio/speech, LFM3-Audio-2B Q8_0 with its mmproj
#
# The name a client puts in "model" picks the child the router forwards to; the router loads the three at
# its start and keeps them (they are small: 6.5 GB of pinned GGUF files in all).
#
#   ./install-lfm3.sh                 first install, or update in place
#   ./install-lfm3.sh --no-smoke      skip the proof at the end
#   ./install-lfm3.sh --uninstall     remove the unit (the GGUF files and the image kept, said)
#   ./install-lfm3.sh --units-only    the unit alone: no image, no checkpoints (install.sh, on demand)
#   ./install-lfm3.sh --prepare       its image, pulled with no sudo: the cockpit's first Load
#                                     runs it (./prepare-lane.sh lfm3)
#   ./install-lfm3.sh --fetch         its GGUF files, into the cache, with no sudo:
#                                     switch-model.sh lfm3 runs it before --stage
#   ./install-lfm3.sh --stage FILE    the unit for the files in the cache, written to FILE with no
#                                     sudo, its router presets written first: switch-model.sh lfm3
#                                     installs FILE
#
# WHY ONE ROUTER AND NOT THREE LANES. The three models answer three different routes of one OpenAI-shaped
# API (chat, decisions, audio) and weigh about 6.5 GB together: what llama.cpp calls the router server
# loads them in one process, each in its own child, and forwards each request to the one its "model" names
# (their --alias is the model's name, its server-models.cpp). Three lanes of their own would be three
# images, three units, three loads and three waits.
#
# WHY A LANE OF ITS OWN BESIDE THE 27B. The family is the one this box measured (docs/lfm3-lane.md:
# 103.6 tok/s in chat on its image, GSM8K 250 at 0.844, a /v1/systemone decision at the contract in 54 to
# 60 ms, French WER 8.02 % and speech synthesis at 24 kHz): a helper lane that answers chat, decisions and
# audio while the big lanes idle, on 6.5 GB of weights and one engine. Its unit says what every engine
# unit says (Conflicts=): one engine holds the GPU and the text port at a time, like the Kolibri lane and
# the Nemotron lanes; whether it also holds beside a serving 27B at its peak is a box measure, not a claim
# of this script (the pilot measures it before the lane is advertised beside another).
#
# install.sh puts this lane's unit in on every plain install, like the other lanes, and its first Load from
# the cockpit pulls its image and fetches its files; ./install.sh --no-lfm3 leaves it out of the run that
# passes it. It takes about 7 GB of GGUF files and an image of 4.5 GB (2.7 to download), the
# Qwen3.8 Flash-Next Unsloth GGUF lane's own image (b12316, pinned by the same digest): when the disk does
# not have the room, this refuses before downloading anything.
#
# The pins are sha256 and size read from the cache's own blobs on the reference box on 2026-11-15
# (~/qa-tools/etudes/nouvelles-lanes/EPINGLES.md, run essai-20261114-1436); check-pins.sh asks the Hub's
# LFS pointers every day. The router's per-model flags are the ones that engine answered with on the box
# the 14/11 (essai.log): d1 with -b 8192 -ub 8192, the chat model with -c 65536 -np 8 --jinja, the audio
# model with -c 32768. Nothing here adds a flag the server was not measured with: they live in
# qwen38-lfm3-models.ini.template, which this script renders with the pinned files' paths.
set -euo pipefail
# One spelling of the home, without a trailing slash, as install.sh says why: the cockpit's
# exact-argv sudoers is rendered from it and has to match the cockpit's own calls (2026-10-22).
while [ -n "${HOME:-}" ] && [ "$HOME" != / ] && [ "${HOME%/}" != "$HOME" ]; do HOME="${HOME%/}"; done; export HOME
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT=qwen38-lfm3.service
# where systemd reads the units, the one variable lanes.sh and llamacpp-lane.sh read the other units by
INSTALLED="${LANES_UNIT_DIR:-/etc/systemd/system}/$UNIT"
CONTAINER=qwen38-lfm3
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
  # The two halves of a lane installed on demand, as for the other lanes: --units-only puts in
  # what needs root (the unit, its start scripts) and nothing heavy, and --prepare pulls the
  # lane's image as the user, with no sudo, when the cockpit first loads the lane. The GGUF files
  # come with that Load's switch (--fetch), as a checkpoint comes with any Load.
  --units-only) [ "$MODE" = full ] || die "--units-only and --prepare are the two halves of an install: one at a time"; MODE=units ;;
  --prepare) [ "$MODE" = full ] || die "--units-only and --prepare are the two halves of an install: one at a time"; MODE=prepare ;;
  --fetch) [ "$MODE" = full ] || die "--fetch is a mode of its own"; MODE=fetch ;;
  --stage) [ "$MODE" = full ] || die "--stage is a mode of its own"; MODE=stage ;;
  -h|--help) awk 'NR > 1 && !/^#/ { exit } NR > 1' "$0"; exit 0 ;;
  *) if [ "$MODE" = stage ] && [ -z "${STAGE_OUT:-}" ]; then STAGE_OUT="$a"; else die "unknown option: $a"; fi ;;
esac; done
if [ "$MODE" = stage ] && [ -z "${STAGE_OUT:-}" ]; then die "--stage takes the file to write the unit to"; fi

CONFIG_DIR="$HOME/.config/qwen38"
# the text lane the box comes back to, one record shared with switch-model.sh (lanes.sh)
# shellcheck source=lanes.sh
. "$HERE/lanes.sh"
# A 64 GB DGX Spark has never run this lane (boxmem.py): refused there before anything is fetched,
# and --uninstall still takes away what an older release put there.
[ "$ACTION" = uninstall ] || [ "$(box_mem_class)" != 64 ] || lane_fits lfm3 || die "The Liquid AI LFM3 lane needs a 128 GB DGX Spark, and this box is a 64 GB one."
# A terminal opened before the account joined the docker group: the run goes on with it (lanes.sh).
docker_session bash "$HERE/${BASH_SOURCE[0]##*/}" "$@"
# what the llama.cpp lanes' installers share: the engine's cache and port, the uninstall, the service, the
# preflight and the room, the fetch of the GGUF files, and the proof (part of it vllm-lane.sh's)
# shellcheck source=llamacpp-lane.sh
. "$HERE/llamacpp-lane.sh"
VLANE=lfm3; VLANE_SHORT="Liquid AI LFM3"

# The pins, at column 0 like install.sh's. switch-model.sh, check-pins.sh and the cockpit read them here.
# The three Liquid AI GGUF releases this lane serves, each file by its sha256 and its size in bytes
# (PATH=SHA256:SIZE, gguf-fetch.py's form), read from the cache's own blobs on the reference box
# (~/qa-tools/etudes/nouvelles-lanes/EPINGLES.md, 2026-11-15). The chat model carries no mmproj; the
# decision model's and the audio model's each carry theirs, and the router pairs a model with the
# projector of its section (its common/preset.cpp reads the mmproj option per preset).
LFM3_CHAT_REPO="LiquidAI/LFM3-1.7B-GGUF"
LFM3_CHAT_REV="cd0c74c575b5c416f3011b5b4c298077f26ca8c2"
LFM3_CHAT_FILE="LFM3-1.7B-Q8_0.gguf=8f6b24182258205e9e01e21f1f24a1f208c1e8115578825d202ce39976af20b6:1846238496"
LFM3_D1_REPO="LiquidAI/d1-2B-GGUF"
LFM3_D1_REV="4a181094d1b6addfaa3a0956f78e7ab2aa406a0c"
LFM3_D1_FILE="d1-2B-Q8_0.gguf=073313d92b53b0ac8193db6e906cb3c78a98183d244b15b7b8f5426115c6738c:1846240576"
LFM3_D1_MMPROJ="mmproj-d1-2B-F16.gguf=c54d0c0cc66c7e5b6b516204cab556e41d1da9e9cd970245a57d7b509af68ab3:378995744"
LFM3_AUDIO_REPO="LiquidAI/LFM3-Audio-2B-GGUF"
LFM3_AUDIO_REV="39670cb5c42e9c4bfe046519ba7ffbe47f9552ad"
LFM3_AUDIO_FILE="LFM3-Audio-2B-Q8_0.gguf=d4e36902ba8f0ee7b333596030a59d650f88ac76660b4d9ba5ff77c3c2258fc0:1846236704"
LFM3_AUDIO_MMPROJ="mmproj-LFM3-Audio-2B-F16.gguf=bd0fce0670c96e405a753f996701dee6568bc6a1318a80091d74fb58f7510d79:570846400"
# The names clients put in "model". The router gives each child its instance's name as --alias (its
# server-models.cpp overwrites any alias of the preset), so these are the served names of the lane, as
# flash-unsloth's --alias is its lane's: what the clients ask for, what the proxy lists, what the cockpit
# shows. They are the names the box's 2026-11-14 run gave its engines (--alias lfm3-1.7b, d1-2b,
# lfm3-audio, essai.log), kept here so a switch to the lane answers the same requests the study proved.
LFM3_NAME_CHAT="lfm3-1.7b"
LFM3_NAME_D1="d1-2b"
LFM3_NAME_AUDIO="lfm3-audio"
# llama.cpp's server image for arm64 with CUDA 13 (server-cuda13: b12316, commit 5d4c8eeb24db), by its
# digest: the same image the Qwen3.8 Flash-Next Unsloth GGUF lane pins (install-flash-unsloth.sh says where
# it was read). One image for the two llama.cpp lanes, as the two Nemotron lanes share one: nothing is built
# on the box.
LFM3_IMAGE="ghcr.io/ggml-org/llama.cpp@sha256:a5600a59c704208491adbb96505454a8d3d4cd3a9db48b03cf2760b660b7fd16"
# the five files at these revisions, 6.49 GB (10^9 bytes), rounded up: the room their fetch asks.
LFM3_GB="7"
# Its pull at its peak: the Unsloth GGUF lane's count of the same image (docker image inspect and docker
# manifest inspect on the reference box, 2026-11-12): 4.50 GB unpacked, 2.70 to download, 6.7 GiB with the
# containerd store's copy beside it; ceil(x 1.02) + 1 as df -BG rounds up. The image is the other lane's, so
# a box that holds it needs nothing here (vlane_room takes what the box has off).
IMAGE_GB=8
LOCAL_IMAGE="$LFM3_IMAGE"
# The router presets, rendered with the pinned files' container paths (the template is at column 0
# readable: its three sections name the three served models and hold the measured per-model flags).
MODELS_INI="$HERE/qwen38-lfm3-models.ini.template"
INI="$CONFIG_DIR/lfm3-models.ini"
# What the router's model list must not read: its cached-models scan takes the Hugging Face cache's own
# layout (common/hf-cache.cpp get_cache_directory, read in the pinned image's commit: LLAMA_CACHE first, then
# HOME/.cache/huggingface/hub), and the container mounts the box's real cache. Without one empty cache of its
# own, every GGUF the box holds, the Unsloth lane's 112 GB among them, would be a model the router lists and
# a client could ask it to load. LLAMA_CACHE set here wins the precedence, and the lane loads its three by
# absolute path, which needs no cache. Read in the image's own code the day the lane was written (2026-11-15).
ROUTER_CACHE=/tmp/qwen38-lfm3-router-cache

# The text lanes' cache and port: this lane is one of them for the proxy (vllm-lane.sh).
vlane_engine_env "${LFM3_PORT:-}"

# The files this lane fetches and its unit names, as a list of (repo, revision, PATH=SHA256:SIZE) triples.
lf_all_files(){
  printf '%s\n' "$LFM3_CHAT_REPO|$LFM3_CHAT_REV|$LFM3_CHAT_FILE" \
                "$LFM3_D1_REPO|$LFM3_D1_REV|$LFM3_D1_FILE" \
                "$LFM3_D1_REPO|$LFM3_D1_REV|$LFM3_D1_MMPROJ" \
                "$LFM3_AUDIO_REPO|$LFM3_AUDIO_REV|$LFM3_AUDIO_FILE" \
                "$LFM3_AUDIO_REPO|$LFM3_AUDIO_REV|$LFM3_AUDIO_MMPROJ"
}
# The path the engine reads one triple at inside the container: where the cache is mounted, in the same
# hub/models--.../snapshots/... layout the host keeps (the flash-unsloth unit says why both spellings).
lf_box(){ local r="${1%%|*}" rest="${1#*|}" file; file="${1##*|}"; file="${file%=*}"
  printf '/root/.cache/huggingface/hub/models--%s/snapshots/%s/%s\n' "${r//\//--}" "${rest%%|*}" "$file"; }

if [ "$ACTION" = uninstall ]; then
  step "Removing the Liquid AI LFM3 lane"
  vlane_uninstall_unit
  vlane_uninstall_clients
  # The image is pulled, not built: its local tag goes (lanes.sh), and the GGUF files stay, said.
  llane_uninstall_runtime "$LFM3_CHAT_REPO" "$LFM3_D1_REPO" "$LFM3_AUDIO_REPO"
  rm -f "$INI"
  exit 0
fi

# The router presets for the files in the cache: the template with each pinned path written in. Refuses
# (with the reason) when the template or a pin is not what this script says, so a unit never names a file
# the fetch did not verify.
render_ini(){  # $1 the file to write
  # the five pinned files, each read twice: the path the engine opens in the container, and the path the
  # start check opens on the host (the cache is mounted where the first spells the second)
  CHAT_BOX="$(lf_box "$LFM3_CHAT_REPO|$LFM3_CHAT_REV|$LFM3_CHAT_FILE")"
  CHAT_HOST="$(llane_path "$LFM3_CHAT_REPO" "$LFM3_CHAT_REV" "$LFM3_CHAT_FILE")"
  D1_BOX="$(lf_box "$LFM3_D1_REPO|$LFM3_D1_REV|$LFM3_D1_FILE")"
  D1_HOST="$(llane_path "$LFM3_D1_REPO" "$LFM3_D1_REV" "$LFM3_D1_FILE")"
  D1_MM_BOX="$(lf_box "$LFM3_D1_REPO|$LFM3_D1_REV|$LFM3_D1_MMPROJ")"
  D1_MM_HOST="$(llane_path "$LFM3_D1_REPO" "$LFM3_D1_REV" "$LFM3_D1_MMPROJ")"
  AUDIO_BOX="$(lf_box "$LFM3_AUDIO_REPO|$LFM3_AUDIO_REV|$LFM3_AUDIO_FILE")"
  AUDIO_HOST="$(llane_path "$LFM3_AUDIO_REPO" "$LFM3_AUDIO_REV" "$LFM3_AUDIO_FILE")"
  AUDIO_MM_BOX="$(lf_box "$LFM3_AUDIO_REPO|$LFM3_AUDIO_REV|$LFM3_AUDIO_MMPROJ")"
  AUDIO_MM_HOST="$(llane_path "$LFM3_AUDIO_REPO" "$LFM3_AUDIO_REV" "$LFM3_AUDIO_MMPROJ")"
  sed -e "s|__CHAT_MODEL__|$CHAT_BOX|g" -e "s|__D1_MODEL__|$D1_BOX|g" -e "s|__D1_MMPROJ__|$D1_MM_BOX|g" \
      -e "s|__AUDIO_MODEL__|$AUDIO_BOX|g" -e "s|__AUDIO_MMPROJ__|$AUDIO_MM_BOX|g" \
      "$MODELS_INI" > "$1"
  if grep -q '__[A-Z][A-Z0-9_]*__' "$1"; then die "the router presets still hold an unsubstituted placeholder"; fi
  # The five files, on the host, for the check the unit runs before it starts (llamacpp-files.sh).
  FILES_HOST="$CHAT_HOST $D1_HOST $D1_MM_HOST $AUDIO_HOST $AUDIO_MM_HOST"
}

# The unit, rendered for this box: written to $1. Its start checks the files the engine reads
# (llamacpp-files.sh): the five pinned GGUF and the presets the engine loads.
render_unit(){
  local files f
  render_ini /dev/null >/dev/null || die "the router presets could not be rendered"
  files=""
  for f in $FILES_HOST "$INI"; do files="$files $f"; done
  sed -e "s|__USER__|$(id -un)|g" -e "s|__GROUP__|$(id -gn "$(id -un)")|g" -e "s|__HOME__|$HOME|g" \
      -e "s|__IMAGE__|$LFM3_IMAGE|g" -e "s|__HF_CACHE__|$HF_CACHE|g" -e "s|__FILES__|${files# }|g" \
      -e "s|__INI_HOST__|$INI|g" -e "s|__INI_BOX__|/out/lfm3-models.ini|g" \
      -e "s|__ROUTER_CACHE__|$ROUTER_CACHE|g" \
      -e "s|__ENGINE_BIND__|$ENGINE_BIND|g" -e "s|__PORT__|$PORT|g" \
      "$HERE/$UNIT.template" > "$1"
  if grep -q '__[A-Z][A-Z0-9_]*__' "$1"; then die "the unit template still holds an unsubstituted placeholder"; fi
}

# The presets are the unit's input: written with the unit, at every install, every switch and every
# --stage (the file names the pinned paths, so a pin that moved renders it again, mtime included).
put_ini(){
  mkdir -p "$CONFIG_DIR"
  local r; r="$(mktemp)"
  render_ini "$r"
  cmp -s "$r" "$INI" || install -m 644 "$r" "$INI"
  rm -f "$r"
}

# Step 4, the one part that needs root: the unit and its start scripts. A function, since a lane
# installed on demand (--units-only) takes it straight after the preflight.
install_service(){
  step "4/5 Service"
  llane_service_files
  put_ini
  RENDER="$(mktemp)"; trap 'rm -f "$RENDER"' EXIT
  render_unit "$RENDER"
  # with what the engine reads at its start besides the unit, its router presets: a lane started before
  # they changed serves the old ones (vllm-lane.sh)
  vlane_put_unit "$RENDER" "$INI"
  # Installed, not switched to: the lane this box serves stays the lane it serves.
  echo "installed as a lane of its own. Switch to it like any lane:"
  echo "  the cockpit: Load on Liquid AI LFM3 in the Lanes view"
  echo "  a terminal : ./switch-model.sh lfm3, then the commands it prints"
}

# The room the files still missing need on the cache's disk, said before a byte is downloaded.
missing_gb(){
  local t need=0 blob size
  while read -r t; do
    [ -n "$t" ] || continue
    blob="$(llane_blob "${t%%|*}" "${t##*|}")"; size="${t##*:}"
    if [ ! -f "$blob" ] || [ "$(stat -c %s "$blob" 2>/dev/null || echo 0)" != "$size" ]; then
      need=$(( need + size ))
    fi
  done <<< "$(lf_all_files)"
  echo $(( (need + 999999999) / 1000000000 ))
}
fetch_files(){
  local t
  while read -r t; do
    [ -n "$t" ] || continue
    llane_fetch "${t%%|*}" "$(printf '%s\n' "$t" | cut -d'|' -f2)" "${t##*|}"
  done <<< "$(lf_all_files)"
}

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

# --stage: the unit for the files in the cache, for switch-model.sh, with no sudo: its router presets
# first, as the unit names them (llane_template's rule for the chat template, here for the presets file).
if [ "$MODE" = stage ]; then
  put_ini >&2
  [ -f "$CONFIG_DIR/llamacpp-files.sh" ] || install -m 755 "$HERE/llamacpp-files.sh" "$CONFIG_DIR/llamacpp-files.sh"
  render_unit "$STAGE_OUT"
  exit 0
fi

step "1/5 Preflight (Liquid AI LFM3: $LFM3_CHAT_REPO @ ${LFM3_CHAT_REV:0:12}, $LFM3_D1_REPO @ ${LFM3_D1_REV:0:12}, $LFM3_AUDIO_REPO @ ${LFM3_AUDIO_REV:0:12})"
vlane_preflight engine-preflight.sh llamacpp-files.sh gguf-fetch.py
# what is left to download of the pinned files, not what the repositories' folders hold (llamacpp-lane.sh)
WEIGHTS_GB="$(missing_gb)"
vlane_room "$LOCAL_IMAGE"
if [ "$MODE" = units ]; then
  echo "OK: $(uname -m), key present; the image and the GGUF files come with the lane's first Load"
  install_service
  step "Done: the Liquid AI LFM3 lane is installed; its first Load prepares it"
  echo "  the cockpit: Load on Liquid AI LFM3 in the Lanes view pulls its image (about 4.5 GB, 2.7 to"
  echo "               download, already there on a box that loaded the Qwen3.8 Flash-Next Unsloth GGUF"
  echo "               lane), fetches its GGUF files (about $LFM3_GB GB), then starts it"
  echo "  a terminal : ./install-lfm3.sh --prepare, then ./switch-model.sh lfm3 and the commands it prints"
  exit 0
fi

step "2/5 Image (${LFM3_IMAGE:0:40}...)"
if docker image inspect "$LFM3_IMAGE" >/dev/null 2>&1; then
  echo "already here"
else
  docker pull "$LFM3_IMAGE" \
    || die "docker pull of $LFM3_IMAGE failed. Causes: no internet, a GitHub Container Registry rate limit (retry in a few minutes), or the digest removed upstream."
fi
# its local tag, so that a docker image prune leaves it alone, and the tags of the images it was pinned to before
# go (lanes.sh)
pin_tag lfm3 "$LFM3_IMAGE"
unpin_stale lfm3 "$LFM3_IMAGE" || true

if [ "$MODE" = prepare ]; then
  step "Done: the Liquid AI LFM3 lane's image is ready; its Load fetches the GGUF files and starts it"
  exit 0
fi

step "3/5 GGUF files (the three pinned releases, five files, resumable) and the router presets"
fetch_files
put_ini

install_service

if [ "$SMOKE" -eq 0 ]; then step "Done (proof skipped)"; exit 0; fi

step "5/5 Proving it serves: chat and a tool call, a decision at the contract, speech made and heard back"
VLANE_SERVED=$LFM3_NAME_CHAT; VLANE_KWARGS='{}'
VLANE_PARSERS="llama.cpp's parser of the template's tool calls is"; VLANE_BOOT="about 1 min"; VLANE_WAIT_MIN=15
vlane_prove

# The two routes the chat proof does not reach, asked of the engine that just answered (vlane_prove left
# it up): each is what the lane's card says it serves, and the box measured them on 2026-11-14 (the
# bodies below are its essai-nouveaux.sh's, verbatim). The key is read the way vlane_ask reads it, and
# never put on a command line.
lf_ask(){  # $1 the route, $2 the body file or - (stdin), $3 the output file; prints the HTTP code
  local code
  code="$(curl -sS -o "$3" -w '%{http_code}' -m 300 "http://127.0.0.1:$PORT$1" \
    -H @<(printf 'Authorization: Bearer %s\n' "$KEY") -H 'Content-Type: application/json' --data-binary @"$2" || true)"
  printf '%s\n' "${code:-000}"
}
lf_body(){ printf '%s' "$1" > "$SMOKE_DIR/body.json"; lf_ask "$2" "$SMOKE_DIR/body.json" "$SMOKE_DIR/out.json"; }

SMOKE_DIR="$(mktemp -d)"; trap 'rm -rf "$SMOKE_DIR"' EXIT
# GET, not POST: the router lists its models and their states (a name it does not list is a lane that
# does not serve what its card says)
res="$(curl -sS -m 15 "http://127.0.0.1:$PORT/v1/models" -H @<(printf 'Authorization: Bearer %s\n' "$KEY") || true)"
printf '%s\n' "$res" | python3 - "$LFM3_NAME_CHAT" "$LFM3_NAME_D1" "$LFM3_NAME_AUDIO" <<'PY' || die "the router does not list its three models: the lane serves less than its card says."
import json, sys
want = set(sys.argv[1:4])
try:
    have = {m.get("id") for m in json.load(sys.stdin).get("data", [])}
except Exception as e:
    raise SystemExit(f"/v1/models answered something that is not its list: {e}")
missing = sorted(want - have)
if missing:
    raise SystemExit("the router is missing: " + ", ".join(missing))
print(f"   /v1/models lists the three: {', '.join(sorted(want))}")
PY
REQ='{"state": "Hi, I have been trying to connect my Stripe account for 3 days and it keeps failing. I am losing sales. Please help ASAP.", "model": "d1-2b", "questions": {"department": {"type": "choice", "instructions": "Which team should handle this", "criteria": {"billing": "Payment or subscription issues", "technical": "Bugs or integration problems", "sales": "Pricing or account questions"}}, "frustration": {"type": "score", "instructions": "How frustrated the customer appears", "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"]}, "is_urgent": {"type": "noul", "instructions": "The message conveys urgency or time-sensitivity"}}}'
code="$(lf_body "$REQ" /v1/systemone)"
[ "$code" = 200 ] || { head -c 300 "$SMOKE_DIR/out.json" 2>/dev/null; echo; die "the decision model did not answer /v1/systemone (HTTP ${code:-000})."; }
python3 - "$SMOKE_DIR/out.json" <<'PY' || die "the decision answer does not hold the contract the study measured."
import json, sys
a = json.load(open(sys.argv[1]))["answers"]
assert a["department"]["choice"] in ("billing", "technical", "sales"), a["department"]
assert a["frustration"]["type"] == "score" and 0.0 <= a["frustration"]["score"] <= 2.0, a["frustration"]
assert a["is_urgent"]["type"] == "noul" and 0.0 <= a["is_urgent"]["noul"] <= 1.0, a["is_urgent"]
print(f"   /v1/systemone at the contract: {a['department']['choice']}, frustration {a['frustration']['score']}, urgent {a['is_urgent']['noul']}")
PY
code="$(lf_body "{\"model\":\"$LFM3_NAME_AUDIO\",\"input\":\"Bonjour, ceci est un essai de synthèse vocale sur la DGX Spark.\",\"response_format\":\"wav\"}" /v1/audio/speech)"
[ "$code" = 200 ] || { head -c 300 "$SMOKE_DIR/out.json" 2>/dev/null; echo; die "the audio model did not answer /v1/audio/speech (HTTP ${code:-000})."; }
mv "$SMOKE_DIR/out.json" "$SMOKE_DIR/say.wav"
python3 - "$SMOKE_DIR/say.wav" <<'PY' || die "the speech answer is not a WAV file."
import sys
b = open(sys.argv[1], "rb").read(12)
if b[:4] != b"RIFF" or b[8:12] != b"WAVE":
    raise SystemExit(f"not a WAV: {b[:12]!r}")
print(f"   /v1/audio/speech: {len(open(sys.argv[1],'rb').read())} bytes of WAV")
PY
code="$(curl -sS -o "$SMOKE_DIR/heard.json" -w '%{http_code}' -m 300 "http://127.0.0.1:$PORT/v1/audio/transcriptions" \
  -H @<(printf 'Authorization: Bearer %s\n' "$KEY") -F "file=@$SMOKE_DIR/say.wav" -F "model=$LFM3_NAME_AUDIO" -F language=fr || true)"
[ "$code" = 200 ] || { head -c 300 "$SMOKE_DIR/heard.json" 2>/dev/null; echo; die "the audio model did not answer /v1/audio/transcriptions (HTTP ${code:-000})."; }
python3 - "$SMOKE_DIR/heard.json" <<'PY' || die "what the audio model heard back does not carry its own sentence: the route answered, not the model."
import json, re, sys, unicodedata
heard = json.load(open(sys.argv[1])).get("text") or ""
norm = lambda s: re.sub(r"\W+", " ", unicodedata.normalize("NFKD", s.lower()).encode("ascii", "ignore").decode()).split()
said = norm("bonjour ceci est un essai de synthese vocale sur la dgx spark")
got = set(norm(heard))
keep = [w for w in said if w in got]
print(f"   /v1/audio/transcriptions heard: {heard[:90]!r}")
if len(keep) < 5:
    raise SystemExit(f"only {len(keep)} of {len(said)} words came back")
PY

step "Done: the Liquid AI LFM3 lane is installed and proved it serves on $ENGINE_BIND:$PORT"
echo "  clients      : the proxy on :30001 with the box's key: chat asks $LFM3_NAME_CHAT, decisions ask"
echo "                 $LFM3_NAME_D1 (POST /v1/systemone), audio asks $LFM3_NAME_AUDIO (/v1/audio/*); the"
echo "                 audio and decision routes answered the engine directly on $ENGINE_BIND:$PORT here, and"
echo "                 the proxy's own relay of them is docs/lfm3-lane.md's account"
