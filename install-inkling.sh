#!/usr/bin/env bash
# Install the Inkling 2 Mini lane: Thinking Machines' Inkling 2 Mini (a looped mixture of experts, 26.4B with 3.4B
# active per token, its middle layers run three times; text and images in, text out; a window of 1M tokens), with the
# DSpark draft head its checkpoint carries, served by SGLang on the text lanes' port behind the proxy and its key.
# Idempotent, and safe to run alone.
#
#   ./install-inkling.sh                 first install, or update in place
#   ./install-inkling.sh --no-smoke      skip the proof at the end
#   ./install-inkling.sh --uninstall     remove the unit (the checkpoint and the image kept, said)
#   ./install-inkling.sh --units-only    the unit alone: no image, no checkpoint (install.sh, on demand)
#   ./install-inkling.sh --prepare       its image, pulled with no sudo: the cockpit's first Load
#                                        runs it (./prepare-lane.sh inkling)
#   ./install-inkling.sh --stage FILE    the unit for this box, written to FILE with no sudo:
#                                        switch-model.sh inkling installs FILE, once it has fetched the
#                                        checkpoint in the lane's image
#
# install.sh runs this on every plain install, like the other lanes; ./install.sh --no-inkling leaves it out of the
# run that passes it. It takes 20.2 GB of weights (one checkpoint, the draft head inside it) and an SGLang image of its
# own (38.6 GB unpacked); when the disk does not have the room, this refuses before downloading anything.
#
# WHY SGLANG, AND THIS IMAGE. Inkling 2 Mini is in no SGLang release yet: SGLang's cookbook serves it from the
# lmsysorg/sglang:inkling-2-mini image (pull request sgl-project/sglang#51926), pinned here by the digest of its arm64
# build, with the cookbook's DGX Spark recipe and its DSpark draft, at the 27B lane's memory share (0.76, where the recipe
# gives 0.45), so that its KV cache holds the model's whole window of 1,048,576 tokens. On the reference box
# (~/qa-tools/etudes/inkling), at 0.76 on 2026-11-24: ready in 237 s, a KV cache of 1,091,200 tokens, MemAvailable 24.37
# GiB at its lowest while it loaded. At the recipe's 0.45 on 2026-11-22 to 24: a KV cache of 490,496 tokens, GSM8K 250
# at 0.964 (its default effort, temperature 0) and 0.980 (its default effort, temperature 1.0), HumanEval 0.976, three
# tool calls of three, and a median of 64.7 tokens/s in bench.sh against 72.3 for the 27B lane.
#
# ONE ENGINE AT A TIME. The unit says so with Conflicts=: starting it stops whichever lane serves, and starting
# another lane stops it.
set -euo pipefail
# One spelling of the home, without a trailing slash, as install.sh says why: the cockpit's
# exact-argv sudoers is rendered from it and has to match the cockpit's own calls (2026-10-22).
while [ -n "${HOME:-}" ] && [ "$HOME" != / ] && [ "${HOME%/}" != "$HOME" ]; do HOME="${HOME%/}"; done; export HOME
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT=qwen38-inkling.service
# where systemd reads the units, the one variable lanes.sh and vllm-lane.sh read the other units by
INSTALLED="${LANES_UNIT_DIR:-/etc/systemd/system}/$UNIT"
CONTAINER=qwen38-inkling
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
  # The two halves of a lane installed on demand, as for the other lanes: --units-only puts in what needs root (the
  # unit, its start scripts) and nothing heavy, and --prepare pulls the lane's image as the user, with no sudo, when
  # the cockpit first loads the lane. The checkpoint comes with that Load's switch, as for any Load.
  --units-only) [ "$MODE" = full ] || die "--units-only and --prepare are the two halves of an install: one at a time"; MODE=units ;;
  --prepare) [ "$MODE" = full ] || die "--units-only and --prepare are the two halves of an install: one at a time"; MODE=prepare ;;
  --stage) [ "$MODE" = full ] || die "--stage is a mode of its own"; MODE=stage ;;
  -h|--help) awk 'NR > 1 && !/^#/ { exit } NR > 1' "$0"; exit 0 ;;
  *) if [ "$MODE" = stage ] && [ -z "${STAGE_OUT:-}" ]; then STAGE_OUT="$a"; else die "unknown option: $a"; fi ;;
esac; done
if [ "$MODE" = stage ] && [ -z "${STAGE_OUT:-}" ]; then die "--stage takes the file to write the unit to"; fi

CONFIG_DIR="$HOME/.config/qwen38"
# the text lane the box comes back to, one record shared with switch-model.sh (lanes.sh)
# shellcheck source=lanes.sh
. "$HERE/lanes.sh"
# A 64 GB DGX Spark cannot hold this lane as measured (boxmem.py): 84.4 GiB of GPU memory once ready on a 128 GB box,
# and no 64 GB run was made. Refused there before anything is fetched; --uninstall still takes away what was put.
[ "$ACTION" = uninstall ] || [ "$(box_mem_class)" != 64 ] || lane_fits inkling || die "The Inkling 2 Mini lane needs a 128 GB DGX Spark, and this box is a 64 GB one."
# A terminal opened before the account joined the docker group: the run goes on with it (lanes.sh).
docker_session bash "$HERE/${BASH_SOURCE[0]##*/}" "$@"
# what the SGLang lanes' installers share: the engine's cache and port, the uninstall, the service, the preflight
# and the room, the fetch, the proof (most of it vllm-lane.sh's)
# shellcheck source=sglang-lane.sh
. "$HERE/sglang-lane.sh"
VLANE=inkling; VLANE_SHORT="Inkling 2 Mini"

# The pins, at column 0 like install.sh's. switch-model.sh, check-pins.sh and the cockpit read them here.
# The image SGLang's cookbook serves Inkling 2 Mini with (lmsysorg/sglang:inkling-2-mini), by the digest of its arm64
# build, the one the reference box ran it on (2026-11-22).
INKLING_IMAGE="lmsysorg/sglang@sha256:bc811ac7ac68c6fca116a49750baca817b8613a30abface8a347682b55d132e3"
# Thinking Machines' checkpoint (Apache-2.0), at the revision measured on 2026-11-22: mixed NVFP4 and MXFP8 experts,
# the DSpark draft head beside them (mtp.safetensors), which SGLang takes from the checkpoint itself.
INKLING_REPO="thinkingmachines/Inkling-2-Mini"
INKLING_REV="b85c3cfa615fc413cc9af3379f491b963f3b586a"
# The repository at this revision, 20,191,643,333 bytes (the Hub's API, its files' sizes, read 2026-11-24), 20.19 GB
# (10^9 bytes), rounded up: the room its fetch asks.
INKLING_GB="21"
LOCAL_IMAGE="$INKLING_IMAGE"
MODEL="$INKLING_REPO"; MODEL_REV="$INKLING_REV"; WEIGHTS_GB="$INKLING_GB"
# Its pull at its peak: 38.6 GB unpacked (docker image inspect on the reference box) and 16.4 GB to download (its arm64
# build on Docker Hub, read 2026-11-22), which the containerd image store (Docker 29's default on a new install) keeps
# beside it, 51.3 GiB; ceil(x 1.02) + 1 as df -BG rounds up (install-lfm3.sh counts its image so).
IMAGE_GB=54

# The text lanes' cache and port: this lane is one of them for the proxy (vllm-lane.sh).
vlane_engine_env "${INKLING_PORT:-}"

if [ "$ACTION" = uninstall ]; then
  step "Removing the Inkling 2 Mini lane"
  vlane_uninstall_unit
  vlane_uninstall_clients
  slane_uninstall_runtime "$INKLING_REPO"
  exit 0
fi

# The unit, rendered for this box: written to $1.
render_unit(){
  sed -e "s|__USER__|$(id -un)|g" -e "s|__GROUP__|$(id -gn "$(id -un)")|g" -e "s|__HOME__|$HOME|g" \
      -e "s|__IMAGE__|$INKLING_IMAGE|g" -e "s|__HF_CACHE__|$HF_CACHE|g" \
      -e "s|__MODEL__|$INKLING_REPO|g" -e "s|__REV__|$INKLING_REV|g" \
      -e "s|__ENGINE_BIND__|$ENGINE_BIND|g" -e "s|__PORT__|$PORT|g" \
      "$HERE/$UNIT.template" > "$1"
  if grep -q '__[A-Z][A-Z0-9_]*__' "$1"; then die "the unit template still holds an unsubstituted placeholder: $(grep -o '__[A-Z][A-Z0-9_]*__' "$1" | sort -u | tr '\n' ' ')"; fi
}

# The lane's caches, made as this user so that docker does not make them as root when it mounts them: torch's compile
# cache, and the two kernel caches the engine fills at its first start (SGLang's and b12x's), which the reference box's
# runs kept from one start to the next (379 s to the first answer at the first start there, 222 s at the next).
INKLING_CACHES=("$CONFIG_DIR/$VLANE-cache" "$CONFIG_DIR/$VLANE-cache/sglang" "$CONFIG_DIR/$VLANE-cache/b12x")

# Step 4, the one part that needs root: the unit and its start scripts. A function, since a lane installed on
# demand (--units-only) takes it straight after the preflight.
install_service(){
  step "4/5 Service"
  slane_service_files "${INKLING_CACHES[@]}"
  RENDER="$(mktemp)"; trap 'rm -f "$RENDER"' EXIT
  render_unit "$RENDER"
  vlane_put_unit "$RENDER"
  # Installed, not switched to: the lane this box serves stays the lane it serves.
  echo "installed as a lane of its own. Switch to it like any lane:"
  echo "  the cockpit: Load on Inkling 2 Mini in the Lanes view"
  echo "  a terminal : ./switch-model.sh inkling, then the commands it prints"
}

# The checkpoint, in the lane's image (vllm-lane.sh's fetch, ckpt-fetch.py). Pinned as refs/main: the unit runs with
# HF_HUB_OFFLINE=1, and SGLang takes the DSpark draft from the checkpoint by its name ("defaulting
# --speculative-draft-model-path to --model-path", the engine's journal on the reference box), with no revision,
# through refs/main, which a download by commit never writes.
fetch_files(){
  vlane_fetch "$INKLING_REPO" "$INKLING_REV" pin
}

# --stage: the unit for this box, for switch-model.sh, with no sudo, its cache folders made first (docker would make
# them as root).
if [ "$MODE" = stage ]; then
  mkdir -p "${INKLING_CACHES[@]}"
  render_unit "$STAGE_OUT"
  exit 0
fi

step "1/5 Preflight (Inkling 2 Mini: $INKLING_REPO @ ${INKLING_REV:0:12})"
vlane_preflight engine-preflight.sh engine-secrets.sh ckpt-fetch.py
vlane_room "$INKLING_IMAGE" "$INKLING_REPO"
if [ "$MODE" = units ]; then
  echo "OK: $(uname -m), key present; the image and the checkpoint come with the lane's first Load"
  install_service
  step "Done: the Inkling 2 Mini lane is installed; its first Load prepares it"
  echo "  the cockpit: Load on Inkling 2 Mini in the Lanes view pulls its image (16.4 GB to download), fetches its"
  echo "               checkpoint (about $INKLING_GB GB), then starts it"
  echo "  a terminal : ./install-inkling.sh --prepare, then ./switch-model.sh inkling and the commands it prints"
  exit 0
fi

step "2/5 Image (${INKLING_IMAGE:0:40}...)"
if docker image inspect "$INKLING_IMAGE" >/dev/null 2>&1; then
  echo "already here"
else
  docker pull "$INKLING_IMAGE" \
    || die "docker pull of $INKLING_IMAGE failed. Causes: no internet, a Docker Hub rate limit (retry in a few minutes, or docker login), or the digest removed upstream."
fi
# its local tag, so that a docker image prune leaves it alone, and the tags of the images it was pinned to before
# go (lanes.sh)
pin_tag inkling "$INKLING_IMAGE"
unpin_stale inkling "$INKLING_IMAGE" || true

if [ "$MODE" = prepare ]; then
  step "Done: the Inkling 2 Mini lane's image is ready; its Load fetches the checkpoint and starts it"
  exit 0
fi

step "3/5 Checkpoint ($INKLING_REPO @ ${INKLING_REV:0:12}, resumable)"
fetch_files

install_service

if [ "$SMOKE" -eq 0 ]; then step "Done (proof skipped)"; exit 0; fi

step "5/5 Proving it serves (a reply and a tool call, then the box goes back to the lane it was serving)"
# Its effort at the template's lowest level (reasoning_effort 0.0, "none"), which its card says does not turn the
# reasoning off: the reply is given room for that reasoning before it (2,048 tokens).
VLANE_SERVED=inkling-2-mini; VLANE_KWARGS='{"reasoning_effort":0.0}'; VLANE_REPLY_TOKENS=2048
VLANE_PARSERS="the inkling2 tool-call parser is"; VLANE_BOOT="about 4 min (6 at its first start)"; VLANE_WAIT_MIN=20
vlane_prove

step "Done: the Inkling 2 Mini lane is installed and proved it serves on $ENGINE_BIND:$PORT"
echo "  switch to it : Load on Inkling 2 Mini in the cockpit, or ./switch-model.sh inkling"
echo "  clients      : the proxy on :30001 with the box's key, model inkling-2-mini (OpenAI and Anthropic APIs)"
