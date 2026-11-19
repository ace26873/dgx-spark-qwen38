#!/usr/bin/env bash
# Install the Liquid AI LFM2.5 24B-A2B lane: Liquid AI's official FP8 of LFM2.5 24B-A2B (a hybrid mixture of
# experts, 24B with 2B active per token) and its official DSpark draft, served by SGLang on the text lanes' port
# behind the proxy and its key. Idempotent, and safe to run alone.
#
#   ./install-lfm25.sh                 first install, or update in place
#   ./install-lfm25.sh --no-smoke      skip the proof at the end
#   ./install-lfm25.sh --uninstall     remove the unit (the checkpoints and the image kept, said)
#   ./install-lfm25.sh --units-only    the unit alone: no image, no checkpoints (install.sh, on demand)
#   ./install-lfm25.sh --prepare       its image, pulled with no sudo: the cockpit's first Load
#                                      runs it (./prepare-lane.sh lfm25)
#   ./install-lfm25.sh --stage FILE    the unit for this box, written to FILE with no sudo:
#                                      switch-model.sh lfm25 installs FILE, once it has fetched the
#                                      checkpoint and its draft in the lane's image
#
# install.sh runs this on every plain install, like the other lanes; ./install.sh --no-lfm25 leaves it out of the
# run that passes it. It takes 25.4 GB of weights (the FP8 checkpoint and the draft) and the 27B lane's SGLang image
# (33.7 GB unpacked, already on a box that serves the 27B); when the disk does not have the room, this refuses
# before downloading anything.
#
# WHY SGLANG, AND THE 27B LANE'S IMAGE. Liquid AI's draft card gives SGLang's command for this pair, and the 27B
# lane's own image (v0.5.22, the same digest) served it on the reference box on 2026-11-14: ready in 286 s, 17 x 23
# answered right, GSM8K 250 at 0.912 (the 27B scores 0.99 there), a generation of 362 tokens/s in all at the median
# under 8 requests, the draft accepting 5.28 tokens at the median, a KV cache of 1,667,049 tokens and MemAvailable
# 24 GiB at its lowest (~/qa-tools/etudes/nouvelles-lanes/RESULTATS.md, essai-20261114-1436). One image for the
# SGLang lanes: nothing is built, nothing more is pulled on a box that has the 27B.
#
# ONE ENGINE AT A TIME. The unit says so with Conflicts=: starting it stops whichever lane serves, and starting
# another lane stops it.
set -euo pipefail
# One spelling of the home, without a trailing slash, as install.sh says why: the cockpit's
# exact-argv sudoers is rendered from it and has to match the cockpit's own calls (2026-10-22).
while [ -n "${HOME:-}" ] && [ "$HOME" != / ] && [ "${HOME%/}" != "$HOME" ]; do HOME="${HOME%/}"; done; export HOME
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT=qwen38-lfm25.service
# where systemd reads the units, the one variable lanes.sh and vllm-lane.sh read the other units by
INSTALLED="${LANES_UNIT_DIR:-/etc/systemd/system}/$UNIT"
CONTAINER=qwen38-lfm25
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
  # the cockpit first loads the lane. The checkpoints come with that Load's switch, as for any Load.
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
# A 64 GB DGX Spark cannot hold this lane (boxmem.py): its run on a 128 GB box left 24 GiB available at its lowest,
# and none was made on a 64 GB one. Refused there before anything is fetched; --uninstall still takes away what was put.
[ "$ACTION" = uninstall ] || [ "$(box_mem_class)" != 64 ] || lane_fits lfm25 || die "The Liquid AI LFM2.5 24B-A2B lane needs a 128 GB DGX Spark, and this box is a 64 GB one."
# A terminal opened before the account joined the docker group: the run goes on with it (lanes.sh).
docker_session bash "$HERE/${BASH_SOURCE[0]##*/}" "$@"
# what the SGLang lanes' installers share: the engine's cache and port, the uninstall, the service, the preflight
# and the room, the fetch, the proof (most of it vllm-lane.sh's)
# shellcheck source=sglang-lane.sh
. "$HERE/sglang-lane.sh"
VLANE=lfm25; VLANE_SHORT="Liquid AI LFM2.5 24B-A2B"

# The pins, at column 0 like install.sh's. switch-model.sh, check-pins.sh and the cockpit read them here.
# The 27B lane's SGLang image (install.sh's IMAGE: v0.5.22, by its digest), the one the reference box ran this pair on.
LFM25_IMAGE="lmsysorg/sglang@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae"
# Liquid AI's official FP8 (compressed-tensors) and its official DSpark draft, at the revisions measured on
# 2026-11-14.
LFM25_REPO="LiquidAI/LFM2.5-24B-A2B-FP8"
LFM25_REV="f54bf39547fd5b7955abb9fe8d1b151e31fbbddb"
LFM25_DRAFT_REPO="LiquidAI/LFM2.5-24B-A2B-DSpark"
LFM25_DRAFT_REV="e052e5b1dd103c38cdd4f8ddc068d7dea4586f55"
# The two repositories at these revisions, 24,766,905,221 and 656,291,264 bytes (the Hub's API, its files' sizes,
# read 2026-11-16), 25.42 GB (10^9 bytes) together, rounded up: the room their fetch asks.
LFM25_GB="26"
# The parser that separates the reasoning: the checkpoint's template opens <think> itself at every answer, so the
# model writes "... </think>" with no opening tag. deepseek-r1's parser takes every answer as begun in a thought,
# whatever a request's enable_thinking says (DeepSeekR1Detector, force_reasoning=True, read in this image's source,
# 2026-11-16), as the template does. One line, so that the box's proof with the real clients can settle it.
LFM25_REASONING_PARSER="deepseek-r1"
LOCAL_IMAGE="$LFM25_IMAGE"
MODEL="$LFM25_REPO"; MODEL_REV="$LFM25_REV"; WEIGHTS_GB="$LFM25_GB"
# The image's pull at its peak, as install.sh counts the 27B lane's (33.7 GB unpacked, 14.8 GB to download, 45.1
# GiB with the containerd store's copy, measured 2026-10-15; ceil(x 1.02) + 1 as df -BG rounds up). A box that serves
# the 27B has it, and needs nothing here (vlane_room takes what the box has off).
IMAGE_GB=48

# The text lanes' cache and port: this lane is one of them for the proxy (vllm-lane.sh).
vlane_engine_env "${LFM25_PORT:-}"

if [ "$ACTION" = uninstall ]; then
  step "Removing the Liquid AI LFM2.5 24B-A2B lane"
  vlane_uninstall_unit
  vlane_uninstall_clients
  slane_uninstall_runtime "$LFM25_REPO" "$LFM25_DRAFT_REPO"
  exit 0
fi

# The unit, rendered for this box: written to $1.
render_unit(){
  sed -e "s|__USER__|$(id -un)|g" -e "s|__GROUP__|$(id -gn "$(id -un)")|g" -e "s|__HOME__|$HOME|g" \
      -e "s|__IMAGE__|$LFM25_IMAGE|g" -e "s|__HF_CACHE__|$HF_CACHE|g" \
      -e "s|__MODEL__|$LFM25_REPO|g" -e "s|__REV__|$LFM25_REV|g" \
      -e "s|__DRAFT__|$LFM25_DRAFT_REPO|g" -e "s|__DRAFT_REV__|$LFM25_DRAFT_REV|g" \
      -e "s|__REASONING_PARSER__|$LFM25_REASONING_PARSER|g" \
      -e "s|__ENGINE_BIND__|$ENGINE_BIND|g" -e "s|__PORT__|$PORT|g" \
      "$HERE/$UNIT.template" > "$1"
  if grep -q '__[A-Z][A-Z0-9_]*__' "$1"; then die "the unit template still holds an unsubstituted placeholder: $(grep -o '__[A-Z][A-Z0-9_]*__' "$1" | sort -u | tr '\n' ' ')"; fi
}

# Step 4, the one part that needs root: the unit and its start scripts. A function, since a lane installed on
# demand (--units-only) takes it straight after the preflight.
install_service(){
  step "4/5 Service"
  # SGLang's compile cache
  slane_service_files "$CONFIG_DIR/$VLANE-cache"
  RENDER="$(mktemp)"; trap 'rm -f "$RENDER"' EXIT
  render_unit "$RENDER"
  vlane_put_unit "$RENDER"
  # Installed, not switched to: the lane this box serves stays the lane it serves.
  echo "installed as a lane of its own. Switch to it like any lane:"
  echo "  the cockpit: Load on Liquid AI LFM2.5 24B-A2B in the Lanes view"
  echo "  a terminal : ./switch-model.sh lfm25, then the commands it prints"
}

# The checkpoint and its draft, in the lane's image (vllm-lane.sh's fetch, ckpt-fetch.py). Both pinned
# as refs/main: the unit runs with HF_HUB_OFFLINE=1 and SGLang's speculative hook reads the draft's
# config by its name, through refs/main, which a download by commit never writes.
fetch_files(){
  vlane_fetch "$LFM25_REPO" "$LFM25_REV" pin
  vlane_fetch "$LFM25_DRAFT_REPO" "$LFM25_DRAFT_REV" pin
}

# --stage: the unit for this box, for switch-model.sh, with no sudo, its cache folder made first (docker would make
# it as root).
if [ "$MODE" = stage ]; then
  mkdir -p "$CONFIG_DIR/$VLANE-cache"
  render_unit "$STAGE_OUT"
  exit 0
fi

step "1/5 Preflight (Liquid AI LFM2.5 24B-A2B: $LFM25_REPO @ ${LFM25_REV:0:12}, $LFM25_DRAFT_REPO @ ${LFM25_DRAFT_REV:0:12})"
vlane_preflight engine-preflight.sh engine-secrets.sh ckpt-fetch.py
vlane_room "$LFM25_IMAGE" "$LFM25_REPO" "$LFM25_DRAFT_REPO"
if [ "$MODE" = units ]; then
  echo "OK: $(uname -m), key present; the image and the checkpoints come with the lane's first Load"
  install_service
  step "Done: the Liquid AI LFM2.5 24B-A2B lane is installed; its first Load prepares it"
  echo "  the cockpit: Load on Liquid AI LFM2.5 24B-A2B in the Lanes view pulls its image (the 27B lane's, already"
  echo "               there on a box that serves it), fetches its checkpoints (about $LFM25_GB GB), then starts it"
  echo "  a terminal : ./install-lfm25.sh --prepare, then ./switch-model.sh lfm25 and the commands it prints"
  exit 0
fi

step "2/5 Image (${LFM25_IMAGE:0:40}...)"
if docker image inspect "$LFM25_IMAGE" >/dev/null 2>&1; then
  echo "already here"
else
  docker pull "$LFM25_IMAGE" \
    || die "docker pull of $LFM25_IMAGE failed. Causes: no internet, a Docker Hub rate limit (retry in a few minutes, or docker login), or the digest removed upstream."
fi
# its local tag, so that a docker image prune leaves it alone, and the tags of the images it was pinned to before
# go (lanes.sh)
pin_tag lfm25 "$LFM25_IMAGE"
unpin_stale lfm25 "$LFM25_IMAGE" || true

if [ "$MODE" = prepare ]; then
  step "Done: the Liquid AI LFM2.5 24B-A2B lane's image is ready; its Load fetches the checkpoints and starts it"
  exit 0
fi

step "3/5 Checkpoints ($LFM25_REPO @ ${LFM25_REV:0:12} and its draft, resumable)"
fetch_files

install_service

if [ "$SMOKE" -eq 0 ]; then step "Done (proof skipped)"; exit 0; fi

step "5/5 Proving it serves (a reply and a tool call, then the box goes back to the lane it was serving)"
# Its template opens <think> at every answer and has no switch to turn the reasoning off: the reply is given room for
# its reasoning before it (2,048 tokens; 75 for "17 x 23" with its reasoning on the reference box).
VLANE_SERVED=lfm2.5-24b-a2b; VLANE_KWARGS='{}'; VLANE_REPLY_TOKENS=2048
VLANE_PARSERS="the lfm2 tool-call parser is"; VLANE_BOOT="about 5 min"; VLANE_WAIT_MIN=20
vlane_prove

step "Done: the Liquid AI LFM2.5 24B-A2B lane is installed and proved it serves on $ENGINE_BIND:$PORT"
echo "  switch to it : Load on Liquid AI LFM2.5 24B-A2B in the cockpit, or ./switch-model.sh lfm25"
echo "  clients      : the proxy on :30001 with the box's key, model lfm2.5-24b-a2b (OpenAI and Anthropic APIs)"
