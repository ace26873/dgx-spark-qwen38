#!/usr/bin/env bash
# Install the Liquid AI LFM3 lane: Liquid AI's LFM3 1.7B (a hybrid of short convolutions and attention, post-trained for
# agent harnesses) in its official NVFP4 build, served by vLLM's own nightly image on the text lanes' port behind the
# proxy and its key. Idempotent, and safe to run alone.
#
#   ./install-lfm3.sh                 first install, or update in place
#   ./install-lfm3.sh --no-smoke      skip the proof at the end
#   ./install-lfm3.sh --uninstall     remove the unit (the checkpoint and the image kept, said)
#   ./install-lfm3.sh --units-only    the unit alone: no image, no checkpoint (install.sh, on demand)
#   ./install-lfm3.sh --prepare       its image, pulled with no sudo: the cockpit's first Load
#                                     runs it (./prepare-lane.sh lfm3)
#   ./install-lfm3.sh --stage FILE    the unit for the checkpoint in the cache, written to FILE
#                                     with no sudo: switch-model.sh lfm3 installs FILE
#
# install.sh puts this lane's unit in on every plain install, like the other lanes, and its first Load from the
# cockpit pulls its image and fetches its checkpoint; ./install.sh --no-lfm3 leaves it out of the run that passes it.
# It takes a checkpoint of 1.83 GB and vLLM's nightly image, 28 GB (11.5 to download): when the disk does not have the
# room, this refuses before downloading anything.
#
# WHY THIS IMAGE. LFM3 came to vLLM on 2026-11-06 (PR 65826), in no release yet: the nightly image of commit 60bd6dd
# (vLLM 0.34.0rc1.dev571, CUDA 13.2, arm64) serves it, and started on the box's driver (CUDA 13.0) on 2026-11-16, its
# NVFP4 read by FlashInfer. As the reference box measured it then, before LFM3_GMU below was set
# (~/qa-tools/etudes/lfm3-vllm/NOTES.md): GSM8K 250 at 0.896 (llama.cpp's Q8_0 of the same model: 0.844), 87.7
# tokens/s alone and 3,912 in all at 128 requests, the whole window of 262,144 tokens, an agent's next turn served 99
# to 100 % from the prefix cache. Pulled as published, by digest: nothing is built on the box.
#
# ONE ENGINE AT A TIME. The unit says so with Conflicts=: starting it stops whichever lane serves, and starting another
# lane stops it.
set -euo pipefail
# One spelling of the home, without a trailing slash, as install.sh says why: the cockpit's
# exact-argv sudoers is rendered from it and has to match the cockpit's own calls (2026-10-22).
while [ -n "${HOME:-}" ] && [ "$HOME" != / ] && [ "${HOME%/}" != "$HOME" ]; do HOME="${HOME%/}"; done; export HOME
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT=qwen38-lfm3.service
# where systemd reads the units, the one variable lanes.sh and vllm-lane.sh read the other units by
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
  # lane's image as the user, with no sudo, when the cockpit first loads the lane. The checkpoint
  # comes with that Load's switch, as for any Load.
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
# A 64 GB DGX Spark cannot hold this lane (boxmem.py): its share, 0.75 of the memory, is a 128 GB box's, and none was
# measured on a 64 GB one. Refused there before anything is fetched; --uninstall still takes away what was put.
[ "$ACTION" = uninstall ] || [ "$(box_mem_class)" != 64 ] || lane_fits lfm3 || die "The Liquid AI LFM3 lane needs a 128 GB DGX Spark, and this box is a 64 GB one."
# A terminal opened before the account joined the docker group: the run goes on with it (lanes.sh).
docker_session bash "$HERE/${BASH_SOURCE[0]##*/}" "$@"
# what the vLLM lanes' installers share: the engine's cache and port, the uninstall, the service,
# the preflight and the room, the fetch, the proof
# shellcheck source=vllm-lane.sh
. "$HERE/vllm-lane.sh"
VLANE=lfm3; VLANE_SHORT="Liquid AI LFM3"

# The pins, at column 0 like install.sh's. switch-model.sh, check-pins.sh and the cockpit read them here.
# vLLM's own nightly image of commit 60bd6dd2edd175a1c55fd3488f871e4c3d2ef27d (vLLM 0.34.0rc1.dev571+g60bd6dd2e, read
# in the image), arm64, by its digest: the image the reference box measured on 2026-11-16.
LFM3_IMAGE="vllm/vllm-openai@sha256:19cfbac06cfed28a2eb2c08950e62b4b07fd6e20685ad6e10f9422236d39e519"
# Liquid AI's official NVFP4 (ModelOpt, weights and activations by groups of 16), at the revision measured then.
LFM3_REPO="LiquidAI/LFM3-1.7B-NVFP4"
LFM3_REV="cfe4b5eabe33e828256f760f72c0af600ceff0ad"
# The repository at this revision, 1,847,737,703 bytes (its 13 files' sizes on the Hub's API, read 2026-11-16), 1.85 GB
# (10^9 bytes), rounded up: the room its fetch asks.
LFM3_GB="2"
# The engine's share of the memory it sees. This nightly sizes its KV cache after its warmup and extends it into every
# byte left free (v1/worker/extensible_kv_cache.py, which says to set a share "on a unified-memory device"): with none,
# the box kept 1,087 MiB available at its lowest under load (2026-11-16), its cache then 5,239,455 tokens, the window
# twenty times over. 0.75 caps the engine at about 91 GiB on a 128 GB box. One line: a box that measures another share
# sets it here.
LFM3_GMU="0.75"
LOCAL_IMAGE="$LFM3_IMAGE"
MODEL="$LFM3_REPO"; MODEL_REV="$LFM3_REV"; WEIGHTS_GB="$LFM3_GB"; GMU="$LFM3_GMU"
# Its pull at its peak: 27.8 GB unpacked and 11.5 GB to download (its 42 layers on the registry's manifest, read
# 2026-11-16), which the containerd image store (Docker 29's default on a new install) keeps beside it, 36.6 GiB;
# ceil(x 1.02) + 1 as df -BG rounds up (install-flash-local.sh counts its image so).
IMAGE_GB=39

# The text lanes' cache and port: this lane is one of them for the proxy (vllm-lane.sh).
vlane_engine_env "${LFM3_PORT:-}"

if [ "$ACTION" = uninstall ]; then
  step "Removing the Liquid AI LFM3 lane"
  vlane_uninstall_unit
  vlane_uninstall_clients
  # The image is pulled, not built: no build of it to prune (qwen38-lfm3 names none), and its local tag goes (lanes.sh)
  vlane_uninstall_runtime qwen38-lfm3 "$LFM3_IMAGE" "$LFM3_REPO"
  pin_untag lfm3
  exit 0
fi

# The unit, rendered for this box: written to $1.
render_unit(){
  sed -e "s|__USER__|$(id -un)|g" -e "s|__GROUP__|$(id -gn "$(id -un)")|g" -e "s|__HOME__|$HOME|g" \
      -e "s|__IMAGE__|$LOCAL_IMAGE|g" -e "s|__MODEL__|$MODEL|g" -e "s|__REV__|$MODEL_REV|g" \
      -e "s|__GMU__|$GMU|g" \
      -e "s|__HF_CACHE__|$HF_CACHE|g" -e "s|__ENGINE_BIND__|$ENGINE_BIND|g" -e "s|__PORT__|$PORT|g" \
      "$HERE/$UNIT.template" > "$1"
  if grep -q '__[A-Z][A-Z0-9_]*__' "$1"; then die "the unit template still holds an unsubstituted placeholder: $(grep -o '__[A-Z][A-Z0-9_]*__' "$1" | sort -u | tr '\n' ' ')"; fi
}

# This lane's own folders: vLLM's compile cache, FlashInfer's and Triton's kernels, built at the first boot.
CACHES=("$CONFIG_DIR/lfm3-cache/vllm" "$CONFIG_DIR/lfm3-cache/flashinfer" "$CONFIG_DIR/lfm3-cache/triton")

# Step 4, the one part that needs root: the unit and its start scripts. A function, since a lane
# installed on demand (--units-only) takes it straight after the preflight.
install_service(){
  step "4/5 Service"
  vlane_service_files "${CACHES[@]}"
  # the key's yaml the unit mounts, written now: --mount refuses a source that is not there, and the unit writes it
  # again before every start
  bash "$CONFIG_DIR/engine-secrets.sh" "$CONFIG_DIR" \
    || die "the key's yaml could not be written from $CONFIG_DIR/api-key (the line above says why)"
  RENDER="$(mktemp)"; trap 'rm -f "$RENDER"' EXIT
  render_unit "$RENDER"
  vlane_put_unit "$RENDER"
  # Installed, not switched to: the lane this box serves stays the lane it serves.
  echo "installed as a lane of its own. Switch to it like any lane:"
  echo "  the cockpit: Load on Liquid AI LFM3 in the Lanes view"
  echo "  a terminal : ./switch-model.sh lfm3, then the commands it prints"
}

# --stage: the unit for the checkpoint in the cache, for switch-model.sh, with no sudo. This lane's own files only, as
# the other lanes' switches touch nothing but their unit: the start scripts every lane runs (vlane_service_files) are
# ./install.sh's to put in. Its caches made as the user, so that docker does not make them as root.
if [ "$MODE" = stage ]; then
  mkdir -p "${CACHES[@]}"
  render_unit "$STAGE_OUT"
  exit 0
fi

step "1/5 Preflight (Liquid AI LFM3: $MODEL @ ${MODEL_REV:0:12})"
vlane_preflight engine-preflight.sh engine-secrets.sh vllm_engine_auth.py ckpt-fetch.py
vlane_room "$LFM3_IMAGE" "$MODEL"
if [ "$MODE" = units ]; then
  echo "OK: $(uname -m), key present; the image and the checkpoint come with the lane's first Load"
  install_service
  step "Done: the Liquid AI LFM3 lane is installed; its first Load prepares it"
  echo "  the cockpit: Load on Liquid AI LFM3 in the Lanes view pulls its image (about 28 GB, 11.5 to download),"
  echo "               fetches its checkpoint (about $WEIGHTS_GB GB), then starts it"
  echo "  a terminal : ./install-lfm3.sh --prepare, then ./switch-model.sh lfm3 and the commands it prints"
  exit 0
fi

step "2/5 Image (${LFM3_IMAGE:0:40}...)"
if docker image inspect "$LFM3_IMAGE" >/dev/null 2>&1; then
  echo "already here"
else
  docker pull "$LFM3_IMAGE" \
    || die "docker pull of $LFM3_IMAGE failed. Causes: no internet, a Docker Hub rate limit (retry in a few minutes, or docker login), or the digest removed upstream."
fi
# its local tag, so that a docker image prune leaves it alone while this lane is installed, and the tags of the images
# it was pinned to before go (lanes.sh)
pin_tag lfm3 "$LFM3_IMAGE"
unpin_stale lfm3 "$LFM3_IMAGE" || true

if [ "$MODE" = prepare ]; then
  step "Done: the Liquid AI LFM3 lane's image is ready; its Load fetches the checkpoint and starts it"
  exit 0
fi

step "3/5 Checkpoint ($MODEL @ ${MODEL_REV:0:12}, resumable)"
# Pinned as refs/main, as the LFM2.5 lane's fetches are: the unit runs with HF_HUB_OFFLINE=1, and a read of the repo
# by its name then finds the pinned commit. None is known in this image (vLLM reads the tokenizer at --revision).
vlane_fetch "$MODEL" "$MODEL_REV" pin

install_service

if [ "$SMOKE" -eq 0 ]; then step "Done (proof skipped)"; exit 0; fi

step "5/5 Proving it serves (a reply and a tool call, then the box goes back to the lane it was serving)"
VLANE_SERVED=lfm3-1.7b; VLANE_KWARGS='{"enable_thinking":false}'
VLANE_PARSERS="the qwen3_coder parser is"; VLANE_BOOT="2 min 40 s at its first boot"; VLANE_WAIT_MIN=20
vlane_prove

step "Done: the Liquid AI LFM3 lane is installed and proved it serves on $ENGINE_BIND:$PORT"
echo "  switch to it : Load on Liquid AI LFM3 in the cockpit, or ./switch-model.sh lfm3"
echo "  clients      : the proxy on :30001 with the box's key, model lfm3-1.7b (OpenAI and Anthropic APIs)"
