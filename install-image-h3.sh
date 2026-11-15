#!/usr/bin/env bash
# Install the MiniMax-H3-Image lane: text-to-image and image editing up to about 16 megapixels,
# served by SGLang Diffusion on this box, beside the Qwen-Image 2.1 lane. Idempotent, and safe
# to run alone.
#
#   ./install-image-h3.sh                 first install, or update in place
#   ./install-image-h3.sh --no-smoke      skip the generation at the end
#   ./install-image-h3.sh --uninstall     remove the unit and the venv (weights kept)
#   ./install-image-h3.sh --units-only    the unit alone: no runtime, no weights (install.sh, on demand)
#   ./install-image-h3.sh --prepare       the runtime alone, no sudo: the cockpit's first Load runs it
#                                         (./prepare-lane.sh image-h3)
#
# install.sh puts its unit in on a plain install, as for every lane, and the lane's first Load
# builds its runtime and fetches its weights: the checkpoint (68.7 GB, 64 GiB in WEIGHTS_GB below),
# the text encoder it loads from MiniMax-H3 (66.7 GB, 63 GiB in ENCODER_GB; nothing to fetch on a
# box whose video lane has its weights, the files being the same), and the runtime with its build
# (RUNTIME_GB). When the disk does not have the room, this refuses before downloading anything,
# install.sh says so and goes on, and ./install.sh --no-image-h3 leaves the lane out of the run
# that passes it.
#
# WHY A LANE OF ITS OWN. SGLang serves this model from its main branch only (sgl-project/sglang
# PR 49552, merged on 2026-11-10, in no release yet), and the Qwen-Image 2.1 lane runs the v0.5.21
# release: two runtimes, so two venvs, two units and two ports. The release wheel goes in first,
# v0.5.23, whose torch, kernels and transformers are the ones that main asks for, then the source
# at the commit measured on the reference box (2026-11-11), then the kernel packages of SGLang's
# DGX Spark recipe for this model.
#
# ITS LICENCE. The MiniMax-H3-Image Community License, and the MiniMax H3 one for the text encoder:
# neither is granted in the European Union, the United Kingdom, South Korea or the United States
# (their "Excluded Territories"); elsewhere, MiniMax's written authorization is needed above USD 1
# million of annual revenue or for a service hosted for others, and a commercial product shows the
# name "MiniMax-H3-Image". Read both before any use (docs/image-h3-lane.md).
#
# ONE ENGINE AT A TIME. It peaked at 92.3 GB on the reference box and its recipe asks for 106 GiB
# available at its start: it fits beside no serving lane. The unit says so with Conflicts=, so
# starting this stops the other lanes and starting another lane stops this.
set -euo pipefail
# One spelling of the home, without a trailing slash, as install.sh says why: the cockpit's
# exact-argv sudoers is rendered from it and has to match the cockpit's own calls (2026-10-22).
while [ -n "${HOME:-}" ] && [ "$HOME" != / ] && [ "${HOME%/}" != "$HOME" ]; do HOME="${HOME%/}"; done; export HOME
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT=qwen38-image-h3.service
INSTALLED="/etc/systemd/system/$UNIT"
die(){ printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
step(){ printf '\n\033[1;36m── %s\033[0m\n' "$*"; }

# Under sudo every path below moves to /root: the venv, the weights and the key the
# unit reads. install.sh refuses the same way, for the same reason.
if [ "$(id -u)" = "0" ] && [ "${ALLOW_ROOT:-0}" != "1" ]; then
  die "run this as the user who will use the box, not as root (everything it installs is addressed from \$HOME)."
fi

SMOKE=1; ACTION=install; MODE=full
for a in "$@"; do case "$a" in
  --no-smoke) SMOKE=0 ;;
  --uninstall) ACTION=uninstall ;;
  # The two halves of a lane installed on demand, as for the image and video lanes: --units-only
  # puts in what needs root (the unit) and nothing heavy, and --prepare builds the runtime as the
  # user, with no sudo, when the cockpit first loads the lane. The weights come with that Load's
  # switch, as for any Load.
  --units-only) [ "$MODE" = full ] || die "--units-only and --prepare are the two halves of an install: one at a time"; MODE=units ;;
  --prepare) [ "$MODE" = full ] || die "--units-only and --prepare are the two halves of an install: one at a time"; MODE=prepare ;;
  -h|--help) awk 'NR > 1 && !/^#/ { exit } NR > 1' "$0"; exit 0 ;;
  *) die "unknown option: $a" ;;
esac; done

# What the installed unit says, so a re-run without variables changes nothing.
installed(){ { grep -m1 -E "^$1=" "$INSTALLED" 2>/dev/null || true; } | cut -d= -f2-; }
unit_flag(){ { grep -m1 -oE -- "$1 [^ \\\\]+" "$INSTALLED" 2>/dev/null || true; } | awk '{print $2}'; }

CONFIG_DIR="$HOME/.config/qwen38"
# the text lane the box comes back to, one record shared with switch-model.sh (lanes.sh)
# shellcheck source=lanes.sh
. "$HERE/lanes.sh"
# A 64 GB DGX Spark cannot hold this lane (boxmem.py): refused there before anything is fetched,
# and --uninstall still takes away what was put there.
[ "$ACTION" = uninstall ] || [ "$(box_mem_class)" != 64 ] || lane_fits image-h3 || die "The MiniMax-H3-Image lane needs a 128 GB DGX Spark, and this box is a 64 GB one."
# A terminal opened before the account joined the docker group: the run goes on with it (lanes.sh).
docker_session bash "$HERE/${BASH_SOURCE[0]##*/}" "$@"
# Read back one slash at a time, as unit_hf_cache reads the cache (lanes.sh): a unit rendered from a
# home with a trailing slash says "//", and its own spelling kept it as it was at every update.
LANE_DIR="${IMAGE_H3_LANE_DIR:-$(installed WorkingDirectory | tr -s /)}"; LANE_DIR="${LANE_DIR:-$HOME/.local/share/qwen38-image-h3}"
VENV="$LANE_DIR/venv"
SRC="$LANE_DIR/sglang"
WHEEL_NOTE="$LANE_DIR/sglang-wheel"   # the wheel release the venv's dependencies come from (step 3)
SOURCE_NOTE="$LANE_DIR/sglang-source" # the SGLang commit the venv runs from (step 3)
RUNTIME_NOTE="$LANE_DIR/runtime-changed"   # touched whenever this script changes the runtime the lane runs (step 3)
PORT="${IMAGE_H3_PORT:-$(unit_flag --port)}"; PORT="${PORT:-30026}"
# The checkpoint at the revision of 2026-11-12: the weights measured on the reference box on
# 2026-11-11 (492fd8c4), every file of transformer/ and vae/ the same at both, its licence's
# address and its README the only changes. IMAGE_H3_MODEL_REV overrides it.
IMAGE_H3_MODEL_PIN="MiniMaxAI/MiniMax-H3-Image"
IMAGE_H3_MODEL_PIN_REV="5f90473023975e6271b0ed1dd84c4e88e4bbc6d3"
MODEL="$IMAGE_H3_MODEL_PIN"; MODEL_REV="${IMAGE_H3_MODEL_REV:-$IMAGE_H3_MODEL_PIN_REV}"
# Its text encoder, which its model_index.json loads from MiniMax-H3's text_encoder/ folder: the
# video lane's model, at the revision that lane pins (install-video.sh VIDEO_MODEL_PIN_REV), whose
# text_encoder/ holds the same 23 files as its FL2VA/ and Ref2VA/ ones (their blobs compared on the
# reference box, 2026-11-14): on a box whose video lane is prepared, its fetch downloads nothing.
IMAGE_H3_ENCODER_PIN="MiniMaxAI/MiniMax-H3"
IMAGE_H3_ENCODER_PIN_REV="42ed227ee7df40d41602854ae760620d6eb651fe"
# Only what the lane serves is fetched, by allowlist: the repository also carries a copy of the
# weights in the diffusers layout (transformer_diffusers/, vae_diffusers/, 68.6 GB), a copy of the
# text encoder (66.7 GB, which the runtime refuses beside the one model_index.json names), and the
# model card's assets and examples. The same list in switch-model.sh, which fetches it at a Load.
ALLOW="transformer/* vae/* scheduler/* tokenizer/* processor/* model_index.json modular_model_index.json pdd_config.json pdd_fused_heads.safetensors README.md LICENSE"
# The installed unit's cache, then the one the text lane mounts, then the default, as for the
# image and video lanes (install-image.sh says why).
installed_env(){ { grep -m1 -E "^Environment=$1=" "$INSTALLED" 2>/dev/null || true; } | cut -d= -f3-; }
TEXT_UNITS="${TEXT_UNITS:-/etc/systemd/system/qwen38-sglang.service $CONFIG_DIR/launch-flash.sh}"
text_cache(){
  # shellcheck disable=SC2086  # a list of paths
  unit_hf_cache $TEXT_UNITS
}
HF_CACHE="${HF_CACHE:-$(installed_env HF_HOME | tr -s /)}"; HF_CACHE="${HF_CACHE:-$(text_cache)}"   # one slash at a time, as LANE_DIR
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
# The text encoder's folder in that cache, which the unit names: the files of the pinned revision.
ENCODER_DIR="$HF_CACHE/hub/models--${IMAGE_H3_ENCODER_PIN//\//--}/snapshots/$IMAGE_H3_ENCODER_PIN_REV/text_encoder"
# The SGLang source this lane runs: main at the commit SGLang's DGX Spark recipe was measured on
# by this project (study G8, 2026-11-11: 2048 x 2048 in 15.6 s, 4736 x 3552 in 81.1 s). It holds the
# recipe's own commit (539fea720b86, merged by PR 49552 as 170804b40a40) and the runtime's
# --sleep-on-idle (PR 46007), which the unit passes. IMAGE_H3_SGLANG_PIN overrides it only for
# someone deliberately testing another one.
PIN="${IMAGE_H3_SGLANG_PIN:-49f1a44ee348373d29c9292ea36cd45f054ba18a}"
# The released wheel whose dependencies are main's at that commit: torch 2.14.1, sglang-kernel
# 0.4.10, transformers 5.19.0 and diffusers 0.37.0 are pinned the same in both, and FlashInfer 0.7.2
# is the recipe's. It also carries the prebuilt aarch64 kernels the source tree reuses.
WHEEL="${IMAGE_H3_SGLANG_WHEEL:-0.5.23}"
# The recipe's kernel packages, installed with no dependency over the wheel, as the recipe does:
# flash-attn-4 at the beta it names (the wheel asks for 4.0.0b18 or later and would take the
# newest), FlashInfer's cubins and its jit-cache package at 0.7.2 for CUDA 13.0 (the package is the
# version note FlashInfer checks; the kernels for each GPU are packages of their own, which the
# recipe leaves out). Each wheel of FlashInfer's release is fetched by its URL and checked by its
# sha256, which GitHub publishes for the release's assets.
FA4_VERSION="4.0.0b38"
FLASHINFER_RELEASE="https://github.com/flashinfer-ai/flashinfer/releases/download/v0.7.2"
FLASHINFER_CUBIN_FILE="flashinfer_cubin-0.7.2-py3-none-any.whl"
FLASHINFER_CUBIN_SHA256="31704fa5b2562d459d34c67cee9e80e0abb96c82f89943d542276d5a92c88079"
FLASHINFER_JIT_FILE="flashinfer_jit_cache-0.7.2+cu130-cp39-abi3-manylinux_2_28_aarch64.whl"
# what importlib.metadata answers for each once installed: the file name's own version part
# (comparing the whole file name was true forever: every update re-downloaded 1.6 GB and touched
# the runtime note, which kept the cockpit saying the serving lane ran older files; review of v1.50)
FLASHINFER_CUBIN_VERSION="${FLASHINFER_CUBIN_FILE#flashinfer_cubin-}"; FLASHINFER_CUBIN_VERSION="${FLASHINFER_CUBIN_VERSION%%-*}"
FLASHINFER_JIT_VERSION="${FLASHINFER_JIT_FILE#flashinfer_jit_cache-}"; FLASHINFER_JIT_VERSION="${FLASHINFER_JIT_VERSION%%-cp*}"
FLASHINFER_JIT_SHA256="157e74a786c84df49c59cf1d7568a531a8833ec47a58e9fd52de52737705cc21"
# Loopback only. The runtime has no --api-key, so a bind past loopback puts a keyless image
# generator on that interface; and at this commit its scheduler, a socket that unpickles what it is
# sent, follows an explicit IPv4 --host (scheduler_endpoint_for), where the image and video lanes
# patch it to stay on loopback (scheduler-loopback). The cockpit, on this machine, is the door.
IMAGE_H3_BIND="${IMAGE_H3_BIND:-$(unit_flag --host)}"; IMAGE_H3_BIND="${IMAGE_H3_BIND:-127.0.0.1}"
case "$IMAGE_H3_BIND" in
  localhost) ;;
  127.*) case "$IMAGE_H3_BIND" in *[!0-9.]*|*..*) die "IMAGE_H3_BIND takes an IPv4 loopback address, got: $IMAGE_H3_BIND" ;; esac
         IFS=. read -r -a _h3o <<<"$IMAGE_H3_BIND"; _h3bad=0
         [ "${#_h3o[@]}" -eq 4 ] || _h3bad=1
         for _o in ${_h3o[@]+"${_h3o[@]}"}; do case "$_o" in ''|*[!0-9]*) _h3bad=1 ;; *) [ "$_o" -le 255 ] || _h3bad=1 ;; esac; done
         [ "$_h3bad" = 0 ] || die "IMAGE_H3_BIND takes an IPv4 loopback address, got: $IMAGE_H3_BIND (four dotted octets, each up to 255)" ;;
  *) die "IMAGE_H3_BIND=$IMAGE_H3_BIND: this lane binds loopback only (127.0.0.1 or localhost). It has no API key, and at its SGLang commit its scheduler's unauthenticated socket would follow the address. The cockpit is the door to it." ;;
esac
# Its licence's Excluded Territories, said as its weights come and as its unit goes in (docs/image-h3-lane.md).
LICENCE='Its licence, the MiniMax-H3-Image Community License, is not granted in the European Union, the United Kingdom, South Korea or the United States (its Excluded Territories: definition 5 and clause V.4 of its LICENSE).'
# 68.65 GB in 34 files (63.94 GiB) and 66.73 GB in 23 (62.15 GiB), from the Hub's listing of the two pinned
# revisions; the runtime the video lane's 11 (the venv, the checkout and the wheel, 8.6 to 8.8 GiB measured, and
# the pip build tree) and FlashInfer's cubins (1.56 GB). Each at the start of its line: switch-model.sh reads them
# by name there (review of v1.50).
WEIGHTS_GB=64
ENCODER_GB=63
RUNTIME_GB=14
# TODO(measure): the runtime at its first install on the reference box.

if [ "$ACTION" = uninstall ]; then
  step "Removing the MiniMax-H3-Image lane"
  if [ -f "$INSTALLED" ]; then
    WAS_BOOT=0; systemctl is-enabled --quiet "$UNIT" 2>/dev/null && WAS_BOOT=1
    sudo systemctl disable --now "$UNIT" 2>/dev/null || true
    sudo rm -f "$INSTALLED"; sudo systemctl daemon-reload
    echo "unit removed"
    # When this lane was the box's lane at boot, whatever it replaced was disabled by that switch:
    # removing it left no engine at all (the image lane's trap). The text lane comes back.
    TEXT_ENABLED=0
    for u in qwen38-sglang.service qwen38-flash.service qwen38-image.service qwen38-video.service qwen38-kolibri.service qwen38-nemotron.service qwen38-nemotron-super.service qwen38-flash-local.service qwen38-ornith-122b.service qwen38-flash-unsloth.service; do
      systemctl is-enabled --quiet "$u" 2>/dev/null && TEXT_ENABLED=1
    done
    if [ "$WAS_BOOT" -eq 1 ] && [ "$TEXT_ENABLED" -eq 0 ]; then
      text_lane_give_back image-h3 "the MiniMax-H3-Image lane"
    fi
  fi
  # Only what this script put there: LANE_DIR can be a directory shared with other work.
  if [ -d "$LANE_DIR" ]; then
    rm -rf "$VENV" "$SRC" "$WHEEL_NOTE" "$SOURCE_NOTE" "$RUNTIME_NOTE"
    echo "runtime removed: $VENV and $SRC"
    rmdir "$LANE_DIR" 2>/dev/null || echo "kept $LANE_DIR: it holds files the MiniMax-H3-Image lane did not put there"
  fi
  # The text encoder is MiniMax-H3's, which the video lane serves: never named here for removal.
  if [ -d "$HF_CACHE/hub/models--${MODEL//\//--}" ]; then
    echo "the checkpoint is left in $HF_CACHE (about 69 GB); delete it yourself if you want the space:"
    echo "  rm -rf $HF_CACHE/hub/models--${MODEL//\//--}"
    echo "its text encoder is MiniMax-H3's, in the same cache: the video lane's model, left in place"
  fi
  exit 0
fi

# Step 5, the one part that needs root: the unit. A function, since a lane installed on demand
# (--units-only) takes it straight after the preflight.
install_service(){
  step "5/6 Service"
  # The unit runs engine-preflight.sh before every start: run on its own, this installer
  # must put it in place, or the lane would never start.
  mkdir -p "$CONFIG_DIR"
  cmp -s "$HERE/engine-preflight.sh" "$CONFIG_DIR/engine-preflight.sh" \
    || install -m 755 "$HERE/engine-preflight.sh" "$CONFIG_DIR/engine-preflight.sh"
  RENDER="$(mktemp)"; trap 'rm -f "$RENDER"' EXIT
  sed -e "s|__USER__|$(id -un)|g" -e "s|__GROUP__|$(id -gn "$(id -un)")|g" \
      -e "s|__IMAGE_H3_LANE_DIR__|$LANE_DIR|g" -e "s|__IMAGE_H3_VENV__|$VENV|g" \
      -e "s|__IMAGE_H3_MODEL__|$MODEL|g" -e "s|__IMAGE_H3_ENCODER__|$ENCODER_DIR|g" \
      -e "s|__IMAGE_H3_PORT__|$PORT|g" -e "s|__IMAGE_H3_BIND__|$IMAGE_H3_BIND|g" -e "s|__HF_CACHE__|$HF_CACHE|g" \
      -e "s|__HOME__|$HOME|g" \
      "$HERE/$UNIT.template" > "$RENDER"
  grep -q '__[A-Z][A-Z0-9_]*__' "$RENDER" && die "the unit template still holds an unsubstituted placeholder: $(grep -o '__[A-Z][A-Z0-9_]*__' "$RENDER" | sort -u | tr '\n' ' ')"
  # 0644 like every other unit (mktemp creates 0600): what reads the unit, the switch and the
  # cockpit, failed quietly on a 0600 one (install-image.sh says when).
  UNIT_PUT="$(unit_put "$RENDER" "$INSTALLED")"
  case "$UNIT_PUT" in
    same) echo "unit unchanged"
          sudo chmod 644 "$INSTALLED" ;;
    deps) sudo systemctl daemon-reload
          echo "unit updated: only the lines naming the other lanes changed, which the lane does not read" ;;
    *)    sudo systemctl daemon-reload
          echo "unit installed at $INSTALLED" ;;
  esac
  # A lane that serves goes on serving the unit and the runtime it started with: said here when
  # no proof follows to restart it (a boot is minutes, and an image being made is lost).
  LANE_FILES=("$INSTALLED" "$RUNTIME_NOTE" "$WHEEL_NOTE" "$SOURCE_NOTE" "$CONFIG_DIR/engine-preflight.sh")
  if [ "$SMOKE" -eq 0 ] && lane_serves_older "$UNIT" "${LANE_FILES[@]}"; then
    echo "NOTE: The MiniMax-H3-Image lane runs older files than the ones installed now (its unit, its runtime or its start scripts); it runs these after: sudo systemctl restart $UNIT   (or Load it again from the cockpit)"
  fi
  # Installed, not switched to: the lane this box serves stays the lane it serves.
  echo "installed as a lane of its own. Switch to it like any lane:"
  echo "  the cockpit: Load on MiniMax-H3-Image in the Lanes view (it switches, stops the serving lane, starts this one)"
  echo "  a terminal : ./switch-model.sh image-h3, then the two commands it prints"
  echo "MiniMax-H3-Image: ${LICENCE/Its licence/its licence}"
}

step "1/6 Preflight"
case "$(uname -m)" in aarch64|arm64) : ;; *) die "this lane is verified on the DGX Spark's ARM64 GB10 only (this box is $(uname -m))." ;; esac
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: this needs the NVIDIA driver."
command -v git >/dev/null || die "git is required (stock on DGX OS)."
command -v curl >/dev/null || die "curl is required (stock on DGX OS)."
# Run from a terminal, this installs what the lane needs and the box lacks (lanes.sh
# host_packages); the checks below say what could not be put in.
HOST_PKGS=()
python3 -c 'import venv, ensurepip' 2>/dev/null || HOST_PKGS+=(python3-venv)
[ -f "$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])' 2>/dev/null)/Python.h" ] || HOST_PKGS+=(python3-dev)
[ "${#HOST_PKGS[@]}" -eq 0 ] || host_packages "the MiniMax-H3-Image lane" "${HOST_PKGS[@]}" || true
# ensurepip, not just venv: a box without python3-venv makes a venv with no pip.
python3 -c 'import venv, ensurepip' 2>/dev/null || die "python3-venv is missing (a venv made without it has no pip). Fix: sudo apt-get install -y python3-venv"
# python3-dev too: xatlas, which sglang[diffusion] asks for, is built here against Python's
# headers (PyPI has no aarch64 wheel of it), and DGX OS installs python3-pip without them.
[ -f "$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')/Python.h" ] || die "python3-dev is missing (xatlas, which this lane needs, is built here against Python's headers). Fix: sudo apt-get install -y python3-dev"
[ -s "$CONFIG_DIR/api-key" ] || die "no API key at $CONFIG_DIR/api-key. Run ./install.sh first: the cockpit is the authenticated door in front of this lane, and it reads that file."
if [ "$MODE" = units ]; then
  echo "OK: $(uname -m), key present; the runtime and the weights come with the lane's first Load"
  install_service
  step "Done: the MiniMax-H3-Image lane is installed; its first Load prepares it"
  echo "  the cockpit: Load on MiniMax-H3-Image in the Lanes view builds its runtime (about 15 GB),"
  echo "               fetches its checkpoint (about 69 GB) and its text encoder (67 GB, none beside the video lane), then starts it"
  echo "  a terminal : ./install-image-h3.sh --prepare, then ./switch-model.sh image-h3 and the commands it prints"
  exit 0
fi
# What is left to fetch, measured where each part lands, by bytes in the blobs (the folder alone
# says nothing: huggingface_hub creates it before the first byte), in whole GiB down.
# existing: lanes.sh's (the nearest folder of a path that exists)
free_gb(){ { df -BG --output=avail "$(existing "$1")" 2>/dev/null || true; } | tail -1 | tr -dc '0-9'; }
HAVE_B="$({ du -s --apparent-size -B1 "$HF_CACHE/hub/models--${MODEL//\//--}/blobs" 2>/dev/null || true; } | cut -f1)"
WEIGHTS_NEED=$(( (WEIGHTS_GB * 1073741824 - ${HAVE_B:-0}) / 1073741824 )); [ "$WEIGHTS_NEED" -ge 0 ] || WEIGHTS_NEED=0
# The text encoder: nothing when the pinned revision of MiniMax-H3 holds its 14 shards whole in
# any of its three text_encoder folders (the root's, FL2VA's or Ref2VA's: the same blobs), since
# a fetch of text_encoder/ then links the blobs already there.
encoder_cached(){ h3_encoder_cached "$HF_CACHE/hub/models--${IMAGE_H3_ENCODER_PIN//\//--}/snapshots/$IMAGE_H3_ENCODER_PIN_REV"; }
ENCODER_NEED="$ENCODER_GB"; encoder_cached && ENCODER_NEED=0
# --prepare: the weights are the Load's switch's to fetch, and to measure
if [ "$MODE" = prepare ]; then WEIGHTS_NEED=0; ENCODER_NEED=0; fi
# Summed per disk: on a stock box all of it lands on one.
declare -A NEED_ON=() FREE_ON=() WHAT_ON=()
PARTS=("$HF_CACHE|$WEIGHTS_NEED|the checkpoint" "$HF_CACHE|$ENCODER_NEED|its text encoder" "$LANE_DIR|$RUNTIME_GB|the runtime and its build")
for PART in "${PARTS[@]}"; do
  IFS='|' read -r WHERE GB WHAT <<< "$PART"
  DEV="$(stat -L -c %d "$(existing "$WHERE")")"
  NEED_ON[$DEV]=$(( ${NEED_ON[$DEV]:-0} + GB ))
  FREE_ON[$DEV]="$(free_gb "$WHERE")"
  WHAT_ON[$DEV]="${WHAT_ON[$DEV]:+${WHAT_ON[$DEV]}, }$GB for $WHAT ($WHERE)"
done
for DEV in "${!NEED_ON[@]}"; do
  [ "${FREE_ON[$DEV]:-0}" -ge "${NEED_ON[$DEV]}" ] \
    || die "${FREE_ON[$DEV]:-?} GiB free on one disk, and this lane needs about ${NEED_ON[$DEV]} GiB there: ${WHAT_ON[$DEV]}. Free some space, or point HF_CACHE or IMAGE_H3_LANE_DIR at a bigger disk."
done
echo "OK: $(uname -m), key present, weights $([ "$MODE" = prepare ] && echo "left to the Load's switch" || { [ $((WEIGHTS_NEED + ENCODER_NEED)) -eq 0 ] && echo 'already cached'; } || echo "$((WEIGHTS_NEED + ENCODER_NEED)) GiB to download") into $HF_CACHE"

step "2/6 Runtime ($LANE_DIR)"
mkdir -p "$LANE_DIR"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV" || die "could not create the venv at $VENV"
  echo "venv created on $("$VENV/bin/python" -V)"
elif ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  # A venv made while python3-venv was missing has its python and no pip, and every
  # later run took it for a finished one, then died on its first pip call.
  echo "the venv at $VENV has no pip: making it again"
  python3 -m venv --clear "$VENV" || die "could not make the venv at $VENV again"
else
  echo "venv already at $VENV ($("$VENV/bin/python" -V))"
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip >/dev/null

step "3/6 SGLang Diffusion (released wheel first, the pinned source over it, then the recipe's kernels)"
# Order matters, as for the image lane (install-image.sh step 3): the wheel resolves the whole
# dependency tree and ships sglang-kernel built for aarch64, and its version is written down
# once the whole runtime is in (below): a venv from before, or one a run left half built, takes
# it again.
if ! "$VENV/bin/python" -c 'import sglang' 2>/dev/null || [ "$(cat "$WHEEL_NOTE" 2>/dev/null)" != "$WHEEL" ]; then
  echo "installing sglang[diffusion]==$WHEEL (~10 min on a new venv: torch and the kernels are large; an existing one only takes what changed)"
  "$VENV/bin/pip" install --quiet --pre "sglang[diffusion]==$WHEEL" \
    || die "the released wheel would not install. Re-run: pip resumes. If it keeps failing, IMAGE_H3_SGLANG_WHEEL=<version> picks another."
  touch "$RUNTIME_NOTE"
fi
if [ ! -d "$SRC/.git" ]; then
  git clone --quiet https://github.com/sgl-project/sglang "$SRC" || die "could not clone SGLang."
fi
# No local change to this source: the three the image and video lanes patch in (image-sglang/)
# are flags of the runtime at this commit, which the unit passes (its template says which).
CURRENT="$(git -C "$SRC" rev-parse HEAD 2>/dev/null || true)"
if [ "$CURRENT" != "$PIN" ]; then
  git -C "$SRC" fetch --quiet origin "$PIN" 2>/dev/null || git -C "$SRC" fetch --quiet origin
  git -C "$SRC" checkout --quiet "$PIN" \
    || die "could not check out $PIN: not in the SGLang repository, or local edits in $SRC block it (git -C $SRC status says which)."
  echo "source checked out at ${PIN:0:12}"
else
  echo "source already at ${PIN:0:12}"
fi
HEAD_NOW="$(git -C "$SRC" rev-parse HEAD)"
[ "$(cat "$SOURCE_NOTE" 2>/dev/null)" = "$HEAD_NOW" ] || { printf '%s\n' "$HEAD_NOW" > "$SOURCE_NOTE"; touch "$RUNTIME_NOTE"; }
# --no-deps: the wheel above resolved them. SGLANG_BUILD_RUST_EXTS=none: the source declares Rust
# extensions of the LLM runtime only, none imported by the diffusion server, and building them needs
# cargo, which DGX OS does not ship (install-image.sh says when it was found).
if ! "$VENV/bin/python" -c 'import sglang, pathlib, sys; sys.exit(0 if str(pathlib.Path(sglang.__file__).parent).startswith("'"$SRC"'") else 1)' 2>/dev/null; then
  echo "overlaying the pinned source (editable, no dependency resolution, no Rust extensions)"
  SGLANG_BUILD_RUST_EXTS=none "$VENV/bin/pip" install --quiet --no-deps -e "$SRC/python" \
    || die "the editable overlay failed. The venv still holds the released wheel; re-run to retry."
  touch "$RUNTIME_NOTE"
fi
# The recipe's kernel packages, each only when the venv holds another version.
kernel_version(){ "$VENV/bin/python" -c 'import importlib.metadata as m, sys
try: print(m.version(sys.argv[1]))
except m.PackageNotFoundError: print("")' "$1" 2>/dev/null || true; }
if [ "$(kernel_version flash-attn-4)" != "$FA4_VERSION" ]; then
  echo "installing flash-attn-4==$FA4_VERSION (the recipe's, no dependency resolution)"
  "$VENV/bin/pip" install --quiet --no-deps "flash-attn-4==$FA4_VERSION" \
    || die "flash-attn-4==$FA4_VERSION would not install (no internet, or PyPI dropped it). Re-run to retry."
  touch "$RUNTIME_NOTE"
fi
if [ "$(kernel_version flashinfer-cubin)" != "$FLASHINFER_CUBIN_VERSION" ] \
   || [ "$(kernel_version flashinfer-jit-cache)" != "$FLASHINFER_JIT_VERSION" ]; then
  echo "installing FlashInfer's cubins and jit-cache package 0.7.2 (the recipe's, 1.6 GB, each checked by its sha256)"
  WHEELS="$(mktemp -d)"
  for W in "$FLASHINFER_CUBIN_FILE|$FLASHINFER_CUBIN_SHA256" "$FLASHINFER_JIT_FILE|$FLASHINFER_JIT_SHA256"; do
    F="${W%%|*}"; SUM="${W##*|}"
    curl -fsSL --retry 3 -o "$WHEELS/$F" "$FLASHINFER_RELEASE/${F//+/%2B}" \
      || { rm -rf "$WHEELS"; die "could not download $F from FlashInfer's release. Re-run to retry."; }
    echo "$SUM  $WHEELS/$F" | sha256sum -c --quiet - \
      || { rm -rf "$WHEELS"; die "$F does not match the sha256 FlashInfer's release published for it; refusing to install it."; }
  done
  "$VENV/bin/pip" install --quiet --no-deps "$WHEELS/$FLASHINFER_CUBIN_FILE" "$WHEELS/$FLASHINFER_JIT_FILE" \
    || { rm -rf "$WHEELS"; die "FlashInfer's packages would not install. Re-run to retry."; }
  rm -rf "$WHEELS"
  touch "$RUNTIME_NOTE"
fi
# Asked of the registry itself, as the image lane asks it of Qwen-Image 2.1: the exact id maps to
# MiniMax-H3-Image's own config (registered under it at this commit, ahead of MiniMax-H3's), and
# the pipeline its model_index.json names resolves to MiniMax-H3's image pipeline.
"$VENV/bin/python" - <<'PY' || die "the runtime does not know MiniMax-H3-Image. The pin may be wrong for this checkout."
import pathlib, sglang
from sglang.multimodal_gen import registry
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3_image import MiniMaxH3ImagePipelineConfig
from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import MiniMaxH3ImagePipeline
registry._ensure_registry_initialized()
info = registry._CONFIG_REGISTRY.get(registry._MODEL_HF_PATH_TO_NAME.get("MiniMaxAI/MiniMax-H3-Image", ""))
if info is None or info.pipeline_config_cls is not MiniMaxH3ImagePipelineConfig:
    raise SystemExit("MiniMaxAI/MiniMax-H3-Image does not map to MiniMax-H3-Image's own config in the model registry")
if registry.get_pipeline_class("MiniMaxH3ImagePipeline") is not MiniMaxH3ImagePipeline:
    raise SystemExit("MiniMaxH3ImagePipeline does not resolve to MiniMax-H3's image pipeline in the model registry")
print(f"   runtime: {pathlib.Path(sglang.__file__).parent}")
PY
# The runtime is whole: its note says so, and it is what the cockpit and install.sh read as this
# lane prepared. Written last, and only when it changes (install-image.sh says why).
[ "$(cat "$WHEEL_NOTE" 2>/dev/null)" = "$WHEEL" ] || printf '%s\n' "$WHEEL" > "$WHEEL_NOTE"

if [ "$MODE" = prepare ]; then
  step "Done: the MiniMax-H3-Image lane's runtime is ready; its Load fetches the weights and starts it"
  exit 0
fi

step "4/6 Weights ($MODEL at ${MODEL_REV:0:12} and its text encoder, one-time, resumable)"
# ckpt-fetch.py, the one fetch of every lane's checkpoint (resumed attempts, the room, the byte
# count). Xet off and HF_TOKEN passed through, the lessons of the text lanes. The unit serves the
# checkpoint by name with HF_HUB_OFFLINE=1, which resolves refs/main, and a download by commit
# writes no ref: --refs-main pin points main at the pinned commit. The text encoder is named in
# the unit by its folder at its revision, and its refs/main is the video lane's: left as it is.
echo "MiniMax-H3-Image: ${LICENCE/Its licence/its licence}"
echo "fetching with allow: $ALLOW"
HF_HOME="$HF_CACHE" HF_HUB_DOWNLOAD_TIMEOUT=30 HF_HUB_DISABLE_XET=1 \
  "$VENV/bin/python" "$HERE/ckpt-fetch.py" "$MODEL" "$MODEL_REV" --allow "$ALLOW" --refs-main pin \
  || die "checkpoint download failed. Causes: no internet, HuggingFace throttling an unauthenticated download (set HF_TOKEN=<your token>), a pinned revision removed upstream (IMAGE_H3_MODEL_REV=main serves the current one), or a permission error in the cache. Re-running resumes."
HF_HOME="$HF_CACHE" HF_HUB_DOWNLOAD_TIMEOUT=30 HF_HUB_DISABLE_XET=1 \
  "$VENV/bin/python" "$HERE/ckpt-fetch.py" "$IMAGE_H3_ENCODER_PIN" "$IMAGE_H3_ENCODER_PIN_REV" --allow "text_encoder/*" --offline-first \
  || die "the text encoder's download failed (the same causes as the checkpoint's). Re-running resumes."

install_service

if [ "$SMOKE" -eq 0 ]; then step "Done (smoke test skipped)"; exit 0; fi

step "6/6 Proving it serves (one image, then the box goes back to the lane it was serving)"
# The restore mirrors this unit's Conflicts= set: starting it stops every other lane, so
# whatever was serving has to be started back.
WAS_OTHER=""
for u in qwen38-sglang.service qwen38-flash.service qwen38-image.service qwen38-video.service qwen38-kolibri.service qwen38-nemotron.service qwen38-nemotron-super.service qwen38-flash-local.service qwen38-ornith-122b.service qwen38-flash-unsloth.service; do
  systemctl is-active --quiet "$u" 2>/dev/null && WAS_OTHER="$u"
done
# If this lane was already serving, the test leaves it serving.
WAS_H3=0
systemctl is-active --quiet "$UNIT" 2>/dev/null && WAS_H3=1
[ -n "$WAS_OTHER" ] && echo "note: $WAS_OTHER is serving and will be stopped for this test, then started again."
# From here on the other lane is DOWN: the trap puts it back on success and on every failure alike.
restore_other_lane() {
  [ "$WAS_H3" -eq 1 ] || sudo systemctl stop "$UNIT" 2>/dev/null || true
  if [ -n "$WAS_OTHER" ]; then
    echo "starting $WAS_OTHER again"
    sudo systemctl start "$WAS_OTHER" || echo "WARNING: could not restart $WAS_OTHER. Do it by hand: sudo systemctl start $WAS_OTHER"
  fi
  rm -f "$RENDER" "${OUT:-}"
}
trap restore_other_lane EXIT
# A lane that runs older files than the ones installed now restarts for the proof.
if [ "$WAS_H3" -eq 1 ] && lane_serves_older "$UNIT" "${LANE_FILES[@]}"; then
  echo "The MiniMax-H3-Image lane runs older files than the ones installed now (its unit, its runtime or its start scripts): it restarts to prove these"
  sudo systemctl restart "$UNIT"
else
  sudo systemctl start "$UNIT"
fi
READY=0
for i in $(seq 1 120); do
  curl -fsS -m 5 "http://$IMAGE_H3_BIND:$PORT/health" >/dev/null 2>&1 && { READY=1; break; }
  sleep 10
done
[ "$READY" -eq 1 ] || { sudo journalctl -u "$UNIT" -n 40 --no-pager || true; die "the lane did not answer /health within 20 min. The journal above says why."; }
echo "up after ~$(( (i - 1) * 10 )) s; generating a 1024x1024 image (made on a 2048x2048 canvas, about 16 s)"
OUT="$(mktemp)"
# `|| true`: a transport failure made curl exit non-zero inside the assignment, and set -e ended
# the script before the message and the journal below (the image lane's lesson).
CODE=$(curl -sS -o "$OUT" -w '%{http_code}' -m 600 "http://$IMAGE_H3_BIND:$PORT/v1/images/generations" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A capybara reading a book by candlelight","size":"1024x1024","output_format":"png","response_format":"b64_json","generator_device":"cpu","seed":42}' || true)
if [ "$CODE" != 200 ]; then
  echo "HTTP ${CODE:-000}: $(head -c 300 "$OUT" 2>/dev/null)"
  sudo journalctl -u "$UNIT" -n 40 --no-pager || true
  die "the lane started but did not make the smoke image (HTTP ${CODE:-000}); the journal above says why."
fi
"$VENV/bin/python" - "$OUT" <<'PY' || die "the lane answered 200 with something that is not a 1024x1024 image."
import base64, json, struct, sys
d = json.load(open(sys.argv[1]))["data"][0]
raw = base64.b64decode(d["b64_json"])
if raw[:8] != b"\x89PNG\r\n\x1a\n":
    raise SystemExit("not a PNG")
w, h = struct.unpack(">II", raw[16:24])
if (w, h) != (1024, 1024):
    raise SystemExit(f"got {w}x{h}")
print(f"   {w}x{h}, {len(raw)/1e6:.1f} MB")
PY
# What the lane costs with nothing to do, from its own cgroup: the CPU time systemd accounts to
# it over 10 s, after 5 s for the request's tail to finish. About 0 with --sleep-on-idle, about 1
# core without it. A note, not a failure: it serves either way.
CG="/sys/fs/cgroup$(systemctl show -p ControlGroup --value "$UNIT" 2>/dev/null)"
if [ -r "$CG/cpu.stat" ]; then
  sleep 5
  U0=$(awk '/^usage_usec/{print $2}' "$CG/cpu.stat"); sleep 10
  U1=$(awk '/^usage_usec/{print $2}' "$CG/cpu.stat")
  IDLE_CORES=$(awk -v a="$U0" -v b="$U1" 'BEGIN{printf "%.2f", (b-a)/1e7}')
  echo "   at rest: $IDLE_CORES CPU cores"
  awk -v c="$IDLE_CORES" 'BEGIN{exit !(c > 0.5)}' \
    && echo "NOTE: the lane holds a CPU core while idle; its unit's --sleep-on-idle is not in effect (a lane started before it keeps the old loop until its next start)."
fi
# the trap stops the lane and brings the other lane back, whichever way this ends

step "Done: the MiniMax-H3-Image lane is installed and proved it serves on $IMAGE_H3_BIND:$PORT"
echo "  switch to it : Load on MiniMax-H3-Image in the cockpit's Lanes view, or ./switch-model.sh image-h3"
echo "  back         : Load on any other lane there, or ./switch-model.sh stock"
echo "  generate     : the cockpit's Image view, or the API on :$PORT (loopback, no key)"
