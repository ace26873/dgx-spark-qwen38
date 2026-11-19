#!/usr/bin/env bash
# Surgical target-model switch on a live install, between these targets:
#
#   ./switch-model.sh stock           # RadixArk/Qwen3.8-27B-NVFP4 (SGLang)
#   ./switch-model.sh uncensored      # edp1096/Huihui-...-abliterated-NVFP4 (SGLang)
#   ./switch-model.sh fp8             # Qwen/Qwen3.8-27B-FP8 (SGLang)
#   ./switch-model.sh uncensored-fp8  # edp1096/Huihui-...-abliterated-FP8 (SGLang)
#   ./switch-model.sh arete           # atakans/Arete-27B-NVFP4, Nous Research's fine-tune of the 27B (SGLang)
#   ./switch-model.sh flash           # RadixArk/Qwen3.8-Flash-Next-NVFP4 (SGLang)
#   ./switch-model.sh flash-nvda      # nvidia/Qwen3.8-Flash-Next-NVFP4, ModelOpt mixed precision
#   ./switch-model.sh flash-uncensored # the abliterated build of the same tree
#   ./switch-model.sh image           # Qwen/Qwen-Image-2.1 (SGLang Diffusion, own venv)
#   ./switch-model.sh image-turbo     # Qwen/Qwen-Image-2.1-Turbo, its eight-step distillation
#   ./switch-model.sh video           # MiniMaxAI/MiniMax-H3 (SGLang Diffusion, own venv)
#   ./switch-model.sh kolibri         # Aleph-Alpha/Kolibri-1, the official FP8 (vLLM + Aleph Alpha's plugin)
#   ./switch-model.sh kolibri-nvfp4   # Quantinine/Kolibri-1-NVFP4A16-FP8, its experts in NVFP4
#   ./switch-model.sh nemotron        # nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB and its drafter (vLLM nightly + b12x)
#   ./switch-model.sh nemotron-super  # nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 and its drafter (the same image)
#   ./switch-model.sh flash-local     # nvidia/Qwen3.8-Flash-Next-FP4-Local, NVIDIA's FP4 build of the flash model (vLLM + b12x)
#   ./switch-model.sh ornith-122b     # ornith-ai/Ornith-1.5-122B-NVFP4, Ornith AI's mixture of experts (vLLM + b12x)
#   ./switch-model.sh flash-unsloth   # unsloth/Qwen3.8-Flash-Next-GGUF, Unsloth's GGUF of the flash model (llama.cpp)
#   ./switch-model.sh flash-unsloth-iq3 # the same lane on its UD-IQ3_XXS quantization, its experts in 3-bit
#   ./switch-model.sh image-h3        # MiniMaxAI/MiniMax-H3-Image (SGLang Diffusion at SGLang's main, own venv)
#   ./switch-model.sh transcribe      # NVIDIA's Nemotron-3.5-Transcribe-30B-A4B in GGUF (llama.cpp, its own port, 30036)
#   ./switch-model.sh lfm25           # LiquidAI/LFM2.5-24B-A2B-FP8 and its DSpark draft (SGLang, the 27B lane's image)
#   ./switch-model.sh lfm3            # LiquidAI/LFM3-1.7B-NVFP4 (vLLM's own nightly image)
#   ./switch-model.sh <target> --fetch  # that target's checkpoint only, into the cache; nothing else changes
#   ./switch-model.sh <flash-local|flash-unsloth|flash-unsloth-iq3> --clients
#                                     # that lane's clients for its unit's window; nothing else
#
# Within the 27B lane (stock, uncensored, fp8, uncensored-fp8, arete) it does what it
# always did:
#   1. downloads the checkpoint into $HF_CACHE (resumable, pinned revision);
#   2. applies the 1M YaRN config patch ONLY if the installed unit uses it, then, for
#      arete, whose checkpoint declares none, the lane's fp8 KV cache declaration;
#   3. regenerates the patched chat template from the target's own snapshot;
#   4. rewrites ONLY --model-path/--revision in the qwen38-sglang unit.
#
# Across lanes (27B <-> flash) both stacks must already be installed once
# (each by its own ./install.sh run, or a lane installed on demand prepared by
# its first Load, ./prepare-lane.sh: the switch is surgical, it does not
# download images or build overlays). It then re-verifies the checkpoint,
# regenerates the target's template, flips which unit is enabled at boot,
# and points the opencode default model at the target.
#
# The DFlash2 drafter (27B) and the NEXTN/MTP head (flash) are lossless speculative
# paths: drafts are verified against the target, quality is the target's own.
# The switch takes effect on the engine's NEXT restart: this script never starts, stops
# or restarts an engine, and prints the exact commands instead. The services around it: it
# restarts opencode-web when the limits it reads change (a switch across lanes), and, on a
# box whose proxy unit is from before v1.18.7, the proxy, the old way of moving its
# ceiling (since v1.18.7 the ceiling follows the lane that serves). For the 27B lane it
# rewrites --model-path, --revision and the KV cache line of the unit.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
die() { printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
# One spelling of the home, without a trailing slash, as install.sh says why: the cockpit's
# exact-argv sudoers is rendered from it and has to match the cockpit's own calls (2026-10-22).
while [ -n "${HOME:-}" ] && [ "$HOME" != / ] && [ "${HOME%/}" != "$HOME" ]; do HOME="${HOME%/}"; done; export HOME
# A download says its bytes every 10 s (hf-progress.py), and with no terminal on stdout (the
# cockpit's Load runs this as a job) snapshot_download's own bars go: they count files, a
# 53 GB safetensors the same as a 2 KB config, and in a job each redraw is one more line.
DL_NO_BARS=""; [ -t 1 ] || DL_NO_BARS=1

_ENV_HF_CACHE="${HF_CACHE:-}"      # before the pins below give it install.sh's default
# No target, no switch: run bare, this used to switch the box to stock, re-enabling the
# 27B lane and restarting what follows it, where the usage was all anyone wanted (found
# in review, 2026-09-24). MODEL_CHOICE= still names one, as install.sh takes it.
CHOICE="${1:-${MODEL_CHOICE:-}}"
case "$CHOICE" in stock|uncensored|fp8|uncensored-fp8|arete|flash|flash-nvda|flash-uncensored|image|image-turbo|video|kolibri|kolibri-nvfp4|nemotron|nemotron-super|flash-local|ornith-122b|flash-unsloth|flash-unsloth-iq3|image-h3|transcribe|lfm25|lfm3) ;; *) printf 'usage: ./switch-model.sh <stock|uncensored|fp8|uncensored-fp8|arete|flash|flash-nvda|flash-uncensored|image|image-turbo|video|kolibri|kolibri-nvfp4|nemotron|nemotron-super|flash-local|ornith-122b|flash-unsloth|flash-unsloth-iq3|image-h3|transcribe|lfm25|lfm3> [--fetch]\n       ./switch-model.sh <flash-local|flash-unsloth|flash-unsloth-iq3> --clients\n' >&2; [ -z "$CHOICE" ] && exit 2; die "unknown target: $CHOICE" ;; esac
# --fetch: the target's checkpoint only, fetched as a switch fetches it (the cache first,
# resumable, its room checked), and nothing else: no unit, no boot, no engine, no record.
# Library's Download runs it, and a lane installed without its weights gets them so.
# --clients: the clients of a lane that follow its unit's window (FLASH_LOCAL_CONTEXT, FLASH_UNSLOTH_CONTEXT): opencode,
# and through its config pi and claude-local, set for the window its installed unit starts with, as a switch to it sets
# them, and nothing else: no fetch, no unit, no boot, no engine, no record. The cockpit's Load of such a lane that serves
# without being the boot lane, with an update owed to it, runs it before the restart: a switch would have pointed the boot
# at it, which no update asks for, and a window chosen with --units-only reached the engine and not the clients (review
# of v1.48.0, 2026-11-13).
FETCH_ONLY=0; CLIENTS_ONLY=0
case "${2:-}" in
  "") ;;
  --fetch) FETCH_ONLY=1 ;;
  --clients)
    case "$CHOICE" in flash-local|flash-unsloth|flash-unsloth-iq3) CLIENTS_ONLY=1 ;;
      *) die "--clients is for flash-local, flash-unsloth and flash-unsloth-iq3, the lanes whose clients follow their unit's window" ;; esac ;;
  *) die "unknown option: $2 (the options are --fetch, and --clients for flash-local, flash-unsloth and flash-unsloth-iq3)" ;;
esac
fetched(){  # $1 what: in --fetch mode the checkpoint is in the cache, and the run ends here
  [ "$FETCH_ONLY" -eq 1 ] || return 0
  printf '\n\033[1;32m── %s is in the cache; nothing else was changed (--fetch)\033[0m\n' "$1"
  exit 0
}


PINS="$(grep -E '^(IMAGE|STOCK_REPO|STOCK_REV|UNC_REPO|UNC_REV|FP8_REPO|FP8_REV|UNCFP8_REPO|UNCFP8_REV|ARETE_REPO|ARETE_REV|FLASH_REPO|FLASH_REV|FLASH_NVDA_REPO|FLASH_NVDA_REV|FLASH_UNC_REPO|FLASH_UNC_REV|FLASH_IMAGE|FLASH_SERVE_IMAGE|SERVE_IMAGE|MODEL_CHOICE|HF_CACHE|CONFIG_DIR)=' "$REPO_DIR/install.sh" || true)"
# A count of matched lines was the old check, and adding a pin broke both scripts
# at once (it did, on 2026-09-08). What matters is not how many lines matched but
# whether every name this script goes on to use is defined, so that is what is
# asserted, and a failure says which one.
eval "$PINS"
for _v in IMAGE SERVE_IMAGE STOCK_REPO STOCK_REV UNC_REPO UNC_REV FP8_REPO FP8_REV UNCFP8_REPO UNCFP8_REV ARETE_REPO ARETE_REV FLASH_REPO FLASH_REV FLASH_NVDA_REPO FLASH_NVDA_REV FLASH_UNC_REPO FLASH_UNC_REV FLASH_IMAGE FLASH_SERVE_IMAGE MODEL_CHOICE HF_CACHE CONFIG_DIR; do
  eval "[ -n \"\${$_v:-}\" ]" || die "install.sh no longer defines $_v (repo layout changed?)"
done
unset _v

IMAGE_UNIT="/etc/systemd/system/qwen38-image.service"
IMAGE_UNIT_NAME="qwen38-image.service"
# The image unit is rewritten between its two checkpoints (image, image-turbo) with sudo,
# so it is staged at a fixed path the cockpit's exact-argv sudoers allowlist names, as the
# 27B unit is below.
IMG_STAGE="$CONFIG_DIR/qwen38-image.service.switch-stage"
VIDEO_UNIT="/etc/systemd/system/qwen38-video.service"
VIDEO_UNIT_NAME="qwen38-video.service"
# The video and image-h3 units are put from a waiting unit (pending) with sudo, so they are staged
# at fixed paths the cockpit's exact-argv sudoers allowlist names, as the image unit is above.
VID_STAGE="$CONFIG_DIR/qwen38-video.service.switch-stage"
IMAGE_H3_UNIT="/etc/systemd/system/qwen38-image-h3.service"
IMAGE_H3_UNIT_NAME="qwen38-image-h3.service"
H3_STAGE="$CONFIG_DIR/qwen38-image-h3.service.switch-stage"
TRANSCRIBE_UNIT="/etc/systemd/system/qwen38-transcribe.service"
TRANSCRIBE_UNIT_NAME="qwen38-transcribe.service"
LFM25_UNIT="/etc/systemd/system/qwen38-lfm25.service"
LFM25_UNIT_NAME="qwen38-lfm25.service"
LFM3_UNIT="/etc/systemd/system/qwen38-lfm3.service"
LFM3_UNIT_NAME="qwen38-lfm3.service"

# The one fetch of every lane's checkpoint, run in the lane's image or venv. Checked here,
# before any branch mounts it: docker -v of a host file that is not there makes a
# root-owned directory in its place.
[ -f "$REPO_DIR/ckpt-fetch.py" ] || die "ckpt-fetch.py is missing next to this script: run it from the repository."

# Free GB under a path's filesystem, whole numbers, "" when df cannot answer. Every
# branch's room check reads through this, so it lives before the branches run.
dl_free_gb(){ { df -BG --output=avail "$(existing "${1:-$HF_CACHE}")" 2>/dev/null || true; } | tail -1 | tr -dc '0-9'; }

# The text lane the box comes back to, one record (lanes.sh says why): a switch to a side lane
# notes the SGLang text unit it takes the boot from, a switch to a text target notes its own.
# shellcheck source=lanes.sh
. "$REPO_DIR/lanes.sh"
# A target a 64 GB DGX Spark cannot hold is refused there before any download (boxmem.py).
[ "$(box_mem_class)" != 64 ] || target_fits "$CHOICE" || die "The $CHOICE target needs a 128 GB DGX Spark, and this box is a 64 GB one."
# A terminal opened before the account joined the docker group: the run goes on with it (lanes.sh).
docker_session bash "$REPO_DIR/${BASH_SOURCE[0]##*/}" "$@"
[ "$FETCH_ONLY" -eq 1 ] || [ "$CLIENTS_ONLY" -eq 1 ] || text_lane_migrate

# ── The Kolibri lane: a text lane on vLLM, switched to the way the image lane is ────
# For the proxy and its clients Kolibri is a text lane like the 27B and the flash: the same
# port, the same key. But nothing in the four hundred lines below is about it: they rewrite
# SGLang units, patch Qwen chat templates and move Qwen prompt ceilings. So it is handled
# here, whole, and exits, with the image lane's contract: verify the checkpoint (the first
# switch to a variant fetches it), name it in the unit, make it the one unit enabled at
# boot, never start or stop anything, print the exact commands.
KOLIBRI_UNIT="/etc/systemd/system/qwen38-kolibri.service"
KOLIBRI_UNIT_NAME="qwen38-kolibri.service"
# Rewritten with sudo between its two checkpoints, so staged at a fixed path the cockpit's
# exact-argv sudoers allowlist names, as the image unit is.
KOL_STAGE="$CONFIG_DIR/qwen38-kolibri.service.switch-stage"
if [ "$CHOICE" = "kolibri" ] || [ "$CHOICE" = "kolibri-nvfp4" ]; then
  [ -f "$KOLIBRI_UNIT" ] || die "the Kolibri-1 lane is not installed on this box (no $KOLIBRI_UNIT). Install it once: ./install.sh --with-kolibri (about 102 GB)"
  # The lane's two checkpoints, their pinned revisions, memory shares and sizes, from
  # install-kolibri.sh, the one place they are declared (the shares and sizes were copied
  # here by hand, and a share tuned there came back at the next switch: found in review,
  # 2026-10-14).
  eval "$(grep -E '^KOLIBRI_(FP8|NVFP4)_(REPO|REV|GMU|GB)=' "$REPO_DIR/install-kolibri.sh" || true)"
  for _v in KOLIBRI_FP8_REPO KOLIBRI_FP8_REV KOLIBRI_FP8_GMU KOLIBRI_FP8_GB \
            KOLIBRI_NVFP4_REPO KOLIBRI_NVFP4_REV KOLIBRI_NVFP4_GMU KOLIBRI_NVFP4_GB; do
    eval "[ -n \"\${$_v:-}\" ]" || die "install-kolibri.sh no longer defines $_v (repo layout changed?)"
  done
  unset _v
  if [ "$CHOICE" = "kolibri-nvfp4" ]; then
    KOL_MODEL="$KOLIBRI_NVFP4_REPO"; KOL_REV="$KOLIBRI_NVFP4_REV"; KOL_GMU="$KOLIBRI_NVFP4_GMU"; KOL_NEED_GB="$KOLIBRI_NVFP4_GB"
  else
    KOL_MODEL="$KOLIBRI_FP8_REPO"; KOL_REV="$KOLIBRI_FP8_REV"; KOL_GMU="$KOLIBRI_FP8_GMU"; KOL_NEED_GB="$KOLIBRI_FP8_GB"
  fi
  KOL_IMAGE="$(grep -m1 -oE 'qwen38-kolibri:[^ ]+' "$KOLIBRI_UNIT" || true)"
  [ -n "$KOL_IMAGE" ] && docker image inspect "$KOL_IMAGE" >/dev/null 2>&1 \
    || die "the Kolibri-1 lane's image (${KOL_IMAGE:-none named in its unit}) is missing: ./prepare-lane.sh kolibri builds it (no sudo; its first Load from the cockpit runs it), then switch again"
  KOL_HF="$(unit_hf_cache "$KOLIBRI_UNIT")"
  KOL_HF="${KOL_HF:-$HF_CACHE}"
  printf '\n\033[1;36m── Verifying %s @ %s in the cache (resumable)\033[0m\n' "$KOL_MODEL" "${KOL_REV:0:12}"
  KOL_TOKEN_ARGS=()
  # by name: docker takes the value from its own environment, so the token is in no argv
  if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; KOL_TOKEN_ARGS=(-e HF_TOKEN); fi
  KOL_PROGRESS_ARGS=()
  if [ -f "$REPO_DIR/hf-progress.py" ]; then
    KOL_PROGRESS_ARGS=(-v "$REPO_DIR/hf-progress.py:/opt/qwen38/hf-progress.py:ro")
  fi
  # In the lane's own image, as the text lanes fetch theirs in their serving image; --init
  # for the same reason as below: a cockpit job stopped past its timeout stops this too.
  # ckpt-fetch.py is the one fetch of every lane's checkpoint (it says why).
  docker run --rm --init --network host --user "$(id -u):$(id -g)" --entrypoint python3 \
    -e HF_HOME=/hf -e HF_HUB_DOWNLOAD_TIMEOUT=30 -e HF_HUB_DISABLE_XET=1 \
    -e HF_HUB_DISABLE_PROGRESS_BARS="$DL_NO_BARS" \
    "${KOL_TOKEN_ARGS[@]}" "${KOL_PROGRESS_ARGS[@]}" \
    -v "$REPO_DIR/ckpt-fetch.py:/opt/qwen38/ckpt-fetch.py:ro" \
    -v "$KOL_HF":/hf "$KOL_IMAGE" /opt/qwen38/ckpt-fetch.py "$KOL_MODEL" "$KOL_REV" --offline-first \
    --need-gb "$KOL_NEED_GB" --free-gb "$(dl_free_gb "$KOL_HF")" \
    || die "the Kolibri-1 checkpoint could not be verified or fetched (re-run to resume; set HF_TOKEN if it stalls)"
  fetched "$KOL_MODEL @ ${KOL_REV:0:12}"
  # The unit names one checkpoint, its revision and its memory share, on one line each; a
  # switch rewrites those and nothing else.
  KOL_CUR="$(grep -m1 -oE -- "[^ ]+ --revision [0-9a-f]{40}" "$KOLIBRI_UNIT" || true)"
  KOL_CUR_GMU="$(grep -m1 -oE -- "--gpu-memory-utilization [0-9.]+" "$KOLIBRI_UNIT" || true)"
  [ -n "$KOL_CUR" ] && [ -n "$KOL_CUR_GMU" ] || die "the Kolibri-1 unit names no checkpoint or memory share where this expects it (hand-edited unit?): re-run ./install-kolibri.sh --no-smoke"
  KOL_WANT="$KOL_MODEL --revision $KOL_REV"; KOL_WANT_GMU="--gpu-memory-utilization $KOL_GMU"
  KOL_REWRITTEN=""
  # The update that waited for this Load (if any) is posed whole first when it serves the asked target:
  # this lane's local image is tagged from the pins its installer pins, so a waiting unit's image line can
  # differ from the one in place, and the rewrite below would lose it (review of v1.52.0, 2026-11-19, GLM 5.3).
  # When it serves the lane's OTHER target, it is posed whole and then rewritten to the asked one: the asked
  # target is the one served, never the reverse, and the checkpoint fetched above is the asked one's, so
  # nothing is downloaded for a target that will not serve (third review of v1.52.0, 2026-11-19, GLM 5.3: the
  # wait was posed whole and the asked target dropped, after its 102 GB checkpoint had been fetched).
  KOL_POSE=""
  if [ -f "$CONFIG_DIR/pending/$KOLIBRI_UNIT_NAME" ]; then
    if [ -f "$CONFIG_DIR/pending/$KOLIBRI_UNIT_NAME.base" ] && [ -f "$KOLIBRI_UNIT" ] \
       && [ "$(cat "$CONFIG_DIR/pending/$KOLIBRI_UNIT_NAME.base")" != "$(sha256sum "$KOLIBRI_UNIT" | cut -d' ' -f1)" ]; then
      unit_pending_clear "$KOLIBRI_UNIT"
      echo "the update that waited for this Load was dropped: $KOLIBRI_UNIT_NAME changed since it was rendered; ./install.sh writes it again"
    else
      KOL_POSE="$CONFIG_DIR/pending/$KOLIBRI_UNIT_NAME"
      KOL_POSE_CUR="$(grep -m1 -oE -- "[^ ]+ --revision [0-9a-f]{40}" "$KOL_POSE" || true)"
      KOL_POSE_GMU="$(grep -m1 -oE -- "--gpu-memory-utilization [0-9.]+" "$KOL_POSE" || true)"
      if [ -z "$KOL_POSE_CUR" ] || [ "$KOL_POSE_CUR" = "$KOL_WANT" ]; then
        # it serves the asked target (or names no checkpoint): it waits for what it lacks, as before
        if [ -n "$(unit_needs "$KOL_POSE")" ]; then
          echo "NOTE: the update that waited for this Load still lacks $(unit_needs "$KOL_POSE" | paste -sd ';' | sed 's/;/; /g'): it stays waiting; ./install.sh --with-kolibri applies it"
          KOL_POSE=""   # it stays waiting: the unit in place is rewritten to the asked target below
        fi
      elif [ -n "$(unit_needs "$KOL_POSE" | grep -v "^checkpoint ${KOL_POSE_CUR%% --revision *}@${KOL_POSE_CUR##* --revision }$" || true)" ]; then
        # it serves the other target and still lacks what the rewrite to the asked one does not replace
        # (its image, a mounted file): it cannot be posed whole, and is dropped, the asked target served
        # from the unit in place
        echo "NOTE: the update that waited for this Load serves ${KOL_POSE_CUR%% --revision *}, not the asked $KOL_MODEL, and still lacks $(unit_needs "$KOL_POSE" | grep -v "^checkpoint ${KOL_POSE_CUR%% --revision *}@${KOL_POSE_CUR##* --revision }$" | paste -sd ';' | sed 's/;/; /g'): it is dropped; ./install.sh writes it again"
        unit_pending_clear "$KOLIBRI_UNIT"
        KOL_POSE=""
      fi
    fi
  fi
  KOL_FROM="$KOLIBRI_UNIT"; KOL_FROM_CUR="$KOL_CUR"; KOL_FROM_GMU="$KOL_CUR_GMU"
  if [ -n "$KOL_POSE" ]; then KOL_FROM="$KOL_POSE"; KOL_FROM_CUR="$KOL_POSE_CUR"; KOL_FROM_GMU="$KOL_POSE_GMU"; fi
  # The wait that serves the asked target but still lacks something (its image) is not posed: the unit in place
  # is rewritten to the asked target anyway, whose checkpoint was just fetched, and the wait is kept, not
  # cleared: the output's "Switch queued" is then true (seventh pass of the v1.52.0 reviews, 2026-11-19:
  # with the stays-branch the unit in place kept the other target while the output named the asked one).
  if [ -n "$KOL_POSE" ] || { [ "$KOL_CUR" != "$KOL_WANT" ] || [ "$KOL_CUR_GMU" != "$KOL_WANT_GMU" ]; }; then
    # the waiting unit posed whole when there is one, then rewritten to the asked target, which is the one
    # served, never the reverse
    if [ -n "$KOL_FROM_CUR" ]; then
      KOL_TEXT="$(cat "$KOL_FROM")"
      KOL_TEXT="${KOL_TEXT//"$KOL_FROM_CUR"/"$KOL_WANT"}"
      printf '%s\n' "${KOL_TEXT//"$KOL_FROM_GMU"/"$KOL_WANT_GMU"}" > "$KOL_STAGE"
      if ! grep -q -- "$KOL_WANT " "$KOL_STAGE" || ! grep -q -- "$KOL_WANT_GMU " "$KOL_STAGE"; then
        rm -f "$KOL_STAGE"; die "the Kolibri-1 unit's rewrite to $KOL_MODEL failed (hand-edited unit?): re-run ./install-kolibri.sh --no-smoke"
      fi
    else
      cp "$KOL_FROM" "$KOL_STAGE"
    fi
    if ! cmp -s "$KOL_STAGE" "$KOLIBRI_UNIT"; then
      diff "$KOL_STAGE" "$KOLIBRI_UNIT" || true   # show exactly what changes
      sudo install -m 644 "$KOL_STAGE" "$KOLIBRI_UNIT"
      [ -n "$KOL_POSE" ] && unit_pending_clear "$KOLIBRI_UNIT"
      # a wait that stays waiting (KOL_POSE empty) keeps a .base naming the unit from before this
      # rewrite: the next Load or switch would drop it as changed since it was rendered; it names
      # the unit in place now
      [ -f "$CONFIG_DIR/pending/$KOLIBRI_UNIT_NAME" ] \
        && sha256sum "$KOLIBRI_UNIT" | cut -d' ' -f1 > "$CONFIG_DIR/pending/$KOLIBRI_UNIT_NAME.base"
      rm -f "$KOL_STAGE"
      sudo systemctl daemon-reload
      [ -n "$KOL_POSE" ] && echo "the update that waited for this Load is in place: $KOLIBRI_UNIT_NAME"
    else
      [ -n "$KOL_POSE" ] && unit_pending_clear "$KOLIBRI_UNIT"   # an unchanged waiting unit is cleared without install
    fi
    KOL_REWRITTEN=1
  fi
  # Which lane this box served at boot, written down before it is disabled: removing the
  # Kolibri lane later gives the boot back to it (install-kolibri.sh --uninstall).
  for TEXT_UNIT_NAME in qwen38-sglang.service qwen38-flash.service; do
    if systemctl is-enabled --quiet "$TEXT_UNIT_NAME" 2>/dev/null; then
      text_lane_note "$TEXT_UNIT_NAME"; break   # both enabled: the 27B, as install.sh follows it
    fi
  done
  # The loop variable ends in UNIT_NAME on purpose: CI checks every
  # `sudo systemctl <verb> "$..UNIT_NAME"` against the cockpit's sudoers allowlist.
  for TEXT_UNIT_NAME in qwen38-sglang.service qwen38-flash.service qwen38-image.service qwen38-video.service qwen38-nemotron.service qwen38-nemotron-super.service qwen38-flash-local.service qwen38-ornith-122b.service qwen38-flash-unsloth.service qwen38-image-h3.service qwen38-transcribe.service qwen38-lfm25.service qwen38-lfm3.service; do
    if systemctl is-enabled --quiet "$TEXT_UNIT_NAME" 2>/dev/null; then
      echo "disabling $TEXT_UNIT_NAME at boot (unit file kept for switching back)"
      sudo systemctl disable "$TEXT_UNIT_NAME"
    fi
  done
  sudo systemctl enable "$KOLIBRI_UNIT_NAME" >/dev/null 2>&1 || sudo systemctl enable "$KOLIBRI_UNIT_NAME"
  sudo systemctl daemon-reload
  RUNNING=""
  for u in qwen38-sglang.service qwen38-flash.service qwen38-image.service qwen38-video.service qwen38-nemotron.service qwen38-nemotron-super.service qwen38-flash-local.service qwen38-ornith-122b.service qwen38-flash-unsloth.service qwen38-image-h3.service qwen38-transcribe.service qwen38-lfm25.service qwen38-lfm3.service; do
    systemctl is-active --quiet "$u" 2>/dev/null && RUNNING="$u"
  done
  printf '\n\033[1;32mSwitch queued: %s (%s)\033[0m\n' "$KOL_MODEL" "$KOLIBRI_UNIT_NAME"
  # The checkpoint is the operator's choice from here, as a text lane's target (2026-10-17):
  # recorded in an installed.json that exists, erased when it is the recommended one.
  python3 "$REPO_DIR/conformity.py" choose --config-dir "$CONFIG_DIR" --repo-dir "$REPO_DIR" \
    KOLIBRI_VARIANT "$([ "$KOL_MODEL" = "$KOLIBRI_NVFP4_REPO" ] && echo nvfp4 || echo fp8)" || true
  if [ -n "$RUNNING" ]; then
    echo "Effective after:  sudo systemctl stop $RUNNING && sudo systemctl start $KOLIBRI_UNIT_NAME   (or next reboot)"
  elif systemctl is-active --quiet "$KOLIBRI_UNIT_NAME" 2>/dev/null; then
    if [ -n "$KOL_REWRITTEN" ]; then
      echo "Effective after:  sudo systemctl restart $KOLIBRI_UNIT_NAME   (it serves the previous checkpoint until then)"
    else
      echo "Its unit already named this checkpoint. Started before it did, the lane serves what it loaded: sudo systemctl restart $KOLIBRI_UNIT_NAME"
    fi
  else
    echo "Effective after:  sudo systemctl start $KOLIBRI_UNIT_NAME   (or next reboot; ready in about 10 min)"
  fi
  # opencode follows the lane, as on a text switch: the kolibri provider (added here to a
  # config written before the lane was installed) becomes the default of the generated
  # config and of the one opencode reads, and the server that reads them restarts when they
  # changed. A box installed with --no-opencode is left alone.
  if [ -f "$CONFIG_DIR/opencode.off" ]; then
    echo "opencode integration is off on this box (./install.sh --no-opencode): default model left alone"
  else
    KOL_OC_SUM="$({ cat "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json" 2>/dev/null || true; } | sha256sum)"
    KOL_PAIR="$("$REPO_DIR/oc-limits.sh" "$CHOICE")" || KOL_PAIR=""
    KOL_NAME="$(python3 "$REPO_DIR/oc-point-default.py" --label "$CHOICE" 262144 || true)"
    KOL_PORT="$({ grep -m1 -oE 'PROXY_PORT=[0-9]+|keepalive-proxy.py [0-9]+' /etc/systemd/system/qwen38-keepalive.service 2>/dev/null || true; } | grep -oE '[0-9]+$' || true)"
    KOL_SRC="$(mktemp)"
    if [ -n "$KOL_PAIR" ] && python3 "$REPO_DIR/oc-kolibri.py" "http://127.0.0.1:${KOL_PORT:-30001}/v1" "$CONFIG_DIR/api-key" \
         "${KOL_PAIR%% *}" "$(echo "$KOL_PAIR" | cut -d' ' -f2)" "${KOL_NAME:-Kolibri-1 (local, 262K)}" > "$KOL_SRC"; then
      # The compaction block is global: sized from Kolibri's window as every text switch sizes
      # it from its own. Left as the Qwen lane wrote it (132,000 on a 27B in 1M mode), opencode
      # kept almost all of a session verbatim against Kolibri's threshold and compacted again
      # every step or two (found in review, 2026-10-14). pi's keepRecentTokens follows (pi-gen).
      KOL_KEEP="$("$REPO_DIR/oc-limits.sh" --preserve "${KOL_PAIR%% *}")" || KOL_KEEP=""
      for OC_JSON in "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json"; do
        [ -f "$OC_JSON" ] || continue
        python3 "$REPO_DIR/oc-merge-limits.py" "$OC_JSON" --add-providers "$KOL_SRC" >/dev/null || true
        python3 "$REPO_DIR/oc-point-default.py" "$OC_JSON" kolibri "$CHOICE" 262144 \
          || echo "NOTE: could not update $OC_JSON (hand-edited?); set its \"model\" field yourself"
        [ -n "$KOL_KEEP" ] && { python3 "$REPO_DIR/oc-merge-limits.py" "$OC_JSON" --compaction "$KOL_KEEP" \
          || echo "NOTE: could not update the compaction block in $OC_JSON; check it by hand ($KOL_KEEP)"; }
      done
    fi
    rm -f "$KOL_SRC"
    if [ -d "$CONFIG_DIR/pi" ] || [ -d "$CONFIG_DIR/omp" ]; then
      python3 "$REPO_DIR/pi-gen.py" --config "$CONFIG_DIR/opencode.json" --out "$CONFIG_DIR" \
        || echo "NOTE: could not regenerate the pi/omp artifacts; run ./pi-gen.py by hand"
    fi
    if [ "$KOL_OC_SUM" != "$({ cat "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json" 2>/dev/null || true; } | sha256sum)" ] \
       && systemctl list-unit-files opencode-web.service >/dev/null 2>&1 && systemctl is-active --quiet opencode-web.service; then
      sudo systemctl restart opencode-web.service \
        && echo "opencode-web.service restarted so it reads the new default" \
        || echo "NOTE: could not restart opencode-web.service; restart it by hand"
    fi
  fi
  echo "Clients reach it through the proxy on :30001 with the box's key, as model kolibri-1."
  echo "Switch back:      ./switch-model.sh stock   (or any other target)"
  exit 0
fi

# ── The Nemotron lanes: text lanes on vLLM, switched to the way the Kolibri lane is ──────
# Bolt (nemotron) and Super VL (nemotron-super), each its own unit and installer and one image for
# the two, under the same contract: verify the checkpoints (the model and its drafter; the first
# switch fetches them), name them in the unit with the memory share its installer pins, make it the
# one unit enabled at boot, never start or stop anything, print the exact commands.
NEMOTRON_UNIT="/etc/systemd/system/qwen38-nemotron.service"
NEMOTRON_UNIT_NAME="qwen38-nemotron.service"
NEMOTRON_SUPER_UNIT="/etc/systemd/system/qwen38-nemotron-super.service"
NEMOTRON_SUPER_UNIT_NAME="qwen38-nemotron-super.service"
# Rewritten with sudo when its pins moved, so staged at a fixed path the cockpit's exact-argv
# sudoers allowlist names, as the Kolibri unit is.
NEM_STAGE="$CONFIG_DIR/qwen38-nemotron.service.switch-stage"
NSV_STAGE="$CONFIG_DIR/qwen38-nemotron-super.service.switch-stage"
if [ "$CHOICE" = "nemotron" ] || [ "$CHOICE" = "nemotron-super" ]; then
  if [ "$CHOICE" = "nemotron" ]; then
    NEM_UNIT="$NEMOTRON_UNIT"; NEM_UNIT_NAME="$NEMOTRON_UNIT_NAME"; NEM_LANE_STAGE="$NEM_STAGE"
    NEM_INSTALLER=install-nemotron.sh; NEM_PIN=NEMOTRON_BOLT; NEM_TITLE="Nemotron 3.5 Bolt"
    NEM_SERVED=nemotron-3.5-bolt; NEM_WINDOW=262144; NEM_SIZE=51; NEM_BOOT="about 8 min"
    # Bolt loads its reasoning parser from its checkpoint's snapshot: its unit names that snapshot
    NEM_NEEDS_SNAP=1
  else
    NEM_UNIT="$NEMOTRON_SUPER_UNIT"; NEM_UNIT_NAME="$NEMOTRON_SUPER_UNIT_NAME"; NEM_LANE_STAGE="$NSV_STAGE"
    NEM_INSTALLER=install-nemotron-super.sh; NEM_PIN=NEMOTRON_SUPER; NEM_TITLE="Nemotron 3.5 Super VL"
    NEM_SERVED=nemotron-3.5-super-vl; NEM_WINDOW=1048576; NEM_SIZE=102; NEM_BOOT="about 11 min"
    NEM_NEEDS_SNAP=0
  fi
  [ -f "$NEM_UNIT" ] || die "the $NEM_TITLE lane is not installed on this box (no $NEM_UNIT). Install it once: ./install.sh --with-$CHOICE (about $NEM_SIZE GB)"
  # The checkpoint, its drafter, their pinned revisions, the memory share and the size, from the
  # lane's installer, the one place they are declared.
  eval "$(grep -E "^${NEM_PIN}_(REPO|REV|DRAFT_REPO|DRAFT_REV|GMU|GMU_64|GB)=" "$REPO_DIR/$NEM_INSTALLER" || true)"
  for _v in REPO REV DRAFT_REPO DRAFT_REV GMU GB; do
    eval "[ -n \"\${${NEM_PIN}_$_v:-}\" ]" || die "$NEM_INSTALLER no longer defines ${NEM_PIN}_$_v (repo layout changed?)"
  done
  unset _v
  eval "NEM_MODEL=\"\$${NEM_PIN}_REPO\"; NEM_REV=\"\$${NEM_PIN}_REV\"; NEM_GMU=\"\$${NEM_PIN}_GMU\"; NEM_NEED_GB=\"\$${NEM_PIN}_GB\""
  # on a 64 GB DGX Spark, the share its installer gives that box (boxmem.py)
  if [ "$(box_mem_class)" = 64 ]; then eval "NEM_GMU=\"\${${NEM_PIN}_GMU_64:-\$NEM_GMU}\""; fi
  eval "NEM_DRAFT=\"\$${NEM_PIN}_DRAFT_REPO\"; NEM_DREV=\"\$${NEM_PIN}_DRAFT_REV\""
  # the two lanes run the one image the Bolt lane builds (install-nemotron-super.sh says why)
  NEM_IMAGE="$(grep -m1 -oE 'qwen38-nemotron:[^ ]+' "$NEM_UNIT" || true)"
  [ -n "$NEM_IMAGE" ] && docker image inspect "$NEM_IMAGE" >/dev/null 2>&1 \
    || die "the $NEM_TITLE lane's image (${NEM_IMAGE:-none named in its unit}) is missing: ./prepare-lane.sh $CHOICE builds it (no sudo; its first Load from the cockpit runs it), then switch again"
  NEM_HF="$(unit_hf_cache "$NEM_UNIT")"
  NEM_HF="${NEM_HF:-$HF_CACHE}"
  NEM_TOKEN_ARGS=()
  # by name: docker takes the value from its own environment, so the token is in no argv
  if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; NEM_TOKEN_ARGS=(-e HF_TOKEN); fi
  NEM_PROGRESS_ARGS=()
  if [ -f "$REPO_DIR/hf-progress.py" ]; then
    NEM_PROGRESS_ARGS=(-v "$REPO_DIR/hf-progress.py:/opt/qwen38/hf-progress.py:ro")
  fi
  # In the lane's own image, as the text lanes fetch theirs in their serving image; --init so
  # that a cockpit job stopped past its timeout stops this too. ckpt-fetch.py is the one fetch
  # of every lane's checkpoint (it says why). The room asked for is the pair's, given to the
  # first fetch, which the second then needs no more than its own share of.
  NEM_ASK="$NEM_NEED_GB"
  for NEM_CKPT in "$NEM_MODEL $NEM_REV" "$NEM_DRAFT $NEM_DREV"; do
    NEM_R="${NEM_CKPT%% *}"; NEM_V="${NEM_CKPT##* }"
    printf '\n\033[1;36m── Verifying %s @ %s in the cache (resumable)\033[0m\n' "$NEM_R" "${NEM_V:0:12}"
    docker run --rm --init --network host --user "$(id -u):$(id -g)" --entrypoint python3 \
      -e HF_HOME=/hf -e HF_HUB_DOWNLOAD_TIMEOUT=30 -e HF_HUB_DISABLE_XET=1 \
      -e HF_HUB_DISABLE_PROGRESS_BARS="$DL_NO_BARS" \
      "${NEM_TOKEN_ARGS[@]}" "${NEM_PROGRESS_ARGS[@]}" \
      -v "$REPO_DIR/ckpt-fetch.py:/opt/qwen38/ckpt-fetch.py:ro" \
      -v "$NEM_HF":/hf "$NEM_IMAGE" /opt/qwen38/ckpt-fetch.py "$NEM_R" "$NEM_V" --offline-first \
      --need-gb "$NEM_ASK" --free-gb "$(dl_free_gb "$NEM_HF")" \
      || die "the $NEM_TITLE checkpoint $NEM_R could not be verified or fetched (re-run to resume; set HF_TOKEN if it stalls)"
    NEM_ASK=1
  done
  # --fetch ends once both are in the cache: the lane loads the drafter with the model
  fetched "$NEM_MODEL @ ${NEM_REV:0:12}, with its drafter,"
  # The unit names the checkpoint's revision after the checkpoint (and Bolt's again, in the path of
  # the reasoning parser in that snapshot), the drafter and its revision, and the memory share: a
  # switch after the pins moved rewrites each of them. It rewrote the first and the share only, and
  # the engine would have started the new weights with the old snapshot's parser and a drafter this
  # switch had not checked (found in review, 2026-10-16).
  NEM_CUR="$(grep -m1 -oE -- "[^ ]+ --revision [0-9a-f]{40}" "$NEM_UNIT" || true)"
  NEM_CUR_GMU="$(grep -m1 -oE -- "--gpu-memory-utilization [0-9.]+" "$NEM_UNIT" || true)"
  # The requests at a time, for the lane whose installer pins them (Super VL's: its memory share was
  # measured with them), rewritten with the share: the two move together. The switch rewrote the
  # share alone, a pairing nobody had measured (found in review, 2026-10-16). Bolt's are its
  # template's own, as it renders them.
  NEM_SEQS="$({ grep -m1 -oE "^${NEM_PIN}_SEQS=\"?[0-9]+" "$REPO_DIR/$NEM_INSTALLER" || true; } | grep -oE '[0-9]+$' || true)"
  NEM_CUR_SEQS="$(grep -m1 -oE -- "--max-num-seqs [0-9]+" "$NEM_UNIT" || true)"
  NEM_WANT_SEQS="$NEM_CUR_SEQS"; [ -n "$NEM_SEQS" ] && NEM_WANT_SEQS="--max-num-seqs $NEM_SEQS"
  NEM_CUR_D="$(grep -m1 -oE -- "--speculative-config.model [^ ]+" "$NEM_UNIT" || true)"
  NEM_CUR_DREV="$(grep -m1 -oE -- "--speculative-config.revision [0-9a-f]{40}" "$NEM_UNIT" || true)"
  NEM_CUR_REPO="${NEM_CUR%% *}"; NEM_CUR_SNAP="models--${NEM_CUR_REPO//\//--}/snapshots/${NEM_CUR##* }/"
  # Bolt's unit must name its revision's snapshot (its parser's path), as before the block was the
  # two lanes': made optional for both, a hand-edited path was left behind the revision rewritten
  # around it, and the engine would have started offline on a parser not there (found in review,
  # 2026-10-16).
  [ -n "$NEM_CUR" ] && [ -n "$NEM_CUR_GMU" ] && [ -n "$NEM_CUR_D" ] && [ -n "$NEM_CUR_DREV" ] \
    && { [ "$NEM_NEEDS_SNAP" -eq 0 ] || grep -qF -- "$NEM_CUR_SNAP" "$NEM_UNIT"; } \
    && { [ -z "$NEM_SEQS" ] || [ -n "$NEM_CUR_SEQS" ]; } \
    || die "the $NEM_TITLE unit names no checkpoint, drafter or memory share where this expects it (hand-edited unit?): re-run ./$NEM_INSTALLER --no-smoke"
  NEM_WANT="$NEM_MODEL --revision $NEM_REV"; NEM_WANT_GMU="--gpu-memory-utilization $NEM_GMU"
  NEM_WANT_D="--speculative-config.model $NEM_DRAFT"; NEM_WANT_DREV="--speculative-config.revision $NEM_DREV"
  NEM_WANT_SNAP="models--${NEM_MODEL//\//--}/snapshots/$NEM_REV/"
  NEM_REWRITTEN=""
  NEM_POSE=""
  # The update that waited for this Load (if any) goes in place whole first: this lane's local image is
  # tagged from the pins its installer pins, so a waiting unit's image line can differ from the one in
  # place, and the rewrite below would lose it (review of v1.52.0, 2026-11-19, GLM 5.3). It is then
  # rewritten to the asked revision like any unit in place: posed whole and left as it was, it would
  # serve the revision the update was rendered with, not the one this switch just fetched (seventh
  # pass of the v1.52.0 reviews, 2026-11-19). The wait is cleared only once the posed unit is in place
  # AND rewritten to the asked revision: a rewrite that fails leaves it waiting for the retry.
  if [ -f "$CONFIG_DIR/pending/$NEM_UNIT_NAME" ]; then
    if [ -f "$CONFIG_DIR/pending/$NEM_UNIT_NAME.base" ] && [ -f "$NEM_UNIT" ] \
       && [ "$(cat "$CONFIG_DIR/pending/$NEM_UNIT_NAME.base")" != "$(sha256sum "$NEM_UNIT" | cut -d' ' -f1)" ]; then
      unit_pending_clear "$NEM_UNIT"
      echo "the update that waited for this Load was dropped: $NEM_UNIT_NAME changed since it was rendered; ./install.sh writes it again"
    elif [ -n "$(unit_needs "$CONFIG_DIR/pending/$NEM_UNIT_NAME")" ]; then
      echo "NOTE: the update that waited for this Load still lacks $(unit_needs "$CONFIG_DIR/pending/$NEM_UNIT_NAME" | paste -sd ';' | sed 's/;/; /g'): it stays waiting; ./install.sh --with-$CHOICE applies it"
    else
      NEM_POSE="$CONFIG_DIR/pending/$NEM_UNIT_NAME"
    fi
  fi
  if [ -n "$NEM_POSE" ]; then
    if ! cmp -s "$NEM_POSE" "$NEM_UNIT"; then
      cp "$NEM_POSE" "$NEM_LANE_STAGE"
      # one exact command per lane, as the cockpit's sudoers allowlist names it
      if [ "$CHOICE" = "nemotron" ]; then
        sudo install -m 644 "$NEM_STAGE" "$NEMOTRON_UNIT"
      else
        sudo install -m 644 "$NSV_STAGE" "$NEMOTRON_SUPER_UNIT"
      fi
      rm -f "$NEM_LANE_STAGE"
      sudo systemctl daemon-reload
      echo "the update that waited for this Load is in place: $NEM_UNIT_NAME"
      # the posed unit is in place: a wait that stays kept (the revalidation below can die, the
      # rewrite can fail) names the unit in place now, so the retry does not read it as changed
      # since it was rendered and drop it
      [ -f "$CONFIG_DIR/pending/$NEM_UNIT_NAME" ] \
        && sha256sum "$NEM_UNIT" | cut -d' ' -f1 > "$CONFIG_DIR/pending/$NEM_UNIT_NAME.base"
      # read back what is in place now, for the rewrite to the asked revision below
      NEM_CUR="$(grep -m1 -oE -- "[^ ]+ --revision [0-9a-f]{40}" "$NEM_UNIT" || true)"
      NEM_CUR_GMU="$(grep -m1 -oE -- "--gpu-memory-utilization [0-9.]+" "$NEM_UNIT" || true)"
      NEM_CUR_D="$(grep -m1 -oE -- "--speculative-config.model [^ ]+" "$NEM_UNIT" || true)"
      NEM_CUR_DREV="$(grep -m1 -oE -- "--speculative-config.revision [0-9a-f]{40}" "$NEM_UNIT" || true)"
      NEM_CUR_SEQS="$({ grep -m1 -oE -- "--max-num-seqs [0-9]+" "$NEM_UNIT" || true; })"
      NEM_CUR_REPO="${NEM_CUR%% *}"; NEM_CUR_SNAP="models--${NEM_CUR_REPO//\//--}/snapshots/${NEM_CUR##* }/"
      # the fields read back from the posed unit are validated as the ones of the unit in place
      # above were: a malformed waiting unit dies here, before the wait is cleared below
      [ -n "$NEM_CUR" ] && [ -n "$NEM_CUR_GMU" ] && [ -n "$NEM_CUR_D" ] && [ -n "$NEM_CUR_DREV" ] \
        && { [ "$NEM_NEEDS_SNAP" -eq 0 ] || grep -qF -- "$NEM_CUR_SNAP" "$NEM_UNIT"; } \
        && { [ -z "$NEM_SEQS" ] || [ -n "$NEM_CUR_SEQS" ]; } \
        || die "the $NEM_TITLE unit names no checkpoint, drafter or memory share where this expects it (hand-edited unit?): re-run ./$NEM_INSTALLER --no-smoke"
    fi
    NEM_REWRITTEN=1
  fi
  if [ "$NEM_CUR|$NEM_CUR_GMU|$NEM_CUR_D|$NEM_CUR_DREV|$NEM_CUR_SEQS" != "$NEM_WANT|$NEM_WANT_GMU|$NEM_WANT_D|$NEM_WANT_DREV|$NEM_WANT_SEQS" ]; then
    NEM_TEXT="$(cat "$NEM_UNIT")"
    NEM_TEXT="${NEM_TEXT//"$NEM_CUR"/"$NEM_WANT"}"
    NEM_TEXT="${NEM_TEXT//"$NEM_CUR_SNAP"/"$NEM_WANT_SNAP"}"
    NEM_TEXT="${NEM_TEXT//"$NEM_CUR_D "/"$NEM_WANT_D "}"
    NEM_TEXT="${NEM_TEXT//"$NEM_CUR_DREV"/"$NEM_WANT_DREV"}"
    [ -n "$NEM_CUR_SEQS" ] && NEM_TEXT="${NEM_TEXT//"$NEM_CUR_SEQS "/"$NEM_WANT_SEQS "}"
    printf '%s\n' "${NEM_TEXT//"$NEM_CUR_GMU"/"$NEM_WANT_GMU"}" > "$NEM_LANE_STAGE"
    if ! grep -qF -- "$NEM_WANT " "$NEM_LANE_STAGE" || ! grep -qF -- "$NEM_WANT_GMU " "$NEM_LANE_STAGE" \
       || ! grep -qF -- "$NEM_WANT_D " "$NEM_LANE_STAGE" || ! grep -qF -- "$NEM_WANT_DREV " "$NEM_LANE_STAGE" \
       || { [ "$NEM_NEEDS_SNAP" -eq 1 ] && ! grep -qF -- "$NEM_WANT_SNAP" "$NEM_LANE_STAGE"; } \
       || { [ -n "$NEM_SEQS" ] && ! grep -qF -- "$NEM_WANT_SEQS " "$NEM_LANE_STAGE"; }; then
      rm -f "$NEM_LANE_STAGE"; die "the $NEM_TITLE unit's rewrite to $NEM_MODEL failed (hand-edited unit?): re-run ./$NEM_INSTALLER --no-smoke"
    fi
    diff "$NEM_LANE_STAGE" "$NEM_UNIT" || true   # show exactly what changes
    # one exact command per lane, as the cockpit's sudoers allowlist names it
    if [ "$CHOICE" = "nemotron" ]; then
      sudo install -m 644 "$NEM_STAGE" "$NEMOTRON_UNIT"
      [ -n "$NEM_POSE" ] && unit_pending_clear "$NEMOTRON_UNIT"   # posed and rewritten to the asked revision: the wait is applied
    else
      sudo install -m 644 "$NSV_STAGE" "$NEMOTRON_SUPER_UNIT"
      [ -n "$NEM_POSE" ] && unit_pending_clear "$NEMOTRON_SUPER_UNIT"   # posed and rewritten to the asked revision: the wait is applied
    fi
    # a wait that stays waiting (not posed: it still lacks something) keeps a .base naming the unit
    # from before this rewrite: the next Load would drop it as changed since it was rendered; it
    # names the unit in place now
    [ -f "$CONFIG_DIR/pending/$NEM_UNIT_NAME" ] \
      && sha256sum "$NEM_UNIT" | cut -d' ' -f1 > "$CONFIG_DIR/pending/$NEM_UNIT_NAME.base"
    rm -f "$NEM_LANE_STAGE"
    NEM_REWRITTEN=1
  else
    # posed whole and already the asked revision: the wait is applied too, now that the unit is in
    # place; a rewrite that died above left the wait in place for the retry
    [ -n "$NEM_POSE" ] && unit_pending_clear "$NEM_UNIT"
  fi
  # Which lane this box served at boot, written down before it is disabled: removing the
  # Nemotron lane later gives the boot back to it (its installer's --uninstall).
  for TEXT_UNIT_NAME in qwen38-sglang.service qwen38-flash.service; do
    if systemctl is-enabled --quiet "$TEXT_UNIT_NAME" 2>/dev/null; then
      text_lane_note "$TEXT_UNIT_NAME"; break   # both enabled: the 27B, as install.sh follows it
    fi
  done
  # The loop variable ends in UNIT_NAME on purpose: CI checks every
  # `sudo systemctl <verb> "$..UNIT_NAME"` against the cockpit's sudoers allowlist.
  for TEXT_UNIT_NAME in qwen38-sglang.service qwen38-flash.service qwen38-image.service qwen38-video.service qwen38-kolibri.service qwen38-nemotron.service qwen38-nemotron-super.service qwen38-flash-local.service qwen38-ornith-122b.service qwen38-flash-unsloth.service qwen38-image-h3.service qwen38-transcribe.service qwen38-lfm25.service qwen38-lfm3.service; do
    [ "$TEXT_UNIT_NAME" = "$NEM_UNIT_NAME" ] && continue
    if systemctl is-enabled --quiet "$TEXT_UNIT_NAME" 2>/dev/null; then
      echo "disabling $TEXT_UNIT_NAME at boot (unit file kept for switching back)"
      sudo systemctl disable "$TEXT_UNIT_NAME"
    fi
  done
  if [ "$CHOICE" = "nemotron" ]; then
    sudo systemctl enable "$NEMOTRON_UNIT_NAME" >/dev/null 2>&1 || sudo systemctl enable "$NEMOTRON_UNIT_NAME"
  else
    sudo systemctl enable "$NEMOTRON_SUPER_UNIT_NAME" >/dev/null 2>&1 || sudo systemctl enable "$NEMOTRON_SUPER_UNIT_NAME"
  fi
  sudo systemctl daemon-reload
  RUNNING=""
  for u in qwen38-sglang.service qwen38-flash.service qwen38-image.service qwen38-video.service qwen38-kolibri.service qwen38-nemotron.service qwen38-nemotron-super.service qwen38-flash-local.service qwen38-ornith-122b.service qwen38-flash-unsloth.service qwen38-image-h3.service qwen38-transcribe.service qwen38-lfm25.service qwen38-lfm3.service; do
    [ "$u" = "$NEM_UNIT_NAME" ] && continue
    systemctl is-active --quiet "$u" 2>/dev/null && RUNNING="$u"
  done
  printf '\n\033[1;32mSwitch queued: %s (%s)\033[0m\n' "$NEM_MODEL" "$NEM_UNIT_NAME"
  if [ -n "$RUNNING" ]; then
    echo "Effective after:  sudo systemctl stop $RUNNING && sudo systemctl start $NEM_UNIT_NAME   (or next reboot)"
  elif systemctl is-active --quiet "$NEM_UNIT_NAME" 2>/dev/null; then
    if [ -n "$NEM_REWRITTEN" ]; then
      echo "Effective after:  sudo systemctl restart $NEM_UNIT_NAME   (it serves the previous unit until then)"
    else
      echo "Its unit already named this checkpoint. Started before it did, the lane serves what it loaded: sudo systemctl restart $NEM_UNIT_NAME"
    fi
  else
    echo "Effective after:  sudo systemctl start $NEM_UNIT_NAME   (or next reboot; ready in $NEM_BOOT)"
  fi
  # opencode follows the lane, as on a text switch: the lane's provider becomes the default of
  # the generated config and of the one opencode reads, and the server that reads them restarts
  # when they changed. A box installed with --no-opencode is left alone.
  if [ -f "$CONFIG_DIR/opencode.off" ]; then
    echo "opencode integration is off on this box (./install.sh --no-opencode): default model left alone"
  else
    NEM_OC_SUM="$({ cat "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json" 2>/dev/null || true; } | sha256sum)"
    NEM_PAIR="$("$REPO_DIR/oc-limits.sh" "$CHOICE")" || NEM_PAIR=""
    NEM_NAME="$(python3 "$REPO_DIR/oc-point-default.py" --label "$CHOICE" "$NEM_WINDOW" || true)"
    NEM_PORT="$({ grep -m1 -oE 'PROXY_PORT=[0-9]+|keepalive-proxy.py [0-9]+' /etc/systemd/system/qwen38-keepalive.service 2>/dev/null || true; } | grep -oE '[0-9]+$' || true)"
    NEM_SRC="$(mktemp)"
    if [ -n "$NEM_PAIR" ] && python3 "$REPO_DIR/oc-nemotron.py" "http://127.0.0.1:${NEM_PORT:-30001}/v1" "$CONFIG_DIR/api-key" \
         "${NEM_PAIR%% *}" "$(echo "$NEM_PAIR" | cut -d' ' -f2)" "${NEM_NAME:-$NEM_TITLE (local)}" "$CHOICE" > "$NEM_SRC"; then
      # The compaction block is global: sized from this lane's window as every text switch sizes
      # it from its own. pi's keepRecentTokens follows (pi-gen).
      NEM_KEEP="$("$REPO_DIR/oc-limits.sh" --preserve "${NEM_PAIR%% *}")" || NEM_KEEP=""
      for OC_JSON in "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json"; do
        [ -f "$OC_JSON" ] || continue
        python3 "$REPO_DIR/oc-merge-limits.py" "$OC_JSON" --add-providers "$NEM_SRC" >/dev/null || true
        python3 "$REPO_DIR/oc-point-default.py" "$OC_JSON" "$CHOICE" "$CHOICE" "$NEM_WINDOW" \
          || echo "NOTE: could not update $OC_JSON (hand-edited?); set its \"model\" field yourself"
        [ -n "$NEM_KEEP" ] && { python3 "$REPO_DIR/oc-merge-limits.py" "$OC_JSON" --compaction "$NEM_KEEP" \
          || echo "NOTE: could not update the compaction block in $OC_JSON; check it by hand ($NEM_KEEP)"; }
      done
    fi
    rm -f "$NEM_SRC"
    if [ -d "$CONFIG_DIR/pi" ] || [ -d "$CONFIG_DIR/omp" ]; then
      python3 "$REPO_DIR/pi-gen.py" --config "$CONFIG_DIR/opencode.json" --out "$CONFIG_DIR" \
        || echo "NOTE: could not regenerate the pi/omp artifacts; run ./pi-gen.py by hand"
    fi
    if [ "$NEM_OC_SUM" != "$({ cat "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json" 2>/dev/null || true; } | sha256sum)" ] \
       && systemctl list-unit-files opencode-web.service >/dev/null 2>&1 && systemctl is-active --quiet opencode-web.service; then
      sudo systemctl restart opencode-web.service \
        && echo "opencode-web.service restarted so it reads the new default" \
        || echo "NOTE: could not restart opencode-web.service; restart it by hand"
    fi
  fi
  echo "Clients reach it through the proxy on :30001 with the box's key, as model $NEM_SERVED."
  echo "Switch back:      ./switch-model.sh stock   (or any other target)"
  exit 0
fi

# ── The Qwen3.8 Flash-Next FP4-Local lane: NVIDIA's FP4 build of the flash model, on vLLM ─────────
# A text lane for the proxy, served by vLLM with b12x like the Nemotron lanes, under the same
# contract: verify the checkpoint (the first switch fetches it), put the unit for it in place, make
# it the one unit enabled at boot, never start or stop anything, print the exact commands. The unit
# is rendered by the lane's installer (install-flash-local.sh --stage, no sudo), which first makes
# its chat template and, when the cache is on another disk than the home directory's, the copies of
# its n-gram table the unit mounts (ple-table.py says why): one renderer, so a switch after the pins
# moved, or after the cache moved, puts in the unit an install would.
FLASH_LOCAL_UNIT="/etc/systemd/system/qwen38-flash-local.service"
FLASH_LOCAL_UNIT_NAME="qwen38-flash-local.service"
# Rewritten with sudo, so staged at a fixed path the cockpit's exact-argv sudoers allowlist names,
# as the other lanes' units are.
FLL_STAGE="$CONFIG_DIR/qwen38-flash-local.service.switch-stage"
if [ "$CHOICE" = "flash-local" ]; then
  [ -f "$FLASH_LOCAL_UNIT" ] || die "the Qwen3.8 Flash-Next FP4-Local lane is not installed on this box (no $FLASH_LOCAL_UNIT). Install it once: ./install.sh --with-flash-local (about 121 GB)"
  # the fetch, the unit and the boot: --clients goes straight to the clients, for the unit in place
  if [ "$CLIENTS_ONLY" -eq 0 ]; then
    # The checkpoint, its pinned revision, the image and the size, from the lane's installer, the one
    # place they are declared.
    eval "$(grep -E '^FLASH_LOCAL_(REPO|REV|IMAGE|GB)=' "$REPO_DIR/install-flash-local.sh" || true)"
    for _v in FLASH_LOCAL_REPO FLASH_LOCAL_REV FLASH_LOCAL_IMAGE FLASH_LOCAL_GB; do
      eval "[ -n \"\${$_v:-}\" ]" || die "install-flash-local.sh no longer defines $_v (repo layout changed?)"
    done
    unset _v
    docker image inspect "$FLASH_LOCAL_IMAGE" >/dev/null 2>&1 \
      || die "the Qwen3.8 Flash-Next FP4-Local lane's image (${FLASH_LOCAL_IMAGE:0:40}...) is missing: ./prepare-lane.sh flash-local pulls it (no sudo; its first Load from the cockpit runs it), then switch again"
    FLL_HF="$(unit_hf_cache "$FLASH_LOCAL_UNIT")"
    FLL_HF="${FLL_HF:-$HF_CACHE}"
    # the room its table's copies need on their disk, before the download (install-flash-local.sh)
    HF_CACHE="$FLL_HF" bash "$REPO_DIR/install-flash-local.sh" --room || exit 1
    printf '\n\033[1;36m── Verifying %s @ %s in the cache (resumable)\033[0m\n' "$FLASH_LOCAL_REPO" "${FLASH_LOCAL_REV:0:12}"
    FLL_TOKEN_ARGS=()
    # by name: docker takes the value from its own environment, so the token is in no argv
    if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; FLL_TOKEN_ARGS=(-e HF_TOKEN); fi
    FLL_PROGRESS_ARGS=()
    if [ -f "$REPO_DIR/hf-progress.py" ]; then
      FLL_PROGRESS_ARGS=(-v "$REPO_DIR/hf-progress.py:/opt/qwen38/hf-progress.py:ro")
    fi
    # In the lane's own image, as the other vLLM lanes fetch theirs; --init so that a cockpit job
    # stopped past its timeout stops this too. ckpt-fetch.py is the one fetch of every lane's
    # checkpoint (it says why).
    docker run --rm --init --network host --user "$(id -u):$(id -g)" --entrypoint python3 \
      -e HF_HOME=/hf -e HF_HUB_DOWNLOAD_TIMEOUT=30 -e HF_HUB_DISABLE_XET=1 \
      -e HF_HUB_DISABLE_PROGRESS_BARS="$DL_NO_BARS" \
      "${FLL_TOKEN_ARGS[@]}" "${FLL_PROGRESS_ARGS[@]}" \
      -v "$REPO_DIR/ckpt-fetch.py:/opt/qwen38/ckpt-fetch.py:ro" \
      -v "$FLL_HF":/hf "$FLASH_LOCAL_IMAGE" /opt/qwen38/ckpt-fetch.py "$FLASH_LOCAL_REPO" "$FLASH_LOCAL_REV" --offline-first \
      --need-gb "$FLASH_LOCAL_GB" --free-gb "$(dl_free_gb "$FLL_HF")" \
      || die "the Qwen3.8 Flash-Next FP4-Local checkpoint could not be verified or fetched (re-run to resume; set HF_TOKEN if it stalls)"
    fetched "$FLASH_LOCAL_REPO @ ${FLASH_LOCAL_REV:0:12}"
    # its chat template, its n-gram table where the engine reads it, and its unit for this checkpoint
    HF_CACHE="$FLL_HF" bash "$REPO_DIR/install-flash-local.sh" --stage "$FLL_STAGE" \
      || { rm -f "$FLL_STAGE"; die "the Qwen3.8 Flash-Next FP4-Local unit could not be prepared (the lines above say why)"; }
    FLL_REWRITTEN=""
    if ! cmp -s "$FLL_STAGE" "$FLASH_LOCAL_UNIT"; then
      diff "$FLL_STAGE" "$FLASH_LOCAL_UNIT" || true   # show exactly what changes
      sudo install -m 644 "$FLL_STAGE" "$FLASH_LOCAL_UNIT"
      unit_pending_clear "$FLASH_LOCAL_UNIT"   # its new unit is in place: an update that waited for this Load is applied
      FLL_REWRITTEN=1
    fi
    rm -f "$FLL_STAGE"
    # Which lane this box served at boot, written down before it is disabled: removing this lane
    # later gives the boot back to it (its installer's --uninstall).
    for TEXT_UNIT_NAME in qwen38-sglang.service qwen38-flash.service; do
      if systemctl is-enabled --quiet "$TEXT_UNIT_NAME" 2>/dev/null; then
        text_lane_note "$TEXT_UNIT_NAME"; break   # both enabled: the 27B, as install.sh follows it
      fi
    done
    # The loop variable ends in UNIT_NAME on purpose: CI checks every
    # `sudo systemctl <verb> "$..UNIT_NAME"` against the cockpit's sudoers allowlist.
    for TEXT_UNIT_NAME in qwen38-sglang.service qwen38-flash.service qwen38-image.service qwen38-video.service qwen38-kolibri.service qwen38-nemotron.service qwen38-nemotron-super.service qwen38-ornith-122b.service qwen38-flash-unsloth.service qwen38-image-h3.service qwen38-transcribe.service qwen38-lfm25.service qwen38-lfm3.service; do
      if systemctl is-enabled --quiet "$TEXT_UNIT_NAME" 2>/dev/null; then
        echo "disabling $TEXT_UNIT_NAME at boot (unit file kept for switching back)"
        sudo systemctl disable "$TEXT_UNIT_NAME"
      fi
    done
    sudo systemctl enable "$FLASH_LOCAL_UNIT_NAME" >/dev/null 2>&1 || sudo systemctl enable "$FLASH_LOCAL_UNIT_NAME"
    sudo systemctl daemon-reload
    RUNNING=""
    for u in qwen38-sglang.service qwen38-flash.service qwen38-image.service qwen38-video.service qwen38-kolibri.service qwen38-nemotron.service qwen38-nemotron-super.service qwen38-ornith-122b.service qwen38-flash-unsloth.service qwen38-image-h3.service qwen38-transcribe.service qwen38-lfm25.service qwen38-lfm3.service; do
      systemctl is-active --quiet "$u" 2>/dev/null && RUNNING="$u"
    done
    printf '\n\033[1;32mSwitch queued: %s (%s)\033[0m\n' "$FLASH_LOCAL_REPO" "$FLASH_LOCAL_UNIT_NAME"
    if [ -n "$RUNNING" ]; then
      echo "Effective after:  sudo systemctl stop $RUNNING && sudo systemctl start $FLASH_LOCAL_UNIT_NAME   (or next reboot)"
    elif systemctl is-active --quiet "$FLASH_LOCAL_UNIT_NAME" 2>/dev/null; then
      if [ -n "$FLL_REWRITTEN" ]; then
        echo "Effective after:  sudo systemctl restart $FLASH_LOCAL_UNIT_NAME   (it serves the previous unit until then)"
      else
        echo "Its unit already named this checkpoint. Started before it did, the lane serves what it loaded: sudo systemctl restart $FLASH_LOCAL_UNIT_NAME"
      fi
    else
      echo "Effective after:  sudo systemctl start $FLASH_LOCAL_UNIT_NAME   (or next reboot; ready in about 6 min, 17 at its first boot)"
    fi
  fi
  # opencode follows the lane, as on a text switch: the lane's provider becomes the default of the
  # generated config and of the one opencode reads, and the server that reads them restarts when
  # they changed. A box installed with --no-opencode is left alone.
  if [ -f "$CONFIG_DIR/opencode.off" ]; then
    echo "opencode integration is off on this box (./install.sh --no-opencode): default model left alone"
  else
    FLL_OC_SUM="$({ cat "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json" 2>/dev/null || true; } | sha256sum)"
    # the pair and the window of the unit just staged: 1,048,576 tokens with FLASH_LOCAL_CONTEXT=1m (v1.45.2)
    FLL_WINDOW="$({ grep -oE -- '--max-model-len [0-9]+' "$FLASH_LOCAL_UNIT" 2>/dev/null || true; } | awk 'NR == 1 {print $2}')"
    FLL_PAIR="$("$REPO_DIR/oc-limits.sh" flash-local --from "$FLASH_LOCAL_UNIT")" || FLL_PAIR=""
    read -r FLL_CTX FLL_OUT _ <<< "$FLL_PAIR"
    FLL_NAME="$(python3 "$REPO_DIR/oc-point-default.py" --label flash-local "${FLL_WINDOW:-262144}" || true)"
    FLL_PORT="$({ grep -m1 -oE 'PROXY_PORT=[0-9]+|keepalive-proxy.py [0-9]+' /etc/systemd/system/qwen38-keepalive.service 2>/dev/null || true; } | grep -oE '[0-9]+$' || true)"
    FLL_SRC="$(mktemp)"
    if [ -n "$FLL_PAIR" ] && python3 "$REPO_DIR/oc-flash-local.py" "http://127.0.0.1:${FLL_PORT:-30001}/v1" "$CONFIG_DIR/api-key" \
         "$FLL_CTX" "$FLL_OUT" "${FLL_NAME:-Qwen3.8-Flash-Next FP4-Local (local, 262K)}" > "$FLL_SRC"; then
      # The compaction block is global: sized from this lane's window as every text switch sizes
      # it from its own. pi's keepRecentTokens follows (pi-gen).
      FLL_KEEP="$("$REPO_DIR/oc-limits.sh" --preserve "$FLL_CTX")" || FLL_KEEP=""
      for OC_JSON in "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json"; do
        [ -f "$OC_JSON" ] || continue
        python3 "$REPO_DIR/oc-merge-limits.py" "$OC_JSON" --add-providers "$FLL_SRC" >/dev/null || true
        # and an entry already there takes the pair of the unit's window: --add-providers adds a missing provider only,
        # and a Load to 1M kept 205,000 under a compaction block sized for 840,000 (second review of v1.45.2)
        python3 "$REPO_DIR/oc-merge-limits.py" "$OC_JSON" flash-local qwen3.8-flash-next-fp4 "$FLL_CTX" "$FLL_OUT" >/dev/null \
          || echo "NOTE: could not update the Qwen3.8 Flash-Next FP4-Local limits in $OC_JSON; check them by hand ($FLL_PAIR)"
        # --clients: the limits and the entry's name for its window, alone. The boot stays on another lane, and so do
        # opencode's default model and its compaction, sized for the lane it boots with (independent review of v1.48.1)
        if [ "${CLIENTS_ONLY:-0}" -eq 1 ]; then
          python3 "$REPO_DIR/oc-point-default.py" --name-only "$OC_JSON" flash-local flash-local "${FLL_WINDOW:-262144}" \
            || echo "NOTE: could not update $OC_JSON (hand-edited?); the FP4-Local entry keeps its name"
          continue
        fi
        python3 "$REPO_DIR/oc-point-default.py" "$OC_JSON" flash-local flash-local "${FLL_WINDOW:-262144}" \
          || echo "NOTE: could not update $OC_JSON (hand-edited?); set its \"model\" field yourself"
        [ -n "$FLL_KEEP" ] && { python3 "$REPO_DIR/oc-merge-limits.py" "$OC_JSON" --compaction "$FLL_KEEP" \
          || echo "NOTE: could not update the compaction block in $OC_JSON; check it by hand ($FLL_KEEP)"; }
      done
    fi
    rm -f "$FLL_SRC"
    if [ -d "$CONFIG_DIR/pi" ] || [ -d "$CONFIG_DIR/omp" ]; then
      python3 "$REPO_DIR/pi-gen.py" --config "$CONFIG_DIR/opencode.json" --out "$CONFIG_DIR" \
        || echo "NOTE: could not regenerate the pi/omp artifacts; run ./pi-gen.py by hand"
    fi
    if [ "$FLL_OC_SUM" != "$({ cat "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json" 2>/dev/null || true; } | sha256sum)" ] \
       && systemctl list-unit-files opencode-web.service >/dev/null 2>&1 && systemctl is-active --quiet opencode-web.service; then
      sudo systemctl restart opencode-web.service \
        && echo "opencode-web.service restarted so it reads the new default" \
        || echo "NOTE: could not restart opencode-web.service; restart it by hand"
    fi
  fi
  echo "Clients reach it through the proxy on :30001 with the box's key, as model qwen3.8-flash-next-fp4."
  [ "$CLIENTS_ONLY" -eq 0 ] || { echo "--clients: the clients of $FLASH_LOCAL_UNIT_NAME, for the window of the unit in place; nothing else was changed"; exit 0; }
  echo "Switch back:      ./switch-model.sh flash   (or any other target)"
  exit 0
fi

# ── The Ornith 1.5 122B lane: Ornith AI's mixture of experts, on vLLM ─────────────────────
# A text lane for the proxy, served by vLLM with b12x in the Qwen3.8 Flash-Next FP4-Local lane's image, under the
# same contract: verify the checkpoint (the first switch fetches it), put the unit for it in place,
# make it the one unit enabled at boot, never start or stop anything, print the exact commands. The
# unit is rendered by the lane's installer (install-ornith-122b.sh --stage, no sudo): one renderer,
# so a switch after the pins moved, or after the cache moved, puts in the unit an install would.
ORNITH_122B_UNIT="/etc/systemd/system/qwen38-ornith-122b.service"
ORNITH_122B_UNIT_NAME="qwen38-ornith-122b.service"
# Rewritten with sudo, so staged at a fixed path the cockpit's exact-argv sudoers allowlist names,
# as the other lanes' units are.
ORN_STAGE="$CONFIG_DIR/qwen38-ornith-122b.service.switch-stage"
if [ "$CHOICE" = "ornith-122b" ]; then
  [ -f "$ORNITH_122B_UNIT" ] || die "the Ornith 1.5 122B lane is not installed on this box (no $ORNITH_122B_UNIT). Install it once: ./install.sh --with-ornith-122b (about 96 GB)"
  # The checkpoint, its pinned revision and the size, from the lane's installer, and the image from
  # the Qwen3.8 Flash-Next FP4-Local lane's, the one place each is declared.
  eval "$(grep -E '^ORNITH_122B_(REPO|REV|GB)=' "$REPO_DIR/install-ornith-122b.sh" || true)"
  eval "$(grep -E '^FLASH_LOCAL_IMAGE=' "$REPO_DIR/install-flash-local.sh" || true)"
  for _v in ORNITH_122B_REPO ORNITH_122B_REV ORNITH_122B_GB FLASH_LOCAL_IMAGE; do
    eval "[ -n \"\${$_v:-}\" ]" || die "install-ornith-122b.sh or install-flash-local.sh no longer defines $_v (repo layout changed?)"
  done
  unset _v
  docker image inspect "$FLASH_LOCAL_IMAGE" >/dev/null 2>&1 \
    || die "the Ornith 1.5 122B lane's image (${FLASH_LOCAL_IMAGE:0:40}..., the Qwen3.8 Flash-Next FP4-Local lane's) is missing: ./prepare-lane.sh ornith-122b pulls it (no sudo; its first Load from the cockpit runs it), then switch again"
  ORN_HF="$(unit_hf_cache "$ORNITH_122B_UNIT")"
  ORN_HF="${ORN_HF:-$HF_CACHE}"
  printf '\n\033[1;36m── Verifying %s @ %s in the cache (resumable)\033[0m\n' "$ORNITH_122B_REPO" "${ORNITH_122B_REV:0:12}"
  ORN_TOKEN_ARGS=()
  # by name: docker takes the value from its own environment, so the token is in no argv
  if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; ORN_TOKEN_ARGS=(-e HF_TOKEN); fi
  ORN_PROGRESS_ARGS=()
  if [ -f "$REPO_DIR/hf-progress.py" ]; then
    ORN_PROGRESS_ARGS=(-v "$REPO_DIR/hf-progress.py:/opt/qwen38/hf-progress.py:ro")
  fi
  # In the lane's own image, as the other vLLM lanes fetch theirs; --init so that a cockpit job
  # stopped past its timeout stops this too. ckpt-fetch.py is the one fetch of every lane's
  # checkpoint (it says why).
  docker run --rm --init --network host --user "$(id -u):$(id -g)" --entrypoint python3 \
    -e HF_HOME=/hf -e HF_HUB_DOWNLOAD_TIMEOUT=30 -e HF_HUB_DISABLE_XET=1 \
    -e HF_HUB_DISABLE_PROGRESS_BARS="$DL_NO_BARS" \
    "${ORN_TOKEN_ARGS[@]}" "${ORN_PROGRESS_ARGS[@]}" \
    -v "$REPO_DIR/ckpt-fetch.py:/opt/qwen38/ckpt-fetch.py:ro" \
    -v "$ORN_HF":/hf "$FLASH_LOCAL_IMAGE" /opt/qwen38/ckpt-fetch.py "$ORNITH_122B_REPO" "$ORNITH_122B_REV" --offline-first \
    --need-gb "$ORNITH_122B_GB" --free-gb "$(dl_free_gb "$ORN_HF")" \
    || die "the Ornith 1.5 122B checkpoint could not be verified or fetched (re-run to resume; set HF_TOKEN if it stalls)"
  fetched "$ORNITH_122B_REPO @ ${ORNITH_122B_REV:0:12}"
  # its unit for this checkpoint
  HF_CACHE="$ORN_HF" bash "$REPO_DIR/install-ornith-122b.sh" --stage "$ORN_STAGE" \
    || { rm -f "$ORN_STAGE"; die "the Ornith 1.5 122B unit could not be prepared (the lines above say why)"; }
  ORN_REWRITTEN=""
  if ! cmp -s "$ORN_STAGE" "$ORNITH_122B_UNIT"; then
    diff "$ORN_STAGE" "$ORNITH_122B_UNIT" || true   # show exactly what changes
    sudo install -m 644 "$ORN_STAGE" "$ORNITH_122B_UNIT"
    unit_pending_clear "$ORNITH_122B_UNIT"   # its new unit is in place: an update that waited for this Load is applied
    ORN_REWRITTEN=1
  fi
  rm -f "$ORN_STAGE"
  # Which lane this box served at boot, written down before it is disabled: removing this lane
  # later gives the boot back to it (its installer's --uninstall).
  for TEXT_UNIT_NAME in qwen38-sglang.service qwen38-flash.service; do
    if systemctl is-enabled --quiet "$TEXT_UNIT_NAME" 2>/dev/null; then
      text_lane_note "$TEXT_UNIT_NAME"; break   # both enabled: the 27B, as install.sh follows it
    fi
  done
  # The loop variable ends in UNIT_NAME on purpose: CI checks every
  # `sudo systemctl <verb> "$..UNIT_NAME"` against the cockpit's sudoers allowlist.
  for TEXT_UNIT_NAME in qwen38-sglang.service qwen38-flash.service qwen38-image.service qwen38-video.service qwen38-kolibri.service qwen38-nemotron.service qwen38-nemotron-super.service qwen38-flash-local.service qwen38-flash-unsloth.service qwen38-image-h3.service qwen38-transcribe.service qwen38-lfm25.service qwen38-lfm3.service; do
    if systemctl is-enabled --quiet "$TEXT_UNIT_NAME" 2>/dev/null; then
      echo "disabling $TEXT_UNIT_NAME at boot (unit file kept for switching back)"
      sudo systemctl disable "$TEXT_UNIT_NAME"
    fi
  done
  sudo systemctl enable "$ORNITH_122B_UNIT_NAME" >/dev/null 2>&1 || sudo systemctl enable "$ORNITH_122B_UNIT_NAME"
  sudo systemctl daemon-reload
  RUNNING=""
  for u in qwen38-sglang.service qwen38-flash.service qwen38-image.service qwen38-video.service qwen38-kolibri.service qwen38-nemotron.service qwen38-nemotron-super.service qwen38-flash-local.service qwen38-flash-unsloth.service qwen38-image-h3.service qwen38-transcribe.service qwen38-lfm25.service qwen38-lfm3.service; do
    systemctl is-active --quiet "$u" 2>/dev/null && RUNNING="$u"
  done
  printf '\n\033[1;32mSwitch queued: %s (%s)\033[0m\n' "$ORNITH_122B_REPO" "$ORNITH_122B_UNIT_NAME"
  if [ -n "$RUNNING" ]; then
    echo "Effective after:  sudo systemctl stop $RUNNING && sudo systemctl start $ORNITH_122B_UNIT_NAME   (or next reboot)"
  elif systemctl is-active --quiet "$ORNITH_122B_UNIT_NAME" 2>/dev/null; then
    if [ -n "$ORN_REWRITTEN" ]; then
      echo "Effective after:  sudo systemctl restart $ORNITH_122B_UNIT_NAME   (it serves the previous unit until then)"
    else
      echo "Its unit already named this checkpoint. Started before it did, the lane serves what it loaded: sudo systemctl restart $ORNITH_122B_UNIT_NAME"
    fi
  else
    echo "Effective after:  sudo systemctl start $ORNITH_122B_UNIT_NAME   (or next reboot; ready in about 7 min)"
  fi
  # opencode follows the lane, as on a text switch: the lane's provider becomes the default of the
  # generated config and of the one opencode reads, and the server that reads them restarts when
  # they changed. A box installed with --no-opencode is left alone.
  if [ -f "$CONFIG_DIR/opencode.off" ]; then
    echo "opencode integration is off on this box (./install.sh --no-opencode): default model left alone"
  else
    ORN_OC_SUM="$({ cat "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json" 2>/dev/null || true; } | sha256sum)"
    ORN_PAIR="$("$REPO_DIR/oc-limits.sh" ornith-122b)" || ORN_PAIR=""
    ORN_NAME="$(python3 "$REPO_DIR/oc-point-default.py" --label ornith-122b 1048576 || true)"
    ORN_PORT="$({ grep -m1 -oE 'PROXY_PORT=[0-9]+|keepalive-proxy.py [0-9]+' /etc/systemd/system/qwen38-keepalive.service 2>/dev/null || true; } | grep -oE '[0-9]+$' || true)"
    ORN_SRC="$(mktemp)"
    if [ -n "$ORN_PAIR" ] && python3 "$REPO_DIR/oc-ornith.py" "http://127.0.0.1:${ORN_PORT:-30001}/v1" "$CONFIG_DIR/api-key" \
         "${ORN_PAIR%% *}" "$(echo "$ORN_PAIR" | cut -d' ' -f2)" "${ORN_NAME:-Ornith 1.5 122B (local, 1M)}" > "$ORN_SRC"; then
      # The compaction block is global: sized from this lane's window as every text switch sizes
      # it from its own. pi's keepRecentTokens follows (pi-gen).
      ORN_KEEP="$("$REPO_DIR/oc-limits.sh" --preserve "${ORN_PAIR%% *}")" || ORN_KEEP=""
      for OC_JSON in "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json"; do
        [ -f "$OC_JSON" ] || continue
        python3 "$REPO_DIR/oc-merge-limits.py" "$OC_JSON" --add-providers "$ORN_SRC" >/dev/null || true
        python3 "$REPO_DIR/oc-point-default.py" "$OC_JSON" ornith-122b ornith-122b 1048576 \
          || echo "NOTE: could not update $OC_JSON (hand-edited?); set its \"model\" field yourself"
        [ -n "$ORN_KEEP" ] && { python3 "$REPO_DIR/oc-merge-limits.py" "$OC_JSON" --compaction "$ORN_KEEP" \
          || echo "NOTE: could not update the compaction block in $OC_JSON; check it by hand ($ORN_KEEP)"; }
      done
    fi
    rm -f "$ORN_SRC"
    if [ -d "$CONFIG_DIR/pi" ] || [ -d "$CONFIG_DIR/omp" ]; then
      python3 "$REPO_DIR/pi-gen.py" --config "$CONFIG_DIR/opencode.json" --out "$CONFIG_DIR" \
        || echo "NOTE: could not regenerate the pi/omp artifacts; run ./pi-gen.py by hand"
    fi
    if [ "$ORN_OC_SUM" != "$({ cat "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json" 2>/dev/null || true; } | sha256sum)" ] \
       && systemctl list-unit-files opencode-web.service >/dev/null 2>&1 && systemctl is-active --quiet opencode-web.service; then
      sudo systemctl restart opencode-web.service \
        && echo "opencode-web.service restarted so it reads the new default" \
        || echo "NOTE: could not restart opencode-web.service; restart it by hand"
    fi
  fi
  echo "Clients reach it through the proxy on :30001 with the box's key, as model ornith-1.5-122b."
  echo "Switch back:      ./switch-model.sh stock   (or any other target)"
  exit 0
fi

# ── The Qwen3.8 Flash-Next Unsloth GGUF lane: Unsloth's GGUF of the flash model, on llama.cpp ───────────────
# A text lane for the proxy, served by llama.cpp's own server, under the vLLM lanes' contract: verify its files (the
# first switch fetches them), put the unit for them in place, make it the one unit enabled at boot, never start or
# stop anything, print the exact commands. The files and the unit come from the lane's installer, with no sudo
# (install-flash-unsloth.sh --fetch, then --stage): its GGUF files are pinned there one by one and fetched with
# gguf-fetch.py, its image having no Python; one renderer, so a switch after the pins moved, or after the cache
# moved, puts in the unit an install would. Its two targets, UD-Q4_K_XL (flash-unsloth) and UD-IQ3_XXS
# (flash-unsloth-iq3), are one lane under one served name, as the Kolibri lane's two checkpoints are: the target is
# handed to the installer (FLASH_UNSLOTH_TARGET), which fetches its files and renders its forms.
FLASH_UNSLOTH_UNIT="/etc/systemd/system/qwen38-flash-unsloth.service"
FLASH_UNSLOTH_UNIT_NAME="qwen38-flash-unsloth.service"
# Rewritten with sudo, so staged at a fixed path the cockpit's exact-argv sudoers allowlist names,
# as the other lanes' units are.
FLU_STAGE="$CONFIG_DIR/qwen38-flash-unsloth.service.switch-stage"
if [ "$CHOICE" = "flash-unsloth" ] || [ "$CHOICE" = "flash-unsloth-iq3" ]; then
  [ -f "$FLASH_UNSLOTH_UNIT" ] || die "the Qwen3.8 Flash-Next Unsloth GGUF lane is not installed on this box (no $FLASH_UNSLOTH_UNIT). Install it once: ./install.sh --with-flash-unsloth (about 90 GB for the UD-IQ3_XXS default, 83 more for UD-Q4_K_XL's)"
  # the fetch, the unit and the boot: --clients goes straight to the clients, for the unit in place
  if [ "$CLIENTS_ONLY" -eq 0 ]; then
    # The model, its pinned revision and the image, from the lane's installer, the one place they are declared.
    eval "$(grep -E '^FLASH_UNSLOTH_(REPO|REV|IMAGE)=' "$REPO_DIR/install-flash-unsloth.sh" || true)"
    for _v in FLASH_UNSLOTH_REPO FLASH_UNSLOTH_REV FLASH_UNSLOTH_IMAGE; do
      eval "[ -n \"\${$_v:-}\" ]" || die "install-flash-unsloth.sh no longer defines $_v (repo layout changed?)"
    done
    unset _v
    FLU_HF="$(unit_hf_cache "$FLASH_UNSLOTH_UNIT")"
    FLU_HF="${FLU_HF:-$HF_CACHE}"
    printf '\n\033[1;36m── Verifying %s @ %s and its draft in the cache (resumable)\033[0m\n' "$FLASH_UNSLOTH_REPO" "${FLASH_UNSLOTH_REV:0:12}"
    # As the user, by the lane's installer: the files pinned there, the room checked before a byte comes, the token
    # read from the environment (gguf-fetch.py), so it is in no argv. The image is not needed for it, so a --fetch
    # of the Library works before the lane's first Load pulled it.
    HF_CACHE="$FLU_HF" FLASH_UNSLOTH_TARGET="$CHOICE" bash "$REPO_DIR/install-flash-unsloth.sh" --fetch \
      || die "the Qwen3.8 Flash-Next Unsloth GGUF files could not be verified or fetched (re-run to resume; set HF_TOKEN if it stalls)"
    fetched "$FLASH_UNSLOTH_REPO @ ${FLASH_UNSLOTH_REV:0:12}"
    docker image inspect "$FLASH_UNSLOTH_IMAGE" >/dev/null 2>&1 \
      || die "the Qwen3.8 Flash-Next Unsloth GGUF lane's image (${FLASH_UNSLOTH_IMAGE:0:40}...) is missing: ./prepare-lane.sh flash-unsloth pulls it (no sudo; its first Load from the cockpit runs it), then switch again"
    # its chat template, and its unit for these files
    HF_CACHE="$FLU_HF" FLASH_UNSLOTH_TARGET="$CHOICE" bash "$REPO_DIR/install-flash-unsloth.sh" --stage "$FLU_STAGE" \
      || { rm -f "$FLU_STAGE"; die "the Qwen3.8 Flash-Next Unsloth GGUF unit could not be prepared (the lines above say why)"; }
    FLU_REWRITTEN=""
    if ! cmp -s "$FLU_STAGE" "$FLASH_UNSLOTH_UNIT"; then
      diff "$FLU_STAGE" "$FLASH_UNSLOTH_UNIT" || true   # show exactly what changes
      sudo install -m 644 "$FLU_STAGE" "$FLASH_UNSLOTH_UNIT"
      unit_pending_clear "$FLASH_UNSLOTH_UNIT"   # its new unit is in place: an update that waited for this Load is applied
      FLU_REWRITTEN=1
    fi
    rm -f "$FLU_STAGE"
    # Which lane this box served at boot, written down before it is disabled: removing this lane
    # later gives the boot back to it (its installer's --uninstall).
    for TEXT_UNIT_NAME in qwen38-sglang.service qwen38-flash.service; do
      if systemctl is-enabled --quiet "$TEXT_UNIT_NAME" 2>/dev/null; then
        text_lane_note "$TEXT_UNIT_NAME"; break   # both enabled: the 27B, as install.sh follows it
      fi
    done
    # The loop variable ends in UNIT_NAME on purpose: CI checks every
    # `sudo systemctl <verb> "$..UNIT_NAME"` against the cockpit's sudoers allowlist.
    for TEXT_UNIT_NAME in qwen38-sglang.service qwen38-flash.service qwen38-image.service qwen38-video.service qwen38-kolibri.service qwen38-nemotron.service qwen38-nemotron-super.service qwen38-flash-local.service qwen38-ornith-122b.service qwen38-image-h3.service qwen38-transcribe.service qwen38-lfm25.service qwen38-lfm3.service; do
      if systemctl is-enabled --quiet "$TEXT_UNIT_NAME" 2>/dev/null; then
        echo "disabling $TEXT_UNIT_NAME at boot (unit file kept for switching back)"
        sudo systemctl disable "$TEXT_UNIT_NAME"
      fi
    done
    sudo systemctl enable "$FLASH_UNSLOTH_UNIT_NAME" >/dev/null 2>&1 || sudo systemctl enable "$FLASH_UNSLOTH_UNIT_NAME"
    sudo systemctl daemon-reload
    RUNNING=""
    for u in qwen38-sglang.service qwen38-flash.service qwen38-image.service qwen38-video.service qwen38-kolibri.service qwen38-nemotron.service qwen38-nemotron-super.service qwen38-flash-local.service qwen38-ornith-122b.service qwen38-image-h3.service qwen38-transcribe.service qwen38-lfm25.service qwen38-lfm3.service; do
      systemctl is-active --quiet "$u" 2>/dev/null && RUNNING="$u"
    done
    printf '\n\033[1;32mSwitch queued: %s, %s (%s)\033[0m\n' "$FLASH_UNSLOTH_REPO" "$CHOICE" "$FLASH_UNSLOTH_UNIT_NAME"
    # The target is the operator's choice from here, as the Kolibri lane's checkpoint (v1.49.0): recorded in an
    # installed.json that exists, erased when it is the recommended one, so an update moves no chosen UD-Q4_K_XL.
    python3 "$REPO_DIR/conformity.py" choose --config-dir "$CONFIG_DIR" --repo-dir "$REPO_DIR" \
      FLASH_UNSLOTH_TARGET "$CHOICE" || true
    if [ -n "$RUNNING" ]; then
      echo "Effective after:  sudo systemctl stop $RUNNING && sudo systemctl start $FLASH_UNSLOTH_UNIT_NAME   (or next reboot)"
    elif systemctl is-active --quiet "$FLASH_UNSLOTH_UNIT_NAME" 2>/dev/null; then
      if [ -n "$FLU_REWRITTEN" ]; then
        echo "Effective after:  sudo systemctl restart $FLASH_UNSLOTH_UNIT_NAME   (it serves the previous unit until then)"
      else
        echo "Its unit already named these files. Started before it did, the lane serves what it loaded: sudo systemctl restart $FLASH_UNSLOTH_UNIT_NAME"
      fi
    else
      echo "Effective after:  sudo systemctl start $FLASH_UNSLOTH_UNIT_NAME   (or next reboot; ready in about 4 to 7 min)"
    fi
  fi
  # opencode follows the lane, as on a text switch: the lane's provider becomes the default of the
  # generated config and of the one opencode reads, and the server that reads them restarts when
  # they changed. A box installed with --no-opencode is left alone.
  if [ -f "$CONFIG_DIR/opencode.off" ]; then
    echo "opencode integration is off on this box (./install.sh --no-opencode): default model left alone"
  else
    FLU_OC_SUM="$({ cat "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json" 2>/dev/null || true; } | sha256sum)"
    # the window of the unit just staged (its -c, a window for each request), and its pair: 262,144 by default,
    # 1,048,576 with FLASH_UNSLOTH_CONTEXT=1m, the choice its installer kept (decisions 1316 and 1318); and the target it
    # serves, which the entry's name says (lanes.sh): the unit in place's with --clients, the one just staged otherwise
    FLU_WINDOW="$(llamacpp_window "$FLASH_UNSLOTH_UNIT")"
    FLU_TARGET="$(flash_unsloth_target "$FLASH_UNSLOTH_UNIT" "$REPO_DIR/install-flash-unsloth.sh")"
    FLU_TARGET="${FLU_TARGET:-$CHOICE}"
    FLU_PAIR="$("$REPO_DIR/oc-limits.sh" "$FLU_TARGET" --from "$FLASH_UNSLOTH_UNIT")" || FLU_PAIR=""
    read -r FLU_CTX FLU_OUT _ <<< "$FLU_PAIR"
    FLU_NAME="$(python3 "$REPO_DIR/oc-point-default.py" --label "$FLU_TARGET" "${FLU_WINDOW:-262144}" || true)"
    FLU_PORT="$({ grep -m1 -oE 'PROXY_PORT=[0-9]+|keepalive-proxy.py [0-9]+' /etc/systemd/system/qwen38-keepalive.service 2>/dev/null || true; } | grep -oE '[0-9]+$' || true)"
    FLU_SRC="$(mktemp)"
    if [ -n "$FLU_PAIR" ] && python3 "$REPO_DIR/oc-flash-unsloth.py" "http://127.0.0.1:${FLU_PORT:-30001}/v1" "$CONFIG_DIR/api-key" \
         "$FLU_CTX" "$FLU_OUT" "${FLU_NAME:-Qwen3.8-Flash-Next Unsloth GGUF + DFlash2 (local)}" > "$FLU_SRC"; then
      # The compaction block is global: sized from this lane's window as every text switch sizes
      # it from its own. pi's keepRecentTokens follows (pi-gen).
      FLU_KEEP="$("$REPO_DIR/oc-limits.sh" --preserve "$FLU_CTX")" || FLU_KEEP=""
      for OC_JSON in "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json"; do
        [ -f "$OC_JSON" ] || continue
        python3 "$REPO_DIR/oc-merge-limits.py" "$OC_JSON" --add-providers "$FLU_SRC" >/dev/null || true
        # and an entry already there takes the pair of the unit's window: --add-providers adds a missing provider only
        # (the FP4-Local lane's rule, second review of v1.45.2)
        python3 "$REPO_DIR/oc-merge-limits.py" "$OC_JSON" flash-unsloth qwen3.8-flash-next-unsloth "$FLU_CTX" "$FLU_OUT" >/dev/null \
          || echo "NOTE: could not update the Qwen3.8 Flash-Next Unsloth GGUF limits in $OC_JSON; check them by hand ($FLU_PAIR)"
        # --clients: the limits and the entry's name for its window, alone, as for FP4-Local
        if [ "${CLIENTS_ONLY:-0}" -eq 1 ]; then
          python3 "$REPO_DIR/oc-point-default.py" --name-only "$OC_JSON" flash-unsloth "$FLU_TARGET" "${FLU_WINDOW:-262144}" \
            || echo "NOTE: could not update $OC_JSON (hand-edited?); the Unsloth GGUF entry keeps its name"
          continue
        fi
        python3 "$REPO_DIR/oc-point-default.py" "$OC_JSON" flash-unsloth "$FLU_TARGET" "${FLU_WINDOW:-262144}" \
          || echo "NOTE: could not update $OC_JSON (hand-edited?); set its \"model\" field yourself"
        [ -n "$FLU_KEEP" ] && { python3 "$REPO_DIR/oc-merge-limits.py" "$OC_JSON" --compaction "$FLU_KEEP" \
          || echo "NOTE: could not update the compaction block in $OC_JSON; check it by hand ($FLU_KEEP)"; }
      done
    fi
    rm -f "$FLU_SRC"
    if [ -d "$CONFIG_DIR/pi" ] || [ -d "$CONFIG_DIR/omp" ]; then
      python3 "$REPO_DIR/pi-gen.py" --config "$CONFIG_DIR/opencode.json" --out "$CONFIG_DIR" \
        || echo "NOTE: could not regenerate the pi/omp artifacts; run ./pi-gen.py by hand"
    fi
    if [ "$FLU_OC_SUM" != "$({ cat "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json" 2>/dev/null || true; } | sha256sum)" ] \
       && systemctl list-unit-files opencode-web.service >/dev/null 2>&1 && systemctl is-active --quiet opencode-web.service; then
      sudo systemctl restart opencode-web.service \
        && echo "opencode-web.service restarted so it reads the new default" \
        || echo "NOTE: could not restart opencode-web.service; restart it by hand"
    fi
  fi
  echo "Clients reach it through the proxy on :30001 with the box's key, as model qwen3.8-flash-next-unsloth."
  [ "$CLIENTS_ONLY" -eq 0 ] || { echo "--clients: the clients of $FLASH_UNSLOTH_UNIT_NAME, for the window of the unit in place; nothing else was changed"; exit 0; }
  echo "Switch back:      ./switch-model.sh flash   (or any other target)"
  exit 0
fi

# ── The image lane: a third lane, switched to the same way as the other two ─────
# Same contract as a cross-lane switch: verify the checkpoint, make it the one unit
# enabled at boot, never start or stop anything, print the exact commands. What it
# does NOT share with the text lanes is everything that is about text: there is no
# serving image to inspect (it runs from its own venv), nothing in the unit to rewrite but
# its --model-path (between the lane's checkpoints), no prompt ceiling on the proxy (the
# proxy is a text door and nothing reaches this lane through it), and no opencode default
# to move (opencode is a text client, and pointing it at an image lane would break every
# session it opens). So it is handled here, whole, and exits, instead of threading
# "unless image" through four hundred lines written for SGLang's text server.
if [ "$CHOICE" = "image" ] || [ "$CHOICE" = "image-turbo" ]; then
  [ -f "$IMAGE_UNIT" ] || die "the image lane is not installed on this box (no $IMAGE_UNIT). Install it once: ./install.sh --with-image (46 GB)"
  IMG_WD="$(grep -m1 -E '^WorkingDirectory=' "$IMAGE_UNIT" | cut -d= -f2- || true)"
  IMG_PY="${IMG_WD:-$HOME/.local/share/qwen38-image}/venv/bin/python"
  [ -x "$IMG_PY" ] || die "the image lane's runtime is missing ($IMG_PY): ./prepare-lane.sh image builds it (no sudo; its first Load from the cockpit runs it), then switch again"
  # The lane's two checkpoints and their pinned revisions, from install-image.sh, the one
  # place they are declared.
  eval "$(grep -E '^(IMAGE_MODEL_PIN|IMAGE_MODEL_PIN_REV|IMAGE_TURBO_PIN|IMAGE_TURBO_PIN_REV)=' "$REPO_DIR/install-image.sh" || true)"
  for _v in IMAGE_MODEL_PIN IMAGE_MODEL_PIN_REV IMAGE_TURBO_PIN IMAGE_TURBO_PIN_REV; do
    eval "[ -n \"\${$_v:-}\" ]" || die "install-image.sh no longer defines $_v (repo layout changed?)"
  done
  unset _v
  # An update left this lane's new unit waiting for this Load (lanes.sh's unit_put, decision 1408): what this
  # switch fetches is what THAT unit reads, not what the one in place reads (review of v1.52.0, 2026-11-19,
  # GLM 5.3: this branch read the wait nowhere, so a Load whose pin had moved fetched the old pin, posed
  # nothing, cleared nothing, and the card's Load looped on a restart that changed nothing).
  IMG_SRC="$IMAGE_UNIT"
  if [ -f "$CONFIG_DIR/pending/$IMAGE_UNIT_NAME" ]; then
    IMG_SRC="$CONFIG_DIR/pending/$IMAGE_UNIT_NAME"
    echo "the image lane's update waits for this Load: fetching what its new unit reads"
  fi
  IMG_CUR="$(grep -oE -- '--model-path [^ ]+' "$IMAGE_UNIT" | head -1 | cut -d' ' -f2 || true)"
  IMG_CUR="${IMG_CUR:-$IMAGE_MODEL_PIN}"
  IMG_SERVED="$(grep -oE -- '--model-path [^ ]+' "$IMG_SRC" | head -1 | cut -d' ' -f2 || true)"
  # image-turbo serves the Turbo; image serves the base, or the checkpoint the unit was
  # installed with (IMAGE_MODEL=) when it is neither, written down on the way to the Turbo
  # so that the way back finds it.
  IMG_BEFORE_TURBO="$CONFIG_DIR/image-model-before-turbo"
  if [ "$CHOICE" = "image-turbo" ]; then
    IMG_MODEL="$IMAGE_TURBO_PIN"
  elif [ "$IMG_CUR" = "$IMAGE_TURBO_PIN" ]; then
    IMG_MODEL="$(cat "$IMG_BEFORE_TURBO" 2>/dev/null || true)"; IMG_MODEL="${IMG_MODEL:-$IMAGE_MODEL_PIN}"
  else
    IMG_MODEL="${IMG_SERVED:-$IMG_CUR}"
  fi
  # The revision this box serves of that checkpoint, as the video lane reads it: the unit
  # names the repo with HF_HUB_OFFLINE=1, so its refs/main IS the served commit, and one an
  # install made with IMAGE_MODEL_REV= stays served. The installer's pin answers when the ref
  # is missing or holds no commit, and main for a checkpoint of the unit's own.
  # the cache read from the same unit the fetch is for (IMG_SRC, the pending one when an update waits), not
  # from the unit in place: the fetch verifies and fills the cache the waiting unit reads (sixth review of
  # v1.52.0, 2026-11-19, GLM 5.3: it read the one in place, and a moved cache looped the wait forever)
  IMG_HF="$(grep -m1 -E '^Environment=HF_HOME=' "$IMG_SRC" | cut -d= -f3- || true)"
  IMG_MODEL_DIR="${IMG_HF:-$HF_CACHE}/hub/models--${IMG_MODEL//\//--}"
  IMG_REV=""
  if [ -f "$IMG_MODEL_DIR/refs/main" ]; then IMG_REV="$(cat "$IMG_MODEL_DIR/refs/main")"; fi
  [[ "$IMG_REV" =~ ^[0-9a-f]{40}$ ]] || IMG_REV=""
  if [ -z "$IMG_REV" ]; then
    case "$IMG_MODEL" in
      "$IMAGE_MODEL_PIN") IMG_REV="$IMAGE_MODEL_PIN_REV" ;;
      "$IMAGE_TURBO_PIN") IMG_REV="$IMAGE_TURBO_PIN_REV" ;;
      *) IMG_REV=main ;;
    esac
  fi
  if [ "$IMG_MODEL" = "$IMAGE_TURBO_PIN" ]; then
    # The Turbo samples on the eight-step grid its model_index.json carries, which only a
    # runtime with sgl-project/sglang#43391 follows; one without it loads the weights and
    # samples on a uniform schedule instead, answering 200 with worse images.
    IMG_WHY="$("$IMG_PY" "$REPO_DIR/image-sglang/turbo-grid-check.py" 2>&1)" \
      || die "the image lane's runtime cannot sample on the Turbo's eight-step grid (${IMG_WHY:-no reason given}). Update it first: ./install.sh (or ./install-image.sh), then switch again."
  fi
  printf '\n\033[1;36m── Verifying %s @ %s in the cache (resumable)\033[0m\n' "$IMG_MODEL" "$IMG_REV"
  # The same two lessons the text lanes learned: Xet stalls on this box, and an
  # unauthenticated pull gets throttled. ckpt-fetch.py, the one fetch of every lane: the
  # cache first, so a complete one answers without touching the network at all; both
  # checkpoints of this lane are about 33 GB (33.13 and 32.49: --need-gb 34 is the larger, rounded
  # up), refused before the first byte when the room is not there. A revision is fetched as such
  # and refs/main pointed at it, because the unit runs offline by name, which resolves refs/main:
  # fetched at main, a push upstream
  # would have changed what the lane serves (the same lesson as install-image.sh, 2026-09-24).
  HF_HOME="${IMG_HF:-$HF_CACHE}" HF_HUB_DOWNLOAD_TIMEOUT=30 HF_HUB_DISABLE_XET=1 \
    HF_HUB_DISABLE_PROGRESS_BARS="$DL_NO_BARS" \
    "$IMG_PY" "$REPO_DIR/ckpt-fetch.py" "$IMG_MODEL" "$IMG_REV" --offline-first \
    --need-gb 34 --free-gb "$(dl_free_gb "${IMG_HF:-$HF_CACHE}")" --refs-main pin \
    || die "the image checkpoint could not be verified or fetched (re-run to resume; set HF_TOKEN if it stalls)"
  fetched "$IMG_MODEL @ ${IMG_REV:0:12}"
  # The unit names one checkpoint; a switch between the lane's two rewrites that name and
  # nothing else. The update that waited for this Load (if any) goes in place whole instead: this
  # switch just fetched what its new unit reads, and the block is the video and MiniMax-H3-Image
  # lanes' (review of v1.52.0, 2026-11-19, GLM 5.3: it was never posed nor cleared here).
  IMG_REWRITTEN=""
  # The update that waited for this Load (if any) is posed whole first when it serves the asked checkpoint
  # (the fetch above read what it reads, IMG_SRC): this switch just fetched it. When it serves the lane's
  # OTHER one, it is posed whole and then its --model-path is rewritten to the asked one: the asked target
  # is the one served, never the reverse, and the checkpoint fetched above is the asked one's, so nothing is
  # downloaded for a target that will not serve (third review of v1.52.0, 2026-11-19, GLM 5.3: a Load of
  # image-turbo with a wait for the base posed the wait whole and dropped the Turbo, after its 33 GB had
  # been fetched).
  IMG_POSE=""
  if [ -f "$CONFIG_DIR/pending/$IMAGE_UNIT_NAME" ]; then
    if [ -f "$CONFIG_DIR/pending/$IMAGE_UNIT_NAME.base" ] && [ -f "$IMAGE_UNIT" ] \
       && [ "$(cat "$CONFIG_DIR/pending/$IMAGE_UNIT_NAME.base")" != "$(sha256sum "$IMAGE_UNIT" | cut -d' ' -f1)" ]; then
      unit_pending_clear "$IMAGE_UNIT"
      echo "the update that waited for this Load was dropped: $IMAGE_UNIT_NAME changed since it was rendered; ./install.sh writes it again"
    else
      IMG_POSE="$CONFIG_DIR/pending/$IMAGE_UNIT_NAME"
      IMG_POSE_CUR="$(grep -oE -- '--model-path [^ ]+' "$IMG_POSE" | head -1 | cut -d' ' -f2 || true)"
      if [ -z "$IMG_POSE_CUR" ] || [ "$IMG_POSE_CUR" = "$IMG_MODEL" ]; then
        # it serves the asked checkpoint (or names none): it waits for what it lacks, as before
        if [ -n "$(unit_needs "$IMG_POSE")" ]; then
          echo "NOTE: the update that waited for this Load still lacks $(unit_needs "$IMG_POSE" | paste -sd ';' | sed 's/;/; /g'): it stays waiting; ./install.sh --with-image applies it"
          IMG_POSE=""   # it stays waiting: the unit in place is rewritten to the asked checkpoint below
        fi
      elif [ -n "$(unit_needs "$IMG_POSE" | grep -v "^checkpoint ${IMG_POSE_CUR}@" || true)" ]; then
        # it serves the other checkpoint and still lacks what the rewrite to the asked one does not
        # replace (its runtime, a mounted file): it cannot be posed whole, and is dropped, the asked
        # target served from the unit in place
        echo "NOTE: the update that waited for this Load serves $IMG_POSE_CUR, not the asked $IMG_MODEL, and still lacks $(unit_needs "$IMG_POSE" | grep -v "^checkpoint ${IMG_POSE_CUR}@" | paste -sd ';' | sed 's/;/; /g'): it is dropped; ./install.sh writes it again"
        unit_pending_clear "$IMAGE_UNIT"
        IMG_POSE=""
      fi
    fi
  fi
  IMG_FROM="$IMAGE_UNIT"; IMG_FROM_CUR="$IMG_CUR"
  if [ -n "$IMG_POSE" ]; then IMG_FROM="$IMG_POSE"; IMG_FROM_CUR="$IMG_POSE_CUR"; fi
  # The wait that serves the asked checkpoint but still lacks something (its runtime) is not posed: the unit in
  # place is rewritten to the asked checkpoint anyway, whose checkpoint was just fetched, and the wait is
  # kept, not cleared: the output's "Switch queued" is then true (seventh pass of the v1.52.0 reviews,
  # 2026-11-19: with the stays-branch the unit in place kept the other checkpoint while the output named
  # the asked one).
  if [ -n "$IMG_POSE" ] || [ "$IMG_CUR" != "$IMG_MODEL" ]; then
    # the waiting unit posed whole when there is one, then rewritten to the asked checkpoint, which is the
    # one served, never the reverse
    if [ -n "$IMG_FROM_CUR" ]; then
      IMG_UNIT_TEXT="$(cat "$IMG_FROM")"
      printf '%s\n' "${IMG_UNIT_TEXT//"--model-path $IMG_FROM_CUR "/"--model-path $IMG_MODEL "}" > "$IMG_STAGE"
      if ! grep -q -- "--model-path $IMG_MODEL " "$IMG_STAGE" && ! grep -q -- "--model-path $IMG_MODEL\$" "$IMG_STAGE"; then
        rm -f "$IMG_STAGE"; die "the image unit's rewrite to $IMG_MODEL failed (hand-edited unit?): re-run ./install-image.sh"
      fi
    else
      cp "$IMG_FROM" "$IMG_STAGE"
    fi
    if ! cmp -s "$IMG_STAGE" "$IMAGE_UNIT"; then
      diff "$IMG_STAGE" "$IMAGE_UNIT" || true   # show exactly what changes
      sudo install -m 644 "$IMG_STAGE" "$IMAGE_UNIT"
      [ -n "$IMG_POSE" ] && unit_pending_clear "$IMAGE_UNIT"
      # a wait that stays waiting (IMG_POSE empty) keeps a .base naming the unit from before this
      # rewrite: the next Load or switch would drop it as changed since it was rendered; it names
      # the unit in place now
      [ -f "$CONFIG_DIR/pending/$IMAGE_UNIT_NAME" ] \
        && sha256sum "$IMAGE_UNIT" | cut -d' ' -f1 > "$CONFIG_DIR/pending/$IMAGE_UNIT_NAME.base"
      rm -f "$IMG_STAGE"
      sudo systemctl daemon-reload
      [ -n "$IMG_POSE" ] && echo "the update that waited for this Load is in place: $IMAGE_UNIT_NAME"
    else
      [ -n "$IMG_POSE" ] && unit_pending_clear "$IMAGE_UNIT"   # an unchanged waiting unit is cleared without install
    fi
    IMG_REWRITTEN=1
    if [ "$IMG_MODEL" = "$IMAGE_TURBO_PIN" ] && [ "$IMG_CUR" != "$IMAGE_MODEL_PIN" ]; then
      printf '%s\n' "$IMG_CUR" > "$IMG_BEFORE_TURBO"
    else
      rm -f "$IMG_BEFORE_TURBO"
    fi
  fi
  # Which text lane this box served, written down before it is disabled: install.sh
  # updates that lane on a later run, and enablement cannot tell it once both are off.
  for TEXT_UNIT_NAME in qwen38-sglang.service qwen38-flash.service; do
    if systemctl is-enabled --quiet "$TEXT_UNIT_NAME" 2>/dev/null; then
      text_lane_note "$TEXT_UNIT_NAME"; break   # both enabled: the 27B, as install.sh follows it
    fi
  done
  # The loop variable ends in UNIT_NAME on purpose: CI checks every
  # `sudo systemctl <verb> "$..UNIT_NAME"` against the cockpit's sudoers allowlist,
  # and a variable called $u would have walked straight past it.
  for TEXT_UNIT_NAME in qwen38-sglang.service qwen38-flash.service qwen38-video.service qwen38-kolibri.service qwen38-nemotron.service qwen38-nemotron-super.service qwen38-flash-local.service qwen38-ornith-122b.service qwen38-flash-unsloth.service qwen38-image-h3.service qwen38-transcribe.service qwen38-lfm25.service qwen38-lfm3.service; do
    if systemctl is-enabled --quiet "$TEXT_UNIT_NAME" 2>/dev/null; then
      echo "disabling $TEXT_UNIT_NAME at boot (unit file kept for switching back)"
      sudo systemctl disable "$TEXT_UNIT_NAME"
    fi
  done
  sudo systemctl enable "$IMAGE_UNIT_NAME" >/dev/null 2>&1 || sudo systemctl enable "$IMAGE_UNIT_NAME"
  sudo systemctl daemon-reload
  RUNNING=""
  for u in qwen38-sglang.service qwen38-flash.service qwen38-video.service qwen38-kolibri.service qwen38-nemotron.service qwen38-nemotron-super.service qwen38-flash-local.service qwen38-ornith-122b.service qwen38-flash-unsloth.service qwen38-image-h3.service qwen38-transcribe.service qwen38-lfm25.service qwen38-lfm3.service; do
    systemctl is-active --quiet "$u" 2>/dev/null && RUNNING="$u"
  done
  printf '\n\033[1;32mSwitch queued: %s (%s)\033[0m\n' "$IMG_MODEL" "$IMAGE_UNIT_NAME"
  # Its checkpoint is the operator's choice from here, as Kolibri's (2026-10-17).
  python3 "$REPO_DIR/conformity.py" choose --config-dir "$CONFIG_DIR" --repo-dir "$REPO_DIR" \
    IMAGE_MODEL "$IMG_MODEL" || true
  if [ -n "$RUNNING" ]; then
    echo "Effective after:  sudo systemctl stop $RUNNING && sudo systemctl start $IMAGE_UNIT_NAME   (or next reboot)"
  elif systemctl is-active --quiet "$IMAGE_UNIT_NAME" 2>/dev/null; then
    if [ -n "$IMG_REWRITTEN" ]; then
      echo "Effective after:  sudo systemctl restart $IMAGE_UNIT_NAME   (it serves the previous checkpoint until then)"
    else
      echo "Its unit already named this checkpoint. Started before it did, the lane serves what it loaded: sudo systemctl restart $IMAGE_UNIT_NAME"
    fi
  else
    echo "Effective after:  sudo systemctl start $IMAGE_UNIT_NAME   (or next reboot; ready in about 70 s)"
  fi
  echo "The proxy on :30001 and opencode are text clients and were left as they are."
  echo "Switch back:      ./switch-model.sh stock   (or any text target)"
  exit 0
fi

# ── The video lane: a fourth lane, switched to the same way as the image one ───
# Same contract as the image lane: verify the checkpoint, make it the one unit
# enabled at boot, never start or stop anything, print the exact commands. Like the
# image lane it shares nothing with the text lanes: no serving image to inspect (it
# runs from its own venv), no unit to rewrite (one checkpoint, one variant), no
# prompt ceiling on the proxy (the proxy is a text door and nothing reaches this
# lane through it), and no opencode default to move (opencode is a text client).
# Handled here, whole, and exits.
if [ "$CHOICE" = "video" ]; then
  [ -f "$VIDEO_UNIT" ] || die "the video lane is not installed on this box (no $VIDEO_UNIT). Install it once: ./install.sh --with-video (about 174 GB of headroom)"
  VID_WD="$(grep -m1 -E '^WorkingDirectory=' "$VIDEO_UNIT" | cut -d= -f2- || true)"
  VID_PY="${VID_WD:-$HOME/.local/share/qwen38-video}/venv/bin/python"
  [ -x "$VID_PY" ] || die "the video lane's runtime is missing ($VID_PY): ./prepare-lane.sh video builds it (no sudo; its first Load from the cockpit runs it), then switch again"
  # An update left this lane's new unit waiting for this Load (lanes.sh's unit_put, decision 1408): what this
  # switch fetches is what THAT unit reads, not what the one in place reads.
  VID_SRC="$VIDEO_UNIT"
  if [ -f "$CONFIG_DIR/pending/$VIDEO_UNIT_NAME" ]; then
    VID_SRC="$CONFIG_DIR/pending/$VIDEO_UNIT_NAME"
    echo "the video lane's update waits for this Load: fetching what its new unit reads"
  fi
  VID_MODEL="$(grep -oE -- '--model-path [^ ]+' "$VID_SRC" | head -1 | cut -d' ' -f2 || true)"
  VID_MODEL="${VID_MODEL:-MiniMaxAI/MiniMax-H3}"
  VID_HF="$(grep -m1 -E '^Environment=HF_HOME=' "$VID_SRC" | cut -d= -f3- || true)"
  printf '\n\033[1;36m── Verifying %s in the cache (resumable)\033[0m\n' "$VID_MODEL"
  echo "MiniMax-H3: its licence, the MiniMax H3 Community License, is not granted in the European Union, the United Kingdom, South Korea or the United States (its Excluded Territories: definition 5 and clause V.4 of its LICENSE)."
  # Only the partition the installed unit serves is verified and fetched, by the same
  # allowlist as install-video.sh: a complete fl2va cache answers without touching the
  # network, and without it a switch re-downloaded the whole repo (Ref2VA included)
  # onto a full disk (reference box, 2026-09-26).
  VID_VARIANT="$(grep -oE -- '--model-variant [^ ]+' "$VID_SRC" | head -1 | cut -d' ' -f2 || true)"
  VID_VARIANT="${VID_VARIANT:-fl2va}"
  if [ "$VID_MODEL" = "MiniMaxAI/MiniMax-H3" ]; then
    if [ "$VID_VARIANT" = "fl2va" ]; then VID_ALLOW="FL2VA/* model_index.json modular_model_index.json scheduler/* audio_scheduler/* *.md LICENSE*"
    else VID_ALLOW="Ref2VA/* model_index.json modular_model_index.json scheduler/* audio_scheduler/* *.md LICENSE*"; fi
  else VID_ALLOW="";
  fi
  # Xet stalls on this box, and an unauthenticated pull gets throttled.
  # Fetch the revision this box actually serves, never the upstream main tip: the
  # unit names the repo with HF_HUB_OFFLINE=1, so refs/main IS the served commit.
  # A download without revision resolves main over the network, writes refs/main
  # at the tip BEFORE the first byte is downloaded (huggingface_hub writes the ref
  # before fetching: _snapshot_download.py), and resumes that half-fetched tip on
  # the next boot attempt: a switched-off pin, ~145 GB of another commit, a dead
  # offline unit. refs/main says what to fetch; the installer's pin answers when
  # the ref is missing (a download by sha wrote no ref).
  VID_MODEL_DIR="${VID_HF:-$HF_CACHE}/hub/models--${VID_MODEL//\//--}"
  VID_REV=""
  if [ -f "$VID_MODEL_DIR/refs/main" ]; then VID_REV="$(cat "$VID_MODEL_DIR/refs/main")"; fi
  [[ "$VID_REV" =~ ^[0-9a-f]{40}$ ]] || VID_REV=""
  if [ -z "$VID_REV" ]; then
    if [ "$VID_MODEL" = "MiniMaxAI/MiniMax-H3" ]; then
      VID_REV="$(grep -m1 -oE '^VIDEO_MODEL_PIN_REV="[0-9a-f]{40}"' "$REPO_DIR/install-video.sh" | cut -d'"' -f2 || true)"
    else VID_REV="main"; fi
  fi
  [ -n "$VID_REV" ] || die "no refs/main under $VID_MODEL_DIR and install-video.sh no longer defines its pin: re-run ./install-video.sh, it resumes"
  # Room first, the law the text branch learned on 2026-09-24 and this branch
  # skipped: the cockpit's Load journey runs this under a serving engine, and a
  # 145 GB weights fetch onto a fuller disk is a failure of everything on it
  # (found in the branch's reviews, 2026-09-28). What the cache already holds
  # comes off the need, so a complete cache (this box's normal state) needs only
  # the floor and downloads nothing; the number is install-video.sh's own.
  VID_WANT_GB="$(grep -m1 -oE '^WEIGHTS_GB=[0-9]+' "$REPO_DIR/install-video.sh" | cut -d= -f2 || true)"
  # install-video.sh's number when its line cannot be read: GiB, as it counts them since v1.20.0 (this
  # said 145, the GB it counted before)
  VID_WANT_GB="${VID_WANT_GB:-135}"
  VID_HAVE_B="$({ du -s --apparent-size -B1 "$VID_MODEL_DIR/blobs" 2>/dev/null || true; } | cut -f1)"
  VID_WANT_GB=$((VID_WANT_GB - ${VID_HAVE_B:-0} / 1073741824)); [ "$VID_WANT_GB" -ge 10 ] || VID_WANT_GB=10
  # The two caches the runtime writes at its first start (lanes.sh), with the checkpoint on the
  # disk they share and on their own otherwise: a first Load's preparation counts them with the
  # runtime, before the checkpoint, and this counted the checkpoint alone, so a disk with room
  # for each part passed both and filled up at the first start (found in review, 2026-10-15).
  dev_of(){ stat -L -c %d "$(existing "$1")"; }
  VID_DEV="$(dev_of "${VID_HF:-$HF_CACHE}")"; VID_WHAT="about $VID_WANT_GB GiB for the checkpoint"
  declare -A C_NEED=() C_AT=() C_DIRS=() C_WHAT_ON=()
  while IFS='|' read -r C_DIR C_GB C_WHAT; do
    [ "$C_GB" -gt 0 ] || continue
    C_DEV="$(dev_of "$C_DIR")"
    if [ "$C_DEV" = "$VID_DEV" ]; then
      VID_WANT_GB=$((VID_WANT_GB + C_GB)); VID_WHAT="$VID_WHAT, $C_GB for $C_WHAT"
    else
      # summed on their own disk: each was checked alone there (found in review, 2026-10-15)
      C_NEED[$C_DEV]=$(( ${C_NEED[$C_DEV]:-0} + C_GB )); C_AT[$C_DEV]="${C_AT[$C_DEV]:-$C_DIR}"
      C_DIRS[$C_DEV]="${C_DIRS[$C_DEV]:+${C_DIRS[$C_DEV]} and }$C_DIR"     # each named, not the last alone
      C_WHAT_ON[$C_DEV]="${C_WHAT_ON[$C_DEV]:+${C_WHAT_ON[$C_DEV]}, }$C_GB for $C_WHAT"
    fi
  done < <(video_first_start_caches)
  for C_DEV in "${!C_NEED[@]}"; do
    C_FREE="$(dl_free_gb "${C_AT[$C_DEV]}")"
    [ "${C_FREE:-0}" -ge "${C_NEED[$C_DEV]}" ] \
      || die "not enough room for the video lane's first start under ${C_DIRS[$C_DEV]}: ${C_FREE:-?} GiB free, about ${C_NEED[$C_DEV]} GiB needed (${C_WHAT_ON[$C_DEV]}; nothing was changed). Free some space first."
  done
  VID_FREE_GB="$(dl_free_gb "${VID_HF:-$HF_CACHE}")"
  [ "${VID_FREE_GB:-0}" -ge "$VID_WANT_GB" ] \
    || die "not enough room for $VID_MODEL under HF_HOME=${VID_HF:-$HF_CACHE}: ${VID_FREE_GB:-?} GiB free, about $VID_WANT_GB GiB needed ($VID_WHAT; nothing was changed). Free some space first, or point HF_HOME at a bigger disk."
  # ckpt-fetch.py, the one fetch of every lane: the cache first (a full sha resolves straight
  # from snapshots/, refs or no refs), then the rest with resumed attempts. refs/main is what
  # the offline unit resolves, and a download by sha never writes it: --refs-main fill creates
  # it when missing and repairs it when it holds no commit hash (a corrupt ref strands the unit
  # offline whatever just succeeded here), and never moves a valid one the installer or the pin
  # watch will answer for. It runs on the local-cache success too: that is the common path.
  HF_HOME="${VID_HF:-$HF_CACHE}" HF_HUB_DOWNLOAD_TIMEOUT=30 HF_HUB_DISABLE_XET=1 \
    HF_HUB_DISABLE_PROGRESS_BARS="$DL_NO_BARS" \
    "$VID_PY" "$REPO_DIR/ckpt-fetch.py" "$VID_MODEL" "$VID_REV" --allow "$VID_ALLOW" --offline-first --refs-main fill \
    || die "the video checkpoint could not be verified or fetched (re-run to resume; set HF_TOKEN if it stalls; if the pinned revision was removed upstream, ./install-video.sh VIDEO_MODEL_REV=main re-pins it)"
  fetched "$VID_MODEL @ ${VID_REV:0:12}"
  # The update that waited for this Load (if any) goes in place now: this switch just fetched what its unit reads.
  # The video lane's --prepare builds the runtime only, so its switch is the one place its Load can apply an
  # update (the cockpit's Load journey ends on this step for a lane beside the text lanes; review of v1.52.0,
  # 2026-11-19: nothing posed it, and the Load left the lane on the old release, its card saying an update
  # waited, forever).
  if [ -f "$CONFIG_DIR/pending/$VIDEO_UNIT_NAME" ]; then
    if [ -f "$CONFIG_DIR/pending/$VIDEO_UNIT_NAME.base" ] && [ -f "$VIDEO_UNIT" ] \
       && [ "$(cat "$CONFIG_DIR/pending/$VIDEO_UNIT_NAME.base")" != "$(sha256sum "$VIDEO_UNIT" | cut -d' ' -f1)" ]; then
      unit_pending_clear "$VIDEO_UNIT"
      echo "the update that waited for this Load was dropped: $VIDEO_UNIT_NAME changed since it was rendered; ./install.sh writes it again"
    elif [ -n "$(unit_needs "$CONFIG_DIR/pending/$VIDEO_UNIT_NAME")" ]; then
      echo "NOTE: the update that waited for this Load still lacks $(unit_needs "$CONFIG_DIR/pending/$VIDEO_UNIT_NAME" | paste -sd ';' | sed 's/;/; /g'): it stays waiting; ./install.sh --with-video applies it"
    else
      if ! cmp -s "$CONFIG_DIR/pending/$VIDEO_UNIT_NAME" "$VIDEO_UNIT"; then
        cp "$CONFIG_DIR/pending/$VIDEO_UNIT_NAME" "$VID_STAGE"
        sudo install -m 644 "$VID_STAGE" "$VIDEO_UNIT"
        rm -f "$VID_STAGE"
        sudo systemctl daemon-reload
        echo "the update that waited for this Load is in place: $VIDEO_UNIT_NAME"
      fi
      unit_pending_clear "$VIDEO_UNIT"
    fi
  fi
  # Which lane this box served, written down before it is disabled: install.sh
  # updates that lane on a later run, and enablement cannot tell it once all are off.
  for TEXT_UNIT_NAME in qwen38-sglang.service qwen38-flash.service; do
    if systemctl is-enabled --quiet "$TEXT_UNIT_NAME" 2>/dev/null; then
      text_lane_note "$TEXT_UNIT_NAME"; break   # both enabled: the 27B, as install.sh follows it
    fi
  done
  # The loop variable ends in UNIT_NAME on purpose: CI checks every
  # `sudo systemctl <verb> "$..UNIT_NAME"` against the cockpit's sudoers allowlist,
  # and a variable called $u would have walked straight past it.
  for OTHER_UNIT_NAME in qwen38-sglang.service qwen38-flash.service qwen38-image.service qwen38-kolibri.service qwen38-nemotron.service qwen38-nemotron-super.service qwen38-flash-local.service qwen38-ornith-122b.service qwen38-flash-unsloth.service qwen38-image-h3.service qwen38-transcribe.service qwen38-lfm25.service qwen38-lfm3.service; do
    if systemctl is-enabled --quiet "$OTHER_UNIT_NAME" 2>/dev/null; then
      echo "disabling $OTHER_UNIT_NAME at boot (unit file kept for switching back)"
      sudo systemctl disable "$OTHER_UNIT_NAME"
    fi
  done
  sudo systemctl enable "$VIDEO_UNIT_NAME" >/dev/null 2>&1 || sudo systemctl enable "$VIDEO_UNIT_NAME"
  sudo systemctl daemon-reload
  RUNNING=""
  for u in qwen38-sglang.service qwen38-flash.service qwen38-image.service qwen38-kolibri.service qwen38-nemotron.service qwen38-nemotron-super.service qwen38-flash-local.service qwen38-ornith-122b.service qwen38-flash-unsloth.service qwen38-image-h3.service qwen38-transcribe.service qwen38-lfm25.service qwen38-lfm3.service; do
    systemctl is-active --quiet "$u" 2>/dev/null && RUNNING="$u"
  done
  printf '\n\033[1;32mSwitch queued: %s (%s)\033[0m\n' "$VID_MODEL" "$VIDEO_UNIT_NAME"
  if [ -n "$RUNNING" ]; then
    echo "Effective after:  sudo systemctl stop $RUNNING && sudo systemctl start $VIDEO_UNIT_NAME   (or next reboot)"
  else
    echo "Effective after:  sudo systemctl start $VIDEO_UNIT_NAME   (or next reboot; ready in about 12 min)"
  fi
  echo "The proxy on :30001 and opencode are text clients and were left as they are."
  echo "Switch back:      ./switch-model.sh stock   (or any text target, or image)"
  exit 0
fi

# ── The MiniMax-H3-Image lane: a second image lane, switched to the same way ──────
# The image and video lanes' contract: verify the weights, make it the one unit enabled at boot,
# never start or stop anything, print the exact commands. One checkpoint and one target: no unit
# to rewrite, no prompt ceiling on the proxy and no opencode default (text clients both).
# Its weights are two: the checkpoint, served by name (refs/main, pinned as the installer pins
# it), and the text encoder it loads from MiniMax-H3's text_encoder/ folder, which its unit names
# by its path at the revision the installer pins: on a box whose video lane has its weights, the
# same files, and nothing is downloaded for it. Handled here, whole, and exits.
if [ "$CHOICE" = "image-h3" ]; then
  [ -f "$IMAGE_H3_UNIT" ] || die "the MiniMax-H3-Image lane is not installed on this box (no $IMAGE_H3_UNIT). Install it once: ./install.sh --with-image-h3 (about 151 GB of headroom, 84 beside the video lane)"
  H3_WD="$(grep -m1 -E '^WorkingDirectory=' "$IMAGE_H3_UNIT" | cut -d= -f2- || true)"
  H3_PY="${H3_WD:-$HOME/.local/share/qwen38-image-h3}/venv/bin/python"
  [ -x "$H3_PY" ] || die "the MiniMax-H3-Image lane's runtime is missing ($H3_PY): ./prepare-lane.sh image-h3 builds it (no sudo; its first Load from the cockpit runs it), then switch again"
  # An update left this lane's new unit waiting for this Load (lanes.sh's unit_put, decision 1408): what this
  # switch fetches is what THAT unit reads (its checkpoint's snapshot and its text encoder's), not what the
  # one in place reads.
  H3_SRC="$IMAGE_H3_UNIT"
  if [ -f "$CONFIG_DIR/pending/$IMAGE_H3_UNIT_NAME" ]; then
    H3_SRC="$CONFIG_DIR/pending/$IMAGE_H3_UNIT_NAME"
    echo "the MiniMax-H3-Image lane's update waits for this Load: fetching what its new unit reads"
  fi
  # The pins, the allowlist and the sizes, from install-image-h3.sh, the one place they are declared.
  eval "$(grep -E '^(IMAGE_H3_MODEL_PIN|IMAGE_H3_MODEL_PIN_REV|IMAGE_H3_ENCODER_PIN|IMAGE_H3_ENCODER_PIN_REV|ALLOW|WEIGHTS_GB|ENCODER_GB)=' "$REPO_DIR/install-image-h3.sh" || true)"
  for _v in IMAGE_H3_MODEL_PIN IMAGE_H3_MODEL_PIN_REV IMAGE_H3_ENCODER_PIN IMAGE_H3_ENCODER_PIN_REV ALLOW WEIGHTS_GB ENCODER_GB; do
    eval "[ -n \"\${$_v:-}\" ]" || die "install-image-h3.sh no longer defines $_v (repo layout changed?)"
  done
  unset _v
  H3_MODEL="$(grep -oE -- '--model-path [^ ]+' "$H3_SRC" | head -1 | cut -d' ' -f2 || true)"
  H3_HF="$(grep -m1 -E '^Environment=HF_HOME=' "$H3_SRC" | cut -d= -f3- || true)"
  H3_REV=""
  # The unit names the checkpoint by its snapshot's path (v1.50.0: by name, an offline start refused the snapshot this
  # lane fetches only in part): the repository and the revision are that path's. A unit of before names the repository,
  # and its revision is then refs/main's, which the offline unit resolved, else the installer's pin.
  if [[ "$H3_MODEL" =~ /models--([^/]+)--([^/]+)/snapshots/([0-9a-f]{40})/?$ ]]; then
    H3_MODEL="${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"; H3_REV="${BASH_REMATCH[3]}"
  fi
  H3_MODEL="${H3_MODEL:-$IMAGE_H3_MODEL_PIN}"
  H3_MODEL_DIR="${H3_HF:-$HF_CACHE}/hub/models--${H3_MODEL//\//--}"
  if [ -z "$H3_REV" ] && [ -f "$H3_MODEL_DIR/refs/main" ]; then H3_REV="$(cat "$H3_MODEL_DIR/refs/main")"; fi
  [[ "$H3_REV" =~ ^[0-9a-f]{40}$ ]] || H3_REV=""
  if [ -z "$H3_REV" ]; then
    if [ "$H3_MODEL" = "$IMAGE_H3_MODEL_PIN" ]; then H3_REV="$IMAGE_H3_MODEL_PIN_REV"; else H3_REV=main; fi
  fi
  # Room first, for the two at once, on the cache's disk: what the cache holds of the checkpoint
  # comes off its share, and the text encoder counts nothing when the folder the unit names, or
  # one of its partitions' copies of it, is whole (install-image-h3.sh, the same rule).
  H3_HAVE_B="$({ du -s --apparent-size -B1 "$H3_MODEL_DIR/blobs" 2>/dev/null || true; } | cut -f1)"
  H3_NEED=$((WEIGHTS_GB - ${H3_HAVE_B:-0} / 1073741824)); [ "$H3_NEED" -ge 10 ] || H3_NEED=10
  H3_ENC_SNAP="${H3_HF:-$HF_CACHE}/hub/models--${IMAGE_H3_ENCODER_PIN//\//--}/snapshots/$IMAGE_H3_ENCODER_PIN_REV"
  if ! h3_encoder_cached "$H3_ENC_SNAP"; then H3_NEED=$((H3_NEED + ENCODER_GB)); fi
  H3_FREE="$(dl_free_gb "${H3_HF:-$HF_CACHE}")"
  [ "${H3_FREE:-0}" -ge "$H3_NEED" ] \
    || die "not enough room for $H3_MODEL and its text encoder under HF_HOME=${H3_HF:-$HF_CACHE}: ${H3_FREE:-?} GiB free, about $H3_NEED GiB needed (nothing was changed). Free some space first, or point HF_HOME at a bigger disk."
  printf '\n\033[1;36m── Verifying %s @ %s and its text encoder in the cache (resumable)\033[0m\n' "$H3_MODEL" "${H3_REV:0:12}"
  echo "MiniMax-H3-Image: its licence, the MiniMax-H3-Image Community License, is not granted in the European Union, the United Kingdom, South Korea or the United States (its Excluded Territories: definition 5 and clause V.4 of its LICENSE)."
  # ckpt-fetch.py, the one fetch of every lane: the cache first, then what is missing, Xet off.
  HF_HOME="${H3_HF:-$HF_CACHE}" HF_HUB_DOWNLOAD_TIMEOUT=30 HF_HUB_DISABLE_XET=1 \
    HF_HUB_DISABLE_PROGRESS_BARS="$DL_NO_BARS" \
    "$H3_PY" "$REPO_DIR/ckpt-fetch.py" "$H3_MODEL" "$H3_REV" --allow "$ALLOW" --offline-first --refs-main pin \
    || die "the MiniMax-H3-Image checkpoint could not be verified or fetched (re-run to resume; set HF_TOKEN if it stalls)"
  HF_HOME="${H3_HF:-$HF_CACHE}" HF_HUB_DOWNLOAD_TIMEOUT=30 HF_HUB_DISABLE_XET=1 \
    HF_HUB_DISABLE_PROGRESS_BARS="$DL_NO_BARS" \
    "$H3_PY" "$REPO_DIR/ckpt-fetch.py" "$IMAGE_H3_ENCODER_PIN" "$IMAGE_H3_ENCODER_PIN_REV" --allow "text_encoder/*" --offline-first \
    || die "the MiniMax-H3-Image text encoder could not be verified or fetched (re-run to resume; set HF_TOKEN if it stalls)"
  fetched "$H3_MODEL @ ${H3_REV:0:12} and its text encoder"
  # The update that waited for this Load (if any) goes in place now: this switch just fetched what its unit reads
  # (its checkpoint and its text encoder). The image-h3 lane's --prepare builds the runtime only, so its switch
  # is the one place its Load can apply an update (review of v1.52.0, 2026-11-19: nothing posed it, and the Load
  # left the lane on the old release, its card saying an update waited, forever).
  if [ -f "$CONFIG_DIR/pending/$IMAGE_H3_UNIT_NAME" ]; then
    if [ -f "$CONFIG_DIR/pending/$IMAGE_H3_UNIT_NAME.base" ] && [ -f "$IMAGE_H3_UNIT" ] \
       && [ "$(cat "$CONFIG_DIR/pending/$IMAGE_H3_UNIT_NAME.base")" != "$(sha256sum "$IMAGE_H3_UNIT" | cut -d' ' -f1)" ]; then
      unit_pending_clear "$IMAGE_H3_UNIT"
      echo "the update that waited for this Load was dropped: $IMAGE_H3_UNIT_NAME changed since it was rendered; ./install.sh writes it again"
    elif [ -n "$(unit_needs "$CONFIG_DIR/pending/$IMAGE_H3_UNIT_NAME")" ]; then
      echo "NOTE: the update that waited for this Load still lacks $(unit_needs "$CONFIG_DIR/pending/$IMAGE_H3_UNIT_NAME" | paste -sd ';' | sed 's/;/; /g'): it stays waiting; ./install.sh --with-image-h3 applies it"
    else
      if ! cmp -s "$CONFIG_DIR/pending/$IMAGE_H3_UNIT_NAME" "$IMAGE_H3_UNIT"; then
        cp "$CONFIG_DIR/pending/$IMAGE_H3_UNIT_NAME" "$H3_STAGE"
        sudo install -m 644 "$H3_STAGE" "$IMAGE_H3_UNIT"
        rm -f "$H3_STAGE"
        sudo systemctl daemon-reload
        echo "the update that waited for this Load is in place: $IMAGE_H3_UNIT_NAME"
      fi
      unit_pending_clear "$IMAGE_H3_UNIT"
    fi
  fi
  # Which text lane this box served, written down before it is disabled.
  for TEXT_UNIT_NAME in qwen38-sglang.service qwen38-flash.service; do
    if systemctl is-enabled --quiet "$TEXT_UNIT_NAME" 2>/dev/null; then
      text_lane_note "$TEXT_UNIT_NAME"; break   # both enabled: the 27B, as install.sh follows it
    fi
  done
  # The loop variable ends in UNIT_NAME on purpose: CI checks every
  # `sudo systemctl <verb> "$..UNIT_NAME"` against the cockpit's sudoers allowlist.
  for OTHER_UNIT_NAME in qwen38-sglang.service qwen38-flash.service qwen38-image.service qwen38-video.service qwen38-kolibri.service qwen38-nemotron.service qwen38-nemotron-super.service qwen38-flash-local.service qwen38-ornith-122b.service qwen38-flash-unsloth.service qwen38-transcribe.service qwen38-lfm25.service qwen38-lfm3.service; do
    if systemctl is-enabled --quiet "$OTHER_UNIT_NAME" 2>/dev/null; then
      echo "disabling $OTHER_UNIT_NAME at boot (unit file kept for switching back)"
      sudo systemctl disable "$OTHER_UNIT_NAME"
    fi
  done
  sudo systemctl enable "$IMAGE_H3_UNIT_NAME" >/dev/null 2>&1 || sudo systemctl enable "$IMAGE_H3_UNIT_NAME"
  sudo systemctl daemon-reload
  RUNNING=""
  for u in qwen38-sglang.service qwen38-flash.service qwen38-image.service qwen38-video.service qwen38-kolibri.service qwen38-nemotron.service qwen38-nemotron-super.service qwen38-flash-local.service qwen38-ornith-122b.service qwen38-flash-unsloth.service qwen38-transcribe.service qwen38-lfm25.service qwen38-lfm3.service; do
    systemctl is-active --quiet "$u" 2>/dev/null && RUNNING="$u"
  done
  printf '\n\033[1;32mSwitch queued: %s (%s)\033[0m\n' "$H3_MODEL" "$IMAGE_H3_UNIT_NAME"
  if [ -n "$RUNNING" ]; then
    echo "Effective after:  sudo systemctl stop $RUNNING && sudo systemctl start $IMAGE_H3_UNIT_NAME   (or next reboot)"
  else
    echo "Effective after:  sudo systemctl start $IMAGE_H3_UNIT_NAME   (or next reboot; ready in about 8 min)"
  fi
  echo "The proxy on :30001 and opencode are text clients and were left as they are."
  echo "Switch back:      ./switch-model.sh stock   (or any text target, or image)"
  exit 0
fi

# ── The NVIDIA Nemotron Transcribe lane: speech heard on the GPU, on a port of its own ───────────────────────────
# The image and video lanes' contract with the Unsloth GGUF lane's renderer: verify its files (the first switch fetches
# them), put the unit for them in place, make it the one unit enabled at boot, never start or stop anything, print the
# exact commands. The files and the unit come from the lane's installer, with no sudo (install-transcribe.sh --fetch,
# then --stage). It answers audio on its own port (30036), not the text port: no proxy ceiling, no opencode default.
TRANSCRIBE_STAGE="$CONFIG_DIR/qwen38-transcribe.service.switch-stage"
if [ "$CHOICE" = "transcribe" ]; then
  [ -f "$TRANSCRIBE_UNIT" ] || die "the NVIDIA Nemotron Transcribe lane is not installed on this box (no $TRANSCRIBE_UNIT). Install it once: ./install.sh --with-transcribe (about 37 GB of GGUF files and 8 of image)"
  eval "$(grep -E '^TRANSCRIBE_(REPO|REV|IMAGE)=' "$REPO_DIR/install-transcribe.sh" || true)"
  for _v in TRANSCRIBE_REPO TRANSCRIBE_REV TRANSCRIBE_IMAGE; do
    eval "[ -n \"\${$_v:-}\" ]" || die "install-transcribe.sh no longer defines $_v (repo layout changed?)"
  done
  unset _v
  TR_HF="$(unit_hf_cache "$TRANSCRIBE_UNIT")"
  TR_HF="${TR_HF:-$HF_CACHE}"
  printf '\n\033[1;36m── Verifying %s @ %s in the cache (resumable)\033[0m\n' "$TRANSCRIBE_REPO" "${TRANSCRIBE_REV:0:12}"
  HF_CACHE="$TR_HF" bash "$REPO_DIR/install-transcribe.sh" --fetch \
    || die "the NVIDIA Nemotron Transcribe files could not be verified or fetched (re-run to resume; set HF_TOKEN if it stalls)"
  fetched "$TRANSCRIBE_REPO @ ${TRANSCRIBE_REV:0:12}"
  docker image inspect "$TRANSCRIBE_IMAGE" >/dev/null 2>&1 \
    || die "the NVIDIA Nemotron Transcribe lane's image (${TRANSCRIBE_IMAGE:0:40}...) is missing: ./prepare-lane.sh transcribe pulls it (no sudo; its first Load from the cockpit runs it), then switch again"
  HF_CACHE="$TR_HF" bash "$REPO_DIR/install-transcribe.sh" --stage "$TRANSCRIBE_STAGE" \
    || { rm -f "$TRANSCRIBE_STAGE"; die "the NVIDIA Nemotron Transcribe unit could not be prepared (the lines above say why)"; }
  TR_REWRITTEN=""
  if ! cmp -s "$TRANSCRIBE_STAGE" "$TRANSCRIBE_UNIT"; then
    diff "$TRANSCRIBE_STAGE" "$TRANSCRIBE_UNIT" || true   # show exactly what changes
    sudo install -m 644 "$TRANSCRIBE_STAGE" "$TRANSCRIBE_UNIT"
    unit_pending_clear "$TRANSCRIBE_UNIT"   # its new unit is in place: an update that waited for this Load is applied
    TR_REWRITTEN=1
  fi
  rm -f "$TRANSCRIBE_STAGE"
  # Which text lane this box served, written down before it is disabled.
  for TEXT_UNIT_NAME in qwen38-sglang.service qwen38-flash.service; do
    if systemctl is-enabled --quiet "$TEXT_UNIT_NAME" 2>/dev/null; then
      text_lane_note "$TEXT_UNIT_NAME"; break   # both enabled: the 27B, as install.sh follows it
    fi
  done
  # The loop variable ends in UNIT_NAME on purpose: CI checks every
  # `sudo systemctl <verb> "$..UNIT_NAME"` against the cockpit's sudoers allowlist.
  for OTHER_UNIT_NAME in qwen38-sglang.service qwen38-flash.service qwen38-image.service qwen38-video.service qwen38-kolibri.service qwen38-nemotron.service qwen38-nemotron-super.service qwen38-flash-local.service qwen38-ornith-122b.service qwen38-flash-unsloth.service qwen38-image-h3.service qwen38-lfm25.service qwen38-lfm3.service; do
    if systemctl is-enabled --quiet "$OTHER_UNIT_NAME" 2>/dev/null; then
      echo "disabling $OTHER_UNIT_NAME at boot (unit file kept for switching back)"
      sudo systemctl disable "$OTHER_UNIT_NAME"
    fi
  done
  sudo systemctl enable "$TRANSCRIBE_UNIT_NAME" >/dev/null 2>&1 || sudo systemctl enable "$TRANSCRIBE_UNIT_NAME"
  sudo systemctl daemon-reload
  RUNNING=""
  for u in qwen38-sglang.service qwen38-flash.service qwen38-image.service qwen38-video.service qwen38-kolibri.service qwen38-nemotron.service qwen38-nemotron-super.service qwen38-flash-local.service qwen38-ornith-122b.service qwen38-flash-unsloth.service qwen38-image-h3.service qwen38-lfm25.service qwen38-lfm3.service; do
    systemctl is-active --quiet "$u" 2>/dev/null && RUNNING="$u"
  done
  printf '\n\033[1;32mSwitch queued: %s (%s)\033[0m\n' "$TRANSCRIBE_REPO" "$TRANSCRIBE_UNIT_NAME"
  if [ -n "$RUNNING" ]; then
    echo "Effective after:  sudo systemctl stop $RUNNING && sudo systemctl start $TRANSCRIBE_UNIT_NAME   (or next reboot)"
  elif systemctl is-active --quiet "$TRANSCRIBE_UNIT_NAME" 2>/dev/null; then
    if [ -n "$TR_REWRITTEN" ]; then
      echo "Effective after:  sudo systemctl restart $TRANSCRIBE_UNIT_NAME   (it serves the previous unit until then)"
    else
      echo "Its unit already named these files. Started before it did, the lane serves what it loaded: sudo systemctl restart $TRANSCRIBE_UNIT_NAME"
    fi
  else
    echo "Effective after:  sudo systemctl start $TRANSCRIBE_UNIT_NAME   (or next reboot; ready in about 15 s once its files are on disk)"
  fi
  echo "The proxy on :30001 and opencode are text clients and were left as they are: it answers /v1/audio/transcriptions"
  echo "and /v1/realtime on its own port with the box's key (docs/transcribe-lane.md)."
  echo "Switch back:      ./switch-model.sh stock   (or any text target)"
  exit 0
fi

# ── The Liquid AI LFM2.5 24B-A2B lane: Liquid AI's mixture of experts, on SGLang ─────────────────────────────────
# A text lane for the proxy, served by SGLang in the 27B lane's own image, under the side lanes' contract: verify its
# checkpoint and its draft (the first switch fetches them), put the unit for them in place, make it the one unit
# enabled at boot, never start or stop anything, print the exact commands. The unit is rendered by the lane's
# installer (install-lfm25.sh --stage, no sudo): one renderer, so a switch after the pins moved, or after the cache
# moved, puts in the unit an install would.
# Rewritten with sudo, so staged at a fixed path the cockpit's exact-argv sudoers allowlist names,
# as the other lanes' units are.
LFM25_STAGE="$CONFIG_DIR/qwen38-lfm25.service.switch-stage"
if [ "$CHOICE" = "lfm25" ]; then
  [ -f "$LFM25_UNIT" ] || die "the Liquid AI LFM2.5 24B-A2B lane is not installed on this box (no $LFM25_UNIT). Install it once: ./install.sh --with-lfm25 (about 26 GB, and the 27B lane's image)"
  # The checkpoint, its draft, their pinned revisions, their size and the image, from the lane's installer, the one
  # place each is declared.
  eval "$(grep -E '^LFM25_(IMAGE|REPO|REV|DRAFT_REPO|DRAFT_REV|GB)=' "$REPO_DIR/install-lfm25.sh" || true)"
  for _v in LFM25_IMAGE LFM25_REPO LFM25_REV LFM25_DRAFT_REPO LFM25_DRAFT_REV LFM25_GB; do
    eval "[ -n \"\${$_v:-}\" ]" || die "install-lfm25.sh no longer defines $_v (repo layout changed?)"
  done
  unset _v
  docker image inspect "$LFM25_IMAGE" >/dev/null 2>&1 \
    || die "the Liquid AI LFM2.5 24B-A2B lane's image (${LFM25_IMAGE:0:40}..., the 27B lane's) is missing: ./prepare-lane.sh lfm25 pulls it (no sudo; its first Load from the cockpit runs it), then switch again"
  LFM25_HF="$(unit_hf_cache "$LFM25_UNIT")"
  LFM25_HF="${LFM25_HF:-$HF_CACHE}"
  printf '\n\033[1;36m── Verifying %s @ %s and its draft in the cache (resumable)\033[0m\n' "$LFM25_REPO" "${LFM25_REV:0:12}"
  LFM25_TOKEN_ARGS=()
  # by name: docker takes the value from its own environment, so the token is in no argv
  if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; LFM25_TOKEN_ARGS=(-e HF_TOKEN); fi
  LFM25_PROGRESS_ARGS=()
  if [ -f "$REPO_DIR/hf-progress.py" ]; then
    LFM25_PROGRESS_ARGS=(-v "$REPO_DIR/hf-progress.py:/opt/qwen38/hf-progress.py:ro")
  fi
  # In the lane's own image, as the vLLM lanes fetch theirs; --init so that a cockpit job stopped past its timeout
  # stops this too. ckpt-fetch.py is the one fetch of every lane's checkpoint (it says why). The room is asked for
  # the two at the first fetch; the draft's own, 0.66 GB, rounded to 1. Both pinned as refs/main: the unit runs with
  # HF_HUB_OFFLINE=1 and SGLang's speculative hook reads the draft's config by its name, through refs/main, which a
  # download by commit never writes.
  for LFM25_PAIR in "$LFM25_REPO $LFM25_REV $LFM25_GB" "$LFM25_DRAFT_REPO $LFM25_DRAFT_REV 1"; do
    read -r LFM25_R LFM25_V LFM25_N <<< "$LFM25_PAIR"
    docker run --rm --init --network host --user "$(id -u):$(id -g)" --entrypoint python3 \
      -e HF_HOME=/hf -e HF_HUB_DOWNLOAD_TIMEOUT=30 -e HF_HUB_DISABLE_XET=1 \
      -e HF_HUB_DISABLE_PROGRESS_BARS="$DL_NO_BARS" \
      "${LFM25_TOKEN_ARGS[@]}" "${LFM25_PROGRESS_ARGS[@]}" \
      -v "$REPO_DIR/ckpt-fetch.py:/opt/qwen38/ckpt-fetch.py:ro" \
      -v "$LFM25_HF":/hf "$LFM25_IMAGE" /opt/qwen38/ckpt-fetch.py "$LFM25_R" "$LFM25_V" --offline-first --refs-main pin \
      --need-gb "$LFM25_N" --free-gb "$(dl_free_gb "$LFM25_HF")" \
      || die "$LFM25_R could not be verified or fetched (re-run to resume; set HF_TOKEN if it stalls)"
  done
  fetched "$LFM25_REPO @ ${LFM25_REV:0:12} and its draft"
  unset LFM25_PAIR LFM25_R LFM25_V LFM25_N
  # its unit for these checkpoints
  HF_CACHE="$LFM25_HF" bash "$REPO_DIR/install-lfm25.sh" --stage "$LFM25_STAGE" \
    || { rm -f "$LFM25_STAGE"; die "the Liquid AI LFM2.5 24B-A2B unit could not be prepared (the lines above say why)"; }
  LFM25_REWRITTEN=""
  if ! cmp -s "$LFM25_STAGE" "$LFM25_UNIT"; then
    diff "$LFM25_STAGE" "$LFM25_UNIT" || true   # show exactly what changes
    sudo install -m 644 "$LFM25_STAGE" "$LFM25_UNIT"
    unit_pending_clear "$LFM25_UNIT"   # its new unit is in place: an update that waited for this Load is applied
    LFM25_REWRITTEN=1
  fi
  rm -f "$LFM25_STAGE"
  # Which lane this box served at boot, written down before it is disabled: removing this lane
  # later gives the boot back to it (its installer's --uninstall).
  for TEXT_UNIT_NAME in qwen38-sglang.service qwen38-flash.service; do
    if systemctl is-enabled --quiet "$TEXT_UNIT_NAME" 2>/dev/null; then
      text_lane_note "$TEXT_UNIT_NAME"; break   # both enabled: the 27B, as install.sh follows it
    fi
  done
  # The loop variable ends in UNIT_NAME on purpose: CI checks every
  # `sudo systemctl <verb> "$..UNIT_NAME"` against the cockpit's sudoers allowlist.
  for TEXT_UNIT_NAME in qwen38-sglang.service qwen38-flash.service qwen38-image.service qwen38-video.service qwen38-kolibri.service qwen38-nemotron.service qwen38-nemotron-super.service qwen38-flash-local.service qwen38-ornith-122b.service qwen38-flash-unsloth.service qwen38-image-h3.service qwen38-transcribe.service qwen38-lfm3.service; do
    if systemctl is-enabled --quiet "$TEXT_UNIT_NAME" 2>/dev/null; then
      echo "disabling $TEXT_UNIT_NAME at boot (unit file kept for switching back)"
      sudo systemctl disable "$TEXT_UNIT_NAME"
    fi
  done
  sudo systemctl enable "$LFM25_UNIT_NAME" >/dev/null 2>&1 || sudo systemctl enable "$LFM25_UNIT_NAME"
  sudo systemctl daemon-reload
  RUNNING=""
  for u in qwen38-sglang.service qwen38-flash.service qwen38-image.service qwen38-video.service qwen38-kolibri.service qwen38-nemotron.service qwen38-nemotron-super.service qwen38-flash-local.service qwen38-ornith-122b.service qwen38-flash-unsloth.service qwen38-image-h3.service qwen38-transcribe.service qwen38-lfm3.service; do
    systemctl is-active --quiet "$u" 2>/dev/null && RUNNING="$u"
  done
  printf '\n\033[1;32mSwitch queued: %s (%s)\033[0m\n' "$LFM25_REPO" "$LFM25_UNIT_NAME"
  if [ -n "$RUNNING" ]; then
    echo "Effective after:  sudo systemctl stop $RUNNING && sudo systemctl start $LFM25_UNIT_NAME   (or next reboot)"
  elif systemctl is-active --quiet "$LFM25_UNIT_NAME" 2>/dev/null; then
    if [ -n "$LFM25_REWRITTEN" ]; then
      echo "Effective after:  sudo systemctl restart $LFM25_UNIT_NAME   (it serves the previous unit until then)"
    else
      echo "Its unit already named these checkpoints. Started before it did, the lane serves what it loaded: sudo systemctl restart $LFM25_UNIT_NAME"
    fi
  else
    echo "Effective after:  sudo systemctl start $LFM25_UNIT_NAME   (or next reboot; ready in about 5 min)"
  fi
  # opencode follows the lane, as on a text switch: the lane's provider becomes the default of the
  # generated config and of the one opencode reads, and the server that reads them restarts when
  # they changed. A box installed with --no-opencode is left alone.
  if [ -f "$CONFIG_DIR/opencode.off" ]; then
    echo "opencode integration is off on this box (./install.sh --no-opencode): default model left alone"
  else
    LFM25_OC_SUM="$({ cat "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json" 2>/dev/null || true; } | sha256sum)"
    LFM25_LIMITS="$("$REPO_DIR/oc-limits.sh" lfm25)" || LFM25_LIMITS=""
    LFM25_NAME="$(python3 "$REPO_DIR/oc-point-default.py" --label lfm25 131072 || true)"
    LFM25_PORT="$({ grep -m1 -oE 'PROXY_PORT=[0-9]+|keepalive-proxy.py [0-9]+' /etc/systemd/system/qwen38-keepalive.service 2>/dev/null || true; } | grep -oE '[0-9]+$' || true)"
    LFM25_SRC="$(mktemp)"
    if [ -n "$LFM25_LIMITS" ] && python3 "$REPO_DIR/oc-lfm25.py" "http://127.0.0.1:${LFM25_PORT:-30001}/v1" "$CONFIG_DIR/api-key" \
         "${LFM25_LIMITS%% *}" "$(echo "$LFM25_LIMITS" | cut -d' ' -f2)" "${LFM25_NAME:-Liquid AI LFM2.5 24B-A2B (local, 131K)}" > "$LFM25_SRC"; then
      # The compaction block is global: sized from this lane's window as every text switch sizes
      # it from its own. pi's keepRecentTokens follows (pi-gen).
      LFM25_KEEP="$("$REPO_DIR/oc-limits.sh" --preserve "${LFM25_LIMITS%% *}")" || LFM25_KEEP=""
      for OC_JSON in "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json"; do
        [ -f "$OC_JSON" ] || continue
        python3 "$REPO_DIR/oc-merge-limits.py" "$OC_JSON" --add-providers "$LFM25_SRC" >/dev/null || true
        python3 "$REPO_DIR/oc-point-default.py" "$OC_JSON" lfm25 lfm25 131072 \
          || echo "NOTE: could not update $OC_JSON (hand-edited?); set its \"model\" field yourself"
        [ -n "$LFM25_KEEP" ] && { python3 "$REPO_DIR/oc-merge-limits.py" "$OC_JSON" --compaction "$LFM25_KEEP" \
          || echo "NOTE: could not update the compaction block in $OC_JSON; check it by hand ($LFM25_KEEP)"; }
      done
    fi
    rm -f "$LFM25_SRC"
    if [ -d "$CONFIG_DIR/pi" ] || [ -d "$CONFIG_DIR/omp" ]; then
      python3 "$REPO_DIR/pi-gen.py" --config "$CONFIG_DIR/opencode.json" --out "$CONFIG_DIR" \
        || echo "NOTE: could not regenerate the pi/omp artifacts; run ./pi-gen.py by hand"
    fi
    if [ "$LFM25_OC_SUM" != "$({ cat "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json" 2>/dev/null || true; } | sha256sum)" ] \
       && systemctl list-unit-files opencode-web.service >/dev/null 2>&1 && systemctl is-active --quiet opencode-web.service; then
      sudo systemctl restart opencode-web.service \
        && echo "opencode-web.service restarted so it reads the new default" \
        || echo "NOTE: could not restart opencode-web.service; restart it by hand"
    fi
  fi
  echo "Clients reach it through the proxy on :30001 with the box's key, as model lfm2.5-24b-a2b."
  echo "Switch back:      ./switch-model.sh stock   (or any other target)"
  exit 0
fi

# ── The Liquid AI LFM3 lane: Liquid AI's LFM3 1.7B, on vLLM ─────────────────────────────────────────────────────
# A text lane for the proxy, served by vLLM's own nightly image, under the side lanes' contract: verify its checkpoint
# (the first switch fetches it), put the unit for it in place, make it the one unit enabled at boot, never start or stop
# anything, print the exact commands. The unit is rendered by the lane's installer (install-lfm3.sh --stage, no sudo):
# one renderer, so a switch after the pins moved, or after the cache moved, puts in the unit an install would.
# Rewritten with sudo, so staged at a fixed path the cockpit's exact-argv sudoers allowlist names,
# as the other lanes' units are.
LFM3_STAGE="$CONFIG_DIR/qwen38-lfm3.service.switch-stage"
if [ "$CHOICE" = "lfm3" ]; then
  [ -f "$LFM3_UNIT" ] || die "the Liquid AI LFM3 lane is not installed on this box (no $LFM3_UNIT). Install it once: ./install.sh --with-lfm3 (about 2 GB, and vLLM's nightly image, 39 GB at its pull's peak)"
  # The checkpoint, its pinned revision, its size and the image, from the lane's installer, the one place each is
  # declared.
  eval "$(grep -E '^LFM3_(IMAGE|REPO|REV|GB)=' "$REPO_DIR/install-lfm3.sh" || true)"
  for _v in LFM3_IMAGE LFM3_REPO LFM3_REV LFM3_GB; do
    eval "[ -n \"\${$_v:-}\" ]" || die "install-lfm3.sh no longer defines $_v (repo layout changed?)"
  done
  unset _v
  docker image inspect "$LFM3_IMAGE" >/dev/null 2>&1 \
    || die "the Liquid AI LFM3 lane's image (${LFM3_IMAGE:0:40}...) is missing: ./prepare-lane.sh lfm3 pulls it (no sudo; its first Load from the cockpit runs it), then switch again"
  LFM3_HF="$(unit_hf_cache "$LFM3_UNIT")"
  LFM3_HF="${LFM3_HF:-$HF_CACHE}"
  printf '\n\033[1;36m── Verifying %s @ %s in the cache (resumable)\033[0m\n' "$LFM3_REPO" "${LFM3_REV:0:12}"
  LFM3_TOKEN_ARGS=()
  # by name: docker takes the value from its own environment, so the token is in no argv
  if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; LFM3_TOKEN_ARGS=(-e HF_TOKEN); fi
  LFM3_PROGRESS_ARGS=()
  if [ -f "$REPO_DIR/hf-progress.py" ]; then
    LFM3_PROGRESS_ARGS=(-v "$REPO_DIR/hf-progress.py:/opt/qwen38/hf-progress.py:ro")
  fi
  # In the lane's own image, as the other vLLM lanes fetch theirs; --init so that a cockpit job stopped past its
  # timeout stops this too. ckpt-fetch.py is the one fetch of every lane's checkpoint (it says why). Pinned as
  # refs/main, as install-lfm3.sh pins it: the unit runs with HF_HUB_OFFLINE=1, and a read of the repo by its name
  # then finds the pinned commit. None is known in this image (vLLM reads the tokenizer at --revision).
  docker run --rm --init --network host --user "$(id -u):$(id -g)" --entrypoint python3 \
    -e HF_HOME=/hf -e HF_HUB_DOWNLOAD_TIMEOUT=30 -e HF_HUB_DISABLE_XET=1 \
    -e HF_HUB_DISABLE_PROGRESS_BARS="$DL_NO_BARS" \
    "${LFM3_TOKEN_ARGS[@]}" "${LFM3_PROGRESS_ARGS[@]}" \
    -v "$REPO_DIR/ckpt-fetch.py:/opt/qwen38/ckpt-fetch.py:ro" \
    -v "$LFM3_HF":/hf "$LFM3_IMAGE" /opt/qwen38/ckpt-fetch.py "$LFM3_REPO" "$LFM3_REV" --offline-first --refs-main pin \
    --need-gb "$LFM3_GB" --free-gb "$(dl_free_gb "$LFM3_HF")" \
    || die "the Liquid AI LFM3 checkpoint could not be verified or fetched (re-run to resume; set HF_TOKEN if it stalls)"
  fetched "$LFM3_REPO @ ${LFM3_REV:0:12}"
  # its unit for this checkpoint
  HF_CACHE="$LFM3_HF" bash "$REPO_DIR/install-lfm3.sh" --stage "$LFM3_STAGE" \
    || { rm -f "$LFM3_STAGE"; die "the Liquid AI LFM3 unit could not be prepared (the lines above say why)"; }
  LFM3_REWRITTEN=""
  if ! cmp -s "$LFM3_STAGE" "$LFM3_UNIT"; then
    diff "$LFM3_STAGE" "$LFM3_UNIT" || true   # show exactly what changes
    sudo install -m 644 "$LFM3_STAGE" "$LFM3_UNIT"
    unit_pending_clear "$LFM3_UNIT"   # its new unit is in place: an update that waited for this Load is applied
    LFM3_REWRITTEN=1
  fi
  rm -f "$LFM3_STAGE"
  # Which lane this box served at boot, written down before it is disabled: removing this lane
  # later gives the boot back to it (its installer's --uninstall).
  for TEXT_UNIT_NAME in qwen38-sglang.service qwen38-flash.service; do
    if systemctl is-enabled --quiet "$TEXT_UNIT_NAME" 2>/dev/null; then
      text_lane_note "$TEXT_UNIT_NAME"; break   # both enabled: the 27B, as install.sh follows it
    fi
  done
  # The loop variable ends in UNIT_NAME on purpose: CI checks every
  # `sudo systemctl <verb> "$..UNIT_NAME"` against the cockpit's sudoers allowlist.
  for TEXT_UNIT_NAME in qwen38-sglang.service qwen38-flash.service qwen38-image.service qwen38-video.service qwen38-kolibri.service qwen38-nemotron.service qwen38-nemotron-super.service qwen38-flash-local.service qwen38-ornith-122b.service qwen38-flash-unsloth.service qwen38-image-h3.service qwen38-transcribe.service qwen38-lfm25.service; do
    if systemctl is-enabled --quiet "$TEXT_UNIT_NAME" 2>/dev/null; then
      echo "disabling $TEXT_UNIT_NAME at boot (unit file kept for switching back)"
      sudo systemctl disable "$TEXT_UNIT_NAME"
    fi
  done
  sudo systemctl enable "$LFM3_UNIT_NAME" >/dev/null 2>&1 || sudo systemctl enable "$LFM3_UNIT_NAME"
  sudo systemctl daemon-reload
  RUNNING=""
  for u in qwen38-sglang.service qwen38-flash.service qwen38-image.service qwen38-video.service qwen38-kolibri.service qwen38-nemotron.service qwen38-nemotron-super.service qwen38-flash-local.service qwen38-ornith-122b.service qwen38-flash-unsloth.service qwen38-image-h3.service qwen38-transcribe.service qwen38-lfm25.service; do
    systemctl is-active --quiet "$u" 2>/dev/null && RUNNING="$u"
  done
  printf '\n\033[1;32mSwitch queued: %s (%s)\033[0m\n' "$LFM3_REPO" "$LFM3_UNIT_NAME"
  if [ -n "$RUNNING" ]; then
    echo "Effective after:  sudo systemctl stop $RUNNING && sudo systemctl start $LFM3_UNIT_NAME   (or next reboot)"
  elif systemctl is-active --quiet "$LFM3_UNIT_NAME" 2>/dev/null; then
    if [ -n "$LFM3_REWRITTEN" ]; then
      echo "Effective after:  sudo systemctl restart $LFM3_UNIT_NAME   (it serves the previous unit until then)"
    else
      echo "Its unit already named this checkpoint. Started before it did, the lane serves what it loaded: sudo systemctl restart $LFM3_UNIT_NAME"
    fi
  else
    echo "Effective after:  sudo systemctl start $LFM3_UNIT_NAME   (or next reboot; ready in 2 min 40 s at its first boot)"
  fi
  # opencode follows the lane, as on a text switch: the lane's provider becomes the default of the
  # generated config and of the one opencode reads, and the server that reads them restarts when
  # they changed. A box installed with --no-opencode is left alone.
  if [ -f "$CONFIG_DIR/opencode.off" ]; then
    echo "opencode integration is off on this box (./install.sh --no-opencode): default model left alone"
  else
    LFM3_OC_SUM="$({ cat "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json" 2>/dev/null || true; } | sha256sum)"
    LFM3_LIMITS="$("$REPO_DIR/oc-limits.sh" lfm3)" || LFM3_LIMITS=""
    LFM3_NAME="$(python3 "$REPO_DIR/oc-point-default.py" --label lfm3 262144 || true)"
    LFM3_PORT="$({ grep -m1 -oE 'PROXY_PORT=[0-9]+|keepalive-proxy.py [0-9]+' /etc/systemd/system/qwen38-keepalive.service 2>/dev/null || true; } | grep -oE '[0-9]+$' || true)"
    LFM3_SRC="$(mktemp)"
    if [ -n "$LFM3_LIMITS" ] && python3 "$REPO_DIR/oc-lfm3.py" "http://127.0.0.1:${LFM3_PORT:-30001}/v1" "$CONFIG_DIR/api-key" \
         "${LFM3_LIMITS%% *}" "$(echo "$LFM3_LIMITS" | cut -d' ' -f2)" "${LFM3_NAME:-Liquid AI LFM3 (local, 262K)}" > "$LFM3_SRC"; then
      # The compaction block is global: sized from this lane's window as every text switch sizes
      # it from its own. pi's keepRecentTokens follows (pi-gen).
      LFM3_KEEP="$("$REPO_DIR/oc-limits.sh" --preserve "${LFM3_LIMITS%% *}")" || LFM3_KEEP=""
      for OC_JSON in "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json"; do
        [ -f "$OC_JSON" ] || continue
        python3 "$REPO_DIR/oc-merge-limits.py" "$OC_JSON" --add-providers "$LFM3_SRC" >/dev/null || true
        python3 "$REPO_DIR/oc-point-default.py" "$OC_JSON" lfm3 lfm3 262144 \
          || echo "NOTE: could not update $OC_JSON (hand-edited?); set its \"model\" field yourself"
        [ -n "$LFM3_KEEP" ] && { python3 "$REPO_DIR/oc-merge-limits.py" "$OC_JSON" --compaction "$LFM3_KEEP" \
          || echo "NOTE: could not update the compaction block in $OC_JSON; check it by hand ($LFM3_KEEP)"; }
      done
    fi
    rm -f "$LFM3_SRC"
    if [ -d "$CONFIG_DIR/pi" ] || [ -d "$CONFIG_DIR/omp" ]; then
      python3 "$REPO_DIR/pi-gen.py" --config "$CONFIG_DIR/opencode.json" --out "$CONFIG_DIR" \
        || echo "NOTE: could not regenerate the pi/omp artifacts; run ./pi-gen.py by hand"
    fi
    if [ "$LFM3_OC_SUM" != "$({ cat "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json" 2>/dev/null || true; } | sha256sum)" ] \
       && systemctl list-unit-files opencode-web.service >/dev/null 2>&1 && systemctl is-active --quiet opencode-web.service; then
      sudo systemctl restart opencode-web.service \
        && echo "opencode-web.service restarted so it reads the new default" \
        || echo "NOTE: could not restart opencode-web.service; restart it by hand"
    fi
  fi
  echo "Clients reach it through the proxy on :30001 with the box's key, as model lfm3-1.7b."
  echo "Switch back:      ./switch-model.sh stock   (or any other target)"
  exit 0
fi

SGL_UNIT="/etc/systemd/system/qwen38-sglang.service"
FLASH_UNIT="/etc/systemd/system/qwen38-flash.service"

case "$CHOICE" in
  stock)      TARGET_REPO="$STOCK_REPO"; TARGET_REV="$STOCK_REV"; TARGET_LANE=27b ;;
  uncensored) TARGET_REPO="$UNC_REPO";   TARGET_REV="$UNC_REV";   TARGET_LANE=27b ;;
  fp8)        TARGET_REPO="$FP8_REPO";   TARGET_REV="$FP8_REV";   TARGET_LANE=27b ;;
  uncensored-fp8) TARGET_REPO="$UNCFP8_REPO"; TARGET_REV="$UNCFP8_REV"; TARGET_LANE=27b ;;
  arete)      TARGET_REPO="$ARETE_REPO"; TARGET_REV="$ARETE_REV"; TARGET_LANE=27b ;;
  flash)      TARGET_REPO="$FLASH_REPO"; TARGET_REV="$FLASH_REV"; TARGET_LANE=flash ;;
  flash-nvda) TARGET_REPO="$FLASH_NVDA_REPO"; TARGET_REV="$FLASH_NVDA_REV"; TARGET_LANE=flash ;;
  flash-uncensored) TARGET_REPO="$FLASH_UNC_REPO"; TARGET_REV="$FLASH_UNC_REV"; TARGET_LANE=flash ;;
esac

# --fetch honors MODEL_REV= as the revision to fetch, not the pin: prepare-lane.sh passes it when the unit
# that waits names another revision than the pin (the pin moved since the update waited, or a MODEL_REV=
# custom install), and a fetch of the pin would leave that wait forever unfilled (seventh pass of the
# v1.52.0 reviews, 2026-11-19). A switch (no --fetch) still points the unit at the pin.
if [ "$FETCH_ONLY" -eq 1 ] && [ -n "${MODEL_REV:-}" ]; then TARGET_REV="$MODEL_REV"; fi

# The pins come from install.sh through eval, so a pin renamed or emptied there
# would otherwise reach the download step as an empty repo or revision. CI
# guards the names; this guards the values on a box that already diverged.
[ -n "${TARGET_REPO:-}" ] && [ -n "${TARGET_REV:-}" ] \
  || die "target '$CHOICE' resolved to an empty checkpoint or revision; install.sh is missing its pin"

# A GPU outside its ATS addressing mode cannot start the flash lane (issue #54): no checkpoint is
# fetched for it and the boot is not pointed at it. A cockpit Load that points the boot at it runs
# this before it stops the serving lane (a failed step ends the journey), and so does the Library's
# Download (--fetch).
if [ "$TARGET_LANE" = flash ] && ATS_WHY="$(flash_cannot_start)"; then
  die "the flash lane cannot start on this box: $ATS_WHY"
fi

if [ "$TARGET_LANE" = "27b" ]; then
  [ -f "$SGL_UNIT" ] || die "the 27B stack is not installed on this box (no $SGL_UNIT). Install it once first: MODEL_CHOICE=$CHOICE ./install.sh"
  TARGET_UNIT="$SGL_UNIT"; TARGET_UNIT_NAME="qwen38-sglang.service"
  OTHER_UNIT="$FLASH_UNIT"; OTHER_UNIT_NAME="qwen38-flash.service"
  INVOCATION="$SGL_UNIT"
  PIN_IMAGE="$SERVE_IMAGE"
else
  [ -f "$FLASH_UNIT" ] || die "the Flash-Next stack is not installed on this box (no $FLASH_UNIT). Install it once first: MODEL_CHOICE=$CHOICE ./install.sh (~152 GB download: the checkpoint 136, the image 16; no image build)"
  TARGET_UNIT="$FLASH_UNIT"; TARGET_UNIT_NAME="qwen38-flash.service"
  OTHER_UNIT="$SGL_UNIT"; OTHER_UNIT_NAME="qwen38-sglang.service"
  INVOCATION="$CONFIG_DIR/launch-flash.sh"
  PIN_IMAGE="$FLASH_SERVE_IMAGE"
fi
# The cache the installed engine mounts, read back the way install.sh does (CUR_HF),
# not the default: a box installed with HF_CACHE=/data/hf and switched from the cockpit,
# which passes no HF_CACHE, downloaded and patched into ~/.cache/huggingface while its
# unit kept mounting /data/hf, and the next start found no checkpoint (found in review,
# 2026-09-24). Both text lanes mount the same cache; an explicit HF_CACHE still wins.
if [ -z "$_ENV_HF_CACHE" ]; then
  for _inv in "$INVOCATION" "$SGL_UNIT" "$CONFIG_DIR/launch-flash.sh"; do
    _hf="$(unit_hf_cache "$_inv")"
    if [ -n "$_hf" ]; then
      [ "$_hf" != "$HF_CACHE" ] && echo "using the HF cache the installed engine mounts: $_hf"
      HF_CACHE="$_hf"
      break
    fi
  done
  unset _inv _hf
fi
# The image whose presence decides a switch is the one the installed invocation
# will actually run, not the base it may have been built from. Since v1.8 a box
# can serve the pinned official image and keep no base image at all, and both
# base images are re-pullable, so gating on a base turned a working switch into
# a false "run install.sh once". Read it the way the cockpit does: the line just
# above the one that calls the server.
unit_image() {
  [ -f "$1" ] || return 0
  awk '/sglang\.launch_server/ { print prev; exit } { prev = $0 }' "$1" \
    | sed -e 's/[[:space:]]*\\$//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}
# Point the flash launcher at another checkpoint of the same lane: the model
# path, the revision, and the one flag line that belongs to the checkpoint.
#
# That line is REPLACED WHOLE, not patched in place. Patching fragments left one
# checkpoint's draft quantization standing behind the other's scheme (seen live
# on 2026-09-08 switching nvda -> flash), because the two strings hold a
# different number of flags. The rendered line always ends with the FP4 GEMM
# backend, so rebuilding it from the indent is exact.
#
# Tested by CI against both input shapes, sourced out of this file so a copy of
# the logic cannot drift from it.
# Point the 27B unit at another checkpoint of the same lane: the model path, the
# revision, and the KV cache dtype, which belongs to the checkpoint too.
#
# That last one is why this is a function and not two sed expressions. Until
# 2026-09-08 the switch rewrote only the path and the revision, so a switch from
# an NVFP4 target to Qwen's FP8 release left the unit with NO --kv-cache-dtype:
# that checkpoint carries no KV scales, "auto" falls back to a bf16 KV cache, and
# the pool halves (this repo measured 771,139 tokens against 382,706 on the same
# 1M unit). In the other direction the flag stayed on an NVFP4 target, which is a
# no-op there (both NVFP4 exports declare kv_cache_quant_algo: FP8, checked) but
# is what left this box reporting a drift for days.
#
# The fraction on that line is the box's own choice (0.50 native, 0.76 for 1M,
# and a hand-tuned value must survive a switch), so it is read from the unit and
# written back unchanged. Tested by CI against both directions and both modes,
# sourced out of this file so no copy of the logic can drift from it.
rewrite_27b_unit() {  # $1 unit, $2 repo, $3 revision, $4 kv args ("" or "--kv-cache-dtype fp8_e4m3 ")
  awk -v repo="$2" -v rev="$3" -v kv="$4" '
    {
      line = $0
      sub(/--model-path [A-Za-z0-9_.\/-]+/, "--model-path " repo, line)
      sub(/--revision [A-Za-z0-9]+/, "--revision " rev, line)
      if (match(line, /--mem-fraction-static [0-9.]+/)) {
        frac = substr(line, RSTART, RLENGTH)
        match(line, /^ */)
        line = substr(line, 1, RLENGTH) frac " " kv "\\"
      }
      print line
    }
  ' "$1"
}

rewrite_flash_launcher() {  # $1 launcher, $2 repo, $3 revision, $4 quant args
  awk -v repo="$2" -v rev="$3" -v quant="$4" '
    {
      line = $0
      sub(/--model-path [A-Za-z0-9_.\/-]+/, "--model-path " repo, line)
      sub(/--revision [A-Za-z0-9]+/, "--revision " rev, line)
      if (line ~ /^ *(--quantization|--moe-runner-backend)/) {
        match(line, /^ */)
        line = substr(line, 1, RLENGTH) quant "--fp4-gemm-backend flashinfer_cutlass \\"
      }
      print line
    }
  ' "$1"
}

DL_IMAGE="$(unit_image "$INVOCATION")"
case "$DL_IMAGE" in
  *:*) ;;                                  # name:tag or name@sha256:... both match
  *) DL_IMAGE="$PIN_IMAGE" ;;              # unreadable invocation: fall back to the repo's pin
esac
docker image inspect "$DL_IMAGE" >/dev/null 2>&1 \
  || die "the image this box would serve is not present ($DL_IMAGE): a lane installed on demand is prepared by its first Load from the cockpit (or ./prepare-lane.sh $TARGET_LANE); otherwise run MODEL_CHOICE=$CHOICE ./install.sh once, the switch stays surgical"

if [ "$FETCH_ONLY" -eq 1 ]; then
  echo "fetching $TARGET_REPO @ $TARGET_REV; nothing else changes (--fetch)"
elif [ "$TARGET_LANE" = "27b" ]; then
  CUR="$(grep -oE -- '--model-path [^ ]+' "$TARGET_UNIT" | head -1 | cut -d' ' -f2 || true)"
  if [ "$CUR" = "$TARGET_REPO" ] && ! systemctl is-enabled --quiet "$OTHER_UNIT_NAME" 2>/dev/null; then
    echo "unit already points at $TARGET_REPO"
  else
    echo "switching: ${CUR:-<none>} -> $TARGET_REPO @ $TARGET_REV"
  fi
else
  echo "switching the serving lane to Flash-Next: $TARGET_REPO @ $TARGET_REV"
fi

# 1) checkpoint in cache (docker + pinned image python, same as install.sh step 4)
printf '\n\033[1;36m── Verifying/downloading %s @ %s (resumable)\033[0m\n' "$TARGET_REPO" "$TARGET_REV"
# Room first, as install.sh does: a cockpit button can start this, and a download that
# fills the disk under a serving engine is a failure of everything on it, while the old
# message blamed an HF_TOKEN (found in review, 2026-09-24). What the cache already holds
# for the target comes off the lane's need (install.sh's numbers: 45 GB for a 27B
# checkpoint and its caches, 180 for flash).
DL_NEED_GB="$(ckpt_room_gb "$TARGET_LANE" "$HF_CACHE" "$TARGET_REPO" "$TARGET_REV")"   # lanes.sh, the first Load's too
DL_FREE_GB="$(dl_free_gb)"
[ "${DL_FREE_GB:-0}" -ge "$DL_NEED_GB" ] \
  || die "not enough room for $TARGET_REPO under HF_CACHE=$HF_CACHE: ${DL_FREE_GB:-?} GiB free, about $DL_NEED_GB GiB needed (nothing was changed). Free some space first."
DL_TOKEN_ARGS=()
# by name: docker takes the value from its own environment, so the token is not in the
# docker client's argv, which any local user can read in /proc (found in review, 2026-09-24)
if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; DL_TOKEN_ARGS=(-e HF_TOKEN); fi
# The byte count reads hf-progress.py in the container, beside ckpt-fetch.py. Mounted only
# when it is there: a bind of a missing file makes docker create a root-owned directory in
# its place.
DL_PROGRESS_ARGS=()
if [ -f "$REPO_DIR/hf-progress.py" ]; then
  DL_PROGRESS_ARGS=(-v "$REPO_DIR/hf-progress.py:/opt/qwen38/hf-progress.py:ro")
fi
# --init: the cockpit stops a switch that overruns its job timeout with a TERM to the
# whole process group, and docker run passes it on to the container, where python3 as
# PID 1 has no handler for it and ignores it. Measured on the reference box: without
# --init the container outlived the TERM and then a SIGKILL of its docker client, and
# went on downloading with nobody attached; with it, both were gone within 6 s.
# ckpt-fetch.py, the one fetch of every lane's checkpoint (resumed attempts, the Xet exception
# the flash checkpoint's 53.7 GB n-gram table needs, the byte count). The cache first, as the
# other lanes' switches ask it: a checkpoint every shard of which is there answers without
# the network, so a cockpit Load of a text lane works offline or with the Hub throttling
# (until v1.31 it asked the Hub every time, and died without it). Offline serving resolves
# "main" via refs/main, which a pinned-sha download never writes: --refs-main fill writes it
# where it is missing and never moves one the box serves.
docker run --rm --init --network host --user "$(id -u):$(id -g)" \
  --entrypoint python3 \
  -e HF_HOME=/hf -e HF_HUB_DOWNLOAD_TIMEOUT=30 -e HF_HUB_DISABLE_XET=1 \
  -e HF_HUB_DISABLE_PROGRESS_BARS="$DL_NO_BARS" \
  "${DL_TOKEN_ARGS[@]}" "${DL_PROGRESS_ARGS[@]}" \
  -v "$REPO_DIR/ckpt-fetch.py:/opt/qwen38/ckpt-fetch.py:ro" \
  -v "$HF_CACHE":/hf \
  "$DL_IMAGE" /opt/qwen38/ckpt-fetch.py "$TARGET_REPO" "$TARGET_REV" --offline-first --refs-main fill \
  || die "download failed ($(dl_free_gb) GiB left under $HF_CACHE; if that is near 0 the disk filled up, otherwise re-run to resume: HuggingFace throttles unauthenticated downloads, set HF_TOKEN=<your token> if it stalls)"
fetched "$TARGET_REPO @ ${TARGET_REV:0:12}"

# 2, 2a, 2b) the fetched checkpoint's own files, patched for what will serve them: one function
# lanes.sh holds, which the switch calls here with the unit in place, and the cockpit's
# apply_pending (patch_waiting_checkpoint) with the unit that waited, once that unit is in place,
# each with the cache that unit mounts. A Load's prepare (prepare-lane.sh) does not call it since
# v1.52.1: in --fetch, fetched() ends the run before this line, and a Load poses a wait through
# apply_pending, which patches after the pose (sixth review of v1.52.0, 2026-11-19, GLM 5.3: the
# Load's --fetch stopped right after the download, and a lane serving its target with an update
# waiting restarted on a config the switch's steps never patched, which does not load in 1M).
patch_served_checkpoint "$TARGET_LANE" "$CHOICE" "$TARGET_UNIT" "$HF_CACHE" "$TARGET_REPO" "$TARGET_REV"

# 3) point the target unit at the exact checkpoint. For the 27B pair, rewrite
#    ONLY --model-path/--revision (the draft's own revision flag has a
#    different name and is never touched). For flash, refresh --revision to
#    the repo pin. The 27B unit is installed with sudo, so it is staged at a
#    FIXED path the cockpit's exact-argv sudoers allowlist pins (a mktemp name
#    cannot be pinned); the user-owned flash launcher keeps a temp file.
TMP_UNIT="$(mktemp)"
STAGE_UNIT="$CONFIG_DIR/qwen38-sglang.service.switch-stage"
STAGE_CEIL="$CONFIG_DIR/keepalive-ceiling.conf.switch-stage"
trap 'rm -f "$TMP_UNIT" "$STAGE_UNIT" "$STAGE_CEIL"' EXIT
if [ "$TARGET_LANE" = "27b" ]; then
  # The KV cache dtype follows the checkpoint, exactly as install.sh decides it:
  # Qwen's FP8 release carries no KV scales and must ask for fp8 explicitly, the
  # NVFP4 exports declare it in their own quant config and must not be forced
  # (Arete's since 2a above).
  # CI asserts these two strings still match install.sh's KV_CACHE_ARGS.
  case "$CHOICE" in
    fp8|uncensored-fp8) NEW_KV="--kv-cache-dtype fp8_e4m3 " ;;
    *)                  NEW_KV="" ;;
  esac
  rewrite_27b_unit "$TARGET_UNIT" "$TARGET_REPO" "$TARGET_REV" "$NEW_KV" > "$STAGE_UNIT"
  grep -q -- "--model-path $TARGET_REPO" "$STAGE_UNIT" || die "unit rewrite failed"
  KV_SEEN="$(grep -c -- '--kv-cache-dtype fp8_e4m3' "$STAGE_UNIT" || true)"
  case "$CHOICE" in
    fp8|uncensored-fp8)
      [ "$KV_SEEN" -eq 1 ] || die "the rewritten unit lost the fp8 KV cache this target needs (it costs half the pool); re-run MODEL_CHOICE=$CHOICE ./install.sh" ;;
    *)
      [ "$KV_SEEN" -eq 0 ] || die "the rewritten unit still forces an fp8 KV cache; re-run MODEL_CHOICE=$CHOICE ./install.sh" ;;
  esac
  grep -qE -- '--mem-fraction-static [0-9.]+' "$STAGE_UNIT" || die "the rewritten unit lost its memory fraction"
  if grep -q -- '--revision ' "$TARGET_UNIT"; then
    grep -q -- "--revision $TARGET_REV" "$STAGE_UNIT" || die "unit revision rewrite failed"
  fi
  # A unit that already says all this keeps its date: rewritten, it read as an update the
  # serving engine had not taken, and the cockpit offered a restart for nothing (found in
  # review, 2026-10-17). Kolibri, Nemotron and the image lane compare the same way.
  if cmp -s "$STAGE_UNIT" "$TARGET_UNIT"; then
    echo "unit unchanged: it already loads $CHOICE"
  else
    diff "$STAGE_UNIT" "$TARGET_UNIT" || true   # show exactly what changes
    sudo install -m 644 "$STAGE_UNIT" "$TARGET_UNIT"
    # the switch rewrote the unit: an update that waited, rendered against the unit before, is caduque, and is
    # cleared here, as every lane beside clears its wait at its own switch (decision 1408; review of v1.52.0,
    # 2026-11-19: it stayed, and a Load after this switch downloaded what it lacked then dropped it)
    if [ -f "$CONFIG_DIR/pending/$TARGET_UNIT_NAME" ]; then
      unit_pending_clear "$TARGET_UNIT"
      echo "the update that waited for this lane was dropped: $TARGET_UNIT_NAME changed since it was rendered; ./install.sh writes it again"
    fi
  fi
else
  # Flash keeps its serving flags in a plain launch script; the unit just
  # points at it. Refresh --revision there.
  FLASH_LAUNCH="$CONFIG_DIR/launch-flash.sh"
  [ -f "$FLASH_LAUNCH" ] || die "flash launch script missing ($FLASH_LAUNCH): re-run MODEL_CHOICE=flash ./install.sh"
  # The two flash checkpoints differ in more than their name: the RadixArk
  # export is plain NVFP4 and declares it, NVIDIA's is a mixed-precision export
  # that resolves its own scheme and needs the MoE runner pinned. So the switch
  # rewrites the model path, the revision and that one flag pair together.
  # These two strings are install.sh's FLASH_QUANT_ARGS, and CI asserts they
  # still match it character for character. They carry BOTH the target's scheme
  # and the draft's, because both belong to the checkpoint: the NVFP4 tree's
  # in-checkpoint MTP tensors are BF16, NVIDIA's are fp8 block-scaled.
  case "$CHOICE" in
    flash-nvda) NEW_QUANT="--moe-runner-backend flashinfer_cutlass " ;;
    *)          NEW_QUANT="--quantization modelopt_fp4 --speculative-draft-model-quantization unquant " ;;
  esac
  rewrite_flash_launcher "$FLASH_LAUNCH" "$TARGET_REPO" "$TARGET_REV" "$NEW_QUANT" > "$TMP_UNIT"
  grep -qF -- "$NEW_QUANT--fp4-gemm-backend" "$TMP_UNIT" \
    || die "the launcher's quantization line did not take this checkpoint's flags; re-run MODEL_CHOICE=$CHOICE ./install.sh"
  grep -q -- "--model-path $TARGET_REPO " "$TMP_UNIT" || die "the flash launch script does not serve $TARGET_REPO (hand-edited?); re-run MODEL_CHOICE=$CHOICE ./install.sh"

  grep -q -- "--revision $TARGET_REV" "$TMP_UNIT" || die "launch script revision rewrite failed"
  bash -n "$TMP_UNIT" || die "rewritten launch script does not parse"
  # the same for the launch script the flash unit runs (see the unit's case above)
  if cmp -s "$TMP_UNIT" "$FLASH_LAUNCH"; then
    echo "launch script unchanged: it already loads $CHOICE"
  else
    diff "$TMP_UNIT" "$FLASH_LAUNCH" || true   # show exactly what changes
    install -m 755 "$TMP_UNIT" "$FLASH_LAUNCH"
    # the switch rewrote the launcher: an update that waited for it is caduque, and is cleared here, as every
    # lane beside clears its wait at its own switch (decision 1408; review of v1.52.0, 2026-11-19: it stayed, and
    # a Load after this switch downloaded what it lacked then dropped it)
    if [ -f "$CONFIG_DIR/pending/launch-flash.sh" ] || [ -f "$CONFIG_DIR/pending/$TARGET_UNIT_NAME" ]; then
      unit_pending_clear "$FLASH_LAUNCH"
      unit_pending_clear "$FLASH_UNIT"
      echo "the update that waited for this lane was dropped: the launch script changed since it was rendered; ./install.sh writes it again"
    fi
  fi
  # A switch changes the checkpoint, never the serving image: only install.sh builds
  # that. So a box that pulled a repo whose image pin moved would keep serving the old
  # one, silently, and the image is where the kernel fixes live (v1.6: the merged
  # sm_121 kernel). Say so instead of letting it pass.
  # Since v1.8 the launcher names the pinned official image (a digest); a box
  # installed with OVERLAY_FLASH=1 before v1.18.7 retired it names a local
  # qwen38-flash:tag, and is told the same thing: adopt the pin.
  # Same class as OC_WINDOW below: the empty case is handled three lines down
  # ([ -n "$LAUNCH_IMAGE" ]), and without || true set -e never lets it get there.
  LAUNCH_IMAGE="$(grep -oE '^[[:space:]]*(qwen38-flash:[A-Za-z0-9._-]+|lmsysorg/sglang@sha256:[0-9a-f]{64})' "$FLASH_LAUNCH" | tr -d '[:space:]' | head -1 || true)"
  WANT_IMAGE="${FLASH_IMAGE:-}"
  if [ -n "$LAUNCH_IMAGE" ] && [ -n "$WANT_IMAGE" ] && [ "$LAUNCH_IMAGE" != "$WANT_IMAGE" ]; then
    printf '\n\033[1;33mWARNING:\033[0m this box serves flash from %s, but this repo pins %s.\n' \
      "$LAUNCH_IMAGE" "$WANT_IMAGE"
    printf '         The serving image is where the kernel fixes live, and only the installer adopts it.\n'
    printf '         Run ./install.sh to adopt the pinned image before relying on this lane.\n'
  fi
fi

# 4) exactly one serving unit enabled at boot; the opencode default model
#    follows the switch (providers themselves are kept as installed).
if [ -f "$OTHER_UNIT" ] && systemctl is-enabled --quiet "$OTHER_UNIT_NAME" 2>/dev/null; then
  echo "disabling $OTHER_UNIT_NAME at boot (unit file kept for switching back)"
  sudo systemctl disable "$OTHER_UNIT_NAME"
fi
# The image lane is the third serving unit, the video lane the fourth. Leaving either
# enabled while a text lane is made the boot lane would bring two engines up at the
# next reboot.
if [ -f "$IMAGE_UNIT" ] && systemctl is-enabled --quiet "$IMAGE_UNIT_NAME" 2>/dev/null; then
  echo "disabling $IMAGE_UNIT_NAME at boot (unit file kept for switching back)"
  sudo systemctl disable "$IMAGE_UNIT_NAME"
fi
if [ -f "$VIDEO_UNIT" ] && systemctl is-enabled --quiet "$VIDEO_UNIT_NAME" 2>/dev/null; then
  echo "disabling $VIDEO_UNIT_NAME at boot (unit file kept for switching back)"
  sudo systemctl disable "$VIDEO_UNIT_NAME"
fi
# The Kolibri lane is the fifth: the same reason.
if [ -f "$KOLIBRI_UNIT" ] && systemctl is-enabled --quiet "$KOLIBRI_UNIT_NAME" 2>/dev/null; then
  echo "disabling $KOLIBRI_UNIT_NAME at boot (unit file kept for switching back)"
  sudo systemctl disable "$KOLIBRI_UNIT_NAME"
fi
# The Nemotron lanes are the sixth and the seventh: the same reason.
if [ -f "$NEMOTRON_UNIT" ] && systemctl is-enabled --quiet "$NEMOTRON_UNIT_NAME" 2>/dev/null; then
  echo "disabling $NEMOTRON_UNIT_NAME at boot (unit file kept for switching back)"
  sudo systemctl disable "$NEMOTRON_UNIT_NAME"
fi
if [ -f "$NEMOTRON_SUPER_UNIT" ] && systemctl is-enabled --quiet "$NEMOTRON_SUPER_UNIT_NAME" 2>/dev/null; then
  echo "disabling $NEMOTRON_SUPER_UNIT_NAME at boot (unit file kept for switching back)"
  sudo systemctl disable "$NEMOTRON_SUPER_UNIT_NAME"
fi
# The Qwen3.8 Flash-Next FP4-Local lane is the eighth: the same reason.
if [ -f "$FLASH_LOCAL_UNIT" ] && systemctl is-enabled --quiet "$FLASH_LOCAL_UNIT_NAME" 2>/dev/null; then
  echo "disabling $FLASH_LOCAL_UNIT_NAME at boot (unit file kept for switching back)"
  sudo systemctl disable "$FLASH_LOCAL_UNIT_NAME"
fi
# The Ornith 1.5 122B lane is the ninth: the same reason.
if [ -f "$ORNITH_122B_UNIT" ] && systemctl is-enabled --quiet "$ORNITH_122B_UNIT_NAME" 2>/dev/null; then
  echo "disabling $ORNITH_122B_UNIT_NAME at boot (unit file kept for switching back)"
  sudo systemctl disable "$ORNITH_122B_UNIT_NAME"
fi
# The Qwen3.8 Flash-Next Unsloth GGUF lane is the tenth: the same reason.
if [ -f "$FLASH_UNSLOTH_UNIT" ] && systemctl is-enabled --quiet "$FLASH_UNSLOTH_UNIT_NAME" 2>/dev/null; then
  echo "disabling $FLASH_UNSLOTH_UNIT_NAME at boot (unit file kept for switching back)"
  sudo systemctl disable "$FLASH_UNSLOTH_UNIT_NAME"
fi
# The MiniMax-H3-Image lane is the eleventh: the same reason.
if [ -f "$IMAGE_H3_UNIT" ] && systemctl is-enabled --quiet "$IMAGE_H3_UNIT_NAME" 2>/dev/null; then
  echo "disabling $IMAGE_H3_UNIT_NAME at boot (unit file kept for switching back)"
  sudo systemctl disable "$IMAGE_H3_UNIT_NAME"
fi
# The NVIDIA Nemotron Transcribe lane is the twelfth: the same reason.
if [ -f "$TRANSCRIBE_UNIT" ] && systemctl is-enabled --quiet "$TRANSCRIBE_UNIT_NAME" 2>/dev/null; then
  echo "disabling $TRANSCRIBE_UNIT_NAME at boot (unit file kept for switching back)"
  sudo systemctl disable "$TRANSCRIBE_UNIT_NAME"
fi
# The Liquid AI LFM2.5 24B-A2B lane is the thirteenth: the same reason.
if [ -f "$LFM25_UNIT" ] && systemctl is-enabled --quiet "$LFM25_UNIT_NAME" 2>/dev/null; then
  echo "disabling $LFM25_UNIT_NAME at boot (unit file kept for switching back)"
  sudo systemctl disable "$LFM25_UNIT_NAME"
fi
# The Liquid AI LFM3 lane is the fourteenth: the same reason.
if [ -f "$LFM3_UNIT" ] && systemctl is-enabled --quiet "$LFM3_UNIT_NAME" 2>/dev/null; then
  echo "disabling $LFM3_UNIT_NAME at boot (unit file kept for switching back)"
  sudo systemctl disable "$LFM3_UNIT_NAME"
fi
sudo systemctl enable "$TARGET_UNIT_NAME" >/dev/null 2>&1 || sudo systemctl enable "$TARGET_UNIT_NAME"
text_lane_note "$TARGET_UNIT_NAME"
sudo systemctl daemon-reload
# 4b) the keepalive proxy's one-prompt ceiling follows the lane (v1.5.6 contract, see
#     install.sh: flash 250000 tokens by default, the 27B lane none), applied now so the
#     proxy matches the lane that serves after the restart below. Written as a systemd
#     drop-in, never by editing the unit in place: an in-place sed needs a sudoers
#     wildcard to stay valid, and a wildcard on sed is root (its w command writes any
#     file). The drop-in overrides the unit's Environment line; install.sh bakes the
#     ceiling into the unit and removes this drop-in, so either path converges.
KA_UNIT="/etc/systemd/system/qwen38-keepalive.service"
if [ -f "$KA_UNIT" ] && grep -q '^Environment=FLASH_PROMPT_CEILING_TOKENS=' "$KA_UNIT"; then
  # The proxy of this unit follows the lane that serves by itself (v6.25): nothing to move,
  # and nothing to restart. Moving it here applied the target's ceiling while the old lane
  # still served (a 27B serving 1M refused every prompt past 250,000 until the next boot),
  # and the restart cut every stream in flight through :30001 (found in review, 2026-09-24).
  echo "keepalive proxy: its one-prompt ceiling follows the lane that serves; left as it is"
elif [ -f "$KA_UNIT" ]; then
  # A unit from before v1.18.7, whose proxy only knows one ceiling: the old way, until a
  # ./install.sh writes the new unit.
  CEIL=0
  [ "$TARGET_LANE" = "flash" ] && CEIL="${PROMPT_CEILING_TOKENS:-250000}"
  [[ "$CEIL" =~ ^[0-9]+$ ]] || die "PROMPT_CEILING_TOKENS must be a number (got '$CEIL')"
  printf '[Service]\nEnvironment=PROMPT_CEILING_TOKENS=%s\n' "$CEIL" > "$STAGE_CEIL"
  sudo install -m 644 -D "$STAGE_CEIL" "/etc/systemd/system/qwen38-keepalive.service.d/ceiling.conf"
  sudo systemctl daemon-reload
  sudo systemctl restart qwen38-keepalive.service
  echo "keepalive proxy: one-prompt ceiling ${CEIL} tokens for the $TARGET_LANE lane"
fi
# Both files when present: the generated artifact (CONFIG_DIR) and the config
# opencode actually reads (~/.config/opencode). Updating only the artifact left
# the real default on the previous lane (seen on the reference box 2026-08-30).
# A box installed with --no-opencode (marker file) is left entirely alone.
if [ -f "$CONFIG_DIR/opencode.off" ]; then
  echo "opencode integration is off on this box (./install.sh --no-opencode): default model left alone"
else
# The limits follow the target, not just the model name. opencode compacts when a
# conversation reaches limit.context, and a limit larger than the lane can serve
# means the proxy refuses the request with a 400 before opencode ever compacts:
# a hard failure mid-session. Switching a 27B box in 1M mode (700,000) to the
# flash lane (200,000 ceiling) is exactly that, and until 2026-09-08 the switch
# left the old numbers in place. The table is oc-limits.sh, shared with
# install.sh so the two cannot disagree; the tier or context mode is read from
# the invocation this switch just wrote, which is what the box will serve.
# The label is read but not used here: the picker name is set by the python
# block below, from the same table's LABEL dict, so it is read into a throwaway.
# what the opencode server reads, before this switch writes any of it (see 4d); a missing
# file is part of the answer, not a failure (install.sh's first installs died on the same
# line under pipefail)
oc_configs_sum(){ { cat "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json" 2>/dev/null || true; } | sha256sum; }
SW_OC_SUM_BEFORE="$(oc_configs_sum)"
if read -r SW_CTX SW_OUT _SW_LABEL \
     <<<"$("$REPO_DIR/oc-limits.sh" "$CHOICE" --from "$INVOCATION")" \
   && [ -n "${SW_CTX:-}" ]; then
  if [ "$TARGET_LANE" = "flash" ]; then SW_PROV=flashnext; SW_MODEL=qwen3.8-flash-next
  else SW_PROV=qwen38; SW_MODEL=qwen3.8-27b; fi
  # The compaction block is global, so it is sized from the target's context and
  # rewritten with the limits: a switch that moved the threshold and left the old
  # preserve_recent_tokens behind would compact the new lane against the old
  # lane's window. Same table, same call as install.sh.
  SW_KEEP="$("$REPO_DIR/oc-limits.sh" --preserve "$SW_CTX")" || SW_KEEP=""
  for OC_JSON in "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json"; do
    [ -f "$OC_JSON" ] || continue
    python3 "$REPO_DIR/oc-merge-limits.py" "$OC_JSON" "$SW_PROV" "$SW_MODEL" "$SW_CTX" "$SW_OUT" \
      || echo "NOTE: could not update the limits in $OC_JSON; check them by hand ($SW_CTX/$SW_OUT)"
    [ -n "$SW_KEEP" ] && { python3 "$REPO_DIR/oc-merge-limits.py" "$OC_JSON" --compaction "$SW_KEEP" \
      || echo "NOTE: could not update the compaction block in $OC_JSON; check it by hand ($SW_KEEP)"; }
  done
else
  echo "NOTE: oc-limits.sh gave no limits for $CHOICE; the opencode limits were left as they were"
fi
for OC_JSON in "$CONFIG_DIR/opencode.json" "$HOME/.config/opencode/opencode.json"; do
  [ -f "$OC_JSON" ] || continue
  # The window is read from the INVOCATION, not from the unit: on the flash lane
  # the unit is a thin pointer at the launcher and carries no --context-length,
  # so reading the unit labelled every flash switch "(local)" instead of
  # "(local, 262K)". Default model and picker name share one table in
  # oc-point-default.py with install.sh, so the three spellings cannot drift.
  # || true, and it is the whole reason a switch to a native 27B target died
  # here on 2026-09-14: a native unit carries NO --context-length (262144 is the
  # engine default), grep exits 1 on zero matches, and under set -e with
  # pipefail that killed the script silently, three lines before the end. The
  # lane had already been swapped and enabled, so the box served the 27B while
  # opencode's default model still pointed at flashnext: a switch that reported
  # failure and had in fact done nine tenths of the job. install.sh carries the
  # same pipeline and was fixed the same way on 2026-09-12; this copy was not.
  OC_WINDOW="$(grep -oE -- '--context-length [0-9]+' "$INVOCATION" 2>/dev/null | awk '{print $2}' | head -1 || true)"
  python3 "$REPO_DIR/oc-point-default.py" "$OC_JSON" "$TARGET_LANE" "$CHOICE" "$OC_WINDOW" \
    || echo "NOTE: could not update $OC_JSON (hand-edited?); set its \"model\" field yourself"
done
# 4d) the opencode server, if this box runs one, is restarted so the numbers above
#     actually reach it. `opencode serve` reads opencode.json ONCE at startup and
#     never again: measured 2026-09-13, the file on disk said 225,000 while the
#     running server still answered 175,000 on /config. Every switch before this
#     rewrote the limits correctly and left the Agent tab compacting against the
#     previous lane's window, which is a silent version of the mid-session 400
#     this whole table exists to prevent.
#     Only when they changed: a switch between two targets of one lane (stock to
#     uncensored) writes the same numbers, and the restart ended the turn the Agent tab
#     was in all the same (found in review, 2026-09-24).
if systemctl list-unit-files opencode-web.service >/dev/null 2>&1 \
   && systemctl is-active --quiet opencode-web.service; then
  if [ "$(oc_configs_sum)" != "$SW_OC_SUM_BEFORE" ]; then
    sudo systemctl restart opencode-web.service \
      && echo "opencode-web.service restarted so it reads the new limits" \
      || echo "NOTE: could not restart opencode-web.service; restart it by hand or the Agent view keeps the old limits"
  else
    echo "opencode-web.service kept: the limits it reads did not change"
  fi
fi
fi

# 4f) pi and omp (docs/pi.md): the same dynamic the block above gives opencode.
#     Their configs carry no second table of names and limits: pi-gen.py
#     regenerates them from the opencode.json this switch just rewrote, as
#     artifacts under ~/.config/qwen38 (pi/ JSON, omp/ YAML converted from it).
#     No agent directory is ever written: the agents' own files are the user's
#     copies, so re-copy them after a switch to pick up the new lane. Only on a
#     box that asked for them once (./pi-gen.py made the directories): a switch
#     writes no file for an agent nobody uses.
if [ -d "$CONFIG_DIR/pi" ] || [ -d "$CONFIG_DIR/omp" ]; then
  python3 "$REPO_DIR/pi-gen.py" --config "$CONFIG_DIR/opencode.json" --out "$CONFIG_DIR" \
    || echo "NOTE: could not regenerate the pi/omp artifacts; run ./pi-gen.py by hand"
fi

RUNNING=""
systemctl is-active --quiet "$OTHER_UNIT_NAME" 2>/dev/null && RUNNING="$OTHER_UNIT_NAME"
systemctl is-active --quiet "$IMAGE_UNIT_NAME" 2>/dev/null && RUNNING="$IMAGE_UNIT_NAME"
systemctl is-active --quiet "$VIDEO_UNIT_NAME" 2>/dev/null && RUNNING="$VIDEO_UNIT_NAME"
systemctl is-active --quiet "$KOLIBRI_UNIT_NAME" 2>/dev/null && RUNNING="$KOLIBRI_UNIT_NAME"
systemctl is-active --quiet "$NEMOTRON_UNIT_NAME" 2>/dev/null && RUNNING="$NEMOTRON_UNIT_NAME"
systemctl is-active --quiet "$NEMOTRON_SUPER_UNIT_NAME" 2>/dev/null && RUNNING="$NEMOTRON_SUPER_UNIT_NAME"
systemctl is-active --quiet "$FLASH_LOCAL_UNIT_NAME" 2>/dev/null && RUNNING="$FLASH_LOCAL_UNIT_NAME"
systemctl is-active --quiet "$ORNITH_122B_UNIT_NAME" 2>/dev/null && RUNNING="$ORNITH_122B_UNIT_NAME"
systemctl is-active --quiet "$FLASH_UNSLOTH_UNIT_NAME" 2>/dev/null && RUNNING="$FLASH_UNSLOTH_UNIT_NAME"
systemctl is-active --quiet "$IMAGE_H3_UNIT_NAME" 2>/dev/null && RUNNING="$IMAGE_H3_UNIT_NAME"
systemctl is-active --quiet "$TRANSCRIBE_UNIT_NAME" 2>/dev/null && RUNNING="$TRANSCRIBE_UNIT_NAME"
systemctl is-active --quiet "$LFM25_UNIT_NAME" 2>/dev/null && RUNNING="$LFM25_UNIT_NAME"
systemctl is-active --quiet "$LFM3_UNIT_NAME" 2>/dev/null && RUNNING="$LFM3_UNIT_NAME"
printf '\n\033[1;32mSwitch queued: %s (%s)\033[0m\n' "$TARGET_REPO" "$TARGET_UNIT_NAME"
# The target is the operator's choice from here (2026-10-17): an installed.json that a run of
# v1.36.0 or later wrote records it, and a switch back to the recommended target erases the
# entry (conformity.py choose). Never fatal: the next ./install.sh classes what the unit serves.
python3 "$REPO_DIR/conformity.py" choose --config-dir "$CONFIG_DIR" --repo-dir "$REPO_DIR" \
  "target.$TARGET_LANE" "$CHOICE" || true
if [ -n "$RUNNING" ]; then
  echo "Effective after:  sudo systemctl stop $RUNNING && sudo systemctl start $TARGET_UNIT_NAME   (or next reboot)"
else
  echo "Effective after:  sudo systemctl restart $TARGET_UNIT_NAME   (or next reboot)"
fi
# The limits written above are the ones this mode DESERVES, not the ones this
# box can serve: the pool is only known once the engine has booted, and it moves
# with the checkpoint and the image. install.sh fits them at install time; a
# switch cannot, because the engine it just pointed at is not up yet. Say so,
# or the operator learns it from a proxy 400 late in a session (or, on a box
# with the cockpit, from its banner: this is the "Fit the limits to this
# engine" button). Measured three times on 2026-09-18 while testing switches.
echo "Then fit them:    python3 oc-fit-limits.py   (the limits above are this target's"
echo "                  nominal pair; the pool is only known once the engine is up)"
# The first target a hint names is a standard one: the project advises the standard
# checkpoints. The abliterated targets stay named, last in the parentheses, so that they are
# known without ever being the default. This line named `uncensored` first after every
# switch to stock.
case "$CHOICE" in
  stock)      echo "Switch back:      ./switch-model.sh flash   (or fp8, image, video, uncensored)" ;;
  uncensored) echo "Switch back:      ./switch-model.sh stock   (or fp8, flash)" ;;
  fp8)        echo "Switch back:      ./switch-model.sh stock   (or flash, uncensored, uncensored-fp8)" ;;
  uncensored-fp8) echo "Switch back:      ./switch-model.sh fp8   (or stock, flash, uncensored)" ;;
  arete)      echo "Switch back:      ./switch-model.sh stock   (or fp8, flash, uncensored)" ;;
  flash)      echo "Switch back:      ./switch-model.sh stock   (or fp8, flash-nvda, uncensored, flash-uncensored)" ;;
  flash-nvda) echo "Switch back:      ./switch-model.sh flash   (or stock, fp8, flash-uncensored, uncensored)" ;;
  flash-uncensored) echo "Switch back:      ./switch-model.sh flash   (or flash-nvda, stock, fp8, uncensored)" ;;
esac
