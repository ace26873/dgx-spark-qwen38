# shellcheck shell=bash
# sglang-lane.sh: what an SGLang lane installed by an installer of its own shares (install-lfm25.sh, the first;
# install.sh installs the 27B and flash lanes itself, on the same engine). Sourced after lanes.sh, never run. Each
# installer keeps what is its own (its pins, its unit's placeholders, its words) and names itself first, as a vLLM
# lane's does:
#
#   VLANE        the lane (lfm25): its opencode provider, the record of the text lane it took the boot from
#                (lanes.sh), its --no-<lane>, its cache folder
#   VLANE_SHORT  the lane in a sentence, by its official name (Liquid AI LFM2.5 24B-A2B)
#   UNIT, INSTALLED, CONTAINER, CONFIG_DIR, HERE, LOCAL_IMAGE (the image, by its digest); and, for each part
#                below, what it says it reads.
#
# What vllm-lane.sh holds that is no vLLM's (the engine's cache, port and bind, the uninstall of the unit and of the
# clients' entries, the preflight, the room, the fetch of a checkpoint in the lane's image, the proof, and the way a
# rendered unit is put in place) is every text lane's on an engine of its own: it is sourced from there, not copied,
# as the llama.cpp lanes' llamacpp-lane.sh does. What is the SGLang lanes' own is here: the key by the file SGLang
# merges in memory (engine-secrets.sh, as the 27B lane's unit reads it), no key middleware, and the engine's
# uninstall, whose image is the 27B lane's and stays.

# shellcheck source=vllm-lane.sh
. "$HERE/vllm-lane.sh"

# ── The service: the one part that needs root ────────────────────────────────────────────────
# The start scripts the unit runs before every start: the memory guard every lane runs, and the key's yaml written
# again from the api-key file (engine-secrets.sh); and the lane's caches, made here as this user so that docker does
# not make them as root when it mounts them. No key middleware: SGLang asks the key itself (--config).
slane_service_files(){  # $1... the lane's cache folders
  mkdir -p "$CONFIG_DIR" "$@"
  cmp -s "$HERE/engine-preflight.sh" "$CONFIG_DIR/engine-preflight.sh" \
    || install -m 755 "$HERE/engine-preflight.sh" "$CONFIG_DIR/engine-preflight.sh"
  cmp -s "$HERE/engine-secrets.sh" "$CONFIG_DIR/engine-secrets.sh" \
    || install -m 755 "$HERE/engine-secrets.sh" "$CONFIG_DIR/engine-secrets.sh"
  # the yaml the unit mounts, written now: --mount refuses a source that is not there, and the unit writes it
  # again before every start
  bash "$CONFIG_DIR/engine-secrets.sh" "$CONFIG_DIR" \
    || die "the key's yaml could not be written from $CONFIG_DIR/api-key (the line above says why)"
}

# ── --uninstall ──────────────────────────────────────────────────────────────────────────────
# Its container, its image's local tag and its own caches; the checkpoints stay, said. The image is the 27B lane's,
# whose own tag keeps it, as pin_untag says (lanes.sh). The key's yaml stays: the 27B lane's unit writes and reads the
# same file.
slane_uninstall_runtime(){  # $1... the checkpoints' repos
  local r d
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  pin_untag "$VLANE"
  # SGLang writes its caches as root inside the container: what the user cannot remove, sudo does
  rm -rf "$CONFIG_DIR/$VLANE-cache" 2>/dev/null || sudo rm -rf --one-file-system "$CONFIG_DIR/$VLANE-cache"
  for r in "$@"; do
    d="$HF_CACHE/hub/models--${r//\//--}"
    [ -d "$d" ] && echo "checkpoint kept, delete it yourself for the space: rm -rf $d"
  done
  return 0
}
