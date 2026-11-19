#!/usr/bin/env bash
# Before an engine starts: the memory the last one held has to be back.
#
# A crashed engine normally hands its memory back within seconds (measured on the reference
# box, 2026-10-01: the 27B's 92 GB back 2 to 5 s after its scheduler died, and systemd
# relaunches 15 s after the container exits). sglang#40948 is the other case: the driver
# kept a dead engine's memory, the next start piled its allocations on top of it, the driver
# logged NV_ERR_NO_MEMORY and its lock seized, and the whole machine froze until a power
# cycle, 9.5 h later. Waiting longer does not help there; not starting does. So this waits,
# bounded, and refuses a start only when that is what it finds:
#   - GPU memory still held by what is left of an engine: an SGLang or a vLLM process, a
#     llama.cpp lane's llama-server (in its own container), or a process the driver still lists
#     and the system no longer has;
#   - memory the driver keeps that no process accounts for: on this unified-memory machine
#     a GPU allocation is in none of the kernel's own categories, so MemTotal minus all of
#     them is what the driver holds (1.1 GiB with no engine and a desktop session, the
#     engine's size plus 1 to 3 GiB with one, measured);
#   - a driver that does not answer nvidia-smi.
# Anything else never blocks a start, another program's GPU memory included (the journal
# names it): the engine sizes its pool to what it finds, as it always did.
#
# One attempt waits at most 60 s, plus one nvidia-smi call of at most 12 s and, when the driver
# lists a llama-server, one docker call of at most 7 s: systemd kills
# a start step after TimeoutStartSec, 90 s on every engine unit, and the waiting goes on
# across attempts, since each unit starts again 15 s after a failed one. The lines it
# writes start with "preflight:", and the cockpit reads them to say why a lane is not up.
#
# usage: engine-preflight.sh <unit name>
set -u
LANE="${1:-engine}"
WAIT_S="${QWEN38_PREFLIGHT_WAIT_S:-60}"
HELD_MIB_MAX="${QWEN38_PREFLIGHT_HELD_MIB:-2048}"
ORPHAN_GIB_MAX="${QWEN38_PREFLIGHT_ORPHAN_GIB:-16}"
MEMINFO="${QWEN38_PREFLIGHT_MEMINFO:-/proc/meminfo}"
SMI="${QWEN38_PREFLIGHT_SMI:-nvidia-smi}"
PROC="${QWEN38_PREFLIGHT_PROC:-/proc}"
STEP_S="${QWEN38_PREFLIGHT_STEP_S:-2}"
SMI_TIMEOUT_S="${QWEN38_PREFLIGHT_SMI_TIMEOUT_S:-10}"
# The llama.cpp lanes' containers, and the docker that says their ids (2026-11-12; every lane
# of that engine since 2026-11-19, the transcribe lane's included): each lane the registry calls
# llamacpp, its unit's name without ".service" (lane_registry.py, beside this script in the
# repository, so a lane added there is covered here with no list to forget). The copy the units
# run (~/.config/qwen38/engine-preflight.sh) has no registry beside it: it reads the units'
# directory, where each llama.cpp lane's unit runs the files' check (llamacpp-files.sh, the
# marker the uninstall reads too, llamacpp-lane.sh) and names its container (--name).
# QWEN38_PREFLIGHT_LLAMACPP_CONTAINER overrides both with one container, for a test.
llamacpp_containers(){
  local dir reg u out
  if [ -n "${QWEN38_PREFLIGHT_LLAMACPP_CONTAINER:-}" ]; then
    printf '%s\n' "$QWEN38_PREFLIGHT_LLAMACPP_CONTAINER"; return 0
  fi
  dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  reg="${QWEN38_PREFLIGHT_REGISTRY:-$dir/lane_registry.py}"
  if [ -f "$reg" ] && command -v python3 >/dev/null 2>&1; then
    out="$(python3 -I -c 'import sys; sys.path.insert(0, sys.argv[1])
import lane_registry as r
print(" ".join(r.LANES[l]["unit"][:-len(".service")] for l in r.lanes(engine="llamacpp")))' \
      "$(dirname "$reg")" 2>/dev/null)" && [ -n "$out" ] && { printf '%s\n' "$out"; return 0; }
  fi
  for u in $(grep -lsF 'llamacpp-files.sh' "${QWEN38_PREFLIGHT_UNITS_DIR:-/etc/systemd/system}"/qwen38-*.service 2>/dev/null); do
    grep -hoE -- '--name qwen38-[a-z0-9-]+' "$u" 2>/dev/null | awk '{print $2}'
  done | sort -u
}
LLAMACPP_CONTAINERS="$(llamacpp_containers)"
DOCKER="${QWEN38_PREFLIGHT_DOCKER:-docker}"
DOCKER_TIMEOUT_S="${QWEN38_PREFLIGHT_DOCKER_TIMEOUT_S:-5}"

# The id of each llama.cpp lane's container, one "{{.Name}} {{.Id}}" line each, in one docker
# call; a container docker does not know says nothing. gpu_holders keeps them in llamacpp_ids.
llamacpp_container_ids(){
  local out line name id
  # a container docker does not know answers on stderr and a non-zero status, with the others'
  # lines still on stdout: what it said is kept, not wiped
  out="$(timeout -k 2 "$DOCKER_TIMEOUT_S" "$DOCKER" inspect -f '{{.Name}} {{.Id}}' $LLAMACPP_CONTAINERS 2>/dev/null)" || true
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    name="${line%% *}"; name="${name#/}"
    id="${line#* }"
    case "$id" in ''|*[!0-9a-f]*) continue ;; esac
    llamacpp_ids["$name"]="$id"
  done <<< "$out"
}

# True when the process $1 runs in one of the llama.cpp lanes' containers, whose id its cgroup
# names. Told by its name alone, a llama-server of another program on the GPU (one of the user's
# own) held every lane's start back after 60 s (code review of v1.48.0, 2026-11-12). The ids are
# asked once a pass, and only when the driver lists a llama-server; a docker that does not
# answer says none is ours. No process of these lanes outlives its container docker knows:
# docker drops a container (--rm, rm -f) only once its exit event says its main process, llama-server,
# was reaped, and refuses to while that process still runs, and the kernel kills the other processes of
# a PID namespace when its first one exits, reaping that one only after them (moby daemon/kill.go,
# delete.go and monitor.go at docker 29.6.2, the box's, read 2026-11-12).
ours_llamacpp() {
  local c
  if [ "${llamacpp_asked:-}" != 1 ]; then
    llamacpp_container_ids; llamacpp_asked=1
  fi
  for c in $LLAMACPP_CONTAINERS; do
    [ -n "${llamacpp_ids[$c]:-}" ] && grep -qF -- "${llamacpp_ids[$c]}" "$PROC/$1/cgroup" 2>/dev/null && return 0
  done
  return 1
}

# One line, its fields split on the unit separator (read does not merge two of them when
# one is empty, as it does tabs): MiB held by engine remnants, MiB held in all, the
# remnants, the others. Returns 1 when the driver does not answer.
gpu_holders() {
  local out line pid name mib comm cmd eng=0 all=0 rem="" oth=""
  local -A llamacpp_ids=()
  local llamacpp_asked=""
  out="$(timeout -k 2 "$SMI_TIMEOUT_S" "$SMI" --query-compute-apps=pid,process_name,used_memory \
           --format=csv,noheader,nounits 2>/dev/null)" || return 1
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    pid="${line%%,*}"; pid="${pid// /}"
    mib="${line##*,}"; mib="${mib// /}"
    name="${line#*,}"; name="${name%,*}"; name="${name# }"     # a name may hold commas
    case "$pid" in ''|*[!0-9]*) continue ;; esac
    case "$mib" in ''|*[!0-9]*) continue ;; esac               # [N/A]: nothing to count
    all=$((all + mib))
    comm="$(cat "$PROC/$pid/comm" 2>/dev/null)"
    cmd="$(tr '\0' ' ' < "$PROC/$pid/cmdline" 2>/dev/null)"
    # SGLang's lanes, and the Kolibri lane's vLLM: its engine core is VLLM::EngineCore (the
    # kernel keeps 15 characters of it) and its API server a python3 running vllm serve. And a
    # llama.cpp lane's llama-server, in that lane's container (the speech lane's is built for
    # the CPU, and the driver lists no process of it).
    if [ ! -d "$PROC/$pid" ] || [[ "$comm" == sglang* ]] || [[ "$cmd" == *sglang* ]] \
       || [[ "$comm" == VLLM::* ]] || [[ "$cmd" == *vllm\ serve* ]] \
       || { [[ "$comm" == llama-server ]] && ours_llamacpp "$pid"; }; then
      eng=$((eng + mib)); rem+="$pid ${comm:-gone} $mib MiB; "
    else
      oth+="$pid ${comm:-$name} $mib MiB; "
    fi
  done <<< "$out"
  printf '%d\x1f%d\x1f%s\x1f%s\n' "$eng" "$all" "$rem" "$oth"
}

# GiB the driver holds beyond what the listed processes hold
orphan_gib() {  # $1: MiB held by the listed processes
  awk -v held="$1" '/^(MemTotal|MemFree|Buffers|Cached|SwapCached|AnonPages|KReclaimable|SUnreclaim|KernelStack|PageTables|SecPageTables|Percpu):/ {v[$1] = $2}
    END {u = v["MemTotal:"] - v["MemFree:"] - v["Buffers:"] - v["Cached:"] - v["SwapCached:"] - v["AnonPages:"] \
             - v["KReclaimable:"] - v["SUnreclaim:"] - v["KernelStack:"] - v["PageTables:"] - v["SecPageTables:"] - v["Percpu:"];
         o = (u - held * 1024) / 1048576; printf "%.1f\n", (o > 0 ? o : 0)}' "$MEMINFO"
}

avail_gib() { awk '/^MemAvailable:/ {printf "%.1f\n", $2 / 1048576}' "$MEMINFO"; }

start=$SECONDS; said=""; oth=""
while :; do
  reason=""
  if ! h="$(gpu_holders)"; then
    reason="the GPU driver does not answer nvidia-smi"
  else
    IFS=$'\x1f' read -r eng all rem oth <<< "$h"
    orphan="$(orphan_gib "$all")"
    if [ "$eng" -gt "$HELD_MIB_MAX" ]; then
      reason="what is left of an engine still holds $((eng / 1024)) GiB of GPU memory: ${rem%; }"
    elif awk -v o="$orphan" -v m="$ORPHAN_GIB_MAX" 'BEGIN {exit !(o > m)}'; then
      reason="the GPU driver holds $orphan GiB that no process accounts for"
    fi
  fi
  if [ -z "$reason" ]; then
    echo "preflight: $LANE starts: no GPU memory held from before, $(avail_gib) GiB available" \
         "($((SECONDS - start)) s)${oth:+; other programs hold GPU memory: ${oth%; }}"
    exit 0
  fi
  if [ $((SECONDS - start)) -ge "$WAIT_S" ]; then
    echo "preflight: NOT starting $LANE: $reason, still after ${WAIT_S} s. Starting into memory" \
         "the driver has not given back is how sglang#40948 froze a DGX Spark until a power" \
         "cycle. Stop what holds it, or reboot if nothing does; systemd tries again in 15 s."
    exit 1
  fi
  [ "$reason" != "$said" ] && echo "preflight: $LANE waits: $reason" && said="$reason"
  sleep "$STEP_S"
done
