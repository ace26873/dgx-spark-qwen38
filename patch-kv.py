#!/usr/bin/env python3
"""Declare the 27B lane's fp8 KV cache in a cached checkpoint that declares none.

The lane's NVFP4 export (RadixArk/Qwen3.8-27B-NVFP4) declares its KV cache in its
own configs: kv_cache_scheme {"dynamic": false, "num_bits": 8, "type": "float"}
under quantization_config in config.json, kv_cache_quant_algo "FP8" in
hf_quant_config.json. Its unit passes no --kv-cache-dtype, so SGLang reads that
declaration (ModelOpt's from_config) and gives the target an fp8_e4m3 cache at
scale 1.0, while the DFlash2 draft keeps its own bf16 one. A ModelOpt export that
declares none (atakans/Arete-27B-NVFP4) gets a bf16 cache and about half the pool.
The flag the FP8 targets pass, --kv-cache-dtype fp8_e4m3, is not the same path:
it moves the draft's cache to fp8 too (kv_cache_dtype.py of the pinned image), and
the setting the draft's runner and the KV scales' loader read says fp8_e4m3 where
the lane's says auto (model_runner.py). So this writes the lane's declaration into the checkpoint's two files, the way the A/B of
2026-11-14 served Arete: target cache float8_e4m3fn, draft cache bfloat16, pool
905,941 tokens against the lane's 882,224 at the same unit.

Each file is written atomically through a temp file beside it. A cached file is
a link into the repo's blobs/, and the link is what gets replaced: the blob stays
the checkpoint's own bytes, as patch-yarn.py leaves it. Idempotent: a file that
declares this already is not rewritten. A checkpoint that declares another KV
scheme, or whose quantization is not ModelOpt's, is refused (exit 3): what it
chose is not overwritten. patch-yarn.py --restore puts back the config.json saved
before the YaRN patch, which may predate this declaration, so whatever writes the
window's configs runs this after patch-yarn.py.

Usage: patch-kv.py [--check] <hf_cache_dir> <repo_id> <revision>
The revision is a sha or a ref name, and that exact snapshot is used: no fallback
to another one, whose config the engine would not read. Exit codes: 0 declared
(now or already), 1 usage or no such snapshot, 2 not declared (--check only),
3 refused.
"""
import json
import os
import sys

# The lane's declaration, as RadixArk/Qwen3.8-27B-NVFP4 @ 52d1adc writes it.
KV_SCHEME = {"dynamic": False, "num_bits": 8, "type": "float"}
KV_ALGO = "FP8"


def snapshot(hf_cache: str, repo: str, revision: str) -> str:
    repo_dir = f"{hf_cache}/hub/models--{repo.replace('/', '--')}"
    ref = f"{repo_dir}/refs/{revision}"
    if os.path.isfile(ref):  # a ref name (e.g. 'main') -> the sha it points at
        revision = open(ref, encoding="utf-8").read().strip()
    snap = f"{repo_dir}/snapshots/{revision}"
    if not os.path.isfile(f"{snap}/config.json"):
        sys.exit(f"no cached config.json for {repo} @ {revision[:12]} under {hf_cache}: "
                 "fetch the checkpoint first")
    return snap


def read(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write(path: str, data: dict) -> None:
    # the same encoding both ways, and a rename: a kill mid-write leaves the old file whole
    tmp = path + ".tmp-kv"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def state(cfg: dict, hfq: dict | None) -> tuple[str, str]:
    """("declared" | "absent" | "refused", why) for config.json and hf_quant_config.json together."""
    qc = cfg.get("quantization_config")
    if not isinstance(qc, dict) or qc.get("quant_method") != "modelopt":
        return "refused", "config.json declares no ModelOpt quantization: no KV declaration of this form applies"
    scheme, algo = qc.get("kv_cache_scheme"), qc.get("kv_cache_quant_algo")
    if scheme not in (None, KV_SCHEME) or algo not in (None, KV_ALGO):
        return "refused", f"config.json declares another KV cache already ({scheme or algo}): left as it chose"
    done = scheme == KV_SCHEME
    if hfq is not None:
        q = hfq.get("quantization")
        if not isinstance(q, dict):
            return "refused", "hf_quant_config.json has no quantization section"
        if q.get("kv_cache_quant_algo") not in (None, KV_ALGO):
            return "refused", f"hf_quant_config.json declares another KV cache already ({q['kv_cache_quant_algo']})"
        done = done and q.get("kv_cache_quant_algo") == KV_ALGO
    return ("declared", "") if done else ("absent", "")


def main() -> None:
    args = sys.argv[1:]
    check = bool(args) and args[0] == "--check"
    if check:
        args.pop(0)
    if len(args) != 3:
        sys.exit(__doc__)
    hf_cache, repo, revision = args
    snap = snapshot(hf_cache, repo, revision)
    cfg_path, hfq_path = f"{snap}/config.json", f"{snap}/hf_quant_config.json"
    cfg = read(cfg_path)
    hfq = read(hfq_path) if os.path.isfile(hfq_path) else None
    st, why = state(cfg, hfq)
    if st == "refused":
        print(f"{repo}: {why}", file=sys.stderr)
        sys.exit(3)
    if st == "declared":
        print(f"fp8 KV cache already declared: {snap}")
        return
    if check:
        print(f"no KV cache declared (the engine would serve a bf16 one): {snap}")
        sys.exit(2)
    # each file only where it lacks the declaration, hf_quant_config.json first: config.json is
    # the one the engine reads first, and the one whose declaration says the job is done
    if hfq is not None and hfq["quantization"].get("kv_cache_quant_algo") != KV_ALGO:
        hfq["quantization"]["kv_cache_quant_algo"] = KV_ALGO
        write(hfq_path, hfq)
    if cfg["quantization_config"].get("kv_cache_scheme") != KV_SCHEME:
        cfg["quantization_config"]["kv_cache_scheme"] = dict(KV_SCHEME)
        write(cfg_path, cfg)
    print(f"fp8 KV cache declared, as the lane's NVFP4 declares it: {snap}")


if __name__ == "__main__":
    main()
