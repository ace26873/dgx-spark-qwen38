"""Local registry: what lives on this box (models, revisions, images).

Pure parsing/classification helpers + one filesystem scanner. The scanner
touches metadata only (never file contents) and dedups hardlinked blobs by
inode so a multi-revision model is not double-counted.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import time
from pathlib import Path

# Which pin variable serves which checkpoint (kept in lockstep with
# switch-model.sh; the pin parser itself is generic).
PIN_MODELS = {
    "STOCK_REV": "RadixArk/Qwen3.8-27B-NVFP4",
    "UNC_REV": "edp1096/Huihui-RadixArk-Qwen3.8-27B-abliterated-NVFP4",
    "FP8_REV": "Qwen/Qwen3.8-27B-FP8",
    "UNCFP8_REV": "edp1096/Huihui-Qwen3.8-27B-abliterated-FP8",
    "ARETE_REV": "atakans/Arete-27B-NVFP4",
    "FLASH_REV": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
    "FLASH_NVDA_REV": "nvidia/Qwen3.8-Flash-Next-NVFP4",
    "FLASH_UNC_REV": "dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4",
    "DRAFT_REV": "RadixArk/Qwen3.8-27B-DSpark",
    "DRAFT2_REV": "maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal",
    # The other lanes, each pinned by its own installer. Until 2026-10-14 their weights read
    # as "other models", with no pin and no status.
    "IMAGE_MODEL_PIN_REV": "Qwen/Qwen-Image-2.1",
    "IMAGE_TURBO_PIN_REV": "Qwen/Qwen-Image-2.1-Turbo",
    "VIDEO_MODEL_PIN_REV": "MiniMaxAI/MiniMax-H3",
    "VIDEO_TURBO_REV": "larryvrh/MiniMax-H3-Turbo-Lora",
    "KOLIBRI_FP8_REV": "Aleph-Alpha/Kolibri-1",
    "KOLIBRI_NVFP4_REV": "Quantinine/Kolibri-1-NVFP4A16-FP8",
    "NEMOTRON_BOLT_REV": "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB",
    "NEMOTRON_BOLT_DRAFT_REV": "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr",
    "NEMOTRON_SUPER_REV": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4",
    "NEMOTRON_SUPER_DRAFT_REV": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr",
    "FLASH_LOCAL_REV": "nvidia/Qwen3.8-Flash-Next-FP4-Local",
    "ORNITH_122B_REV": "ornith-ai/Ornith-1.5-122B-NVFP4",
    # the Unsloth GGUF lane's: Unsloth's GGUF repository (the lane serves its UD-Q4_K_XL parts or its UD-IQ3_XXS ones,
    # two targets at this one revision) and its draft's
    "FLASH_UNSLOTH_REV": "unsloth/Qwen3.8-Flash-Next-GGUF",
    "FLASH_UNSLOTH_DRAFT_REV": "ggml-org/Qwen3.8-Flash-Next-GGUF",
    # the MiniMax-H3-Image lane's: its checkpoint, and the text encoder it loads from MiniMax-H3 (the video lane's
    # model, at the revision that lane pins)
    "IMAGE_H3_MODEL_PIN_REV": "MiniMaxAI/MiniMax-H3-Image",
    "IMAGE_H3_ENCODER_PIN_REV": "MiniMaxAI/MiniMax-H3",
}

# Where the pins are written: the text lanes' in the switch and the installer, every other
# lane's in its own installer.
PIN_FILES = ("switch-model.sh", "run.sh", "install.sh",
             "install-image.sh", "install-video.sh", "install-kolibri.sh", "install-nemotron.sh",
             "install-nemotron-super.sh", "install-flash-local.sh", "install-ornith-122b.sh",
             "install-flash-unsloth.sh", "install-image-h3.sh")

PIN_RE = re.compile(r'^\s*([A-Z][A-Z0-9_]*REV)="?\$?\{?[A-Z0-9_:-]*?([0-9a-f]{40})',
                    re.MULTILINE)

# Images that belong to this stack (everything else is none of our business). The Kolibri and
# Nemotron lanes run images built on the box, which carry a tag and no digest: their builds were
# missing from Library's engine images (found on the reference box, 2026-10-16). The Flash
# FP4-Local lane runs the community b12x image its model card names, pulled by digest, and the Unsloth
# GGUF lane llama.cpp's own server image.
ENGINE_IMAGE_RE = re.compile(
    r"^(lmsysorg/sglang|qwen38-flash|qwen38-dflash2|qwen38-sglang|qwen38-kolibri:|qwen38-nemotron:|vllm|eugr/spark-vllm-b12x"
    r"|ghcr\.io/ggml-org/llama\.cpp)")


# Where a lane's unit or launcher says its Hugging Face cache is: a text lane mounts it into
# its container, the image and video lanes set HF_HOME. install.sh takes HF_CACHE.
CACHE_RE = re.compile(r"-v\s+(\S+?):/root/\.cache/huggingface\b|HF_HOME=(\S+)")


def cache_dirs(texts) -> list[str]:
    """The Hugging Face caches the given unit and launcher texts name, in order, once each."""
    out: list[str] = []
    for t in texts:
        for m in CACHE_RE.finditer(t or ""):
            d = (m.group(1) or m.group(2)).strip("\"'")
            if d not in out:
                out.append(d)
    return out


def parse_pins(text: str) -> dict[str, str]:
    """VAR=<40-hex> pairs from shell text (default-expansions included)."""
    return {m.group(1): m.group(2) for m in PIN_RE.finditer(text)}


def parse_docker_images(lines: list[str]) -> list[dict]:
    """'ref size id' rows -> engine-stack images, tagged as ours.

    The caller feeds BOTH shapes docker can print: repo:tag and repo@digest. An
    image pulled by digest has no tag at all, so `{{.Repository}}:{{.Tag}}`
    prints `<none>:<none>` for it and a digest-pinned recipe could never match
    what is on the box: the cockpit said "image missing" about the very image
    the lane was running (2026-09-08, when the flash lane moved to a digest pin).
    Rows whose reference is unusable are dropped rather than carried as noise.
    """
    rows = []
    seen = set()
    for ln in lines:
        parts = ln.split()
        if len(parts) < 3:
            continue
        ref, size, iid = parts[0], parts[1], parts[2]
        if "<none>" in ref or ref in seen:
            continue
        seen.add(ref)
        rows.append({"ref": ref, "size": size, "id": iid,
                     "engine": bool(ENGINE_IMAGE_RE.match(ref))})
    return rows


DOCKER_SIZE_RE = re.compile(r"^([0-9.]+)\s*(B|kB|KB|MB|GB|TB|PB)$")


def parse_docker_size(text: str) -> int | None:
    """Bytes of a size as docker prints it (go-units, powers of 1000: "1.838kB", "639MB", "33.92GB", "0B"); None
    when it reads otherwise."""
    m = DOCKER_SIZE_RE.match((text or "").strip())
    if not m:
        return None
    return int(float(m.group(1)) * {"B": 1, "kB": 1e3, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12, "PB": 1e15}[m.group(2)])


# A blob huggingface_hub is still writing has been written to within this long.
ACTIVE_S = 600


def allowed(rel: str, allow) -> bool:
    """Whether a snapshot path counts under allow patterns, as ckpt-fetch.py (and
    huggingface_hub) read them: fnmatch, case kept, a trailing / standing for the folder."""
    return not allow or any(fnmatch.fnmatchcase(rel, p + "*" if p.endswith("/") else p) for p in allow)


def snapshot_complete(snap: Path, allow=None) -> bool:
    """Every weight file each of the snapshot's indexes names is in it (a file appears there
    only once its blob is whole): the root index of a checkpoint, what install.sh's
    ckpt_cached checks, or one per component in a diffusers layout. With no index, a weight
    file at all. A snapshot folder exists before the first byte of its first file. Only what
    the allow patterns cover counts, as in ckpt-fetch.py's complete(): the video lane serves
    one partition of MiniMax-H3, and the index of another a full pull began does not make
    the lane's checkpoint incomplete (found building the models view, 2026-10-14)."""
    indexes = [p for p in snap.rglob("*.safetensors.index.json") if allowed(p.relative_to(snap).as_posix(), allow)]
    if not indexes:
        # a link counts once its blob is there, as in an index (a link to a blob gone did)
        if any(allowed(p.relative_to(snap).as_posix(), allow) and p.exists() for p in snap.rglob("*.safetensors")):
            return True
        return gguf_complete(snap, allow)
    for idx in indexes:
        try:
            shards = set(json.loads(idx.read_text())["weight_map"].values())
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return False
        if not shards or not all((idx.parent / s).exists() for s in shards):
            return False
    return True


# A GGUF split in parts names them <stem>-00001-of-00004.gguf and so on (llama.cpp's llama_split_path).
GGUF_PART_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$")


def gguf_complete(snap: Path, allow=None) -> bool:
    """A GGUF checkpoint (a llama.cpp lane's): no index, its files *.gguf, each in the snapshot once its blob is whole
    (gguf-fetch.py and huggingface_hub link a file only then). Complete when one is there and every split GGUF has all
    its parts: a fetch cut short between two parts left a model that llama.cpp refuses to load."""
    files = [p for p in snap.rglob("*.gguf") if allowed(p.relative_to(snap).as_posix(), allow) and p.exists()]
    if not files:
        return False
    return splits_whole(p.relative_to(snap).as_posix() for p in files)


def splits_whole(names) -> bool:
    """Every split GGUF among these names has all its parts, 1 to N of its own stem; a name that is not split counts
    whole (gguf_complete for the files of a snapshot, the cockpit for the parts an installer pins)."""
    names = set(names)
    for name in names:
        m = GGUF_PART_RE.match(name)
        if m and any(f"{m.group(1)}-{i:05d}-of-{m.group(3)}.gguf" not in names for i in range(1, int(m.group(3)) + 1)):
            return False
    return True


def pinned_paths(installer_text: str, prefix: str) -> list[str]:
    """The paths an installer pins one by one as PREFIX<n>="PATH=SHA256:SIZE", in their order: the GGUF parts a target
    of a llama.cpp lane serves of a repository that holds every quantization (install-flash-unsloth.sh's
    FLASH_UNSLOTH_PART1 to 4, and FLASH_UNSLOTH_IQ3_PART1 to 4 for its second target, v1.49)."""
    return re.findall(rf'^{re.escape(prefix)}\d+="([^"=]+)=[0-9a-f]{{64}}:\d+"$', installer_text or "", re.M)


VIDEO_ALLOW_RE = re.compile(r'if \[ "\$VID_VARIANT" = "fl2va" \]; then VID_ALLOW="([^"]+)"\s*else VID_ALLOW="([^"]+)"')


def video_allow(switch_text: str, variant: str):
    """The patterns the video lane's switch fetches and serves for a variant, read from
    switch-model.sh, where they are written; None when it no longer reads that way."""
    m = VIDEO_ALLOW_RE.search(switch_text or "")
    if not m:
        return None
    return (m.group(1) if variant == "fl2va" else m.group(2)).split()


def scan_hf_cache(root: Path) -> list[dict]:
    """[{repo_id, disk_bytes, incomplete, active, revisions:[{rev, bytes, complete}]}] for
    models--* dirs.

    disk_bytes = physical blobs (deduped); a revision's bytes are the blobs its
    snapshot links to, each once (two of its files can link one blob: MiniMax-H3's
    partitions share their weights, and counted per file its one revision read twice
    its size), so a blob two revisions share counts in each, and once in disk_bytes.
    """
    out = []
    now = time.time()
    for d in sorted(root.glob("models--*")):
        repo_id = d.name[len("models--"):].replace("--", "/")
        seen: set[int] = set()
        disk = 0
        incomplete = active = 0
        blobs = d / "blobs"
        if blobs.is_dir():
            for f in blobs.iterdir():
                # huggingface_hub writes <sha>.incomplete while a blob is still arriving.
                # One that nothing writes to any more is what an interrupted download
                # left, and it marked the repo "downloading" for good (found in review,
                # 2026-09-24): only a recent one is a download in progress.
                if f.name.endswith(".incomplete"):
                    incomplete += 1
                    try:
                        active += now - f.stat().st_mtime < ACTIVE_S
                    except OSError:
                        pass
                    continue
                try:
                    st = f.stat()
                except OSError:
                    continue
                if st.st_ino in seen:
                    continue
                seen.add(st.st_ino)
                disk += st.st_size
        revs = []
        snaps = d / "snapshots"
        if snaps.is_dir():
            for rd in sorted(snaps.iterdir()):
                size, mine = 0, set()
                for base, _dirs, files in os.walk(rd):
                    for fn in files:
                        try:
                            st = (Path(base) / fn).stat()
                        except OSError:
                            continue
                        if (st.st_dev, st.st_ino) not in mine:
                            mine.add((st.st_dev, st.st_ino))
                            size += st.st_size
                revs.append({"rev": rd.name, "bytes": size, "complete": snapshot_complete(rd)})
        out.append({"repo_id": repo_id, "disk_bytes": disk,
                    "incomplete": incomplete, "active": active, "revisions": revs})
    return out


REV_RE = re.compile(r"^[0-9a-f]{40}$")


def deletion_plan(repo_dir: Path, rev: str) -> dict:
    """What deleting one cached revision removes: its snapshot folder and the blobs no other
    snapshot of the repo links to, with the bytes that frees. {"refused": why} for a name
    that is not a revision here, and for a revision a ref names, which a load by name reads.
    Nothing is removed here: the cockpit's job does that, after the page showed this."""
    if not REV_RE.match(rev or ""):
        return {"refused": "not a revision hash"}
    snaps = repo_dir / "snapshots"
    target = snaps / rev
    if not target.is_dir():
        return {"refused": "this revision is not in the cache"}
    refs = repo_dir / "refs"
    if refs.is_dir():
        for ref in sorted(refs.rglob("*")):
            try:
                named = ref.is_file() and ref.read_text().strip() == rev
            except OSError:
                named = True                 # unreadable: assume it names it
            if named:
                return {"refused": f"refs/{ref.relative_to(refs)} names it, and a load by name reads it"}
    blobs = (repo_dir / "blobs").resolve()
    mine: set[str] = set()
    others: set[str] = set()
    for d in snaps.iterdir():
        bucket = mine if d.name == rev else others
        for base, _dirs, files in os.walk(d):
            for fn in files:
                link = Path(base) / fn
                if link.is_symlink():
                    bucket.add(os.path.realpath(link))
    # only this repo's own blobs, whatever a link says
    only = sorted(b for b in mine - others if Path(b).parent == blobs)
    size = 0
    for b in only:
        try:
            size += os.stat(b).st_size
        except OSError:
            pass
    return {"snapshot": str(target), "blobs": only, "bytes": size}


def mount_of(path) -> str:
    """The mount point a path's bytes live on: the longest mount of /proc/self/mounts that holds its real path (the
    box's own disk, or the SSD a cache was moved to). What a deletion frees is said for that disk."""
    real = os.path.realpath(path)
    best = "/"
    try:
        with open("/proc/self/mounts", encoding="utf-8", errors="replace") as f:
            for ln in f:
                parts = ln.split()
                if len(parts) < 2:
                    continue
                mp = parts[1].replace("\\040", " ")
                if (real == mp or real.startswith(mp.rstrip("/") + "/")) and len(mp) > len(best):
                    best = mp
    except OSError:
        pass
    return best


def _hop(link: Path) -> str | None:
    """The path a symbolic link names, made absolute beside it (not resolved further)."""
    try:
        return os.path.normpath(os.path.join(link.parent, os.readlink(link)))
    except OSError:
        return None


def _refs_lines(refs: Path) -> list[str] | None:
    """The lines of a shared blob's .refs (the repository blobs that link to it, from the cache root), None when it
    cannot be read: a reader that cannot tell who else uses a blob keeps it."""
    try:
        return [ln.strip() for ln in refs.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except (OSError, UnicodeDecodeError):
        return None


def repo_deletion_plan(repo_dir: Path, keep=None) -> dict:
    """What deleting a repository of a Hugging Face cache removes, and the bytes that frees.

    keep: patterns (as allowed() reads them) of the snapshot paths another lane still reads in it (MiniMax-H3's
    text_encoder/, which MiniMax-H3-Image loads): those links, the blobs they reach and what names those blobs stay.
    Each real file is counted and removed once: the repository's own blob, or the blob of the cache's shared store
    (hub/blobs/<xx>/<sha256>) a repository blob links to. A shared blob goes only when every line of its .refs is a
    repository blob that goes: another repository's line keeps it (the scan counted it in both, and frees it in
    neither). Nothing is removed here: apply_repo_deletion does it, from a plan made just before.
    {"refused": why} for a folder that is not a repository, or one a download is writing to now."""
    repo_dir = Path(repo_dir)
    snaps, blobs_dir = repo_dir / "snapshots", repo_dir / "blobs"
    if not repo_dir.name.startswith(("models--", "datasets--", "spaces--")) or not (snaps.is_dir() or blobs_dir.is_dir()):
        return {"refused": "not a repository of this cache"}
    now = time.time()
    if blobs_dir.is_dir():
        for b in blobs_dir.iterdir():
            try:
                if b.name.endswith(".incomplete") and now - b.stat().st_mtime < ACTIVE_S:
                    return {"refused": "a download is writing to it now"}
            except OSError:
                pass
    hub = repo_dir.parent
    store = os.path.normpath(os.path.realpath(hub / "blobs"))
    own = os.path.normpath(os.path.realpath(blobs_dir))
    links, kept = [], []
    for snap in sorted(snaps.iterdir()) if snaps.is_dir() else []:
        for base, _dirs, files in os.walk(snap):
            for fn in files:
                p = Path(base) / fn
                (kept if keep and allowed(str(p.relative_to(snap)), keep) else links).append(p)
    kept_hops = {_hop(p) for p in kept if p.is_symlink()}
    kept_real = {os.path.realpath(p) for p in kept}
    gone = []
    for b in sorted(blobs_dir.iterdir()) if blobs_dir.is_dir() else []:
        if os.path.normpath(str(b)) not in kept_hops and os.path.realpath(b) not in kept_real:
            gone.append(str(b))
    gone_set = {os.path.normpath(g) for g in gone}
    seen, size, kept_bytes, shared, rewrite = set(), 0, 0, [], {}
    for g in gone:
        real = os.path.realpath(g)
        try:
            st = os.stat(real)
        except OSError:
            continue                                    # a link to nothing: it goes, it frees nothing
        key = (st.st_dev, st.st_ino)
        parent = os.path.dirname(real)
        if os.path.dirname(parent) == store:            # the shared store: by what its .refs says
            lines = _refs_lines(Path(real + ".refs"))
            if not lines:
                continue                                # no owner said: kept, counted nowhere
            mine = [ln for ln in lines if os.path.normpath(str(hub / ln)) in gone_set]
            if len(mine) == len(lines):
                if key not in seen:
                    seen.add(key)
                    size += st.st_size
                    shared.append(real)
            elif mine:
                rewrite[real + ".refs"] = [ln for ln in lines if ln not in mine]
            continue
        if parent != own or key in seen:
            continue                                    # not this cache's file (a link out of it): only the link goes
        seen.add(key)
        size += st.st_size
    for p in kept:
        try:
            st = os.stat(p)
        except OSError:
            continue
        if (st.st_dev, st.st_ino) not in seen:
            seen.add((st.st_dev, st.st_ino))
            kept_bytes += st.st_size
    return {"repo_dir": str(repo_dir), "whole": not kept, "links": [str(p) for p in links] if kept else [],
            "blobs": gone, "shared": shared, "refs_rewrite": rewrite, "bytes": size, "kept_bytes": kept_bytes,
            "mount": mount_of(repo_dir)}


def hub_bytes(hub: Path) -> dict:
    """{mount: bytes} the blobs of a Hugging Face cache hold on each disk: every repository's blobs and the shared
    store's, each real file once (a repository blob that links to the shared store counts there), in blocks, on the
    disk the file really is. A repository folder can be a link to another disk: the reference box keeps MiniMax-H3 on
    its own disk, linked from the cache on the SSD, and counted under the cache's disk it read 62 GiB too many there
    (2026-11-17). scan_hf_cache counts a shared blob in each repository that links to it."""
    seen, out, mounts = set(), {}, {}
    for d in list(Path(hub).glob("*/blobs")) + list(Path(hub).glob("blobs/*")):
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for b in entries:
            try:
                st = os.stat(b)
            except OSError:
                continue
            if (st.st_dev, st.st_ino) in seen or os.path.isdir(b):
                continue
            seen.add((st.st_dev, st.st_ino))
            if st.st_dev not in mounts:
                mounts[st.st_dev] = mount_of(b)
            out[mounts[st.st_dev]] = out.get(mounts[st.st_dev], 0) + st.st_blocks * 512
    return out


def apply_repo_deletion(plan: dict) -> int:
    """Carry out a repo_deletion_plan: the whole repository folder (links, never what they point at) and its lock
    folder, or only the links and blobs the plan names when part of it is kept; then the shared blobs it owned alone,
    with their .refs and .lock, and the .refs of a shared blob it shared, without its lines. The bytes the plan
    counted."""
    repo_dir = Path(plan["repo_dir"])
    if plan["whole"]:
        import shutil
        # a repository kept on another disk, its folder a link from the cache (MiniMax-H3 on the reference box): the
        # folder it links to, then the link (rmtree refuses a link)
        shutil.rmtree(os.path.realpath(repo_dir))
        if repo_dir.is_symlink():
            repo_dir.unlink()
        shutil.rmtree(repo_dir.parent / ".locks" / repo_dir.name, ignore_errors=True)
    else:
        for p in plan["links"] + plan["blobs"]:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
        snaps = repo_dir / "snapshots"
        for d in sorted((d for d in snaps.rglob("*") if d.is_dir() and not d.is_symlink()),
                        key=lambda d: -len(d.parts)):
            if not any(d.iterdir()):
                d.rmdir()
    for b in plan["shared"]:
        for f in (b, b + ".refs", b + ".lock"):
            try:
                os.unlink(f)
            except FileNotFoundError:
                pass
    for refs, lines in plan["refs_rewrite"].items():
        tmp = refs + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("".join(ln + "\n" for ln in lines))
        os.replace(tmp, refs)
    return plan["bytes"]


# A file of a snapshot hashed whole to compare it with the Hub: a config the installers rewrote, or a small file a
# tool cached under another name than its git id (NVIDIA Sync's launcher names them by sha256: the disk cleanup of
# 2026-11-16 read them "different" until their content was hashed). A larger one is not read: it stays unverified.
HASH_MAX = 64 * 2**20


def _hashes(path) -> tuple[str, str]:
    """(git blob id, sha256) of a file's content: the Hub names a small file by the first, an LFS one by the second."""
    import hashlib
    size = os.path.getsize(path)
    g, s = hashlib.sha1(b"blob %d\0" % size), hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            g.update(chunk)
            s.update(chunk)
    return g.hexdigest(), s.hexdigest()


def upstream_verdict(snap: Path, siblings: list[dict]) -> dict:
    """Whether the Hub serves every file of a cached revision as it is here, from the revision's file list with its
    blob ids and LFS sha256 (GET /api/models/<repo>/revision/<rev>?blobs=true): what deleting it loses.

    A linked file is the same when the name of the blob it reaches is the Hub's sha256 or git id at the same size, or
    its content hashes to one of them. A file written in the snapshot (no link) is "rewritten here" when the Hub has a
    file of that name and its content differs: the installers rewrite a few configs (YaRN's config.json, keeping the
    original as config.json.pre-yarn, which is compared under the original name; the fp8 KV declaration of
    hf_quant_config.json) and write them again from the Hub's. {"same", "rewritten": [...], "differs": [...],
    "absent": [...], "unverified": [...], "identical": bool}."""
    up = {s.get("rfilename"): s for s in siblings if isinstance(s, dict)}
    out = {"same": 0, "rewritten": [], "differs": [], "absent": [], "unverified": []}
    for base, _dirs, files in os.walk(snap):
        for fn in sorted(files):
            p = Path(base) / fn
            rel = str(p.relative_to(snap))
            name = rel[:-len(".pre-yarn")] if rel.endswith(".pre-yarn") else rel
            s = up.get(name)
            if s is None:
                out["absent"].append(rel)
                continue
            lfs = s.get("lfs") or {}
            ids, usize = {lfs.get("sha256"), s.get("blobId")} - {None}, lfs.get("size", s.get("size"))
            try:
                size = os.path.getsize(p)
            except OSError:
                out["differs"].append(rel)                  # a link to nothing: what it named is gone here
                continue
            # the repository blob's name, or the shared store's blob it links to, named otherwise (snac_24khz's
            # repository blob is the Hub's sha256, its shared blob another hash)
            names = {os.path.basename(os.path.realpath(p)), os.path.basename(_hop(p) or "")}
            if p.is_symlink() and names & ids and usize in (None, size):
                out["same"] += 1
                continue
            if size > HASH_MAX:
                out["unverified"].append(rel)
                continue
            if set(_hashes(p)) & ids:
                out["same"] += 1
            elif not p.is_symlink() and rel == name:
                out["rewritten"].append(rel)
            else:
                out["differs"].append(rel)
    out["identical"] = not (out["differs"] or out["absent"] or out["unverified"])
    return out


def classify(models: list[dict], pins: dict[str, str]) -> list[dict]:
    """Mark every cached revision pinned (by which var) or stray."""
    # the first variable that pins a revision names it: two may pin one (MiniMax-H3's commit, the video lane's checkpoint
    # and the MiniMax-H3-Image lane's text encoder, v1.50), and the later one overwrote the lane that owns it
    rev2var = {}
    for var, rev in pins.items():
        rev2var.setdefault(rev, var)
    enriched = []
    for m in models:
        pin_vars = [v for v, mid in PIN_MODELS.items() if mid == m["repo_id"]]
        revs = []
        for r in m["revisions"]:
            var = rev2var.get(r["rev"])
            status = ("pinned" if var and var in pin_vars
                      else "pin-elsewhere" if var
                      else "stray" if pin_vars
                      else "unmanaged")
            revs.append({**r, "status": status, "pin": var})
        enriched.append({**m, "revisions": revs,
                         "managed": bool(pin_vars)})
    return enriched
