#!/usr/bin/env python3
"""Library's Delete of a whole repository (registry.repo_deletion_plan, apply_repo_deletion), on a Hugging Face cache
of the test's own: each real file counted and removed once; a blob of the cache's shared store (hub/blobs/<xx>/<sha>,
whose .refs names the repository blobs that link to it) removed only when every repository it serves goes; the part
another lane reads kept whole (MiniMax-H3's text_encoder/, which MiniMax-H3-Image loads: the disk cleanup of
2026-11-16 kept it by hand); a link out of the cache removed, never what it points at (the engram table of the same
cleanup was a link into a study folder); a repository a download writes to now left alone."""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
import registry as rg  # noqa: E402

REV = "a" * 40


def write(p: Path, data: bytes):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def link(src: Path, target: Path):
    src.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(os.path.relpath(target, src.parent), src)


class Cache:
    """A hub folder: repositories whose files are their own blobs, or blobs of the shared store."""

    def __init__(self, root: Path):
        self.hub = root / "hub"
        self.hub.mkdir(parents=True)

    def repo(self, name, files, shared=None, rev=REV):
        """files: {snapshot path: bytes}, each its own blob; shared: {snapshot path: sha}, a repository blob that
        links to the shared store's blob of that sha (created by store())."""
        repo = self.hub / f"models--o--{name}"
        snap = repo / "snapshots" / rev
        for i, (rel, data) in enumerate(sorted(files.items())):
            blob = repo / "blobs" / f"{name}{i:03d}"
            write(blob, data)
            link(snap / rel, blob)
        for rel, sha in sorted((shared or {}).items()):
            blob = repo / "blobs" / sha
            if not blob.is_symlink():
                link(blob, self.hub / "blobs" / sha[:2] / sha)
            link(snap / rel, blob)
        (repo / "refs").mkdir(parents=True, exist_ok=True)
        (repo / "refs" / "main").write_text(rev)
        return repo

    def store(self, sha, data, owners):
        b = self.hub / "blobs" / sha[:2] / sha
        write(b, data)
        Path(str(b) + ".refs").write_text("".join(f"models--o--{o}/blobs/{sha}\n" for o in owners))
        Path(str(b) + ".lock").write_text("")
        return b


class DeleteRepo(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="registry-delete-repo-"))
        self.addCleanup(__import__("shutil").rmtree, self.root, True)
        self.c = Cache(self.root)

    def test_whole_repository_and_its_lock_folder(self):
        repo = self.c.repo("m", {"config.json": b"{}", "model.safetensors": b"x" * 1000})
        (self.c.hub / ".locks" / repo.name).mkdir(parents=True)
        plan = rg.repo_deletion_plan(repo)
        self.assertTrue(plan["whole"])
        self.assertEqual(plan["bytes"], 1002)
        self.assertEqual(rg.apply_repo_deletion(plan), 1002)
        self.assertFalse(repo.exists())
        self.assertFalse((self.c.hub / ".locks" / repo.name).exists())

    def test_a_repository_linked_from_another_disk_goes_with_its_link(self):
        far = Cache(self.root / "interne").repo("h3", {"te.bin": b"z" * 64})
        link = self.c.hub / far.name
        os.symlink(far, link)
        plan = rg.repo_deletion_plan(link)
        self.assertEqual((plan["whole"], plan["bytes"]), (True, 64))
        rg.apply_repo_deletion(plan)
        self.assertFalse(os.path.lexists(link))
        self.assertFalse(far.exists())
        self.assertTrue((self.root / "interne" / "hub").is_dir(), "only the repository goes, not its cache")

    def test_shared_blob_goes_only_with_its_last_repository(self):
        sha, other = "b" * 64, "c" * 64
        alone = self.c.store(sha, b"y" * 500, ["m"])
        both = self.c.store(other, b"z" * 300, ["m", "n"])
        repo = self.c.repo("m", {"config.json": b"{}"}, {"a.safetensors": sha, "b.safetensors": other})
        keeper = self.c.repo("n", {"config.json": b"{}"}, {"b.safetensors": other})
        plan = rg.repo_deletion_plan(repo)
        # its own blob, and the shared blob it alone uses; never the one repository n still links to
        self.assertEqual(plan["bytes"], 2 + 500)
        self.assertEqual(plan["shared"], [os.path.realpath(alone)])
        rg.apply_repo_deletion(plan)
        for f in (alone, Path(str(alone) + ".refs"), Path(str(alone) + ".lock")):
            self.assertFalse(f.exists(), f)
        self.assertEqual((keeper / "snapshots" / REV / "b.safetensors").read_bytes(), b"z" * 300)
        # and n's blob no longer names m in its .refs
        self.assertEqual(Path(str(both) + ".refs").read_text(), f"models--o--n/blobs/{other}\n")

    def test_shared_blob_without_owners_said_is_kept(self):
        sha = "d" * 64
        b = self.c.store(sha, b"w" * 100, [])
        Path(str(b) + ".refs").write_text("")
        repo = self.c.repo("m", {}, {"w.safetensors": sha})
        plan = rg.repo_deletion_plan(repo)
        self.assertEqual((plan["bytes"], plan["shared"]), (0, []))
        rg.apply_repo_deletion(plan)
        self.assertTrue(b.exists())

    def test_part_another_lane_reads_is_kept_whole(self):
        files = {f"text_encoder/model-{i}.safetensors": bytes([i]) * 100 for i in range(3)}
        files.update({f"transformer/model-{i}.safetensors": bytes([10 + i]) * 1000 for i in range(4)})
        files["model_index.json"] = b"{}"
        repo = self.c.repo("h3", files)
        plan = rg.repo_deletion_plan(repo, keep=["text_encoder/*"])
        self.assertFalse(plan["whole"])
        self.assertEqual((plan["bytes"], plan["kept_bytes"]), (4002, 300))
        rg.apply_repo_deletion(plan)
        snap = repo / "snapshots" / REV
        self.assertEqual(sorted(str(p.relative_to(snap)) for p in snap.rglob("*") if p.is_symlink()),
                         [f"text_encoder/model-{i}.safetensors" for i in range(3)])
        self.assertTrue(all(p.exists() for p in snap.rglob("*")), "a kept link broken")
        self.assertEqual(len(list((repo / "blobs").iterdir())), 3)
        self.assertFalse((snap / "transformer").exists())

    def test_link_out_of_the_cache_goes_alone(self):
        outside = self.root / "etudes" / "table.gguf"
        write(outside, b"t" * 700)
        repo = self.c.repo("g", {"config.json": b"{}"})
        link(repo / "snapshots" / REV / "engram" / "table.gguf", outside)
        # and a blob of the repository itself that links out of the cache: counted nowhere, its file kept
        link(repo / "blobs" / ("f" * 64), outside)
        link(repo / "snapshots" / REV / "big.gguf", repo / "blobs" / ("f" * 64))
        plan = rg.repo_deletion_plan(repo)
        self.assertEqual(plan["bytes"], 2)
        rg.apply_repo_deletion(plan)
        self.assertEqual(outside.read_bytes(), b"t" * 700)

    def test_a_folder_this_user_may_not_change_refuses_before_anything_goes(self):
        repo = self.c.repo("m", {"config.json": b"{}", "sub/w.bin": b"w" * 10})
        locked = repo / "snapshots" / REV / "sub"
        os.chmod(locked, 0o555)
        self.addCleanup(os.chmod, locked, 0o755)
        if os.access(locked, os.W_OK):
            self.skipTest("run as root: every folder is writable")
        plan = rg.repo_deletion_plan(repo)
        self.assertIn("may not change", plan.get("refused", ""), plan)
        self.assertTrue((locked / "w.bin").exists())

    def test_a_download_in_progress_is_refused(self):
        repo = self.c.repo("m", {"config.json": b"{}"})
        part = repo / "blobs" / ("e" * 64 + ".incomplete")
        part.write_bytes(b"p")
        self.assertIn("refused", rg.repo_deletion_plan(repo))
        old = time.time() - rg.ACTIVE_S - 60
        os.utime(part, (old, old))
        self.assertEqual(rg.repo_deletion_plan(repo)["bytes"], 3)   # what an interrupted download left goes too

    def test_not_a_repository(self):
        (self.c.hub / "blobs").mkdir(exist_ok=True)
        self.assertIn("refused", rg.repo_deletion_plan(self.c.hub / "blobs"))
        self.assertIn("refused", rg.repo_deletion_plan(self.c.hub / "models--o--absent"))

    def test_mount_of_names_a_mount(self):
        self.assertTrue(os.path.ismount(rg.mount_of(self.root)))



class UpstreamVerdict(unittest.TestCase):
    """What deleting a cached revision loses, read against the Hub's file list for that revision (blob ids, LFS
    sha256): the check verif_hf.py made by hand for the disk cleanup of 2026-11-16, its traps included."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="registry-upstream-"))
        self.addCleanup(__import__("shutil").rmtree, self.root, True)
        self.snap = self.root / "models--o--m" / "snapshots" / REV
        self.blobs = self.root / "models--o--m" / "blobs"

    def linked(self, rel, data, name=None):
        git, sha = self.hashes(data)
        blob = self.blobs / (name or sha)
        write(blob, data)
        link(self.snap / rel, blob)
        return git, sha

    def written(self, rel, data):
        write(self.snap / rel, data)

    @staticmethod
    def hashes(data):
        import hashlib
        return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest(), hashlib.sha256(data).hexdigest()

    def sib(self, rel, data, lfs=True):
        git, sha = self.hashes(data)
        return {"rfilename": rel, "blobId": git, "size": len(data),
                **({"lfs": {"sha256": sha, "size": len(data)}} if lfs else {})}

    def test_identical_by_name_and_by_content(self):
        w, c = b"w" * 50, b'{"a": 1}'
        self.linked("model.safetensors", w)
        # a small file a tool cached under its sha256 rather than its git id: the same once its content is hashed
        self.linked("config.json", c)
        v = rg.upstream_verdict(self.snap, [self.sib("model.safetensors", w), self.sib("config.json", c, lfs=False)])
        self.assertEqual((v["same"], v["identical"]), (2, True))

    def test_configs_the_installers_rewrite(self):
        orig, patched = b'{"rope": null}', b'{"rope": "yarn"}'
        self.written("config.json", patched)
        self.written("config.json.pre-yarn", orig)
        v = rg.upstream_verdict(self.snap, [self.sib("config.json", orig, lfs=False)])
        self.assertEqual((v["same"], v["rewritten"], v["identical"]), (1, ["config.json"], True))
        # the original kept beside it is what the Hub must still serve
        self.written("config.json.pre-yarn", b'{"rope": "other"}')
        v = rg.upstream_verdict(self.snap, [self.sib("config.json", orig, lfs=False)])
        self.assertEqual((v["differs"], v["identical"]), (["config.json.pre-yarn"], False))

    def test_a_linked_blob_the_hub_no_longer_serves(self):
        self.linked("model.safetensors", b"old" * 10)
        v = rg.upstream_verdict(self.snap, [self.sib("model.safetensors", b"new" * 10)])
        self.assertEqual((v["differs"], v["identical"]), (["model.safetensors"], False))

    def test_absent_upstream_and_a_broken_link(self):
        self.linked("extra.bin", b"e")
        link(self.snap / "gone.bin", self.blobs / "nothing")
        v = rg.upstream_verdict(self.snap, [self.sib("gone.bin", b"g")])
        self.assertEqual((v["absent"], v["differs"], v["identical"]), (["extra.bin"], ["gone.bin"], False))

    def test_a_large_file_named_otherwise_stays_unverified(self):
        data = b"L" * 300
        self.linked("big.bin", data, name="not-its-hash")
        old, rg.HASH_MAX = rg.HASH_MAX, 100
        self.addCleanup(setattr, rg, "HASH_MAX", old)
        v = rg.upstream_verdict(self.snap, [self.sib("big.bin", data)])
        self.assertEqual((v["unverified"], v["identical"]), (["big.bin"], False))

    def test_shared_store_blob_named_otherwise(self):
        data = b"s" * 40
        git, sha = self.hashes(data)
        store = self.root / "blobs" / "13" / ("1" * 64)
        write(store, data)
        link(self.blobs / sha, store)
        link(self.snap / "w.bin", self.blobs / sha)
        old, rg.HASH_MAX = rg.HASH_MAX, 10                   # by its names alone, not by its content
        self.addCleanup(setattr, rg, "HASH_MAX", old)
        self.assertTrue(rg.upstream_verdict(self.snap, [self.sib("w.bin", data)])["identical"])



class HubBytes(unittest.TestCase):
    """Library's disks: the bytes a cache's blobs hold on each disk, each real file once, on the disk it really is."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="registry-hub-bytes-"))
        self.addCleanup(__import__("shutil").rmtree, self.root, True)
        self.c = Cache(self.root)

    def test_a_shared_blob_once_and_a_linked_repository_by_its_real_files(self):
        # two disks cannot be made here: the reference box's (SSD 303.0 GiB, its own disk 62.1 for MiniMax-H3, linked
        # from the cache on the SSD) read as du reads each (2026-11-17)
        sha = "b" * 64
        self.c.store(sha, b"y" * 8192, ["m", "n"])
        self.c.repo("m", {"a.bin": b"x" * 4096}, {"s.bin": sha})
        self.c.repo("n", {}, {"s.bin": sha})
        far = Cache(self.root / "interne").repo("h3", {"te.bin": b"z" * 4096})
        os.symlink(far, self.c.hub / far.name)
        got = rg.hub_bytes(self.c.hub)
        # the shared store's .refs too, as du counts them (a block each)
        shared = self.c.hub / "blobs" / sha[:2] / sha
        files = [shared, Path(str(shared) + ".refs"), self.c.hub / "models--o--m" / "blobs" / "m000", far / "blobs" / "h3000"]
        self.assertEqual(got, {rg.mount_of(self.root): sum(os.stat(f).st_blocks * 512 for f in files)})

if __name__ == "__main__":
    unittest.main()
