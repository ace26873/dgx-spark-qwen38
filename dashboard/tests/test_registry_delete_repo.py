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


if __name__ == "__main__":
    unittest.main()
