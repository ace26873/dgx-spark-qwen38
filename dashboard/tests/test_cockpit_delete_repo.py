#!/usr/bin/env python3
"""Library's Delete of a whole repository (decisions 1402 and 1404 of Hasan, 2026-11-16): planned again by the job;
refused while a lane unit that serves, a running container or a process reads it; refused when the Hub does not serve
every cached revision as it is here, unless the page showed that loss and it was confirmed (lose=yes); the part a lane
loads alone kept (MiniMax-H3's text_encoder/). Its parameters are a closed set read from the scan. The cockpit runs live
here (not dry run), on a Hugging Face cache, a /proc and a Hub of the test's own."""
import hashlib
import importlib.util
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
DASH = HERE.parents[1]
REPO = HERE.parents[2]
sys.path.insert(0, str(DASH))
sys.path.insert(0, str(HERE.parent))
from test_registry import fake_repo  # noqa: E402

ENV_BEFORE = dict(os.environ)
DSPARK = "RadixArk/Qwen3.8-27B-DSpark"
OTHER = "someone/other-model"


def tearDownModule():
    os.environ.clear()
    os.environ.update(ENV_BEFORE)


def sibling(rel, data):
    return {"rfilename": rel, "blobId": hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest(), "size": len(data),
            "lfs": {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}}


class DeleteRepo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="cockpit-delete-repo-"))
        (cls.tmp / "config").mkdir()
        (cls.tmp / "config" / "api-key").write_text("k\n")
        closed = "http://127.0.0.1:1"
        os.environ.update(HOME=str(cls.tmp), COCKPIT_DRY_RUN="0", COCKPIT_CONFIG_DIR=str(cls.tmp / "config"),
                          COCKPIT_REPO_DIR=str(REPO), COCKPIT_PORT="0", COCKPIT_AGENT_PORT="0",
                          COCKPIT_AUTOHEAL="0", COCKPIT_ENGINE=closed, COCKPIT_PROXY=closed, COCKPIT_IMAGE=closed,
                          COCKPIT_IMAGE_H3=closed, COCKPIT_UNIT_DIR=str(cls.tmp / "units"))
        spec = importlib.util.spec_from_file_location("cockpit_delete_repo_under_test", DASH / "cockpit.py")
        cls.cp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.cp)
        cls.unit = cls.tmp / "qwen38-kolibri.service"
        # every unit the module reads names the test's own cache only (test_cockpit_delete_strays says why)
        for name, value in list(vars(cls.cp).items()):
            if name.endswith("_UNIT_PATH") and isinstance(value, Path):
                setattr(cls.cp, name, cls.unit)
        cls.cp.UNIT_PATHS = {k: cls.unit for k in cls.cp.UNIT_PATHS}
        cls.cp.UNIT_DIR = cls.tmp / "units"
        cls.cp.PROC = cls.tmp / "proc"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.hub = self.tmp / ".cache" / "huggingface" / "hub"
        shutil.rmtree(self.hub, ignore_errors=True)
        self.hub.mkdir(parents=True)
        for d in ("units", "proc"):
            shutil.rmtree(self.tmp / d, ignore_errors=True)
            (self.tmp / d).mkdir()
        self.unit.write_text("ExecStart=/usr/bin/docker run --rm img\n")
        self.rev = "c" * 40
        self.files = {"config.json": b"{}", "w.safetensors": b"x" * 10}
        self.repo = fake_repo(self.hub, {self.rev: self.files}, name="models--RadixArk--Qwen3.8-27B-DSpark")
        self.cache = str(self.tmp / ".cache" / "huggingface")
        self.active = set()
        self.cp.run = self.fake_run
        self.hub_files = {DSPARK: [sibling(k, v) for k, v in self.files.items()]}
        self.cp._get_json = self.fake_hub

    def fake_run(self, argv, timeout=5.0, merge_err=False, errors="strict"):
        if argv[:2] == ["systemctl", "is-active"]:
            return "active\n" if argv[2] in self.active else "inactive\n"
        return ""

    def fake_hub(self, url, timeout=5.0):
        repo = url.split("/api/models/", 1)[1].split("/revision/", 1)[0]
        if repo not in self.hub_files:
            raise OSError("no network")
        return {"siblings": self.hub_files[repo]}

    def act(self, params):
        own = os.path.realpath(self.tmp) + os.sep
        outside = [c for c in self.cp.lane_caches() if not os.path.realpath(c).startswith(own)]
        self.assertEqual(outside, [], "the delete would scan a cache outside the test's own: nothing was run")
        self.cp.registry_snapshot(max_age=0.0)
        code, body = self.cp.start_action("delete_repo", params)
        if code == 202:
            job = self.cp.JOBS[body["job"]]
            for _ in range(200):
                if job.status != "running":
                    break
                time.sleep(0.05)
            body["status"], body["lines"], body["result"] = job.status, job.lines, job.result
        return code, body

    def params(self, repo=DSPARK, lose="no"):
        return {"repo": repo, "cache": self.cache, "lose": lose}

    def test_a_repository_the_hub_serves_alike_goes_whole(self):
        want = sum(os.stat(b).st_blocks * 512 for b in (self.repo / "blobs").iterdir())
        code, body = self.act(self.params())
        self.assertEqual((code, body["status"]), (202, "done"), body)
        self.assertEqual(body["result"]["bytes"], want, body)
        self.assertFalse(self.repo.exists())
        self.assertTrue(any(f"deleted {DSPARK}" in ln for ln in body["lines"]), body["lines"])

    def test_a_repository_the_hub_serves_otherwise_needs_the_loss_confirmed(self):
        self.hub_files[DSPARK] = [sibling("config.json", b"{}"), sibling("w.safetensors", b"z" * 10)]
        code, body = self.act(self.params())
        self.assertEqual(body["status"], "failed", body)
        self.assertTrue(self.repo.exists())
        self.assertTrue(any("does not serve it as it is here" in ln and "w.safetensors" in ln for ln in body["lines"]),
                        body["lines"])
        code, body = self.act(self.params(lose="yes"))
        self.assertEqual(body["status"], "done", body)
        self.assertFalse(self.repo.exists())

    def test_a_hub_that_cannot_be_asked_is_no_proof(self):
        self.hub_files.pop(DSPARK)
        code, body = self.act(self.params())
        self.assertEqual(body["status"], "failed", body)
        self.assertTrue(self.repo.exists())
        self.assertTrue(any("could not be asked" in ln for ln in body["lines"]), body["lines"])

    def test_what_reads_it_now_keeps_it(self):
        # a process whose command line names it (a container's engine shows in /proc too)
        proc = self.tmp / "proc" / "4242"
        proc.mkdir()
        (proc / "cmdline").write_bytes(b"python3\0-m\0sglang.launch_server\0--model-path\0" + DSPARK.encode() + b"\0")
        code, body = self.act(self.params())
        self.assertEqual(body["status"], "failed", body)
        self.assertTrue(self.repo.exists())
        self.assertTrue(any("process 4242" in ln for ln in body["lines"]), body["lines"])
        shutil.rmtree(proc)
        # a lane unit that serves and names it
        (self.tmp / "units" / "qwen38-sglang.service").write_text(f"ExecStart=... --speculative-draft-model-path {DSPARK}\n")
        self.active.add("qwen38-sglang.service")
        code, body = self.act(self.params())
        self.assertEqual(body["status"], "failed", body)
        self.assertTrue(any("qwen38-sglang.service" in ln for ln in body["lines"]), body["lines"])
        # stopped, it reads nothing: the Delete goes on
        self.active.clear()
        code, body = self.act(self.params())
        self.assertEqual(body["status"], "done", body)

    def test_only_a_repository_the_scan_listed(self):
        code, body = self.act(self.params(repo=OTHER))
        self.assertEqual(code, 400, body)
        code, body = self.act({"repo": DSPARK, "cache": str(self.tmp), "lose": "no"})
        self.assertEqual(code, 400, body)
        self.assertTrue(self.repo.exists())
        # a repository no pin names is listed too, and deleted the same way
        other = fake_repo(self.hub, {self.rev: {"m.bin": b"o"}}, name="models--someone--other-model")
        self.hub_files[OTHER] = [sibling("m.bin", b"o")]
        code, body = self.act(self.params(repo=OTHER))
        self.assertEqual((code, body["status"]), (202, "done"), body)
        self.assertFalse(other.exists())
        self.assertTrue(self.repo.exists())

    def test_the_part_a_lane_loads_alone_is_kept_while_its_checkpoint_is_whole(self):
        enc = self.cp.rg.PIN_MODELS["IMAGE_H3_ENCODER_PIN_REV"]
        reg = {"targets": [{"target": "image-h3", "state": "complete"}]}
        self.assertEqual(self.cp.repo_keep(enc, reg)[0], ["text_encoder/*"])
        reg["targets"][0]["state"] = "absent"
        self.assertEqual(self.cp.repo_keep(enc, reg)[0], [])
        self.assertEqual(self.cp.repo_keep(DSPARK, {"targets": [{"target": "image-h3", "state": "complete"}]})[0], [])

    def test_a_reader_that_appears_during_the_hubs_verdicts_keeps_it(self):
        # the plan reads the readers before asking the Hub (up to 20 s a batch of four): a lane started in between
        # is read again just before anything goes (revue of v1.51.0, 2026-11-19)
        (self.tmp / "units" / "qwen38-sglang.service").write_text(f"ExecStart=... --model-path {DSPARK}\n")
        real_hub = self.cp._get_json

        def hub_then_lane(url, timeout=5.0):
            out = real_hub(url, timeout)
            self.active.add("qwen38-sglang.service")     # the autoheal starts the lane while the Hub is asked
            return out

        self.cp._get_json = hub_then_lane
        self.addCleanup(setattr, self.cp, "_get_json", real_hub)
        code, body = self.act(self.params())
        self.assertEqual(body["status"], "failed", body)
        self.assertTrue(self.repo.exists())
        self.assertTrue(any("read now by" in ln and "qwen38-sglang.service" in ln for ln in body["lines"]),
                        body["lines"])

    def test_a_partial_scan_refuses_the_delete_of_a_repository(self):
        # a checkout in progress: the scan missed a pin, so the image-h3 lane reads "unknown" and the part it loads
        # alone (MiniMax-H3's text_encoder/) would read as deletable. The Delete is refused, like job_delete_strays
        # (revue of v1.51.0, 2026-11-19)
        enc = self.cp.rg.PIN_MODELS["IMAGE_H3_ENCODER_PIN_REV"]
        repo = fake_repo(self.hub, {self.rev: self.files}, name="models--MiniMaxAI--MiniMax-H3")
        self.hub_files[enc] = [sibling(k, v) for k, v in self.files.items()]
        real = self.cp.registry_snapshot

        def partial(max_age=300.0):
            reg = real(max_age=0.0)
            reg["partial"] = ["IMAGE_H3_ENCODER_PIN_REV not found"]
            reg["targets"] = [{**t, "state": "unknown"} if t["target"] == "image-h3" else t for t in reg["targets"]]
            return reg

        self.cp.registry_snapshot = partial
        self.addCleanup(setattr, self.cp, "registry_snapshot", real)
        # what the keep would be without the refusal: nothing, text_encoder/ would go with the repository
        self.assertEqual(self.cp.repo_keep(enc, partial())[0], [])
        code, body = self.act(self.params(repo=enc))
        self.assertEqual(body["status"], "failed", body)
        self.assertTrue(repo.exists())
        self.assertTrue(any("this scan missed something" in ln and "IMAGE_H3_ENCODER_PIN_REV not found" in ln
                            for ln in body["lines"]), body["lines"])


    def runtime(self, item, lose=None):
        code, body = self.cp.start_action("delete_runtime", {"item": item})
        if code == 202:
            job = self.cp.JOBS[body["job"]]
            for _ in range(200):
                if job.status != "running":
                    break
                time.sleep(0.05)
            body["status"], body["lines"], body["result"] = job.status, job.lines, job.result
        return code, body

    def test_a_lanes_runtime_goes_and_what_it_made_stays(self):
        d = self.tmp / ".local" / "share" / "qwen38-image"
        shutil.rmtree(d, ignore_errors=True)
        (d / "venv" / "bin").mkdir(parents=True)
        (d / "venv" / "bin" / "python").write_bytes(b"p" * 5000)
        (d / "sglang-wheel").write_text("sglang-0.5.22\n")
        (d / "outputs").mkdir()
        (d / "outputs" / "cat.png").write_bytes(b"png")
        # IMAGE_LANE_DIR can be a folder shared with other work: only what the installer put there goes
        (d / "work").mkdir()
        (d / "work" / "notes.txt").write_text("mine")
        rows = {r["item"]: r for r in self.cp.runtime_rows()}
        self.assertIn("image", rows)
        self.assertGreaterEqual(rows["image"]["bytes"], 5000)
        self.assertNotIn(str(d / "outputs"), rows["image"]["paths"])
        # its lane serves: kept
        self.active.add(self.unit.name)
        code, body = self.runtime("image")
        self.assertEqual(body["status"], "failed", body)
        self.assertTrue((d / "venv").exists())
        self.active.clear()
        code, body = self.runtime("image")
        self.assertEqual((code, body["status"]), (202, "done"), body)
        self.assertEqual(sorted(x.name for x in d.iterdir()), ["outputs", "work"])
        self.assertEqual((d / "work" / "notes.txt").read_text(), "mine")
        self.assertEqual((d / "outputs" / "cat.png").read_bytes(), b"png")
        self.assertTrue(any("next Load builds it again" in ln for ln in body["lines"]), body["lines"])

    def test_the_diffusion_caches_wait_for_every_diffusion_lane_to_stop(self):
        spill = self.tmp / ".cache" / "sglang" / "diffusion" / "host_spill"
        spill.mkdir(parents=True, exist_ok=True)
        (spill / "w.bin").write_bytes(b"w" * 100)
        self.active.add("qwen38-video.service")
        code, body = self.runtime("diffusion-caches")
        self.assertEqual(body["status"], "failed", body)
        self.assertTrue(spill.exists())
        self.active.clear()
        code, body = self.runtime("diffusion-caches")
        self.assertEqual(body["status"], "done", body)
        self.assertFalse(spill.exists())
        self.assertTrue((self.tmp / ".cache" / "sglang" / "diffusion").exists(), "only the caches go")

    def test_only_a_known_item(self):
        self.assertEqual(self.runtime("../../etc")[0], 400)


    def test_a_process_running_from_a_runtime_keeps_it(self):
        d = self.tmp / ".local" / "share" / "qwen38-image"
        shutil.rmtree(d, ignore_errors=True)
        (d / "venv" / "bin").mkdir(parents=True)
        (d / "venv" / "bin" / "python").write_bytes(b"p")
        proc = self.tmp / "proc" / "777"
        proc.mkdir()
        (proc / "cmdline").write_bytes(str(d / "venv" / "bin" / "python").encode() + b"\0-m\0sglang\0")
        code, body = self.runtime("image")
        self.assertEqual(body["status"], "failed", body)
        self.assertTrue((d / "venv").exists())
        self.assertTrue(any("process 777" in ln for ln in body["lines"]), body["lines"])

    def test_a_process_that_maps_a_shared_blob_it_owns_keeps_it(self):
        # a checkpoint loaded by mmap keeps no file open, only its mapping, of the file the links reached
        sha = "d" * 64
        store = self.hub / "blobs" / sha[:2] / sha
        store.parent.mkdir(parents=True)
        store.write_bytes(b"s" * 20)
        (store.parent / (sha + ".refs")).write_text(f"models--RadixArk--Qwen3.8-27B-DSpark/blobs/{sha}\n")
        os.symlink(os.path.relpath(store, self.repo / "blobs"), self.repo / "blobs" / sha)
        os.symlink(f"../../blobs/{sha}", self.repo / "snapshots" / self.rev / "s.safetensors")
        self.hub_files[DSPARK].append(sibling("s.safetensors", b"s" * 20))
        proc = self.tmp / "proc" / "888"
        proc.mkdir()
        (proc / "cmdline").write_bytes(b"python3\0notebook.py\0")
        (proc / "maps").write_text(f"7f00-7f10 r--s 00000000 00:00 1   /data/hub/blobs/{sha[:2]}/{sha}\n")
        code, body = self.act(self.params())
        self.assertEqual(body["status"], "failed", body)
        self.assertTrue(any("process 888" in ln for ln in body["lines"]), body["lines"])
        self.assertTrue(store.exists())

    def test_a_longer_name_that_starts_with_it_is_not_it(self):
        (self.tmp / "units" / "qwen38-sglang.service").write_text(f"ExecStart=... --model-path {DSPARK}2 --revision x\n")
        self.active.add("qwen38-sglang.service")
        code, body = self.act(self.params())
        self.assertEqual(body["status"], "done", body)

    def test_each_disk_it_touches_says_what_it_freed(self):
        sha = "e" * 64
        store = self.hub / "blobs" / sha[:2] / sha
        store.parent.mkdir(parents=True)
        store.write_bytes(b"s" * 20)
        (store.parent / (sha + ".refs")).write_text(f"models--RadixArk--Qwen3.8-27B-DSpark/blobs/{sha}\n")
        os.symlink(os.path.relpath(store, self.repo / "blobs"), self.repo / "blobs" / sha)
        os.symlink(f"../../blobs/{sha}", self.repo / "snapshots" / self.rev / "s.safetensors")
        self.hub_files[DSPARK].append(sibling("s.safetensors", b"s" * 20))
        real_mount, real_free = self.cp.rg.mount_of, self.cp.disk_free
        self.cp.rg.mount_of = lambda path: "/store" if "/hub/blobs/" in os.path.realpath(path) + "/" else "/repos"
        free = {"/store": 100, "/repos": 1000}
        self.cp.disk_free = lambda m: free[m] + (1000 if not self.repo.exists() else 0)
        self.addCleanup(setattr, self.cp.rg, "mount_of", real_mount)
        self.addCleanup(setattr, self.cp, "disk_free", real_free)
        code, body = self.act(self.params())
        self.assertEqual(body["status"], "done", body)
        self.assertIn("on /repos", body["lines"][0])
        self.assertIn("on /store", body["lines"][0])

    def test_the_registry_scan_walks_no_runtime_and_library_still_shows_them(self):
        calls = []
        real = self.cp.runtime_rows
        self.cp.runtime_rows = lambda: calls.append(1) or real()
        self.addCleanup(setattr, self.cp, "runtime_rows", real)
        self.cp.registry_snapshot(max_age=0.0)
        self.cp.repo_delete_plan(self.cache, DSPARK, hub=False)
        self.assertEqual(calls, [], "a Delete's plan rescans the registry: it walked the venvs too")
        view = self.cp.registry_view(True)
        self.assertEqual(len(calls), 1)
        self.assertIn("runtimes", view)
        self.assertIn("disks", view)

    def test_the_hub_is_asked_for_every_revision_at_once(self):
        revs = [c * 40 for c in "1234"]
        for r in revs:
            fake_repo(self.hub, {r: {"m.bin": b"o"}}, name="models--someone--four") if r == revs[0] else None
        repo = self.hub / "models--someone--four"
        for r in revs[1:]:
            (repo / "snapshots" / r).mkdir(parents=True)
        def slow(url, timeout=5.0):
            time.sleep(0.4)
            return {"siblings": [sibling("m.bin", b"o")]}
        self.cp._get_json = slow
        t0 = time.time()
        plan = self.cp.repo_delete_plan(self.cache, OTHER.replace("other-model", "four"))
        self.assertEqual(len(plan["revisions"]), 4, plan)
        self.assertLess(time.time() - t0, 1.2, "four revisions asked one after the other")

    def test_what_another_project_keeps_has_no_delete(self):
        keep = self.tmp / "config" / "library-keep"
        keep.write_text(f"{DSPARK}  # GTA, brouillon de gta-vlm\n# a comment line\n\n")
        self.addCleanup(keep.unlink)
        code, body = self.act(self.params())
        self.assertEqual(body["status"], "failed", body)
        self.assertTrue(self.repo.exists())
        self.assertTrue(any("kept for GTA, brouillon de gta-vlm (library-keep)" in ln for ln in body["lines"]),
                        body["lines"])
        view = self.cp.registry_view(True)
        mine = next(m for m in view["models"] if m["repo_id"] == DSPARK)
        self.assertEqual(mine["keep"], "GTA, brouillon de gta-vlm")
        self.assertIsNone(self.cp.registry_snapshot().get("models")[0].get("keep"), "the scan's cache is not changed")

if __name__ == "__main__":
    unittest.main()
