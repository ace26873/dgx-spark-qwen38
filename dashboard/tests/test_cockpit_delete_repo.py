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
        code, body = self.act(self.params())
        self.assertEqual((code, body["status"]), (202, "done"), body)
        self.assertEqual(body["result"]["bytes"], 12, body)
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


if __name__ == "__main__":
    unittest.main()
