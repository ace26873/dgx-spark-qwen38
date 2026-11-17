#!/usr/bin/env python3
"""Library's Delete of an engine image (decisions 1402 and 1404 of Hasan, 2026-11-16), against a docker of the test's
own: every name of the image goes (two lanes can run one image: FP4-Local and Ornith), by docker rmi and never -f;
refused while a container starts from it, a stopped one too (the GTA session's gta-vlm runs the 27B's image,
2026-11-17), or while a lane that runs it serves; refused when no registry serves it and no lane builds it, unless the
loss was shown and confirmed (lose=yes); what it frees is docker's own count of the bytes only it holds."""
import importlib.util
import json
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

ENV_BEFORE = dict(os.environ)
DIGEST = "eugr/spark-vllm-b12x@sha256:03a956544e9f" + "0" * 52
FULL = "sha256:" + "a7f76ae9f18c" + "1" * 52


def tearDownModule():
    os.environ.clear()
    os.environ.update(ENV_BEFORE)


class Ran(str):
    ok = True


def ran(text="", ok=True):
    r = Ran(text)
    r.ok = ok
    return r


class DeleteImage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="cockpit-delete-image-"))
        (cls.tmp / "config").mkdir()
        (cls.tmp / "config" / "api-key").write_text("k\n")
        closed = "http://127.0.0.1:1"
        os.environ.update(HOME=str(cls.tmp), COCKPIT_DRY_RUN="0", COCKPIT_CONFIG_DIR=str(cls.tmp / "config"),
                          COCKPIT_REPO_DIR=str(REPO), COCKPIT_PORT="0", COCKPIT_AGENT_PORT="0",
                          COCKPIT_AUTOHEAL="0", COCKPIT_ENGINE=closed, COCKPIT_PROXY=closed, COCKPIT_IMAGE=closed,
                          COCKPIT_IMAGE_H3=closed, COCKPIT_UNIT_DIR=str(cls.tmp / "units"))
        spec = importlib.util.spec_from_file_location("cockpit_delete_image_under_test", DASH / "cockpit.py")
        cls.cp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.cp)
        cls.cp.UNIT_DIR = cls.tmp / "units"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        shutil.rmtree(self.tmp / "units", ignore_errors=True)
        (self.tmp / "units").mkdir()
        self.names = ["qwen38-pinned:flash-local-03a956544e9f", "qwen38-pinned:ornith-122b-03a956544e9f", DIGEST]
        self.containers, self.active, self.served, self.rmi = [], set(), True, []
        for lane in ("flash-local", "ornith-122b"):
            (self.tmp / "units" / f"qwen38-{lane}.service").write_text(f"ExecStart=/usr/bin/docker run --rm {DIGEST}\n")
        self.cp.run = self.fake_run
        self.cp.registry_snapshot = lambda max_age=300.0: {"images": [{"ref": DIGEST, "id": "a7f76ae9f18c",
                                                                         "engine": True}]}
        self.cp.disk_free = lambda path: 10**12 + (17_770_000_000 if self.rmi else 0)

    def fake_run(self, argv, timeout=5.0, merge_err=False, errors="strict"):
        if argv[:3] == ["docker", "image", "inspect"]:
            if argv[-1] != "a7f76ae9f18c" or self.rmi:
                return ran("", ok=False)
            tags = [n for n in self.names if "@" not in n]
            return ran(f"{FULL}|{json.dumps(tags)}|{json.dumps([n for n in self.names if '@' in n])}\n")
        if argv[:2] == ["docker", "ps"]:
            return ran("".join(c + "\n" for c in self.containers))
        if argv[:2] == ["systemctl", "is-active"]:
            return ran("active\n" if argv[2] in self.active else "inactive\n")
        if argv[:3] == ["docker", "system", "df"]:
            return ran(json.dumps([{"ID": FULL, "UniqueSize": "17.77GB", "SharedSize": "415.2MB"}]))
        if argv[:3] == ["docker", "buildx", "imagetools"]:
            return ran('"sha256:..."' if self.served else "", ok=self.served)
        if argv[:2] == ["docker", "info"]:
            return ran("/var/lib/docker\n")
        if argv[:2] == ["docker", "rmi"]:
            self.rmi.append(argv[2:])
            return ran("Untagged: ...\nDeleted: ...\n")
        return ran("")

    def act(self, lose="no", image="a7f76ae9f18c"):
        code, body = self.cp.start_action("delete_image", {"image": image, "lose": lose})
        if code == 202:
            job = self.cp.JOBS[body["job"]]
            for _ in range(200):
                if job.status != "running":
                    break
                time.sleep(0.05)
            body["status"], body["lines"], body["result"] = job.status, job.lines, job.result
        return code, body

    def test_every_name_goes_and_the_lanes_that_ran_it_are_said(self):
        code, body = self.act()
        self.assertEqual((code, body["status"]), (202, "done"), body)
        self.assertEqual(self.rmi, [self.names], "one docker rmi of every name, never -f")
        self.assertEqual(body["result"]["bytes"], 17_770_000_000)
        self.assertTrue(any("flash-local" in ln and "ornith-122b" in ln and "pulls or builds it again" in ln
                            for ln in body["lines"]), body["lines"])

    def test_a_container_even_stopped_keeps_it(self):
        self.containers = ["gta-vlm (exited)"]
        code, body = self.act()
        self.assertEqual(body["status"], "failed", body)
        self.assertEqual(self.rmi, [])
        self.assertTrue(any("gta-vlm" in ln for ln in body["lines"]), body["lines"])

    def test_a_lane_that_serves_with_it_keeps_it(self):
        self.active.add("qwen38-flash-local.service")
        code, body = self.act()
        self.assertEqual(body["status"], "failed", body)
        self.assertEqual(self.rmi, [])

    def test_an_image_no_registry_serves_needs_the_loss_confirmed(self):
        self.served = False
        code, body = self.act()
        self.assertEqual(body["status"], "failed", body)
        self.assertEqual(self.rmi, [])
        code, body = self.act(lose="yes")
        self.assertEqual(body["status"], "done", body)

    def test_an_image_its_lane_builds_here_needs_no_registry(self):
        self.names = ["qwen38-kolibri:vllm0.29.0-aai1.0.0-18372a72249386434"]
        self.served = False
        (self.tmp / "units" / "qwen38-kolibri.service").write_text(f"ExecStart=docker run {self.names[0]}\n")
        code, body = self.act()
        self.assertEqual(body["status"], "done", body)

    def test_only_an_engine_image_the_scan_listed(self):
        code, body = self.act(image="b6a1428a9d48")
        self.assertEqual(code, 400, body)
        # listed by mistake, served by its registry, the loss even confirmed: still not an image of this stack
        self.names = ["gta-relais:1", "gta-relais@sha256:" + "2" * 64]
        code, body = self.act(lose="yes")
        self.assertEqual(body["status"], "failed", body)
        self.assertEqual(self.rmi, [])
        self.assertTrue(any("not an engine image of this stack" in ln for ln in body["lines"]), body["lines"])


if __name__ == "__main__":
    unittest.main()
