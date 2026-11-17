#!/usr/bin/env python3
"""An update that waits for its Load (decisions 1404 and 1408 to 1410 of Hasan, 2026-11-16 and 17): the cockpit reads
the lane units waiting in $CONFIG_DIR/pending (what each lacks beside it), calls such a lane not prepared so that its
Load prepares it first, and apply_pending puts the waiting unit in place once the box has all it reads (lanes.sh's
unit_needs, asked again), through the fixed staging path its exact sudoers line allows. The cockpit runs live here,
on a config folder of the test's own, its sudo and unit_needs stubbed."""
import importlib.util
import os
import shutil
import subprocess
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
UNIT = "qwen38-sglang.service"


def tearDownModule():
    os.environ.clear()
    os.environ.update(ENV_BEFORE)


class Ran(str):
    ok = True


class ApplyPending(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="cockpit-apply-pending-"))
        (cls.tmp / "config").mkdir()
        (cls.tmp / "config" / "api-key").write_text("k\n")
        closed = "http://127.0.0.1:1"
        os.environ.update(HOME=str(cls.tmp), COCKPIT_DRY_RUN="0", COCKPIT_CONFIG_DIR=str(cls.tmp / "config"),
                          COCKPIT_REPO_DIR=str(REPO), COCKPIT_PORT="0", COCKPIT_AGENT_PORT="0",
                          COCKPIT_AUTOHEAL="0", COCKPIT_ENGINE=closed, COCKPIT_PROXY=closed, COCKPIT_IMAGE=closed,
                          COCKPIT_IMAGE_H3=closed, COCKPIT_UNIT_DIR=str(cls.tmp / "units"))
        spec = importlib.util.spec_from_file_location("cockpit_apply_pending_under_test", DASH / "cockpit.py")
        cls.cp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.cp)
        # every unit the module reads or writes is the test's own: never the box's (a test wrote a real checkpoint
        # away on 2026-10-22; this one reached /etc/systemd/system before this line, refused for want of root)
        units = cls.tmp / "units"
        units.mkdir()
        for name, value in list(vars(cls.cp).items()):
            if name.endswith("_UNIT_PATH") and isinstance(value, Path):
                setattr(cls.cp, name, units / value.name)
        cls.cp.UNIT_PATHS = {k: units / k for k in cls.cp.UNIT_PATHS}
        cls.cp.UNIT_DIR = units

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.pdir = self.cp.PENDING_DIR
        shutil.rmtree(self.pdir, ignore_errors=True)
        self.pdir.mkdir(parents=True)
        (self.pdir / UNIT).write_text("[Service]\nExecStart=new\n")
        (self.pdir / (UNIT + ".needs")).write_text("image lmsysorg/sglang@sha256:" + "b" * 64 + "\n")
        # the unit in place the waiting one was rendered against (its sha256 in <unit>.base, lanes.sh)
        self.in_place = self.cp.UNIT_DIR / UNIT
        self.in_place.write_text("[Service]\nExecStart=old\n")
        import hashlib
        (self.pdir / (UNIT + ".base")).write_text(hashlib.sha256(self.in_place.read_bytes()).hexdigest() + "\n")
        self.lacks, self.calls = "", []
        real_sp = subprocess.run

        def fake_sp(argv, **kw):
            if argv[:2] == ["bash", "-c"] and "unit_needs" in argv[2]:
                return subprocess.CompletedProcess(argv, 0, self.lacks, "")
            return real_sp(argv, **kw)
        self.cp.subprocess.run = fake_sp
        self.addCleanup(setattr, self.cp.subprocess, "run", real_sp)

        def fake_run(argv, timeout=5.0, merge_err=False, errors="strict"):
            if argv[0] == "sudo":
                self.calls.append(argv)
            return Ran("")
        self.cp.run = fake_run

    def act(self):
        code, body = self.cp.start_action("apply_pending", {"unit": UNIT})
        self.assertEqual(code, 202, body)
        job = self.cp.JOBS[body["job"]]
        for _ in range(200):
            if job.status != "running":
                break
            time.sleep(0.05)
        return job

    def test_what_waits_is_read_with_what_it_lacks(self):
        self.assertEqual(self.cp.pending_units(), {UNIT: ["image lmsysorg/sglang@sha256:" + "b" * 64]})

    def test_a_lane_whose_update_waits_stays_built(self):
        # it serves, or can start, the release it served: its Load prepares the update (the page's journey), and its
        # Start as installed, its runtime's line and Library's Download stay (code review of v1.52.0)
        src = (self.cp.REPO_DIR / "dashboard" / "cockpit.py").read_text()
        body = src[src.index("def lane_prepared(lane: str):"):src.index("# ---- An update that waits for its Load")]
        self.assertNotIn("PENDING_DIR", body)

    def test_a_waiting_unit_rendered_before_a_switch_is_dropped_not_put(self):
        self.in_place.write_text("[Service]\nExecStart=switched to another target\n")
        job = self.act()
        self.assertEqual(job.status, "failed", job.lines)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.cp.pending_units(), {})
        self.assertTrue(any("changed since the update" in ln for ln in job.lines), job.lines)

    def test_the_checkpoint_is_the_switchs_to_fetch(self):
        self.lacks = "checkpoint RadixArk/Qwen3.8-27B-NVFP4@" + "2" * 40 + "\n"
        job = self.act()
        self.assertEqual(job.status, "done", job.lines)

    def test_a_unit_that_still_lacks_something_is_not_put_in_place(self):
        self.lacks = "image lmsysorg/sglang@sha256:" + "b" * 64 + "\n"
        job = self.act()
        self.assertEqual(job.status, "failed", job.lines)
        self.assertEqual(self.calls, [])
        self.assertTrue((self.pdir / UNIT).exists())
        self.assertTrue(any("still lacks" in ln for ln in job.lines), job.lines)

    def test_once_the_box_has_it_all_it_goes_in_place_by_the_exact_sudoers_line(self):
        job = self.act()
        self.assertEqual(job.status, "done", job.lines)
        stage = str(self.cp.CONFIG_DIR / f"{UNIT}.switch-stage")
        self.assertEqual(self.calls, [["sudo", "-n", "/usr/bin/install", "-m", "644", stage, f"/etc/systemd/system/{UNIT}"],
                                      ["sudo", "-n", "/usr/bin/systemctl", "daemon-reload"]])
        sudoers = (DASH / "sudoers-cockpit.template").read_text()
        self.assertIn(f"/usr/bin/install -m 644 __HOME__/.config/qwen38/{UNIT}.switch-stage /etc/systemd/system/{UNIT}",
                      sudoers)
        self.assertEqual(self.cp.pending_units(), {})
        self.assertFalse(Path(stage).exists())

    def test_every_unit_that_can_wait_has_its_sudoers_line(self):
        sudoers = (DASH / "sudoers-cockpit.template").read_text()
        self.assertEqual(self.cp.PENDING_UNITS, ("qwen38-sglang.service", "qwen38-flash.service"))
        for unit in ("qwen38-sglang.service",):     # the flash lane's launcher is the user's file: no sudo (below)
            with self.subTest(unit=unit):
                self.assertIn(f"/usr/bin/install -m 644 __HOME__/.config/qwen38/{unit}.switch-stage /etc/systemd/system/{unit}",
                              sudoers)


    def test_the_flash_lanes_launcher_goes_in_place_with_no_sudo(self):
        flash = "qwen38-flash.service"
        (self.pdir / flash).write_text("[Service]\nExecStart=launch\n")
        (self.pdir / "launch-flash.sh").write_text("#!/bin/bash\nnew\n")
        (self.pdir / (flash + ".needs")).write_text("image x\n")
        launcher = self.cp.CONFIG_DIR / "launch-flash.sh"
        launcher.write_text("#!/bin/bash\nold\n")
        code, body = self.cp.start_action("apply_pending", {"unit": flash})
        self.assertEqual(code, 202, body)
        job = self.cp.JOBS[body["job"]]
        for _ in range(200):
            if job.status != "running":
                break
            time.sleep(0.05)
        self.assertEqual(job.status, "done", job.lines)
        self.assertEqual(launcher.read_text(), "#!/bin/bash\nnew\n")
        self.assertTrue(os.access(launcher, os.X_OK))
        self.assertEqual(self.calls, [], "no sudo for the user's own file")
        self.assertNotIn(flash, self.cp.pending_units())

if __name__ == "__main__":
    unittest.main()
