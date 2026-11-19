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


class RanFail(str):
    ok = False


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

    def test_the_waits_target_is_read_from_the_pending_unit(self):
        # the Load toward another target than the wait's neither prepares nor poses it (third review of
        # v1.52.0, 2026-11-19, GLM 5.3): the page reads which target each waiting unit serves
        (self.pdir / UNIT).write_text("[Service]\nExecStart=x --model-path RadixArk/Qwen3.8-27B-NVFP4 --revision "
                                      + "1" * 40 + "\n")
        self.assertEqual(self.cp.collect_pending_targets(), {UNIT: "stock"})
        (self.pdir / UNIT).write_text("[Service]\nExecStart=x --model-path Qwen/Qwen3.8-27B-FP8 --revision "
                                      + "2" * 40 + "\n")
        self.assertEqual(self.cp.collect_pending_targets(), {UNIT: "fp8"})
        (self.pdir / UNIT).write_text("[Service]\nExecStart=x --model-path some/other --revision " + "3" * 40 + "\n")
        self.assertEqual(self.cp.collect_pending_targets(), {UNIT: None})

    def test_the_flash_waits_target_is_read_from_its_pending_launcher(self):
        flash = "qwen38-flash.service"
        (self.pdir / flash).write_text("[Service]\nExecStart=launch\n")
        (self.pdir / "launch-flash.sh").write_text("#!/bin/bash\n--model-path nvidia/Qwen3.8-Flash-Next-NVFP4 --revision "
                                                   + "4" * 40 + "\n")
        self.assertEqual(self.cp.collect_pending_targets()[flash], "flash-nvda")

    def test_a_wait_whose_unit_in_place_changed_since_is_marked_stale_before_the_load(self):
        # said by pending_units, so the card says it and proposes no Load that would download then drop it
        # (review of v1.52.0, 2026-11-19)
        self.in_place.write_text("[Service]\nExecStart=switched to another target\n")
        self.assertEqual(self.cp.pending_units(),
                         {UNIT: ["stale: the unit in place changed since this update waited (a switch); "
                                 "it is dropped, ./install.sh writes it again"]})

    def test_the_flash_lanes_wait_is_marked_stale_off_its_launchers_base(self):
        flash = "qwen38-flash.service"
        (self.pdir / flash).write_text("[Service]\nExecStart=launch\n")
        (self.pdir / "launch-flash.sh").write_text("#!/bin/bash\nnew\n")
        (self.pdir / (flash + ".needs")).write_text("image x\n")
        launcher = self.cp.CONFIG_DIR / "launch-flash.sh"
        launcher.write_text("#!/bin/bash\nold\n")
        import hashlib
        (self.pdir / (flash + ".base")).write_text(hashlib.sha256(b"another launcher").hexdigest() + "\n")
        self.assertEqual(self.cp.pending_units()[flash],
                         ["stale: the unit in place changed since this update waited (a switch); "
                          "it is dropped, ./install.sh writes it again"])
        (self.pdir / (flash + ".base")).write_text(hashlib.sha256(launcher.read_bytes()).hexdigest() + "\n")
        self.assertEqual(self.cp.pending_units()[flash], ["image x"])

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

    def test_a_waiting_unit_whose_checkpoint_is_not_on_the_box_is_not_put_in_place(self):
        # apply_pending alone (an API call, steps clicked in another order) must not pose a unit whose
        # weights are not there: the lane would serve in memory and not start at its next boot
        # (review of v1.52.0, 2026-11-19)
        self.lacks = "checkpoint RadixArk/Qwen3.8-27B-NVFP4@" + "2" * 40 + "\n"
        job = self.act()
        self.assertEqual(job.status, "failed", job.lines)
        self.assertEqual(self.calls, [])
        self.assertTrue((self.pdir / UNIT).exists(), "the wait is kept, not cleared")
        self.assertTrue(any("still lacks" in ln and "checkpoint" in ln for ln in job.lines), job.lines)

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

    def test_a_daemon_reload_that_fails_after_an_install_that_worked_leaves_no_wait(self):
        # the unit is in place, so the wait is cleared however the reload goes, and the failure says so:
        # a retry must not read the leftover wait as stale (a switch) and drop it with a false message
        # (sixth review of v1.52.0, 2026-11-19, GLM 5.3)
        def failing_run(argv, timeout=5.0, merge_err=False, errors="strict"):
            if argv[0] == "sudo":
                self.calls.append(argv)
                if "daemon-reload" in argv:
                    return RanFail("systemd busy")
            return Ran("")
        self.cp.run = failing_run
        job = self.act()
        self.assertEqual(job.status, "failed", job.lines)
        self.assertEqual([c[2] for c in self.calls], ["/usr/bin/install", "/usr/bin/systemctl"])
        self.assertTrue(any("the unit is in place, systemd did not reload it" in ln for ln in job.lines), job.lines)
        self.assertFalse(any("changed since" in ln or "stale" in ln for ln in job.lines),
                         "no switch message: the unit IS in place")
        self.assertEqual(self.cp.pending_units(), {}, "the wait is cleared: the unit is in place")
        stage = self.cp.CONFIG_DIR / f"{UNIT}.switch-stage"
        self.assertFalse(stage.exists(), "the stage is not left behind")
        # a retry says there is nothing to apply, not that the update was dropped as stale
        self.calls.clear()
        job = self.act()
        self.assertEqual(self.calls, [])
        self.assertTrue(any("no update waits" in ln for ln in job.lines), job.lines)

    def test_an_install_that_fails_keeps_the_wait_for_the_retry(self):
        def failing_run(argv, timeout=5.0, merge_err=False, errors="strict"):
            if argv[0] == "sudo":
                self.calls.append(argv)
                return RanFail("no")
            return Ran("")
        self.cp.run = failing_run
        job = self.act()
        self.assertEqual(job.status, "failed", job.lines)
        self.assertEqual(len(self.calls), 1, "the reload is not asked after a failed install")
        self.assertTrue(any("install failed" in ln for ln in job.lines), job.lines)
        self.assertTrue((self.pdir / UNIT).exists(), "nothing was put in place: the wait is kept")
        self.assertFalse((self.cp.CONFIG_DIR / f"{UNIT}.switch-stage").exists())

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
