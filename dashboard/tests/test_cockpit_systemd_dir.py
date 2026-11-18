#!/usr/bin/env python3
"""COCKPIT_SYSTEMD_DIR: every unit path the cockpit reads departs from it.

A runner has no /etc/systemd/system/qwen38-*.service, and a box's real units name the box's
real HF cache in absolute paths: with the paths hardcoded, a test that called
registry_snapshot() without redirecting them scanned the developer's real cache locally and
nothing on a runner, and the local coverage was not the runner's (C20, 2026-11-19). With
COCKPIT_SYSTEMD_DIR posed on a witness directory, no unit path of the module stays under
/etc/systemd/system, so a local run reads no real unit, as a runner does."""
import importlib.util
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
DASH = HERE.parents[1]
REPO = HERE.parents[2]

ENV_BEFORE = dict(os.environ)


def tearDownModule():
    os.environ.clear()
    os.environ.update(ENV_BEFORE)


class TheSystemdDir(unittest.TestCase):
    def test_no_unit_path_of_the_module_stays_under_etc_systemd_system(self):
        tmp = Path(tempfile.mkdtemp(prefix="cockpit-systemd-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "config").mkdir()
        (tmp / "config" / "api-key").write_text("k\n")
        witness = tmp / "systemd"
        witness.mkdir()
        closed = "http://127.0.0.1:1"
        os.environ.pop("COCKPIT_AGENT_UNIT_PATH", None)
        os.environ.pop("COCKPIT_UNIT_DIR", None)
        os.environ.update(HOME=str(tmp), COCKPIT_DRY_RUN="1", COCKPIT_CONFIG_DIR=str(tmp / "config"),
                          COCKPIT_REPO_DIR=str(REPO), COCKPIT_PORT="0", COCKPIT_AGENT_PORT="0",
                          COCKPIT_AUTOHEAL="0", COCKPIT_ENGINE=closed, COCKPIT_PROXY=closed,
                          COCKPIT_IMAGE=closed, COCKPIT_IMAGE_H3=closed, COCKPIT_VIDEO=closed,
                          COCKPIT_TRANSCRIBE=closed, COCKPIT_SYSTEMD_DIR=str(witness))
        sys.path.insert(0, str(DASH))
        spec = importlib.util.spec_from_file_location("cockpit_systemd_under_test", DASH / "cockpit.py")
        cp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cp)
        paths = {name: value for name, value in vars(cp).items()
                 if name.endswith("_UNIT_PATH") and isinstance(value, Path)}
        paths["UNIT_DIR"] = cp.UNIT_DIR
        for unit, path in cp.UNIT_PATHS.items():
            paths[f"UNIT_PATHS[{unit}]"] = path
        for lane, path in cp.INSTALLED_INVOCATION.items():
            paths[f"INSTALLED_INVOCATION[{lane}]"] = path
        self.assertTrue(paths, "no unit path found in the module")
        under_etc = {n: str(p) for n, p in paths.items() if str(p).startswith("/etc/systemd/system")}
        self.assertEqual(under_etc, {}, "unit paths still read under /etc/systemd/system with "
                                       "COCKPIT_SYSTEMD_DIR posed")
        witness_real = os.path.realpath(witness)
        outside = {n: str(p) for n, p in paths.items()
                   if not os.path.realpath(p).startswith(witness_real + os.sep)
                   and not os.path.realpath(p).startswith(os.path.realpath(tmp) + os.sep)}
        self.assertEqual(outside, {}, "a unit path departs from neither the witness nor the test's tmp")


if __name__ == "__main__":
    unittest.main()
