#!/usr/bin/env python3
"""Every command of the cockpit's sudoers allowlist names its arguments, or none with "".

In sudoers a command written with no arguments may be run with ANY: the py-spy wrapper's
line allowed whatever arguments a caller added (without effect today, since the wrapper
reads none; found in review, 2026-09-24). An empty "" allows none, and the cockpit calls
it bare."""
import pathlib
import re
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
LINES = [ln for ln in (REPO / "dashboard" / "sudoers-cockpit.template").read_text().splitlines()
         if ln.startswith("__USER__")]


class EveryCommandIsExact(unittest.TestCase):
    def test_no_line_leaves_the_arguments_open(self):
        self.assertTrue(LINES)
        for ln in LINES:
            command = ln.split("NOPASSWD:", 1)[1].strip()
            self.assertGreater(len(command.split()), 1, f"any arguments allowed: {ln}")

    def test_every_systemctl_line_names_its_unit(self):
        # sudoers matches a command's arguments exactly, so "/usr/bin/systemctl start" alone allowed only
        # a bare "systemctl start", which systemctl refuses: five such lines came with the Transcribe
        # lane's wiring, did nothing, and read as a hole to a blind review (2026-11-19). The cockpit
        # calls systemctl with a verb and a unit (cockpit.py, the unit action), daemon-reload aside.
        for ln in LINES:
            argv = ln.split("NOPASSWD:", 1)[1].split()
            if argv[0] == "/usr/bin/systemctl" and argv[1:] != ["daemon-reload"]:
                self.assertEqual(len(argv), 3, f"a systemctl line without its one unit: {ln}")
                self.assertRegex(argv[2], r"^[\w@.-]+\.(service|timer|target)$", ln)

    def test_the_cockpit_calls_the_wrapper_bare(self):
        ck = (REPO / "dashboard" / "cockpit.py").read_text()
        self.assertRegex(ck, re.escape('run(["sudo", "-n", "/usr/local/bin/qwen38-pyspy-scheduler"]'))


if __name__ == "__main__":
    unittest.main(verbosity=2)
