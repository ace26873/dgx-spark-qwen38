#!/usr/bin/env python3
"""A plain install installs the speech lane too (v1.58.0, decision 1564: what the repo ships is there for
everyone). It is no GPU lane: install.sh hands it whole to install-speech.sh, proves it when it is new or
named, and goes on when it fails. --no-speech counts for its run only, like every --no-<lane>.

install.sh's own lines run here under its own shell options, against a stub of install-speech.sh in a
throwaway folder: nothing reaches the box the suite runs on.
"""
import pathlib
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
TEXT = (REPO / "install.sh").read_text()


def between(start, end):
    s = TEXT.index(start)
    return TEXT[s:TEXT.index(end, s)]


STEP = between('SPEECH_UNIT_PATH="/etc/systemd/system/qwen38-speech.service"', "side_mode(){")
WANTED = (between("lane_title(){", "\n}\n") + "\n}\n" + between("lane_wanted(){", "\n}\n") + "\n}\n")


def run_step(want=1, with_=0, had=False, installer_rc=0):
    d = pathlib.Path(tempfile.mkdtemp(prefix="speech-step-"))
    stub = d / "install-speech.sh"
    stub.write_text(f'#!/bin/sh\necho "ARGS=[$*]" >> "{d}/calls"\nexit {installer_rc}\n')
    stub.chmod(0o755)
    unit = d / "qwen38-speech.service"
    if had:
        unit.write_text("[Unit]\n")
    script = ("set -euo pipefail\n" 'step(){ echo "STEP: $*"; }\n'
              f'REPO_DIR="{d}"; WANT_SPEECH={want}; WITH_SPEECH={with_}\n{STEP}\n'
              f'SPEECH_UNIT_PATH="{unit}"\n'
              'speech_step\necho "STATE=$SPEECH_STATE"\nspeech_line\necho "WENT ON"\n')
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                       env={"PATH": "/usr/bin:/bin", "HOME": str(d)})
    calls = (d / "calls").read_text().splitlines() if (d / "calls").exists() else []
    return r, calls


class TheSpeechLaneIsInstalledByAPlainRun(unittest.TestCase):
    def test_a_box_without_it_gets_it_whole_and_proved(self):
        r, calls = run_step()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(calls, ["ARGS=[]"], "the installer runs whole, round trip included")
        self.assertIn("STATE=proved", r.stdout)
        self.assertIn("STEP: Speech lane", r.stdout)
        self.assertIn("installed and proved it hears what it says", r.stdout)

    def test_a_box_that_has_it_is_updated_and_proves_nothing_twice(self):
        r, calls = run_step(had=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(calls, ["ARGS=[--no-smoke]"])
        self.assertIn("STATE=installed", r.stdout)
        self.assertIn("installed and up to date", r.stdout)

    def test_named_it_is_proved_again(self):
        r, calls = run_step(with_=1, had=True)
        self.assertEqual(calls, ["ARGS=[]"])
        self.assertIn("STATE=proved", r.stdout)

    def test_a_failure_is_said_and_the_run_goes_on(self):
        for had in (False, True):
            r, calls = run_step(had=had, installer_rc=1)
            self.assertEqual(r.returncode, 0, "set -e must not end the install on the speech lane: " + r.stderr)
            self.assertEqual(len(calls), 1)
            self.assertIn("STATE=failed", r.stdout)
            self.assertIn("NOTE: the speech lane did not install or update", r.stdout)
            self.assertIn("Speech     : NOT ready", r.stdout)
            self.assertIn("WENT ON", r.stdout)

    def test_left_out_nothing_runs_and_the_summary_says_so(self):
        r, calls = run_step(want=0)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(calls, [])
        self.assertNotIn("STEP:", r.stdout)
        self.assertIn("left out of this run, as asked (--no-speech)", r.stdout)
        r, calls = run_step(want=0, had=True)
        self.assertEqual(calls, [])
        self.assertIn("left as it is, as asked (--no-speech)", r.stdout)


class ItsFlags(unittest.TestCase):
    def want(self, no, with_):
        cfg = pathlib.Path(tempfile.mkdtemp(prefix="speech-want-")) / ".config" / "qwen38"
        script = (f'set -euo pipefail\nRECOMMENDED=0\nMEM_CLASS=128\nCONFIG_DIR="{cfg}"\nPLAN_ONLY=0\nLANE=27b\nSIDE_BOOT=0\n'
                  'serves_anyway(){ [ "$1" = "$LANE" ] && [ "$SIDE_BOOT" -eq 0 ]; }\nserves_said(){ :; }\n'
                  f'{WANTED}\nlane_wanted speech {no} {with_}\necho "WANT=$WANT"\n')
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def test_on_by_default_and_off_for_the_run_that_says_so(self):
        self.assertIn("WANT=1", self.want(0, 0))
        self.assertIn("WANT=0", self.want(1, 0))
        self.assertIn("WANT=1", self.want(0, 1))

    def head(self, *flags):
        cut = TEXT.index('step "1/10 Preflight checks"')
        d = pathlib.Path(tempfile.mkdtemp(prefix="speech-flags-"))
        (d / "install.sh").write_text(TEXT[:cut] + 'echo "FLAGS OK want=$WANT_SPEECH"; exit 0\n')
        (d / "lanes.sh").write_text((REPO / "lanes.sh").read_text())
        return subprocess.run(["bash", str(d / "install.sh"), *flags], capture_output=True, text=True, timeout=60,
                              cwd=str(REPO), env={"PATH": "/usr/bin:/bin", "HOME": str(d)})

    def test_both_flags_are_refused_before_anything(self):
        r = self.head("--with-speech", "--no-speech")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--no-speech and --with-speech contradict each other", r.stderr)
        self.assertNotIn("FLAGS OK", r.stdout)

    def test_named_without_the_services_is_refused(self):
        for flag in ("--no-start", "--no-service"):
            r = self.head("--with-speech", flag)
            self.assertNotEqual(r.returncode, 0, flag)
            self.assertIn("--with-speech needs the full install", r.stderr)
            self.assertIn("./install-speech.sh on its own", r.stderr)

    def test_the_help_names_them(self):
        r = self.head("--help")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("[--no-speech | --with-speech]", r.stdout)
        self.assertIn("--no-speech           leave the speech lane out of this run", r.stdout)


class TheTwoPathsBothReachIt(unittest.TestCase):
    """The run that serves a text lane and the run of a box booting another lane each call the step once and
    print its line once: a box booting the image lane would otherwise never get the speech lane."""

    def test_step_and_line_each_stand_in_both_paths(self):
        side = between("# Since v1.20 a side lane missing here is installed too", "record_install side-boot")
        self.assertEqual(side.count("\n  speech_step\n"), 1)
        self.assertEqual(side.count("\n  speech_line\n"), 1)
        main = TEXT[TEXT.index("record_install side-boot"):]
        self.assertEqual(main.count("\n    speech_step\n"), 1)
        self.assertEqual(main.count("\n    speech_line\n"), 1)
        self.assertLess(main.index("\n    speech_step\n"), main.index("\n    speech_line\n"))


if __name__ == "__main__":
    unittest.main()
