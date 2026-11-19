#!/usr/bin/env python3
"""A switch to what a 27B unit or the flash launch script already loads rewrites nothing.

Rewritten, the file got a new date, which reads as an update the serving engine has not taken:
the cockpit then offered a restart of 8 to 13 minutes for nothing (found in review, 2026-10-17).
Kolibri, Nemotron and the image lane already compared before rewriting; the 27B unit and the
flash launch script now do too. The switch's own lines run here with sudo and install recorded.
"""
import pathlib
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
TEXT = (REPO / "switch-model.sh").read_text()


def block(first):
    start = TEXT.index(first)
    return TEXT[start:TEXT.index("\n  fi\n", start) + 6]


UNIT_27B = block('  if cmp -s "$STAGE_UNIT" "$TARGET_UNIT"; then')
FLASH = block('  if cmp -s "$TMP_UNIT" "$FLASH_LAUNCH"; then')


def run(snippet, staged, installed, pending=False):
    d = pathlib.Path(tempfile.mkdtemp(prefix="sw-same-"))
    (d / "staged").write_text(staged)
    (d / "installed").write_text(installed)
    cfg = d / "cfg" / "pending"
    cfg.mkdir(parents=True)
    if pending:
        (cfg / "qwen38-sglang.service").write_text("waiting\n")
        (cfg / "launch-flash.sh").write_text("waiting\n")
    script = ("set -euo pipefail\n"
              f'log="{d}/calls"; sudo(){{ echo "sudo $*" >> "$log"; }}; install(){{ echo "install $*" >> "$log"; }}\n'
              f'unit_pending_clear(){{ echo "clear $*" >> "$log"; }}\n'
              f'CONFIG_DIR="{d}/cfg"; CHOICE=stock; STAGE_UNIT="{d}/staged"; TARGET_UNIT="{d}/installed"; '
              f'TARGET_UNIT_NAME=qwen38-sglang.service; FLASH_UNIT="{d}/installed"; '
              f'TMP_UNIT="{d}/staged"; FLASH_LAUNCH="{d}/installed"\n' + snippet + "echo REACHED\n")
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
    calls = (d / "calls").read_text() if (d / "calls").exists() else ""
    return r.stdout + r.stderr, calls


class ASwitchToWhatIsInstalledRewritesNothing(unittest.TestCase):
    def test_the_27b_unit_unchanged_keeps_its_date(self):
        out, calls = run(UNIT_27B, "ExecStart=x --revision a\n", "ExecStart=x --revision a\n")
        self.assertEqual(calls, "", out)
        self.assertIn("unit unchanged: it already loads stock", out)
        self.assertIn("REACHED", out)

    def test_the_27b_unit_changed_is_installed(self):
        out, calls = run(UNIT_27B, "ExecStart=x --revision b\n", "ExecStart=x --revision a\n")
        self.assertIn("sudo install -m 644", calls, out)
        self.assertIn("REACHED", out)

    def test_a_changed_unit_clears_the_wait_the_switch_makes_stale(self):
        # an update rendered against the unit before this switch is caduque now (review of v1.52.0, 2026-11-19)
        out, calls = run(UNIT_27B, "ExecStart=x --revision b\n", "ExecStart=x --revision a\n", pending=True)
        self.assertIn("clear ", calls, out)
        self.assertIn("was dropped", out)

    def test_an_unchanged_unit_keeps_its_wait(self):
        out, calls = run(UNIT_27B, "ExecStart=x --revision a\n", "ExecStart=x --revision a\n", pending=True)
        self.assertNotIn("clear ", calls, out)
        self.assertIn("REACHED", out)

    def test_the_flash_launcher_unchanged_keeps_its_date(self):
        out, calls = run(FLASH, "docker run x\n", "docker run x\n")
        self.assertEqual(calls, "", out)
        self.assertIn("launch script unchanged", out)

    def test_the_flash_launcher_changed_is_installed(self):
        out, calls = run(FLASH, "docker run y\n", "docker run x\n")
        self.assertIn("install -m 755", calls, out)

    def test_a_changed_launcher_clears_the_wait_the_switch_makes_stale(self):
        out, calls = run(FLASH, "docker run y\n", "docker run x\n", pending=True)
        self.assertIn("clear ", calls, out)
        self.assertIn("was dropped", out)



LANE_BLOCKS = {
    "transcribe": ('  if ! cmp -s "$TRANSCRIBE_STAGE" "$TRANSCRIBE_UNIT"; then',
                   {"TRANSCRIBE_STAGE": "staged", "TRANSCRIBE_UNIT": "installed",
                    "TRANSCRIBE_UNIT_NAME": "qwen38-transcribe.service"}),
    "lfm25": ('  if ! cmp -s "$LFM25_STAGE" "$LFM25_UNIT"; then',
              {"LFM25_STAGE": "staged", "LFM25_UNIT": "installed",
               "LFM25_UNIT_NAME": "qwen38-lfm25.service"}),
    "lfm3": ('  if ! cmp -s "$LFM3_STAGE" "$LFM3_UNIT"; then',
             {"LFM3_STAGE": "staged", "LFM3_UNIT": "installed",
              "LFM3_UNIT_NAME": "qwen38-lfm3.service"}),
}


def run_lane(snippet, vars_, staged, installed, pending):
    d = pathlib.Path(tempfile.mkdtemp(prefix="sw-lane-same-"))
    (d / "staged").write_text(staged)
    (d / "installed").write_text(installed)
    cfg = d / "cfg" / "pending"
    cfg.mkdir(parents=True)
    if pending:
        for name in {v for v in vars_.values() if v.endswith(".service")}:
            (cfg / name).write_text("waiting\n")
    assigns = "; ".join(f'{k}="{d}/{v}"' if v in ("staged", "installed") else f'{k}="{v}"'
                        for k, v in vars_.items())
    script = ("set -euo pipefail\n"
              f'log="{d}/calls"; sudo(){{ echo "sudo $*" >> "$log"; }}; unit_pending_clear(){{ echo "clear $*" >> "$log"; }}\n'
              f'CONFIG_DIR="{d}/cfg"; {assigns}\n' + snippet + "echo REACHED\n")
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
    calls = (d / "calls").read_text() if (d / "calls").exists() else ""
    return r.stdout + r.stderr, calls


class TheSideLaneSwitchesClearAWaitTheirRenderedUnitMatches(unittest.TestCase):
    """Blind review of v1.53.1 (2026-11-19, GLM 5.3), latent defect (h): the transcribe, lfm25 and
    lfm3 switches had no else branch like Kolibri's and Nemotron's. When the unit they render
    equals the one in place (a pin moved back between two versions before the Load), a wait for
    the lane was not cleared, and the lane's card kept saying an update waited for its Load
    until the next ./install.sh."""

    def test_each_side_lane_clears_the_wait_when_the_rendered_unit_is_the_one_in_place(self):
        for lane, (first, vars_) in LANE_BLOCKS.items():
            with self.subTest(lane=lane):
                out, calls = run_lane(block(first), vars_, "unit text\n", "unit text\n", pending=True)
                self.assertIn("clear ", calls, out)
                self.assertIn("REACHED", out)

    def test_each_side_lane_installs_and_clears_when_the_rendered_unit_differs(self):
        for lane, (first, vars_) in LANE_BLOCKS.items():
            with self.subTest(lane=lane):
                out, calls = run_lane(block(first), vars_, "new text\n", "unit text\n", pending=True)
                self.assertIn("sudo install -m 644", calls, out)
                self.assertIn("clear ", calls, out)
                self.assertIn("REACHED", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
