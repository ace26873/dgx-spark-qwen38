#!/usr/bin/env python3
"""A box whose Unsloth GGUF lane serves UD-Q4_K_XL because v1.48 installed it by default moves to
UD-IQ3_XXS, the default since v1.49.0 (decision 1376), at its update; chosen, it stays.

The lane's installer keeps the target its unit names (install-flash-unsloth.sh), so install.sh
hands it FLASH_UNSLOTH_TARGET when conformity.py classes the unit's UD-Q4_K_XL as an old default:
on a box older than v1.49.0 with no installed.json, or one whose installed.json does not record
it. Recorded (a switch to flash-unsloth records it), on a younger box, given on the command line,
or with --recommended (the installer's default whatever the unit names), nothing is handed over.
This runs install.sh's own lines, with lanes.sh's reading of the unit."""
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import installer_wall as wall  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import conformity  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
TEXT = (REPO / "install.sh").read_text()
START = TEXT.index("# The Unsloth GGUF lane's target: its installer keeps")
BLOCK = TEXT[START:TEXT.index("\nfi\n", START) + 4]
_MS = TEXT.index("old_default_moves() {")
MOVES = 'MOVED_TAG=""\n' + TEXT[_MS:TEXT.index("\n}\n", _MS) + 3]
OLD = conformity.old_default_of("FLASH_UNSLOTH_TARGET")
SNAP = "/root/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/" + "f" * 40
Q4 = f"{SNAP}/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
IQ3 = f"{SNAP}/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf"


def shift(minutes):
    import datetime
    return (datetime.datetime.fromisoformat(OLD.instant) + datetime.timedelta(minutes=minutes)).isoformat()


def run_block(model, key_at=None, record=None, want="1", recommended="0", given=""):
    """(output, FLASH_UNSLOTH_TARGET after the block) for a unit whose -m is `model` (none when None)."""
    d = pathlib.Path(tempfile.mkdtemp(prefix="unsloth-target-"))
    unit = d / "qwen38-flash-unsloth.service"
    if model is not None:
        # its header comment names "-m" too, as the real unit's does: the reading leaves comments out
        unit.write_text("# Unsloth's GGUF, -m naming the first part\n[Service]\nExecStart=/usr/bin/docker run x \\\n"
                        f"    -m {model} \\\n    --alias qwen3.8-flash-next-unsloth\n")
    cfg = pathlib.Path(wall.box_home(key_at=key_at, record=record)) / ".config/qwen38"
    script = ("set -euo pipefail\n" + f'. "{REPO}/lanes.sh"\n'
              + f'REPO_DIR="{REPO}"; CONFIG_DIR="{cfg}"; FLASH_UNSLOTH_UNIT_PATH="{unit}"\n'
              + f'WANT_FLASH_UNSLOTH={want}; RECOMMENDED={recommended}; _ENV_FLASH_UNSLOTH_TARGET="{given}"\n'
              + (f'export FLASH_UNSLOTH_TARGET="{given}"\n' if given else "unset FLASH_UNSLOTH_TARGET\n")
              + MOVES + BLOCK + '\necho "TARGET=${FLASH_UNSLOTH_TARGET:-}"\n')
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30,
                       env={"PATH": "/usr/bin:/bin", "HOME": str(cfg.parents[1])})
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("TARGET=")]
    assert line, r.stdout + r.stderr
    assert "command not found" not in r.stderr, r.stderr
    return r.stdout + r.stderr, line[0].split("=", 1)[1]


class TheOldDefaultMoves(unittest.TestCase):
    def test_on_a_box_older_than_v1_49_it_moves_to_iq3_and_says_how_to_keep_it(self):
        out, target = run_block(Q4, key_at=shift(-60))
        self.assertEqual(target, "flash-unsloth-iq3")
        self.assertIn("Moving the Unsloth GGUF lane from flash-unsloth to flash-unsloth-iq3, the default since v1.49.0", out)
        self.assertIn("To keep it: FLASH_UNSLOTH_TARGET=flash-unsloth ./install.sh", out)

    def test_an_installed_json_that_does_not_record_it_moves_it_whatever_the_age(self):
        _, target = run_block(Q4, key_at=shift(60), record={})
        self.assertEqual(target, "flash-unsloth-iq3")


class WhatStays(unittest.TestCase):
    def test_recorded_as_a_choice_it_stays(self):
        out, target = run_block(Q4, key_at=shift(-60), record={"FLASH_UNSLOTH_TARGET": "flash-unsloth"})
        self.assertEqual(target, "")
        self.assertNotIn("Moving", out)

    def test_on_a_box_younger_than_v1_49_it_stays(self):
        _, target = run_block(Q4, key_at=shift(60))
        self.assertEqual(target, "")

    def test_a_box_whose_age_cannot_be_told_keeps_it(self):
        _, target = run_block(Q4)
        self.assertEqual(target, "")

    def test_given_on_the_command_line_it_decides(self):
        out, target = run_block(Q4, key_at=shift(-60), given="flash-unsloth")
        self.assertEqual(target, "flash-unsloth")
        self.assertNotIn("Moving", out)

    def test_recommended_leaves_it_to_the_installer(self):
        _, target = run_block(Q4, key_at=shift(-60), recommended="1")
        self.assertEqual(target, "")

    def test_a_run_that_does_not_install_the_lane_hands_nothing(self):
        _, target = run_block(Q4, key_at=shift(-60), want="0")
        self.assertEqual(target, "")

    def test_a_unit_on_the_default_or_none_hands_nothing(self):
        for model in (IQ3, None, "/elsewhere/own.gguf"):
            with self.subTest(model=model):
                out, target = run_block(model, key_at=shift(-60))
                self.assertEqual(target, "")
                self.assertNotIn("Moving", out)


class TheClientsNameIt(unittest.TestCase):
    def test_the_entry_names_the_target_handed_to_the_installer(self):
        """The clients' block runs before the installer renders the unit: it names the target the run
        hands over (given or moved), not the one the unit still serves."""
        i = TEXT.index('  OC_FLASH_UNSLOTH_TARGET="$(flash_unsloth_target "$FLASH_UNSLOTH_UNIT_PATH"')
        nxt = TEXT[i:TEXT.index("\n", TEXT.index("\n", i) + 1)]
        self.assertIn('[ "$WANT_FLASH_UNSLOTH" -ne 1 ] || [ -z "${FLASH_UNSLOTH_TARGET:-}" ] '
                      '|| OC_FLASH_UNSLOTH_TARGET="$FLASH_UNSLOTH_TARGET"', nxt)
        self.assertLess(START, i, "the move is decided before the clients' block reads it")


if __name__ == "__main__":
    unittest.main()
