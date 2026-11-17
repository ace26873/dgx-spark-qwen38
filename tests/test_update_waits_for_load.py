#!/usr/bin/env python3
"""install.sh on an installed box (decisions 1404 and 1408 to 1410 of Hasan, 2026-11-16 and 17): every unit it writes
goes through unit_put with PENDING_PUT=1, and the text lane, when it can serve and its new pins are not all here (the
image, the target whole at its revision, the 27B's draft), gets its units only: none of its downloads, its update
waiting for its Load. --with-<lane>, --all-weights, a custom model kept, --no-service and a new box download as before.
The block of install.sh itself, run with the box's facts stubbed."""
import pathlib
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
TEXT = (REPO / "install.sh").read_text()
BLOCK = TEXT[TEXT.index("# An update downloads nothing large (decisions 1404 and 1408 to 1410 of Hasan, 2026-11-16 and 17): weights"):
             TEXT.index("# Measured where the weights actually land")]
IMAGE = "lmsysorg/sglang@sha256:" + "a" * 64
REV, DREV = "1" * 40, "d" * 40


class UpdateWaits(unittest.TestCase):
    def run_block(self, unit=True, prepared=True, image=True, ckpt=True, draft=True, all_weights=0, named=0,
                  no_service=0, verbatim=0, units_only=0, lane="27b"):
        d = pathlib.Path(tempfile.mkdtemp(prefix="update-waits-"))
        hf = d / "hf"
        if draft:
            snap = hf / "hub" / "models--maurienne-ai--Draft" / "snapshots" / DREV
            snap.mkdir(parents=True)
            (snap / "config.json").write_text("{}")
            (snap / "model.safetensors").write_bytes(b"w")
        etc = d / "etc"
        etc.mkdir()
        if unit:
            (etc / "qwen38-sglang.service").write_text("x")
        script = f"""
set -euo pipefail
LANE={lane}; UNIT_NAME=qwen38-sglang.service; ALL_WEIGHTS={all_weights}; UNITS_ONLY={units_only}; NO_SERVICE={no_service}
TEXT_NAMED={named}; KEEP_MODEL_VERBATIM={verbatim}; IMAGE={IMAGE}; FLASH_IMAGE={IMAGE}; HF_CACHE={hf}
MODEL_REPO=RadixArk/Qwen3.8-27B-NVFP4; MODEL_REV={REV}; DRAFT2_REPO=maurienne-ai/Draft; DRAFT2_REV={DREV}
text_lane_prepared(){{ return {0 if prepared else 1}; }}
docker(){{ return {0 if image else 1}; }}
ckpt_cached(){{ return {0 if ckpt else 1}; }}
""" + BLOCK.replace("/etc/systemd/system/", f"{etc}/") + """
echo "STATE PENDING_PUT=$PENDING_PUT WAITS_LOAD=$WAITS_LOAD UNITS_ONLY=$UNITS_ONLY"
"""
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def test_an_update_that_lacks_the_new_pins_waits_for_the_load(self):
        out = self.run_block(image=False, ckpt=False, draft=False)
        self.assertIn("STATE PENDING_PUT=1 WAITS_LOAD=1 UNITS_ONLY=1", out)
        self.assertIn(f"its image {'a' * 64}, RadixArk/Qwen3.8-27B-NVFP4 at 1111111, its draft maurienne-ai/Draft at ddddddd", out)
        self.assertIn("keeps serving what it served until its next Load", out)

    def test_an_update_whose_pins_are_all_here_runs_whole(self):
        self.assertIn("STATE PENDING_PUT=1 WAITS_LOAD=0 UNITS_ONLY=0", self.run_block())

    def test_asked_downloads_and_a_new_box_download_as_before(self):
        for kw in ({"named": 1}, {"no_service": 1}, {"verbatim": 1}):
            with self.subTest(**kw):
                self.assertIn("WAITS_LOAD=0 UNITS_ONLY=0", self.run_block(image=False, **kw))
        self.assertIn("STATE PENDING_PUT=0 WAITS_LOAD=0 UNITS_ONLY=0", self.run_block(image=False, all_weights=1))
        self.assertIn("STATE PENDING_PUT=0 WAITS_LOAD=0 UNITS_ONLY=0", self.run_block(image=False, unit=False))

    def test_a_lane_that_cannot_serve_as_it_is_is_not_this_case(self):
        # emptied or never loaded: the paths before say what it gets (left out, not prepared), not this one
        self.assertIn("STATE PENDING_PUT=1 WAITS_LOAD=0 UNITS_ONLY=0", self.run_block(prepared=False, image=False))

    def test_a_run_already_on_its_units_only_stays_as_it_was(self):
        self.assertIn("STATE PENDING_PUT=1 WAITS_LOAD=0 UNITS_ONLY=1", self.run_block(image=False, units_only=1))

    def test_the_flash_lane_has_no_draft_to_lack(self):
        self.assertIn("STATE PENDING_PUT=1 WAITS_LOAD=0 UNITS_ONLY=0", self.run_block(lane="flash", draft=False))


if __name__ == "__main__":
    unittest.main()
