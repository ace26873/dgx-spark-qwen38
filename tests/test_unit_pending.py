#!/usr/bin/env python3
"""An update downloads nothing large (decisions 1404 and 1408 to 1410 of Hasan, 2026-11-16 and 17). lanes.sh's
unit_needs says what a lane's unit reads that the box lacks (an image, a checkpoint whole at its revision, the 27B's
draft, a model named by its path); unit_put, with PENDING_PUT=1, leaves a lane that can serve as it is when its new
unit reads what the box lacks: the new unit waits in $CONFIG_DIR/pending/, what it lacks beside it, for the lane's
Load. Run against a docker and a Hugging Face cache of the test's own."""
import json
import os
import pathlib
import subprocess
import tempfile
import time
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
OLD_IMG = "lmsysorg/sglang@sha256:" + "a" * 64
NEW_IMG = "lmsysorg/sglang@sha256:" + "b" * 64
REV1, REV2, DREV = "1" * 40, "2" * 40, "d" * 40


class Box:
    def __init__(self):
        self.d = pathlib.Path(tempfile.mkdtemp(prefix="unit-pending-"))
        self.bin = self.d / "bin"
        self.bin.mkdir()
        (self.bin / "sudo").write_text('#!/bin/bash\nexec "$@"\n')
        self.images = self.d / "images"
        self.images.write_text(OLD_IMG + "\n")
        (self.bin / "docker").write_text(
            f'#!/bin/bash\n[ "$1 $2" = "image inspect" ] || exit 2\ngrep -qxF -- "${{@: -1}}" "{self.images}"\n')
        for f in ("sudo", "docker"):
            (self.bin / f).chmod(0o755)
        self.hf = self.d / "hf"
        self.config = self.d / "config"
        self.config.mkdir()
        self.unit = self.d / "qwen38-sglang.service"
        self.ckpt("RadixArk/Qwen3.8-27B-NVFP4", REV1)
        self.ckpt("maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal", DREV, draft=True)

    def ckpt(self, repo, rev, draft=False):
        snap = self.hf / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / rev
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}")
        (snap / "model.safetensors").write_bytes(b"w")

    def text(self, img=OLD_IMG, rev=REV1):
        return (f"[Service]\nExecStart=/bin/bash -c 'exec /usr/bin/docker run --rm -v {self.hf}:/root/.cache/huggingface \\\n"
                f"  {img} \\\n  python3 -m sglang.launch_server --model-path RadixArk/Qwen3.8-27B-NVFP4 --revision {rev} \\\n"
                f"  --speculative-draft-model-path maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal "
                f"--speculative-draft-model-revision {DREV} \\\n  --speculative-num-draft-tokens 16'\n")

    def sh(self, script, env=None):
        r = subprocess.run(["bash", "-c", f'set -euo pipefail\nCONFIG_DIR="{self.config}"; . "{REPO}/lanes.sh"\n' + script],
                           capture_output=True, text=True, timeout=60,
                           env={"PATH": f"{self.bin}:/usr/bin:/bin", "HOME": str(self.d), **(env or {})})
        return r

    def needs(self, text):
        f = self.d / "rendered"
        f.write_text(text)
        r = self.sh(f'unit_needs "{f}"')
        self.t.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.splitlines()

    def put(self, new, pending=True):
        f = self.d / "rendered"
        f.write_text(new)
        r = self.sh(f'P="$(unit_put "{f}" "{self.unit}")"\necho "PUT=$P"', env={"PENDING_PUT": "1"} if pending else {})
        self.t.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.strip().splitlines()[-1]


class UnitNeeds(unittest.TestCase):
    def setUp(self):
        self.b = Box()
        self.b.t = self

    def test_a_unit_whose_inputs_are_all_here_needs_nothing(self):
        self.assertEqual(self.b.needs(self.b.text()), [])

    def test_each_missing_input_is_named(self):
        self.assertEqual(self.b.needs(self.b.text(img=NEW_IMG, rev=REV2)),
                         [f"image {NEW_IMG}", f"checkpoint RadixArk/Qwen3.8-27B-NVFP4@{REV2}"])
        draft = self.b.hf / "hub" / "models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal" / "snapshots" / DREV
        (draft / "model.safetensors").unlink()
        self.assertEqual(self.b.needs(self.b.text()),
                         [f"draft maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal@{DREV}"])

    def test_a_checkpoint_begun_is_not_whole(self):
        snap = self.b.hf / "hub" / "models--RadixArk--Qwen3.8-27B-NVFP4" / "snapshots" / REV1
        (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "m-1.safetensors",
                                                                                     "b": "m-2.safetensors"}}))
        (snap / "m-1.safetensors").write_bytes(b"1")
        self.assertEqual(self.b.needs(self.b.text()), [f"checkpoint RadixArk/Qwen3.8-27B-NVFP4@{REV1}"])

    def test_a_gguf_part_named_by_its_path(self):
        part = f"{self.b.hf}/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/{'e' * 40}/UD-IQ3_XXS/m-00001.gguf"
        text = f"ExecStart=/usr/bin/docker run --rm {OLD_IMG} -m {part} --port 30000\n"
        self.assertEqual(self.b.needs(text), [f"path {part}"])


class UnitPutOnAnUpdate(unittest.TestCase):
    def setUp(self):
        self.b = Box()
        self.b.t = self
        self.b.unit.write_text(self.b.text())
        os.utime(self.b.unit, (1_700_000_000, 1_700_000_000))
        self.pending = self.b.config / "pending" / self.b.unit.name

    def test_a_new_unit_that_reads_what_the_box_lacks_waits_and_the_lane_stays_as_it_was(self):
        new = self.b.text(img=NEW_IMG)
        self.assertEqual(self.b.put(new), "PUT=pending")
        self.assertEqual(self.b.unit.read_text(), self.b.text(), "the unit in place serves what it served, at boot too")
        self.assertEqual(int(self.b.unit.stat().st_mtime), 1_700_000_000)
        self.assertEqual(self.pending.read_text(), new)
        self.assertEqual(pathlib.Path(str(self.pending) + ".needs").read_text(), f"image {NEW_IMG}\n")

    def test_what_it_lacks_is_said_in_one_line(self):
        self.assertEqual(self.b.put(self.b.text(img=NEW_IMG, rev=REV2)), "PUT=pending")
        r = self.b.sh(f'unit_pending_says "{self.b.unit}"')
        self.assertEqual(r.stdout.strip(), f"image {NEW_IMG}; checkpoint RadixArk/Qwen3.8-27B-NVFP4@{REV2}")

    def test_once_the_box_has_it_all_the_unit_goes_in_place_and_nothing_waits(self):
        new = self.b.text(img=NEW_IMG)
        self.assertEqual(self.b.put(new), "PUT=pending")
        self.b.images.write_text(OLD_IMG + "\n" + NEW_IMG + "\n")       # the Load pulled it
        self.assertEqual(self.b.put(new), "PUT=new")
        self.assertEqual(self.b.unit.read_text(), new)
        self.assertFalse(self.pending.exists())
        self.assertFalse(pathlib.Path(str(self.pending) + ".needs").exists())

    def test_a_run_that_does_not_ask_for_it_puts_the_unit_in_place_as_before(self):
        new = self.b.text(img=NEW_IMG)
        self.assertEqual(self.b.put(new, pending=False), "PUT=new")
        self.assertEqual(self.b.unit.read_text(), new)

    def test_a_lane_that_cannot_serve_as_it_is_gets_its_new_unit(self):
        # emptied (its image gone): nothing to keep serving, its Load prepares the new unit
        self.b.images.write_text("")
        new = self.b.text(img=NEW_IMG)
        self.assertEqual(self.b.put(new), "PUT=new")
        self.assertEqual(self.b.unit.read_text(), new)
        self.assertFalse(self.pending.exists())

    def test_the_same_unit_again_clears_what_waited(self):
        self.assertEqual(self.b.put(self.b.text(img=NEW_IMG)), "PUT=pending")
        self.assertEqual(self.b.put(self.b.text()), "PUT=same")
        self.assertFalse(self.pending.exists())


if __name__ == "__main__":
    unittest.main()
