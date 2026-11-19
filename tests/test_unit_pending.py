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

    def test_a_model_named_first_on_its_line(self):
        # Kolibri's and the Nemotron lanes' units (code review of v1.52.0: unit_needs did not see their checkpoint)
        text = (f"ExecStart=/bin/bash -c 'exec /usr/bin/docker run -v {self.b.hf}:/root/.cache/huggingface \\\n"
                f"  {OLD_IMG} \\\n    Aleph-Alpha/Kolibri-1 --revision {REV2} \\\n    --port 30000'\n")
        self.assertEqual(self.b.needs(text), [f"checkpoint Aleph-Alpha/Kolibri-1@{REV2}"])
        self.b.ckpt("Aleph-Alpha/Kolibri-1", REV2)
        self.assertEqual(self.b.needs(text), [])

    def test_a_vllm_drafter_named_by_its_speculative_config(self):
        # the Nemotron lanes' drafter (Vibe's blind review of v1.52.0, GLM 5.3: unit_needs did not see it)
        text = (f"ExecStart=/bin/bash -c 'exec /usr/bin/docker run -v {self.b.hf}:/root/.cache/huggingface \\\n"
                f"  {OLD_IMG} \\\n    nvidia/Nemotron-Bolt --revision {REV1} \\\n"
                f"    --speculative-config.method dflash --speculative-config.model nvidia/Bolt-Draft \\\n"
                f"    --speculative-config.revision {REV2} --speculative-config.num_speculative_tokens 7'\n")
        self.b.ckpt("nvidia/Nemotron-Bolt", REV1)
        self.assertEqual(self.b.needs(text), [f"draft nvidia/Bolt-Draft@{REV2}"])
        self.b.ckpt("nvidia/Bolt-Draft", REV2)
        self.assertEqual(self.b.needs(text), [])

    @unittest.skipIf(os.geteuid() == 0, "root reads a file of mode 000")
    def test_a_unit_it_cannot_read_is_not_taken_for_whole(self):
        # a root unit of mode 0600 read as "lacks nothing" (Vibe's blind review of v1.52.0, Mistral Large 4)
        f = self.b.d / "unreadable.service"
        f.write_text(self.b.text())
        f.chmod(0)
        r = self.b.sh(f'unit_needs "{f}"')
        self.assertEqual((r.returncode, r.stdout.splitlines()), (0, [f"unreadable {f}"]), r.stderr)

    def test_a_native_lane_reads_the_cache_its_environment_names(self):
        # the image and video lanes set HF_HOME: their checkpoint is looked for there, not in the default cache
        other = self.b.d / "other-disk"
        snap = other / "hub" / "models--MiniMaxAI--MiniMax-H3" / "snapshots" / REV1
        snap.mkdir(parents=True)
        (snap / "model.safetensors").write_bytes(b"w")
        text = (f"[Service]\nEnvironment=HF_HOME={other}\n"
                f"ExecStart=/home/x/venv/bin/python -m sglang serve --model-path MiniMaxAI/MiniMax-H3 --revision {REV1}\n")
        self.assertEqual(self.b.needs(text), [])
        (snap / "model.safetensors").unlink()
        self.assertEqual(self.b.needs(text), [f"checkpoint MiniMaxAI/MiniMax-H3@{REV1}"])

    def test_a_gguf_part_named_by_its_path(self):
        part = f"{self.b.hf}/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/{'e' * 40}/UD-IQ3_XXS/m-00001.gguf"
        text = f"ExecStart=/usr/bin/docker run --rm {OLD_IMG} -m {part} --port 30000\n"
        self.assertEqual(self.b.needs(text), [f"path {part}"])

    def test_a_gguf_named_by_its_path_in_the_container_is_looked_for_where_its_mount_puts_it_on_the_host(self):
        # the Unsloth lane's unit mounts <hf>:/root/.cache/huggingface and names its -m /root/.cache/huggingface/...
        # (install-flash-unsloth.sh renders it so): /root is 0700 on the host, so the container's path tested as it
        # is never exists, and every update of the lane waited for a file the box has (review of v1.52.0, 2026-11-19)
        rel = f"hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/{'e' * 40}/UD-IQ3_XXS/m-00001-of-00004.gguf"
        text = (f"ExecStart=/usr/bin/docker run --rm -v {self.b.hf}:/root/.cache/huggingface:ro {OLD_IMG} "
                f"-m /root/.cache/huggingface/{rel} --port 30000\n")
        part = self.b.hf / rel
        part.parent.mkdir(parents=True)
        part.write_bytes(b"GGUF")
        self.assertEqual(self.b.needs(text), [])
        part.unlink()
        self.assertEqual(self.b.needs(text), [f"path {self.b.hf}/{rel}"])


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
        # the unit in place it was rendered against (a switch since makes it stale: the cockpit's apply_pending)
        import hashlib
        self.assertEqual(pathlib.Path(str(self.pending) + ".base").read_text().strip(),
                         hashlib.sha256(self.b.unit.read_bytes()).hexdigest())

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
        self.assertFalse(pathlib.Path(str(self.pending) + ".base").exists())

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

    @unittest.skipIf(os.geteuid() == 0, "root reads a file of mode 000")
    def test_an_installed_unit_it_cannot_read_takes_its_new_unit_as_before(self):
        # unreadable, it was taken for whole and its new unit waited (Vibe's blind review of v1.52.0)
        self.b.unit.chmod(0)
        new = self.b.text(img=NEW_IMG)
        self.assertEqual(self.b.put(new), "PUT=new")
        self.assertEqual(self.b.unit.read_text(), new)
        self.assertFalse(self.pending.exists())

    def test_the_flash_lanes_waiting_launcher_survives_its_units_put(self):
        # its wait is its launch script's: the unit that only runs that script clears nothing (code review of v1.52.0)
        flash = self.b.d / "qwen38-flash.service"
        flash.write_text("[Service]\nExecStart=launch-flash.sh\n")
        pend = self.b.config / "pending"
        pend.mkdir()
        for f in ("launch-flash.sh", "qwen38-flash.service", "qwen38-flash.service.needs"):
            (pend / f).write_text("x\n")
        f = self.b.d / "rendered"
        f.write_text(flash.read_text())
        r = self.b.sh(f'P="$(unit_put "{f}" "{flash}")"\necho "PUT=$P"', env={"PENDING_PUT": "1"})
        self.assertIn("PUT=same", r.stdout, r.stderr)
        self.assertEqual(sorted(x.name for x in pend.iterdir()),
                         ["launch-flash.sh", "qwen38-flash.service", "qwen38-flash.service.needs"])

    def test_the_same_unit_again_clears_what_waited(self):
        self.assertEqual(self.b.put(self.b.text(img=NEW_IMG)), "PUT=pending")
        self.assertEqual(self.b.put(self.b.text()), "PUT=same")
        self.assertFalse(self.pending.exists())



class PrepareReadsTheWaitingUnit(unittest.TestCase):
    """prepare-lane.sh 27b prepares what the unit waiting for this Load reads (its image, its draft), not the unit in
    place, which serves the release before."""

    def test_the_unit_it_reads(self):
        text = (REPO / "prepare-lane.sh").read_text()
        block = text[text.index("  # an update left its new unit waiting for this Load"):text.index("  img=\"$(engine_image \"$unit\")\"")]
        d = pathlib.Path(tempfile.mkdtemp(prefix="prepare-pending-"))
        (d / "pending").mkdir()
        for waiting in (False, True):
            with self.subTest(waiting=waiting):
                if waiting:
                    (d / "pending" / "qwen38-sglang.service").write_text("new")
                    (d / "pending" / "qwen38-sglang.service.needs").write_text("image x\n")
                r = subprocess.run(["bash", "-c", f'set -euo pipefail\nCONFIG_DIR="{d}"; . "{REPO}/lanes.sh"\n'
                                    f'unit=/etc/systemd/system/qwen38-sglang.service\n{block}\necho "UNIT=$unit"'],
                                   capture_output=True, text=True, timeout=30)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertIn(f"UNIT={d}/pending/qwen38-sglang.service" if waiting
                              else "UNIT=/etc/systemd/system/qwen38-sglang.service", r.stdout)
                if waiting:
                    self.assertIn("preparing what its new unit reads (image x)", r.stdout)


class TheSwitchPutsALaneBesidesNewUnit(unittest.TestCase):
    def test_each_unit_its_switch_puts_in_place_clears_what_waited(self):
        # a lane beside the text lane takes its new unit from its switch, which renders it (code review of v1.52.0)
        import re
        text = (REPO / "switch-model.sh").read_text()
        puts = re.findall(r'sudo install -m 644 "\$(KOL|NEM|NSV|FLL|ORN|FLU|IMG)_STAGE" "\$([A-Z0-9_]+)"\n\s*unit_pending_clear "\$\2"', text)
        self.assertEqual(len(puts), 7, puts)

if __name__ == "__main__":
    unittest.main()
