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
            f'#!/bin/bash\n[ -n "${{DOCKER_DOWN:-}}" ] && exit 2\n[ "$1" = "info" ] && exit 0\n'
            f'[ "$1 $2" = "image inspect" ] || exit 2\ngrep -qxF -- "${{@: -1}}" "{self.images}"\n')
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

    def test_a_text_encoder_the_unit_names_by_its_path(self):
        # the image-h3 lane's unit loads MiniMax-H3's text_encoder by its folder at the pinned revision
        # (qwen38-image-h3.service.template's --text-encoder-path): a pin that moved and was never fetched is a
        # need, or the unit goes in place and the lane finds no encoder at its next start (review of v1.52.0,
        # 2026-11-19, GLM 5.3)
        enc = "e" * 40
        model_snap = self.b.hf / "hub" / "models--MiniMaxAI--MiniMax-H3-Image" / "snapshots" / REV1
        model_snap.mkdir(parents=True)
        (model_snap / "model.safetensors").write_bytes(b"w")

        def unit(rev):
            return (f"[Service]\nEnvironment=HF_HOME={self.b.hf}\n"
                    f"ExecStart=/x/venv/bin/python -m sglang serve --model-path {model_snap} "
                    f"--text-encoder-path {self.b.hf}/hub/models--MiniMaxAI--MiniMax-H3/snapshots/{rev}/text_encoder"
                    f" --port 30026\n")
        encdir = self.b.hf / "hub" / "models--MiniMaxAI--MiniMax-H3" / "snapshots" / enc / "text_encoder"
        self.assertEqual(self.b.needs(unit(enc)), [f"path {encdir}"])
        encdir.mkdir(parents=True)
        (encdir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "te-1.safetensors"}}))
        (encdir / "te-1.safetensors").write_bytes(b"e")
        self.assertEqual(self.b.needs(unit(enc)), [])
        (encdir / "te-1.safetensors").unlink()
        self.assertEqual(self.b.needs(unit(enc)), [f"path {encdir}"])

    def test_a_docker_that_does_not_answer_is_said_unknown_not_missing(self):
        # unit_put must not read "docker down" as "the image of the unit in place is missing" (review of
        # v1.52.0, 2026-11-19: it did, and a new unit whose checkpoint was not there went in place)
        f = self.b.d / "rendered"
        f.write_text(self.b.text())
        r = self.b.sh(f'unit_needs "{f}"', env={"DOCKER_DOWN": "1"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.splitlines(), [f"unknown image {OLD_IMG} (docker did not answer)"])

    @unittest.skipIf(os.geteuid() == 0, "root reads a folder of mode 000")
    def test_an_unreadable_cache_is_said_unknown_not_missing(self):
        self.b.hf.chmod(0)
        f = self.b.d / "rendered"
        f.write_text(self.b.text())
        r = self.b.sh(f'unit_needs "{f}"')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.splitlines(), [f"unknown cache {self.b.hf}"])
        self.b.hf.chmod(0o755)

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

    def test_a_docker_down_during_the_update_keeps_the_unit_in_place_and_waits(self):
        # the unit in place cannot be concluded on (its image reads unknown): it keeps serving what it served,
        # the new one, whose checkpoint is not there, waits for its Load, never the reverse (review of v1.52.0,
        # 2026-11-19)
        new = self.b.text(img=NEW_IMG, rev=REV2)
        self.b.d.joinpath("rendered").write_text(new)
        r = self.b.sh(f'P="$(unit_put \"{self.b.d}/rendered\" \"{self.b.unit}\")"\necho "PUT=$P"',
                      env={"PENDING_PUT": "1", "DOCKER_DOWN": "1"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("PUT=pending", r.stdout, r.stdout)
        self.assertEqual(self.b.unit.read_text(), self.b.text(), "the unit in place serves what it served, at boot too")
        self.assertEqual(self.pending.read_text(), new)
        self.assertIn(f"checkpoint RadixArk/Qwen3.8-27B-NVFP4@{REV2}",
                      pathlib.Path(str(self.pending) + ".needs").read_text())

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


class TheTextLaneSwitchClearsTheWaitItMakesStale(unittest.TestCase):
    """The 27B's and the flash lane's switch rewrites the unit or the launcher: an update that waited, rendered
    against the file before, is caduque, and is cleared there, as every lane beside clears its wait at its own
    switch (review of v1.52.0, 2026-11-19: it stayed, and a Load after the switch downloaded what it lacked then
    dropped it). The two blocks run here against a pending folder of the test's own."""

    def run_block(self, which, rewritten=True, waiting=True):
        text = (REPO / "switch-model.sh").read_text()
        if which == "27b":
            start = text.index('  if cmp -s "$STAGE_UNIT" "$TARGET_UNIT"; then')
        else:
            start = text.index('  if cmp -s "$TMP_UNIT" "$FLASH_LAUNCH"; then')
        end = text.index("\n  fi\n", start) + len("\n  fi\n")
        block = text[start:end]
        d = pathlib.Path(tempfile.mkdtemp(prefix=f"switch-clear-{which}-"))
        cfg = d / "config"
        (cfg / "pending").mkdir(parents=True)
        log = d / "log"
        log.write_text("")
        unit_path = d / ("qwen38-sglang.service" if which == "27b" else "qwen38-flash.service")
        unit_path.write_text("old\n")
        stage_path = d / "stage"
        stage_path.write_text("new\n" if rewritten else "old\n")
        launch = cfg / "launch-flash.sh"
        launch.write_text("#!/bin/bash\nold\n")
        if waiting:
            (cfg / "pending" / "qwen38-sglang.service").write_text("waiting\n")
            (cfg / "pending" / "qwen38-flash.service").write_text("waiting\n")
            (cfg / "pending" / "launch-flash.sh").write_text("#!/bin/bash\nwaiting\n")
        script = f"""
set -euo pipefail
CONFIG_DIR="{cfg}"
CHOICE=stock
TARGET_UNIT="{unit_path}"; TARGET_UNIT_NAME="{unit_path.name}"
FLASH_UNIT="{unit_path}"; FLASH_LAUNCH="{launch}"
STAGE_UNIT="{stage_path}"; TMP_UNIT="{stage_path}"
sudo(){{ echo "sudo $*" >> "{log}"; if [ "$1" = install ]; then cp "$4" "$5"; fi; }}
unit_pending_clear(){{ echo "clear $1" >> "{log}"; rm -f "$CONFIG_DIR/pending/$(basename "$1")" "$CONFIG_DIR/pending/$(basename "$1").needs" "$CONFIG_DIR/pending/$(basename "$1").base"; }}
""" + block
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
        return r, cfg, log, unit_path

    def test_a_rewritten_unit_clears_what_waited(self):
        r, cfg, log, unit_path = self.run_block("27b")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("was dropped", r.stdout)
        self.assertIn(f"clear {unit_path}", log.read_text())
        self.assertFalse((cfg / "pending" / "qwen38-sglang.service").exists())

    def test_a_rewritten_launcher_clears_what_waited(self):
        r, cfg, log, unit_path = self.run_block("flash")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("was dropped", r.stdout)
        self.assertIn(f"clear {cfg}/launch-flash.sh", log.read_text())
        self.assertIn(f"clear {unit_path}", log.read_text())
        self.assertFalse((cfg / "pending" / "launch-flash.sh").exists())
        self.assertFalse((cfg / "pending" / "qwen38-flash.service").exists())

    def test_an_unchanged_unit_keeps_its_wait(self):
        # the switch did not rewrite the file: the wait was rendered against it and is still applicable
        r, cfg, log, unit_path = self.run_block("27b", rewritten=False)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("\nclear ", "\n" + log.read_text())
        self.assertTrue((cfg / "pending" / "qwen38-sglang.service").exists())

    def test_nothing_waiting_nothing_said(self):
        r, cfg, log, unit_path = self.run_block("27b", waiting=False)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("\nclear ", "\n" + log.read_text())
        self.assertNotIn("dropped", r.stdout)


class TheSwitchPutsALaneBesidesNewUnit(unittest.TestCase):
    def test_each_unit_its_switch_puts_in_place_clears_what_waited(self):
        # a lane beside the text lane takes its new unit from its switch, which renders it (code review of v1.52.0)
        import re
        text = (REPO / "switch-model.sh").read_text()
        puts = re.findall(r'sudo install -m 644 "\$(KOL|NEM|NSV|FLL|ORN|FLU|IMG)_STAGE" "\$([A-Z0-9_]+)"\n\s*unit_pending_clear "\$\2"', text)
        self.assertEqual(len(puts), 7, puts)


class TheVideoAndImageH3SwitchAppliesTheWaitingUnit(unittest.TestCase):
    """The video and MiniMax-H3-Image lanes' switch is the one place their Load can apply an update: their
    --prepare builds the runtime only, and their switch never rewrote a unit (review of v1.52.0, 2026-11-19,
    GLM 5.3: a waiting unit was never posed nor cleared, the card's Load left the lane on the old release and
    said so forever). The block runs here against a pending folder of the test's own."""

    @staticmethod
    def block(unit_var, stage_var, lane):
        text = (REPO / "switch-model.sh").read_text()
        marker = "  # The update that waited for this Load (if any) goes in place now"
        starts = [i for i in range(len(text)) if text.startswith(marker, i)]
        start = starts[0 if lane == "video" else 1]
        end = next(i for i in range(start + len(marker), len(text)) if text.startswith("  # Which", i))
        return text[start:end], text, lane, unit_var, stage_var

    def run_block(self, lane, unit_var, stage_var, pending_text="new unit\n", base=None, needs="", same=False):
        d = pathlib.Path(tempfile.mkdtemp(prefix=f"switch-{lane}-"))
        cfg, etc = d / "config", d / "etc"
        (cfg / "pending").mkdir(parents=True)
        etc.mkdir()
        unit = etc / f"qwen38-{lane}.service"
        unit.write_text("old unit\n" if not same else pending_text)
        pend = cfg / "pending" / f"qwen38-{lane}.service"
        pend.write_text(pending_text)
        (cfg / "pending" / f"qwen38-{lane}.service.needs").write_text("image x\n")
        if base is None:
            import hashlib
            base = hashlib.sha256(unit.read_bytes()).hexdigest()
        (cfg / "pending" / f"qwen38-{lane}.service.base").write_text(base + "\n")
        log = d / "log"
        log.write_text("")
        script = f"""
set -euo pipefail
CONFIG_DIR="{cfg}"
{unit_var}="{unit}"; {unit_var}_NAME="qwen38-{lane}.service"
{stage_var}="$CONFIG_DIR/qwen38-{lane}.service.switch-stage"
sudo(){{ echo "sudo $*" >> "{log}"; if [ "$1" = install ]; then cp "$4" "$5"; fi; }}
unit_needs(){{ {f'echo "{needs}"' if needs else ':'}; }}
unit_pending_clear(){{ echo "clear $1" >> "{log}"; rm -f "$CONFIG_DIR/pending/$(basename "$1")" "$CONFIG_DIR/pending/$(basename "$1").needs" "$CONFIG_DIR/pending/$(basename "$1").base"; }}
""" + self.block(unit_var, stage_var, lane)[0] + f'\necho "UNIT=$(cat "{unit}")"\n'
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
        return r, unit, pend, cfg, log

    def test_the_waiting_unit_goes_in_place_and_the_wait_is_cleared(self):
        for lane, unit_var, stage_var in (("video", "VIDEO_UNIT", "VID_STAGE"),
                                          ("image-h3", "IMAGE_H3_UNIT", "H3_STAGE")):
            with self.subTest(lane=lane):
                r, unit, pend, cfg, log = self.run_block(lane, unit_var, stage_var)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertIn("is in place", r.stdout)
                self.assertEqual(unit.read_text(), "new unit\n")
                self.assertIn(f'sudo install -m 644 {cfg}/qwen38-{lane}.service.switch-stage {unit}', log.read_text())
                self.assertIn(f"clear {unit}", log.read_text())
                self.assertFalse(pend.exists())
                self.assertFalse((cfg / "pending" / f"qwen38-{lane}.service.needs").exists())
                self.assertFalse((cfg / f"qwen38-{lane}.service.switch-stage").exists(), "the stage is left behind")

    def test_a_wait_rendered_before_a_switch_is_dropped_not_put(self):
        r, unit, pend, cfg, log = self.run_block("video", "VIDEO_UNIT", "VID_STAGE", base="0" * 64)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("was dropped", r.stdout)
        self.assertEqual(unit.read_text(), "old unit\n")
        self.assertNotIn("sudo install", log.read_text())
        self.assertFalse(pend.exists())

    def test_a_wait_that_still_lacks_stays_waiting(self):
        r, unit, pend, cfg, log = self.run_block("image-h3", "IMAGE_H3_UNIT", "H3_STAGE", needs="path /hf/enc")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("still lacks path /hf/enc", r.stdout)
        self.assertEqual(unit.read_text(), "old unit\n")
        self.assertNotIn("sudo install", log.read_text())
        self.assertTrue(pend.exists(), "the wait is kept, not cleared")

    def test_an_unchanged_waiting_unit_is_cleared_without_install(self):
        r, unit, pend, cfg, log = self.run_block("video", "VIDEO_UNIT", "VID_STAGE", same=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("sudo install", log.read_text())
        self.assertIn(f"clear {unit}", log.read_text())
        self.assertFalse(pend.exists())

    def test_each_branch_fetches_what_the_waiting_unit_reads(self):
        # the fetch target (the model, its variant) is read from the pending unit when one waits, so the
        # checkpoint the Load downloads is the one the waiting unit names
        text = (REPO / "switch-model.sh").read_text()
        for lane, src, unit_var in (("video", "VID_SRC", "VIDEO_UNIT"), ("image-h3", "H3_SRC", "IMAGE_H3_UNIT")):
            with self.subTest(lane=lane):
                self.assertIn(f'{src}="$CONFIG_DIR/pending/${unit_var}_NAME"', text)
                for flag in ("--model-path", "--model-variant") if lane == "video" else ("--model-path",):
                    self.assertIn(f"grep -oE -- '{flag} [^ ]+' \"${src}\"", text)

    def test_their_staging_paths_have_their_sudoers_lines(self):
        sudoers = (REPO / "dashboard" / "sudoers-cockpit.template").read_text()
        ci = (REPO / ".github/workflows/ci.yml").read_text()
        for lane in ("video", "image-h3"):
            with self.subTest(lane=lane):
                unit = f"qwen38-{lane}.service"
                self.assertIn(f"/usr/bin/install -m 644 __HOME__/.config/qwen38/{unit}.switch-stage /etc/systemd/system/{unit}", sudoers)
                stage = "VID_STAGE" if lane == "video" else "H3_STAGE"
                self.assertRegex((REPO / "switch-model.sh").read_text(), rf'(?m)^{stage}="\$CONFIG_DIR/{unit}\.switch-stage"$')
                gate = next(ln for ln in ci.splitlines() if "stage = dict(re.findall" in ln)
                self.assertIn(stage, gate, "the CI gate's stage regex knows it")
                self.assertIn(f'"{stage}": {{"__HOME__/.config/qwen38/" + stage["{stage}"]}},', ci)


class TheImageSwitchAppliesTheWaitingUnit(unittest.TestCase):
    """The image lane's switch applies the unit that waited for its Load, as the video and MiniMax-H3-Image
    lanes' do: it fetches what the waiting unit reads and poses it whole, then clears the wait (review of
    v1.52.0, 2026-11-19, GLM 5.3: this branch read the wait nowhere, so a pin that moved left the Load
    fetching the old pin, posing nothing, clearing nothing, the card's Load looping forever). The block
    runs here against a pending folder of the test's own."""

    MARKER = "  # The unit names one checkpoint; a switch between the lane's two rewrites that name and"
    END = "  # Which text lane this box served"

    def block(self):
        text = (REPO / "switch-model.sh").read_text()
        start = text.index(self.MARKER)
        end = text.index(self.END, start)
        return text[start:end]

    def run_block(self, pending_text="new unit\n", base=None, needs="", same=False, cur="pin-a", model="pin-a"):
        d = pathlib.Path(tempfile.mkdtemp(prefix="switch-image-"))
        cfg, etc = d / "config", d / "etc"
        (cfg / "pending").mkdir(parents=True)
        etc.mkdir()
        unit = etc / "qwen38-image.service"
        unit.write_text("old unit\n" if not same else pending_text)
        pend = cfg / "pending" / "qwen38-image.service"
        pend.write_text(pending_text)
        (cfg / "pending" / "qwen38-image.service.needs").write_text("image x\n")
        if base is None:
            import hashlib
            base = hashlib.sha256(unit.read_bytes()).hexdigest()
        (cfg / "pending" / "qwen38-image.service.base").write_text(base + "\n")
        log = d / "log"
        log.write_text("")
        script = f"""
set -euo pipefail
CONFIG_DIR="{cfg}"
IMAGE_UNIT="{unit}"; IMAGE_UNIT_NAME="qwen38-image.service"
IMG_STAGE="$CONFIG_DIR/qwen38-image.service.switch-stage"
IMG_CUR="{cur}"; IMG_MODEL="{model}"; NEEDS='{needs}'
sudo(){{ echo "sudo $*" >> "{log}"; if [ "$1" = install ]; then cp "$4" "$5"; fi; }}
unit_needs(){{ if [ -n "$NEEDS" ]; then echo "$NEEDS"; fi; }}
unit_pending_clear(){{ echo "clear $1" >> "{log}"; rm -f "$CONFIG_DIR/pending/$(basename "$1")" "$CONFIG_DIR/pending/$(basename "$1").needs" "$CONFIG_DIR/pending/$(basename "$1").base"; }}
""" + self.block() + f'\necho "UNIT=$(cat "{unit}")"\n'
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
        return r, unit, pend, cfg, log

    def test_the_waiting_unit_goes_in_place_and_the_wait_is_cleared(self):
        # the pin moved: the unit in place names pin-a, the one waiting names pin-b, this Load serves pin-b
        r, unit, pend, cfg, log = self.run_block(cur="pin-a", model="pin-b", pending_text="--model-path pin-b\n")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("is in place", r.stdout)
        self.assertEqual(unit.read_text(), "--model-path pin-b\n")
        self.assertIn(f'sudo install -m 644 {cfg}/qwen38-image.service.switch-stage {unit}', log.read_text())
        self.assertIn(f"clear {unit}", log.read_text())
        self.assertFalse(pend.exists())
        self.assertFalse((cfg / "pending" / "qwen38-image.service.needs").exists())
        self.assertFalse((cfg / "qwen38-image.service.switch-stage").exists(), "the stage is left behind")

    def test_a_wait_rendered_before_a_switch_is_dropped_not_put(self):
        r, unit, pend, cfg, log = self.run_block(base="0" * 64)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("was dropped", r.stdout)
        self.assertEqual(unit.read_text(), "old unit\n")
        self.assertNotIn("sudo install", log.read_text())
        self.assertFalse(pend.exists())

    def test_a_wait_that_still_lacks_stays_waiting(self):
        r, unit, pend, cfg, log = self.run_block(needs="checkpoint repo@rev")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("still lacks checkpoint repo@rev", r.stdout)
        self.assertEqual(unit.read_text(), "old unit\n")
        self.assertNotIn("sudo install", log.read_text())
        self.assertTrue(pend.exists(), "the wait is kept, not cleared")

    def test_an_unchanged_waiting_unit_is_cleared_without_install(self):
        r, unit, pend, cfg, log = self.run_block(same=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("sudo install", log.read_text())
        self.assertIn(f"clear {unit}", log.read_text())
        self.assertFalse(pend.exists())

    def test_without_a_wait_the_switch_between_the_two_checkpoints_still_rewrites(self):
        # no pending unit: the piecemeal rewrite between the lane's two checkpoints is unchanged
        d = pathlib.Path(tempfile.mkdtemp(prefix="switch-image-nopend-"))
        cfg, etc = d / "config", d / "etc"
        cfg.mkdir(parents=True)
        etc.mkdir()
        unit = etc / "qwen38-image.service"
        unit.write_text("--model-path pin-a \nother\n")
        log = d / "log"
        log.write_text("")
        script = f"""
set -euo pipefail
CONFIG_DIR="{cfg}"
IMAGE_UNIT="{unit}"; IMAGE_UNIT_NAME="qwen38-image.service"
IMG_STAGE="$CONFIG_DIR/qwen38-image.service.switch-stage"
IMG_CUR="pin-a"; IMG_MODEL="pin-b"; IMAGE_MODEL_PIN="pin-b"; IMAGE_TURBO_PIN="pin-t"
IMG_BEFORE_TURBO="$CONFIG_DIR/image-model-before-turbo"
sudo(){{ echo "sudo $*" >> "{log}"; if [ "$1" = install ]; then cp "$4" "$5"; fi; }}
unit_needs(){{ :; }}
unit_pending_clear(){{ echo "clear $1" >> "{log}"; rm -f "$CONFIG_DIR/pending/$(basename "$1")"; }}
""" + self.block() + f'\necho "UNIT=$(cat "{unit}")"\n'
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--model-path pin-b ", unit.read_text())
        self.assertIn(f'sudo install -m 644 {cfg}/qwen38-image.service.switch-stage {unit}', log.read_text())

    def test_it_fetches_what_the_waiting_unit_reads(self):
        # the fetch target (the model) is read from the pending unit when one waits, so the checkpoint
        # the Load downloads is the one the waiting unit names
        text = (REPO / "switch-model.sh").read_text()
        self.assertIn('IMG_SRC="$CONFIG_DIR/pending/$IMAGE_UNIT_NAME"', text)
        self.assertIn('grep -oE -- \'--model-path [^ ]+\' "$IMG_SRC"', text)


if __name__ == "__main__":
    unittest.main()
