#!/usr/bin/env python3
"""The NVIDIA Nemotron Transcribe lane's installer (install-transcribe.sh), run as written.

Its --stage renders the unit for the GGUF files already in the cache, with no sudo and no
engine: the shape of the lane on demand. The cache it mounts is the one the fetch fills, read
the way the other lanes beside it read theirs.
"""
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
INSTALLER = REPO / "install-transcribe.sh"
UNIT = "qwen38-transcribe.service"


class TheStagedUnit(unittest.TestCase):
    def setUp(self):
        self.t = pathlib.Path(tempfile.mkdtemp(prefix="transcribe-stage-"))
        self.home = self.t / "home"
        self.cfg = self.home / ".config" / "qwen38"
        self.cfg.mkdir(parents=True)
        (self.cfg / "api-key").write_text("test-key\n")
        self.units = self.t / "units"
        self.units.mkdir()
        self.out = self.t / "stage.service"

    def tearDown(self):
        shutil.rmtree(self.t, ignore_errors=True)

    def stage(self, env=None):
        """The staged unit, with HF_CACHE deliberately absent from the environment."""
        e = {"PATH": "/usr/bin:/bin", "HOME": str(self.home), "LANES_UNIT_DIR": str(self.units), **(env or {})}
        r = subprocess.run(["bash", str(INSTALLER), "--stage", str(self.out)],
                           capture_output=True, text=True, env=e, timeout=120)
        return r.returncode, r.stdout + r.stderr

    def mounted_cache(self):
        return re.search(r"-v (\S+):/root/\.cache/huggingface:ro", self.out.read_text()).group(1)

    def a_unit_mounting(self, unit, cache):
        (self.units / unit).write_text(
            f"ExecStart=/bin/bash -c 'exec /usr/bin/docker run --rm --name {unit[:-len('.service')]} "
            f"-v {cache}:/root/.cache/huggingface:ro x'\n")

    def test_the_cache_the_text_lanes_mount_is_the_one_it_fetches_into(self):
        """A box whose cache lives on another disk (docs/external-disk.md: HF_CACHE=/mnt/ssd/huggingface
        ./install.sh) keeps the lane's 36.5 GB there: the installer reads the text lanes' cache the way
        the lanes beside it do (vllm-lane.sh's vlane_text_cache, install-image.sh's text_cache), not
        $HOME's, into which they would be downloaded again."""
        self.a_unit_mounting("qwen38-sglang.service", "/mnt/ssd/huggingface")
        rc, out = self.stage()
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.mounted_cache(), "/mnt/ssd/huggingface")

    def test_its_own_unit_keeps_priority_over_the_text_lanes(self):
        self.a_unit_mounting(UNIT, "/mnt/own/huggingface")
        self.a_unit_mounting("qwen38-sglang.service", "/mnt/ssd/huggingface")
        rc, out = self.stage()
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.mounted_cache(), "/mnt/own/huggingface")

    def test_with_no_unit_to_read_it_falls_back_to_home(self):
        rc, out = self.stage()
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.mounted_cache(), str(self.home / ".cache" / "huggingface"))

    def test_hf_cache_in_the_environment_wins_over_every_unit(self):
        self.a_unit_mounting("qwen38-sglang.service", "/mnt/ssd/huggingface")
        rc, out = self.stage(env={"HF_CACHE": "/mnt/asked/huggingface"})
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.mounted_cache(), "/mnt/asked/huggingface")


class TheProofRestartsALaneThatServesOlderFiles(unittest.TestCase):
    """A lane already serving when its installer runs holds its engine in memory: a start on an
    active unit is a no-op systemd answers with success, and the proof then asked the old engine,
    the one this run replaced on disk (vlane_prove's own defect, found in review, 2026-10-14;
    this hand-written proof had it too, found in review, 2026-11-19)."""

    def block(self):
        t = INSTALLER.read_text()
        start = t.index("TRANSCRIBE_PROVE_FILES=(")
        return t[start:t.index("fi\n", start) + len("fi\n")]

    def run_block(self, was_self, older):
        cfg = pathlib.Path(tempfile.mkdtemp(prefix="transcribe-prove-"))
        self.addCleanup(shutil.rmtree, cfg, ignore_errors=True)
        code = ("set -euo pipefail\n"
                "UNIT=qwen38-transcribe.service; INSTALLED=/etc/systemd/system/$UNIT\n"
                f"CONFIG_DIR={cfg}; WAS_SELF={was_self}\n"
                'TRANSCRIBE_FILE="a.gguf=sha:1"; TRANSCRIBE_MMPROJ="b.gguf=sha:2"\n'
                'model_host(){ echo "/gguf/$1"; }\n'
                f'lane_serves_older(){{ echo "OLDER $*"; return {older}; }}\n'
                'sudo(){ echo "SUDO $*"; }\n'
                + self.block())
        return subprocess.run(["bash", "-c", code], capture_output=True, text=True, timeout=30,
                              env={"PATH": "/usr/bin:/bin"})

    def test_a_serving_lane_with_older_files_restarts_for_the_proof(self):
        r = self.run_block(1, 0)          # lane_serves_older true: it serves older files
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("SUDO systemctl restart qwen38-transcribe.service", r.stdout)
        self.assertNotIn("systemctl start", r.stdout)

    def test_the_restart_decision_reads_the_files_this_run_touched(self):
        r = self.run_block(1, 0)
        asked = next(ln for ln in r.stdout.splitlines() if ln.startswith("OLDER "))
        for f in ("/etc/systemd/system/qwen38-transcribe.service", "llamacpp-files.sh",
                  "engine-preflight.sh", "/gguf/a.gguf=sha:1", "/gguf/b.gguf=sha:2"):
            self.assertIn(f, asked)

    def test_a_serving_lane_current_on_its_files_is_not_touched(self):
        r = self.run_block(1, 1)          # lane_serves_older false: it serves these files
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("SUDO", r.stdout)

    def test_a_lane_that_was_down_starts(self):
        r = self.run_block(0, 0)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("SUDO systemctl start qwen38-transcribe.service", r.stdout)
        self.assertNotIn("restart", r.stdout)


if __name__ == "__main__":
    unittest.main()
