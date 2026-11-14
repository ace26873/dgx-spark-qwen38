#!/usr/bin/env python3
"""The MiniMax-H3-Image lane (v1.49): its unit, its installer's bind, its weights and the licence said where it is said.

The runtime at the lane's SGLang commit has, as flags, what the image and video lanes patch in at theirs: the unit
passes them, and the CI's gate holds --sleep-on-idle there. The lane binds loopback only: its runtime has no API key,
and at this commit its scheduler's socket follows an explicit address. Its weights are its checkpoint, by an allowlist
that leaves the repository's diffusers copies and its own copy of the text encoder out, and MiniMax-H3's text encoder,
the video lane's, named by its path at the revision both pin. Its licence and MiniMax-H3's are not granted in four
territories (decision 1375 of Hasan, 2026-11-14: said plainly wherever the licence is said, nothing refused)."""
import pathlib
import re
import subprocess
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
UNIT = (REPO / "qwen38-image-h3.service.template").read_text()
INST = (REPO / "install-image-h3.sh").read_text()
SWITCH = (REPO / "switch-model.sh").read_text()
TERRITORIES = "is not granted in the European Union, the United Kingdom, South Korea or the United States"


def exec_start(text):
    return text[text.index("ExecStart="):text.index("\nRestart=")]


class TheUnit(unittest.TestCase):
    def test_it_passes_the_runtimes_own_idle_and_stop_flags_and_no_key(self):
        start = exec_start(UNIT)
        self.assertIn("--sleep-on-idle", start)
        self.assertIn("--http-graceful-shutdown-timeout 5", start)
        self.assertNotIn("--api-key", start)
        self.assertIn('--output-path ""', start)
        self.assertIn('--input-save-path ""', start)

    def test_its_text_encoder_is_named_by_its_path_and_the_filter_is_loopbacks(self):
        self.assertIn("--text-encoder-path __IMAGE_H3_ENCODER__", exec_start(UNIT))
        self.assertIn("IPAddressAllow=localhost\nIPAddressDeny=any", UNIT)

    def test_the_installer_renders_every_placeholder_of_it(self):
        for ph in sorted(set(re.findall(r"__[A-Z][A-Z0-9_]*__", UNIT))):
            self.assertIn(f"s|{ph}|", INST, ph)

    def test_the_encoder_path_is_the_pinned_revision_of_minimax_h3(self):
        self.assertIn('ENCODER_DIR="$HF_CACHE/hub/models--${IMAGE_H3_ENCODER_PIN//\\//--}/snapshots/$IMAGE_H3_ENCODER_PIN_REV/text_encoder"', INST)
        video = (REPO / "install-video.sh").read_text()
        self.assertEqual(re.search(r'^IMAGE_H3_ENCODER_PIN_REV="([0-9a-f]{40})"', INST, re.M).group(1),
                         re.search(r'^VIDEO_MODEL_PIN_REV="([0-9a-f]{40})"', video, re.M).group(1))


class TheBind(unittest.TestCase):
    def bind(self, value):
        block = INST[INST.index('IMAGE_H3_BIND="${IMAGE_H3_BIND:-$(unit_flag --host)}"'):INST.index("\nesac\n", INST.index('case "$IMAGE_H3_BIND" in')) + 6]
        script = ("set -euo pipefail\ndie(){ echo \"ERROR: $*\" >&2; exit 1; }\nunit_flag(){ :; }\n"
                  f"IMAGE_H3_BIND={value}\n" + block + '\necho "bind=$IMAGE_H3_BIND"\n')
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=10)

    def test_loopback_goes_on(self):
        for value in ("127.0.0.1", "127.0.0.2", "localhost"):
            r = self.bind(value)
            self.assertEqual((r.returncode, r.stdout.strip()), (0, f"bind={value}"), r.stderr)

    def test_anything_past_loopback_is_refused_with_the_reason(self):
        for value in ("0.0.0.0", "192.168.1.20", "100.64.0.1", "127.0.0.1;x", "::1"):
            r = self.bind(f"'{value}'")
            self.assertEqual(r.returncode, 1, (value, r.stdout))
            self.assertIn("IMAGE_H3_BIND", r.stderr)


class TheWeights(unittest.TestCase):
    def test_the_checkpoint_comes_by_an_allowlist_without_its_copies(self):
        allow = re.search(r'^ALLOW="([^"]+)"', INST, re.M).group(1).split()
        for need in ("transformer/*", "vae/*", "pdd_fused_heads.safetensors", "pdd_config.json", "model_index.json"):
            self.assertIn(need, allow)
        for left in ("text_encoder/*", "transformer_diffusers/*", "vae_diffusers/*"):
            self.assertNotIn(left, allow)
        self.assertIn('--allow "text_encoder/*"', INST)

    def test_the_switch_reads_them_where_the_installer_declares_them(self):
        names = re.search(r"grep -E '\^\(([A-Z0-9_|]+)\)=' \"\$REPO_DIR/install-image-h3.sh\"", SWITCH).group(1).split("|")
        for name in names:
            self.assertRegex(INST, rf"(?m)^{name}=", name)
        block = SWITCH[SWITCH.index('if [ "$CHOICE" = "image-h3" ]; then'):SWITCH.index("exit 0\nfi", SWITCH.index('if [ "$CHOICE" = "image-h3" ]; then'))]
        code = "\n".join(ln for ln in block.splitlines() if not ln.lstrip().startswith("echo "))
        self.assertNotRegex(code, r"systemctl (start|stop|restart) ", "a switch never starts, stops or restarts an engine")
        self.assertIn('--allow "$ALLOW" --offline-first --refs-main pin', block)


class TheLicenceIsSaidPlainly(unittest.TestCase):
    def test_the_image_lanes_where_its_licence_is_said(self):
        for rel in ("install-image-h3.sh", "switch-model.sh", "docs/image-h3-lane.md", "docs/models/minimax-h3-image.md",
                    "dashboard/static/js/image.js", "README.md"):
            text = " ".join((REPO / rel).read_text().split())
            self.assertIn("the MiniMax-H3-Image Community License, " + TERRITORIES, text, rel)
        self.assertIn('echo "MiniMax-H3-Image: ${LICENCE/Its licence/its licence}"\necho "fetching with allow: $ALLOW"', INST)

    def test_the_video_lanes_where_its_licence_is_said(self):
        for rel in ("install-video.sh", "switch-model.sh", "docs/video-lane.md", "docs/models/minimax-h3.md",
                    "dashboard/static/index.html"):
            text = " ".join((REPO / rel).read_text().split())
            self.assertIn("the MiniMax H3 Community License, " + TERRITORIES, text, rel)
        self.assertNotIn("Read it before commercial use", (REPO / "docs/video-lane.md").read_text())


if __name__ == "__main__":
    unittest.main()
