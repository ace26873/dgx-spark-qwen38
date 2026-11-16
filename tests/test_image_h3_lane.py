#!/usr/bin/env python3
"""The MiniMax-H3-Image lane (v1.50): its unit, its installer's bind, its weights and the licence said where it is said.

The runtime at the lane's SGLang commit has, as flags, what the image and video lanes patch in at theirs: the unit
passes them, and the CI's gate holds --sleep-on-idle there. The lane binds loopback only: its runtime has no API key,
and at this commit its scheduler's socket follows an explicit address. Its weights are its checkpoint, by an allowlist
that leaves the repository's diffusers copies and its own copy of the text encoder out, and MiniMax-H3's text encoder,
the video lane's, named by its path at the revision both pin. Its licence and MiniMax-H3's are not granted in four
territories (decision 1375 of Hasan, 2026-11-14: said plainly wherever the licence is said, nothing refused)."""
import pathlib
import re
import subprocess
import sys
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

    def test_the_checkpoint_is_named_by_its_snapshots_path(self):
        # by name, an offline start checked the snapshot against the repository's whole cached file list, and this lane
        # fetches only what it loads: every start failed (IncompleteSnapshotError, its first proof, 2026-11-16)
        self.assertIn('MODEL_SNAP="$HF_CACHE/hub/models--${MODEL//\\//--}/snapshots/$MODEL_REV"', INST)
        self.assertIn('-e "s|__IMAGE_H3_MODEL__|$MODEL_SNAP|g"', INST)
        self.assertIn('[[ "$MODEL_REV" =~ ^[0-9a-f]{40}$ ]] || die', INST, "a branch name would make no snapshot path")
        self.assertIn('_match_registered_hf_path(sys.argv[1]) != "MiniMaxAI/MiniMax-H3-Image"', INST)

    def test_the_switch_reads_the_repository_and_revision_off_the_units_path(self):
        import tempfile
        sw = (REPO / "switch-model.sh").read_text()
        start = sw.index('  H3_MODEL="$(grep -oE -- \'--model-path [^ ]+\' "$IMAGE_H3_UNIT"')
        end = sw.index("\n", sw.index('    if [ "$H3_MODEL" = "$IMAGE_H3_MODEL_PIN" ]; then H3_REV='))
        block = sw[start:end] + "\n  fi\n"
        d = pathlib.Path(tempfile.mkdtemp(prefix="h3-switch-"))
        self.addCleanup(__import__("shutil").rmtree, d, True)
        rev, pin = "a" * 40, "b" * 40
        cases = {"path": (f"--model-path /c/hub/models--MiniMaxAI--MiniMax-H3-Image/snapshots/{rev} \\", rev),
                 "slash": (f"--model-path /c/hub/models--MiniMaxAI--MiniMax-H3-Image/snapshots/{rev}/ \\", rev),
                 "name": ("--model-path MiniMaxAI/MiniMax-H3-Image \\", pin)}
        for how, (line, want) in cases.items():
            with self.subTest(how):
                unit = d / f"{how}.service"
                unit.write_text(f"[Service]\nExecStart=/x/sglang serve {line}\n")
                script = (f'IMAGE_H3_UNIT="{unit}"; HF_CACHE="{d}"; IMAGE_H3_MODEL_PIN=MiniMaxAI/MiniMax-H3-Image; '
                          f'IMAGE_H3_MODEL_PIN_REV={pin}\n' + block + '\necho "$H3_MODEL $H3_REV"')
                r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
                self.assertEqual(r.stdout.split(), ["MiniMaxAI/MiniMax-H3-Image", want], r.stderr)

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
        # 127.0.0.300 is no address: sglang could not bind it and the smoke would burn 20 min on it
        # (independent review of v1.50)
        for value in ("0.0.0.0", "192.168.1.20", "100.64.0.1", "127.0.0.1;x", "::1",
                      "127.0.0.300", "127.0.256.1", "127.1", "127.0.0"):
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




class TheFlashinferGate(unittest.TestCase):
    """An update of a complete runtime must NOT re-download the 1.6 GB of FlashInfer wheels: the gate
    compares importlib.metadata's version with the version the wheel FILE names (a review of v1.50
    found it comparing "0.7.2" with "0.7.2-py3-none-any.whl", true forever, which also touched the
    runtime note on every run and made the cockpit claim the serving lane ran older files)."""

    def gate(self, versions):
        t = pathlib.Path(self.subdir)
        venv = t / "venv"
        (venv / "bin").mkdir(parents=True, exist_ok=True)
        (venv / "bin" / "python").write_text(
            "#!/bin/sh\n"
            'case "$3" in\n'    # python -c CODE PKG: the package name reaches the stub third
            + "".join(f'  {k}) echo {v} ;;\n' for k, v in versions.items())
            + '  *) echo "" ;;\nesac\n')
        (venv / "bin" / "python").chmod(0o755)
        pip = t / "pip-calls.log"
        (venv / "bin" / "pip").write_text(f'#!/bin/sh\necho pip "$*" >> "{pip}"\n')
        (venv / "bin" / "pip").chmod(0o755)
        fakebin = t / "bin"
        fakebin.mkdir(exist_ok=True)
        (fakebin / "curl").write_text('#!/bin/sh\nout=""; while [ $# -gt 0 ]; do [ "$1" = -o ] && out="$2"; shift; done\n'
                                      'printf x > "$out"\n')
        (fakebin / "curl").chmod(0o755)
        # the wheel FILE names as the installer spells them, digests faked for the stubbed curl
        import hashlib
        x = hashlib.sha256(b"x").hexdigest()
        derives = "\n".join(ln for ln in INST.splitlines()
                              if re.match(r"^FLASHINFER_(CUBIN|JIT)_VERSION=", ln))
        self.assertTrue(derives, "install-image-h3.sh no longer derives the versions from the file names")
        assigns = (
            'FLASHINFER_CUBIN_FILE="flashinfer_cubin-0.7.2-py3-none-any.whl"\n'
            f'FLASHINFER_CUBIN_SHA256="{x}"\n'
            'FLASHINFER_JIT_FILE="flashinfer_jit_cache-0.7.2+cu130-cp39-abi3-manylinux_2_28_aarch64.whl"\n'
            f'FLASHINFER_JIT_SHA256="{x}"\nFLASHINFER_RELEASE="http://127.0.0.1:1"\n' + derives + "\n")
        fn_start = INST.index("kernel_version(){")
        fn_end = INST.index('|| true; }', fn_start) + len('|| true; }')
        fn = INST[fn_start:fn_end] + "\n"
        start = INST.index('if [ "$(kernel_version flashinfer-cubin)"')
        end = INST.index('\nfi\n', INST.index('rm -rf "$WHEELS"\n  touch "$RUNTIME_NOTE"', start)) + len("\nfi\n")
        block = fn + INST[start:end]
        script = ("set -uo pipefail\ndie(){ echo \"DIE: $*\" >&2; exit 1; }\n"
                  f'VENV="{venv}"; RUNTIME_NOTE="{t}/note"; {assigns}\n' + block + 'echo DONE\n')
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30,
                           env={"PATH": f"{fakebin}:/usr/bin:/bin", "HOME": str(t)})
        return r.returncode, r.stdout + r.stderr, pip.exists(), (t / "note").exists()

    def setUp(self):
        import tempfile
        self.subdir = tempfile.mkdtemp(prefix="h3-gate-")
        self.addCleanup(__import__("shutil").rmtree, self.subdir, True)

    def test_a_complete_runtime_is_left_alone(self):
        rc, out, pip_called, note = self.gate({"flash-attn-4": "2.8.3",
                                               "flashinfer-cubin": "0.7.2",
                                               "flashinfer-jit-cache": "0.7.2+cu130"})
        self.assertEqual(rc, 0, out)
        self.assertIn("DONE", out)
        self.assertNotIn("installing FlashInfer", out, out)
        self.assertFalse(pip_called, "the 1.6 GB came down again")
        self.assertFalse(note, "the note must not move: it feeds the cockpit's older-files warning")

    def test_a_missing_package_still_installs(self):
        rc, out, pip_called, _ = self.gate({"flash-attn-4": "2.8.3", "flashinfer-cubin": "",
                                            "flashinfer-jit-cache": ""})
        self.assertEqual(rc, 0, out)
        self.assertIn("installing FlashInfer", out)


class TheNowViewNamesTheServingImagePort(unittest.TestCase):
    """The Now view must not print Qwen-Image's port while MiniMax-H3-Image serves: it asks the
    image route, which answers the port of the lane that serves (found in review of v1.50)."""

    def test_no_default_port_for_an_image_lane(self):
        js = (REPO / "dashboard/static/js/now.js").read_text()
        self.assertNotIn("IMG_NOW.port || 30020", js)
        self.assertIn("const IMG_NOW = {label: '', port: null, unit: null};", js)
        # each lane's own default until the route answers, never the other's (its behaviour: test_page_details)
        self.assertIn("unit === IMAGE_H3_UNIT ? 30026 : 30020", js)
        self.assertIn("fetch('/api/image')", js)


class TheEncoderRuleIsReadOnce(unittest.TestCase):
    """install-image-h3.sh and switch-model.sh asked the same question about MiniMax-H3's text
    encoder with two copies of one python heredoc: a rename of a partition folder by MiniMax fixes
    one copy and lets the other re-download 63 GiB (found in review of v1.50; the lanes.sh lesson
    of 2026-10-16 says the same). One reading, in lanes.sh."""

    def lanes_sh(self):
        return (REPO / "lanes.sh").read_text()

    def test_both_callers_use_it(self):
        lanes = self.lanes_sh()
        self.assertIn("h3_encoder_cached(){", lanes)
        inst = (REPO / "install-image-h3.sh").read_text()
        self.assertIn('encoder_cached(){ h3_encoder_cached "$HF_CACHE', inst)
        sw = (REPO / "switch-model.sh").read_text()
        self.assertIn('if ! h3_encoder_cached "$H3_ENC_SNAP"; then', sw)
        self.assertNotIn('FL2VA\" / \"text_encoder', sw, "the switch keeps no copy of the rule")

    def test_the_reading_itself(self):
        import tempfile, subprocess
        t = pathlib.Path(tempfile.mkdtemp(prefix="h3-enc-"))
        self.addCleanup(__import__("shutil").rmtree, t, True)
        snap = t / "snap"
        def whole(d, names):
            d.mkdir(parents=True, exist_ok=True)
            import json
            (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {f"t{i}": n for i, n in enumerate(names)}}))
            for n in names:
                (d / n).write_text("x")
        def ask():
            r = subprocess.run(["bash", "-c", f'. "{REPO}/lanes.sh"; h3_encoder_cached "{snap}"; echo rc=$?'],
                               capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "HOME": str(t)})
            return r.stdout.strip().splitlines()[-1]
        self.assertEqual(ask(), "rc=1", "an empty snapshot caches nothing")
        whole(snap / "FL2VA" / "text_encoder", ["e1.safetensors", "e2.safetensors"])
        self.assertEqual(ask(), "rc=0", "a partition folder holds the same blobs")
        (snap / "FL2VA" / "text_encoder" / "e2.safetensors").unlink()
        self.assertEqual(ask(), "rc=1", "a missing shard is not the encoder")


class TheSourcesDependencies(unittest.TestCase):
    """SGLang's main at the pin is laid over the released wheel with no dependency resolution: a package it
    added (blake3, 2026-11-16) was missing, and the runtime failed at the model's first import with a
    message that named a wrong pin (found by the lane's first proof on the reference box). The installer
    pins what main adds, and refuses, naming it, any dependency of the pinned source the venv lacks."""

    GUARD_HEAD = '"$VENV/bin/python" - "$SRC/python/pyproject.toml" <<\'PY\''

    def guard(self):
        text = (REPO / "install-image-h3.sh").read_text()
        start = text.index(self.GUARD_HEAD)
        body = text[text.index("\n", start) + 1:]
        return body[:body.index("\nPY\n")]

    def run_guard(self, deps, diffusion=()):
        import subprocess
        import tempfile
        d = pathlib.Path(tempfile.mkdtemp(prefix="h3-deps-"))
        self.addCleanup(__import__("shutil").rmtree, d, True)
        quote = lambda xs: "[" + ", ".join(f"'{x}'" for x in xs) + "]"   # TOML literal strings: a marker holds quotes
        (d / "pyproject.toml").write_text(f"[project]\nname = \"sglang\"\ndependencies = {quote(deps)}\n\n"
                                          f"[project.optional-dependencies]\ndiffusion = {quote(diffusion)}\n")
        return subprocess.run([sys.executable, "-", str(d / "pyproject.toml")], input=self.guard(),
                              capture_output=True, text=True)

    def test_what_the_venv_holds_passes(self):
        r = self.run_guard(["packaging>=20"], diffusion=["sglang[diffusion]"])
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_a_package_the_venv_lacks_is_named(self):
        r = self.run_guard(["packaging"], diffusion=["no-such-package-for-alzeo>=1.0"])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no-such-package-for-alzeo>=1.0", r.stderr)

    def test_a_dependency_of_another_platform_is_left_out(self):
        r = self.run_guard(['no-such-package-for-alzeo; sys_platform == "win32"'])
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_main_s_own_additions_are_pinned_and_installed(self):
        text = (REPO / "install-image-h3.sh").read_text()
        self.assertIn('MAIN_DEPS=("blake3==1.0.11")', text)
        self.assertLess(text.index('for _d in "${MAIN_DEPS[@]}"'), text.index(self.GUARD_HEAD),
                        "the pinned additions go in before the guard reads the venv")


if __name__ == "__main__":
    unittest.main()
