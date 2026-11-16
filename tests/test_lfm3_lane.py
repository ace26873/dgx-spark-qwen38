#!/usr/bin/env python3
"""The Liquid AI LFM3 lane (install-lfm3.sh): the unit its --stage writes for a switch, the room its fetch asks, what its
--uninstall removes and keeps, a switch to it run as written, and its clients' pair, provider and levels (the proxy's
part is test_proxy_lfm3.py's).

--stage runs the installer as written, in a throwaway home, with no sudo and no docker. The uninstall block runs whole
against a fake unit folder and a fake config directory, sudo, systemctl and docker stubs writing down every call, as in
test_ornith_122b_lane.py."""
import importlib.util
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
TEXT = (REPO / "install-lfm3.sh").read_text()
PINS = {k: v.strip('"') for k, v in (ln.split("=", 1) for ln in TEXT.splitlines()
                                       if ln.startswith("LFM3_") and "=" in ln)}
SERVED = "lfm3-1.7b"


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def exec_command(unit: str) -> str:
    """The ExecStart command as bash runs it: systemd joins the lines a backslash ends."""
    line = unit[unit.index("ExecStart=/bin/bash -c '") + len("ExecStart=/bin/bash -c '"):]
    return line[:line.index("'\nExecStop=")].replace("\\\n", " ")


class TheStagedUnit(unittest.TestCase):
    def setUp(self):
        self.t = pathlib.Path(tempfile.mkdtemp(prefix="lfm3-stage-"))
        self.addCleanup(shutil.rmtree, self.t, True)
        self.home = self.t / "home"
        self.cfg = self.home / ".config" / "qwen38"
        self.cfg.mkdir(parents=True)
        (self.cfg / "api-key").write_text("test-key\n")
        self.units = self.t / "units"
        self.units.mkdir()
        self.hf = self.t / "hf"
        self.out = self.t / "stage.service"

    def stage(self, **env):
        e = {"PATH": "/usr/bin:/bin", "HOME": str(self.home), "HF_CACHE": str(self.hf), "LANES_UNIT_DIR": str(self.units),
             "QWEN38_MEMORY_CLASS": "128", **env}
        r = subprocess.run(["bash", str(REPO / "install-lfm3.sh"), "--stage", str(self.out)],
                           capture_output=True, text=True, env=e, timeout=120)
        return r.returncode, r.stdout + r.stderr

    def test_the_unit_serves_the_pinned_checkpoint_on_the_pinned_nightly(self):
        rc, out = self.stage()
        self.assertEqual(rc, 0, out)
        unit = self.out.read_text()
        self.assertNotRegex(unit, r"__[A-Z][A-Z0-9_]*__", "a placeholder left")
        cmd = re.sub(r"\s+", " ", exec_command(unit))
        self.assertEqual(PINS["LFM3_IMAGE"],
                         "vllm/vllm-openai@sha256:19cfbac06cfed28a2eb2c08950e62b4b07fd6e20685ad6e10f9422236d39e519")
        self.assertIn(f" {PINS['LFM3_IMAGE']} {PINS['LFM3_REPO']} --revision {PINS['LFM3_REV']} ", cmd)
        self.assertIn(f"-v {self.hf}:/root/.cache/huggingface ", cmd)
        for d in ("vllm", "flashinfer"):
            self.assertIn(f"-v {self.home}/.config/qwen38/lfm3-cache/{d}:/root/.cache/{d} ", cmd)
            self.assertTrue((self.cfg / "lfm3-cache" / d).is_dir(), "made as this user, not by docker as root")
        self.assertIn("--host 127.0.0.1 --port 30000 ", cmd)

    def test_it_serves_the_measured_command_with_its_share_of_the_memory(self):
        """The study's command (2026-11-16) and a share: without one, this nightly's KV cache grew into all the memory
        left, the box at 1,087 MiB available at its lowest."""
        self.assertEqual(PINS["LFM3_GMU"], "0.75")
        self.assertEqual(self.stage()[0], 0)
        cmd = re.sub(r"\s+", " ", exec_command(self.out.read_text()))
        for flag in ("--served-model-name lfm3-1.7b", "--max-model-len 262144", "--gpu-memory-utilization 0.75",
                     "--reasoning-parser qwen3", "--tool-call-parser qwen3_coder", "--enable-auto-tool-choice",
                     "--enable-prompt-tokens-details"):
            self.assertIn(f" {flag} ", f" {cmd} ", flag)
        self.assertNotIn("--chat-template", cmd, "the checkpoint's own template, as published")
        self.assertEqual(self.stage(LFM3_GMU="0.5")[0], 0)
        self.assertIn("--gpu-memory-utilization 0.75 ", self.out.read_text(), "a pin, not an environment variable")

    def test_the_key_is_on_no_command_line_and_every_route_but_two_asks_it(self):
        self.assertEqual(self.stage()[0], 0)
        unit = self.out.read_text()
        cmd = exec_command(unit)
        self.assertNotIn("--api-key", cmd)
        self.assertNotIn("test-key", unit)
        self.assertIn("--config /out/engine-secrets.yaml --middleware vllm_engine_auth.RequireKey ", cmd)
        cfg = f"{self.home}/.config/qwen38"
        self.assertIn(f"--mount type=bind,source={cfg}/engine-secrets.yaml,target=/out/engine-secrets.yaml,readonly ", cmd)
        self.assertIn(f"--mount type=bind,source={cfg}/api-key,target=/out/api-key,readonly ", cmd)
        self.assertIn(f"-v {cfg}/vllm-auth:/opt/qwen38-auth:ro ", cmd)
        self.assertNotIn(f"-v {cfg}:/out", cmd, "the config directory's other files stay out of the engine")
        self.assertIn(f"ExecStartPre=/bin/bash {cfg}/engine-secrets.sh\n", unit)
        self.assertIn(f"ExecStartPre=/bin/bash {cfg}/engine-preflight.sh qwen38-lfm3.service\n", unit)
        for env in ("HF_HUB_OFFLINE=1", "VLLM_NO_USAGE_STATS=1", "DO_NOT_TRACK=1", "PYTHONPATH=/opt/qwen38-auth",
                    "QWEN38_KEY_FILE=/out/api-key"):
            self.assertIn(f"-e {env} ", cmd, env)

    def test_a_stop_while_it_loads_is_a_stop_not_a_failure(self):
        self.assertEqual(self.stage()[0], 0)
        unit = self.out.read_text()
        self.assertIn("|| { rc=$$?; [ -e \"$$RUNTIME_DIRECTORY/stopping\" ] && exit 143; exit $$rc; }'", unit)
        self.assertIn("ExecStop=-/bin/bash -c 'touch \"$$RUNTIME_DIRECTORY/stopping\"'\n", unit)
        self.assertIn("RuntimeDirectory=qwen38-lfm3\n", unit)
        self.assertIn("SuccessExitStatus=137 143\n", unit)

    def test_its_engine_is_the_one_engine_on_the_box(self):
        self.assertEqual(self.stage()[0], 0)
        head = self.out.read_text().split("[Service]")[0]
        conflicts = set(re.search(r"^Conflicts=(.*)$", head, re.M).group(1).split())
        after = set(re.search(r"^After=(.*)$", head, re.M).group(1).split())
        others = {t.name[:-len(".template")] for t in REPO.glob("qwen38-*.service.template")
                  if re.search(r"^Conflicts=", t.read_text(), re.M)} - {"qwen38-lfm3.service", "qwen38-sglang-1m.service",
                                                                         "qwen38-sglang-64.service"}
        self.assertEqual(conflicts, others)
        self.assertLessEqual(conflicts, after)


class TheRoomItTakes(unittest.TestCase):
    def test_the_checkpoint_rounded_up(self):
        m = re.search(r"^# The repository at this revision, ([\d,]+) bytes .*\n(?:#.*\n)*LFM3_GB=\"(\d+)\"$", TEXT, re.M)
        self.assertIsNotNone(m, "install-lfm3.sh: no measure of the repository above LFM3_GB")
        self.assertEqual(int(m.group(2)), math.ceil(int(m.group(1).replace(",", "")) / 1e9))
        js = (REPO / "dashboard" / "static" / "js" / "base.js").read_text()
        self.assertRegex(js, rf"(?m)^  lfm3: '[^']*The first switch downloads about {m.group(2)} GB\.'(?:}};|,)$",
                         "base.js's note says another size")

    def test_the_image_at_its_peak_as_install_flash_local_counts_it(self):
        m = re.search(r"^# Its pull at its peak: ([\d.]+) GB unpacked and ([\d.]+) GB to download.*\n(?:#.*\n)*IMAGE_GB=(\d+)$",
                      TEXT, re.M)
        self.assertIsNotNone(m, "install-lfm3.sh: no measure of the image above IMAGE_GB")
        gib = (float(m.group(1)) + float(m.group(2))) * 1e9 / 2 ** 30
        self.assertEqual(int(m.group(3)), math.ceil(gib * 1.02) + 1)


class TheUninstall(unittest.TestCase):
    """Its unit, its container, its local tag and its caches go; the image, the checkpoint, the key's yaml and, while
    another vLLM lane is installed, the middleware stay."""

    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp(prefix="lfm3-uninstall-"))
        self.addCleanup(shutil.rmtree, self.d, True)
        self.home = self.d / "home"
        self.cfg = self.home / ".config" / "qwen38"
        (self.cfg / "lfm3-cache" / "vllm").mkdir(parents=True)
        (self.cfg / "vllm-auth").mkdir()
        (self.cfg / "engine-secrets.yaml").write_text('api-key: "k"\n')
        self.sd = self.d / "systemd"
        self.sd.mkdir()
        self.unit = self.sd / "qwen38-lfm3.service"
        self.unit.write_text("ExecStart=/bin/bash -c '/usr/bin/docker run x'\n")
        self.hf = self.d / "hf"
        (self.hf / "hub" / f"models--{PINS['LFM3_REPO'].replace('/', '--')}").mkdir(parents=True)
        self.oc = self.cfg / "opencode.json"
        self.oc.write_text(json.dumps({"model": f"lfm3/{SERVED}", "provider": {
            "qwen38": {"models": {"qwen3.8-27b": {"limit": {"context": 173000, "output": 64000}}}},
            "lfm3": {"models": {SERVED: {"limit": {"context": 205000, "output": 32000}}}}}}))
        self.bin = self.d / "bin"
        self.bin.mkdir()
        self.log = self.d / "calls.log"
        self.tag = f"qwen38-pinned:lfm3-{PINS['LFM3_IMAGE'].split('@sha256:')[1][:12]}"
        (self.bin / "docker").write_text("#!/bin/bash\n" f'echo "docker $*" >> "{self.log}"\n'
                                         f'case "$1" in images) echo "{self.tag} sha256:feed" ;; esac\nexit 0\n')
        (self.bin / "systemctl").write_text(f'#!/bin/bash\necho "systemctl $*" >> "{self.log}"\n'
                                            '[ "$1" = is-enabled ] && exit 1\n[ "$1" = is-active ] && exit 1\nexit 0\n')
        (self.bin / "sudo").write_text(f'#!/bin/bash\necho "sudo $*" >> "{self.log}"\nexec "$@"\n')
        for tool in ("docker", "systemctl", "sudo"):
            (self.bin / tool).chmod(0o755)

    def run_block(self):
        start = TEXT.index('if [ "$ACTION" = uninstall ]; then')
        block = TEXT[start:TEXT.index("\n  exit 0\nfi\n", start) + len("\n  exit 0\nfi\n")]
        pins = [ln for ln in TEXT.splitlines() if ln.startswith(("LFM3_", "LOCAL_IMAGE=", "VLANE="))]
        script = "\n".join(["set -euo pipefail", "step(){ :; }", "die(){ echo \"ERROR: $*\" >&2; exit 1; }",
                            f'LANES_UNIT_DIR="{self.sd}"', f'HERE="{REPO}"', f'. "{REPO}/lanes.sh"',
                            f'. "{REPO}/vllm-lane.sh"', *pins, "ACTION=uninstall", "UNIT=qwen38-lfm3.service",
                            f'INSTALLED="{self.unit}"', "CONTAINER=qwen38-lfm3", f'CONFIG_DIR="{self.cfg}"',
                            f'HF_CACHE="{self.hf}"', block])
        env = dict(os.environ, PATH=f"{self.bin}:/usr/bin:/bin", HOME=str(self.home))
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r.stdout, self.log.read_text() if self.log.exists() else ""

    def test_its_own_parts_go_and_the_image_and_the_checkpoint_stay(self):
        out, calls = self.run_block()
        self.assertFalse(self.unit.exists())
        self.assertIn("docker rm -f qwen38-lfm3", calls)
        self.assertIn(f"docker rmi {self.tag}", calls)
        self.assertNotIn(f"rmi {PINS['LFM3_IMAGE']}", calls)
        self.assertFalse((self.cfg / "lfm3-cache").exists())
        self.assertTrue((self.cfg / "engine-secrets.yaml").exists())
        d = self.hf / "hub" / f"models--{PINS['LFM3_REPO'].replace('/', '--')}"
        self.assertIn(f"checkpoint kept, delete it yourself for the space: rm -rf {d}", out)

    def test_the_vllm_lanes_middleware_goes_with_the_last_of_them(self):
        self.run_block()
        self.assertFalse((self.cfg / "vllm-auth").exists(), "no other vLLM lane is installed")

    def test_the_middleware_stays_while_another_vllm_lane_reads_it(self):
        (self.sd / "qwen38-kolibri.service").write_text("ExecStart=/bin/true\n")
        out, _ = self.run_block()
        self.assertTrue((self.cfg / "vllm-auth").is_dir())
        self.assertIn("vllm-auth kept: another vLLM lane uses it too", out)

    def test_another_vllm_lanes_uninstall_keeps_it_while_this_lane_reads_it(self):
        """vllm-lane.sh's list of the units that read the middleware names this lane's: without it, an uninstall of the
        Ornith lane took the middleware from a box whose LFM3 lane still started with it."""
        self.unit.write_text("ExecStart=/bin/true\n")
        script = "\n".join(["set -euo pipefail", "step(){ :; }", f'LANES_UNIT_DIR="{self.sd}"', f'HERE="{REPO}"',
                            f'. "{REPO}/lanes.sh"', f'. "{REPO}/vllm-lane.sh"', "VLANE=ornith-122b",
                            "UNIT=qwen38-ornith-122b.service", "CONTAINER=qwen38-ornith-122b", f'CONFIG_DIR="{self.cfg}"',
                            f'HF_CACHE="{self.hf}"', 'vlane_uninstall_runtime qwen38-ornith-122b "x@sha256:0" org/none'])
        env = dict(os.environ, PATH=f"{self.bin}:/usr/bin:/bin", HOME=str(self.home))
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue((self.cfg / "vllm-auth").is_dir())

    def test_its_opencode_provider_goes_and_the_default_comes_back_to_the_27b(self):
        self.run_block()
        cfg = json.loads(self.oc.read_text())
        self.assertNotIn("lfm3", cfg["provider"])
        self.assertEqual(cfg["model"], "qwen38/qwen3.8-27b")


class TheSwitch(unittest.TestCase):
    """./switch-model.sh lfm3, run as written on a copy whose unit paths point at this test's folder."""

    def setUp(self):
        self.t = pathlib.Path(tempfile.mkdtemp(prefix="lfm3-switch-"))
        self.addCleanup(shutil.rmtree, self.t, True)
        self.home = self.t / "home"
        self.cfg = self.home / ".config" / "qwen38"
        self.cfg.mkdir(parents=True)
        (self.cfg / "api-key").write_text("test-key\n")
        self.units = self.t / "units"
        self.units.mkdir()
        repo = self.t / "repo"
        repo.mkdir()
        text = (REPO / "switch-model.sh").read_text()
        for unit in ("qwen38-sglang.service", "qwen38-flash.service", "qwen38-lfm3.service"):
            text = text.replace(f'"/etc/systemd/system/{unit}"', f'"{self.units}/{unit}"')
        (repo / "switch-model.sh").write_text(text)
        (repo / "switch-model.sh").chmod(0o755)
        for f in ("install.sh", "lanes.sh", "boxmem.py", "ckpt-fetch.py", "install-lfm3.sh", "vllm-lane.sh",
                  "qwen38-lfm3.service.template", "oc-limits.sh", "oc-lfm3.py", "oc-merge-limits.py",
                  "oc-point-default.py", "pi-gen.py"):
            shutil.copy(REPO / f, repo / f)
        self.hf = self.t / "hf"
        self.unit = self.units / "qwen38-lfm3.service"
        self.unit.write_text("[Service]\nExecStart=/bin/bash -c '/usr/bin/docker run --rm --name qwen38-lfm3 \\\n"
                             f"  -v {self.hf}:/root/.cache/huggingface \\\n  {PINS['LFM3_IMAGE']} \\\n"
                             f"    {PINS['LFM3_REPO']} --revision old'\n")
        (self.units / "qwen38-sglang.service").write_text("[Service]\n")
        self.oc = self.cfg / "opencode.json"
        self.oc.write_text(json.dumps({"model": "qwen38/qwen3.8-27b", "provider": {"qwen38": {
            "options": {"baseURL": "http://127.0.0.1:30001/v1"},
            "models": {"qwen3.8-27b": {"limit": {"context": 700000, "output": 200000}}}}}}))
        self.bin = self.t / "bin"
        self.bin.mkdir()
        self.log = self.t / "calls.log"
        stub = {"docker": f'echo "docker $*" >> "{self.log}"\nexit 0',
                "systemctl": f'echo "systemctl $*" >> "{self.log}"\n'
                             'case "$*" in "is-enabled --quiet qwen38-sglang.service") exit 0 ;; esac\n'
                             'case "$1" in is-enabled|is-active|list-unit-files) exit 1 ;; esac\nexit 0',
                "sudo": f'echo "sudo $*" >> "{self.log}"\nexec "$@"',
                "df": "echo Avail; echo 500G"}
        for name, body in stub.items():
            (self.bin / name).write_text("#!/bin/bash\n" + body + "\n")
            (self.bin / name).chmod(0o755)
        self.script = repo / "switch-model.sh"

    def switch(self, *args):
        e = {"PATH": f"{self.bin}:/usr/bin:/bin", "HOME": str(self.home), "QWEN38_MEMORY_CLASS": "128"}
        r = subprocess.run([str(self.script), "lfm3", *args], capture_output=True, text=True, env=e, timeout=120)
        return r.returncode, r.stdout + r.stderr, self.log.read_text() if self.log.exists() else ""

    def test_it_puts_in_its_unit_moves_the_boot_and_points_opencode_at_it(self):
        rc, out, calls = self.switch()
        self.assertEqual(rc, 0, out)
        unit = self.unit.read_text()
        self.assertIn(f" {PINS['LFM3_REPO']} --revision {PINS['LFM3_REV']} ", unit.replace("\\\n", " "))
        self.assertIn(f"-v {self.hf}:/root/.cache/huggingface ", unit, "the cache its unit mounted")
        fetches = [c for c in calls.splitlines() if c.startswith("docker run") and "/opt/qwen38/ckpt-fetch.py" in c]
        self.assertEqual(len(fetches), 1, calls)
        self.assertIn(f" {PINS['LFM3_IMAGE']} ", fetches[0])
        self.assertIn("sudo systemctl disable qwen38-sglang.service", calls)
        self.assertIn("sudo systemctl enable qwen38-lfm3.service", calls)
        self.assertFalse(re.search(r"systemctl (start|stop|restart) qwen38-", calls), calls)
        self.assertIn("Effective after:  sudo systemctl start qwen38-lfm3.service", out)
        cfg = json.loads(self.oc.read_text())
        self.assertEqual(cfg["model"], f"lfm3/{SERVED}")
        self.assertEqual(cfg["provider"]["lfm3"]["models"][SERVED]["limit"], {"context": 205000, "input": 205000, "output": 32000})

    def test_with_no_unit_installed_it_says_how_to_install_it_and_changes_nothing(self):
        self.unit.unlink()
        rc, out, calls = self.switch()
        self.assertNotEqual(rc, 0)
        self.assertIn("./install.sh --with-lfm3", out)
        self.assertNotIn("docker run", calls)
        self.assertNotIn("sudo", calls)


class ItsClients(unittest.TestCase):
    def test_its_pair_fits_its_window_by_both_rules(self):
        r = subprocess.run(["bash", str(REPO / "oc-limits.sh"), "lfm3"], capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "205000 32000 local, 262K", r.stderr)
        ctx, out = 205000, 32000
        self.assertLessEqual(ctx - 20000 + 43863 + out, 262144)
        self.assertLessEqual(ctx + 25000 + out, 262144)

    def test_opencode_offers_the_two_states_its_template_holds(self):
        r = subprocess.run(["python3", str(REPO / "oc-lfm3.py"), "http://127.0.0.1:30001/v1", "/k", "205000", "32000", "n"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        block = json.loads(r.stdout)["provider"]["lfm3"]
        model = block["models"][SERVED]
        self.assertEqual(model["variants"], {"high": {"chat_template_kwargs": {"enable_thinking": True}},
                                             "none": {"chat_template_kwargs": {"enable_thinking": False}}})
        self.assertNotIn("reasoning", model, "opencode 1.19.5 would add low, medium and high beside them")
        self.assertNotIn("modalities", model, "it reads no image")
        self.assertEqual(block["options"]["apiKey"], "{file:/k}")

    def test_its_template_reads_that_switch_and_no_level(self):
        """The two variants are what the checkpoint's chat template reads (at the pinned revision, when it is in this
        machine's cache): enable_thinking, true by default, and no reasoning_effort."""
        snap = pathlib.Path(os.environ.get("HF_HOME", pathlib.Path.home() / ".cache/huggingface")) / "hub" / \
            f"models--{PINS['LFM3_REPO'].replace('/', '--')}" / "snapshots" / PINS["LFM3_REV"] / "chat_template.jinja"
        if not snap.is_file():
            self.skipTest("the checkpoint is not in this machine's cache")
        tpl = snap.read_text()
        self.assertIn("{%- set enable_thinking = enable_thinking | default(true) -%}", tpl)
        self.assertNotIn("reasoning_effort", tpl)

    def test_its_name_in_the_picker(self):
        r = subprocess.run(["python3", str(REPO / "oc-point-default.py"), "--label", "lfm3", "262144"],
                           capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "Liquid AI LFM3 1.7B NVFP4 (local, 262K)")

    def test_pi_offers_off_and_high_alone_and_sends_them(self):
        prov = json.loads(subprocess.run(["python3", str(REPO / "oc-lfm3.py"), "http://127.0.0.1:30001/v1", "/k",
                                          "205000", "32000", "n"], capture_output=True, text=True).stdout)
        art = {"model": f"lfm3/{SERVED}", **prov}
        pi = load("pi_gen_lfm3", "pi-gen.py")
        models, settings, efforts = pi.build(art)
        entry = models["providers"]["lfm3"]["models"][0]
        tlm = entry["thinkingLevelMap"]
        # pi 1.2.0's getSupportedThinkingLevels, ported: a level mapped to null is hidden, xhigh and max need a mapping
        shown = [lvl for lvl in pi.PI_LEVELS if not (lvl in tlm and tlm[lvl] is None)
                 and (lvl not in ("xhigh", "max") or lvl in tlm)]
        self.assertEqual(shown, ["off", "high"])
        self.assertEqual(tlm["off"], "none", "sent as none, which the proxy turns into enable_thinking false")
        self.assertTrue(entry["compat"]["supportsReasoningEffort"])
        self.assertEqual(settings["modelThinkingLevels"][f"lfm3/{SERVED}"], "high")
        omp, _ = pi.omp_convert(models, settings, efforts)
        thinking = omp["providers"]["lfm3"]["models"][0]["thinking"]
        self.assertEqual((thinking["efforts"], thinking["defaultLevel"]), (["high", "minimal"], "high"))


if __name__ == "__main__":
    unittest.main()
