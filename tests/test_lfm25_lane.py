#!/usr/bin/env python3
"""The Liquid AI LFM2.5 24B-A2B lane (install-lfm25.sh): the unit its --stage writes for a switch, the room its fetch
asks, what its --uninstall removes and keeps, and its clients' pair and provider (the proxy's part is
test_proxy_lfm25.py's).

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
TEXT = (REPO / "install-lfm25.sh").read_text()
PINS = {k: v.strip('"') for k, v in (ln.split("=", 1) for ln in TEXT.splitlines()
                                       if ln.startswith("LFM25_") and "=" in ln)}
INSTALL_IMAGE = re.search(r'^IMAGE="\$\{IMAGE:-([^}]+)\}"', (REPO / "install.sh").read_text(), re.M).group(1)
SERVED = "lfm2.5-24b-a2b"


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
        self.t = pathlib.Path(tempfile.mkdtemp(prefix="lfm25-stage-"))
        self.home = self.t / "home"
        self.cfg = self.home / ".config" / "qwen38"
        self.cfg.mkdir(parents=True)
        (self.cfg / "api-key").write_text("test-key\n")
        self.units = self.t / "units"
        self.units.mkdir()
        self.hf = self.t / "hf"
        self.out = self.t / "stage.service"

    def tearDown(self):
        shutil.rmtree(self.t, ignore_errors=True)

    def stage(self, **env):
        e = {"PATH": "/usr/bin:/bin", "HOME": str(self.home), "HF_CACHE": str(self.hf), "LANES_UNIT_DIR": str(self.units),
             "QWEN38_MEMORY_CLASS": "128", **env}
        r = subprocess.run(["bash", str(REPO / "install-lfm25.sh"), "--stage", str(self.out)],
                           capture_output=True, text=True, env=e, timeout=120)
        return r.returncode, r.stdout + r.stderr

    def test_the_unit_is_whole_and_serves_the_pinned_pair_on_the_27b_lanes_image(self):
        rc, out = self.stage()
        self.assertEqual(rc, 0, out)
        unit = self.out.read_text()
        self.assertNotRegex(unit, r"__[A-Z][A-Z0-9_]*__", "a placeholder left")
        cmd = exec_command(unit)
        self.assertEqual(PINS["LFM25_IMAGE"], INSTALL_IMAGE, "the 27B lane's image, by its digest")
        self.assertIn(f" {INSTALL_IMAGE} ", cmd)
        self.assertIn(f"--model-path {PINS['LFM25_REPO']} --revision {PINS['LFM25_REV']} ", cmd)
        self.assertIn(f"--speculative-draft-model-path {PINS['LFM25_DRAFT_REPO']} ", cmd)
        self.assertIn(f"--speculative-draft-model-revision {PINS['LFM25_DRAFT_REV']} ", cmd)
        self.assertIn(f"--served-model-name {SERVED} ", cmd)
        self.assertIn(f"-v {self.hf}:/root/.cache/huggingface ", cmd)
        self.assertIn(f"-v {self.home}/.config/qwen38/lfm25-cache:/cache ", cmd)
        self.assertTrue((self.cfg / "lfm25-cache").is_dir(), "made as this user, not by docker as root")
        self.assertIn("--host 127.0.0.1 --port 30000", cmd)

    def test_it_serves_the_measured_command(self):
        """The draft's card's command, as the reference box ran it on 2026-11-14: the model's window, a static share of
        0.7, the DSpark draft by blocks of 9 with 10 tokens to verify, and the 48 requests SGLang set itself then."""
        self.assertEqual(self.stage()[0], 0)
        cmd = re.sub(r"\s+", " ", exec_command(self.out.read_text()))
        for flag in ("--context-length 131072", "--mem-fraction-static 0.7", "--max-running-requests 48",
                     "--speculative-algorithm DSPARK", "--speculative-dspark-block-size 9",
                     "--speculative-num-draft-tokens 10", "--tp-size 1"):
            self.assertIn(f" {flag} ", f" {cmd} ", flag)

    def test_its_parsers_are_the_model_cards_and_the_installers_one_line(self):
        self.assertEqual(PINS["LFM25_REASONING_PARSER"], "deepseek-r1")
        self.assertEqual(self.stage()[0], 0)
        cmd = exec_command(self.out.read_text())
        self.assertIn("--reasoning-parser deepseek-r1 --tool-call-parser lfm2 ", cmd)
        self.assertNotIn("--chat-template", cmd, "the checkpoint's own template, as published")

    def test_the_key_is_on_no_command_line_and_every_belt_of_an_sglang_lane_is_there(self):
        self.assertEqual(self.stage()[0], 0)
        unit = self.out.read_text()
        cmd = exec_command(unit)
        self.assertNotIn("--api-key", cmd)
        self.assertNotIn("test-key", unit)
        self.assertIn("--config /out/engine-secrets.yaml ", cmd)
        self.assertIn(f"--mount type=bind,source={self.home}/.config/qwen38/engine-secrets.yaml,"
                      "target=/out/engine-secrets.yaml,readonly ", cmd)
        self.assertIn(f"ExecStartPre=/bin/bash {self.home}/.config/qwen38/engine-secrets.sh\n", unit)
        self.assertIn(f"ExecStartPre=/bin/bash {self.home}/.config/qwen38/engine-preflight.sh qwen38-lfm25.service\n", unit)
        for env in ("HF_HUB_OFFLINE=1", "SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES=1", "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0",
                    "SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE=file:///tmp/sglang-rdv"):
            self.assertIn(f"-e {env} ", cmd, env)
        for flag in ("--sleep-on-idle", "--enable-metrics", "--enable-cache-report"):
            self.assertIn(f"{flag} ", cmd, flag)

    def test_its_engine_is_the_one_engine_on_the_box(self):
        self.assertEqual(self.stage()[0], 0)
        head = self.out.read_text().split("[Service]")[0]
        conflicts = set(re.search(r"^Conflicts=(.*)$", head, re.M).group(1).split())
        after = set(re.search(r"^After=(.*)$", head, re.M).group(1).split())
        others = {f"{t.name[:-len('.template')]}" for t in REPO.glob("qwen38-*.service.template")
                  if re.search(r"^Conflicts=", t.read_text(), re.M)} - {"qwen38-lfm25.service", "qwen38-sglang-1m.service",
                                                                         "qwen38-sglang-64.service"}
        self.assertEqual(conflicts, others)
        self.assertLessEqual(conflicts, after)

    def test_with_no_file_to_write_to_it_refuses(self):
        e = {"PATH": "/usr/bin:/bin", "HOME": str(self.home), "LANES_UNIT_DIR": str(self.units)}
        r = subprocess.run(["bash", str(REPO / "install-lfm25.sh"), "--stage"], capture_output=True, text=True, env=e, timeout=60)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--stage takes the file to write the unit to", r.stderr)


class TheRoomItsFetchAsks(unittest.TestCase):
    def test_the_two_repositories_sizes_rounded_up(self):
        """LFM25_GB is what the fetch asks room for and what the help and the cockpit say it downloads: the Hub's file
        sizes of the two repositories at their revisions, written above it, summed and rounded up to the GB."""
        m = re.search(r"^# The two repositories at these revisions, ([\d,]+) and ([\d,]+) bytes .*\n(?:#.*\n)*LFM25_GB=\"(\d+)\"$",
                      TEXT, re.M)
        self.assertIsNotNone(m, "install-lfm25.sh: no measure of the two repositories above LFM25_GB")
        total = int(m.group(1).replace(",", "")) + int(m.group(2).replace(",", ""))
        self.assertEqual(int(m.group(3)), math.ceil(total / 1e9))
        js = (REPO / "dashboard" / "static" / "js" / "base.js").read_text()
        self.assertIsNotNone(re.search(rf"^  lfm25: '[^']*The first switch downloads about {m.group(3)} GB\.'}};$", js, re.M),
                             "base.js's note says another size")


class TheUninstall(unittest.TestCase):
    """Its unit, its container, its local tag on the 27B lane's image, its compile cache and its opencode provider go;
    the image, the key's yaml (the 27B lane's unit reads it too), the vLLM lanes' middleware and the checkpoints stay."""

    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp(prefix="lfm25-uninstall-"))
        self.home = self.d / "home"
        self.cfg = self.home / ".config" / "qwen38"
        (self.cfg / "lfm25-cache" / "inductor").mkdir(parents=True)
        (self.cfg / "vllm-auth").mkdir()
        (self.cfg / "engine-secrets.yaml").write_text('api-key: "k"\n')
        self.sd = self.d / "systemd"
        self.sd.mkdir()
        self.unit = self.sd / "qwen38-lfm25.service"
        self.unit.write_text("ExecStart=/bin/bash -c 'exec /usr/bin/docker run x'\n")
        self.hf = self.d / "hf"
        for repo in (PINS["LFM25_REPO"], PINS["LFM25_DRAFT_REPO"]):
            (self.hf / "hub" / f"models--{repo.replace('/', '--')}").mkdir(parents=True)
        self.oc = self.cfg / "opencode.json"
        self.oc.write_text(json.dumps({"model": f"lfm25/{SERVED}", "provider": {
            "qwen38": {"models": {"qwen3.8-27b": {"limit": {"context": 173000, "output": 64000}}}},
            "lfm25": {"models": {SERVED: {"limit": {"context": 74000, "output": 32000}}}}}}))
        self.bin = self.d / "bin"
        self.bin.mkdir()
        self.log = self.d / "calls.log"
        tag = f"qwen38-pinned:lfm25-{PINS['LFM25_IMAGE'].split('@sha256:')[1][:12]}"
        self.tag = tag
        docker = ("#!/bin/bash\n"
                  f'echo "docker $*" >> "{self.log}"\n'
                  f'case "$1" in images) echo "{tag} sha256:feed" ;;\n'
                  '  image) echo "qwen38-pinned:27b-' + PINS["LFM25_IMAGE"].split("@sha256:")[1][:12] + '" ;;\nesac\nexit 0\n')
        (self.bin / "docker").write_text(docker)
        (self.bin / "systemctl").write_text(f'#!/bin/bash\necho "systemctl $*" >> "{self.log}"\n'
                                            '[ "$1" = is-enabled ] && exit 1\n[ "$1" = is-active ] && exit 1\nexit 0\n')
        # sudo runs what it is given, the stubs above on PATH: rm on this test's own folders, systemctl and docker logged
        (self.bin / "sudo").write_text(f'#!/bin/bash\necho "sudo $*" >> "{self.log}"\nexec "$@"\n')
        for tool in ("docker", "systemctl", "sudo"):
            (self.bin / tool).chmod(0o755)

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def run_block(self):
        start = TEXT.index('if [ "$ACTION" = uninstall ]; then')
        block = TEXT[start:TEXT.index("\n  exit 0\nfi\n", start) + len("\n  exit 0\nfi\n")]
        pins = [ln for ln in TEXT.splitlines() if ln.startswith(("LFM25_", "LOCAL_IMAGE=", "VLANE="))]
        script = "\n".join(["set -euo pipefail", "step(){ :; }", "die(){ echo \"ERROR: $*\" >&2; exit 1; }",
                            f'LANES_UNIT_DIR="{self.sd}"', f'HERE="{REPO}"', f'. "{REPO}/lanes.sh"',
                            f'. "{REPO}/sglang-lane.sh"', *pins, "ACTION=uninstall", "UNIT=qwen38-lfm25.service",
                            f'INSTALLED="{self.unit}"', "CONTAINER=qwen38-lfm25", f'CONFIG_DIR="{self.cfg}"',
                            f'HF_CACHE="{self.hf}"', block])
        env = dict(os.environ, PATH=f"{self.bin}:/usr/bin:/bin", HOME=str(self.home))
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r.stdout, self.log.read_text() if self.log.exists() else ""

    def test_its_own_parts_go(self):
        out, calls = self.run_block()
        self.assertFalse(self.unit.exists())
        self.assertIn("unit removed", out)
        self.assertIn("docker rm -f qwen38-lfm25", calls)
        self.assertIn(f"docker rmi {self.tag}", calls)
        self.assertFalse((self.cfg / "lfm25-cache").exists())

    def test_the_27b_lanes_image_and_key_and_the_vllm_lanes_middleware_stay(self):
        out, calls = self.run_block()
        self.assertNotIn(f"rmi {PINS['LFM25_IMAGE']}", calls)
        self.assertNotRegex(calls, r"docker rmi lmsysorg/sglang")
        self.assertIn(f"untagged {self.tag}: the image stays", out)
        self.assertTrue((self.cfg / "engine-secrets.yaml").exists(), "the 27B lane's unit reads it too")
        self.assertTrue((self.cfg / "vllm-auth").is_dir(), "the vLLM lanes' middleware is none of this lane's")

    def test_the_checkpoints_stay_and_it_says_so(self):
        out, _ = self.run_block()
        for repo in (PINS["LFM25_REPO"], PINS["LFM25_DRAFT_REPO"]):
            d = self.hf / "hub" / f"models--{repo.replace('/', '--')}"
            self.assertTrue(d.is_dir())
            self.assertIn(f"checkpoint kept, delete it yourself for the space: rm -rf {d}", out)

    def test_its_opencode_provider_goes_and_the_default_comes_back_to_the_27b(self):
        self.run_block()
        cfg = json.loads(self.oc.read_text())
        self.assertNotIn("lfm25", cfg["provider"])
        self.assertIn("qwen38", cfg["provider"])
        self.assertEqual(cfg["model"], "qwen38/qwen3.8-27b")


class TheSwitch(unittest.TestCase):
    """./switch-model.sh lfm25, run as written on a copy whose unit paths point at this test's folder: its checkpoint and
    its draft fetched in the 27B lane's image, its unit rendered by its installer's --stage and put in place, the boot
    moved to it from the text lane noted first, opencode pointed at its provider, and nothing started or stopped."""

    def setUp(self):
        self.t = pathlib.Path(tempfile.mkdtemp(prefix="lfm25-switch-"))
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
        for unit in ("qwen38-sglang.service", "qwen38-flash.service", "qwen38-lfm25.service"):
            text = text.replace(f'"/etc/systemd/system/{unit}"', f'"{self.units}/{unit}"')
        (repo / "switch-model.sh").write_text(text)
        (repo / "switch-model.sh").chmod(0o755)
        for f in ("install.sh", "lanes.sh", "boxmem.py", "ckpt-fetch.py", "install-lfm25.sh", "sglang-lane.sh", "vllm-lane.sh",
                  "qwen38-lfm25.service.template", "oc-limits.sh", "oc-lfm25.py", "oc-merge-limits.py",
                  "oc-point-default.py", "pi-gen.py"):
            shutil.copy(REPO / f, repo / f)
        self.hf = self.t / "hf"
        # the unit an install put there: the 27B lane's image, this box's cache
        self.unit = self.units / "qwen38-lfm25.service"
        self.unit.write_text("[Service]\nExecStart=/bin/bash -c 'exec /usr/bin/docker run --rm --name qwen38-lfm25 \\\n"
                             f"  -v {self.hf}:/root/.cache/huggingface \\\n  {PINS['LFM25_IMAGE']} \\\n"
                             f"  python3 -m sglang.launch_server --model-path {PINS['LFM25_REPO']} --revision old'\n")
        (self.units / "qwen38-sglang.service").write_text("[Service]\n")
        self.oc = self.cfg / "opencode.json"
        self.oc.write_text(json.dumps({"model": "qwen38/qwen3.8-27b", "provider": {"qwen38": {
            "options": {"baseURL": "http://127.0.0.1:30001/v1"},
            "models": {"qwen3.8-27b": {"limit": {"context": 700000, "output": 200000}}}}}}))
        self.bin = self.t / "bin"
        self.bin.mkdir()
        self.log = self.t / "calls.log"
        stub = {"docker": f'echo "docker $*" >> "{self.log}"\nexit 0',
                # the 27B lane at boot, nothing running
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
        r = subprocess.run([str(self.script), "lfm25", *args], capture_output=True, text=True, env=e, timeout=120)
        return r.returncode, r.stdout + r.stderr, self.log.read_text() if self.log.exists() else ""

    def test_it_puts_in_its_unit_moves_the_boot_and_points_opencode_at_it(self):
        rc, out, calls = self.switch()
        self.assertEqual(rc, 0, out)
        unit = self.unit.read_text()
        self.assertIn(f"--model-path {PINS['LFM25_REPO']} --revision {PINS['LFM25_REV']} ", unit, "its installer's own render")
        self.assertIn(f"-v {self.hf}:/root/.cache/huggingface ", unit, "the cache its unit mounted")
        self.assertFalse((self.cfg / "qwen38-lfm25.service.switch-stage").exists())
        fetches = [c for c in calls.splitlines() if c.startswith("docker run") and "/opt/qwen38/ckpt-fetch.py" in c]
        self.assertEqual(len(fetches), 2, calls)
        self.assertIn(f" {PINS['LFM25_DRAFT_REPO']} {PINS['LFM25_DRAFT_REV']} ", fetches[1])
        self.assertIn("sudo systemctl disable qwen38-sglang.service", calls)
        self.assertIn("sudo systemctl enable qwen38-lfm25.service", calls)
        self.assertEqual((self.cfg / "text-lane").read_text().strip(), "qwen38-sglang.service")
        self.assertFalse(re.search(r"systemctl (start|stop|restart) qwen38-", calls), calls)
        self.assertIn("Effective after:  sudo systemctl start qwen38-lfm25.service", out)
        cfg = json.loads(self.oc.read_text())
        self.assertEqual(cfg["model"], f"lfm25/{SERVED}")
        self.assertEqual(cfg["provider"]["lfm25"]["models"][SERVED]["limit"], {"context": 74000, "input": 74000, "output": 32000})
        self.assertIn("qwen38", cfg["provider"])

    def test_with_no_unit_installed_it_says_how_to_install_it_and_changes_nothing(self):
        self.unit.unlink()
        rc, out, calls = self.switch()
        self.assertNotEqual(rc, 0)
        self.assertIn("./install.sh --with-lfm25", out)
        self.assertNotIn("docker run", calls)
        self.assertNotIn("sudo", calls)


class ItsClients(unittest.TestCase):
    def test_its_pair_fits_its_window_by_both_rules(self):
        r = subprocess.run(["bash", str(REPO / "oc-limits.sh"), "lfm25"], capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "74000 32000 local, 131K", r.stderr)
        ctx, out = 74000, 32000
        self.assertLessEqual(ctx - 20000 + 43863 + out, 131072, "opencode's compaction point, one agent step, the answer")
        fit = load("oc_fit_limits_lfm25", "oc-fit-limits.py")
        self.assertEqual(fit.LANE_MODEL[SERVED], "lfm25")
        # the pool measured on the reference box, 1,667,049 tokens, past the window 12.7 times: the window decides
        self.assertEqual(fit.fit(1667049, 0, 131072, (ctx, out)), (ctx, out))

    def test_the_fit_reads_its_pair_from_its_own_unit(self):
        with tempfile.TemporaryDirectory(prefix="lfm25-fit-") as d:
            unit = pathlib.Path(d, "qwen38-lfm25.service")
            unit.write_text("  --context-length 131072 \\\n  python3 -m sglang.launch_server \\\n")
            os.environ["OC_LFM25_UNIT"] = str(unit)
            try:
                fit = load("oc_fit_limits_lfm25_unit", "oc-fit-limits.py")
                self.assertEqual(fit.lane_cap(SERVED), (74000, 32000))
            finally:
                del os.environ["OC_LFM25_UNIT"]

    def test_its_provider_is_text_with_no_effort_to_choose(self):
        r = subprocess.run(["python3", str(REPO / "oc-lfm25.py"), "http://127.0.0.1:30001/v1", "/k", "74000", "32000", "n"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        block = json.loads(r.stdout)["provider"]["lfm25"]
        self.assertEqual(list(block["models"]), [SERVED])
        model = block["models"][SERVED]
        self.assertEqual(model["limit"], {"context": 74000, "input": 74000, "output": 32000})
        # opencode 1.19.5 offers no level for a model with neither: its template reads none, and "reasoning" alone
        # would make opencode offer low, medium and high (ProviderTransform.variants on @ai-sdk/openai-compatible)
        for key in ("variants", "reasoning", "reasoning_options"):
            self.assertNotIn(key, model, "its template reads no level: none to offer")
        self.assertNotIn("modalities", model, "it reads no image")
        self.assertEqual(block["options"]["apiKey"], "{file:/k}")

    def test_its_name_in_the_picker_says_its_checkpoint_and_window(self):
        r = subprocess.run(["python3", str(REPO / "oc-point-default.py"), "--label", "lfm25", "131072"],
                           capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "Liquid AI LFM2.5 24B-A2B FP8 + DSpark (local, 131K)")

    def test_opencode_is_pointed_at_it_and_pi_reads_it(self):
        with tempfile.TemporaryDirectory(prefix="lfm25-oc-") as d:
            oc = pathlib.Path(d, "opencode.json")
            prov = json.loads(subprocess.run(["python3", str(REPO / "oc-lfm25.py"), "http://127.0.0.1:30001/v1", "/k",
                                              "74000", "32000", "n"], capture_output=True, text=True).stdout)
            oc.write_text(json.dumps({"model": "qwen38/qwen3.8-27b", "provider": {"qwen38": {
                "options": {"baseURL": "http://127.0.0.1:30001/v1"}, "models": {"qwen3.8-27b": {}}}}}))
            src = pathlib.Path(d, "src.json")
            src.write_text(json.dumps(prov))
            r = subprocess.run(["python3", str(REPO / "oc-merge-limits.py"), str(oc), "--add-providers", str(src)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            r = subprocess.run(["python3", str(REPO / "oc-point-default.py"), str(oc), "lfm25", "lfm25", "131072"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            cfg = json.loads(oc.read_text())
            self.assertEqual(cfg["model"], f"lfm25/{SERVED}")
            self.assertIn("lfm25", cfg["provider"])
            pi = load("pi_gen_lfm25", "pi-gen.py")
            models, settings, _ = pi.build(cfg)
            self.assertIn("lfm25", models["providers"])
            self.assertEqual((settings["defaultProvider"], settings["defaultModel"]), ("lfm25", SERVED))


if __name__ == "__main__":
    unittest.main()
