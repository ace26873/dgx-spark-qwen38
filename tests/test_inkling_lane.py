#!/usr/bin/env python3
"""The Inkling 2 Mini lane (install-inkling.sh): the unit its --stage writes for a switch, the room its fetch asks, what
its --uninstall removes and keeps, and its clients' pair, provider and efforts (the proxy's part is
test_proxy_inkling.py's).

--stage runs the installer as written, in a throwaway home, with no sudo and no docker. The uninstall block runs whole
against a fake unit folder and a fake config directory, sudo, systemctl and docker stubs writing down every call, as in
test_lfm25_lane.py."""
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
TEXT = (REPO / "install-inkling.sh").read_text()
PINS = {k: v.strip('"') for k, v in (ln.split("=", 1) for ln in TEXT.splitlines()
                                       if ln.startswith("INKLING_") and "=" in ln)}
SERVED = "inkling-2-mini"
# the KV cache the reference box's engine held at the cookbook's share (its journal, 2026-11-22: max_total_num_tokens)
POOL = 490496
WINDOW = 1048576


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
        self.t = pathlib.Path(tempfile.mkdtemp(prefix="inkling-stage-"))
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
        r = subprocess.run(["bash", str(REPO / "install-inkling.sh"), "--stage", str(self.out)],
                           capture_output=True, text=True, env=e, timeout=120)
        return r.returncode, r.stdout + r.stderr

    def test_the_unit_is_whole_and_serves_the_pinned_checkpoint_on_its_own_image(self):
        rc, out = self.stage()
        self.assertEqual(rc, 0, out)
        unit = self.out.read_text()
        self.assertNotRegex(unit, r"__[A-Z][A-Z0-9_]*__", "a placeholder left")
        cmd = exec_command(unit)
        self.assertEqual(PINS["INKLING_IMAGE"],
                         "lmsysorg/sglang@sha256:bc811ac7ac68c6fca116a49750baca817b8613a30abface8a347682b55d132e3",
                         "the arm64 build of lmsysorg/sglang:inkling-2-mini the reference box ran")
        self.assertIn(f" {PINS['INKLING_IMAGE']} ", cmd)
        self.assertIn(f"--model-path {PINS['INKLING_REPO']} --revision {PINS['INKLING_REV']} ", cmd)
        self.assertIn(f"--served-model-name {SERVED} ", cmd)
        self.assertIn(f"-v {self.hf}:/root/.cache/huggingface ", cmd)
        self.assertIn("--host 127.0.0.1 --port 30000", cmd)

    def test_its_caches_are_kept_from_one_start_to_the_next_and_made_as_this_user(self):
        """The kernels SGLang and b12x compile at the first start (379 s, then 222 s on the reference box, whose runs kept
        them): their two folders are the lane's, beside torch's compile cache, made by the user, not by docker as root."""
        self.assertEqual(self.stage()[0], 0)
        cmd = exec_command(self.out.read_text())
        self.assertIn(f"-v {self.home}/.config/qwen38/inkling-cache:/cache ", cmd)
        for d in ("sglang", "b12x"):
            self.assertIn(f"-v {self.home}/.config/qwen38/inkling-cache/{d}:/root/.cache/{d} ", cmd)
            self.assertTrue((self.cfg / "inkling-cache" / d).is_dir(), d)

    def test_it_serves_the_cookbooks_dgx_spark_recipe_with_dspark(self):
        """The recipe the reference box measured on 2026-11-22 (~/qa-tools/etudes/inkling/banc-inkling.sh, its sg-dspark
        arm), and the window and the 48 requests SGLang set itself at that run."""
        self.assertEqual(self.stage()[0], 0)
        cmd = re.sub(r"\s+", " ", exec_command(self.out.read_text()))
        for flag in ("--tp 1", "--ep 1", "--attention-backend fa4", "--dtype bfloat16", "--page-size 128",
                     "--mem-fraction-static 0.45", "--speculative-algorithm DSPARK", "--speculative-num-draft-tokens 8",
                     "--grammar-backend xgrammar", "--enable-multimodal", "--context-length 1048576",
                     "--max-running-requests 48"):
            self.assertIn(f" {flag} ", f" {cmd} ", flag)
        for env in ("MAX_JOBS=4", "SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL=1", "SGLANG_OPT_DSV41_DSPARK_MID_FOLD=1",
                    "SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE=1", "SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS=1"):
            self.assertIn(f"-e {env} ", cmd, env)
        # the draft is in the checkpoint: no other repository to name
        self.assertNotIn("--speculative-draft-model-path", cmd)

    def test_its_parsers_are_the_recipes_and_its_template_its_own(self):
        self.assertEqual(self.stage()[0], 0)
        cmd = exec_command(self.out.read_text())
        self.assertIn("--reasoning-parser inkling --tool-call-parser inkling2 ", cmd)
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
        self.assertIn(f"ExecStartPre=/bin/bash {self.home}/.config/qwen38/engine-preflight.sh qwen38-inkling.service\n", unit)
        for env in ("HF_HUB_OFFLINE=1", "SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES=1", "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0",
                    "SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE=file:///tmp/sglang-rdv", "GLOO_SOCKET_IFNAME=lo",
                    "NCCL_SOCKET_IFNAME=lo"):
            self.assertIn(f"-e {env} ", cmd, env)
        for flag in ("--sleep-on-idle", "--enable-metrics", "--enable-cache-report"):
            self.assertIn(f"{flag} ", cmd, flag)

    def test_its_engine_is_the_one_engine_on_the_box(self):
        self.assertEqual(self.stage()[0], 0)
        head = self.out.read_text().split("[Service]")[0]
        conflicts = set(re.search(r"^Conflicts=(.*)$", head, re.M).group(1).split())
        after = set(re.search(r"^After=(.*)$", head, re.M).group(1).split())
        others = {f"{t.name[:-len('.template')]}" for t in REPO.glob("qwen38-*.service.template")
                  if re.search(r"^Conflicts=", t.read_text(), re.M)} - {"qwen38-inkling.service", "qwen38-sglang-1m.service",
                                                                         "qwen38-sglang-64.service"}
        self.assertEqual(conflicts, others)
        self.assertLessEqual(conflicts, after)

    def test_with_no_file_to_write_to_it_refuses(self):
        e = {"PATH": "/usr/bin:/bin", "HOME": str(self.home), "LANES_UNIT_DIR": str(self.units)}
        r = subprocess.run(["bash", str(REPO / "install-inkling.sh"), "--stage"], capture_output=True, text=True, env=e,
                           timeout=60)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--stage takes the file to write the unit to", r.stderr)


class TheRoomItsFetchAsks(unittest.TestCase):
    def test_the_repositorys_size_rounded_up(self):
        """INKLING_GB is what the fetch asks room for and what the help and the cockpit say it downloads: the Hub's file
        sizes of the repository at its revision, written above it, rounded up to the GB."""
        m = re.search(r"^# The repository at this revision, ([\d,]+) bytes .*\n(?:#.*\n)*INKLING_GB=\"(\d+)\"$", TEXT, re.M)
        self.assertIsNotNone(m, "install-inkling.sh: no measure of the repository above INKLING_GB")
        self.assertEqual(int(m.group(2)), math.ceil(int(m.group(1).replace(",", "")) / 1e9))
        js = (REPO / "dashboard" / "static" / "js" / "base.js").read_text()
        self.assertIsNotNone(re.search(rf"^  inkling: '[^']*The first switch downloads about {m.group(2)} GB\.'(?:}};|,)$", js,
                                       re.M), "base.js's note says another size")

    def test_the_image_at_its_pulls_peak_rounded_up_as_df_counts_it(self):
        """IMAGE_GB is the image unpacked and its download kept beside it by the containerd store, in GiB, times 1.02,
        rounded up, plus one: the rule install-lfm3.sh counts its image by."""
        m = re.search(r"^# Its pull at its peak: ([\d.]+) GB unpacked .* and ([\d.]+) GB to download .*\n(?:#.*\n)*IMAGE_GB=(\d+)$",
                      TEXT, re.M)
        self.assertIsNotNone(m, "install-inkling.sh: no measure of the image above IMAGE_GB")
        gib = (float(m.group(1)) + float(m.group(2))) * 1e9 / 2 ** 30
        self.assertEqual(int(m.group(3)), math.ceil(gib * 1.02) + 1)
        sh = (REPO / "install.sh").read_text()
        self.assertIn(f"--no-inkling          leave the Inkling 2 Mini lane (SGLang, {int(PINS['INKLING_GB']) + int(m.group(3))} GB)",
                      sh)


class TheFetch(unittest.TestCase):
    def test_the_repository_is_pinned_as_refs_main(self):
        """The unit runs with HF_HUB_OFFLINE=1 and SGLang takes the DSpark draft from the checkpoint by its name with no
        revision, through refs/main, which a download by commit never writes."""
        self.assertIn('vlane_fetch "$INKLING_REPO" "$INKLING_REV" pin', TEXT)
        self.assertIn("--refs-main pin", (REPO / "switch-model.sh").read_text().split('if [ "$CHOICE" = "inkling" ]; then')[1]
                      .split("\nfi\n")[0])


class TheUninstall(unittest.TestCase):
    """Its unit, its container, its local tag on its image, its caches and its opencode provider go; the key's yaml (the
    27B lane's unit reads it too), the vLLM lanes' middleware and the checkpoint stay. The image is no other lane's: once
    its tag is gone, a docker image prune can reclaim it, and the uninstall says so."""

    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp(prefix="inkling-uninstall-"))
        self.home = self.d / "home"
        self.cfg = self.home / ".config" / "qwen38"
        for sub in ("inductor", "sglang", "b12x"):
            (self.cfg / "inkling-cache" / sub).mkdir(parents=True)
        (self.cfg / "vllm-auth").mkdir()
        (self.cfg / "engine-secrets.yaml").write_text('api-key: "k"\n')
        self.sd = self.d / "systemd"
        self.sd.mkdir()
        self.unit = self.sd / "qwen38-inkling.service"
        self.unit.write_text("ExecStart=/bin/bash -c 'exec /usr/bin/docker run x'\n")
        self.hf = self.d / "hf"
        (self.hf / "hub" / f"models--{PINS['INKLING_REPO'].replace('/', '--')}").mkdir(parents=True)
        self.oc = self.cfg / "opencode.json"
        self.oc.write_text(json.dumps({"model": f"inkling/{SERVED}", "provider": {
            "qwen38": {"models": {"qwen3.8-27b": {"limit": {"context": 173000, "output": 64000}}}},
            "inkling": {"models": {SERVED: {"limit": {"context": 374000, "output": 32000}}}}}}))
        self.bin = self.d / "bin"
        self.bin.mkdir()
        self.log = self.d / "calls.log"
        self.tag = f"qwen38-pinned:inkling-{PINS['INKLING_IMAGE'].split('@sha256:')[1][:12]}"
        # its tag and no other on the image
        docker = ("#!/bin/bash\n"
                  f'echo "docker $*" >> "{self.log}"\n'
                  f'case "$1" in images) echo "{self.tag} sha256:feed" ;;\nesac\nexit 0\n')
        (self.bin / "docker").write_text(docker)
        (self.bin / "systemctl").write_text(f'#!/bin/bash\necho "systemctl $*" >> "{self.log}"\n'
                                            '[ "$1" = is-enabled ] && exit 1\n[ "$1" = is-active ] && exit 1\nexit 0\n')
        (self.bin / "sudo").write_text(f'#!/bin/bash\necho "sudo $*" >> "{self.log}"\nexec "$@"\n')
        for tool in ("docker", "systemctl", "sudo"):
            (self.bin / tool).chmod(0o755)

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def run_block(self):
        start = TEXT.index('if [ "$ACTION" = uninstall ]; then')
        block = TEXT[start:TEXT.index("\n  exit 0\nfi\n", start) + len("\n  exit 0\nfi\n")]
        pins = [ln for ln in TEXT.splitlines() if ln.startswith(("INKLING_", "LOCAL_IMAGE=", "VLANE="))]
        script = "\n".join(["set -euo pipefail", "step(){ :; }", "die(){ echo \"ERROR: $*\" >&2; exit 1; }",
                            f'LANES_UNIT_DIR="{self.sd}"', f'HERE="{REPO}"', f'. "{REPO}/lanes.sh"',
                            f'. "{REPO}/sglang-lane.sh"', f'CONFIG_DIR="{self.cfg}"', *pins, "ACTION=uninstall",
                            "UNIT=qwen38-inkling.service", f'INSTALLED="{self.unit}"', "CONTAINER=qwen38-inkling",
                            f'HF_CACHE="{self.hf}"', block])
        env = dict(os.environ, PATH=f"{self.bin}:/usr/bin:/bin", HOME=str(self.home))
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r.stdout, self.log.read_text() if self.log.exists() else ""

    def test_its_own_parts_go(self):
        out, calls = self.run_block()
        self.assertFalse(self.unit.exists())
        self.assertIn("unit removed", out)
        self.assertIn("docker rm -f qwen38-inkling", calls)
        self.assertIn(f"docker rmi {self.tag}", calls)
        self.assertFalse((self.cfg / "inkling-cache").exists())

    def test_its_image_is_left_to_a_prune_and_the_key_and_the_middleware_stay(self):
        out, calls = self.run_block()
        self.assertNotRegex(calls, r"docker rmi lmsysorg/sglang")
        self.assertIn(f"untagged {self.tag}: a docker image prune can reclaim it", out)
        self.assertTrue((self.cfg / "engine-secrets.yaml").exists(), "the 27B lane's unit reads it too")
        self.assertTrue((self.cfg / "vllm-auth").is_dir(), "the vLLM lanes' middleware is none of this lane's")

    def test_the_checkpoint_stays_and_it_says_so(self):
        out, _ = self.run_block()
        d = self.hf / "hub" / f"models--{PINS['INKLING_REPO'].replace('/', '--')}"
        self.assertTrue(d.is_dir())
        self.assertIn(f"checkpoint kept, delete it yourself for the space: rm -rf {d}", out)

    def test_its_opencode_provider_goes_and_the_default_comes_back_to_the_27b(self):
        self.run_block()
        cfg = json.loads(self.oc.read_text())
        self.assertNotIn("inkling", cfg["provider"])
        self.assertIn("qwen38", cfg["provider"])
        self.assertEqual(cfg["model"], "qwen38/qwen3.8-27b")


class TheSwitch(unittest.TestCase):
    """./switch-model.sh inkling, run as written on a copy whose unit paths point at this test's folder: its checkpoint
    fetched in its image, its unit rendered by its installer's --stage and put in place, the boot moved to it from the
    text lane noted first, opencode pointed at its provider, and nothing started or stopped."""

    def setUp(self):
        self.t = pathlib.Path(tempfile.mkdtemp(prefix="inkling-switch-"))
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
        for unit in ("qwen38-sglang.service", "qwen38-flash.service", "qwen38-inkling.service"):
            text = text.replace(f'"/etc/systemd/system/{unit}"', f'"{self.units}/{unit}"')
        (repo / "switch-model.sh").write_text(text)
        (repo / "switch-model.sh").chmod(0o755)
        for f in ("install.sh", "lanes.sh", "boxmem.py", "ckpt-fetch.py", "install-inkling.sh", "sglang-lane.sh",
                  "vllm-lane.sh", "qwen38-inkling.service.template", "oc-limits.sh", "oc-inkling.py", "oc-merge-limits.py",
                  # the merger loads its sibling, the server matcher
                  "oc-permissions.py", "oc-point-default.py", "pi-gen.py"):
            shutil.copy(REPO / f, repo / f)
        self.hf = self.t / "hf"
        # the unit an install put there: its image, this box's cache
        self.unit = self.units / "qwen38-inkling.service"
        self.unit.write_text("[Service]\nExecStart=/bin/bash -c 'exec /usr/bin/docker run --rm --name qwen38-inkling \\\n"
                             f"  -v {self.hf}:/root/.cache/huggingface \\\n  {PINS['INKLING_IMAGE']} \\\n"
                             f"  python3 -m sglang.launch_server --model-path {PINS['INKLING_REPO']} --revision old'\n")
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
        r = subprocess.run([str(self.script), "inkling", *args], capture_output=True, text=True, env=e, timeout=120)
        return r.returncode, r.stdout + r.stderr, self.log.read_text() if self.log.exists() else ""

    def test_it_puts_in_its_unit_moves_the_boot_and_points_opencode_at_it(self):
        rc, out, calls = self.switch()
        self.assertEqual(rc, 0, out)
        unit = self.unit.read_text()
        self.assertIn(f"--model-path {PINS['INKLING_REPO']} --revision {PINS['INKLING_REV']} ", unit,
                      "its installer's own render")
        self.assertIn(f"-v {self.hf}:/root/.cache/huggingface ", unit, "the cache its unit mounted")
        self.assertFalse((self.cfg / "qwen38-inkling.service.switch-stage").exists())
        fetches = [c for c in calls.splitlines() if c.startswith("docker run") and "/opt/qwen38/ckpt-fetch.py" in c]
        self.assertEqual(len(fetches), 1, calls)
        self.assertIn(f" {PINS['INKLING_IMAGE']} /opt/qwen38/ckpt-fetch.py {PINS['INKLING_REPO']} {PINS['INKLING_REV']} "
                      f"--offline-first --refs-main pin --need-gb {PINS['INKLING_GB']} ", fetches[0])
        self.assertIn("sudo systemctl disable qwen38-sglang.service", calls)
        self.assertIn("sudo systemctl enable qwen38-inkling.service", calls)
        self.assertEqual((self.cfg / "text-lane").read_text().strip(), "qwen38-sglang.service")
        self.assertFalse(re.search(r"systemctl (start|stop|restart) qwen38-", calls), calls)
        self.assertIn("Effective after:  sudo systemctl start qwen38-inkling.service", out)
        cfg = json.loads(self.oc.read_text())
        self.assertEqual(cfg["model"], f"inkling/{SERVED}")
        self.assertEqual(cfg["provider"]["inkling"]["models"][SERVED]["limit"],
                         {"context": 374000, "input": 374000, "output": 32000})
        self.assertIn("qwen38", cfg["provider"])

    def test_with_no_unit_installed_it_says_how_to_install_it_and_changes_nothing(self):
        self.unit.unlink()
        rc, out, calls = self.switch()
        self.assertNotEqual(rc, 0)
        self.assertIn("./install.sh --with-inkling", out)
        self.assertNotIn("docker run", calls)
        self.assertNotIn("sudo", calls)


class ItsClients(unittest.TestCase):
    def test_its_pair_fits_its_pool_and_its_window(self):
        """Its window is past twice its pool: the pool decides, by the cockpit's check with a pool 10 % smaller than the
        one measured, and the window holds the prompt at opencode's compaction point, one agent step and the answer."""
        r = subprocess.run(["bash", str(REPO / "oc-limits.sh"), "inkling"], capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "374000 32000 local, 1M", r.stderr)
        ctx, out = 374000, 32000
        self.assertLessEqual(ctx + out, int(POOL * 0.92 * 0.90))
        self.assertLessEqual(ctx - 20000 + 43863 + out, WINDOW)

    def test_the_fit_keeps_its_answer_whatever_its_window(self):
        """A window past 262,144 tokens gave the answer a quarter of the pool (101,000 tokens on the pool measured): its
        card recommends limits on output tokens, and its pair is its ceiling (ANSWER_CAPPED), so the fit and the table
        land on one pair on that pool."""
        fit = load("oc_fit_limits_inkling", "oc-fit-limits.py")
        self.assertEqual(fit.LANE_MODEL[SERVED], "inkling")
        self.assertIn(SERVED, fit.ANSWER_CAPPED)
        self.assertEqual(fit.fit(POOL, 0, WINDOW, None), (304000, 101000), "the rule without the cap")
        self.assertEqual(fit.fit(POOL, 0, WINDOW, (374000, 32000)), (374000, 32000))
        with tempfile.TemporaryDirectory(prefix="inkling-fit-") as d:
            unit = pathlib.Path(d, "qwen38-inkling.service")
            unit.write_text("  --context-length 1048576 \\\n  python3 -m sglang.launch_server \\\n")
            os.environ["OC_INKLING_UNIT"] = str(unit)
            try:
                fit = load("oc_fit_limits_inkling_unit", "oc-fit-limits.py")
                self.assertEqual(fit.lane_cap(SERVED), (374000, 32000))
            finally:
                del os.environ["OC_INKLING_UNIT"]

    def test_its_provider_offers_the_templates_levels_and_images(self):
        r = subprocess.run(["python3", str(REPO / "oc-inkling.py"), "http://127.0.0.1:30001/v1", "/k", "374000", "32000",
                            "n"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        block = json.loads(r.stdout)["provider"]["inkling"]
        self.assertEqual(list(block["models"]), [SERVED])
        model = block["models"][SERVED]
        self.assertEqual(model["limit"], {"context": 374000, "input": 374000, "output": 32000})
        # its default first, as a picker lists it, then the others from the lowest: each a level its template maps
        self.assertEqual(list(model["variants"]), ["high", "none", "low", "medium", "xhigh"])
        # the names its chat_template.jinja maps (effort_map at the pinned revision): none 0.0, minimal 0.1, low 0.2,
        # medium 0.7, high 0.9, xhigh and max 0.99
        for level, v in model["variants"].items():
            self.assertIn(level, {"none", "minimal", "low", "medium", "high", "xhigh", "max"})
            self.assertEqual(v, {"chat_template_kwargs": {"reasoning_effort": level}})
        self.assertEqual(model["modalities"], {"input": ["text", "image"], "output": ["text"]})
        self.assertTrue(model["attachment"])
        self.assertEqual(block["options"]["apiKey"], "{file:/k}")

    def test_its_name_in_the_picker_says_its_checkpoint_and_window(self):
        r = subprocess.run(["python3", str(REPO / "oc-point-default.py"), "--label", "inkling", "1048576"],
                           capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "Inkling 2 Mini + DSpark (local, 1M)")


class ItIsNoLaneOfA64GbBox(unittest.TestCase):
    def test_boxmem_holds_it_for_a_128_gb_box_only(self):
        """Its memory was measured on a 128 GB box only (53.1 GiB of GPU memory at its peak): no 64 GB share is claimed."""
        bm = load("boxmem_inkling", "boxmem.py")
        self.assertTrue(bm.fits("inkling", "128"))
        self.assertFalse(bm.fits("inkling", "64"))
        self.assertEqual(bm.TARGETS["inkling"], "inkling")


if __name__ == "__main__":
    unittest.main()
