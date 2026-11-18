#!/usr/bin/env python3
"""The daisyrose target is served with the 27B lane's fp8 KV cache, declared in its checkpoint's config.

RadixArk/DaisyRose-27B-NVFP4 declares no KV cache, and SGLang gives such a checkpoint a bf16 one and about
half the pool. Like arete, it is served with the lane's declaration written into its config: target cache
fp8, draft cache bf16 (patch-kv.py says why that and not the flag the FP8 targets pass). So its switch and
its install write that declaration, after the window's configs (a YaRN restore puts back a config.json
saved before it), and nothing else: its unit is stock's, with no --kv-cache-dtype, and no other target's
config is touched. run.sh, which serves what ./install.sh --no-service prepared, refuses it undeclared.
Nothing is measured on the reference box yet. The scripts' own lines run here, patch-yarn.py and
patch-kv.py replaced by stubs that record their calls in one log, in order."""
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
SWITCH = (REPO / "switch-model.sh").read_text()
INSTALL = (REPO / "install.sh").read_text()
RUN = (REPO / "run.sh").read_text()
PINS = dict(re.findall(r'^([A-Z0-9_]+_(?:REPO|REV))="([^"$]+)"', INSTALL, re.M))
DAISYROSE = (PINS["DAISYROSE_REPO"], PINS["DAISYROSE_REV"])
TARGETS_27B = {"stock": "STOCK", "uncensored": "UNC", "fp8": "FP8", "uncensored-fp8": "UNCFP8", "arete": "ARETE",
               "daisyrose": "DAISYROSE"}


def block(text, start, end_after):
    """text from the line holding `start` to the end of the first `fi` line after the line holding `end_after`."""
    s = text.rindex("\n", 0, text.index(start)) + 1
    return text[s:text.index("\nfi\n", text.index(end_after, s)) + 4]


def stubs(d, kv_rc=0):
    """patch-yarn.py and patch-kv.py in d, each line of the shared log: the script, then its arguments."""
    for name, rc in (("patch-yarn.py", 0), ("patch-kv.py", kv_rc)):
        (d / name).write_text(
            "import os, sys\n"
            "open(os.path.join(os.path.dirname(sys.argv[0]), 'calls'), 'a').write("
            "os.path.basename(sys.argv[0]) + ' ' + ' '.join(sys.argv[1:]) + '\\n')\n"
            f"sys.exit({rc})\n")


def bash(script, d):
    r = subprocess.run(["bash", "-c", 'set -euo pipefail\ndie(){ echo "DIE: $*"; exit 1; }\n' + script],
                       capture_output=True, text=True, timeout=30)
    calls = d / "calls"
    return r.stdout + r.stderr, calls.read_text().splitlines() if calls.exists() else []


class TheSwitch(unittest.TestCase):
    # steps 2 and 2a as the switch runs them since v1.52.0's seventh pass: lanes.sh's patch_served_checkpoint, which a
    # Load's prepare calls too when an update waits (its step 2b, the chat template, is stubbed and left out of the
    # calls compared here)
    def test_the_switch_runs_them_through_the_shared_function(self):
        self.assertIn('patch_served_checkpoint "$TARGET_LANE" "$CHOICE" "$TARGET_UNIT" "$HF_CACHE" "$TARGET_REPO" "$TARGET_REV"', SWITCH)

    def switch(self, choice, window, kv_rc=0):
        d = pathlib.Path(tempfile.mkdtemp(prefix="daisyrose-switch-"))
        self.addCleanup(shutil.rmtree, d, True)
        stubs(d, kv_rc)
        (d / "patch-template.py").write_text(
            "import os, sys\n"
            "open(os.path.join(os.path.dirname(sys.argv[0]), 'calls'), 'a').write("
            "os.path.basename(sys.argv[0]) + ' ' + ' '.join(sys.argv[1:]) + '\\n')\n")
        (d / "lanes.sh").write_text((REPO / "lanes.sh").read_text())
        (d / "unit").write_text(f"ExecStart=... --context-length {window} ...\n")
        repo, rev = PINS[TARGETS_27B[choice] + "_REPO"], PINS[TARGETS_27B[choice] + "_REV"]
        script = (f'CONFIG_DIR="{d}"; . "{d}/lanes.sh"\n'
                  f'patch_served_checkpoint 27b {choice} "{d}/unit" /hf {repo} {rev}\necho REACHED\n')
        out, calls = bash(script, d)
        return out, [c for c in calls if not c.startswith("patch-template.py")]

    def test_daisyrose_gets_the_declaration_after_the_window_in_both_modes(self):
        repo, rev = DAISYROSE
        for window, yarn in ((1010000, f"patch-yarn.py /hf {repo} {rev}"), (262144, f"patch-yarn.py --restore /hf {repo} {rev}")):
            with self.subTest(window=window):
                out, calls = self.switch("daisyrose", window)
                self.assertEqual(calls, [yarn, f"patch-kv.py /hf {repo} {rev}"], out)
                self.assertIn("REACHED", out)

    def test_no_other_target_has_its_config_declared(self):
        for choice in ("stock", "uncensored", "fp8", "uncensored-fp8"):
            for window in (1010000, 262144):
                with self.subTest(choice=choice, window=window):
                    out, calls = self.switch(choice, window)
                    self.assertEqual([c for c in calls if c.startswith("patch-kv.py")], [], out)
                    self.assertEqual(len(calls), 1, out)
                    self.assertIn("REACHED", out)

    def test_a_declaration_that_fails_stops_the_switch_before_the_unit(self):
        out, _ = self.switch("daisyrose", 1010000, kv_rc=3)
        self.assertIn("DIE: the fp8 KV cache declaration failed", out)
        self.assertNotIn("REACHED", out)

    def test_its_unit_is_stocks_with_no_kv_flag(self):
        # the KV line of the rewritten unit, as switch-model.sh decides it and then checks it
        lines = block(SWITCH, '  case "$CHOICE" in\n    fp8|uncensored-fp8) NEW_KV=', '    *)                  NEW_KV=""')
        lines = lines[:lines.index("  esac\n") + len("  esac\n")]
        for choice, kv in (("daisyrose", ""), ("stock", ""), ("fp8", "--kv-cache-dtype fp8_e4m3 ")):
            with self.subTest(choice):
                r = subprocess.run(["bash", "-c", f"set -eu\nCHOICE={choice}\n{lines}printf '%s' \"$NEW_KV\""],
                                   capture_output=True, text=True, timeout=30)
                self.assertEqual((r.returncode, r.stdout), (0, kv), r.stderr)


class TheInstaller(unittest.TestCase):
    FUNC = INSTALL[INSTALL.index("apply_context_configs(){"):INSTALL.index("\n}\n", INSTALL.index("apply_context_configs(){")) + 3]

    def configs(self, choice, mode, units_only=0, kv_rc=0):
        d = pathlib.Path(tempfile.mkdtemp(prefix="daisyrose-install-"))
        self.addCleanup(shutil.rmtree, d, True)
        stubs(d, kv_rc)
        repo, rev = PINS[TARGETS_27B[choice] + "_REPO"], PINS[TARGETS_27B[choice] + "_REV"]
        script = (f'REPO_DIR="{d}"; HF_CACHE=/hf; MODEL_CHOICE={choice}; MODEL_REPO={repo}; MODEL_REV={rev}; '
                  f'LANE=27b; CONTEXT_MODE={mode}; UNITS_ONLY={units_only}; KEEP_MODEL_VERBATIM=0; NO_SERVICE=0; '
                  f'SGL_UNIT_PATH="{d}/none"; DRAFT2_REPO=org/draft; DRAFT2_REV=dddd\n'
                  + self.FUNC + "apply_context_configs\necho REACHED\n")
        return bash(script, d)

    def test_daisyrose_gets_the_declaration_last_in_both_modes(self):
        repo, rev = DAISYROSE
        for mode, opt in (("1m", ""), ("native", "--restore ")):
            with self.subTest(mode):
                out, calls = self.configs("daisyrose", mode)
                self.assertEqual(calls, [f"patch-yarn.py {opt}/hf {repo} {rev}", f"patch-yarn.py {opt}/hf org/draft dddd",
                                         f"patch-kv.py /hf {repo} {rev}"], out)
                self.assertIn("REACHED", out)

    def test_no_other_target_has_its_config_declared(self):
        for choice in ("stock", "uncensored", "fp8", "uncensored-fp8"):
            for mode in ("1m", "native"):
                with self.subTest(choice=choice, mode=mode):
                    out, calls = self.configs(choice, mode)
                    self.assertEqual(len(calls), 2, out)
                    self.assertFalse(any(c.startswith("patch-kv.py") for c in calls), out)

    def test_a_lane_installed_on_demand_leaves_it_to_its_first_load(self):
        out, calls = self.configs("daisyrose", "1m", units_only=1)
        self.assertEqual(calls, [], out)
        self.assertIn("REACHED", out)

    def test_a_declaration_that_fails_stops_the_install(self):
        out, _ = self.configs("daisyrose", "1m", kv_rc=3)
        self.assertIn("DIE: the fp8 KV cache declaration failed", out)
        self.assertNotIn("REACHED", out)

    def test_its_unit_is_stocks_with_no_kv_flag(self):
        lines = block(INSTALL, 'case "$MODEL_CHOICE" in\n  fp8|uncensored-fp8) KV_CACHE_ARGS=', '  *)                  KV_CACHE_ARGS=""')
        lines = lines[:lines.index("esac\n") + len("esac\n")]
        for choice, kv in (("daisyrose", ""), ("stock", ""), ("uncensored-fp8", "--kv-cache-dtype fp8_e4m3 ")):
            with self.subTest(choice):
                r = subprocess.run(["bash", "-c", f"set -eu\nMODEL_CHOICE={choice}\n{lines}printf '%s' \"$KV_CACHE_ARGS\""],
                                   capture_output=True, text=True, timeout=30)
                self.assertEqual((r.returncode, r.stdout), (0, kv), r.stderr)


class RunSh(unittest.TestCase):
    LINES = block(RUN, 'if [ "$MODEL_CHOICE" = daisyrose ]; then', 'if [ "$MODEL_CHOICE" = daisyrose ]; then')

    def run_check(self, choice, check_rc):
        d = pathlib.Path(tempfile.mkdtemp(prefix="daisyrose-run-"))
        self.addCleanup(shutil.rmtree, d, True)
        stubs(d, check_rc)
        repo, rev = PINS[TARGETS_27B[choice] + "_REPO"], PINS[TARGETS_27B[choice] + "_REV"]
        script = (f'REPO_DIR="{d}"; HF_CACHE=/hf; MODEL_CHOICE={choice}; MODEL_REPO={repo}; MODEL_REV={rev}; '
                  f'PREP="MODEL_CHOICE={choice} ./install.sh --no-service"\n' + self.LINES + "echo REACHED\n")
        return bash(script, d)

    def test_an_undeclared_daisyrose_is_refused_with_its_preparation(self):
        out, calls = self.run_check("daisyrose", 2)
        self.assertEqual(calls, [f"patch-kv.py --check /hf {DAISYROSE[0]} {DAISYROSE[1]}"])
        self.assertIn("DIE: checkpoint RadixArk/DaisyRose-27B-NVFP4 does not declare the fp8 KV cache", out)
        self.assertIn("Run: MODEL_CHOICE=daisyrose ./install.sh --no-service", out)
        self.assertNotIn("REACHED", out)

    def test_a_declared_daisyrose_runs_and_the_others_are_not_asked(self):
        out, calls = self.run_check("daisyrose", 0)
        self.assertIn("REACHED", out)
        out, calls = self.run_check("stock", 2)
        self.assertEqual(calls, [])
        self.assertIn("REACHED", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
