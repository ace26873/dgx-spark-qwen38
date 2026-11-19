#!/usr/bin/env python3
"""The Load of an update that waits (decision 1408), as prepare-lane.sh runs it for the two text
lanes, on a fabricated cache.

Sixth review of v1.52.0 (2026-11-19, GLM 5.3), two defects of the Load's prepare:
1. its `switch-model.sh <target> --fetch` stopped right after the download, so the switch's steps
   after the fetch never ran on the Load's short path (the lane serves the target the update
   waits for: prepare, apply_pending, restart, no switch). Those steps now run in apply_pending,
   once the unit is in place (cockpit.py's patch_waiting_checkpoint -> lanes.sh's
   patch_served_checkpoint): not in the prepare, where they touched the config the unit in place
   still serves when the wait serves the same snapshot with another window (seventh pass, 2026-11-19).
2. that fetch went into the cache of the unit in place (the switch reads it from the invocation
   when no HF_CACHE is passed), while prepare read the waiting unit's: with an HF_CACHE moved by
   a re-run of install, the Load downloaded into the old cache and looped. The fetch now passes
   the waiting unit's cache.

Seventh pass (2026-11-19): the prepare also fetched the PIN's revision instead of the one the
waiting unit names; it now reads --revision in the waiting file and passes it as MODEL_REV=.

The prepare runs as written in a HOME of the test's own, against a docker that records its calls
and knows the pin image is there, a df with room, and a switch-model.sh stub that records its
argv and the HF_CACHE and MODEL_REV it was given.
"""
import importlib.util
import json
import pathlib
import re
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
_INSTALL = (REPO / "install.sh").read_text()
PINS = dict(re.findall(r'^([A-Z0-9_]+(?:_REPO|_REV))="([^"$]+)"', _INSTALL, re.M))
PINS["IMAGE"] = re.search(r'^IMAGE="\$\{IMAGE:-([^}]+)\}"', _INSTALL, re.M).group(1)
_spec = importlib.util.spec_from_file_location("patch_template_under_test", REPO / "patch-template.py")
PT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PT)
# a chat template carrying every anchor patch-template.py looks for, as the checkpoints' own do
TEMPLATE = "\n".join((PT.EFFORT_ANCHOR, PT.LEAN_ANCHOR, PT.MSG_ANCHOR, PT.SYSTEM_ANCHOR,
                      PT.TOOL_REF_ANCHOR, PT.TOOL_REF_RAISE)) + "\n"

NATIVE = {"text_config": {"max_position_embeddings": 262144, "rope_scaling": {"type": "linear"}}}
YARNED = {"text_config": {"max_position_embeddings": 1010000,
                          "rope_scaling": {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144}}}


class Box:
    """prepare-lane.sh 27b|flash, with a waiting unit (or launcher) in $CONFIG_DIR/pending and a
    fabricated cache holding the new revision's snapshot in the cache that unit mounts."""

    def __init__(self, lane, target, window="1m", model=None, pending=True, quant=None,
                 pend_rev=None, pin_rev=None):
        self.t = t = pathlib.Path(tempfile.mkdtemp(prefix=f"prepare-wait-{lane}-"))
        self.d = d = t / "repo"
        d.mkdir()
        install_text = (REPO / "install.sh").read_text()
        if pin_rev is not None:
            # the pin moved in this checkout of the repo since the update waited (no ./install.sh re-run)
            key = "STOCK_REV" if lane == "27b" else "FLASH_REV"
            install_text = re.sub(rf'(?m)^{key}="[0-9a-f]{{40}}"', f'{key}="{pin_rev}"', install_text)
        for f in ("lanes.sh", "flash-ats-check.sh", "boxmem.py", "patch-yarn.py",
                  "patch-kv.py", "patch-template.py"):
            (d / f).write_text((REPO / f).read_text())
        (d / "install.sh").write_text(install_text)
        self.units = t / "units"
        self.units.mkdir()
        (d / "prepare-lane.sh").write_text(
            (REPO / "prepare-lane.sh").read_text().replace("/etc/systemd/system/", f"{self.units}/"))
        # the switch the Load's prepare calls: a stub recording its argv and the HF_CACHE and MODEL_REV it got
        (d / "switch-model.sh").write_text(
            f'#!/bin/bash\necho "switch-model.sh $*" >> "{t}/calls"\n'
            f'echo "${{HF_CACHE:-}}" >> "{t}/hfenv"\necho "${{MODEL_REV:-}}" >> "{t}/modrev"\n')
        (d / "switch-model.sh").chmod(0o755)
        self.bin = t / "bin"
        self.bin.mkdir()
        (self.bin / "docker").write_text(
            f'#!/bin/bash\necho "docker $*" >> "{t}/calls"\ncase "$1" in\n'
            f'  info) echo "{t}/docker-root" ;;\n'
            '  image) exit 0 ;;    # the pin image is there\n'
            'esac\nexit 0\n')
        (self.bin / "df").write_text('#!/bin/sh\necho Avail; echo 900G\n')
        (self.bin / "nvidia-smi").write_text('#!/bin/sh\necho ATS\n')
        for n in ("docker", "df", "nvidia-smi"):
            (self.bin / n).chmod(0o755)
        (t / "meminfo").write_text("MemTotal:       127533104 kB\nMemAvailable:   121000000 kB\n")
        self.home = t / "home"
        self.cfg = self.home / ".config" / "qwen38"
        self.cfg.mkdir(parents=True)
        self.hf_inplace = t / "hf-inplace"
        self.hf_wait = t / "hf-wait"
        self.hf_inplace.mkdir()
        self.hf_wait.mkdir()
        img = PINS["IMAGE"]
        repo = model or PINS[{"27b": "STOCK_REPO", "flash": "FLASH_REPO"}[lane]]
        rev = PINS[{"27b": "STOCK_REV", "flash": "FLASH_REV"}[lane]] if model is None else PINS["ARETE_REV"]
        if model is not None:
            repo = PINS["ARETE_REPO"]
        wrev = pend_rev or rev   # the revision the waiting unit names (the pin's unless it moved)
        window_line = "    --context-length 1010000 \\\\\n" if window == "1m" else ""
        unit_text = (
            "[Service]\nExecStart=/usr/bin/docker run --rm --name qwen38 --gpus all \\\n"
            f"  -v {{hf}}:/root/.cache/huggingface \\\\\n"
            f"  {img} \\\\\n  python3 -m sglang.launch_server \\\\\n"
            f"    --model-path {repo} --revision {{rev}} \\\\\n{window_line}    --port 30000\n")
        # the unit in place mounts the old cache and serves the target, the waiting one the new cache
        (self.units / "qwen38-sglang.service").write_text(
            unit_text.replace("{hf}", str(self.hf_inplace)).replace("{rev}", rev))
        (self.cfg / "launch-flash.sh").write_text(
            "exec /usr/bin/docker run --rm --name qwen38-flash --gpus all \\\n"
            f"  -v {self.hf_inplace}:/root/.cache/huggingface \\\n  -e HF_HUB_OFFLINE=1 \\\n"
            f"  {img} \\\n  python3 -m sglang.launch_server \\\n    --model-path {PINS['FLASH_REPO']} "
            f"--revision {rev}\\n")
        if pending:
            (self.cfg / "pending").mkdir()
            if lane == "27b":
                (self.cfg / "pending" / "qwen38-sglang.service").write_text(
                    unit_text.replace("{hf}", str(self.hf_wait)).replace("{rev}", wrev))
                (self.cfg / "pending" / "qwen38-sglang.service.needs").write_text("checkpoint gone\n")
            else:
                (self.cfg / "pending" / "launch-flash.sh").write_text(
                    "exec /usr/bin/docker run --rm --name qwen38-flash --gpus all \\\n"
                    f"  -v {self.hf_wait}:/root/.cache/huggingface \\\n  -e HF_HUB_OFFLINE=1 \\\n"
                    f"  {img} \\\n  python3 -m sglang.launch_server \\\n    --model-path {PINS['FLASH_REPO']} "
                    f"--revision {wrev}\n")
                (self.cfg / "pending" / "qwen38-flash.service.needs").write_text("checkpoint gone\n")
        # the fabricated cache: the new revision's snapshot, as if the fetch had just brought it
        self.repo, self.rev = repo, wrev
        snap = self.hf_wait / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / wrev
        snap.mkdir(parents=True)
        cfg = dict(NATIVE)
        if quant == "arete":
            cfg = {"text_config": NATIVE["text_config"],
                   "quantization_config": {"quant_method": "modelopt", "quant_algo": "FP8"}}
            (snap / "hf_quant_config.json").write_text(json.dumps({"quantization": {"format": "float8"}}))
        if quant == "yarned":
            cfg = dict(YARNED)
            (snap / "config.json.pre-yarn").write_text(json.dumps(NATIVE))
        (snap / "config.json").write_text(json.dumps(cfg))
        (snap / "chat_template.jinja").write_text(TEMPLATE)
        self.snap = snap

    def run(self, *args):
        env = {"PATH": f"{self.bin}:/usr/bin:/bin", "HOME": str(self.home), "QWEN38_MEMINFO": str(self.t / "meminfo")}
        r = subprocess.run(["bash", str(self.d / "prepare-lane.sh"), *args], capture_output=True,
                           text=True, timeout=60, env=env)
        calls = (self.t / "calls").read_text().splitlines() if (self.t / "calls").exists() else []
        hfenv = (self.t / "hfenv").read_text().splitlines() if (self.t / "hfenv").exists() else []
        self.modrev = (self.t / "modrev").read_text().splitlines() if (self.t / "modrev").exists() else []
        return r.returncode, r.stdout + r.stderr, calls, hfenv

    def config(self):
        return json.loads((self.snap / "config.json").read_text())


class TheLoadsFetchGoesIntoTheWaitingUnitsCache(unittest.TestCase):
    """Defect 2: prepare-lane.sh's `switch-model.sh <target> --fetch` passes the cache the unit
    that waits mounts, so a cache moved by a re-run of install with HF_CACHE= gets the download
    and the Load does not loop (the switch would read the one in place)."""

    def test_27b(self):
        b = Box("27b", "stock")
        rc, out, calls, hfenv = b.run("27b", "stock")
        self.assertEqual(rc, 0, out)
        self.assertIn("switch-model.sh stock --fetch", calls)
        self.assertEqual(hfenv, [str(b.hf_wait)], out)

    def test_flash(self):
        b = Box("flash", "flash")
        rc, out, calls, hfenv = b.run("flash", "flash")
        self.assertEqual(rc, 0, out)
        self.assertIn("switch-model.sh flash --fetch", calls)
        self.assertEqual(hfenv, [str(b.hf_wait)], out)


class ThePrepareLeavesThePatchToThePose(unittest.TestCase):
    """Seventh pass of the v1.52.0 reviews (2026-11-19): the prepare used to run the switch's steps after
    the fetch (patch_served_checkpoint) itself, against the unit that waits, BEFORE apply_pending put it
    in place. When the waiting unit serves the same snapshot as the one in place with another window
    (an update whose image moved, with CONTEXT_MODE changed at the same run), that patch touched the
    config the serving lane still reads: a restart of the lane before the pose left a 1M lane that no
    longer loads. The patch now runs in apply_pending, once the unit is in place (cockpit.py's
    patch_waiting_checkpoint), so the prepare leaves the cached checkpoint exactly as the fetch left it."""

    def test_a_1m_wait_leaves_the_config_and_template_to_the_pose(self):
        b = Box("27b", "stock", window="1m")
        rc, out, calls, _ = b.run("27b", "stock")
        self.assertEqual(rc, 0, out)
        tc = b.config().get("text_config", {})
        self.assertEqual(tc.get("max_position_embeddings"), 262144, out)   # not patched here anymore
        self.assertFalse((b.cfg / "chat-template-sglang.jinja").exists(), out)

    def test_a_native_wait_leaves_the_yarned_config_untouched(self):
        b = Box("27b", "stock", window="native", quant="yarned")
        rc, out, _, _ = b.run("27b", "stock")
        self.assertEqual(rc, 0, out)
        tc = b.config().get("text_config", {})
        self.assertEqual(tc.get("max_position_embeddings"), 1010000, out)  # still yarned: the pose restores it
        self.assertTrue((b.snap / "config.json.pre-yarn").exists(), "the restore did not run in the prepare")

    def test_an_arete_wait_leaves_the_kv_declaration_to_the_pose(self):
        b = Box("27b", "arete", window="1m", model=PINS["ARETE_REPO"], quant="arete")
        rc, out, _, _ = b.run("27b", "arete")
        self.assertEqual(rc, 0, out)
        self.assertNotIn("kv_cache_scheme", b.config().get("quantization_config", {}), out)

    def test_a_flash_wait_leaves_its_template_to_the_pose(self):
        b = Box("flash", "flash")
        rc, out, _, _ = b.run("flash", "flash")
        self.assertEqual(rc, 0, out)
        self.assertFalse((b.cfg / "chat-template-flashnext.jinja").exists(), out)

    def test_without_a_wait_the_prepare_patches_nothing(self):
        # no update waits: the switch step of the journey runs the steps, against the unit in place
        b = Box("27b", "stock", window="1m", pending=False)
        rc, out, _, hfenv = b.run("27b", "stock")
        self.assertEqual(rc, 0, out)
        self.assertEqual(hfenv, [str(b.hf_inplace)], out)   # no wait: the cache of the unit in place
        tc = b.config().get("text_config", {})
        self.assertEqual(tc.get("max_position_embeddings"), 262144, out)
        self.assertFalse((b.cfg / "chat-template-sglang.jinja").exists(), out)


class ThePrepareFetchesTheRevisionTheWaitingUnitNames(unittest.TestCase):
    """Seventh pass of the v1.52.0 reviews (2026-11-19): when an update waits, the prepare fetched and
    patched the PIN's revision, not the one the waiting unit names in its --revision. With the pin moved
    since the wait was posed (a newer checkout without a re-run of ./install.sh), or a MODEL_REV= install,
    each Load fetched the pin's revision while the wait kept asking for its own, and apply_pending
    refused forever. The prepare now reads the revision in the waiting unit and passes it to the fetch
    (MODEL_REV=, honored by switch-model.sh --fetch)."""

    OLD, NEW = "1" * 40, "2" * 40   # the revision the waiting unit names, the pin's (moved) one

    def test_27b_the_waiting_units_revision_is_fetched(self):
        b = Box("27b", "stock", window="1m", pend_rev=self.OLD, pin_rev=self.NEW)
        rc, out, calls, hfenv = b.run("27b", "stock")
        self.assertEqual(rc, 0, out)
        self.assertEqual(b.modrev, [self.OLD], "the fetch was given the waiting unit's revision, not the pin's")
        self.assertIn("switch-model.sh stock --fetch", calls)
        tc = b.config().get("text_config", {})
        self.assertEqual(tc.get("max_position_embeddings"), 262144, out)
        # the config is left as the fetch left it: the patch runs at the pose (apply_pending), not here

    def test_flash_the_waiting_launchers_revision_is_fetched(self):
        b = Box("flash", "flash", pend_rev=self.OLD, pin_rev=self.NEW)
        rc, out, calls, hfenv = b.run("flash", "flash")
        self.assertEqual(rc, 0, out)
        self.assertEqual(b.modrev, [self.OLD], "the fetch was given the waiting launcher's revision")
        self.assertIn("switch-model.sh flash --fetch", calls)

    def test_without_a_wait_the_pins_revision_is_fetched(self):
        b = Box("27b", "stock", window="1m", pending=False)
        rc, out, calls, hfenv = b.run("27b", "stock")
        self.assertEqual(rc, 0, out)
        self.assertEqual(b.modrev, [PINS["STOCK_REV"]], out)   # no wait: the pin, as the switch reads it


if __name__ == "__main__":
    unittest.main()
