#!/usr/bin/env python3
"""The cockpit follows the Liquid AI LFM3 lane as a vLLM lane read from the table of the b12x family's (TABLE_LANES):
the same routes, /metrics and engine panel, its own name, its own refusals, no drafter, and its boot read from vLLM's
own lines with markers of its own (its KV cache sized after its CUDA graphs, no drafter to name). The module is
imported with its environment pointed at a throwaway box.
"""
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
DASH = HERE.parents[1]
REPO = HERE.parents[2]
U = "qwen38-lfm3.service"
INSTALLER = (REPO / "install-lfm3.sh").read_text()
PIN = {k: re.search(rf'^{k}="([^"]+)"', INSTALLER, re.M).group(1) for k in ("LFM3_IMAGE", "LFM3_REPO", "LFM3_REV")}
# the lines of a boot of this lane that the stages read, as vLLM's nightly 60bd6dd wrote them on the reference box
# (2026-11-16), each up to the next
BOOT = [   # the reference box's first boot, 2026-11-16, its lines as vLLM wrote them (cut)
    "(EngineCore pid=173) INFO 11-16 14:35:04 [core.py:153] Initializing a V1 LLM engine (v0.34.0rc1.dev571+g60bd6dd2e) with config: ...",
    "Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]",
    "Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  1.05it/s]",
    "(EngineCore pid=173) INFO 11-16 14:35:07 [default_loader.py:539] Loading weights took 0.95 seconds",
    "(EngineCore pid=173) INFO 11-16 14:35:07 [model_runner.py:449] Model loading took 1.86 GiB memory and 2.090623 seconds",
    "Capturing CUDA graphs (PIECEWISE):   0%|          | 0/51 [00:00<?, ?it/s]Capturing CUDA graphs (PIECEWISE):   4%|",
    "(EngineCore pid=173) INFO 11-16 14:36:49 [kv_cache_utils.py:1134] GPU KV cache size: 5,239,455 tokens, Maximum concurrency for 262,144 tokens per request: 19.99x",
    "(EngineCore pid=173) INFO 11-16 14:37:14 [core.py:436] init engine (profile, create kv cache, warmup model) took 127.54 s (compilation: 11.06 s)",
    "INFO:     Application startup complete.",
]
ENV_BEFORE = {}


def setUpModule():
    ENV_BEFORE.update(os.environ)


def tearDownModule():
    for name in set(os.environ) - set(ENV_BEFORE):
        del os.environ[name]
    os.environ.update(ENV_BEFORE)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.box = Path(tempfile.mkdtemp(prefix="cockpit-lfm3-"))
        os.environ.update(COCKPIT_DRY_RUN="1", COCKPIT_CONFIG_DIR=str(cls.box), COCKPIT_REPO_DIR=str(REPO),
                          COCKPIT_PORT="0", COCKPIT_AGENT_PORT="0", COCKPIT_AUTOHEAL="0",
                          COCKPIT_ENGINE="http://127.0.0.1:1", COCKPIT_PROXY="http://127.0.0.1:1")
        (cls.box / "api-key").write_text("test-key\n")
        sys.path.insert(0, str(DASH))
        spec = importlib.util.spec_from_file_location("cockpit_lfm3_under_test", DASH / "cockpit.py")
        cls.ck = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.ck)
        # its unit as its installer renders it (--stage)
        cls.hf = cls.box / "hf"
        cls.unit = cls.box / U
        home = cls.box / "home"
        (home / ".config" / "qwen38").mkdir(parents=True)
        r = subprocess.run(["bash", str(REPO / "install-lfm3.sh"), "--stage", str(cls.unit)], capture_output=True, text=True,
                           env={"PATH": "/usr/bin:/bin", "HOME": str(home), "HF_CACHE": str(cls.hf),
                                "LANES_UNIT_DIR": str(cls.box / "units"), "QWEN38_MEMORY_CLASS": "128"}, timeout=120)
        assert r.returncode == 0, r.stdout + r.stderr
        cls.ck.LFM3_UNIT_PATH = cls.unit
        cls.real_run = cls.ck.run

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.box, ignore_errors=True)

    def setUp(self):
        self.ck.run = self.real_run
        self.ck.UNIT_TARGET_CACHE.clear()

    def lane_serving(self, state="ready"):
        with self.ck.LIFE_LOCK:
            saved = dict(self.ck.LIFE.get("states") or {})
            self.ck.LIFE["states"] = {U: state}
        self.addCleanup(lambda: self.ck.LIFE.__setitem__("states", saved))


class ItsPlace(Base):
    def test_a_vllm_lane_of_the_table_with_no_drafter_and_none_of_the_sglang_belts(self):
        lc = self.ck.lc
        self.assertIn(U, lc.ENGINE_UNITS)
        self.assertNotIn(U, lc.TEXT_UNITS, "the canary, the pool guard and the autoheal are SGLang's")
        self.assertEqual(lc.LANE_PROVIDER[U], "lfm3")
        self.assertEqual(self.ck.TABLE_LANES[U], ("qwen38-lfm3", "lfm3", "lfm3-1.7b", "nvfp4"))
        self.assertNotIn(U, self.ck.NEMOTRON_LANES, "no b12x")
        self.assertNotIn(U, self.ck.B12X_SPEC, "no drafter")
        self.assertIn(U, self.ck.VLLM_UNITS, "its start reads the vLLM lanes' key middleware")
        self.assertIn(U, self.ck.OTHER_TEXT_UNITS)
        self.assertEqual(self.ck.nemotron_unit_serving([{"id": "lfm3-1.7b"}]), U)
        self.assertEqual(self.ck.nemotron_unit_path(U), self.unit)
        self.assertTrue(lc.blocked_reasons("unit", {"unit": U, "verb": "start"}, {"qwen38-sglang.service": "ready"}))

    def test_it_is_a_switch_target_a_unit_the_page_may_drive_and_a_lane_it_prepares(self):
        self.assertIn("lfm3", self.ck.ACTIONS["switch"]["params"]["target"])
        self.assertIn(U, self.ck.SERVING_UNITS)
        self.assertIn(U, self.ck.JOURNAL_UNITS)
        self.assertIn("qwen38-lfm3", self.ck.CONTAINERS)
        self.assertEqual(self.ck.target_lane("lfm3"), "lfm3")
        self.assertEqual(self.ck.target_unit("lfm3"), self.ck.LFM3_UNIT_PATH)
        self.assertEqual(self.ck.ACTIONS["prepare"]["argv"]({"target": "lfm3"})[-1], "lfm3")
        self.assertEqual(self.ck.TARGET_PIN["lfm3"], "LFM3_REV")
        self.assertEqual(self.ck.rg.PIN_MODELS["LFM3_REV"], PIN["LFM3_REPO"])
        self.assertEqual(self.ck.en.ENGINES[U]["pin"], PIN["LFM3_IMAGE"])

    def test_its_journal_goes_in_the_diagnostics_bundle(self):
        src = (DASH / "cockpit.py").read_text()
        self.assertIn('(tdp / "journal-lfm3.txt").write_text(run(["journalctl", "-u", LFM3_UNIT', src)

    def test_its_target_is_read_from_its_unit(self):
        self.assertEqual(self.ck.unit_target(U), {"model": PIN["LFM3_REPO"], "target": "lfm3"})

    def test_it_is_prepared_once_the_image_its_unit_names_is_here(self):
        img = PIN["LFM3_IMAGE"]
        self.ck.run = lambda argv, **kw: "sha256:abc\n" if argv[:3] == ["docker", "image", "inspect"] and argv[-1] == img else ""
        self.assertTrue(self.ck.lane_prepared("lfm3"))
        self.ck.run = lambda argv, **kw: ""
        self.assertFalse(self.ck.lane_prepared("lfm3"))


class ItsBoot(Base):
    def test_each_stage_of_its_boot_is_named_for_what_it_does(self):
        lc = self.ck.lc
        want = [("init", "starting vLLM"), ("loading-weights", "shard 0 of 1 (0 %)"),
                ("loading-weights", "shard 1 of 1 (100 %)"), ("loading-weights", "shard 1 of 1 (100 %)"),
                ("allocating-kv", "compiling the model and tuning its NVFP4 kernels"),
                ("capturing-graphs", "capturing the CUDA graphs"), ("warming-up", "sizing the KV cache to its share"),
                ("warming-up", "starting the API"), ("warming-up", "ready")]
        for n, (stage, detail) in enumerate(want, 1):
            with self.subTest(line=n):
                b = lc.parse_nemotron_boot_log(BOOT[:n], lc.LFM3_MARKERS)
                self.assertEqual((b["stage"], b["detail"]), (stage, detail))
        self.assertTrue(lc.parse_nemotron_boot_log(BOOT, lc.LFM3_MARKERS)["fired_up"])
        # read with the Nemotron markers, its weights' end named a drafter it does not have
        self.assertNotIn("drafter", " ".join(d for _, _, d in lc.LFM3_MARKERS))

    def test_its_state_reads_its_own_container_with_its_own_markers(self):
        seen = []
        self.ck.run = lambda argv, **kw: "abc\n" if argv[:2] == ["docker", "ps"] and "name=^qwen38-lfm3$" in argv else ""
        saved = self.ck._nemotron_log, self.ck.kolibri_healthy
        self.ck._nemotron_log = lambda container="": (seen.append(container), BOOT[:5])[1]
        self.ck.kolibri_healthy = lambda: False
        try:
            st, boot, running = self.ck.nemotron_engine_state(U, active="active", sub="running", prev_state="starting",
                                                              enter_key="7")
        finally:
            self.ck._nemotron_log, self.ck.kolibri_healthy = saved
        self.assertTrue(running)
        self.assertEqual(seen, ["qwen38-lfm3"])
        self.assertEqual((boot["stage"], boot["detail"]), ("allocating-kv", "compiling the model and tuning its NVFP4 kernels"))


class WhatItCannotDo(Base):
    def test_its_refusals_are_its_own_and_say_its_name(self):
        self.lane_serving()
        self.assertTrue(self.ck.nemotron_serving())
        self.assertEqual(self.ck.b12x_lane_name(), "the Liquid AI LFM3 lane")
        for action in ("flush_cache", "abort_all", "fit_opencode"):
            with self.subTest(action):
                self.assertEqual(self.ck.b12x_cannot(action), self.ck.LFM3_CANNOT[action])
                self.assertTrue(self.ck.LFM3_CANNOT[action].startswith("The Liquid AI LFM3 lane"))

    def test_the_canary_leaves_it_to_its_own_health(self):
        self.lane_serving()
        out = self.ck.collect_canary()
        self.assertTrue(out["skipped"])
        self.assertEqual(out["why"], "the Liquid AI LFM3 lane is checked by vLLM's own /health, not by a generation")

    def test_system_one_is_said_unavailable_without_asking_the_proxy(self):
        self.lane_serving()
        asked = []
        saved = self.ck.urllib.request.urlopen
        self.ck.urllib.request.urlopen = lambda req, timeout=None: asked.append(req.full_url)
        self.ck.SYSTEMONE_CACHE.update(data=None, ts=0.0, key=None)
        try:
            out = self.ck.systemone_available(max_age=0)
        finally:
            self.ck.urllib.request.urlopen = saved
        self.assertFalse(out["available"])
        self.assertIn("the Liquid AI LFM3 lane is serving", out["reason"])
        self.assertEqual(asked, [])

    def test_with_no_engine_info_its_name_is_its_own(self):
        self.lane_serving()
        with self.ck.STATE_LOCK:
            saved = self.ck.STATE.pop("engine_info", None)
        try:
            self.assertEqual(self.ck.served_model_name(), "lfm3-1.7b")
        finally:
            if saved is not None:
                with self.ck.STATE_LOCK:
                    self.ck.STATE["engine_info"] = saved


if __name__ == "__main__":
    unittest.main()
