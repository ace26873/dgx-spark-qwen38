#!/usr/bin/env python3
"""The cockpit follows the Liquid AI LFM2.5 24B-A2B lane as an SGLang text lane, with its own facts where the 27B's
would be wrong.

Its engine is SGLang on the 27B lane's image, on the text port behind the proxy: the SGLang belts are its own (the canary,
the pool guard, the autoheal), its target is read from its unit's --model-path as the 27B's is, its draft is part of what
it needs to start, the 27B's pinned pool is not its, and System One, which its template cannot answer, is said
unavailable there without asking the proxy. The module is imported with its environment pointed at a throwaway box.
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
U = "qwen38-lfm25.service"
INSTALLER = (REPO / "install-lfm25.sh").read_text()
PIN = {k: re.search(rf'^{k}="([^"]+)"', INSTALLER, re.M).group(1)
       for k in ("LFM25_IMAGE", "LFM25_REPO", "LFM25_REV", "LFM25_DRAFT_REPO", "LFM25_DRAFT_REV")}
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
        cls.box = Path(tempfile.mkdtemp(prefix="cockpit-lfm25-"))
        os.environ.update(COCKPIT_DRY_RUN="1", COCKPIT_CONFIG_DIR=str(cls.box), COCKPIT_REPO_DIR=str(REPO),
                          COCKPIT_PORT="0", COCKPIT_AGENT_PORT="0", COCKPIT_AUTOHEAL="0",
                          COCKPIT_ENGINE="http://127.0.0.1:1", COCKPIT_PROXY="http://127.0.0.1:1")
        (cls.box / "api-key").write_text("test-key\n")
        sys.path.insert(0, str(DASH))
        spec = importlib.util.spec_from_file_location("cockpit_lfm25_under_test", DASH / "cockpit.py")
        cls.ck = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.ck)
        # its unit as its installer renders it (--stage), in this box's own cache
        cls.hf = cls.box / "hf"
        cls.unit = cls.box / U
        home = cls.box / "home"
        (home / ".config" / "qwen38").mkdir(parents=True)
        r = subprocess.run(["bash", str(REPO / "install-lfm25.sh"), "--stage", str(cls.unit)], capture_output=True, text=True,
                           env={"PATH": "/usr/bin:/bin", "HOME": str(home), "HF_CACHE": str(cls.hf),
                                "LANES_UNIT_DIR": str(cls.box / "units"), "QWEN38_MEMORY_CLASS": "128"}, timeout=120)
        assert r.returncode == 0, r.stdout + r.stderr
        cls.ck.LFM25_UNIT_PATH = cls.unit
        cls.ck.UNIT_PATHS[U] = cls.unit
        cls.real_run = cls.ck.run

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.box, ignore_errors=True)

    def setUp(self):
        self.ck.run = self.real_run
        self.ck.UNIT_TARGET_CACHE.clear()

    def snap(self, repo, rev):
        return self.hf / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / rev

    def whole(self):
        """Its checkpoint (an index and its two shards) and its draft (one weights file) in the cache its unit mounts."""
        m = self.snap(PIN["LFM25_REPO"], PIN["LFM25_REV"])
        m.mkdir(parents=True, exist_ok=True)
        (m / "config.json").write_text("{}")
        (m / "model.safetensors.index.json").write_text(
            '{"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}')
        for f in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
            (m / f).write_bytes(b"x")
        d = self.snap(PIN["LFM25_DRAFT_REPO"], PIN["LFM25_DRAFT_REV"])
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text("{}")
        (d / "model.safetensors").write_bytes(b"x")
        return m, d

    def image_here(self, here=True):
        img = PIN["LFM25_IMAGE"]
        self.ck.run = lambda argv, **kw: ("sha256:abc\n" if here and argv[:3] == ["docker", "image", "inspect"]
                                          and argv[-1] == img else "")


class ItsUnit(Base):
    def test_its_target_is_read_from_its_model_path_as_the_27bs(self):
        self.assertEqual(self.ck.unit_target(U), {"model": PIN["LFM25_REPO"], "target": "lfm25"})

    def test_it_is_an_sglang_text_lane_with_every_belt(self):
        lc = self.ck.lc
        self.assertIn(U, lc.ENGINE_UNITS)
        self.assertIn(U, lc.TEXT_UNITS, "the canary, the pool guard and the autoheal are SGLang's, and its")
        self.assertEqual(lc.LANE_PROVIDER[U], "lfm25")
        self.assertEqual(self.ck.UNIT2CONT[U], "qwen38-lfm25")
        self.assertIn("qwen38-lfm25", self.ck.SGL_CONTAINERS, "its scheduler lines are SGLang's")
        self.assertNotIn(U, self.ck.OTHER_TEXT_UNITS)
        self.assertEqual(self.ck.TEXT_BOOT_DEFAULT_S[U], 290.0)
        self.assertTrue(lc.blocked_reasons("unit", {"unit": U, "verb": "start"}, {"qwen38-sglang.service": "ready"}))
        self.assertTrue(lc.blocked_reasons("unit", {"unit": "qwen38-sglang.service", "verb": "start"}, {U: "ready"}))

    def test_it_is_a_switch_target_a_unit_the_page_may_drive_and_a_lane_it_prepares(self):
        self.assertIn("lfm25", self.ck.ACTIONS["switch"]["params"]["target"])
        self.assertIn(U, self.ck.SERVING_UNITS)
        self.assertIn(U, self.ck.JOURNAL_UNITS)
        self.assertEqual(self.ck.target_lane("lfm25"), "lfm25")
        self.assertEqual(self.ck.target_unit("lfm25"), self.ck.LFM25_UNIT_PATH)
        self.assertEqual(self.ck.ACTIONS["prepare"]["argv"]({"target": "lfm25"})[-1], "lfm25")
        self.assertEqual(self.ck.TARGET_PIN["lfm25"], "LFM25_REV")
        self.assertEqual(self.ck.TARGET_DRAFT_PIN["lfm25"], "LFM25_DRAFT_REV")
        self.assertEqual(self.ck.rg.PIN_MODELS["LFM25_REV"], PIN["LFM25_REPO"])
        self.assertEqual(self.ck.rg.PIN_MODELS["LFM25_DRAFT_REV"], PIN["LFM25_DRAFT_REPO"])
        self.assertIn("install-lfm25.sh", self.ck.rg.PIN_FILES)

    def test_its_engine_is_the_27b_lanes(self):
        eng = self.ck.en.ENGINES
        self.assertEqual(eng[U]["pin"], PIN["LFM25_IMAGE"])
        self.assertEqual({k: eng[U][k] for k in ("engine", "version", "reports", "pin")},
                         {k: eng["qwen38-sglang.service"][k] for k in ("engine", "version", "reports", "pin")})


class ItStartsOnlyWhole(Base):
    def setUp(self):
        super().setUp()
        self.image_here()

    def test_it_is_prepared_once_the_image_its_unit_names_is_here(self):
        self.assertTrue(self.ck.lane_prepared("lfm25"))
        self.image_here(False)
        self.assertFalse(self.ck.lane_prepared("lfm25"))

    def test_whole_it_can_start(self):
        self.whole()
        self.assertTrue(self.ck.lane_startable("lfm25"))

    def test_its_checkpoint_or_its_draft_missing_holds_the_start_back(self):
        """The unit starts offline (HF_HUB_OFFLINE=1): a draft not there crash-loops it as surely as a checkpoint."""
        for what in ("a shard of the checkpoint", "the draft"):
            with self.subTest(what):
                m, d = self.whole()
                (m / "model-00002-of-00002.safetensors" if what.startswith("a shard") else d / "model.safetensors").unlink()
                try:
                    self.assertFalse(self.ck.lane_startable("lfm25"))
                finally:
                    self.whole()


class TheTwentySevenBsFactsAreNotItsOwn(Base):
    def test_the_27bs_pinned_pool_is_not_its(self):
        """A boot of it was measured against the --max-total-tokens the 27B's unit pins: any unit but the flash's read
        the 27B's invocation."""
        u27 = self.box / "qwen38-sglang.service"
        u27.write_text("ExecStart=/bin/bash -c 'exec docker run x python3 -m sglang.launch_server --model-path m "
                       "--max-total-tokens 123456 --port 30000'\n")
        saved = dict(self.ck.INSTALLED_INVOCATION)
        self.ck.INSTALLED_INVOCATION["27b"] = u27
        try:
            self.assertEqual(self.ck.pinned_pool("qwen38-sglang.service"), 123456, "the 27B's own, read as before")
            self.assertIsNone(self.ck.pinned_pool(U))
        finally:
            self.ck.INSTALLED_INVOCATION.clear()
            self.ck.INSTALLED_INVOCATION.update(saved)

    def lane_serving(self, state="ready"):
        with self.ck.LIFE_LOCK:
            saved = dict(self.ck.LIFE.get("states") or {})
            self.ck.LIFE["states"] = {U: state}
        self.addCleanup(lambda: self.ck.LIFE.__setitem__("states", saved))

    def test_with_no_engine_info_its_name_is_its_own_not_the_27bs(self):
        self.lane_serving()
        with self.ck.STATE_LOCK:
            saved = self.ck.STATE.pop("engine_info", None)
        try:
            self.assertEqual(self.ck.served_model_name(), "lfm2.5-24b-a2b")
        finally:
            if saved is not None:
                with self.ck.STATE_LOCK:
                    self.ck.STATE["engine_info"] = saved

    def test_system_one_is_said_unavailable_without_asking_the_proxy(self):
        """The proxy refuses it on this lane (v6.60), but its schema refuses the cockpit's probe first, with the 422
        that reads as "it serves": the cockpit says why before any probe."""
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
        self.assertIn("the Liquid AI LFM2.5 24B-A2B lane is serving", out["reason"])
        self.assertIn("begins every answer with its reasoning", out["reason"])
        self.assertEqual(asked, [])


if __name__ == "__main__":
    unittest.main()
