#!/usr/bin/env python3
"""The cockpit follows the Inkling 2 Mini lane as an SGLang text lane, with its own facts where the 27B's would be wrong.

Its engine is SGLang on the image its cookbook gives the model, on the text port behind the proxy: the SGLang belts are
its own (the canary, the pool guard, the autoheal), its target is read from its unit's --model-path as the 27B's is, its
checkpoint (the DSpark draft head inside) is what it needs to start, the 27B's pinned pool is not its, and System One,
which it cannot answer, is said unavailable there without asking the proxy. The module is imported with its environment
pointed at a throwaway box.
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
U = "qwen38-inkling.service"
INSTALLER = (REPO / "install-inkling.sh").read_text()
PIN = {k: re.search(rf'^{k}="([^"]+)"', INSTALLER, re.M).group(1) for k in ("INKLING_IMAGE", "INKLING_REPO", "INKLING_REV")}
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
        cls.box = Path(tempfile.mkdtemp(prefix="cockpit-inkling-"))
        os.environ.update(COCKPIT_DRY_RUN="1", COCKPIT_CONFIG_DIR=str(cls.box), COCKPIT_REPO_DIR=str(REPO),
                          COCKPIT_PORT="0", COCKPIT_AGENT_PORT="0", COCKPIT_AUTOHEAL="0",
                          COCKPIT_ENGINE="http://127.0.0.1:1", COCKPIT_PROXY="http://127.0.0.1:1")
        (cls.box / "api-key").write_text("test-key\n")
        sys.path.insert(0, str(DASH))
        spec = importlib.util.spec_from_file_location("cockpit_inkling_under_test", DASH / "cockpit.py")
        cls.ck = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.ck)
        # its unit as its installer renders it (--stage), in this box's own cache
        cls.hf = cls.box / "hf"
        cls.unit = cls.box / U
        home = cls.box / "home"
        (home / ".config" / "qwen38").mkdir(parents=True)
        r = subprocess.run(["bash", str(REPO / "install-inkling.sh"), "--stage", str(cls.unit)], capture_output=True,
                           text=True, env={"PATH": "/usr/bin:/bin", "HOME": str(home), "HF_CACHE": str(cls.hf),
                                           "LANES_UNIT_DIR": str(cls.box / "units"), "QWEN38_MEMORY_CLASS": "128"},
                           timeout=120)
        assert r.returncode == 0, r.stdout + r.stderr
        cls.ck.INKLING_UNIT_PATH = cls.unit
        cls.ck.UNIT_PATHS[U] = cls.unit
        cls.real_run = cls.ck.run

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.box, ignore_errors=True)

    def setUp(self):
        self.ck.run = self.real_run
        self.ck.UNIT_TARGET_CACHE.clear()

    def whole(self):
        """Its checkpoint as its repository holds it: an index, its one shard, and the draft head beside them."""
        m = self.hf / "hub" / f"models--{PIN['INKLING_REPO'].replace('/', '--')}" / "snapshots" / PIN["INKLING_REV"]
        m.mkdir(parents=True, exist_ok=True)
        (m / "config.json").write_text("{}")
        (m / "model.safetensors.index.json").write_text('{"weight_map": {"a": "model-00001-of-00001.safetensors"}}')
        for f in ("model-00001-of-00001.safetensors", "mtp.safetensors"):
            (m / f).write_bytes(b"x")
        return m

    def image_here(self, here=True):
        img = PIN["INKLING_IMAGE"]
        self.ck.run = lambda argv, **kw: ("sha256:abc\n" if here and argv[:3] == ["docker", "image", "inspect"]
                                          and argv[-1] == img else "")


class ItsUnit(Base):
    def test_its_target_is_read_from_its_model_path_as_the_27bs(self):
        self.assertEqual(self.ck.unit_target(U), {"model": PIN["INKLING_REPO"], "target": "inkling"})

    def test_it_is_an_sglang_text_lane_with_every_belt(self):
        lc = self.ck.lc
        self.assertIn(U, lc.ENGINE_UNITS)
        self.assertIn(U, lc.TEXT_UNITS, "the canary, the pool guard and the autoheal are SGLang's, and its")
        self.assertEqual(lc.LANE_PROVIDER[U], "inkling")
        self.assertEqual(self.ck.UNIT2CONT[U], "qwen38-inkling")
        self.assertIn("qwen38-inkling", self.ck.SGL_CONTAINERS, "its scheduler lines are SGLang's")
        self.assertNotIn(U, self.ck.OTHER_TEXT_UNITS)
        self.assertEqual(self.ck.TEXT_BOOT_DEFAULT_S[U], 230.0)
        self.assertTrue(lc.blocked_reasons("unit", {"unit": U, "verb": "start"}, {"qwen38-sglang.service": "ready"}))
        self.assertTrue(lc.blocked_reasons("unit", {"unit": "qwen38-sglang.service", "verb": "start"}, {U: "ready"}))

    def test_it_is_a_switch_target_a_unit_the_page_may_drive_and_a_lane_it_prepares(self):
        self.assertIn("inkling", self.ck.ACTIONS["switch"]["params"]["target"])
        self.assertIn(U, self.ck.SERVING_UNITS)
        self.assertIn(U, self.ck.JOURNAL_UNITS)
        self.assertEqual(self.ck.target_lane("inkling"), "inkling")
        self.assertEqual(self.ck.target_unit("inkling"), self.ck.INKLING_UNIT_PATH)
        self.assertEqual(self.ck.ACTIONS["prepare"]["argv"]({"target": "inkling"})[-1], "inkling")
        self.assertEqual(self.ck.TARGET_PIN["inkling"], "INKLING_REV")
        self.assertNotIn("inkling", self.ck.TARGET_DRAFT_PIN, "its draft head is in its checkpoint")
        self.assertEqual(self.ck.rg.PIN_MODELS["INKLING_REV"], PIN["INKLING_REPO"])
        self.assertIn("install-inkling.sh", self.ck.rg.PIN_FILES)

    def test_its_engine_is_sglang_on_its_pinned_image(self):
        e = self.ck.en.ENGINES[U]
        self.assertEqual(e["pin"], PIN["INKLING_IMAGE"])
        self.assertEqual(e["engine"], "SGLang")


class ItStartsOnlyWhole(Base):
    def setUp(self):
        super().setUp()
        self.image_here()

    def test_it_is_prepared_once_the_image_its_unit_names_is_here(self):
        self.assertTrue(self.ck.lane_prepared("inkling"))
        self.image_here(False)
        self.assertFalse(self.ck.lane_prepared("inkling"))

    def test_whole_it_can_start_and_a_missing_shard_holds_it_back(self):
        """The unit starts offline (HF_HUB_OFFLINE=1): a shard not there crash-loops it."""
        m = self.whole()
        self.assertTrue(self.ck.lane_startable("inkling"))
        (m / "model-00001-of-00001.safetensors").unlink()
        try:
            self.assertFalse(self.ck.lane_startable("inkling"))
        finally:
            self.whole()


class TheTwentySevenBsFactsAreNotItsOwn(Base):
    def test_the_27bs_pinned_pool_is_not_its(self):
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
            self.assertEqual(self.ck.served_model_name(), "inkling-2-mini")
        finally:
            if saved is not None:
                with self.ck.STATE_LOCK:
                    self.ck.STATE["engine_info"] = saved


if __name__ == "__main__":
    unittest.main()
