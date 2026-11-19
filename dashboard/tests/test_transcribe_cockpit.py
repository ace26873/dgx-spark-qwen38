#!/usr/bin/env python3
"""The cockpit follows the NVIDIA Nemotron Transcribe lane on its own facts (v1.51).

Its engine is llama.cpp's own server, the Unsloth GGUF lane's image, on a port of its own (30036 by default, its unit's
--port): its /health open and generating nothing, its audio routes behind the box's key, and nothing of it on the text
port, where the SGLang belts and the llama.cpp text lane's reads look. The engine here answers /health as llama-server
does; its boot is the log of llama.cpp's image the Unsloth GGUF lane's fixture holds (the same server, the same
milestones). The module is imported with its environment pointed at a throwaway box.
"""
import http.server
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
DASH = HERE.parents[1]
REPO = HERE.parents[2]
BOOT = (HERE.parent / "fixtures" / "flash-unsloth-boot.log").read_text().splitlines()
U = "qwen38-transcribe.service"
INSTALLER = (REPO / "install-transcribe.sh").read_text()
REPO_ID = re.search(r'^TRANSCRIBE_REPO="([^"]+)"', INSTALLER, re.M).group(1)
REV = re.search(r'^TRANSCRIBE_REV="([0-9a-f]{40})"', INSTALLER, re.M).group(1)
FILES = [re.search(rf'^{v}="([^"=]+)=', INSTALLER, re.M).group(1) for v in ("TRANSCRIBE_FILE", "TRANSCRIBE_MMPROJ")]
ENV_BEFORE = {}


def setUpModule():
    ENV_BEFORE.update(os.environ)


def tearDownModule():
    for name in set(os.environ) - set(ENV_BEFORE):
        del os.environ[name]
    os.environ.update(ENV_BEFORE)


class Health(http.server.BaseHTTPRequestHandler):
    """llama-server's /health on the lane's own port: 200 once its model is loaded, 503 before."""
    asked: list = []
    loaded = True

    def log_message(self, *a):
        pass

    def do_GET(self):
        Health.asked.append(self.path)
        code = 200 if self.path == "/health" and Health.loaded else 503
        out = b'{"status":"ok"}' if code == 200 else b'{"error":{"code":503,"message":"Loading model"}}'
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.box = Path(tempfile.mkdtemp(prefix="cockpit-transcribe-"))
        cls.lane = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Health)
        cls.lane.daemon_threads = True
        threading.Thread(target=cls.lane.serve_forever, daemon=True).start()
        # the text port answers nothing: the lane is never asked there
        os.environ.update(COCKPIT_DRY_RUN="1", COCKPIT_CONFIG_DIR=str(cls.box), COCKPIT_REPO_DIR=str(REPO),
                          COCKPIT_PORT="0", COCKPIT_AGENT_PORT="0", COCKPIT_AUTOHEAL="0",
                          COCKPIT_ENGINE="http://127.0.0.1:1", COCKPIT_PROXY="http://127.0.0.1:1")
        os.environ.pop("COCKPIT_TRANSCRIBE", None)
        (cls.box / "api-key").write_text("test-key\n")
        sys.path.insert(0, str(DASH))
        spec = importlib.util.spec_from_file_location("cockpit_transcribe_under_test", DASH / "cockpit.py")
        cls.ck = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.ck)
        # its unit as its installer renders it, the files it names in this box's own cache, its port the fake's
        cls.hf = cls.box / "hf"
        snap = cls.hf / "hub" / f"models--{REPO_ID.replace('/', '--')}" / "snapshots" / REV
        cls.files = [snap / f for f in FILES]
        cls.check = cls.box / "llamacpp-files.sh"
        in_box = f"/root/.cache/huggingface/hub/models--{REPO_ID.replace('/', '--')}/snapshots/{REV}"
        cls.unit = cls.box / U
        cls.unit.write_text(
            "# its header names -m and --port in words: --port 9 is no port of its engine\n"
            f"ExecStart=/bin/bash -c 'check={cls.check}; [ -f \"$$check\" ] || exit 78; \\\n"
            f"  bash \"$$check\" {U} transcribe {' '.join(map(str, cls.files))} || exit $$?; \\\n"
            "  exec /usr/bin/docker run --rm --name qwen38-transcribe --gpus all \\\n"
            f"  -v {cls.hf}:/root/.cache/huggingface:ro \\\n"
            "  ghcr.io/ggml-org/llama.cpp@sha256:" + "0" * 64 + " \\\n"
            f"    -m {in_box}/{FILES[0]} \\\n"
            f"    --mmproj {in_box}/{FILES[1]} \\\n"
            "    --reasoning off -c 65536 \\\n"
            f"    --metrics --no-ui --api-key-file /out/api-key --host 127.0.0.1 --port {cls.lane.server_port}'\n")
        cls.ck.TRANSCRIBE_UNIT_PATH = cls.unit
        cls.real_run = cls.ck.run

    @classmethod
    def tearDownClass(cls):
        cls.lane.shutdown()
        cls.lane.server_close()
        shutil.rmtree(cls.box, ignore_errors=True)

    def setUp(self):
        self.ck.run = self.real_run
        self.ck.UNIT_TARGET_CACHE.clear()
        Health.asked[:] = []
        Health.loaded = True

    def whole(self):
        """Every file its unit names there, and the check of them."""
        for f in self.files:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b"GGUF")
        self.check.write_text("#!/bin/bash\nexit 0\n")


class ItsState(Base):
    def setUp(self):
        super().setUp()
        self.logs = []
        self.ck.run = lambda argv, **kw: "abc\n" if argv[:2] == ["docker", "ps"] and "name=^qwen38-transcribe$" in argv else ""
        self.ck._llamacpp_log = lambda container: (self.logs.append(container), list(self.log))[1]
        self.log = ()

    def state(self, prev="ready", enter="100"):
        st, boot, running = self.ck.transcribe_engine_state(U, active="active", sub="running", prev_state=prev,
                                                            enter_key=enter)
        return st["state"], boot, running

    def test_its_port_is_its_units(self):
        self.assertEqual(self.ck.transcribe_port(), self.lane.server_port)
        self.assertEqual(self.ck.transcribe_base(), f"http://127.0.0.1:{self.lane.server_port}")

    def test_healthy_on_its_own_port_it_is_ready_and_its_log_is_not_read(self):
        self.assertEqual(self.state(prev="warming-up")[0], "ready")
        self.assertEqual(Health.asked, ["/health"])
        self.assertEqual(self.logs, [])

    def test_a_boot_is_read_from_its_own_containers_log(self):
        Health.loaded = False
        upto = next(i for i, ln in enumerate(BOOT) if "loading model" in ln.lower()) + 1
        self.log = BOOT[:upto]
        st, boot, running = self.state(prev="starting", enter="200")
        self.assertIn(st, ("starting", "loading-weights"))
        self.assertTrue(running)
        self.assertEqual(self.logs, ["qwen38-transcribe"])

    def test_the_lifecycle_follows_it_on_its_own_facts(self):
        lc = self.ck.lc
        self.assertIn(U, lc.ENGINE_UNITS)
        self.assertNotIn(U, lc.TEXT_UNITS, "the SGLang belts are not its")
        self.assertNotIn(U, lc.LANE_PROVIDER, "no opencode provider: it serves no text")
        self.assertNotIn(U, self.ck.LLAMACPP_LANES, "the llama.cpp text lane's reads ask the text port")
        warn = lc.warn_reasons("unit", {"unit": U, "verb": "stop"}, {U: "ready"})
        self.assertEqual(len(warn), 1)
        self.assertIn("the NVIDIA Nemotron Transcribe lane", warn[0])
        self.assertNotIn(":30001", warn[0])
        # one engine at a time: a start of it is held while another lane serves, and the other way round
        self.assertTrue(lc.blocked_reasons("unit", {"unit": U, "verb": "start"}, {"qwen38-sglang.service": "ready"}))
        self.assertTrue(lc.blocked_reasons("unit", {"unit": "qwen38-sglang.service", "verb": "start"}, {U: "ready"}))


class ItsUnit(Base):
    def test_the_target_is_read_from_the_path_of_its_model(self):
        self.assertEqual(self.ck.unit_target(U), {"model": REPO_ID, "target": "transcribe"})

    def test_it_is_a_switch_target_a_unit_the_page_may_drive_and_a_lane_it_prepares(self):
        self.assertIn("transcribe", self.ck.ACTIONS["switch"]["params"]["target"])
        self.assertIn(U, self.ck.SERVING_UNITS)
        self.assertIn(U, self.ck.JOURNAL_UNITS)
        self.assertIn("qwen38-transcribe", self.ck.CONTAINERS)
        self.assertEqual(self.ck.target_lane("transcribe"), "transcribe")
        self.assertEqual(self.ck.target_unit("transcribe"), self.ck.TRANSCRIBE_UNIT_PATH)
        self.assertEqual(self.ck.ACTIONS["prepare"]["argv"]({"target": "transcribe"})[-1], "transcribe")
        self.assertEqual(self.ck.TARGET_PIN["transcribe"], "TRANSCRIBE_REV")
        self.assertEqual(self.ck.rg.PIN_MODELS["TRANSCRIBE_REV"], REPO_ID)

    def test_its_library_state_counts_the_two_files_it_pins_of_a_repository_that_holds_others(self):
        # mradermacher's repository holds every quantization: the Q8_0 and its projector alone count
        self.assertEqual(self.ck.rg.pinned_files(INSTALLER, ("TRANSCRIBE_FILE", "TRANSCRIBE_MMPROJ")), FILES)
        self.assertEqual(self.ck.rg.pinned_files(INSTALLER, ("TRANSCRIBE_FILE", "TRANSCRIBE_GONE")), FILES[:1])
        read = self.ck._read_once()
        self.assertEqual(read["files"]["transcribe"], FILES)
        self.assertNotIn("transcribe", read["targets"])

    def test_it_is_prepared_once_the_image_its_unit_names_is_here(self):
        img = "ghcr.io/ggml-org/llama.cpp@sha256:" + "0" * 64
        self.ck.run = lambda argv, **kw: "sha256:abc\n" if argv[:3] == ["docker", "image", "inspect"] and argv[-1] == img else ""
        self.assertTrue(self.ck.lane_prepared("transcribe"))
        self.ck.run = lambda argv, **kw: ""
        self.assertFalse(self.ck.lane_prepared("transcribe"))


class ItStartsOnlyWithItsFilesThere(Base):
    """A Start with no switch runs the check of its files, then llama-server on them: a file gone, an empty one or the
    check itself gone, and its unit refuses every start (exit 78), so the page does not offer it."""

    def setUp(self):
        super().setUp()
        img = "ghcr.io/ggml-org/llama.cpp@sha256:" + "0" * 64
        self.ck.run = lambda argv, **kw: "sha256:abc\n" if argv[:3] == ["docker", "image", "inspect"] and argv[-1] == img else ""
        self.whole()

    def test_whole_it_can_start(self):
        self.assertTrue(self.ck.lane_startable("transcribe"))

    def test_a_file_gone_empty_or_its_check_gone_holds_the_start_back(self):
        for what, undo in (("the model gone", lambda: self.files[0].unlink()),
                           ("the projector empty", lambda: self.files[1].write_bytes(b"")),
                           ("the check gone", lambda: self.check.unlink())):
            with self.subTest(what):
                undo()
                try:
                    self.assertFalse(self.ck.lane_startable("transcribe"))
                finally:
                    self.whole()


class ATextActionIsRefusedWhileItServes(Base):
    """The lane holds the GPU but serves no text, on its own port: the text port is closed, so a
    text action (smoke, flush, abort) is refused by the server with the reason, as the page refuses
    it (base.js's NEEDS_ENGINE) - before, the job ran against the closed :30000 (review 2026-11-19)."""

    def setUp(self):
        super().setUp()
        with self.ck.LIFE_LOCK:
            self.ck.LIFE["states"] = {U: "ready"}

    def tearDown(self):
        with self.ck.LIFE_LOCK:
            self.ck.LIFE["states"] = {}
        if self.ck.JOB_LOCK.locked():
            try:
                self.ck.JOB_LOCK.release()
            except RuntimeError:
                pass

    def test_smoke_flush_and_abort_are_refused_with_the_reason(self):
        for name in ("smoke", "flush_cache", "abort_all"):
            with self.subTest(name):
                code, out = self.ck.start_action(name, {})
                self.assertEqual(code, 409, out)
                self.assertEqual(out["error"], "blocked")
                self.assertIn("The NVIDIA Nemotron Transcribe lane is serving", out["reasons"][0])
                self.assertIn(":30000", out["reasons"][0])
                self.assertFalse(self.ck.JOB_LOCK.locked(), "a refusal holds no job lock")

    def test_a_text_action_is_allowed_again_once_it_does_not_serve(self):
        with self.ck.LIFE_LOCK:
            self.ck.LIFE["states"] = {}
        code, out = self.ck.start_action("smoke", {})
        self.assertEqual(code, 202, out)


class ItsContainerIsInTheUnitsTable(Base):
    """UNIT2CONT is what the orphans list is built from (collect_lifecycle) and so the banner
    (base.js reads it): without the transcribe entry, an orphan container of the lane refused every
    other lane's Start with "remove it first, see the banner" and no banner appeared (review of
    2026-11-19)."""

    def test_the_unit_maps_to_its_container(self):
        self.assertEqual(self.ck.UNIT2CONT[U], "qwen38-transcribe")

    def test_an_orphan_of_it_is_listed_and_the_other_lanes_start_is_refused(self):
        self.ck.run = lambda argv, **kw: (
            "c0ffee\n" if argv[:2] == ["docker", "ps"] and "name=^qwen38-transcribe$" in argv else
            "ActiveState=failed\nSubState=failed\nResult=timeout\nActiveEnterTimestampMonotonic=1000\n"
            if argv[:3] == ["systemctl", "show", U] else "")
        out = self.ck.collect_lifecycle()
        d = out.get("data", out)
        self.assertEqual(d["engines"][U]["state"], "orphan")
        self.assertEqual([(o["unit"], o["container"]) for o in d["orphans"]], [(U, "qwen38-transcribe")])
        blocked = d["blocked"].get("unit:start:qwen38-sglang.service") or []
        self.assertTrue(blocked and "outside systemd" in blocked[0], blocked)


class ItsAddressIsWhatThePageOffers(Base):
    """The api collector carries the lane's own address (its unit's bind and port) so the page can
    offer it while the lane serves; no key leaves the unit (review of v1.53.0, 2026-11-19)."""

    def test_collect_api_carries_its_base_and_port_and_no_key(self):
        self.ck.run = lambda argv, **kw: ""
        d = self.ck.collect_api()
        self.assertEqual(d["transcribe"], {"installed": True,
                                           "base": f"http://127.0.0.1:{self.lane.server_port}",
                                           "port": self.lane.server_port})
        self.assertNotIn("test-key", json.dumps(d))

    def test_without_its_unit_the_default_port_is_said(self):
        self.ck.run = lambda argv, **kw: ""
        saved = self.ck.TRANSCRIBE_UNIT_PATH
        self.ck.TRANSCRIBE_UNIT_PATH = self.box / "no-such-unit"
        try:
            self.assertEqual(self.ck.collect_api()["transcribe"],
                             {"installed": False, "base": "http://127.0.0.1:30036", "port": 30036})
        finally:
            self.ck.TRANSCRIBE_UNIT_PATH = saved



if __name__ == "__main__":
    unittest.main()
