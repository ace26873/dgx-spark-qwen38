#!/usr/bin/env python3
"""The Image view's routes on the MiniMax-H3-Image lane (v1.50): one view and one set of routes for the image lane
that serves, else the one installed, Qwen-Image 2.1 first; each call by the rules of the lane it goes to. Same harness
as test_image_routes.py, its cockpit loaded with run() silenced and the engine's urlopen captured."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# its module fixtures too: they put back the environment Base's cockpit changes (the suite's isolation check)
from test_image_routes import Base, setUpModule, tearDownModule  # noqa: E402,F401

H3 = "qwen38-image-h3.service"
QWEN = "qwen38-image.service"


class TheLaneTheViewAddresses(Base):
    def setUp(self):
        super().setUp()
        self.t = Path(tempfile.mkdtemp(prefix="image-h3-units-"))
        self.kept = (self.ck.IMAGE_UNIT_PATH, self.ck.IMAGE_H3_UNIT_PATH, dict(self.ck.LIFE.get("states", {})))
        self.ck.IMAGE_UNIT_PATH, self.ck.IMAGE_H3_UNIT_PATH = self.t / QWEN, self.t / H3

    def tearDown(self):
        self.ck.IMAGE_UNIT_PATH, self.ck.IMAGE_H3_UNIT_PATH, states = self.kept
        self.ck.LIFE["states"] = states
        super().tearDown()

    def box(self, installed, states):
        for u in (QWEN, H3):
            (self.t / u).unlink(missing_ok=True)
        for u in installed:
            (self.t / u).write_text("[Service]\nExecStart=/x/sglang serve --model-path m --host 127.0.0.1 --port 30026\n")
        self.ck.LIFE["states"] = states

    def test_the_lane_that_serves_whichever_it_is(self):
        self.box([QWEN, H3], {H3: "ready", QWEN: "stopped"})
        self.assertEqual(self.ck.image_unit(), H3)
        self.box([QWEN, H3], {QWEN: "ready", H3: "stopped"})
        self.assertEqual(self.ck.image_unit(), QWEN)

    def test_in_a_switch_the_lane_coming_up_before_the_one_stopping(self):
        self.box([QWEN, H3], {QWEN: "stopping", H3: "starting"})
        self.assertEqual(self.ck.image_unit(), H3)

    def test_none_up_the_one_installed_qwen_first(self):
        self.box([QWEN, H3], {})
        self.assertEqual(self.ck.image_unit(), QWEN)
        self.box([H3], {})
        self.assertEqual(self.ck.image_unit(), H3)
        self.box([], {})
        self.assertEqual(self.ck.image_unit(), QWEN)

    def test_its_port_is_read_from_its_unit(self):
        self.box([H3], {})
        self.assertEqual(self.ck.image_h3_port(), 30026)
        (self.t / H3).write_text("ExecStart=/x/sglang serve --host 127.0.0.1 --port 31026\n")
        self.ck.IMAGE_UNIT_CACHE.clear()
        self.assertEqual(self.ck.image_h3_port(), 31026)
        # the real image_base, which the harness stands in for: it reads the H3 unit's bind and port
        self.assertEqual(type(self).saved["image_base"](H3), "http://127.0.0.1:31026")

    def test_the_tab_is_told_which_lane_and_which_lanes(self):
        self.box([QWEN, H3], {H3: "ready", QWEN: "stopped"})
        out = self.ck.image_status()
        self.assertEqual((out["unit"], out["units"], out["port"], out["state"]), (H3, [QWEN, H3], 30026, "ready"))


class ACallToMiniMaxH3Image(Base):
    def setUp(self):
        super().setUp()
        self.saved_unit = self.ck.image_unit
        self.ck.image_unit = lambda: H3

    def tearDown(self):
        self.ck.image_unit = self.saved_unit
        super().tearDown()

    def test_one_image_per_call(self):
        code, out = self.call({"prompt": "x", "n": 2, "width": 2048, "height": 2048})
        self.assertEqual(code, 400)
        self.assertIn("MiniMax-H3-Image makes one image per call", out["error"])
        self.assertIsNone(self.spy.body)

    def test_up_to_the_largest_size_measured_here_and_not_past_it(self):
        for w, h in ((4736, 3552), (6272, 2688), (4096, 4096)):
            code, _ = self.call({"prompt": "x", "width": w, "height": h})
            self.assertEqual(code, 200, (w, h))
            self.assertEqual((json.loads(self.spy.body)["width"], json.loads(self.spy.body)["height"]), (w, h))
        code, out = self.call({"prompt": "x", "width": 5024, "height": 3360})
        self.assertEqual(code, 400)
        self.assertIn("one 6272x2688 image", out["error"])

    def test_what_it_does_not_read_stays_out_and_jpeg_is_its_to_make(self):
        code, _ = self.call({"prompt": "x", "background": "transparent", "flow_shift": 3.0, "max_sequence_length": 512,
                             "output_format": "jpeg", "seed": 7})
        self.assertEqual(code, 200)
        body = json.loads(self.spy.body)
        self.assertNotIn("background", body)
        self.assertNotIn("flow_shift", body)
        self.assertNotIn("max_sequence_length", body)
        self.assertEqual((body["output_format"], body["seed"]), ("jpeg", 7))

    def test_it_goes_to_its_own_lane(self):
        calls = []
        self.ck.image_base = lambda unit=None: calls.append(unit) or "http://127.0.0.1:30026"
        try:
            self.call({"prompt": "x", "width": 2048, "height": 2048})
        finally:
            self.ck.image_base = lambda unit=None: "http://127.0.0.1:30020"
        self.assertEqual(calls, [H3])
        self.assertEqual(self.spy.url, "http://127.0.0.1:30026/v1/images/generations")

    def test_a_lane_that_does_not_answer_names_its_own_load(self):
        def refused(req, timeout=None):
            raise OSError("connection refused")
        self.ck.urllib.request.urlopen = refused
        code, out = self.call({"prompt": "x"})
        self.assertEqual(code, 502)
        self.assertIn("Load MiniMax-H3-Image", out["error"])
        self.assertIn("./switch-model.sh image-h3", out["error"])


if __name__ == "__main__":
    unittest.main()
