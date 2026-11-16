#!/usr/bin/env python3
"""The proxy and the Liquid AI LFM2.5 24B-A2B lane (keepalive-proxy v6.60): its served name is a lane of the pair's
routes, and System One, whose one token read is never an option's label on it, is refused there."""
import importlib.util
import pathlib
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
SERVED = "lfm2.5-24b-a2b"


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TheProxy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kp = load("kproxy_lfm25", "keepalive-proxy.py")

    def serving(self, names, engine="sglang"):
        saved = self.kp.served_models, self.kp._SERVED["engine"]
        self.kp.served_models = lambda: names
        self.kp._SERVED["engine"] = engine
        self.addCleanup(lambda: (setattr(self.kp, "served_models", saved[0]), self.kp._SERVED.__setitem__("engine", saved[1])))

    def test_its_name_is_a_lane_of_the_pair(self):
        self.assertIn(SERVED, self.kp.LANE_NAMES)

    def test_system_one_is_refused_on_it_with_the_lanes_it_serves_on_named(self):
        """Its template opens a thought at every answer and reads no switch: the one token System One reads would be
        the thought's, never an option's label."""
        self.serving((SERVED,))
        with self.assertRaises(self.kp.SystemOneUsage) as e:
            self.kp.systemone_model("jev-latest")
        self.assertIn(f"this box serves {SERVED} now", str(e.exception))
        self.assertIn("the 27B and the flash lanes", str(e.exception))
        with self.assertRaises(self.kp.SystemOneUsage):
            self.kp.systemone_model(SERVED)

    def test_the_27b_still_answers_it(self):
        self.serving(("qwen3.8-27b",))
        self.assertEqual(self.kp.systemone_model("jev-latest"), "qwen3.8-27b")

    def test_its_vocabulary_is_no_smaller_than_the_floor(self):
        """An id under UNKNOWN_VOCAB_FLOOR is relayed before the served vocabulary is learned: every lane's vocabulary
        must hold it, and LFM2.5's is 128,000 (its config.json at the pinned revision)."""
        self.assertLessEqual(self.kp.UNKNOWN_VOCAB_FLOOR, 128000)



if __name__ == "__main__":
    unittest.main()
