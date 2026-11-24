#!/usr/bin/env python3
"""The proxy and the Inkling 2 Mini lane (keepalive-proxy v6.62): its served name is a lane of the pair's routes, a
request's reasoning effort reaches its template as it came, "none" included (the template's lowest level, which the Qwen
templates refuse and the proxy turns into enable_thinking false for them), and System One, whose one token read is never
an option's label on it, is refused there."""
import importlib.util
import json
import pathlib
import time
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
SERVED = "inkling-2-mini"
CHAT = "/v1/chat/completions"


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TheProxy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kp = load("kproxy_inkling", "keepalive-proxy.py")

    def serving(self, names, engine="sglang"):
        saved = self.kp.served_models, dict(self.kp._SERVED)
        self.kp.served_models = lambda: names
        self.kp._SERVED.update(names=names, engine=engine, ts=time.time())
        self.addCleanup(lambda: (setattr(self.kp, "served_models", saved[0]), self.kp._SERVED.clear(),
                                 self.kp._SERVED.update(saved[1])))

    def test_its_name_is_a_lane_of_the_pair(self):
        self.assertIn(SERVED, self.kp.LANE_NAMES)

    def test_none_reaches_its_template_as_it_came(self):
        """Its template reads none (0.0) and no enable_thinking: a none made enable_thinking false lost the level, and
        the lane reasoned at its default, high (0.9)."""
        self.serving((SERVED,))
        for where in ("top", "kwargs"):
            with self.subTest(where=where):
                body = {"model": SERVED, "messages": [{"role": "user", "content": "hi"}]}
                if where == "top":
                    body["reasoning_effort"] = "none"
                else:
                    body["chat_template_kwargs"] = {"reasoning_effort": "none"}
                raw = json.dumps(body).encode()
                self.assertEqual(self.kp.qwen_effort_none(raw, CHAT), (raw, False))

    def test_the_qwen_lanes_still_get_enable_thinking_false(self):
        self.serving(("qwen3.8-27b",))
        raw = json.dumps({"model": "qwen3.8-27b", "reasoning_effort": "none", "messages": []}).encode()
        out, changed = self.kp.qwen_effort_none(raw, CHAT)
        self.assertTrue(changed)
        self.assertEqual(json.loads(out)["chat_template_kwargs"], {"enable_thinking": False})

    def test_system_one_is_refused_on_it_with_the_lanes_it_serves_on_named(self):
        """Its card says its lowest effort does not turn the reasoning off: the one token System One reads would be the
        thought's, never an option's label."""
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
        must hold it, and Inkling 2 Mini's is 201,024 (its config.json at the pinned revision)."""
        self.assertLessEqual(self.kp.UNKNOWN_VOCAB_FLOOR, 201024)


if __name__ == "__main__":
    unittest.main()
