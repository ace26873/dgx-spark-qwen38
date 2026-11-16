#!/usr/bin/env python3
"""The proxy and the Liquid AI LFM3 lane (keepalive-proxy v6.61): its served name is a lane of the pair's routes, its
template reads enable_thinking alone, as Ornith's, so a request's reasoning effort reaches it the way it reaches Ornith
(none and minimal off, every other level on), and System One stays refused there, as on every vLLM lane."""
import http.server
import io
import json
import os
import socket
import sys
import threading
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_proxy_vllm_lane import KEY, FakeVllm, _keep_env, load_proxy  # noqa: E402

LFM3 = "lfm3-1.7b"
CHAT = "/v1/chat/completions"


class TheTranslation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _keep_env(cls)
        os.environ["UPSTREAM"] = "http://127.0.0.1:9"
        cls.mod = load_proxy("kproxy_lfm3_effort_pure")

    def serve(self, name, engine="vllm"):
        self.mod._SERVED.update(names=(name,), engine=engine, ts=time.time())

    def go(self, body):
        raw = json.dumps(body).encode()
        out, level = self.mod.ornith_effort(raw, CHAT)
        return (json.loads(out) if out != raw else None), level

    def test_its_name_is_a_lane_of_the_pair(self):
        self.assertIn(LFM3, self.mod.LANE_NAMES)

    def test_none_and_minimal_turn_its_reasoning_off_and_every_other_level_keeps_it(self):
        self.serve(LFM3)
        want = {"none": False, "minimal": False, "low": True, "medium": True, "high": True, "xhigh": True, "max": True}
        for level, on in want.items():
            for where in ("top", "kwargs"):
                with self.subTest(level=level, where=where):
                    body = {"model": LFM3, "messages": [{"role": "user", "content": "hi"}]}
                    if where == "top":
                        body["reasoning_effort"] = level
                    else:
                        body["chat_template_kwargs"] = {"reasoning_effort": level}
                    sent, got = self.go(body)
                    self.assertEqual(got, level)
                    self.assertEqual(sent["chat_template_kwargs"], {"enable_thinking": on})
                    self.assertNotIn("reasoning_effort", sent)

    def test_another_vllm_lane_and_an_sglang_lane_named_so_go_as_they_came(self):
        body = {"model": "x", "reasoning_effort": "none", "messages": []}
        self.serve("kolibri-1")
        self.assertEqual(self.go(body), (None, None))
        self.serve(LFM3, engine="sglang")
        self.assertEqual(self.go(body), (None, None))

    def test_system_one_stays_refused_there_as_on_every_vllm_lane(self):
        saved = self.mod.served_models
        self.mod.served_models = lambda: (LFM3,)
        self.serve(LFM3)
        try:
            with self.assertRaises(self.mod.SystemOneUsage) as e:
                self.mod.systemone_model("jev-latest")
        finally:
            self.mod.served_models = saved
        self.assertIn(f"this box serves {LFM3} on vLLM now", str(e.exception))


class ThroughTheProxy(unittest.TestCase):
    """A fake vLLM serving lfm3-1.7b behind the real proxy."""

    @classmethod
    def setUpClass(cls):
        _keep_env(cls)
        cls.engine = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeVllm)
        cls.engine.daemon_threads = True
        threading.Thread(target=cls.engine.serve_forever, daemon=True).start()
        os.environ.update(UPSTREAM=f"http://127.0.0.1:{cls.engine.server_port}", KEEPALIVE_S="0.4",
                          CLIENT_IO_S="5", PROMPT_CEILING_TOKENS="0", FLASH_PROMPT_CEILING_TOKENS="0")
        cls.mod = load_proxy("kproxy_lfm3_effort")
        cls.mod._api_key = lambda: KEY
        cls.lines = []
        cls.mod.log = lambda msg: cls.lines.append(f"[proxy] {msg}")
        cls.err = redirect_stderr(io.StringIO())
        cls.err.__enter__()
        cls.proxy = cls.mod.Server(("127.0.0.1", 0), cls.mod.H)
        threading.Thread(target=cls.proxy.serve_forever, daemon=True).start()
        cls.port = cls.proxy.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.err.__exit__(None, None, None)
        cls.proxy.shutdown()
        cls.proxy.server_close()
        cls.engine.shutdown()
        cls.engine.server_close()
        FakeVllm.served, FakeVllm.owner = "kolibri-1", "vllm"

    def post(self, body):
        raw = json.dumps(body).encode()
        s = socket.create_connection(("127.0.0.1", self.port), timeout=20)
        s.sendall(f"POST {CHAT} HTTP/1.1\r\nHost: p\r\nAuthorization: Bearer {KEY}\r\n"
                  f"Content-Type: application/json\r\nContent-Length: {len(raw)}\r\n\r\n".encode() + raw)
        s.settimeout(20)
        got = b""
        while True:
            c = s.recv(65536)
            if not c:
                break
            got += c
        s.close()
        return got

    def test_pis_off_reaches_it_as_thinking_off_and_says_so_in_its_name(self):
        FakeVllm.served, FakeVllm.owner = LFM3, "vllm"
        with FakeVllm.lock:
            FakeVllm.bodies.clear()
        self.mod.invalidate_pool()
        self.mod.served_models()
        got = self.post({"model": LFM3, "stream": True, "reasoning_effort": "none",
                         "messages": [{"role": "user", "content": "What is 17 * 23?"}]})
        self.assertTrue(got.startswith(b"HTTP/1.1 200"), got[:200])
        sent = FakeVllm.bodies[-1]
        self.assertEqual(sent.get("chat_template_kwargs"), {"enable_thinking": False})
        self.assertNotIn("reasoning_effort", sent)
        self.assertTrue(any("not what the LFM3 template reads: relayed as enable_thinking false" in ln
                            for ln in self.lines), self.lines[-5:])


if __name__ == "__main__":
    unittest.main()
