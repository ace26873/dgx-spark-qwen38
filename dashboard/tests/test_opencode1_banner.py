#!/usr/bin/env python3
"""A box that chose ./install.sh --opencode 1 is told it is behind (decision 1564 of Hasan, 2026-11-22): since
v1.56.0 every box serves OpenCode 2 by default, and the marker "1" keeps one on OpenCode 1 at every update; the
cockpit says so in an orange banner, with the command that moves it, so nobody stays behind without knowing it.
Same harness as test_page_behaviour.py."""
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_page_behaviour import run  # noqa: E402


class AnOpencode1BoxIsToldItIsBehind(unittest.TestCase):

    def banners(self, agent):
        return run(self, r"""
        const units = JSON.parse(JSON.stringify(UNITS));
        units.units['qwen38-sglang.service'] = {active: 'active', enabled: 'enabled'};
        const now = Date.now() / 1000;
        apply({config: {data: CONFIG, ts: now}, units: {data: units, ts: now},
               update: {data: {installed: 'v1.56.0'}, ts: now},
               agent: {data: %s, ts: now},
               lifecycle: {data: life({'qwen38-sglang.service': eng('ready', {target: 'stock'})}), ts: now}});
        report(txt('banners'));
        """ % json.dumps(agent))

    def test_a_box_marked_1_gets_the_orange_banner_with_the_command(self):
        r = self.banners({"major_choice": "1", "server": {"healthy": True, "version": "1.19.6"}, "pinned": "1.19.6"})
        self.assertIn("opencode 1 on this box", r)
        self.assertIn("./install.sh --opencode 2", r)

    def test_a_box_on_the_default_gets_none(self):
        for agent in ({"major_choice": "2", "server": {"healthy": True, "version": "2.3.2"}, "pinned": "2.3.2"},
                      {"major_choice": None, "server": {"healthy": True, "version": "2.3.2"}, "pinned": "2.3.2"}):
            with self.subTest(choice=agent["major_choice"]):
                self.assertNotIn("opencode 1 on this box", self.banners(agent))


if __name__ == "__main__":
    unittest.main(verbosity=2)
