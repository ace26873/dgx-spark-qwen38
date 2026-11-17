#!/usr/bin/env python3
"""An update that waits for a lane's Load, on the page as it runs (decisions 1408 and 1409 of Hasan, 2026-11-17): the
lane keeps serving the release it served, its card says an update waits and what its Load downloads, the Lanes view's
badge is orange, and its Load prepares what the waiting unit lacks, puts that unit in place, then restarts the lane.
Same harness as test_page_behaviour.py."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_page_behaviour import run  # noqa: E402

LACKS = ["image lmsysorg/sglang@sha256:" + "b" * 64, "checkpoint RadixArk/Qwen3.8-27B-NVFP4@" + "2" * 40]


def page(test, pending):
    return run(test, r"""
    const units = JSON.parse(JSON.stringify(UNITS));
    units.units['qwen38-sglang.service'] = {active: 'active', enabled: 'enabled'};
    const now = Date.now() / 1000;
    apply({config: {data: CONFIG, ts: now}, units: {data: units, ts: now},
           update: {data: {installed: 'v1.51.0'}, ts: now},
           lanes_ready: {data: {'27b': %s}, ts: now},
           pending: {data: %s, ts: now},
           lifecycle: {data: life({'qwen38-sglang.service': eng('ready', {target: 'stock'})}), ts: now}});
    showView('lanes');
    const c = LCARDS.get('qwen38-sglang.service');
    const out = {loadHidden: c.load.hidden, loadText: c.load.textContent, pick: c.pickLine.textContent,
                 badge: txt('bdg-lanes'), name: LANE_NAME['qwen38-sglang.service']};
    askJourney('stock');
    report(Object.assign(out, {open: !$('scrim').hidden, steps: [...$('sh-journey').children].map(li => li.textContent)}));
    """ % ("false" if pending else "true",
           ("{'qwen38-sglang.service': %s}" % __import__("json").dumps(LACKS)) if pending else "{}"))


class AnUpdateWaitsForItsLoad(unittest.TestCase):
    def test_the_card_says_it_and_the_badge_is_orange(self):
        r = page(self, True)
        self.assertFalse(r["loadHidden"], r)
        self.assertEqual(r["loadText"], f"Load the update of {r['name']}")
        self.assertIn("An update waits for its Load, which downloads " + "; ".join(LACKS), r["pick"])
        self.assertIn("until then it serves the release it served", r["pick"])
        self.assertEqual(r["badge"], "update")

    def test_its_load_prepares_then_puts_the_unit_in_place_then_restarts(self):
        r = page(self, True)
        self.assertTrue(r["open"], r)
        steps = r["steps"]
        self.assertTrue(any("prepare-lane.sh 27b" in s for s in steps), steps)
        self.assertIn(f"Download what the update of {r['name']} needs", steps[0])
        self.assertNotIn("first Load", steps[0])
        i = next(i for i, s in enumerate(steps) if f"Put in place the update {r['name']} waited for" in s)
        self.assertTrue(any("prepare-lane.sh 27b" in s for s in steps[:i]), "what it lacks is fetched before")
        self.assertIn("systemctl restart qwen38-sglang.service", steps[-1])
        self.assertFalse(any("Point the boot" in s for s in steps), steps)

    def test_nothing_waits_nothing_said(self):
        r = page(self, False)
        self.assertTrue(r["loadHidden"], r)
        self.assertNotIn("An update waits", r["pick"])
        self.assertEqual(r["badge"], "")


if __name__ == "__main__":
    unittest.main()
