#!/usr/bin/env python3
"""A bench that closes one side lane's address closes every one.

The cockpit reads where each side lane answers from the environment first (COCKPIT_IMAGE and the
like), else from the lane's unit on this box. A bench closes those addresses so that its
collectors probe nothing real; one that closes some leaves the others to the box's units. The
benches closed COCKPIT_IMAGE and not COCKPIT_IMAGE_H3, which nothing saw until the reference box
had the MiniMax-H3-Image unit: then the offline suite probed the real 127.0.0.1:30026 and the CI
failed (2026-11-16). The serving-port gate of tests/run_suite.py sees a contact only on a box
whose unit is there; this reads the source, and sees it on any box.
"""
import re
import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parent
COCKPIT = TESTS.parent / "cockpit.py"
# a side lane's address: an environment variable first, else one built from what its unit says (an f-string)
SIDE_URL = re.compile(r'os\.environ\.get\(\s*"(COCKPIT_[A-Z0-9_]+)",\s*f"https?://')


def side_lane_vars() -> set:
    return set(SIDE_URL.findall(COCKPIT.read_text()))


def benches() -> dict:
    """Each test file that sets one of them: its name and the ones it sets."""
    names = side_lane_vars()
    out = {}
    for path in sorted(TESTS.glob("test_*.py")):
        if path.name == Path(__file__).name:
            continue
        found = {n for n in names if re.search(rf"\b{n}=", path.read_text())}
        if found:
            out[path.name] = found
    return out


class EveryBenchClosesEverySideLane(unittest.TestCase):
    def test_the_cockpit_reads_its_side_lanes_from_the_environment(self):
        # the image lanes, the video lane and the transcribe lane at least: an empty set would pass anything
        self.assertGreaterEqual(side_lane_vars(), {"COCKPIT_IMAGE", "COCKPIT_IMAGE_H3", "COCKPIT_VIDEO"})

    def test_a_bench_that_closes_one_closes_them_all(self):
        names = side_lane_vars()
        self.assertTrue(benches(), "no bench closes a side lane: the pattern no longer matches them")
        for name, found in benches().items():
            with self.subTest(bench=name):
                self.assertEqual(sorted(names - found), [], f"{name} leaves these side lanes to the box's units")


if __name__ == "__main__":
    unittest.main(verbosity=2)
