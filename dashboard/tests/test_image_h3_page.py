#!/usr/bin/env python3
"""The Image view on the MiniMax-H3-Image lane (v1.50): the view addresses the image lane that serves, else the one
installed, Qwen-Image 2.1 first, by the rule the cockpit's routes follow (cockpit.py image_unit), and takes that
lane's sizes, steps, controls and licence. Same harness as test_page_behaviour.py."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_page_behaviour import run  # noqa: E402


def lanes(qwen=None, h3=None):
    engines = {"'qwen38-sglang.service'": "eng('stopped')"}
    if qwen:
        engines["'qwen38-image.service'"] = f"eng('{qwen}', {{target: 'image'}})"
    if h3:
        engines["'qwen38-image-h3.service'"] = f"eng('{h3}', {{target: 'image-h3'}})"
    body = ", ".join(f"{k}: {v}" for k, v in engines.items())
    return f"feed({{config: CONFIG, units: UNITS, lifecycle: life({{{body}}})}});\nshowView('image');\n"


VIEW = r"""
imgSync();
const opts = [...$('img-size').querySelectorAll('option')].map(o => o.value);
report({unit: imgUnit(), size: $('img-size').value, opts, steps: [$('img-steps').value, $('img-steps').disabled],
        nHidden: $('img-n-f').hidden, bgHidden: $('img-bg-f').hidden, negHidden: $('img-neg-f').hidden,
        payload: imgPayload(), licence: $('img-license').textContent, lede: txt('img-lede')});
"""


class TheViewAddressesTheLaneThatServes(unittest.TestCase):
    def test_minimax_h3_image_serving_takes_its_sizes_steps_and_controls(self):
        r = run(self, lanes(qwen="stopped", h3="ready") + VIEW)
        self.assertEqual(r["unit"], "qwen38-image-h3.service")
        self.assertEqual(r["size"], "2048x2048")
        self.assertIn("4736x3552", r["opts"])
        self.assertIn("6272x2688", r["opts"])
        self.assertNotIn("2752x1536", r["opts"])
        self.assertEqual(r["steps"], ["6", True])
        self.assertEqual((r["nHidden"], r["bgHidden"], r["negHidden"]), (True, True, True))
        for k in ("num_inference_steps", "n", "background", "true_cfg_scale", "negative_prompt", "flow_shift"):
            self.assertNotIn(k, r["payload"], k)
        self.assertEqual((r["payload"]["width"], r["payload"]["height"]), (2048, 2048))
        self.assertIn("not granted in the European Union, the United Kingdom, South Korea or the United States", r["licence"])
        self.assertIn("definition 5 and clause V.4", r["licence"])
        self.assertIn("MiniMax-H3-Image", r["lede"])

    def test_qwen_image_serving_keeps_its_own(self):
        r = run(self, lanes(qwen="ready", h3="stopped") + VIEW)
        self.assertEqual(r["unit"], "qwen38-image.service")
        self.assertEqual(r["size"], "1024x1024")
        self.assertIn("2752x1536", r["opts"])
        self.assertNotIn("4736x3552", r["opts"])
        self.assertEqual(r["steps"], ["40", False])
        self.assertEqual((r["nHidden"], r["bgHidden"]), (False, False))
        self.assertEqual(r["payload"]["num_inference_steps"], 40)
        self.assertIn("not commercial use", r["licence"])

    def test_its_cost_line_counts_its_one_image_whatever_the_hidden_slider_kept(self):
        # Images per call left at 4 on Qwen-Image, then MiniMax-H3-Image serves: its one image is no "each" (review of v1.50)
        r = run(self, lanes(qwen="stopped", h3="ready") + "$('img-n').value = '4'; imgSync(); report(txt('img-cost-k'));")
        self.assertIn("megapixels", r)
        self.assertNotIn("each", r)

    def test_neither_serving_the_one_installed_qwen_first(self):
        self.assertEqual(run(self, lanes(qwen="stopped", h3="stopped") + "report(imgUnit());"), "qwen38-image.service")
        self.assertEqual(run(self, lanes(h3="stopped") + "report(imgUnit());"), "qwen38-image-h3.service")

    def test_a_lane_that_failed_is_the_one_the_view_says(self):
        # its crash shows, not the other lane's controls (review of v1.50; cockpit.py's image_unit, the same rule)
        self.assertEqual(run(self, lanes(qwen="stopped", h3="failed") + "report(imgUnit());"), "qwen38-image-h3.service")

    def test_in_a_switch_the_lane_coming_up_before_the_one_stopping(self):
        self.assertEqual(run(self, lanes(qwen="stopping", h3="starting") + "report(imgUnit());"), "qwen38-image-h3.service")

    def test_a_size_past_the_largest_measured_is_refused_before_it_is_sent(self):
        r = run(self, lanes(h3="ready") + r"""
        $('img-size').value = 'custom'; imgSync(); $('img-w').value = '5024'; $('img-h').value = '3360'; $('img-prompt').value = 'x';
        const big = imgProblem();
        $('img-w').value = '6272'; $('img-h').value = '2688';
        report({big, ok: imgProblem()});
        """)
        self.assertIn("the largest measured here is 16.9", r["big"])
        self.assertEqual(r["ok"], "")

    def test_with_both_installed_and_none_serving_each_has_its_load(self):
        r = run(self, lanes(qwen="stopped", h3="stopped") + r"""
        imgRenderLane();
        report([...$('img-lane').querySelectorAll('button')].map(b => b.textContent));
        """)
        self.assertEqual(r, ["Load Qwen-Image", "Load MiniMax-H3-Image"])


class TheRefusalSaysOneUnit(unittest.TestCase):
    """The refusal of a too large H3-Image call mixed Go and Gio in one sentence: "92.3 GB for the
    engine" beside the Qwen-Image lane's "45.5 GiB" (review of v1.53.0, 2026-11-19; the rule: Go =
    10^9, Gio = 2^30, never mixed). The study measured a 92.3 GB peak (images-h3-flux3/BILAN.md),
    which is 85.97 GiB, shown rounded to one decimal, and the cost line keeps the measured figure
    as its source."""

    # the chiffre the page must show: the measured peak converted, rounded, not truncated
    PEAK_GIB = f"{92.3e9 / 2**30:.1f}"

    def test_the_refusal_says_gib(self):
        self.assertEqual(self.PEAK_GIB, "86.0")
        r = run(self, lanes(h3="ready") + r"""
        $('img-size').value = 'custom'; imgSync(); $('img-w').value = '5024'; $('img-h').value = '3360'; $('img-prompt').value = 'x';
        report(imgProblem());
        """)
        self.assertIn(f"{self.PEAK_GIB} GiB for the engine at its peak", r)
        self.assertNotIn("GB for the engine", r)

    def test_the_cost_line_keeps_the_measured_figure_as_its_source(self):
        r = run(self, lanes(h3="ready") + r"""
        $('img-prompt').value = 'x'; imgSync();
        report(txt('img-cost-b'));
        """)
        self.assertIn(f"{self.PEAK_GIB} GiB at its peak", r)
        self.assertIn("92.3 GB as measured", r)



if __name__ == "__main__":
    unittest.main()
