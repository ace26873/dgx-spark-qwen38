#!/usr/bin/env python3
"""The opencode config install.sh generates gives each lane the limits it serves.

oc and the Agent tab load this file over the user's own (OPENCODE_CONFIG), so its
numbers are the ones they run on. A flash install runs in native mode and used that mode
for the 27B block too: on a box whose 27B serves 1M, the block came out at the native
pair (reference box, 2026-09-23). The 27B block now comes from oc-limits.sh, for the
checkpoint and mode its unit serves: a copy of the numbers here had drifted from the
table and gave an FP8 box the NVFP4 1M pair (found in review, 2026-09-24). These run the
installer's own lines, as written, in a throwaway config dir.""" 
import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
INSTALL = REPO / "install.sh"


def block() -> str:
    text = INSTALL.read_text()
    start = text.index("OC_27B=0; OC_FLASH=0; OC_KOLIBRI=0; OC_NEMOTRON=0; OC_NEMOTRON_SUPER=0; OC_FLASH_LOCAL=0; OC_ORNITH_122B=0; "
                       "OC_FLASH_UNSLOTH=0; OC_LFM25=0; OC_LFM3=0\n")
    end = text.index("\nPYEOF\n", start) + len("\nPYEOF\n")
    return text[start:end]


PINS = "\n".join(line for line in INSTALL.read_text().splitlines()
                 if line.startswith(("STOCK_REPO=", "UNC_REPO=", "FP8_REPO=", "UNCFP8_REPO=", "ARETE_REPO=", "DAISYROSE_REPO=",
                                     "FLASH_REPO=", "FLASH_UNC_REPO=", "FLASH_NVDA_REPO=")))


def generate(lane, mode, unit_ctx=None, flash_unit=False, model="RadixArk/Qwen3.8-27B-NVFP4",
             pair=None, names=False, flash_running=None, kolibri_unit=False, kolibri_boot=False, doc=False,
             nemotron_unit=False, nemotron_boot=False, super_unit=False, super_boot=False,
             flocal_unit=False, flocal_boot=False, ornith_unit=False, ornith_boot=False, unsloth_unit=False,
             unsloth_boot=False, unsloth_slots="-c 524288 -np 2",
             unsloth_model="snapshots/" + "d" * 40 + "/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
             unsloth_want=False, unsloth_handed="", lfm25_unit=False, lfm25_boot=False, lfm3_unit=False, lfm3_boot=False,
             major="1", serve_major=None):
    t = pathlib.Path(tempfile.mkdtemp(prefix="oc-art-"))
    kolibri = t / "qwen38-kolibri.service"
    if kolibri_unit:
        kolibri.write_text("ExecStart=... Aleph-Alpha/Kolibri-1 --revision 258cae8f ...\n")
    nemotron = t / "qwen38-nemotron.service"
    if nemotron_unit:
        nemotron.write_text("ExecStart=... nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB --revision c6cc2e8f ...\n")
    nsuper = t / "qwen38-nemotron-super.service"
    if super_unit:
        nsuper.write_text("ExecStart=... nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 --revision 8a115b9d ...\n")
    flocal = t / "qwen38-flash-local.service"
    if flocal_unit:
        flocal.write_text("ExecStart=... nvidia/Qwen3.8-Flash-Next-FP4-Local --revision b9ce18a6 ...\n")
    ornith = t / "qwen38-ornith-122b.service"
    if ornith_unit:
        ornith.write_text("ExecStart=... ornith-ai/Ornith-1.5-122B-NVFP4 --revision f049036f ...\n")
    unsloth = t / "qwen38-flash-unsloth.service"
    if unsloth_unit:
        unsloth.write_text(f"ExecStart=... -m /root/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/{unsloth_model} "
                           f"--alias qwen3.8-flash-next-unsloth {unsloth_slots} ...\n")
    lfm25 = t / "qwen38-lfm25.service"
    if lfm25_unit:
        lfm25.write_text("ExecStart=... --model-path LiquidAI/LFM2.5-24B-A2B-FP8 --revision f54bf395 ...\n")
    lfm3 = t / "qwen38-lfm3.service"
    if lfm3_unit:
        lfm3.write_text("ExecStart=... LiquidAI/LFM3-1.7B-NVFP4 --revision cfe4b5ea ...\n")
    if flash_running:
        (t / "launch-flash.sh").write_text(
            "docker run img python3 -m sglang.launch_server --model-path RadixArk/Qwen3.8-Flash-Next-NVFP4 "
            f"--max-running-requests {flash_running} --context-length 262144\n")
    sgl = t / "qwen38-sglang.service"
    if unit_ctx:
        sgl.write_text(f"ExecStart=... --model-path {model} --context-length {unit_ctx} ...\n")
    flash = t / "qwen38-flash.service"
    if flash_unit:
        flash.write_text("x\n")
    ctx, out = pair or (("205000", "32000") if lane == "flash" else ("700000", "200000"))
    # lanes.sh first, as install.sh sources it: the block reads a llama.cpp unit's window with its llamacpp_window (v1.48)
    script = ("set -euo pipefail\ndie(){ echo \"DIE: $*\"; exit 1; }\n" + f'. "{REPO}/lanes.sh"\n' + PINS + "\n"
              f'LANE={lane}; CONTEXT_MODE={mode}; SGL_UNIT_PATH="{sgl}"; FLASH_UNIT_PATH="{flash}"\n'
              f'KOLIBRI_UNIT_PATH="{kolibri}"; KOLIBRI_BOOT={1 if kolibri_boot else 0}\n'
              f'NEMOTRON_UNIT_PATH="{nemotron}"; NEMOTRON_BOOT={1 if nemotron_boot else 0}\n'
              f'NEMOTRON_SUPER_UNIT_PATH="{nsuper}"; NEMOTRON_SUPER_BOOT={1 if super_boot else 0}\n'
              f'FLASH_LOCAL_UNIT_PATH="{flocal}"; FLASH_LOCAL_BOOT={1 if flocal_boot else 0}\n'
              f'ORNITH_122B_UNIT_PATH="{ornith}"; ORNITH_122B_BOOT={1 if ornith_boot else 0}\n'
              f'FLASH_UNSLOTH_UNIT_PATH="{unsloth}"; FLASH_UNSLOTH_BOOT={1 if unsloth_boot else 0}\n'
              f'LFM25_UNIT_PATH="{lfm25}"; LFM25_BOOT={1 if lfm25_boot else 0}\n'
              f'LFM3_UNIT_PATH="{lfm3}"; LFM3_BOOT={1 if lfm3_boot else 0}\n'
              # whether this run installs the Unsloth lane, and the target it hands its installer (given or moved, v1.49)
              f'WANT_FLASH_UNSLOTH={1 if unsloth_want else 0}\n'
              + (f'export FLASH_UNSLOTH_TARGET={unsloth_handed}\n' if unsloth_handed else "")
              + f'MODEL_CHOICE={"flash" if lane == "flash" else "stock"}\n'
              # the box's class, which install.sh reads before this block: the block reads it since
              # e65a364, and with no class here every generation stopped on an unbound variable
              # (found by the CI, 2026-10-24)
              'MEM_CLASS=128\n'
              f'REPO_DIR="{REPO}"; CONFIG_DIR="{t}"; OC_PORT=30001; OC_CTX={ctx}; OC_OUT={out}\n'
              f'OPENCODE_MAJOR={major}\n'
              + (f'OC_SERVE_MAJOR={serve_major}\n' if serve_major else "")
              + 'OC_LABEL=local; OPENCODE_PIN=1\n' + block())
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                       env={"PATH": "/usr/bin:/bin", "HOME": str(t)})
    assert r.returncode == 0, r.stdout + r.stderr
    doc_ = json.loads((t / "opencode.json").read_text())
    if doc:
        return doc_
    doc = doc_
    if names:
        return {p: b["models"][next(iter(b["models"]))]["name"] for p, b in doc["provider"].items()}
    return {p: b["models"][next(iter(b["models"]))]["limit"] for p, b in doc["provider"].items()}


class EachLaneKeepsItsOwnLimits(unittest.TestCase):
    def test_a_flash_install_keeps_a_1m_27b_at_its_1m_limits(self):
        lim = generate("flash", "native", unit_ctx=1010000)
        self.assertEqual(lim["qwen38"], {"context": 700000, "input": 700000, "output": 200000})
        self.assertEqual(lim["flashnext"]["context"], 205000)

    def test_a_flash_install_keeps_a_native_27b_native(self):
        lim = generate("flash", "native", unit_ctx=262144)
        self.assertEqual(lim["qwen38"], {"context": 173000, "input": 173000, "output": 64000})

    def test_a_flash_install_keeps_an_fp8_27b_at_the_fp8_pair(self):
        lim = generate("flash", "native", unit_ctx=1010000, model="Qwen/Qwen3.8-27B-FP8")
        self.assertEqual(lim["qwen38"], {"context": 480000, "input": 480000, "output": 160000})

    def test_the_27b_block_is_read_from_the_table(self):
        # the numbers themselves live in oc-limits.sh, once
        self.assertNotIn("(194048, 64000", INSTALL.read_text())
        self.assertNotIn("(700000, 200000", INSTALL.read_text())

    def test_the_flash_block_beside_a_27b_is_read_from_the_table_too(self):
        """A copy of one pair here, 110000/32000 (the throughput tier's), gave the flash its
        205,000 window cut to 110,000 at every run that served the 27B (2026-10-06)."""
        self.assertFalse("(110000, 32000)" in INSTALL.read_text(), "a flash pair is copied in install.sh")
        lim = generate("27b", "1m", unit_ctx=1010000, flash_unit=True, flash_running=8)
        self.assertEqual(lim["flashnext"], {"context": 205000, "input": 205000, "output": 32000})

    def test_the_flash_block_follows_the_tier_its_launcher_runs(self):
        lim = generate("27b", "1m", unit_ctx=1010000, flash_unit=True, flash_running=24)
        self.assertEqual(lim["flashnext"], {"context": 110000, "input": 110000, "output": 32000})

    def test_a_flash_only_box_lists_no_27b(self):
        self.assertNotIn("qwen38", generate("flash", "native"))

    def test_a_27b_install_uses_its_own_numbers(self):
        # This run's target, not the unit on disk, which step 8 rewrites after this block:
        # an update from the NVFP4 checkpoint to FP8 still finds the NVFP4 unit here. The
        # pair must differ from that unit's own for the test to see which one was used
        # (700000/200000 here was both, found in review, 2026-09-24).
        lim = generate("27b", "1m", unit_ctx=1010000, flash_unit=True, pair=("480000", "160000"))
        self.assertEqual(lim["qwen38"], {"context": 480000, "input": 480000, "output": 160000})
        self.assertIn("flashnext", lim)



class TheLaneNotInstalledKeepsItsCheckpointsName(unittest.TestCase):
    """A flash install named the 27B entry generically, whatever its unit serves; the next
    switch back to the 27B renamed it, so the first update after every switch rewrote the
    file and restarted opencode-web for a name (the reference box, 2026-09-30)."""

    def test_an_abliterated_27b_keeps_its_name_under_a_flash_install(self):
        names = generate("flash", "native", unit_ctx=1010000,
                         model="edp1096/Huihui-RadixArk-Qwen3.8-27B-abliterated-NVFP4", names=True)
        self.assertEqual(names["qwen38"], "Qwen3.8-27B NVFP4 abliterated + DFlash2 (local, 1M)")

    def test_arete_keeps_its_name_under_a_flash_install(self):
        names = generate("flash", "native", unit_ctx=1010000, model="atakans/Arete-27B-NVFP4", names=True)
        self.assertEqual(names["qwen38"], "Arete-27B NVFP4 + DFlash2 (local, 1M)")

    def test_daisyrose_keeps_its_name_under_a_flash_install(self):
        names = generate("flash", "native", unit_ctx=1010000, model="RadixArk/DaisyRose-27B-NVFP4", names=True)
        self.assertEqual(names["qwen38"], "DaisyRose-27B NVFP4 + DFlash2 (local, 1M)")

    def test_an_fp8_27b_is_named_fp8_and_a_native_one_says_its_window(self):
        names = generate("flash", "native", unit_ctx=262144, model="Qwen/Qwen3.8-27B-FP8", names=True)
        self.assertEqual(names["qwen38"], "Qwen3.8-27B FP8 official + DFlash2 (local, 262K)")


class ABoxBootingKolibri(unittest.TestCase):
    """Kolibri serving at boot: opencode's default is Kolibri, and the compaction block, global,
    is sized from its window (46,000 kept of its 185,000 threshold), not from the 27B's (132,000
    on a 1M 27B, and opencode compacted again every step or two: found in review, 2026-10-14)."""

    def test_the_default_and_the_compaction_are_kolibris(self):
        d = generate("27b", "1m", unit_ctx=1010000, flash_unit=True, pair=("551000", "160000"),
                     kolibri_unit=True, kolibri_boot=True, doc=True)
        self.assertEqual(d["model"], "kolibri/kolibri-1")
        self.assertIn("kolibri", d["provider"])
        self.assertEqual(d["compaction"]["preserve_recent_tokens"], 46000)

    def test_kolibris_pair_compacts_late_and_still_fits_its_window(self):
        # 173,000 + 64,000 compacted at 153,000 tokens, its 64,000 of answer unused (the longest
        # reply of a session on the reference box was 1,742 tokens): 205,000 + 32,000, the same
        # rule, compacts at 185,000 (Hasan, 2026-10-14)
        for target in ("kolibri", "kolibri-nvfp4"):
            ctx, out, *_ = subprocess.run([str(REPO / "oc-limits.sh"), target], capture_output=True,
                                          text=True, check=True).stdout.split()
            ctx, out = int(ctx), int(out)
            self.assertEqual((ctx, out), (205000, 32000), target)
            self.assertLessEqual(ctx - 20000 + 43863 + out, 262144, "threshold + worst step + answer fit the window")
            self.assertGreaterEqual(ctx - 20000, 185000)

    def test_with_kolibri_installed_but_not_at_boot_the_27b_keeps_its_own(self):
        d = generate("27b", "1m", unit_ctx=1010000, pair=("551000", "160000"), kolibri_unit=True, doc=True)
        self.assertEqual(d["model"], "qwen38/qwen3.8-27b")
        self.assertIn("kolibri", d["provider"], "the installed lane is offered")
        self.assertEqual(d["compaction"]["preserve_recent_tokens"], 132000)

    def test_without_its_unit_there_is_no_kolibri_provider(self):
        d = generate("27b", "1m", unit_ctx=1010000, pair=("551000", "160000"), doc=True)
        self.assertNotIn("kolibri", d["provider"])

class ABoxBootingNemotron(unittest.TestCase):
    """The Nemotron lane, the same way: serving at boot it is opencode's default, its compaction
    sized from its window; installed beside another lane it is offered; with no unit, absent."""

    def test_the_default_the_compaction_and_the_provider_are_nemotrons(self):
        d = generate("27b", "1m", unit_ctx=1010000, flash_unit=True, pair=("551000", "160000"),
                     nemotron_unit=True, nemotron_boot=True, doc=True)
        self.assertEqual(d["model"], "nemotron/nemotron-3.5-bolt")
        self.assertEqual(d["compaction"]["preserve_recent_tokens"], 46000)
        m = d["provider"]["nemotron"]["models"]["nemotron-3.5-bolt"]
        self.assertEqual(m["limit"], {"context": 205000, "input": 205000, "output": 32000})
        self.assertEqual(m["name"], "Nemotron 3.5 Bolt (local, 262K)")
        # it reads images, and its levels are its template's own words
        self.assertTrue(m["attachment"])
        self.assertEqual(m["modalities"]["input"], ["text", "image"])
        self.assertEqual(m["variants"]["none"], {"chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(m["variants"]["medium"], {"chat_template_kwargs": {"enable_thinking": True, "medium_effort": True}})
        self.assertEqual(m["variants"]["high"], {"chat_template_kwargs": {"enable_thinking": True}})
        self.assertEqual(d["provider"]["nemotron"]["options"]["baseURL"], "http://127.0.0.1:30001/v1")

    def test_its_pair_compacts_late_and_still_fits_its_window(self):
        ctx, out, *_ = subprocess.run([str(REPO / "oc-limits.sh"), "nemotron"], capture_output=True,
                                      text=True, check=True).stdout.split()
        ctx, out = int(ctx), int(out)
        self.assertEqual((ctx, out), (205000, 32000))
        self.assertLessEqual(ctx - 20000 + 43863 + out, 262144, "threshold + worst step + answer fit the window")

    def test_installed_but_not_at_boot_it_is_offered_and_the_27b_keeps_its_own(self):
        d = generate("27b", "1m", unit_ctx=1010000, pair=("551000", "160000"), nemotron_unit=True, doc=True)
        self.assertEqual(d["model"], "qwen38/qwen3.8-27b")
        self.assertIn("nemotron", d["provider"])
        self.assertEqual(d["compaction"]["preserve_recent_tokens"], 132000)

    def test_without_its_unit_there_is_no_nemotron_provider(self):
        d = generate("27b", "1m", unit_ctx=1010000, pair=("551000", "160000"), kolibri_unit=True, doc=True)
        self.assertNotIn("nemotron", d["provider"])


class ABoxBootingNemotronSuper(unittest.TestCase):
    """The Nemotron Super lane, its own provider (its own model and window): serving at boot it is
    opencode's default, its compaction sized from its window; installed beside another lane it is
    offered, Bolt's provider beside it; with no unit, absent."""

    def test_the_default_the_compaction_and_the_provider_are_super_vls(self):
        d = generate("27b", "1m", unit_ctx=1010000, flash_unit=True, pair=("551000", "160000"),
                     nemotron_unit=True, super_unit=True, super_boot=True, doc=True)
        self.assertEqual(d["model"], "nemotron-super/nemotron-3.5-super-vl")
        keep = int(subprocess.run([str(REPO / "oc-limits.sh"), "--preserve", "990000"], capture_output=True,
                                  text=True, check=True).stdout)
        self.assertEqual(d["compaction"]["preserve_recent_tokens"], keep)
        m = d["provider"]["nemotron-super"]["models"]["nemotron-3.5-super-vl"]
        self.assertEqual(m["limit"], {"context": 990000, "input": 990000, "output": 32000})
        self.assertEqual(m["name"], "Nemotron 3.5 Super VL (local, 1M)")
        self.assertTrue(m["attachment"])
        self.assertEqual(m["modalities"]["input"], ["text", "image"])
        self.assertEqual(m["variants"]["none"], {"chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(m["variants"]["low"], {"chat_template_kwargs": {"enable_thinking": True, "medium_effort": True}})
        self.assertEqual(m["variants"]["high"], {"chat_template_kwargs": {"enable_thinking": True}})
        self.assertEqual(d["provider"]["nemotron-super"]["name"], "Nemotron 3.5 Super VL (DGX Spark)")
        # Bolt's provider beside it, its own model and window
        self.assertIn("nemotron-3.5-bolt", d["provider"]["nemotron"]["models"])

    def test_its_pair_compacts_late_and_still_fits_its_window(self):
        ctx, out, *_ = subprocess.run([str(REPO / "oc-limits.sh"), "nemotron-super"], capture_output=True,
                                      text=True, check=True).stdout.split()
        ctx, out = int(ctx), int(out)
        self.assertEqual((ctx, out), (990000, 32000))
        self.assertLessEqual(ctx - 20000 + 43863 + out, 1048576, "threshold + worst step + answer fit the window")

    def test_installed_but_not_at_boot_it_is_offered_and_the_27b_keeps_its_own(self):
        d = generate("27b", "1m", unit_ctx=1010000, pair=("551000", "160000"), super_unit=True, doc=True)
        self.assertEqual(d["model"], "qwen38/qwen3.8-27b")
        self.assertIn("nemotron-super", d["provider"])
        self.assertNotIn("nemotron", d["provider"])
        self.assertEqual(d["compaction"]["preserve_recent_tokens"], 132000)

    def test_without_its_unit_there_is_no_super_provider(self):
        d = generate("27b", "1m", unit_ctx=1010000, pair=("551000", "160000"), nemotron_unit=True, doc=True)
        self.assertNotIn("nemotron-super", d["provider"])


class ABoxBootingFlashLocal(unittest.TestCase):
    """The Qwen3.8 Flash-Next FP4-Local lane, its own provider (its own served model): serving at boot it is
    opencode's default, its compaction sized from its window; installed beside another lane it is
    offered; with no unit, absent. Its levels are the flash lane's Qwen levels, xhigh first as since
    v1.37.0, whatever the lane's default (lean since v1.46.4): the order decides no request, and the
    generated config stays as it was."""

    def test_the_default_the_compaction_and_the_provider_are_flash_locals(self):
        d = generate("27b", "1m", unit_ctx=1010000, flash_unit=True, pair=("551000", "160000"),
                     flocal_unit=True, flocal_boot=True, doc=True)
        self.assertEqual(d["model"], "flash-local/qwen3.8-flash-next-fp4")
        keep = int(subprocess.run([str(REPO / "oc-limits.sh"), "--preserve", "205000"], capture_output=True,
                                  text=True, check=True).stdout)
        self.assertEqual(d["compaction"]["preserve_recent_tokens"], keep)
        m = d["provider"]["flash-local"]["models"]["qwen3.8-flash-next-fp4"]
        self.assertEqual(m["limit"], {"context": 205000, "input": 205000, "output": 32000})
        self.assertEqual(m["name"], "Qwen3.8-Flash-Next FP4-Local NVIDIA + MTP (local, 262K)")
        self.assertTrue(m["attachment"])
        self.assertEqual(m["modalities"]["input"], ["text", "image"])
        self.assertEqual(list(m["variants"]), ["xhigh", "lean", "low", "medium"])
        self.assertEqual(m["variants"]["lean"], {"chat_template_kwargs": {"reasoning_effort": "lean"}})
        self.assertEqual(d["provider"]["flash-local"]["name"], "Qwen3.8-Flash-Next FP4-Local (DGX Spark)")

    def test_its_pair_is_the_flash_lanes_and_fits_its_window(self):
        ctx, out, *_ = subprocess.run([str(REPO / "oc-limits.sh"), "flash-local"], capture_output=True,
                                      text=True, check=True).stdout.split()
        fctx, fout, *_ = subprocess.run([str(REPO / "oc-limits.sh"), "flash"], capture_output=True,
                                        text=True, check=True).stdout.split()
        self.assertEqual((int(ctx), int(out)), (int(fctx), int(fout)), "the same model and window as the flash lane")
        self.assertLessEqual(int(ctx) - 20000 + 43863 + int(out), 262144, "threshold + worst step + answer fit the window")

    def test_installed_but_not_at_boot_it_is_offered_and_the_27b_keeps_its_own(self):
        d = generate("27b", "1m", unit_ctx=1010000, pair=("551000", "160000"), flocal_unit=True, doc=True)
        self.assertEqual(d["model"], "qwen38/qwen3.8-27b")
        self.assertIn("flash-local", d["provider"])
        self.assertEqual(d["compaction"]["preserve_recent_tokens"], 132000)

    def test_without_its_unit_there_is_no_flash_local_provider(self):
        d = generate("27b", "1m", unit_ctx=1010000, pair=("551000", "160000"), nemotron_unit=True, doc=True)
        self.assertNotIn("flash-local", d["provider"])


class ABoxBootingOrnith(unittest.TestCase):
    """The Ornith 1.5 122B lane, its own provider (its own served model): serving at boot it is
    opencode's default, its compaction sized from its window; installed beside another lane it is
    offered; with no unit, absent. Its template reads enable_thinking and no effort: two variants,
    its full reasoning first, then none (decided 2026-10-21)."""

    def test_the_default_the_compaction_and_the_provider_are_orniths(self):
        d = generate("27b", "1m", unit_ctx=1010000, flash_unit=True, pair=("551000", "160000"),
                     ornith_unit=True, ornith_boot=True, doc=True)
        self.assertEqual(d["model"], "ornith-122b/ornith-1.5-122b")
        ctx, out, *_ = subprocess.run([str(REPO / "oc-limits.sh"), "ornith-122b"], capture_output=True,
                                      text=True, check=True).stdout.split()
        keep = int(subprocess.run([str(REPO / "oc-limits.sh"), "--preserve", ctx], capture_output=True,
                                  text=True, check=True).stdout)
        self.assertEqual(d["compaction"]["preserve_recent_tokens"], keep)
        m = d["provider"]["ornith-122b"]["models"]["ornith-1.5-122b"]
        self.assertEqual(m["limit"], {"context": int(ctx), "input": int(ctx), "output": int(out)})
        self.assertTrue(m["name"].startswith("Ornith 1.5 122B NVFP4 + MTP (local, "), m["name"])
        self.assertTrue(m["attachment"])
        self.assertEqual(m["modalities"]["input"], ["text", "image"])
        self.assertEqual(list(m["variants"]), ["high", "none"])
        self.assertEqual(m["variants"]["none"], {"chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(m["variants"]["high"], {"chat_template_kwargs": {"enable_thinking": True}})
        self.assertEqual(d["provider"]["ornith-122b"]["name"], "Ornith 1.5 122B (DGX Spark)")

    def test_its_pair_compacts_late_and_still_fits_its_window(self):
        # its window is 1,048,576 tokens (YaRN, four times its own), the Nemotron Super lane's, and the
        # same rule decides: the compaction threshold, one worst step and the answer fit inside it
        ctx, out, label = subprocess.run([str(REPO / "oc-limits.sh"), "ornith-122b"], capture_output=True,
                                         text=True, check=True).stdout.split(" ", 2)
        ctx, out = int(ctx), int(out)
        self.assertLessEqual(ctx - 20000 + 43863 + out, 1048576, "threshold + worst step + answer fit the window")
        self.assertGreater(ctx, 262144, "a pair of the native window would leave three quarters of it unused")
        self.assertEqual(label.strip(), "local, 1M")

    def test_installed_but_not_at_boot_it_is_offered_and_the_27b_keeps_its_own(self):
        d = generate("27b", "1m", unit_ctx=1010000, pair=("551000", "160000"), ornith_unit=True, doc=True)
        self.assertEqual(d["model"], "qwen38/qwen3.8-27b")
        self.assertIn("ornith-122b", d["provider"])
        self.assertEqual(d["compaction"]["preserve_recent_tokens"], 132000)

    def test_without_its_unit_there_is_no_ornith_provider(self):
        d = generate("27b", "1m", unit_ctx=1010000, pair=("551000", "160000"), flocal_unit=True, doc=True)
        self.assertNotIn("ornith-122b", d["provider"])


class ABoxBootingLfm25(unittest.TestCase):
    """The Liquid AI LFM2.5 24B-A2B lane, its own provider (its own served model): serving at boot it is opencode's
    default, its compaction sized from its window; installed beside another lane it is offered; with no unit, absent.
    Its template reads no level and no switch: no variant to offer."""

    def test_the_default_the_compaction_and_the_provider_are_its(self):
        d = generate("27b", "1m", unit_ctx=1010000, flash_unit=True, pair=("551000", "160000"),
                     lfm25_unit=True, lfm25_boot=True, doc=True)
        self.assertEqual(d["model"], "lfm25/lfm2.5-24b-a2b")
        ctx, out, *_ = subprocess.run([str(REPO / "oc-limits.sh"), "lfm25"], capture_output=True,
                                      text=True, check=True).stdout.split()
        keep = int(subprocess.run([str(REPO / "oc-limits.sh"), "--preserve", ctx], capture_output=True,
                                  text=True, check=True).stdout)
        self.assertEqual(d["compaction"]["preserve_recent_tokens"], keep)
        m = d["provider"]["lfm25"]["models"]["lfm2.5-24b-a2b"]
        self.assertEqual(m["limit"], {"context": 74000, "input": 74000, "output": 32000})
        self.assertEqual(m["name"], "Liquid AI LFM2.5 24B-A2B FP8 + DSpark (local, 131K)")
        self.assertNotIn("variants", m)
        self.assertNotIn("modalities", m)
        self.assertEqual(d["provider"]["lfm25"]["name"], "Liquid AI LFM2.5 24B-A2B (DGX Spark)")

    def test_installed_but_not_at_boot_it_is_offered_and_the_27b_keeps_its_own(self):
        d = generate("27b", "1m", unit_ctx=1010000, pair=("551000", "160000"), lfm25_unit=True, doc=True)
        self.assertEqual(d["model"], "qwen38/qwen3.8-27b")
        self.assertIn("lfm25", d["provider"])
        self.assertEqual(d["compaction"]["preserve_recent_tokens"], 132000)

    def test_without_its_unit_there_is_no_lfm25_provider(self):
        d = generate("27b", "1m", unit_ctx=1010000, pair=("551000", "160000"), ornith_unit=True, doc=True)
        self.assertNotIn("lfm25", d["provider"])


class ABoxBootingLfm3(unittest.TestCase):
    """The Liquid AI LFM3 lane, its own provider (its own served model): serving at boot it is opencode's default, its
    compaction sized from its window; installed beside another lane it is offered; with no unit, absent. Its template
    reads enable_thinking alone: two variants, the reasoning first, then none."""

    def test_the_default_the_compaction_and_the_provider_are_its(self):
        d = generate("27b", "1m", unit_ctx=1010000, flash_unit=True, pair=("551000", "160000"),
                     lfm3_unit=True, lfm3_boot=True, doc=True)
        self.assertEqual(d["model"], "lfm3/lfm3-1.7b")
        ctx, out, *_ = subprocess.run([str(REPO / "oc-limits.sh"), "lfm3"], capture_output=True,
                                      text=True, check=True).stdout.split()
        keep = int(subprocess.run([str(REPO / "oc-limits.sh"), "--preserve", ctx], capture_output=True,
                                  text=True, check=True).stdout)
        self.assertEqual(d["compaction"]["preserve_recent_tokens"], keep)
        m = d["provider"]["lfm3"]["models"]["lfm3-1.7b"]
        self.assertEqual(m["limit"], {"context": 205000, "input": 205000, "output": 32000})
        self.assertEqual(m["name"], "Liquid AI LFM3 1.7B NVFP4 (local, 262K)")
        self.assertEqual(list(m["variants"]), ["high", "none"])
        self.assertEqual(m["variants"]["none"], {"chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(m["variants"]["high"], {"chat_template_kwargs": {"enable_thinking": True}})
        self.assertNotIn("modalities", m)
        self.assertEqual(d["provider"]["lfm3"]["name"], "Liquid AI LFM3 (DGX Spark)")

    def test_installed_but_not_at_boot_it_is_offered_and_the_27b_keeps_its_own(self):
        d = generate("27b", "1m", unit_ctx=1010000, pair=("551000", "160000"), lfm3_unit=True, doc=True)
        self.assertEqual(d["model"], "qwen38/qwen3.8-27b")
        self.assertIn("lfm3", d["provider"])

    def test_without_its_unit_there_is_no_lfm3_provider(self):
        d = generate("27b", "1m", unit_ctx=1010000, pair=("551000", "160000"), lfm25_unit=True, doc=True)
        self.assertNotIn("lfm3", d["provider"])


class ABoxBootingFlashUnsloth(unittest.TestCase):
    """The Qwen3.8 Flash-Next Unsloth GGUF lane (llama.cpp, v1.48), its own provider (its own served model): serving at
    boot it is opencode's default, its compaction sized from its window; installed beside another lane it is offered;
    with no unit, absent. Its template is Qwen's with the same fixes, so the 27B's four levels in the 27B's order, and
    text alone: the lane loads no vision projector."""

    def test_the_default_the_compaction_and_the_provider_are_its(self):
        # its window: by default 262,144 for each of its two requests, the native pair, and one request's 1,048,576 in
        # its 1M mode, the 1m pair (decisions 1316 and 1318, 2026-11-12)
        for slots, mode, said in (("-c 524288 -np 2", "native", "262K"),
                                  ("-c 1048576 -np 1 -ctk q8_0 -ctv q8_0", "1m", "1M")):
            with self.subTest(mode=mode):
                d = generate("27b", "1m", unit_ctx=1010000, flash_unit=True, pair=("551000", "160000"),
                             unsloth_unit=True, unsloth_boot=True, unsloth_slots=slots, doc=True)
                self.assertEqual(d["model"], "flash-unsloth/qwen3.8-flash-next-unsloth")
                ctx, out, *_ = subprocess.run([str(REPO / "oc-limits.sh"), "flash-unsloth", mode], capture_output=True,
                                              text=True, check=True).stdout.split()
                keep = int(subprocess.run([str(REPO / "oc-limits.sh"), "--preserve", ctx], capture_output=True,
                                          text=True, check=True).stdout)
                self.assertEqual(d["compaction"]["preserve_recent_tokens"], keep)
                m = d["provider"]["flash-unsloth"]["models"]["qwen3.8-flash-next-unsloth"]
                self.assertEqual(m["limit"], {"context": int(ctx), "input": int(ctx), "output": int(out)})
                self.assertEqual(m["name"], f"Qwen3.8-Flash-Next Unsloth UD-Q4_K_XL + DFlash2 (local, {said})")
        self.assertFalse(m["attachment"])
        self.assertEqual(m["modalities"], {"input": ["text"], "output": ["text"]})
        self.assertEqual(list(m["variants"]), ["lean", "low", "medium", "xhigh"])
        self.assertEqual(m["variants"]["lean"], {"chat_template_kwargs": {"reasoning_effort": "lean"}})
        self.assertEqual(d["provider"]["flash-unsloth"]["name"], "Qwen3.8-Flash-Next Unsloth GGUF (DGX Spark)")

    def test_its_pair_is_the_flash_lanes_and_fits_its_window(self):
        ctx, out, label = subprocess.run([str(REPO / "oc-limits.sh"), "flash-unsloth"], capture_output=True,
                                         text=True, check=True).stdout.split(" ", 2)
        fctx, fout, *_ = subprocess.run([str(REPO / "oc-limits.sh"), "flash"], capture_output=True,
                                        text=True, check=True).stdout.split()
        self.assertEqual((int(ctx), int(out)), (int(fctx), int(fout)), "the same model and window as the flash lane")
        self.assertLessEqual(int(ctx) - 20000 + 43863 + int(out), 262144, "threshold + worst step + answer fit the window")
        self.assertEqual(label.strip(), "local, 262K")
        # its 1M mode's 1,048,576 for one request by YaRN (decisions 1316 and 1318, 2026-11-12), the pool there (no
        # flash ceiling), the pair sized by the FP4-Local lane's rule at its 1M: its pool less 10 % holds context and
        # answer, and its window the worst step too; the lane's answer budget kept
        ctx, out, label = subprocess.run([str(REPO / "oc-limits.sh"), "flash-unsloth", "1m"], capture_output=True,
                                         text=True, check=True).stdout.split(" ", 2)
        self.assertLessEqual(int(ctx) + int(out), int(1048576 * 0.92 * 0.90), "its pool less 10 %")
        self.assertLessEqual(int(ctx) - 20000 + 43863 + int(out), 1048576, "with the worst step, its window")
        self.assertEqual((int(out), label.strip()), (int(fout), "local, 1M"))

    def test_installed_but_not_at_boot_it_is_offered_and_the_27b_keeps_its_own(self):
        d = generate("27b", "1m", unit_ctx=1010000, pair=("551000", "160000"), unsloth_unit=True, doc=True)
        self.assertEqual(d["model"], "qwen38/qwen3.8-27b")
        self.assertIn("flash-unsloth", d["provider"])
        self.assertEqual(d["compaction"]["preserve_recent_tokens"], 132000)

    def test_a_window_for_each_request_is_read_as_one_requests(self):
        # FLASH_UNSLOTH_SHARED=0, the lane's (2026-11-12): -c holds the windows of all its requests, 524,288 tokens for
        # two, 1,048,576 for four; read as the window, opencode's name said "1M" for a lane whose request gets 262,144.
        # Shared (-kvu), -c is the one window.
        for slots in ("-c 524288 -np 2", "-c 1048576 -np 4 -ctk q8_0 -ctv q8_0", "-c 262144 -np 4 -kvu"):
            with self.subTest(slots=slots):
                d = generate("27b", "1m", unit_ctx=1010000, pair=("551000", "160000"), unsloth_unit=True,
                             unsloth_boot=True, unsloth_slots=slots, doc=True)
                m = d["provider"]["flash-unsloth"]["models"]["qwen3.8-flash-next-unsloth"]
                self.assertEqual(m["name"], "Qwen3.8-Flash-Next Unsloth UD-Q4_K_XL + DFlash2 (local, 262K)")

    def test_its_entry_names_the_target_its_unit_serves(self):
        # UD-IQ3_XXS (v1.49), its GGUF in that folder of Unsloth's repository: its four windows of 262,144 the native
        # pair, its 1M mode the 1m pair, under its own name, and opencode's default stays the lane's one served name
        iq3 = "snapshots/" + "d" * 40 + "/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf"
        for slots, mode, said in (("-c 1048576 -np 4 -ctk q8_0 -ctv q8_0", "native", "262K"),
                                  ("-c 1048576 -np 1 -ctk q8_0 -ctv q8_0", "1m", "1M")):
            with self.subTest(mode=mode):
                d = generate("27b", "1m", unit_ctx=1010000, pair=("551000", "160000"), unsloth_unit=True,
                             unsloth_boot=True, unsloth_slots=slots, unsloth_model=iq3, doc=True)
                ctx, out, *_ = subprocess.run([str(REPO / "oc-limits.sh"), "flash-unsloth-iq3", mode], capture_output=True,
                                              text=True, check=True).stdout.split()
                m = d["provider"]["flash-unsloth"]["models"]["qwen3.8-flash-next-unsloth"]
                self.assertEqual(m["limit"], {"context": int(ctx), "input": int(ctx), "output": int(out)})
                self.assertEqual(m["name"], f"Qwen3.8-Flash-Next Unsloth UD-IQ3_XXS + DFlash2 (local, {said})")
                self.assertEqual(d["model"], "flash-unsloth/qwen3.8-flash-next-unsloth")

    def test_the_entry_names_what_the_unit_serves_even_when_the_run_hands_another(self):
        # a UD-Q4_K_XL unit this run moves to UD-IQ3_XXS (an old default, v1.49.0): the entry is written before the
        # lane's installer runs, so it names the unit as it stands; unsloth_clients_follow brings it to the rendered
        # target once the installer has succeeded (independent review of v1.49.0, findings 1 and 2)
        for want, handed in ((True, "flash-unsloth-iq3"), (False, "flash-unsloth-iq3"), (True, "")):
            with self.subTest(want=want, handed=handed):
                d = generate("27b", "1m", unit_ctx=1010000, pair=("551000", "160000"), unsloth_unit=True,
                             unsloth_boot=True, unsloth_want=want, unsloth_handed=handed, doc=True)
                m = d["provider"]["flash-unsloth"]["models"]["qwen3.8-flash-next-unsloth"]
                self.assertTrue(m["name"].startswith("Qwen3.8-Flash-Next Unsloth UD-Q4_K_XL + DFlash2"), m["name"])

    def test_a_unit_naming_neither_folder_takes_the_installer_default_name(self):
        # a hand-edited unit with a GGUF in no target's folder: the entry names the lane's default (v1.49.0),
        # read once from the installer by install.sh (FLASH_UNSLOTH_DEFAULT_TARGET)
        d = generate("27b", "1m", unit_ctx=1010000, pair=("551000", "160000"), unsloth_unit=True,
                     unsloth_boot=True, unsloth_model="snapshots/" + "d" * 40 + "/own-build/model.gguf", doc=True)
        m = d["provider"]["flash-unsloth"]["models"]["qwen3.8-flash-next-unsloth"]
        self.assertTrue(m["name"].startswith("Qwen3.8-Flash-Next Unsloth UD-IQ3_XXS + DFlash2"), m["name"])

    def test_serving_at_boot_its_entry_is_pointed_at_under_the_name_of_its_target(self):
        # install.sh's own lines that point opencode's default at the lane that boots, and name its entry: the name of
        # the target its unit serves, UD-IQ3_XXS's (v1.49), where they named UD-Q4_K_XL's whatever the unit served
        text = INSTALL.read_text()
        start = text.index('OC_POINT_LANE="$LANE"; OC_POINT_CHOICE="$MODEL_CHOICE"; OC_POINT_WINDOW="$OC_WINDOW"\n')
        end = text.index('  || die "could not point the generated opencode config at the installed lane"\n', start)
        d = pathlib.Path(tempfile.mkdtemp(prefix="oc-art-point-"))
        self.addCleanup(shutil.rmtree, d, True)
        cfg = d / "opencode.json"
        cfg.write_text(json.dumps({"model": "qwen38/qwen3.8-27b", "provider": {
            "qwen38": {"models": {"qwen3.8-27b": {"name": "x"}}},
            "flash-unsloth": {"models": {"qwen3.8-flash-next-unsloth": {"name": "y"}}}}}, indent=2) + "\n")
        off = " ".join(f"{lane}_BOOT=0 OC_{lane}=0" for lane in ("KOLIBRI", "NEMOTRON", "NEMOTRON_SUPER", "FLASH_LOCAL", "ORNITH_122B",
                                                                         "LFM25", "LFM3"))
        script = "\n".join(["set -euo pipefail", 'die(){ echo "DIE: $*"; exit 1; }', f'REPO_DIR="{REPO}"; CONFIG_DIR="{d}"',
                             "LANE=27b MODEL_CHOICE=stock OC_WINDOW=1010000 " + off,
                             "FLASH_UNSLOTH_BOOT=1 OC_FLASH_UNSLOTH=1 OC_FLASH_UNSLOTH_TARGET=flash-unsloth-iq3 "
                             "OC_FLASH_UNSLOTH_WINDOW=262144", text[start:end] + "  || die point"])
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                           env={"PATH": "/usr/bin:/bin", "HOME": str(d)})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        doc = json.loads(cfg.read_text())
        self.assertEqual(doc["model"], "flash-unsloth/qwen3.8-flash-next-unsloth")
        self.assertEqual(doc["provider"]["flash-unsloth"]["models"]["qwen3.8-flash-next-unsloth"]["name"],
                         "Qwen3.8-Flash-Next Unsloth UD-IQ3_XXS + DFlash2 (local, 262K)")

    def test_without_its_unit_there_is_no_unsloth_provider(self):
        d = generate("27b", "1m", unit_ctx=1010000, pair=("551000", "160000"), ornith_unit=True, doc=True)
        self.assertNotIn("flash-unsloth", d["provider"])



class TheGeneratedConfigCarriesTheShellRules(unittest.TestCase):
    """Cohere's report (2026-10-26): oc and the Agent tab read this file over the user's own config,
    so the shell rules for the report's hosts, under a key of their own, come after the user's rules
    there whatever those say (docs/agents-and-network.md)."""

    def test_the_rules_are_in_it_under_their_own_key(self):
        d = generate("27b", "1m", unit_ctx=1010000, pair=("551000", "160000"), doc=True)
        self.assertEqual(d["permission"],
                         {"bash*": {"*aliyuncs.com*": "deny", "*169.254.169.254*": "deny",
                                    "*100.100.100.200*": "deny"},
                          "webfetch*": {"*aliyuncs.com*": "deny", "*169.254.169.254*": "deny",
                                        "*100.100.100.200*": "deny"}})


class TheV2CompactionBlock(unittest.TestCase):
    """On a box that chose OpenCode 2 (--opencode 2) the generated config carries the v2
    compaction block: keep.tokens sized by the table, background pinned true (the 2.3.2
    default; the precomputed summary is what lets the auto compaction swap in at once,
    reading the cached prefix), and no v1 key (2.x maps preserve_recent_tokens as a
    deprecated legacy and drops prune with a warning). The v1 block is unchanged."""

    def test_major_2_writes_keep_tokens_and_background(self):
        d = generate("27b", "1m", unit_ctx=1010000, flash_unit=True, pair=("551000", "160000"),
                     doc=True, major="2")
        self.assertEqual(d["compaction"], {"keep": {"tokens": 132000}, "background": True})

    def test_major_2_sizes_keep_from_the_booted_lane(self):
        d = generate("27b", "1m", unit_ctx=1010000, flash_unit=True, pair=("551000", "160000"),
                     kolibri_unit=True, kolibri_boot=True, doc=True, major="2")
        self.assertEqual(d["compaction"], {"keep": {"tokens": 46000}, "background": True})
        self.assertEqual(d["model"], "kolibri/kolibri-1")

    def test_major_1_keeps_the_v1_block(self):
        d = generate("27b", "1m", unit_ctx=1010000, flash_unit=True, pair=("551000", "160000"),
                     doc=True, major="1")
        self.assertEqual(d["compaction"], {"preserve_recent_tokens": 132000, "prune": False})

    def test_no_major_given_defaults_to_the_v1_block(self):
        d = generate("27b", "1m", unit_ctx=1010000, flash_unit=True, pair=("551000", "160000"), doc=True)
        self.assertEqual(d["compaction"], {"preserve_recent_tokens": 132000, "prune": False})


class TheGeneratedConfigFollowsTheBinaryThatStands(unittest.TestCase):
    """A crossing whose download failed leaves the old binary in place, and install.sh writes
    the generated config for THAT major (OC_SERVE_MAJOR), never for the one it was asked to
    switch to: while a 2.x binary stands the v2 keys and the exfiltration denies are in the
    config (the guard holds), and while a 1.x binary stands no v2 key is written at all
    (1.19 refuses to start on "permissions", measured 2026-11-14). Blind review 2026-11-20,
    D2/D3."""

    def test_serve_2_writes_the_v2_keys_and_the_denies(self):
        d = generate("27b", "1m", unit_ctx=1010000, pair=("551000", "160000"),
                     doc=True, major="1", serve_major="2")
        self.assertEqual(d["compaction"], {"keep": {"tokens": 132000}, "background": True})
        self.assertEqual([(p["action"], p["resource"]) for p in d["permissions"]],
                         [(a, r) for a in ("shell", "webfetch")
                          for r in ("*aliyuncs.com*", "*169.254.169.254*", "*100.100.100.200*")])
        self.assertEqual(d["policies"],
                         [{"action": "provider.use", "resource": "opencode", "effect": "deny"}])
        self.assertEqual(d["share"], "disabled")

    def test_serve_1_writes_no_v2_key_at_all(self):
        d = generate("27b", "1m", unit_ctx=1010000, pair=("551000", "160000"),
                     doc=True, major="2", serve_major="1")
        self.assertEqual(d["compaction"], {"preserve_recent_tokens": 132000, "prune": False})
        self.assertNotIn("permissions", d)
        self.assertNotIn("policies", d)
        self.assertNotIn("share", d)
        self.assertNotIn("websearch", d)
        # the v1 denies hold too: the guard is the "permission" key on a 1.x box
        self.assertEqual(d["permission"]["bash*"],
                         {"*aliyuncs.com*": "deny", "*169.254.169.254*": "deny",
                          "*100.100.100.200*": "deny"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
