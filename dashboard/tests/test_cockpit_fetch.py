#!/usr/bin/env python3
"""Library's Download: for each target a Load can switch to, whether its checkpoint is in the
caches the lanes read (whole for what the lane serves, begun, or absent) and whether its lane
is installed; and the action that fetches it, switch-model.sh <target> --fetch, which changes
nothing else. The target list is the switch's own."""
import importlib.util
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
DASH = HERE.parents[1]
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from fake_time import cockpit_time  # noqa: E402
sys.path.insert(0, str(DASH))
ENV_BEFORE = dict(os.environ)


def tearDownModule():
    os.environ.clear()
    os.environ.update(ENV_BEFORE)



class Fetch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="cockpit-fetch-"))
        (cls.tmp / "config").mkdir()
        (cls.tmp / "config" / "api-key").write_text("k\n")
        closed = "http://127.0.0.1:1"
        os.environ.update(HOME=str(cls.tmp), COCKPIT_DRY_RUN="1", COCKPIT_CONFIG_DIR=str(cls.tmp / "config"),
                          COCKPIT_REPO_DIR=str(REPO), COCKPIT_PORT="0", COCKPIT_AGENT_PORT="0",
                          COCKPIT_AUTOHEAL="0", COCKPIT_ENGINE=closed, COCKPIT_PROXY=closed, COCKPIT_IMAGE=closed)
        spec = importlib.util.spec_from_file_location("cockpit_fetch_under_test", DASH / "cockpit.py")
        cls.cp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.cp)
        cls.cp.run = lambda argv, timeout=5.0, merge_err=False: ""
        cls.units = cls.tmp / "units"
        cls.units.mkdir()
        # Every unit the cockpit reads, this test's: the box's own Nemotron unit, once installed, named
        # its real cache (found on the reference box, 2026-10-16), and a fixed list misses the next lane's.
        for name, value in list(vars(cls.cp).items()):
            if name.endswith("_UNIT_PATH") and isinstance(value, Path):
                setattr(cls.cp, name, cls.units / value.name)
        cls.cp.UNIT_PATHS = {"qwen38-sglang.service": cls.units / "qwen38-sglang.service",
                             "qwen38-flash.service": cls.units / "launch-flash.sh"}
        cls.pins = cls.cp.registry_snapshot(max_age=0.0)["pins"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.hub = self.tmp / ".cache" / "huggingface" / "hub"
        shutil.rmtree(self.hub, ignore_errors=True)
        self.hub.mkdir(parents=True)
        for f in self.units.iterdir():
            f.unlink()

    def snap(self, repo, rev, whole=True, part=""):
        d = self.hub / ("models--" + repo.replace("/", "--")) / "snapshots" / rev / part
        d.mkdir(parents=True)
        (d / "model.safetensors.index.json").write_text('{"weight_map": {"a": "1.safetensors", "b": "2.safetensors"}}')
        (d / "1.safetensors").write_text("w")
        if whole:
            (d / "2.safetensors").write_text("w")

    def targets(self):
        return {t["target"]: t for t in self.cp.registry_snapshot(max_age=0.0)["targets"]}

    def test_every_switch_target_is_listed_once(self):
        self.assertEqual(sorted(self.targets()), sorted(self.cp.ACTIONS["switch"]["params"]["target"]))

    def test_whole_begun_or_absent_and_whether_its_lane_is_installed(self):
        self.snap("RadixArk/Qwen3.8-27B-NVFP4", self.pins["STOCK_REV"])
        self.snap("Aleph-Alpha/Kolibri-1", self.pins["KOLIBRI_FP8_REV"], whole=False)
        (self.units / "qwen38-sglang.service").write_text("[Service]\n")
        (self.units / "qwen38-kolibri.service").write_text("[Service]\n")
        t = self.targets()
        self.assertEqual((t["stock"]["state"], t["stock"]["installed"]), ("complete", True))
        self.assertEqual((t["kolibri"]["state"], t["kolibri"]["installed"]), ("incomplete", True))
        self.assertEqual((t["kolibri-nvfp4"]["state"], t["kolibri-nvfp4"]["installed"]), ("absent", True))
        self.assertEqual((t["image"]["state"], t["image"]["installed"]), ("absent", False))
        self.assertEqual(t["flash"]["installed"], False)

    def test_the_video_lane_is_whole_for_the_partition_it_serves(self):
        rev = self.pins["VIDEO_MODEL_PIN_REV"]
        self.snap("MiniMaxAI/MiniMax-H3", rev, part="FL2VA/transformer")
        self.snap("MiniMaxAI/MiniMax-H3", rev, whole=False, part="Ref2VA/transformer")
        (self.hub / "models--MiniMaxAI--MiniMax-H3" / "snapshots" / rev / "model_index.json").write_text("{}")
        (self.units / "qwen38-video.service").write_text(
            "[Service]\nExecStart=/x/python -m sglang.multimodal_gen.runtime.launch_server "
            "--model-path MiniMaxAI/MiniMax-H3 --model-variant fl2va\n")
        self.assertEqual(self.targets()["video"]["state"], "complete")

    def test_a_nemotron_target_is_whole_with_its_drafter_whole_too(self):
        # the two Nemotron lanes load a drafter with the checkpoint, and their Download fetches the
        # two: Bolt's read "on disk" with its drafter missing (found on the reference box, 2026-10-15)
        for target, model, draft in (
                ("nemotron", "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB", "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr"),
                ("nemotron-super", "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4",
                 "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr")):
            pin = self.cp.TARGET_PIN[target]
            dpin = self.cp.TARGET_DRAFT_PIN[target]
            with self.subTest(target=target):
                self.setUp()
                self.assertEqual(self.targets()[target]["state"], "absent")
                self.snap(model, self.pins[pin])
                self.assertEqual(self.targets()[target]["state"], "incomplete", "its drafter is missing")
                self.snap(draft, self.pins[dpin], whole=False)
                self.assertEqual(self.targets()[target]["state"], "incomplete", "its drafter is begun")
                shutil.rmtree(self.hub / ("models--" + draft.replace("/", "--")))
                self.snap(draft, self.pins[dpin])
                self.assertEqual(self.targets()[target]["state"], "complete")
                shutil.rmtree(self.hub / ("models--" + model.replace("/", "--")))
                self.assertEqual(self.targets()[target]["state"], "incomplete", "its drafter alone")
        # a target with no drafter of its own reads its checkpoint alone, as before
        self.setUp()
        self.snap("RadixArk/Qwen3.8-27B-NVFP4", self.pins["STOCK_REV"])
        self.assertEqual(self.targets()["stock"]["state"], "complete")

    def test_the_unsloth_lanes_two_targets_each_count_their_own_parts(self):
        # UD-Q4_K_XL and UD-IQ3_XXS, two folders of one repository at one revision (v1.49): either read whole once the
        # other's parts were there, and the Library offered no Download of a target its Load then fetched (55 GB)
        import re
        text = (REPO / "install-flash-unsloth.sh").read_text()
        parts = {t: re.findall(rf'^{v}[1-4]="([^"=]+)=', text, re.M)
                 for t, v in (("flash-unsloth", "FLASH_UNSLOTH_PART"), ("flash-unsloth-iq3", "FLASH_UNSLOTH_IQ3_PART"))}
        self.assertEqual([len(p) for p in parts.values()], [4, 4])
        snap = self.hub / "models--unsloth--Qwen3.8-Flash-Next-GGUF" / "snapshots" / self.pins["FLASH_UNSLOTH_REV"]
        draft = (self.hub / "models--ggml-org--Qwen3.8-Flash-Next-GGUF" / "snapshots" / self.pins["FLASH_UNSLOTH_DRAFT_REV"]
                 / re.search(r'^FLASH_UNSLOTH_DRAFT="([^"=]+)=', text, re.M).group(1))
        draft.parent.mkdir(parents=True)
        draft.write_text("w")

        def put(paths):
            for rel in paths:
                (snap / rel).parent.mkdir(parents=True, exist_ok=True)
                (snap / rel).write_text("w")

        def states():
            t = self.targets()
            return t["flash-unsloth"]["state"], t["flash-unsloth-iq3"]["state"]

        put(parts["flash-unsloth-iq3"][:2])
        self.assertEqual(states(), ("absent", "incomplete"))
        put(parts["flash-unsloth-iq3"][2:])
        self.assertEqual(states(), ("absent", "complete"))
        put(parts["flash-unsloth"])
        self.assertEqual(states(), ("complete", "complete"))
        shutil.rmtree(snap / parts["flash-unsloth-iq3"][0].split("/")[0])
        self.assertEqual(states(), ("complete", "absent"))
        # the draft both load: whole with it alone, and not taken for a download of the other begun
        draft.unlink()
        self.assertEqual(states(), ("incomplete", "absent"))

    def gguf_on_disk(self):
        """Both GGUF targets whole in the cache, with their draft; the installer's text."""
        import re
        text = (REPO / "install-flash-unsloth.sh").read_text()
        snap = self.hub / "models--unsloth--Qwen3.8-Flash-Next-GGUF" / "snapshots" / self.pins["FLASH_UNSLOTH_REV"]
        for rel in re.findall(r'^FLASH_UNSLOTH_(?:IQ3_)?PART[1-4]="([^"=]+)=', text, re.M):
            (snap / rel).parent.mkdir(parents=True, exist_ok=True)
            (snap / rel).write_text("w")
        draft = (self.hub / "models--ggml-org--Qwen3.8-Flash-Next-GGUF" / "snapshots" / self.pins["FLASH_UNSLOTH_DRAFT_REV"]
                 / re.search(r'^FLASH_UNSLOTH_DRAFT="([^"=]+)=', text, re.M).group(1))
        draft.parent.mkdir(parents=True)
        draft.write_text("w")
        return text

    def half_repo(self, content, name="install-flash-unsloth.sh"):
        """The repository with one script as a checkout may leave it for an instant: gone (None), or this text."""
        half = self.tmp / "half-repo"
        shutil.rmtree(half, ignore_errors=True)
        half.mkdir()
        for f in REPO.iterdir():
            if f.name != name:
                (half / f.name).symlink_to(f)
        if content is not None:
            (half / name).write_text(content)
        real = self.cp.REPO_DIR
        self.cp.REPO_DIR = half
        self.addCleanup(setattr, self.cp, "REPO_DIR", real)
        return half

    BOTH = ("flash-unsloth", "flash-unsloth-iq3")

    def states(self, reg):
        return {t["target"]: t["state"] for t in reg["targets"] if t["target"] in self.BOTH}

    def test_an_installer_read_cut_short_concludes_nothing_of_what_it_misses(self):
        # a checkout rewriting install-flash-unsloth.sh, caught gone, empty or cut short on every read: the two GGUF
        # targets read absent and the Library offered to fetch 85 GB on the disk (second and third reviews of v1.49.0).
        # Each such read leaves them unknown and says what it missed; the other targets are read as ever.
        text = self.gguf_on_disk()
        reg = self.cp.registry_snapshot(max_age=0.0)
        self.assertEqual((self.states(reg), reg["partial"]), (dict.fromkeys(self.BOTH, "complete"), []))
        cut = {"gone": None, "empty": "",
               "before its parts": text[:text.index("FLASH_UNSLOTH_PART1=")],
               "inside its parts": text[:text.index("FLASH_UNSLOTH_IQ3_PART3=")],
               "before its draft's pin": text[:text.index("FLASH_UNSLOTH_DRAFT_REV=")],
               # every pin there but one part's line: only the parts tell
               "one part's line gone": "".join(ln for ln in text.splitlines(keepends=True)
                                               if not ln.startswith("FLASH_UNSLOTH_IQ3_PART4="))}
        cockpit_time(self.cp, self, sleep=lambda s: None)
        for how, content in cut.items():
            with self.subTest(how):
                self.half_repo(content)
                reg = self.cp.registry_snapshot(max_age=0.0)
                self.assertTrue(reg["partial"], "the scan says what it missed")
                # unknown, each target whose pins or parts were not all read; the other, read whole, as ever
                want = dict.fromkeys(self.BOTH, "unknown")
                if how == "one part's line gone":
                    want["flash-unsloth"] = "complete"
                self.assertEqual(self.states(reg), want)
                self.assertEqual(next(t["state"] for t in reg["targets"] if t["target"] == "stock"), "absent",
                                 "a target whose pins were read is read as ever")
                self.assertIn("flash-unsloth-iq3", [t["target"] for t in reg["targets"]])

    def test_a_checkout_that_ends_between_two_reads_is_read_whole(self):
        # the pause between two reads is the instant the checkout needs: the second read finds the file whole
        text = self.gguf_on_disk()
        half = self.half_repo("")
        cockpit_time(self.cp, self, sleep=lambda s: (half / "install-flash-unsloth.sh").write_text(text))
        reg = self.cp.registry_snapshot(max_age=0.0)
        self.assertEqual((self.states(reg), reg["partial"]), (dict.fromkeys(self.BOTH, "complete"), []))

    def test_pins_that_change_between_two_reads_are_a_checkout_in_progress(self):
        # every file read whole, but install.sh of one release and switch-model.sh of the next: a second read finds other
        # pins, and the scan says so (sixth review)
        real = self.cp._read_once
        self.addCleanup(setattr, self.cp, "_read_once", real)
        calls = []

        def moving():
            read = real()
            calls.append(1)
            if len(calls) % 2:
                read["pins"] = {**read["pins"], "STOCK_REV": "f" * 40}
                read["texts"] = {**read["texts"], "install.sh": read["texts"]["install.sh"] + "\n# the next release"}
            return read
        self.cp._read_once = moving
        cockpit_time(self.cp, self, sleep=lambda s: None)
        reg = self.cp.registry_snapshot(max_age=0.0)
        self.assertIn("the scripts changed while read (a checkout?): install.sh", reg["partial"])
        state = {t["target"]: t["state"] for t in reg["targets"]}
        self.assertEqual((state["stock"], state["flash"]), ("unknown", "absent"),
                         "the target whose pin moved is not read; the others are read as ever")
        # once it holds still, the second read agrees and the scan is whole
        self.cp._read_once = real
        self.assertEqual(self.cp.registry_snapshot(max_age=0.0)["partial"], [])

    def test_a_second_read_that_misses_a_file_says_that_file(self):
        # the checkout removes install.sh between the two reads: the cause said is that file, not "changed while read"
        real = self.cp._read_once
        self.addCleanup(setattr, self.cp, "_read_once", real)
        calls = []

        def second_misses():
            calls.append(1)
            if len(calls) % 2 == 0:
                self.half_repo(None, name="install.sh")
                try:
                    return real()
                finally:
                    self.cp.REPO_DIR = REPO
            return real()
        self.cp._read_once = second_misses
        cockpit_time(self.cp, self, sleep=lambda s: None)
        reg = self.cp.registry_snapshot(max_age=0.0)
        self.assertIn("install.sh not read", reg["partial"])
        self.assertFalse([m for m in reg["partial"] if "changed while read" in m])
        self.assertTrue(reg["pins_lost"])

    def test_a_scan_that_missed_a_pin_is_kept_seconds_not_minutes(self):
        # the checkout is over in an instant: kept the cache's 5 minutes, its "not read" outlived it (fifth review)
        self.gguf_on_disk()
        cockpit_time(self.cp, self, sleep=lambda s: None)
        self.half_repo("")
        self.assertTrue(self.cp.registry_snapshot(max_age=0.0)["partial"])
        self.cp.REPO_DIR = REPO
        self.assertTrue(self.cp.registry_snapshot()["partial"], "within its seconds, the same scan")
        self.cp.REGISTRY_CACHE["ts"] -= self.cp.PARTIAL_TTL + 1
        reg = self.cp.registry_snapshot()
        self.assertEqual((reg["partial"], self.states(reg)), ([], dict.fromkeys(self.BOTH, "complete")))

    def test_the_video_lanes_patterns_not_read_leave_it_alone_unknown(self):
        # switch-model.sh cut after its pins and before the patterns the video lane serves: the video target unknown,
        # the others read as ever (fifth review: that read was outside the reads made again)
        text = (REPO / "switch-model.sh").read_text()
        cockpit_time(self.cp, self, sleep=lambda s: None)
        self.half_repo(text[:text.index('if [ "$VID_VARIANT" = "fl2va" ]; then VID_ALLOW=')], name="switch-model.sh")
        reg = self.cp.registry_snapshot(max_age=0.0)
        state = {t["target"]: t["state"] for t in reg["targets"]}
        self.assertIn("the video lane's patterns not found in switch-model.sh", reg["partial"])
        self.assertEqual((state["video"], state["stock"]), ("unknown", "absent"))

    def test_the_upstream_view_keeps_a_scan_that_missed_a_pin_a_minute(self):
        # kept an hour, it would lack that pin's rows for the hour; rebuilt on every call, it would ask the Hub each time
        real, real_get = self.cp.registry_snapshot, self.cp._get_json
        self.addCleanup(setattr, self.cp, "registry_snapshot", real)
        self.addCleanup(setattr, self.cp, "_get_json", real_get)
        asked = []
        lost = {"pins_lost": True}
        self.cp.registry_snapshot = lambda max_age=300.0: asked.append(1) or {"pins": {}, "partial": ["x"], **lost}
        self.cp._get_json = lambda url, timeout=5.0: (_ for _ in ()).throw(OSError(f"offline test: {url}"))
        self.cp.UPSTREAM_CACHE.update(ts=0.0, data=None)
        self.assertTrue(self.cp.upstream_snapshot()["pins_lost"])
        self.cp.upstream_snapshot()
        self.assertEqual(len(asked), 1, "within its minute, the same view")
        self.cp.UPSTREAM_CACHE["ts"] -= self.cp.UPSTREAM_PARTIAL_TTL + 1
        self.cp.upstream_snapshot()
        self.assertEqual(len(asked), 2)
        # a miss that loses no pin (a target's parts, the video lane's patterns) leaves every row: kept the hour
        lost["pins_lost"] = False
        self.cp.UPSTREAM_CACHE.update(ts=0.0, data=None)
        self.cp.upstream_snapshot()
        self.cp.UPSTREAM_CACHE["ts"] -= self.cp.UPSTREAM_PARTIAL_TTL + 1
        self.cp.upstream_snapshot()
        self.assertEqual(len(asked), 3, "the hour holds")

    def test_the_action_runs_the_switch_with_fetch_and_nothing_else(self):
        code, body = self.cp.start_action("fetch", {"target": "kolibri-nvfp4"})
        self.assertEqual(code, 202, body)
        self.assertEqual(body["argv"], ["bash", str(REPO / "switch-model.sh"), "kolibri-nvfp4", "--fetch"])
        for _ in range(100):
            if self.cp.JOBS[body["job"]].status != "running":
                break
            __import__("time").sleep(0.05)
        self.assertEqual(self.cp.start_action("fetch", {"target": "nope"})[0], 400)
        self.assertIs(self.cp.ACTIONS["fetch"]["params"]["target"], self.cp.ACTIONS["switch"]["params"]["target"])


if __name__ == "__main__":
    unittest.main()
