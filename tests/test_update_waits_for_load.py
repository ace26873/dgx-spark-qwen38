#!/usr/bin/env python3
"""install.sh on an installed box (decisions 1404 and 1408 to 1410 of Hasan, 2026-11-16 and 17): every unit it writes
goes through unit_put with PENDING_PUT=1, and the text lane, when it can serve and its new pins are not all here (the
image, the target whole at its revision, the 27B's draft), gets its units only: none of its downloads, its update
waiting for its Load. --with-<lane>, --all-weights, a custom model kept, --no-service and a new box download as before.
The block of install.sh itself, run with the box's facts stubbed."""
import pathlib
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
TEXT = (REPO / "install.sh").read_text()
BLOCK = TEXT[TEXT.index("# An update downloads nothing large (decisions 1404 and 1408 to 1410 of Hasan, 2026-11-16 and 17): weights"):
             TEXT.index("# Measured where the weights actually land")]
IMAGE = "lmsysorg/sglang@sha256:" + "a" * 64
REV, DREV = "1" * 40, "d" * 40


class UpdateWaits(unittest.TestCase):
    def run_block(self, unit=True, prepared=True, image=True, ckpt=True, draft=True, all_weights=0, named=0,
                  no_service=0, verbatim=0, units_only=0, lane="27b"):
        d = pathlib.Path(tempfile.mkdtemp(prefix="update-waits-"))
        hf = d / "hf"
        if draft:
            snap = hf / "hub" / "models--maurienne-ai--Draft" / "snapshots" / DREV
            snap.mkdir(parents=True)
            (snap / "config.json").write_text("{}")
            (snap / "model.safetensors").write_bytes(b"w")
        etc = d / "etc"
        etc.mkdir()
        if unit:
            (etc / "qwen38-sglang.service").write_text("x")
        script = f"""
set -euo pipefail
CONFIG_DIR={d}; . "{REPO}/lanes.sh"
LANE={lane}; UNIT_NAME=qwen38-sglang.service; ALL_WEIGHTS={all_weights}; UNITS_ONLY={units_only}; NO_SERVICE={no_service}
TEXT_NAMED={named}; KEEP_MODEL_VERBATIM={verbatim}; IMAGE={IMAGE}; FLASH_IMAGE={IMAGE}; HF_CACHE={hf}
MODEL_REPO=RadixArk/Qwen3.8-27B-NVFP4; MODEL_REV={REV}; DRAFT2_REPO=maurienne-ai/Draft; DRAFT2_REV={DREV}
text_lane_prepared(){{ return {0 if prepared else 1}; }}
docker(){{ return {0 if image else 1}; }}
ckpt_cached(){{ return {0 if ckpt else 1}; }}
""" + BLOCK.replace("/etc/systemd/system/", f"{etc}/") + """
echo "STATE PENDING_PUT=$PENDING_PUT WAITS_LOAD=$WAITS_LOAD UNITS_ONLY=$UNITS_ONLY"
"""
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def test_an_update_that_lacks_the_new_pins_waits_for_the_load(self):
        out = self.run_block(image=False, ckpt=False, draft=False)
        self.assertIn("STATE PENDING_PUT=1 WAITS_LOAD=1 UNITS_ONLY=1", out)
        self.assertIn(f"image {IMAGE}; checkpoint RadixArk/Qwen3.8-27B-NVFP4@{REV}; draft maurienne-ai/Draft@{DREV}", out)
        self.assertIn("keeps serving what it served until its next Load", out)

    def test_an_update_whose_pins_are_all_here_runs_whole(self):
        self.assertIn("STATE PENDING_PUT=1 WAITS_LOAD=0 UNITS_ONLY=0", self.run_block())

    def test_asked_downloads_and_a_new_box_download_as_before(self):
        for kw in ({"named": 1}, {"no_service": 1}, {"verbatim": 1}):
            with self.subTest(**kw):
                self.assertIn("WAITS_LOAD=0 UNITS_ONLY=0", self.run_block(image=False, **kw))
        self.assertIn("STATE PENDING_PUT=0 WAITS_LOAD=0 UNITS_ONLY=0", self.run_block(image=False, all_weights=1))
        self.assertIn("STATE PENDING_PUT=0 WAITS_LOAD=0 UNITS_ONLY=0", self.run_block(image=False, unit=False))

    def test_a_lane_that_cannot_serve_as_it_is_is_not_this_case(self):
        # emptied or never loaded: the paths before say what it gets (left out, not prepared), not this one
        self.assertIn("STATE PENDING_PUT=1 WAITS_LOAD=0 UNITS_ONLY=0", self.run_block(prepared=False, image=False))

    def test_a_run_already_on_its_units_only_stays_as_it_was(self):
        self.assertIn("STATE PENDING_PUT=1 WAITS_LOAD=0 UNITS_ONLY=1", self.run_block(image=False, units_only=1))

    def test_the_flash_lane_has_no_draft_to_lack(self):
        self.assertIn("STATE PENDING_PUT=1 WAITS_LOAD=0 UNITS_ONLY=0", self.run_block(lane="flash", draft=False))



class ARuntimeThatMovesWaitsToo(unittest.TestCase):
    """The image, MiniMax-H3-Image and video lanes' venv is built again when its pinned SGLang wheel or source moved:
    gigabytes of wheels, which an update leaves to the lane's Load (side_runtime_waits)."""
    FN = TEXT[TEXT.index("side_runtime_waits(){"):TEXT.index("\n}\n", TEXT.index("side_runtime_waits(){")) + 3]

    def run_fn(self, lane="image", note="0.5.21", source=None, pin_source="e" * 40, venv=True, waiting=False,
               waiting_needs="image x\n"):
        d = pathlib.Path(tempfile.mkdtemp(prefix="runtime-waits-"))
        repo, cfg, lanes, etc = d / "repo", d / "config", d / "lanes", d / "etc"
        for x in (repo, cfg, lanes, etc):
            x.mkdir()
        (repo / f"install-{lane}.sh").write_text('WHEEL="${SGLANG_DIFFUSION_WHEEL:-0.5.22}"\n'
                                                 f'PIN="${{SGLANG_DIFFUSION_PIN:-{pin_source}}}"\n')
        (etc / f"qwen38-{lane}.service").write_text("[Service]\nExecStart=old\n")
        ldir = lanes / lane
        if venv:
            (ldir / "venv" / "bin").mkdir(parents=True)
            (ldir / "venv" / "bin" / "python").write_text("#!/bin/sh\n")
            (ldir / "venv" / "bin" / "python").chmod(0o755)
        else:
            ldir.mkdir()
        (ldir / "sglang-wheel").write_text(note + "\n")
        if source:
            (ldir / "sglang-source").write_text(source + "\n")
        if waiting:
            (cfg / "pending").mkdir()
            (cfg / "pending" / f"qwen38-{lane}.service").write_text("newer")
            (cfg / "pending" / f"qwen38-{lane}.service.needs").write_text(waiting_needs)
        script = (f'set -euo pipefail\nREPO_DIR="{repo}"; CONFIG_DIR="{cfg}"\nside_lane_dir(){{ echo "{lanes}/$1"; }}\n'
                  + self.FN.replace("/etc/systemd/system/", f"{etc}/") + f'\nside_runtime_waits {lane}\n')
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        f = cfg / "pending" / f"qwen38-{lane}.service"
        return (f.read_text(), (cfg / "pending" / f"qwen38-{lane}.service.needs").read_text()) if f.exists() else None

    def test_a_wheel_that_moved_waits_with_the_unit_in_place(self):
        unit, needs = self.run_fn()
        self.assertEqual(unit, "[Service]\nExecStart=old\n")
        self.assertEqual(needs, "runtime sglang[diffusion] 0.5.22 (it runs 0.5.21)\n")

    def test_a_source_that_moved_waits(self):
        unit, needs = self.run_fn(lane="image-h3", note="0.5.22", source="d" * 40)
        self.assertEqual(needs, f"runtime SGLang source {'e' * 12}\n")

    def test_nothing_moved_nothing_waits(self):
        self.assertIsNone(self.run_fn(note="0.5.22", source="e" * 40))

    def test_a_venv_not_built_and_a_text_lane_are_not_this_case(self):
        self.assertIsNone(self.run_fn(venv=False))
        self.assertIsNone(self.run_fn(lane="kolibri"))

    def test_a_wheel_and_a_source_that_moved_are_one_line_each(self):
        unit, needs = self.run_fn(lane="image-h3", source="d" * 40)
        self.assertEqual(needs, f"runtime sglang[diffusion] 0.5.22 (it runs 0.5.21)\nruntime SGLang source {'e' * 12}\n")

    def test_a_unit_already_waiting_keeps_its_unit_and_learns_its_runtime(self):
        # an earlier run's wait: its runtime lines were left out (Vibe's blind review of v1.52.0, Mistral Large 4)
        self.assertEqual(self.run_fn(waiting=True),
                         ("newer", "image x\nruntime sglang[diffusion] 0.5.22 (it runs 0.5.21)\n"))

    def test_a_runtime_line_of_an_earlier_run_is_made_current_not_repeated(self):
        self.assertEqual(self.run_fn(waiting=True, waiting_needs="image x\nruntime sglang[diffusion] 0.5.21 (it runs 0.5.20)\n"),
                         ("newer", "image x\nruntime sglang[diffusion] 0.5.22 (it runs 0.5.21)\n"))

    def test_a_unit_already_waiting_with_nothing_moved_is_left_as_it_is(self):
        self.assertEqual(self.run_fn(waiting=True, note="0.5.22"), ("newer", "image x\n"))


class SideInstall(unittest.TestCase):
    """install.sh's side_install, which every lane beside the text lane goes through on a run, the box's boot lane
    included (code review of v1.52.0: the branch for a box booting a side lane called the installers directly)."""
    FNS = (TEXT[TEXT.index("SIDE_WAITED=\" \""):TEXT.index("side_mode(){")])

    def test_no_installer_of_a_side_lane_is_called_but_through_it(self):
        import re
        direct = re.findall(r'"\$REPO_DIR/install-[a-z0-9-]+\.sh" "\$(?:\(side_mode|[A-Z0-9_]+_MODE)', TEXT)
        self.assertEqual(direct, [])

    def run_side(self, lane="kolibri", mode="--no-smoke", named=0, pending_put=1, waits=False, lacks=False,
                 runtime_moves=False):
        d = pathlib.Path(tempfile.mkdtemp(prefix="side-install-"))
        repo, cfg, etc = d / "repo", d / "config", d / "etc"
        for x in (repo, cfg, etc):
            x.mkdir()
        log = d / "calls"
        log.write_text("")
        wait = (f'case "$*" in *--units-only*) mkdir -p "{cfg}/pending"; echo new > "{cfg}/pending/qwen38-{lane}.service"; '
                f'echo "image x" > "{cfg}/pending/qwen38-{lane}.service.needs" ;; esac\n') if waits else ""
        inst = repo / f"install-{lane}.sh"
        inst.write_text(f'#!/bin/bash\necho "install-{lane}.sh $*" >> "{log}"\n' + wait + "exit 0\n")
        inst.chmod(0o755)
        (etc / f"qwen38-{lane}.service").write_text("[Service]\nExecStart=old\n")
        script = (f'set -euo pipefail\nREPO_DIR="{repo}"; CONFIG_DIR="{cfg}"; PENDING_PUT={pending_put}; ALL_WEIGHTS=0\n'
                  f'lane_title(){{ echo "$1"; }}\nunit_pending_says(){{ echo "image x"; }}\n'
                  f'unit_needs(){{ {"echo checkpoint gone" if lacks else "true"}; }}\n'
                  + self.FNS.replace("/etc/systemd/system/", f"{etc}/")
                  # the runtime's own verdict is tested above: here it waits or not, as asked
                  + ('side_runtime_waits(){ mkdir -p "$CONFIG_DIR/pending"; cp "' + str(etc) + '/qwen38-$1.service" '
                     '"$CONFIG_DIR/pending/qwen38-$1.service"; echo "runtime x" > "$CONFIG_DIR/pending/qwen38-$1.service.needs"; }\n'
                     if runtime_moves else 'side_runtime_waits(){ :; }\n')
                  + f'side_install {lane} {mode} {named}\necho "WAITED=$SIDE_WAITED"\n')
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        return log.read_text().splitlines(), r.stdout

    def test_its_new_unit_waits_and_nothing_more_runs(self):
        calls, out = self.run_side(waits=True)
        self.assertEqual(calls, ["install-kolibri.sh --units-only"])
        self.assertIn("WAITED= kolibri ", out)
        self.assertIn("the kolibri lane's update waits for its next Load", out)

    def test_a_runtime_that_moves_waits_with_its_unit_untouched(self):
        calls, out = self.run_side(lane="image", runtime_moves=True)
        self.assertEqual(calls, [], "not even its units: the new unit would run on the old venv at the next boot")
        self.assertIn("WAITED= image ", out)

    def test_a_lane_that_lacks_what_its_unit_reads_gets_its_units_only(self):
        calls, out = self.run_side(lacks=True)
        self.assertEqual(calls, ["install-kolibri.sh --units-only"])
        self.assertIn("its units only, and its next Load prepares it", out)

    def test_nothing_waits_nothing_lacks_the_rest_follows(self):
        calls, _ = self.run_side()
        self.assertEqual(calls, ["install-kolibri.sh --units-only", "install-kolibri.sh --no-smoke"])

    def test_named_or_on_demand_or_a_new_box_as_before(self):
        self.assertEqual(self.run_side(named=1, waits=True)[0], ["install-kolibri.sh --no-smoke"])
        self.assertEqual(self.run_side(mode="--units-only")[0], ["install-kolibri.sh --units-only"])
        self.assertEqual(self.run_side(pending_put=0, waits=True)[0], ["install-kolibri.sh --no-smoke"])

if __name__ == "__main__":
    unittest.main()
