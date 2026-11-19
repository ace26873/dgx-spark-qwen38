#!/usr/bin/env python3
"""install.sh on an installed box (decisions 1404 and 1408 to 1410 of Hasan, 2026-11-16 and 17): every unit it writes
goes through unit_put with PENDING_PUT=1, and the text lane, when it can serve and its new pins are not all here (the
image, the target whole at its revision, the 27B's draft), gets its units only: none of its downloads, its update
waiting for its Load. --with-<lane>, --all-weights, a custom model kept, --no-service and a new box download as before.
The block of install.sh itself, run with the box's facts stubbed."""
import os
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
                  no_service=0, verbatim=0, units_only=0, lane="27b", side_unit=None):
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
        if side_unit:
            (etc / side_unit).write_text("x")
        # docker as a stub file first in PATH, not a shell function: lanes.sh's docker_probe bounds its
        # calls with the timeout binary, which looks docker up in PATH and would miss a function
        binp = d / "bin"
        binp.mkdir()
        (binp / "docker").write_text(f'#!/bin/bash\n[ "$1" = "info" ] && exit 0\nexit {0 if image else 1}\n')
        (binp / "docker").chmod(0o755)
        script = f"""
set -euo pipefail
CONFIG_DIR={d}; . "{REPO}/lanes.sh"
LANE={lane}; UNIT_NAME=qwen38-sglang.service; ALL_WEIGHTS={all_weights}; UNITS_ONLY={units_only}; NO_SERVICE={no_service}
TEXT_NAMED={named}; KEEP_MODEL_VERBATIM={verbatim}; IMAGE={IMAGE}; FLASH_IMAGE={IMAGE}; SERVE_IMAGE={IMAGE}; FLASH_SERVE_IMAGE={IMAGE}; HF_CACHE={hf}
MODEL_REPO=RadixArk/Qwen3.8-27B-NVFP4; MODEL_REV={REV}; DRAFT2_REPO=maurienne-ai/Draft; DRAFT2_REV={DREV}
text_lane_prepared(){{ return {0 if prepared else 1}; }}
ckpt_cached(){{ return {0 if ckpt else 1}; }}
""" + BLOCK.replace("/etc/systemd/system/", f"{etc}/") + """
echo "STATE PENDING_PUT=$PENDING_PUT WAITS_LOAD=$WAITS_LOAD UNITS_ONLY=$UNITS_ONLY"
"""
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30,
                           env={**os.environ, "PATH": f"{binp}:{os.environ['PATH']}"})
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

    def test_a_box_with_only_a_lane_beside_arms_the_wait_too(self):
        # side lanes installed alone (their own installer, no text lane unit on the box): an update of one
        # of them waits for its Load there too, instead of re-downloading what the Load would have fetched
        # (review of v1.52.0, 2026-11-19, GLM 5.3: only the run's text lane armed PENDING_PUT, so side_install
        # called the installer whole, 34 GB for the image lane's checkpoint)
        # every lane beside, the ones install.sh named by their *_UNIT_PATH (whole paths, a doubled path) too
        for side in ("qwen38-image.service", "qwen38-video.service", "qwen38-kolibri.service", "qwen38-nemotron.service",
                     "qwen38-nemotron-super.service", "qwen38-flash-local.service", "qwen38-ornith-122b.service",
                     "qwen38-flash-unsloth.service", "qwen38-image-h3.service", "qwen38-transcribe.service",
                     "qwen38-lfm25.service", "qwen38-lfm3.service"):
            with self.subTest(side=side):
                self.assertIn("STATE PENDING_PUT=1 WAITS_LOAD=0 UNITS_ONLY=0",
                              self.run_block(unit=False, prepared=False, image=False, side_unit=side))

    def test_a_lane_that_cannot_serve_as_it_is_is_not_this_case(self):
        # emptied or never loaded: the paths before say what it gets (left out, not prepared), not this one
        self.assertIn("STATE PENDING_PUT=1 WAITS_LOAD=0 UNITS_ONLY=0", self.run_block(prepared=False, image=False))

    def test_a_run_already_on_its_units_only_stays_as_it_was(self):
        self.assertIn("STATE PENDING_PUT=1 WAITS_LOAD=0 UNITS_ONLY=1", self.run_block(image=False, units_only=1))

    def test_the_flash_lane_has_no_draft_to_lack(self):
        self.assertIn("STATE PENDING_PUT=1 WAITS_LOAD=0 UNITS_ONLY=0", self.run_block(lane="flash", draft=False))



class ARuntimeThatMovesWaitsToo(unittest.TestCase):
    """The image, MiniMax-H3-Image and video lanes' venv is built again when its pinned SGLang wheel or source moved:
    gigabytes of wheels, which an update leaves to the lane's Load. side_runtime_waits only says what the runtime
    lacks, one line each (side_install renders the lane's NEW unit and lets it wait, PENDING_FORCE)."""
    FN = TEXT[TEXT.index("side_runtime_waits(){"):TEXT.index("\n}\n", TEXT.index("side_runtime_waits(){")) + 3]

    def run_fn(self, lane="image", note="0.5.21", source=None, pin_source="e" * 40, venv=True, env=None):
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
        script = (f'set -euo pipefail\nREPO_DIR="{repo}"; CONFIG_DIR="{cfg}"\nside_lane_dir(){{ echo "{lanes}/$1"; }}\n'
                  + self.FN.replace("/etc/systemd/system/", f"{etc}/") + f'\nside_runtime_waits {lane}\n')
        if env:
            for k, v in env.items():
                os.environ[k] = v
                self.addCleanup(os.environ.pop, k, None)
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout, (cfg / "pending").exists()

    def test_a_wheel_that_moved_is_said_and_nothing_is_written(self):
        out, pending = self.run_fn()
        self.assertEqual(out, "runtime sglang[diffusion] 0.5.22 (it runs 0.5.21)\n")
        self.assertFalse(pending, "the function only says: side_install renders the new unit and waits it")

    def test_a_source_that_moved_is_said(self):
        self.assertEqual(self.run_fn(lane="image-h3", note="0.5.22", source="d" * 40)[0],
                         f"runtime SGLang source {'e' * 12}\n")

    def test_the_image_lanes_source_note_is_read_too(self):
        # install-image.sh writes the note like the video and image-h3 lanes do, so a moved source pin
        # waits for the image lane's Load as for the other two (seventh pass of the v1.52.0 reviews,
        # 2026-11-19: it wrote no note, so the move was never said)
        self.assertEqual(self.run_fn(lane="image", note="0.5.22", source="d" * 40)[0],
                         f"runtime SGLang source {'e' * 12}\n")

    def test_an_override_in_this_runs_environment_is_the_value_compared(self):
        # the install was made with SGLANG_DIFFUSION_WHEEL=0.5.23, the note holds 0.5.23: a run with the
        # same environment must not say the runtime moved (the default alone would have, at every update;
        # seventh pass of the v1.52.0 reviews, 2026-11-19)
        out, _ = self.run_fn(note="0.5.23", env={"SGLANG_DIFFUSION_WHEEL": "0.5.23"})
        self.assertEqual(out, "")
        out, _ = self.run_fn(note="0.5.23", env={"SGLANG_DIFFUSION_PIN": "d" * 40}, source="d" * 40)
        self.assertEqual(out, "")

    def test_nothing_moved_says_nothing(self):
        self.assertEqual(self.run_fn(note="0.5.22", source="e" * 40), ("", False))

    def test_a_venv_not_built_and_a_text_lane_are_not_this_case(self):
        self.assertEqual(self.run_fn(venv=False), ("", False))
        self.assertEqual(self.run_fn(lane="kolibri"), ("", False))

    def test_a_wheel_and_a_source_that_moved_are_one_line_each(self):
        self.assertEqual(self.run_fn(lane="image-h3", source="d" * 40)[0],
                         f"runtime sglang[diffusion] 0.5.22 (it runs 0.5.21)\nruntime SGLang source {'e' * 12}\n")


class TheImageLaneNotesItsSGLangSource(unittest.TestCase):
    """install-image.sh runs its venv from a cloned SGLang source checked out at its pin; it now notes the
    commit in sglang-source, as the video and image-h3 lanes do, so install.sh's side_runtime_waits can say
    a moved pin waits for the lane's Load (seventh pass of the v1.52.0 reviews, 2026-11-19: no note was
    written, the move was never said for this lane)."""

    def test_the_note_is_written_at_the_checkout_and_removed_with_the_runtime(self):
        text = (REPO / "install-image.sh").read_text()
        self.assertIn('SOURCE_NOTE="$LANE_DIR/sglang-source"', text)
        self.assertIn("printf '%s\\n' \"$PIN\" > \"$SOURCE_NOTE\"", text)
        self.assertIn('rm -rf "$VENV" "$SRC" "$WHEEL_NOTE" "$SOURCE_NOTE"', text)
        lane_files = text[text.index("LANE_FILES="):text.index("LANE_FILES=") + 220]
        self.assertIn('"$SOURCE_NOTE"', lane_files)


class SideInstall(unittest.TestCase):
    """install.sh's side_install, which every lane beside the text lane goes through on a run, the box's boot lane
    included (code review of v1.52.0: the branch for a box booting a side lane called the installers directly)."""
    FNS = (TEXT[TEXT.index("SIDE_WAITED=\" \""):TEXT.index("side_mode(){")])

    def test_no_installer_of_a_side_lane_is_called_but_through_it(self):
        import re
        direct = re.findall(r'"\$REPO_DIR/install-[a-z0-9-]+\.sh" "\$(?:\(side_mode|[A-Z0-9_]+_MODE)', TEXT)
        self.assertEqual(direct, [])

    def run_side(self, lane="kolibri", mode="--no-smoke", named=0, pending_put=1, waits=False, lacks=False,
                 unknown=False, runtime_moves=False, already_waiting=False):
        d = pathlib.Path(tempfile.mkdtemp(prefix="side-install-"))
        repo, cfg, etc = d / "repo", d / "config", d / "etc"
        for x in (repo, cfg, etc):
            x.mkdir()
        log = d / "calls"
        log.write_text("")
        if already_waiting:
            (cfg / "pending").mkdir()
            (cfg / "pending" / f"qwen38-{lane}.service").write_text("waiting unit\n")
            (cfg / "pending" / f"qwen38-{lane}.service.needs").write_text("checkpoint gone\n")
        # the installer's unit_put, as lanes.sh's: --units-only renders the NEW unit; it waits (pending) when
        # PENDING_FORCE=1 or when it lacks what it reads, and goes in place otherwise, clearing what waited
        inst = repo / f"install-{lane}.sh"
        inst.write_text(f'#!/bin/bash\necho "install-{lane}.sh $* force=${{PENDING_FORCE:-0}}" >> "{log}"\n'
                        f'case " $* " in *" --units-only "*) '
                        f'if [ "${{PENDING_FORCE:-0}}" = 1 ] || {"true" if waits else "false"}; then '
                        f'mkdir -p "{cfg}/pending"; echo "new unit" > "{cfg}/pending/qwen38-{lane}.service"; '
                        f'echo "checkpoint gone" > "{cfg}/pending/qwen38-{lane}.service.needs"; '
                        f'else echo "new unit" > "{etc}/qwen38-{lane}.service"; '
                        f'rm -f "{cfg}/pending/qwen38-{lane}.service" "{cfg}/pending/qwen38-{lane}.service.needs"; fi ;; esac\nexit 0\n')
        inst.chmod(0o755)
        (etc / f"qwen38-{lane}.service").write_text("[Service]\nExecStart=old\n")
        needs_stub = 'echo "unknown cache /mnt/hf"' if unknown else ('echo checkpoint gone' if lacks else 'true')
        script = (f'set -euo pipefail\nREPO_DIR="{repo}"; CONFIG_DIR="{cfg}"; PENDING_PUT={pending_put}; ALL_WEIGHTS=0\n'
                  f'lane_title(){{ echo "$1"; }}\nunit_pending_says(){{ cat "$CONFIG_DIR/pending/qwen38-$1.service.needs" 2>/dev/null || echo "image x"; }}\n'
                  f'unit_needs(){{ {needs_stub}; }}\n'
                  + self.FNS.replace("/etc/systemd/system/", f"{etc}/")
                  # the runtime's own verdict is tested above: here it moves or not, as asked
                  + ('side_runtime_waits(){ echo "runtime x"; }\n' if runtime_moves else 'side_runtime_waits(){ :; }\n')
                  + f'side_install {lane} {mode} {named}\necho "WAITED=$SIDE_WAITED"\n')
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        pend = cfg / "pending" / f"qwen38-{lane}.service"
        needs = (cfg / "pending" / f"qwen38-{lane}.service.needs").read_text() if (cfg / "pending" / f"qwen38-{lane}.service.needs").exists() else ""
        return log.read_text().splitlines(), r.stdout, (pend.read_text() if pend.exists() else None), needs

    def test_its_new_unit_waits_and_nothing_more_runs(self):
        calls, out, pend, needs = self.run_side(waits=True)
        self.assertEqual(calls, ["install-kolibri.sh --units-only force=0"])
        self.assertEqual(pend, "new unit\n")
        self.assertIn("WAITED= kolibri ", out)
        self.assertIn("the kolibri lane's update waits for its next Load", out)

    def test_a_runtime_that_moves_waits_with_the_new_unit_rendered(self):
        # the unit kept waiting is the NEW one the installer rendered, never the one in place: a Load that
        # clears the wait then applies the moved pin too (review of v1.52.0, 2026-11-19, GLM 5.3: the old unit
        # was the one kept, and the Load applied the old pin, losing the new one in silence)
        calls, out, pend, needs = self.run_side(lane="image", runtime_moves=True)
        self.assertEqual(calls, ["install-image.sh --units-only force=1"],
                         "its new unit is rendered and waits, never run on the old venv at the next boot")
        self.assertEqual(pend, "new unit\n")
        self.assertEqual(needs, "checkpoint gone\nruntime x\n")
        self.assertIn("WAITED= image ", out)

    def test_a_runtime_that_moves_with_a_wait_already_there_rerenders_it(self):
        # an earlier run's wait is re-rendered (self-healing), with its runtime lines made current
        calls, out, pend, needs = self.run_side(lane="image", runtime_moves=True, already_waiting=True)
        self.assertEqual(calls, ["install-image.sh --units-only force=1"])
        self.assertEqual(pend, "new unit\n")
        self.assertEqual(needs, "checkpoint gone\nruntime x\n")

    def test_a_runtime_that_moves_with_the_unit_in_place_lacking_still_waits_and_notes_it(self):
        # the unit in place already lacks what it reads (its checkpoint deleted from Library) AND the runtime
        # moved: PENDING_FORCE waits the new unit anyway (lanes.sh's unit_put), so the wait carries the runtime
        # lines, the NOTE is the wait's, and the lane's Load prepares (the venv again) then poses it; the moved
        # runtime is never lost (fourth review of v1.52.0, 2026-11-19, GLM 5.3: the new unit went in place,
        # side_install found no wait, said "its units only", and the Load started the new unit on the old venv)
        for lane in ("image", "video", "image-h3"):
            with self.subTest(lane=lane):
                calls, out, pend, needs = self.run_side(lane=lane, runtime_moves=True, lacks=True)
                self.assertEqual(calls, [f"install-{lane}.sh --units-only force=1"])
                self.assertEqual(pend, "new unit\n")
                self.assertEqual(needs, "checkpoint gone\nruntime x\n")
                self.assertIn(f"the {lane} lane's update waits for its next Load", out)
                self.assertNotIn("its units only", out)

    def test_a_wait_already_there_with_nothing_moved_is_rerendered_and_replaced(self):
        # every update renders its lane's NEW unit and lanes.sh's unit_put decides alone between in place and
        # waiting: a wait of an earlier run is replaced by this one, never reused (third review of v1.52.0,
        # 2026-11-19, GLM 5.3: the render was skipped when a wait was there and nothing had moved, and the
        # Load applied the older version's unit, losing this one's in silence)
        calls, out, pend, needs = self.run_side(already_waiting=True, waits=True)
        self.assertEqual(calls, ["install-kolibri.sh --units-only force=0"], "the new unit is rendered again")
        self.assertEqual(pend, "new unit\n", "the earlier run's wait is replaced by this run's unit")
        self.assertIn("WAITED= kolibri ", out)

    def test_a_wait_already_there_with_nothing_moved_and_nothing_to_wait_for_is_cleared(self):
        # the re-render goes in place (nothing waits) and the earlier wait is cleared with it, the rest of
        # the update following as before
        calls, out, pend, needs = self.run_side(already_waiting=True)
        self.assertEqual(calls, ["install-kolibri.sh --units-only force=0", "install-kolibri.sh --no-smoke force=0"])
        self.assertIsNone(pend, "unit_put put the new unit in place and cleared the wait")
        self.assertNotIn("WAITED= kolibri", out)

    def test_a_lane_that_lacks_what_its_unit_reads_gets_its_units_only(self):
        calls, out, pend, needs = self.run_side(lacks=True)
        self.assertEqual(calls, ["install-kolibri.sh --units-only force=0"])
        self.assertIn("its units only, and its next Load prepares it", out)

    def test_an_unknown_line_is_never_a_lack(self):
        # docker muet ou cache illisible pendant la mise à jour : la lane sert toujours ce qu'elle sert, et le
        # reste de sa mise à jour suit, comme lanes.sh's unit_put_waits lit la même ligne (sixth review of v1.52.0,
        # 2026-11-19, GLM 5.3: la ligne « unknown » était lue comme un manque, la branche sautait le reste de la
        # mise à jour avec un message faux, et aucun Load ne pouvait préparer un « unknown »)
        for lane in ("kolibri", "image", "video", "image-h3", "nemotron", "nemotron-super",
                     "flash-local", "ornith-122b", "flash-unsloth"):
            with self.subTest(lane=lane):
                calls, out, pend, _ = self.run_side(lane=lane, unknown=True)
                self.assertEqual(calls, [f"install-{lane}.sh --units-only force=0",
                                         f"install-{lane}.sh --no-smoke force=0"], out)
                self.assertNotIn("its units only", out)
                self.assertNotIn(f"WAITED= {lane}", out)

    def test_nothing_waits_nothing_lacks_the_rest_follows(self):
        calls, _, pend, _ = self.run_side()
        self.assertEqual(calls, ["install-kolibri.sh --units-only force=0", "install-kolibri.sh --no-smoke force=0"])
        self.assertIsNone(pend)

    def test_named_or_on_demand_or_a_new_box_as_before(self):
        self.assertEqual(self.run_side(named=1, waits=True)[0], ["install-kolibri.sh --no-smoke force=0"])
        self.assertEqual(self.run_side(mode="--units-only")[0], ["install-kolibri.sh --units-only force=0"])
        self.assertEqual(self.run_side(pending_put=0, waits=True)[0], ["install-kolibri.sh --no-smoke force=0"])

class TheVLLMLanesStageTheirTemplateAfterTheFetch(unittest.TestCase):
    """A vLLM lane's switch fetches its checkpoint, then runs its installer's --stage, which writes the chat
    template the unit bind-mounts and renders the unit: one Load applies all of it, never a first Load that
    leaves the wait in place after everything was downloaded (third review of v1.52.0, 2026-11-19, GLM 5.3,
    "à vérifier")."""

    def test_the_stage_runs_after_the_fetch_in_each_lane(self):
        text = (REPO / "switch-model.sh").read_text()
        for lane, fetch_mark, stage_mark in (
                ("flash-local", "ckpt-fetch.py \"$FLASH_LOCAL_REPO\"", "install-flash-local.sh\" --stage"),
                ("ornith-122b", "ckpt-fetch.py \"$ORNITH_122B_REPO\"", "install-ornith-122b.sh\" --stage"),
                ("flash-unsloth", "install-flash-unsloth.sh\" --fetch", "install-flash-unsloth.sh\" --stage")):
            with self.subTest(lane=lane):
                start = text.index(f'if [ "$CHOICE" = "{lane}"')
                end = text.index("\nfi\n", start)
                block = text[start:end]
                self.assertLess(block.index(fetch_mark), block.index(stage_mark),
                                "the fetch precedes the stage that writes the chat template")


class TheWaitsLoadProbeCoversWhatTheRealUnitsRead(unittest.TestCase):
    """The WAITS_LOAD probe of install.sh is a synthetic unit: image, --model-path/--revision, the 27B's draft.
    Every hub-checkpoint flag the real text units and launcher name is one it enumerates, so a need the real
    unit reads cannot be missed by the probe and send the update downloading (third review of v1.52.0,
    2026-11-19, GLM 5.3, "à vérifier")."""

    def test_every_checkpoint_flag_of_the_text_templates_is_in_the_probe(self):
        import re
        text = (REPO / "install.sh").read_text()
        probe = text[text.index('_wunit="$(mktemp)"'):text.index('WAITS_WHAT=', text.index('_wunit="$(mktemp)"'))]
        probe_flags = set(re.findall(r"--[a-z0-9-]+", probe))
        for tpl in ("qwen38-sglang-1m.service.template", "qwen38-sglang-64.service.template",
                    "qwen38-flash-launch.sh.template", "qwen38-flash.service.template"):
            t = (REPO / tpl).read_text()
            for flag in set(re.findall(r"--(?:model-path|revision|speculative-draft-model-path|"
                                       r"speculative-draft-model-revision|speculative-config\.model|"
                                       r"speculative-config\.revision)\b", t)):
                with self.subTest(tpl=tpl, flag=flag):
                    self.assertIn(flag, probe_flags)


class TheWaitsLoadProbeReadsTheImageTheRenderedUnitServes(unittest.TestCase):
    """The WAITS_LOAD probe tests the image the unit this run renders will serve, SERVE_IMAGE_FINAL
    (SERVE_IMAGE= / FLASH_SERVE_IMAGE=, a documented option, substituted into __IMAGE__ below),
    not the pinned one: a box serving an override while the pin is absent does not lack it, and its
    update must not go units-only with a message saying its Load will download an image the unit
    never reads (sixth review of v1.52.0, 2026-11-19, GLM 5.3). The probe block runs here with a
    docker that knows which images are on the box."""

    BLOCK = TEXT[TEXT.index("  _wimg="):TEXT.index("\n  fi\n", TEXT.index("WAITS_LOAD=1", TEXT.index("  _wimg="))) + len("\n  fi\n")]
    PIN = "lmsysorg/sglang@sha256:" + "a" * 64
    OTHER_PIN = "lmsysorg/sglang@sha256:" + "b" * 64

    def run_probe(self, lane, serve_image):
        d = pathlib.Path(tempfile.mkdtemp(prefix="waits-img-"))
        hf = d / "hf"
        hf.mkdir()
        binp = d / "bin"
        binp.mkdir()
        # the pinned image is absent, the serving override is there
        (binp / "docker").write_text(
            '#!/bin/bash\ncase "$1 $2" in\n'
            '  "image inspect") case "$3" in ' + self.PIN + '|' + self.OTHER_PIN + ') exit 1 ;; *) exit 0 ;; esac ;;\n'
            '  "info "*) exit 0 ;;\nesac\nexit 0\n')
        (binp / "docker").chmod(0o755)
        script = f"""
set -euo pipefail
CONFIG_DIR={d}; . "{REPO}/lanes.sh"
LANE={lane}; PENDING_PUT=1; UNITS_ONLY=0; WAITS_LOAD=0; WAITS_WHAT=""
IMAGE={self.PIN}; FLASH_IMAGE={self.OTHER_PIN}; SERVE_IMAGE={serve_image}; FLASH_SERVE_IMAGE={serve_image}
HF_CACHE={hf}
MODEL_REPO=RadixArk/Qwen3.8-27B-NVFP4; MODEL_REV={REV}; DRAFT2_REPO=maurienne-ai/Draft; DRAFT2_REV={DREV}
ckpt_cached(){{ return 0; }}
draft_in_cache(){{ return 0; }}
""" + self.BLOCK + """
echo "STATE WAITS_LOAD=$WAITS_LOAD UNITS_ONLY=$UNITS_ONLY WAITS_WHAT=$WAITS_WHAT"
"""
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30,
                           env={**os.environ, "PATH": f"{binp}:{os.environ['PATH']}"})
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def test_a_serving_override_present_and_the_pin_absent_is_no_lack(self):
        for lane in ("27b", "flash"):
            with self.subTest(lane=lane):
                out = self.run_probe(lane, "mon-build:local")
                self.assertIn("STATE WAITS_LOAD=0 UNITS_ONLY=0 WAITS_WHAT=", out)
                self.assertNotIn("needs downloads", out)

    def test_the_pin_absent_and_no_override_is_a_lack(self):
        # no SERVE_IMAGE=: the pin is what the unit serves, and its absence sends the update to its Load
        for lane in ("27b", "flash"):
            with self.subTest(lane=lane):
                out = self.run_probe(lane, self.PIN if lane == "27b" else self.OTHER_PIN)
                self.assertIn("STATE WAITS_LOAD=1 UNITS_ONLY=1", out)
                self.assertIn("image " + (self.PIN if lane == "27b" else self.OTHER_PIN), out)


class NoDebugLeftovers(unittest.TestCase):
    """Blind review of v1.52.1 (2026-11-19, GLM 5.3): a debug leftover wrote a debug file in /tmp on
    every run of ARuntimeThatMovesWaitsToo.run_fn, outside the test's own tmpdir, and the comment above
    patch_served_checkpoint still said a Load's prepare calls it, which v1.52.1 removed (the cockpit's
    apply_pending calls it, once the waiting unit is in place)."""

    def test_no_test_writes_a_debug_file_in_tmp(self):
        # built in two parts: this test's own source must not hold the marker
        marker = "/tmp/" + "dbg"
        for f in sorted((REPO / "tests").glob("test_*.py")):
            self.assertNotIn(marker, f.read_text(), f"{f.name} writes a debug file outside its tmpdir")

    def test_the_patch_served_checkpoint_comments_do_not_name_the_prepare_as_a_caller(self):
        lanes = (REPO / "lanes.sh").read_text()
        head = lanes[:lanes.index("patch_served_checkpoint(){")]
        comment = head[head.rindex("# The switch's steps 2, 2a and 2b"):]
        self.assertNotIn("prepare-lane.sh with the unit that waits", comment)
        self.assertIn("apply_pending", comment)
        sw = (REPO / "switch-model.sh").read_text()
        i = sw.index("# 2, 2a, 2b) the fetched checkpoint's own files")
        comment2 = sw[i:sw.index('patch_served_checkpoint "$TARGET_LANE"', i)]
        self.assertNotIn("prepare-lane.sh) calls too", comment2)
        self.assertIn("apply_pending", comment2)


if __name__ == "__main__":
    unittest.main()
