#!/usr/bin/env python3
"""A plain install installs the speech lane too (v1.58.0, decision 1564: what the repo ships is there for
everyone). It is no GPU lane: install.sh hands it whole to install-speech.sh, proves it when it is new or
named, and goes on when it fails. --no-speech counts for its run only, like every --no-<lane>.

install.sh's own lines run here under its own shell options, against a stub of install-speech.sh in a
throwaway folder: nothing reaches the box the suite runs on.
"""
import pathlib
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
TEXT = (REPO / "install.sh").read_text()


def between(start, end):
    s = TEXT.index(start)
    return TEXT[s:TEXT.index(end, s)]


STEP = between('SPEECH_UNIT_PATH="/etc/systemd/system/qwen38-speech.service"', "side_mode(){")
WANTED = (between("lane_title(){", "\n}\n") + "\n}\n" + between("lane_wanted(){", "\n}\n") + "\n}\n")


def run_step(want=1, with_=0, had=False, installer_rc=0, beside=False, marker=False):
    d = pathlib.Path(tempfile.mkdtemp(prefix="speech-step-"))
    stub = d / "install-speech.sh"
    stub.write_text(f'#!/bin/sh\necho "ARGS=[$*]" >> "{d}/calls"\nexit {installer_rc}\n')
    stub.chmod(0o755)
    unit = d / "qwen38-speech.service"
    if had:
        unit.write_text("[Unit]\n")
    cfg = d / "cfg"
    cfg.mkdir()
    if marker:
        (cfg / "speech.unproved").write_text("")
    script = ("set -euo pipefail\n" 'step(){ echo "STEP: $*"; }\n'
              f'REPO_DIR="{d}"; CONFIG_DIR="{cfg}"; WANT_SPEECH={want}; WITH_SPEECH={with_}\n{STEP}\n'
              f'SPEECH_UNIT_PATH="{unit}"\n'
              f'speech_step {"beside" if beside else ""}\necho "STATE=$SPEECH_STATE"\nspeech_line\necho "WENT ON"\n')
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                       env={"PATH": "/usr/bin:/bin", "HOME": str(d)})
    calls = (d / "calls").read_text().splitlines() if (d / "calls").exists() else []
    r.marker = (cfg / "speech.unproved").exists()
    return r, calls


class TheSpeechLaneIsInstalledByAPlainRun(unittest.TestCase):
    def test_a_box_without_it_gets_it_whole_and_proved(self):
        r, calls = run_step()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(calls, ["ARGS=[]"], "the installer runs whole, round trip included")
        self.assertIn("STATE=proved", r.stdout)
        self.assertIn("STEP: Speech lane", r.stdout)
        self.assertIn("installed and proved it hears what it says", r.stdout)
        self.assertFalse(r.marker)

    def test_a_box_that_has_it_is_updated_and_proves_nothing_twice(self):
        r, calls = run_step(had=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(calls, ["ARGS=[--no-smoke]"])
        self.assertIn("STATE=installed", r.stdout)
        self.assertIn("installed and up to date", r.stdout)

    def test_named_it_is_proved_again(self):
        r, calls = run_step(with_=1, had=True)
        self.assertEqual(calls, ["ARGS=[]"])
        self.assertIn("STATE=proved", r.stdout)

    def test_a_failure_is_said_and_the_run_goes_on(self):
        for had in (False, True):
            r, calls = run_step(had=had, installer_rc=1)
            self.assertEqual(r.returncode, 0, "set -e must not end the install on the speech lane: " + r.stderr)
            self.assertEqual(len(calls), 1)
            self.assertIn("STATE=failed", r.stdout)
            self.assertIn("NOTE: the speech lane did not install or update", r.stdout)
            self.assertIn("WENT ON", r.stdout)
            self.assertTrue(r.marker, "the lane has not proved it serves: the next run tries again")
        # a lane that never came, and an update that did not go through on a lane that is there: two different lines
        r, _ = run_step(had=False, installer_rc=1)
        self.assertIn("Speech     : NOT ready, see its NOTE above: ./install-speech.sh installs it", r.stdout)
        r, _ = run_step(had=True, installer_rc=1)
        self.assertIn("its update did NOT go through", r.stdout)
        self.assertIn("the lane installed before is as it was", r.stdout)
        self.assertNotIn("installs it", r.stdout)

    def test_a_lane_whose_first_proof_failed_is_proved_again_by_a_plain_run(self):
        """The unit is written before the round trip: a first run whose proof failed left the unit, and the next
        plain run took it for an installed lane, passed --no-smoke and said "up to date" (review of v1.58.0)."""
        r, calls = run_step(had=True, marker=True)
        self.assertEqual(calls, ["ARGS=[]"], "the round trip runs again")
        self.assertIn("STATE=proved", r.stdout)
        self.assertFalse(r.marker, "proved: the marker goes")
        r, calls = run_step(had=True, marker=True, installer_rc=1)
        self.assertIn("STATE=failed", r.stdout)
        self.assertTrue(r.marker)

    def test_beside_a_lane_that_serves_nothing_is_proved_and_it_is_said(self):
        """A box that boots another lane than its text lane: every lane is updated with --no-smoke there, and the
        speech lane's round trip would start two engines beside a lane that may be at work."""
        r, calls = run_step(beside=True)
        self.assertEqual(calls, ["ARGS=[--no-smoke]"])
        self.assertIn("STATE=unproved", r.stdout)
        self.assertIn("its round trip not run beside the lane that serves", r.stdout)
        self.assertTrue(r.marker, "the next plain run of a box back on its text lane proves it")
        r, calls = run_step(beside=True, had=True)
        self.assertEqual(calls, ["ARGS=[--no-smoke]"])
        self.assertIn("STATE=installed", r.stdout)
        r, calls = run_step(beside=True, with_=1)
        self.assertEqual(calls, ["ARGS=[]"], "named, it is proved wherever the box stands")
        self.assertIn("STATE=proved", r.stdout)

    def test_left_out_nothing_runs_and_the_summary_says_so(self):
        r, calls = run_step(want=0)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(calls, [])
        self.assertNotIn("STEP:", r.stdout)
        self.assertIn("left out of this run, as asked (--no-speech)", r.stdout)
        r, calls = run_step(want=0, had=True)
        self.assertEqual(calls, [])
        self.assertIn("left as it is, as asked (--no-speech)", r.stdout)


PKGS = between("SPEECH_PKGS=()\n", "# name the lanes that asked")
REASON = between('_pk_why=(); [ "$WANT_IMAGE" -eq 1 ]', '[ "${#HOST_PKGS[@]}" -eq 0 ] || host_packages')


class ItsBuildToolsAreAskedForOnce(unittest.TestCase):
    """host_packages asks once per run, in install.sh's preflight: the speech lane's own request, later in the run,
    is refused, and it stopped on "cmake is missing" on a box without a compiler (review of v1.58.0)."""

    def ask(self, have, want_speech=1, no_start=0, already=""):
        d = pathlib.Path(tempfile.mkdtemp(prefix="speech-pkgs-"))
        fence = d / "bin"
        fence.mkdir()
        for tool in have:
            (fence / tool).write_text("#!/bin/sh\nexit 0\n")
            (fence / tool).chmod(0o755)
        for tool in ("bash", "printf"):
            pass
        script = ("set -euo pipefail\n"
                  f'NO_START={no_start}; NO_SERVICE=0; WANT_SPEECH={want_speech}; WANT_IMAGE=0; WANT_VIDEO=0; WANT_IMAGE_H3=0\n'
                  f'HOST_PKGS=({already})\n{PKGS}{REASON}\n'
                  'echo "PKGS=[${HOST_PKGS[*]-}]"; echo "WHY=[$_pk_reason]"\n')
        r = subprocess.run(["/bin/bash", "-c", script], capture_output=True, text=True, timeout=60,
                           env={"PATH": str(fence), "HOME": str(d)})
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def test_a_box_without_a_compiler_is_asked_for_what_the_lane_builds_with(self):
        out = self.ask(have=["git", "curl", "taskset", "python3"])      # python3 here is a stub: "import venv" passes
        self.assertIn("PKGS=[cmake build-essential ffmpeg]", out)
        self.assertIn("WHY=[the speech lane]", out)

    def test_a_box_that_has_them_is_asked_nothing(self):
        out = self.ask(have=["git", "curl", "cmake", "c++", "taskset", "ffmpeg", "python3"])
        self.assertIn("PKGS=[]", out)
        self.assertIn("WHY=[]", out)

    def test_nothing_is_asked_twice_nor_for_a_run_that_leaves_it_out(self):
        out = self.ask(have=["git", "curl", "cmake", "c++", "taskset", "python3"], already="ffmpeg python3-dev")
        self.assertIn("PKGS=[ffmpeg python3-dev]", out)
        for kw in ({"want_speech": 0}, {"no_start": 1}):
            out = self.ask(have=[], **kw)
            self.assertIn("PKGS=[]", out)

    def test_the_reason_names_every_lane_that_asked(self):
        d = pathlib.Path(tempfile.mkdtemp(prefix="speech-why-"))
        for lanes, said in ((("image",), "the image lane"), (("image", "video"), "the image and video lanes"),
                            (("image", "video", "MiniMax-H3-Image"), "the image, video and MiniMax-H3-Image lanes"),
                            (("image", "video", "MiniMax-H3-Image", "speech"), "the image, video, MiniMax-H3-Image and speech lanes")):
            script = ("set -euo pipefail\n"
                      f'WANT_IMAGE={int("image" in lanes)}; WANT_VIDEO={int("video" in lanes)}; WANT_IMAGE_H3={int("MiniMax-H3-Image" in lanes)}\n'
                      f'SPEECH_PKGS=({"cmake" if "speech" in lanes else ""})\n{REASON}\necho "WHY=[$_pk_reason]"\n')
            r = subprocess.run(["/bin/bash", "-c", script], capture_output=True, text=True, timeout=60, env={"PATH": "/usr/bin:/bin", "HOME": str(d)})
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn(f"WHY=[{said}]", r.stdout)


SPEECH = (REPO / "install-speech.sh").read_text()
VENV_STEP = SPEECH[SPEECH.index("NEW_VENV=0\n"):SPEECH.index('cmp -s "$HERE/speech-server.py"')]


class AnInstalledLaneAsksPyPINothing(unittest.TestCase):
    """install.sh runs the speech lane's installer at every update since v1.58.0. Its venv step asked PyPI each
    time ("pip install --upgrade pip", with nothing to say its failure), so an update made while PyPI did not
    answer ended on a lane that was installed and serving (review of v1.58.0). A venv filled from this release's
    pins, by the sha256 of the requirements file, is left alone."""

    def step(self, stamp, pip_rc=0, venv=True):
        d = pathlib.Path(tempfile.mkdtemp(prefix="speech-venv-"))
        here, v = d / "repo", d / "venv"
        here.mkdir()
        (here / "speech-requirements.txt").write_text("kokoro-onnx==0.4.9\n")
        import hashlib
        sha = hashlib.sha256((here / "speech-requirements.txt").read_bytes()).hexdigest()
        fake = d / "bin"
        fake.mkdir()
        # python3 -m venv: makes the venv's two tools, as stubs that record what they are asked
        tool = ('#!/bin/sh\necho "$(basename "$0") $*" >> "%s/calls"\ncase "$*" in *freeze*) cat "%s/freeze" 2>/dev/null ;; '
                '*"pip --version"*) exit 0 ;; *install*) echo moved > "%s/freeze"; exit %d ;; esac\n' % (d, d, d, pip_rc))
        (fake / "python3").write_text('#!/bin/sh\necho "python3 $*" >> "%s/calls"\nmkdir -p "%s/bin"\n'
                                      'for t in python pip; do cp "%s/tool" "%s/bin/$t"; chmod 755 "%s/bin/$t"; done\n' % (d, v, d, v, v))
        (fake / "python3").chmod(0o755)
        (d / "tool").write_text(tool)
        if venv:
            (v / "bin").mkdir(parents=True)
            for t in ("python", "pip"):
                (v / "bin" / t).write_text(tool)
                (v / "bin" / t).chmod(0o755)
            if stamp is not None:
                (v / ".qwen38-requirements").write_text((sha if stamp == "same" else "0" * 64) + "\n")
        script = ('set -euo pipefail\ndie(){ echo "DIE: $*"; exit 9; }\n'
                  f'HERE="{here}"; VENV="{v}"; CHANGED=0\n{VENV_STEP}\necho "CHANGED=$CHANGED"\n')
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                           env={"PATH": f"{fake}:/usr/bin:/bin", "HOME": str(d)})
        calls = (d / "calls").read_text().splitlines() if (d / "calls").exists() else []
        stamped = (v / ".qwen38-requirements").read_text().strip() == sha if (v / ".qwen38-requirements").exists() else False
        return r, calls, stamped

    def test_a_venv_at_the_pins_is_left_alone(self):
        r, calls, _ = self.step("same")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse([c for c in calls if "install" in c], calls)
        self.assertIn("packages already at this release's pins", r.stdout)
        self.assertIn("CHANGED=0", r.stdout)

    def test_pins_that_moved_are_installed_once_and_the_lane_restarted_when_a_package_moved(self):
        r, calls, stamped = self.step("other")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(len([c for c in calls if c.startswith("pip install")]), 1, calls)
        self.assertFalse([c for c in calls if "--upgrade pip" in c], "pip's own upgrade is a new venv's only")
        self.assertTrue(stamped)
        self.assertIn("CHANGED=1", r.stdout)

    def test_a_new_venv_is_filled_and_pips_own_upgrade_never_stops_it(self):
        r, calls, stamped = self.step(None, venv=False)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue([c for c in calls if c.startswith("python3 -m venv")], calls)
        self.assertTrue(stamped)

    def test_packages_that_do_not_install_stop_it_and_leave_no_stamp(self):
        r, calls, stamped = self.step("other", pip_rc=1)
        self.assertEqual(r.returncode, 9, r.stdout + r.stderr)
        self.assertIn("DIE: the venv's packages did not install", r.stdout)
        self.assertFalse(stamped, "the next run tries again")


class ItsFlags(unittest.TestCase):
    def want(self, no, with_):
        cfg = pathlib.Path(tempfile.mkdtemp(prefix="speech-want-")) / ".config" / "qwen38"
        script = (f'set -euo pipefail\nRECOMMENDED=0\nMEM_CLASS=128\nCONFIG_DIR="{cfg}"\nPLAN_ONLY=0\nLANE=27b\nSIDE_BOOT=0\n'
                  'serves_anyway(){ [ "$1" = "$LANE" ] && [ "$SIDE_BOOT" -eq 0 ]; }\nserves_said(){ :; }\n'
                  f'{WANTED}\nlane_wanted speech {no} {with_}\necho "WANT=$WANT"\n')
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def test_on_by_default_and_off_for_the_run_that_says_so(self):
        self.assertIn("WANT=1", self.want(0, 0))
        self.assertIn("WANT=0", self.want(1, 0))
        self.assertIn("WANT=1", self.want(0, 1))

    def head(self, *flags):
        cut = TEXT.index('step "1/10 Preflight checks"')
        d = pathlib.Path(tempfile.mkdtemp(prefix="speech-flags-"))
        (d / "install.sh").write_text(TEXT[:cut] + 'echo "FLAGS OK want=$WANT_SPEECH"; exit 0\n')
        (d / "lanes.sh").write_text((REPO / "lanes.sh").read_text())
        return subprocess.run(["bash", str(d / "install.sh"), *flags], capture_output=True, text=True, timeout=60,
                              cwd=str(REPO), env={"PATH": "/usr/bin:/bin", "HOME": str(d)})

    def test_both_flags_are_refused_before_anything(self):
        r = self.head("--with-speech", "--no-speech")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--no-speech and --with-speech contradict each other", r.stderr)
        self.assertNotIn("FLAGS OK", r.stdout)

    def test_named_without_the_services_is_refused(self):
        for flag in ("--no-start", "--no-service"):
            r = self.head("--with-speech", flag)
            self.assertNotEqual(r.returncode, 0, flag)
            self.assertIn("--with-speech needs the full install", r.stderr)
            self.assertIn("./install-speech.sh on its own", r.stderr)

    def test_the_help_names_them(self):
        r = self.head("--help")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("[--no-speech | --with-speech]", r.stdout)
        self.assertIn("--no-speech           leave the speech lane out of this run", r.stdout)


class TheTwoPathsBothReachIt(unittest.TestCase):
    """The run that serves a text lane and the run of a box booting another lane each call the step once and
    print its line once: a box booting the image lane would otherwise never get the speech lane."""

    def test_step_and_line_each_stand_in_both_paths(self):
        side = between("# Since v1.20 a side lane missing here is installed too", "record_install side-boot")
        self.assertEqual(side.count("\n  speech_step beside\n"), 1, "nothing is proved beside the lane that serves")
        self.assertEqual(side.count("\n  speech_line\n"), 1)
        main = TEXT[TEXT.index("record_install side-boot"):]
        self.assertEqual(main.count("\n    speech_step\n"), 1)
        self.assertEqual(main.count("\n    speech_line\n"), 1)
        self.assertLess(main.index("\n    speech_step\n"), main.index("\n    speech_line\n"))


if __name__ == "__main__":
    unittest.main()
