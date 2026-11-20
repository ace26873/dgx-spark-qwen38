#!/usr/bin/env python3
"""install.sh puts the opencode version this repo tests on every box, and only that.

The repo tunes four things to opencode's own behaviour (the compaction threshold, the
hidden 32,000-token output cap, the overflow phrases the proxy answers with, --yolo and
OPENCODE_PERMISSION for the Agent tab), all read out of one version's binary. Before the
pin, the installer took whatever opencode was on PATH, and opencode installs its own
patch releases, so two boxes installed a week apart ran two versions (the reference box
sat on 1.18.27 while 1.18.32 was out). These run the installer's own lines, as written,
in a throwaway HOME, against a fake curl and fake opencode binaries.
"""
import hashlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import tarfile
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
INSTALL = REPO / "install.sh"
PINNED = "1.19.6"

FAKE_OC = """#!/bin/sh
here="$(cd "$(dirname "$0")" && pwd)"
case "$1" in
  --version) cat "$here/VERSION" ;;
  upgrade) echo "upgrade $2" >> "$OC_LOG"; printf '%s\\n' "$2" > "$here/VERSION" ;;
esac
"""
FAKE_CURL = """#!/bin/sh
out=""; url=""
while [ $# -gt 0 ]; do case "$1" in -o) out="$2"; shift 2 ;; -m|--retry) shift 2 ;; -*) shift ;; *) url="$1"; shift ;; esac; done
echo "curl $url" >> "$OC_LOG"
cp "$OC_TARBALL" "$out"
"""


def block(whole: bool = False) -> str:
    text = INSTALL.read_text()
    start = text.index('OC_HOME_BIN="$HOME/.opencode/bin/opencode"')
    if whole:
        # through the marker and the rollback to 1.x, the two steps the chain hands over to (v2 migration)
        return text[start:text.index("# The rest of this run (the Agent tab", start)]
    tail = text.index('echo "added ~/.opencode/bin to your PATH', start)
    end = text.index("\n  fi\nfi\n", tail) + len("\n  fi\nfi\n")
    return text[start:end]


def argloop() -> str:
    """install.sh's own flag parser: from the RECOMMENDED line to the done that closes its loop."""
    text = INSTALL.read_text()
    start = text.index('RECOMMENDED="${QWEN38_RECOMMENDED:-0}"')
    end = text.index("\ndone\n", start) + len("\ndone\n")
    return text[start:end]


def rollback() -> str:
    """install.sh's own lines that take OUR v2 array out when the box goes back to 1.x."""
    text = INSTALL.read_text()
    start = text.index('# Going back to 1.x takes OUR v2 deny array')
    return text[start:text.index("# The rest of this run (the Agent tab", start)]


def major_choice() -> str:
    """install.sh's own lines that resolve which OpenCode this run serves, and the marker as it
    stood at the start of the run, from OPENCODE_MAJOR=2 to the export."""
    text = INSTALL.read_text()
    start = text.index("OPENCODE_MAJOR=2\n# The marker as it stands at the START")
    end = text.index('export OC_MAJOR="$OPENCODE_MAJOR"', start) + len('export OC_MAJOR="$OPENCODE_MAJOR"')
    return text[start:end]


def pinned_default() -> str:
    import re
    return re.search(r'^OPENCODE_VERSION="\$\{OPENCODE_VERSION:-([0-9.]+)\}"', INSTALL.read_text(), re.M).group(1)


class TheOpencodePin(unittest.TestCase):
    def setUp(self):
        self.t = pathlib.Path(tempfile.mkdtemp(prefix="oc-pin-"))
        self.addCleanup(shutil.rmtree, self.t, True)
        self.home = self.t / "home"
        self.home.mkdir()
        (self.home / ".bashrc").write_text("# a user's bashrc\n")
        self.bin = self.t / "fakebin"
        self.bin.mkdir()
        (self.bin / "curl").write_text(FAKE_CURL)
        (self.bin / "curl").chmod(0o755)
        self.log = self.t / "calls.log"
        # the release asset: one "opencode" at the top of the archive, like the real one
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            body = f"#!/bin/sh\n[ \"$1\" = --version ] && echo {PINNED}\n".encode()
            info = tarfile.TarInfo("opencode"); info.size = len(body); info.mode = 0o755
            tf.addfile(info, io.BytesIO(body))
        self.tarball = self.t / "asset.tar.gz"
        self.tarball.write_bytes(buf.getvalue())
        self.sha = hashlib.sha256(buf.getvalue()).hexdigest()

    def opencode_at(self, where: pathlib.Path, version: str) -> pathlib.Path:
        where.mkdir(parents=True, exist_ok=True)
        (where / "opencode").write_text(FAKE_OC)
        (where / "opencode").chmod(0o755)
        (where / "VERSION").write_text(version + "\n")
        return where / "opencode"

    def run_block(self, *, path_extra=(), pin="1", sha=None, env_version="", env_sha="", whole=False):
        script = ("set -euo pipefail\ndie(){ echo \"DIE: $*\"; exit 1; }\n"
                  f'OPENCODE_VERSION="{PINNED}"; OPENCODE_SHA256="{sha or self.sha}"; OPENCODE_PIN="{pin}"\n'
                  f'_ENV_OPENCODE_VERSION="{env_version}"; _ENV_OPENCODE_SHA256="{env_sha}"\n'
                  + block(whole) + 'echo "PATH_AFTER=$PATH"\n')
        path = ":".join([str(self.bin), *map(str, path_extra), "/usr/bin", "/bin"])
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                           env={"HOME": str(self.home), "PATH": path, "OC_LOG": str(self.log),
                                "OC_TARBALL": str(self.tarball)})
        calls = self.log.read_text().splitlines() if self.log.exists() else []
        return r.returncode, r.stdout + r.stderr, calls

    @property
    def home_bin(self):
        return self.home / ".opencode" / "bin" / "opencode"

    def version_of(self, binary):
        return subprocess.run([str(binary), "--version"], capture_output=True, text=True).stdout.strip()

    def test_the_default_pin_is_the_version_these_tests_hold(self):
        self.assertEqual(pinned_default(), PINNED)

    def test_absent_it_installs_the_release_checked_against_its_sha256(self):
        rc, out, calls = self.run_block()
        self.assertEqual(rc, 0, out)
        self.assertIn(f"opencode is not installed: installing {PINNED}", out)
        self.assertIn(f"opencode {PINNED} installed at {self.home_bin}", out)
        self.assertEqual(self.version_of(self.home_bin), PINNED)
        self.assertEqual(calls, [f"curl https://github.com/anomalyco/opencode/releases/download/v{PINNED}"
                                 "/opencode-linux-arm64.tar.gz"])
        self.assertIn(".opencode/bin", (self.home / ".bashrc").read_text())
        self.assertIn(f"{self.home}/.opencode/bin", out.split("PATH_AFTER=")[1])

    def test_one_already_here_but_on_no_path_gets_its_line(self):
        # The same box one update later: opencode in ~/.opencode/bin, no line, and the
        # update never wrote one because opencode was no longer "absent".
        self.opencode_at(self.home / ".opencode" / "bin", PINNED)
        rc, out, _ = self.run_block()
        self.assertEqual(rc, 0, out)
        bashrc = (self.home / ".bashrc").read_text()
        self.assertIn("put on the PATH by dgx-spark-qwen38", bashrc)
        self.assertNotIn("installed by dgx-spark-qwen38", bashrc, "this run did not install it")

    def test_one_already_on_the_path_gets_no_line(self):
        home_bin = self.opencode_at(self.home / ".opencode" / "bin", PINNED)
        rc, out, _ = self.run_block(path_extra=[home_bin.parent])
        self.assertEqual(rc, 0, out)
        self.assertEqual((self.home / ".bashrc").read_text(), "# a user's bashrc\n")

    def test_a_commented_out_path_line_is_not_a_path(self):
        # The reference box, 2026-09-23: a .bashrc whose opencode line was commented
        # out counted as having one, and the opencode just installed stayed off the PATH.
        (self.home / ".bashrc").write_text("# export PATH=$HOME/.opencode/bin:$PATH\n")
        rc, out, _ = self.run_block()
        self.assertEqual(rc, 0, out)
        self.assertIn("installed by dgx-spark-qwen38", (self.home / ".bashrc").read_text())

    def test_the_pinned_version_is_left_alone(self):
        self.opencode_at(self.home / ".opencode" / "bin", PINNED)
        rc, out, calls = self.run_block()
        self.assertEqual(rc, 0, out)
        self.assertIn("the version this repo tests", out)
        self.assertEqual(calls, [])

    def test_an_older_one_where_the_installer_puts_it_is_replaced_the_same_way(self):
        self.opencode_at(self.home / ".opencode" / "bin", "1.18.27")
        (self.home / ".bashrc").write_text('export PATH="$HOME/.opencode/bin:$PATH"\n')
        rc, out, calls = self.run_block()
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.version_of(self.home_bin), PINNED)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].startswith("curl "))
        self.assertNotIn("installed by dgx-spark-qwen38", (self.home / ".bashrc").read_text(),
                         "the user's PATH line was theirs already; nothing to add")

    def test_an_older_one_from_npm_goes_through_opencodes_own_upgrader(self):
        npm_bin = self.opencode_at(self.t / "npm" / "bin", "1.18.20")
        rc, out, calls = self.run_block(path_extra=[npm_bin.parent])
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls, [f"upgrade {PINNED}"])
        self.assertEqual(self.version_of(npm_bin), PINNED)
        self.assertFalse(self.home_bin.exists(), "a second copy was installed beside the npm one")

    def homebrew_opencode(self, version, tap):
        """opencode as Homebrew lays it out: the keg in Cellar, its install receipt (tap None:
        no receipt), and <prefix>/bin/opencode linking to it."""
        keg = self.t / "linuxbrew" / "Cellar" / "opencode" / version
        self.opencode_at(keg / "bin", version)
        if tap is not None:
            (keg / "INSTALL_RECEIPT.json").write_text(json.dumps(
                {"homebrew_version": "4.6.0", "source": {"tap": tap, "spec": "stable"}}, indent=2))
        linked = self.t / "linuxbrew" / "bin"
        linked.mkdir(parents=True, exist_ok=True)
        (linked / "opencode").symlink_to(keg / "bin" / "opencode")
        (linked / "VERSION").write_text(version + "\n")   # the fake reads it beside the path it ran by
        return linked

    def assert_homebrew_kept(self, tap):
        linked = self.homebrew_opencode("1.18.32", tap)
        rc, out, calls = self.run_block(path_extra=[linked])
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls, [], "neither opencode's upgrader nor a download ran")
        self.assertIn("opencode 1.18.32 comes from Homebrew, not from opencode's own tap, kept", out)
        self.assertIn("brew uninstall opencode", out)
        self.assertIn("brew install anomalyco/tap/opencode", out)
        self.assertEqual(self.version_of(linked / "opencode"), "1.18.32")
        self.assertFalse(self.home_bin.exists(), "a second copy was installed beside Homebrew's")

    def test_an_older_one_from_homebrew_core_is_kept_and_its_upgrader_never_runs(self):
        """opencode's upgrader runs brew upgrade whatever version it is asked for (up to 1.19.1),
        and Homebrew Core's formula is OpenCode 2 now (2.1.2, 2026-10-22): an update put
        OpenCode 2 under oc and the Agent tab."""
        self.assert_homebrew_kept("homebrew/core")

    def test_a_homebrew_keg_that_does_not_say_its_tap_is_kept_too(self):
        self.assert_homebrew_kept(None)

    def test_an_older_one_from_opencodes_own_tap_goes_through_its_upgrader(self):
        """opencode's own formula (anomalyco/tap/opencode, sst/tap/opencode before the project
        moved) stays on 1.x, and its upgrader moves it to the tap's version: kept, it would have
        stayed on the Bun without the fix this pin is for."""
        for tap in ("anomalyco/tap", "sst/tap"):
            with self.subTest(tap=tap):
                self.log.unlink(missing_ok=True)
                linked = self.homebrew_opencode("1.18.32", tap)
                rc, out, calls = self.run_block(path_extra=[linked])
                self.assertEqual(rc, 0, out)
                self.assertEqual(calls, [f"upgrade {PINNED}"])
                self.assertNotIn("comes from Homebrew", out)
                shutil.rmtree(self.t / "linuxbrew")

    def test_a_newer_one_is_kept_and_said_so(self):
        # newer than whatever the pin is (1.18.40 was, until the pin moved past it)
        major, minor, patch = PINNED.split(".")
        newer = f"{major}.{minor}.{int(patch) + 6}"
        self.opencode_at(self.home / ".opencode" / "bin", newer)
        rc, out, calls = self.run_block()
        self.assertEqual(rc, 0, out)
        self.assertIn("newer than the", out)
        self.assertEqual(calls, [])
        self.assertEqual(self.version_of(self.home_bin), newer)

    def test_a_first_install_that_cannot_download_says_what_is_missing_and_what_to_run(self):
        (self.bin / "curl").write_text("#!/bin/sh\nexit 22\n")
        rc, out, _ = self.run_block()
        self.assertEqual(rc, 0, out)
        self.assertIn("could not download opencode", out)
        self.assertIn("opencode is still not installed", out)
        self.assertIn("Re-run ./install.sh", out)
        self.assertNotIn("installed at", out)

    def test_a_download_that_does_not_match_its_sha256_is_not_installed(self):
        self.opencode_at(self.home / ".opencode" / "bin", "1.18.27")
        rc, out, _ = self.run_block(sha="0" * 64)
        self.assertEqual(rc, 0, out)                              # opencode is one integration
        self.assertIn("does not match its pinned sha256", out)
        self.assertEqual(self.version_of(self.home_bin), "1.18.27")

    def test_opencode_pin_0_keeps_whatever_is_there(self):
        self.opencode_at(self.home / ".opencode" / "bin", "1.18.27")
        rc, out, calls = self.run_block(pin="0")
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls, [])
        self.assertEqual(self.version_of(self.home_bin), "1.18.27")

    def test_a_version_without_its_checksum_is_refused_before_step_1(self):
        """It was refused at step 7, after the image, the checkpoints and the engine's
        configs (found in review, 2026-09-24)."""
        text = INSTALL.read_text()
        i = text.index("needs OPENCODE_SHA256 too")
        start = text.rindex("\nif ", 0, i) + 1
        refusal = text[start:text.index("\nfi\n", i) + 4]
        self.assertLess(i, text.index('step "1/10 Preflight checks"'))
        for opencode, version, sha, rc in (("1", "1.18.40", "", 1), ("1", "1.18.40", "ab" * 32, 0),
                                           ("0", "1.18.40", "", 0), ("1", "", "", 0)):
            script = ("set -euo pipefail\ndie(){ echo \"DIE: $*\"; exit 1; }\n"
                      f'OPENCODE={opencode}; OPENCODE_VERSION="{version or PINNED}"\n'
                      f'_ENV_OPENCODE_VERSION="{version}"; _ENV_OPENCODE_SHA256="{sha}"\n' + refusal)
            r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
            self.assertEqual(r.returncode, rc, (opencode, version, sha, r.stdout))
            if rc:
                self.assertIn("needs OPENCODE_SHA256 too", r.stdout)


class TheDefaultIsTwo(unittest.TestCase):
    """Since v1.56.0 (decision 1546 of 2026-11-20, replacing the opt-in plan of 2026-10-22) a
    ./install.sh without --opencode serves OpenCode 2: the pinned 2.x asset is downloaded and
    the marker says 2. --opencode 2 is still accepted. --opencode 1 writes the marker "1", so a
    later plain run keeps the box on 1.x; the first update of a pre-v1.56.0 1.x box, which has
    no marker, moves to 2. A stray 2.x on the PATH of a no-marker box is replaced by the pin,
    and a box marked 1 is never moved to 2 by a binary it did not ask for: a failed 1.x
    download keeps it serving what stands, marker 1, nothing written for 2."""

    V2 = "2.3.2"
    # serves the asset whose version the URL names, so a crossing downloads "the right" bytes
    FAKE_CURL_V = ("#!/bin/sh\n"
                   "out=\"\"; url=\"\"\n"
                   "while [ $# -gt 0 ]; do case \"$1\" in -o) out=\"$2\"; shift 2 ;; -m|--retry) shift 2 ;; -*) shift ;; *) url=\"$1\"; shift ;; esac; done\n"
                   "echo \"curl $url\" >> \"$OC_LOG\"\n"
                   "case \"$url\" in *v" + V2 + "*) cp \"$OC_TAR_V2\" \"$out\" ;; *) cp \"$OC_TAR_V1\" \"$out\" ;; esac\n")

    def setUp(self):
        self.t = pathlib.Path(tempfile.mkdtemp(prefix="oc-def2-"))
        self.addCleanup(shutil.rmtree, self.t, True)
        self.home = self.t / "home"
        self.home.mkdir()
        (self.home / ".bashrc").write_text("# a user's bashrc\n")
        self.cfg = self.home / ".config" / "qwen38"
        self.bin = self.t / "fakebin"
        self.bin.mkdir()
        (self.bin / "curl").write_text(self.FAKE_CURL_V)
        (self.bin / "curl").chmod(0o755)
        self.log = self.t / "calls.log"
        self.assets = {}
        for ver in (PINNED, self.V2):
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                body = f"#!/bin/sh\n[ \"$1\" = --version ] && echo {ver}\n".encode()
                info = tarfile.TarInfo("opencode"); info.size = len(body); info.mode = 0o755
                tf.addfile(info, io.BytesIO(body))
            p = self.t / f"asset-{ver}.tar.gz"
            p.write_bytes(buf.getvalue())
            self.assets[ver] = (p, hashlib.sha256(buf.getvalue()).hexdigest())
        self.brew = self.t / "brew"
        self.brew.mkdir()
        (self.brew / "opencode").write_text(FAKE_OC)
        (self.brew / "opencode").chmod(0o755)

    @property
    def home_bin(self):
        return self.home / ".opencode" / "bin" / "opencode"

    def version_of(self, binary):
        return subprocess.run([str(binary), "--version"], capture_output=True, text=True).stdout.strip()

    def run_home(self, *, want="", marker=None, brew_ver=None, v1_sha=None, v2_sha=None, plugins=()):
        if marker is not None:
            self.cfg.mkdir(parents=True, exist_ok=True)
            (self.cfg / "opencode.major").write_text(marker)
        if brew_ver is not None:
            (self.brew / "VERSION").write_text(brew_ver + "\n")
        plugin_dir = self.home / ".config" / "opencode" / "plugins"
        if plugins:
            plugin_dir.mkdir(parents=True, exist_ok=True)
            for name in plugins:
                (plugin_dir / name).write_text(f"// the operator's {name}\n")
        script = ("set -euo pipefail\ndie(){ echo \"DIE: $*\"; exit 1; }\n"
                  f'REPO_DIR="{REPO}"\nCONFIG_DIR="{self.cfg}"\n'
                  f'OPENCODE_VERSION="{PINNED}"; OPENCODE_SHA256="{v1_sha or self.assets[PINNED][1]}"; OPENCODE_PIN="1"\n'
                  f'OPENCODE2_VERSION="{self.V2}"; OPENCODE2_SHA256="{v2_sha or self.assets[self.V2][1]}"\n'
                  f'_ENV_OPENCODE_VERSION=""; _ENV_OPENCODE_SHA256=""\n'
                  f'OPENCODE_MAJOR_WANT="{want}"\n'
                  + major_choice() + "\n" + block(whole=True)
                  + '\necho "CHOICE=${OPENCODE_MAJOR:-1} SERVE=${OC_SERVE_MAJOR:-none} START=${OC_START_MAJOR:-none}"\n')
        path_extra = [str(self.brew)] if brew_ver is not None else []
        path = ":".join([str(self.bin), *path_extra, "/usr/bin", "/bin"])
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                           env={"HOME": str(self.home), "PATH": path, "OC_LOG": str(self.log),
                                "OC_TAR_V1": str(self.assets[PINNED][0]),
                                "OC_TAR_V2": str(self.assets[self.V2][0])})
        marker_file = self.cfg / "opencode.major"
        calls = self.log.read_text().splitlines() if self.log.exists() else []
        self.plugins_after = (sorted(p.name for p in plugin_dir.iterdir())
                              if plugin_dir.is_dir() else None)
        return (r.returncode, r.stdout + r.stderr,
                marker_file.read_text() if marker_file.exists() else None, calls)

    def test_no_flag_installs_the_pinned_2_and_marks_the_box_2(self):
        rc, out, marker, calls = self.run_home()
        self.assertEqual(rc, 0, out)
        self.assertIn("CHOICE=2 SERVE=2 START=none", out)
        self.assertEqual(calls, [f"curl https://github.com/anomalyco/opencode/releases/download/v{self.V2}"
                                 "/opencode-linux-arm64.tar.gz"])
        self.assertEqual(self.version_of(self.home_bin), self.V2)
        self.assertIn(f"installing {self.V2}", out)
        self.assertIsNotNone(marker)
        self.assertEqual(marker.splitlines()[0], "2", "a plain run writes the marker 2")

    def test_opencode_2_flag_is_still_accepted(self):
        rc, out, marker, calls = self.run_home(want="2")
        self.assertEqual(rc, 0, out)
        self.assertIn("CHOICE=2 SERVE=2", out)
        self.assertEqual(self.version_of(self.home_bin), self.V2)
        self.assertEqual(marker.splitlines()[0], "2")

    def test_opencode_1_marks_the_box_1(self):
        rc, out, marker, calls = self.run_home(want="1")
        self.assertEqual(rc, 0, out)
        self.assertIn("CHOICE=1 SERVE=1", out)
        self.assertEqual(calls, [f"curl https://github.com/anomalyco/opencode/releases/download/v{PINNED}"
                                 "/opencode-linux-arm64.tar.gz"])
        self.assertEqual(self.version_of(self.home_bin), PINNED)
        self.assertIsNotNone(marker)
        self.assertEqual(marker.splitlines()[0], "1", "--opencode 1 is memorized in the marker")

    def test_a_box_marked_1_stays_1_on_a_plain_update(self):
        rc, out, marker, calls = self.run_home(marker="1\n# asked with ./install.sh --opencode 1\n",
                                               brew_ver=PINNED)
        self.assertEqual(rc, 0, out)
        self.assertIn("CHOICE=1 SERVE=1", out, "the marker 1 keeps the box on 1.x across updates")
        self.assertEqual(calls, [], "the pinned 1.x is already in place: no download")
        self.assertIn("the version this repo tests", out)
        self.assertEqual(marker.splitlines()[0], "1", "the marker is rewritten as 1, not erased")

    def test_a_box_marked_2_stays_2_on_a_plain_update(self):
        rc, out, marker, calls = self.run_home(marker="2\n", brew_ver=self.V2)
        self.assertEqual(rc, 0, out)
        self.assertIn("CHOICE=2 SERVE=2", out)
        self.assertEqual(calls, [])
        self.assertEqual(marker.splitlines()[0], "2")

    def test_a_plain_update_of_a_pre_v156_1_x_box_moves_to_2(self):
        # no marker (the box was installed before v1.56.0), a 1.x binary in place: the
        # default wins, the pinned 2.x comes down over it, the marker becomes 2
        rc, out, marker, calls = self.run_home(brew_ver=PINNED)
        self.assertEqual(rc, 0, out)
        self.assertIn("CHOICE=2 SERVE=2 START=1", out)
        self.assertEqual(calls, [f"curl https://github.com/anomalyco/opencode/releases/download/v{self.V2}"
                                 "/opencode-linux-arm64.tar.gz"])
        self.assertEqual(self.version_of(self.home_bin), self.V2)
        self.assertEqual(marker.splitlines()[0], "2")
        self.assertIn("installing", out)

    def test_a_stray_2_x_on_a_no_marker_box_is_replaced_by_the_pinned_2(self):
        rc, out, marker, calls = self.run_home(brew_ver="2.3.0")
        self.assertEqual(rc, 0, out)
        self.assertIn("CHOICE=2 SERVE=2", out)
        self.assertEqual(self.version_of(self.home_bin), self.V2, "the pin replaced the stray 2.3.0")
        self.assertEqual(marker.splitlines()[0], "2")

    def test_a_box_marked_1_with_a_stray_2_x_keeps_its_choice_when_the_download_fails(self):
        rc, out, marker, calls = self.run_home(marker="1\n", brew_ver="2.3.0", v1_sha="0" * 64)
        self.assertEqual(rc, 0, out)
        self.assertIn("CHOICE=1 SERVE=2", out, "the 2.x binary in place is what serves")
        self.assertIn("keeps serving the OpenCode already installed", out)
        self.assertIn("2.3.0", out)
        self.assertIn("Re-run ./install.sh --opencode 1", out)
        self.assertIsNotNone(marker)
        self.assertEqual(marker.splitlines()[0], "1", "a failed crossing does not touch the marker")

    def test_migrating_to_2_swaps_the_auto_continue_plugin(self):
        # the operator had the v1 plugin: the v2 port goes in, the v1 file comes out
        # (OpenCode 2 does not run it: module.ts of 2.3.2 requires a default export)
        rc, out, marker, calls = self.run_home(brew_ver=PINNED, plugins=("auto-continue.js",))
        self.assertEqual(rc, 0, out)
        self.assertIn("CHOICE=2 SERVE=2 START=1", out)
        self.assertEqual(self.plugins_after, ["auto-continue-v2.js"])
        v2 = (self.home / ".config" / "opencode" / "plugins" / "auto-continue-v2.js").read_text()
        self.assertEqual(v2, (REPO / "extras" / "opencode" / "auto-continue-v2.js").read_text())
        self.assertIn("auto-continue plugin:", out)

    def test_a_box_that_never_installed_a_plugin_gets_none(self):
        rc, out, marker, calls = self.run_home(brew_ver=PINNED)
        self.assertEqual(rc, 0, out)
        self.assertIsNone(self.plugins_after, "no plugins dir was created")
        self.assertNotIn("auto-continue plugin:", out)

    def test_going_back_to_1_restores_the_v1_plugin(self):
        rc, out, marker, calls = self.run_home(want="1", marker="2\n", brew_ver=self.V2,
                                               plugins=("auto-continue-v2.js",))
        self.assertEqual(rc, 0, out)
        self.assertIn("CHOICE=1 SERVE=1", out)
        self.assertEqual(self.plugins_after, ["auto-continue.js"])
        v1 = (self.home / ".config" / "opencode" / "plugins" / "auto-continue.js").read_text()
        self.assertEqual(v1, (REPO / "extras" / "opencode" / "auto-continue.js").read_text())
        self.assertIn("auto-continue plugin:", out)


class TheFailedCrossingServesWhatStands(unittest.TestCase):
    """A download that fails while the run crosses majors (1 to 2 or 2 to 1) leaves NO mixed
    state: nothing is written for the major the run was asked to switch to (no marker, no
    config key, no unit), the message says what really serves (the binary in place), the
    cloud-metadata and aliyuncs denies stay in the configs of the major that runs, and a
    re-run repairs. Blind review of 2026-11-20, D2/D3."""

    GEN_V2 = {"compaction": {"keep": {"tokens": 135000}, "background": True},
              "provider": {"qwen38": {}}, "autoupdate": False}
    USER_V2 = {"compaction": {"keep": {"tokens": 135000}, "background": True, "prune": False},
               "theme": "dark",
               "permissions": [{"action": "shell", "resource": "*aliyuncs.com*", "effect": "deny"},
                               {"action": "shell", "resource": "*169.254.169.254*", "effect": "deny"},
                               {"action": "shell", "resource": "*100.100.100.200*", "effect": "deny"}]}
    DENIES = ["*aliyuncs.com*", "*169.254.169.254*", "*100.100.100.200*"]

    def setUp(self):
        self.t = pathlib.Path(tempfile.mkdtemp(prefix="oc-xfail-"))
        self.addCleanup(shutil.rmtree, self.t, True)
        self.home = self.t / "home"
        self.cfg = self.home / ".config" / "qwen38"
        self.cfg.mkdir(parents=True)
        (self.home / ".config" / "opencode").mkdir(parents=True)
        (self.home / ".bashrc").write_text("# a user's bashrc\n")
        self.bin = self.t / "fakebin"
        self.bin.mkdir()
        (self.bin / "curl").write_text(FAKE_CURL)
        (self.bin / "curl").chmod(0o755)
        self.log = self.t / "calls.log"
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            body = f"#!/bin/sh\n[ \"$1\" = --version ] && echo {PINNED}\n".encode()
            info = tarfile.TarInfo("opencode"); info.size = len(body); info.mode = 0o755
            tf.addfile(info, io.BytesIO(body))
        self.tarball = self.t / "asset.tar.gz"
        self.tarball.write_bytes(buf.getvalue())
        self.sha = hashlib.sha256(buf.getvalue()).hexdigest()
        self.brew = self.t / "brew"
        self.brew.mkdir()
        (self.brew / "opencode").write_text(FAKE_OC)
        (self.brew / "opencode").chmod(0o755)

    def run_crossing(self, *, brew_version, marker=None, want="", v1_sha=None):
        """The installer's own choice + download + marker + rollback lines, run whole, with a
        binary of the other major on the PATH and a download that fails its sha256 check."""
        (self.brew / "VERSION").write_text(brew_version + "\n")
        if marker:
            (self.cfg / "opencode.major").write_text("2\n# asked with ./install.sh --opencode 2\n")
        gen = self.cfg / "opencode.json"
        gen.write_text(json.dumps(self.GEN_V2, indent=2) + "\n")
        user = self.home / ".config" / "opencode" / "opencode.json"
        user.write_text(json.dumps(self.USER_V2, indent=2) + "\n")
        script = ("set -euo pipefail\ndie(){ echo \"DIE: $*\"; exit 1; }\n"
                  f'REPO_DIR="{REPO}"\nCONFIG_DIR="{self.cfg}"\n'
                  f'OPENCODE_VERSION="{PINNED}"; OPENCODE_SHA256="{v1_sha or self.sha}"; OPENCODE_PIN="1"\n'
                  f'OPENCODE2_VERSION="2.3.2"; OPENCODE2_SHA256="{"0" * 64}"\n'
                  f'_ENV_OPENCODE_VERSION=""; _ENV_OPENCODE_SHA256=""\n'
                  f'OPENCODE_MAJOR_WANT="{want}"\n'
                  + major_choice() + "\n" + block(whole=True)
                  + '\necho "CHOICE=${OPENCODE_MAJOR:-1} SERVE=${OC_SERVE_MAJOR:-none} OCMAJOR=${OC_MAJOR:-none}"\n')
        path = ":".join([str(self.bin), str(self.brew), "/usr/bin", "/bin"])
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                           env={"HOME": str(self.home), "PATH": path, "OC_LOG": str(self.log),
                                "OC_TARBALL": str(self.tarball)})
        marker_file = self.cfg / "opencode.major"
        return (r.returncode, r.stdout + r.stderr, json.loads(gen.read_text()),
                json.loads(user.read_text()),
                marker_file.read_text() if marker_file.exists() else None)

    def test_d2_a_failed_1_x_download_on_a_2_box_keeps_serving_2(self):
        # marker 2, binary 2.3.2, ./install.sh --opencode 1 while GitHub is unreachable
        rc, out, gen, user, marker = self.run_crossing(brew_version="2.3.2", marker="2", want="1",
                                                       v1_sha="0" * 64)
        self.assertEqual(rc, 0, out)
        self.assertIn("CHOICE=1 SERVE=2 OCMAJOR=2", out)
        self.assertIn("could not come down", out)
        self.assertIn("keeps serving the OpenCode already installed", out)
        self.assertIn("2.3.2", out)
        self.assertIn("Re-run ./install.sh --opencode 1", out)
        self.assertIsNotNone(marker)
        self.assertEqual(marker.splitlines()[0], "2", "a failed crossing does not touch the marker")
        # the guard holds: the configs of the major that runs keep the exfiltration denies
        self.assertEqual([p["resource"] for p in user["permissions"]], self.DENIES)
        self.assertEqual(gen["compaction"], {"keep": {"tokens": 135000}, "background": True},
                         "the v2 compaction keys stay while the 2.x binary stands")

    def test_d3_a_failed_2_x_download_on_a_1_x_binary_keeps_serving_1(self):
        # marker 2, binary back at 1.19.0 (brew, another install, OPENCODE_PIN=0), plain run
        rc, out, gen, user, marker = self.run_crossing(brew_version="1.19.0", marker="2", want="")
        self.assertEqual(rc, 0, out)
        self.assertIn("CHOICE=2 SERVE=1 OCMAJOR=1", out)
        self.assertIn("keeps serving the OpenCode already installed", out)
        self.assertIn("1.19.0", out)
        self.assertIn("Re-run ./install.sh --opencode 2", out)
        self.assertIsNotNone(marker)
        self.assertEqual(marker.splitlines()[0], "2",
                         "the marker stays 2, so a plain re-run retries the 2.x download and repairs")
        # 1.x must be able to start: our v2 array is out of the global config, and the v2
        # compaction keys are out of both configs
        self.assertNotIn("permissions", user)
        self.assertNotIn("keep", gen["compaction"])
        self.assertNotIn("background", gen["compaction"])
        self.assertNotIn("keep", user["compaction"])
        self.assertEqual(user["compaction"].get("prune"), False)
        self.assertEqual(user["theme"], "dark")

    def test_everything_downstream_follows_the_binary_that_stands(self):
        text = INSTALL.read_text()
        self.assertIn('OC_MAJOR="${OC_SERVE_MAJOR:-${OPENCODE_MAJOR:-1}}"', text,
                      "the generated config's OC_MAJOR follows the binary in place")
        self.assertEqual(text.count('OPENCODE_MAJOR="${OC_SERVE_MAJOR:-1}" "$REPO_DIR/dashboard/install-agent.sh"'),
                         2, "both install-agent.sh calls hand the served major down (the unit)")
        self.assertIn('OC_SERVE_MAJOR="${OPENCODE_MAJOR:-1}"', text)


class TheRollbackSpeaks(unittest.TestCase):
    """./install.sh back on 1.x takes OUR v2 "permissions" out; when the array is not ours it says
    so plainly: OpenCode 1.19 refuses to start while the config holds "permissions", and a silent
    refusal left the Agent tab dead with no reason on screen (independent review, finding 3)."""

    def run_rollback(self, doc, was2=1):
        t = pathlib.Path(tempfile.mkdtemp(prefix="oc-rollback-"))
        self.addCleanup(shutil.rmtree, t, True)
        home = t / "home"
        (home / ".config" / "opencode").mkdir(parents=True)
        (home / ".config" / "qwen38").mkdir(parents=True)
        (home / ".config" / "opencode" / "opencode.json").write_text(json.dumps(doc))
        script = ("set -euo pipefail\ndie(){ echo \"DIE: $*\"; exit 1; }\n"
                  f'REPO_DIR="{REPO}"\nCONFIG_DIR="{home}/.config/qwen38"\nOPENCODE=1\nOC_SERVE_MAJOR=1\n'
                  f'OC_WAS_2="{was2}"\n' + rollback())
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                           env={"HOME": str(home), "PATH": "/usr/bin:/bin"})
        left = json.loads((home / ".config" / "opencode" / "opencode.json").read_text())
        return r.returncode, r.stdout + r.stderr, left

    def test_our_array_is_taken_out_in_silence(self):
        ours = {"permissions": [{"action": "shell", "resource": "*" + "aliyuncs" + ".com*", "effect": "deny"},
                                {"action": "shell", "resource": "*169.254.169.254*", "effect": "deny"},
                                {"action": "shell", "resource": "*100.100.100.200*", "effect": "deny"}]}
        rc, out, left = self.run_rollback(ours)
        self.assertEqual(rc, 0, out)
        self.assertNotIn("permissions", left)
        self.assertNotIn("NOTE", out)

    def test_a_foreign_array_is_left_and_said_with_the_way_back(self):
        foreign = {"permissions": [{"action": "shell", "resource": "*", "effect": "allow"}]}
        rc, out, left = self.run_rollback(foreign)
        self.assertEqual(rc, 0, out)
        self.assertIn("permissions", left, "the merger of the box only removes the array this repo wrote")
        self.assertIn("NOTE", out)
        self.assertIn("opencode.json", out)
        self.assertIn("permissions", out)

    def test_a_host_that_never_chose_2_is_not_even_asked(self):
        # No marker at the start of the run (OC_WAS_2=0): a "permissions" table of the user's
        # own is not the repo's, and the removal does not even try, so no NOTE on every
        # install. Same guard as the compaction rollback below (blind review 2026-11-20, D4).
        foreign = {"permissions": [{"action": "*", "resource": "*", "effect": "allow"}], "theme": "dark"}
        rc, out, left = self.run_rollback(foreign, was2=0)
        self.assertEqual(rc, 0, out)
        self.assertEqual(left, foreign, "nothing written, nothing said")
        self.assertNotIn("NOTE", out)


class TheRollbackTakesTheV2CompactionKeysOut(unittest.TestCase):
    """./install.sh --opencode 1 on a box that was in 2 (the marker said 2 at the start of the
    run) takes exactly the keys the v2 compaction block wrote (keep, background) out of BOTH
    configs the v2 touched, the generated one and the user's global one, leaving the v1 keys
    and everything else. A box that never chose 2 is not touched at all (no write, no backup)."""

    GEN = {"compaction": {"keep": {"tokens": 135000}, "background": True},
           "provider": {"qwen38": {}}, "autoupdate": False}
    USER = {"compaction": {"preserve_recent_tokens": 135000, "prune": False, "auto": True,
                           "background": True, "keep": {"tokens": 135000}},
            "instructions": ["~/.claude/CLAUDE.md"], "theme": "dark"}

    def run_rollback(self, *, was2, major="1", gen=None, user=None):
        t = pathlib.Path(tempfile.mkdtemp(prefix="oc-rb2-"))
        self.addCleanup(shutil.rmtree, t, True)
        home = t / "home"
        cfg = home / ".config" / "qwen38"
        (home / ".config" / "opencode").mkdir(parents=True)
        cfg.mkdir(parents=True)
        (cfg / "opencode.json").write_text(json.dumps(gen if gen is not None else self.GEN, indent=2) + "\n")
        (home / ".config" / "opencode" / "opencode.json").write_text(
            json.dumps(user if user is not None else self.USER, indent=2) + "\n")
        if was2:
            (cfg / "opencode.major").write_text("2\n# asked with ./install.sh --opencode 2\n")
        script = ("set -euo pipefail\ndie(){ echo \"DIE: $*\"; exit 1; }\n"
                  f'REPO_DIR="{REPO}"\nCONFIG_DIR="{cfg}"\nOPENCODE=1\nOPENCODE_MAJOR="{major}"\n'
                  f'OC_SERVE_MAJOR="{major}"\n'
                  f'OC_WAS_2="{1 if was2 else 0}"\n' + rollback())
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                           env={"HOME": str(home), "PATH": "/usr/bin:/bin"})
        gen_raw = (cfg / "opencode.json").read_bytes()
        user_raw = (home / ".config" / "opencode" / "opencode.json").read_bytes()
        gen_left = json.loads(gen_raw)
        user_left = json.loads(user_raw)
        return (r.returncode, r.stdout + r.stderr, gen_left, user_left, gen_raw, user_raw,
                sorted(p.name for p in cfg.iterdir()),
                sorted(p.name for p in (home / ".config" / "opencode").iterdir()))

    def test_the_marker_and_the_flag_resolve_the_major(self):
        # Since v1.56.0 (decision 1546) OpenCode 2 is the default: no flag and no marker means
        # 2 (that is also what the first update of a pre-v1.56.0 1.x box gets), a marker "1"
        # keeps the box on 1.x across updates, and an explicit flag always wins.
        t = pathlib.Path(tempfile.mkdtemp(prefix="oc-mj-"))
        self.addCleanup(shutil.rmtree, t, True)
        cfg = t / "cfg"
        cfg.mkdir()

        def resolve(marker, want):
            if marker is None:
                (cfg / "opencode.major").unlink(missing_ok=True)
            else:
                (cfg / "opencode.major").write_text(marker)
            script = ("set -euo pipefail\n"
                      f'CONFIG_DIR="{cfg}"\nOPENCODE_MAJOR_WANT="{want}"\n' + major_choice()
                      + '\necho "MAJOR=${OPENCODE_MAJOR} WAS2=${OC_WAS_2}"\n')
            r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30,
                               env={"PATH": "/usr/bin:/bin"})
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            return r.stdout.strip().splitlines()[-1]

        self.assertEqual(resolve("2\n", ""), "MAJOR=2 WAS2=1")
        self.assertEqual(resolve("2\n", "1"), "MAJOR=1 WAS2=1")
        self.assertEqual(resolve("2\n", "2"), "MAJOR=2 WAS2=1")
        self.assertEqual(resolve("1\n", ""), "MAJOR=1 WAS2=0")
        self.assertEqual(resolve("1\n", "2"), "MAJOR=2 WAS2=0")
        self.assertEqual(resolve(None, ""), "MAJOR=2 WAS2=0")
        self.assertEqual(resolve(None, "1"), "MAJOR=1 WAS2=0")

    def test_a_box_that_was_in_2_gets_exactly_the_v2_keys_out_of_both_configs(self):
        rc, out, gen_left, user_left, gen_raw, user_raw, gen_files, user_files = self.run_rollback(was2=True)
        self.assertEqual(rc, 0, out)
        self.assertEqual(gen_left["compaction"], {}, gen_left["compaction"])
        self.assertEqual(gen_left["provider"], {"qwen38": {}})
        self.assertIs(gen_left["autoupdate"], False)
        self.assertEqual(user_left["compaction"],
                         {"preserve_recent_tokens": 135000, "prune": False, "auto": True},
                         user_left["compaction"])
        self.assertEqual(user_left["instructions"], ["~/.claude/CLAUDE.md"])
        self.assertEqual(user_left["theme"], "dark")

    def test_a_box_that_never_was_in_2_is_not_touched(self):
        gen = json.dumps(self.GEN, indent=2) + "\n"
        user = json.dumps(self.USER, indent=2) + "\n"
        rc, out, gen_left, user_left, gen_raw, user_raw, gen_files, user_files = self.run_rollback(
            was2=False, gen=self.GEN, user=self.USER)
        self.assertEqual(rc, 0, out)
        self.assertEqual(gen_raw, gen.encode(), "the generated config was written")
        self.assertEqual(user_raw, user.encode(), "the user's config was written")
        self.assertEqual(gen_files, ["opencode.json"], f"a backup appeared: {gen_files}")
        self.assertEqual(user_files, ["opencode.json"], f"a backup appeared: {user_files}")

    def test_a_box_staying_in_2_is_not_touched(self):
        gen = json.dumps(self.GEN, indent=2) + "\n"
        user = json.dumps(self.USER, indent=2) + "\n"
        rc, out, gen_left, user_left, gen_raw, user_raw, gen_files, user_files = \
            self.run_rollback(was2=True, major="2")
        self.assertEqual(rc, 0, out)
        self.assertEqual(gen_raw, gen.encode())
        self.assertEqual(user_raw, user.encode())
        self.assertEqual(gen_files, ["opencode.json", "opencode.major"])
        self.assertEqual(user_files, ["opencode.json"])


class TheConfigStopsSelfUpdating(unittest.TestCase):
    """autoupdate false: with any other value opencode 1.19 offers OpenCode 2 on its TUI's
    home screen once it is on the newest 1.x (1.18.35, as soon as GitHub's latest release is
    a 2.x), and OpenCode 2's server is not the `opencode serve` the Agent tab relays."""

    def merge(self, cfg, value="false"):
        return subprocess.run(["python3", str(REPO / "oc-merge-limits.py"), str(cfg), "--autoupdate", value],
                              capture_output=True, text=True)

    def config(self, text):
        t = pathlib.Path(tempfile.mkdtemp(prefix="oc-au-"))
        self.addCleanup(shutil.rmtree, t, True)
        (t / "opencode.json").write_text(text)
        return t / "opencode.json"

    def test_the_generated_config_and_the_users_get_false(self):
        text = INSTALL.read_text()
        self.assertIn('doc["autoupdate"] = False', text)
        self.assertIn('oc-merge-limits.py" "$OC_USER_CFG" --autoupdate false', text)
        self.assertNotIn("--autoupdate notify", text)

    def test_a_box_on_notify_gets_the_boolean_and_keeps_the_users_comments(self):
        cfg = self.config('{\n  // mine\n  "model": "x",\n  "autoupdate": "notify"\n}\n')
        r = self.merge(cfg)
        self.assertEqual(r.returncode, 0, r.stdout)
        text = cfg.read_text()
        self.assertIn("// mine", text)
        doc = json.loads("".join(ln for ln in text.splitlines(True) if "//" not in ln))
        self.assertIs(doc["autoupdate"], False, 'the string "false" is a config opencode refuses')
        self.assertIn("unchanged", self.merge(cfg).stdout)

    def test_a_value_opencode_does_not_take_is_refused_and_nothing_written(self):
        cfg = self.config('{"autoupdate": "notify"}\n')
        for bad in ("False", "true", "0", ""):
            r = self.merge(cfg, bad)
            self.assertEqual(r.returncode, 2, (bad, r.stdout))
            self.assertEqual(cfg.read_text(), '{"autoupdate": "notify"}\n', bad)
        self.assertEqual([p.name for p in cfg.parent.iterdir()], ["opencode.json"], "no backup either")

    def test_notify_still_keeps_a_users_false(self):
        cfg = self.config('{"autoupdate": false}\n')
        r = self.merge(cfg, "notify")
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertEqual(cfg.read_text(), '{"autoupdate": false}\n')



class Scratch:
    """A temporary directory per call, removed with the test that asked for it."""
    def scratch(self, prefix):
        d = tempfile.mkdtemp(prefix=prefix)
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        return d


class TheCockpitNamesThePin(Scratch, unittest.TestCase):
    """The Agent tab says when the served opencode is not the version install.sh pins,
    and reads that version out of install.sh, so the two can never name different ones."""

    def test_the_cockpit_reads_the_installers_pin(self):
        code = ("import importlib.util, os, sys\n"
                "t = sys.argv[2]; open(t + '/api-key', 'w').write('k')\n"
                "os.environ.update(COCKPIT_DRY_RUN='1', COCKPIT_CONFIG_DIR=t, COCKPIT_PORT='0',\n"
                "                  COCKPIT_AGENT_PORT='0', COCKPIT_REPO_DIR=sys.argv[1])\n"
                "sys.path.insert(0, sys.argv[1] + '/dashboard')\n"
                "spec = importlib.util.spec_from_file_location('ck', sys.argv[1] + '/dashboard/cockpit.py')\n"
                "ck = importlib.util.module_from_spec(spec); spec.loader.exec_module(ck)\n"
                "print(ck.opencode_pinned())\n")
        cfg, home = self.scratch("oc-ck-cfg-"), self.scratch("oc-ck-")
        r = subprocess.run(["python3", "-c", code, str(REPO), cfg], capture_output=True, text=True,
                           timeout=60, env={**os.environ, "HOME": home})
        self.assertEqual(r.stdout.strip().splitlines()[-1], pinned_default(), r.stderr[-400:])

    def test_the_agent_tab_says_when_it_serves_another_version(self):
        js = (REPO / "dashboard" / "static" / "js" / "agent.js").read_text()
        self.assertIn("if (d.pinned && sv.version && sv.version !== d.pinned)", js)
        self.assertIn("./install.sh brings it in line", js)


class AMissingOpencodeIsSaidPlainly(Scratch, unittest.TestCase):
    """A first-time user with no opencode must read what is missing and what to run, in
    the installer's last lines, from the oc launcher and in the cockpit's Agent tab."""

    def test_the_oc_launcher_says_so_instead_of_failing_on_a_path(self):
        text = INSTALL.read_text()
        start = text.index('cat > "$OC_BIN" <<OCWRAP\n')
        body = text[start:text.index("\nOCWRAP\n", start) + len("\nOCWRAP\n")]
        t = pathlib.Path(tempfile.mkdtemp(prefix="oc-render-"))
        (t / "home").mkdir()
        render = (f'OC_BIN="{t}/oc"; OC_OUT_CAP=200000; REPO_DIR=/opt/repo; OPENCODE_VERSION={PINNED}; '
                  f'CONFIG_DIR="{t}/cfg"\n' + body)
        subprocess.run(["bash", "-c", render], check=True)
        r = subprocess.run(["bash", str(t / "oc"), "--version"], capture_output=True, text=True,
                           env={"HOME": str(t / "home"), "PATH": "/usr/bin:/bin"})
        self.assertEqual(r.returncode, 127)
        self.assertIn("oc: opencode is not installed", r.stderr)
        self.assertIn(f"Re-run ./install.sh in /opt/repo: it installs the opencode this repo tests ({PINNED})", r.stderr)

    def test_the_v2_launcher_ends_with_auto_and_carries_no_dead_cap(self):
        # ./install.sh --opencode 2 writes its own launcher: --yolo does not exist in
        # OpenCode 2 and OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX is read by nobody there
        # (both measured on 2.3.0 the 2026-11-14), and --auto goes last like --yolo did
        text = INSTALL.read_text()
        start = text.index('cat > "$OC_BIN" <<OCWRAP2')
        body = text[start:text.index("\nOCWRAP2\n", start) + len("\nOCWRAP2\n")]
        t = pathlib.Path(tempfile.mkdtemp(prefix="oc-render2-"))
        (t / "home").mkdir()
        (t / "bin").mkdir()
        calls = t / "calls.log"
        (t / "bin" / "opencode").write_text(
            '#!/bin/sh\nif [ "$1" = --version ]; then echo "opencode v2.3.0"; exit 0; fi\n'
            f'printf \'%s\\n\' "$@" >> "{calls}"\n')
        (t / "bin" / "opencode").chmod(0o755)
        render = (f'OC_BIN="{t}/oc"; REPO_DIR=/opt/repo; OPENCODE2_VERSION=2.3.0; '
                  f'OPENCODE_MAJOR=2; CONFIG_DIR="{t}/cfg"\n' + body)
        subprocess.run(["bash", "-c", render], check=True)
        r = subprocess.run(["bash", str(t / "oc"), "run", "hi"], capture_output=True, text=True,
                           env={"HOME": str(t / "home"), "PATH": f"{t}/bin:/usr/bin:/bin"})
        self.assertEqual(r.returncode, 0, r.stderr)
        sent = calls.read_text().strip().splitlines()
        self.assertEqual(sent, ["run", "hi", "--auto"], "--auto last, no --yolo, no env cap")
        self.assertNotIn("OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX", body)
        self.assertIn("--yolo is not a flag of OpenCode 2", body)

    def test_the_v2_launcher_skips_claude_md_and_the_v1_one_does_not(self):
        # OpenCode 2 reads CLAUDE.md (project dirs without AGENTS.md, and the global
        # ~/.claude/CLAUDE.md) into every prompt since 2.3.1; measured on 2.3.2 in a sandbox,
        # both files land in the system message of each request. On a box that is the
        # owner's private file leaving in every prompt, so the v2 launcher exports the
        # documented switch (read at server start, server-process.ts of 2.3.2); "0" is not
        # truthy there, so a user can bring them back. The v1 launcher is unchanged.
        text = INSTALL.read_text()

        def render(body, fname, version_var, version, major=None):
            t = pathlib.Path(tempfile.mkdtemp(prefix=f"oc-{fname}-"))
            (t / "home").mkdir()
            (t / "bin").mkdir()
            calls = t / "calls.log"
            (t / "bin" / "opencode").write_text(
                '#!/bin/sh\nif [ "$1" = --version ]; then echo "opencode v2.3.0"; exit 0; fi\n'
                f'env | grep -E "^OPENCODE_DISABLE_CLAUDE_CODE_PROMPT=" >> "{calls}" || echo ABSENT >> "{calls}"\n')
            (t / "bin" / "opencode").chmod(0o755)
            pre = (f'OC_BIN="{t}/oc"; OC_OUT_CAP=200000; REPO_DIR=/opt/repo; {version_var}={version}; '
                   f'CONFIG_DIR="{t}/cfg"\n')
            if major:
                pre += f"OPENCODE_MAJOR={major}; "
            subprocess.run(["bash", "-c", pre + body], check=True)
            return t, calls

        start = text.index('cat > "$OC_BIN" <<OCWRAP2')
        body2 = text[start:text.index("\nOCWRAP2\n", start) + len("\nOCWRAP2\n")]
        t, calls = render(body2, "v2", "OPENCODE2_VERSION", "2.3.2", major=2)
        base_env = {"HOME": str(t / "home"), "PATH": f"{t}/bin:/usr/bin:/bin"}
        r = subprocess.run(["bash", str(t / "oc"), "run", "hi"], capture_output=True, text=True, env=base_env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("OPENCODE_DISABLE_CLAUDE_CODE_PROMPT=1", calls.read_text(),
                      "the v2 launcher cuts CLAUDE.md from every prompt")
        r = subprocess.run(["bash", str(t / "oc"), "run", "hi"], capture_output=True, text=True,
                           env={**base_env, "OPENCODE_DISABLE_CLAUDE_CODE_PROMPT": "0"})
        self.assertIn("OPENCODE_DISABLE_CLAUDE_CODE_PROMPT=0", calls.read_text(),
                      "the launcher's default yields to an explicit value")

        start = text.index('cat > "$OC_BIN" <<OCWRAP\n')
        body1 = text[start:text.index("\nOCWRAP\n", start) + len("\nOCWRAP\n")]
        t1, calls1 = render(body1, "v1", "OPENCODE_VERSION", PINNED)
        r = subprocess.run(["bash", str(t1 / "oc"), "run", "hi"], capture_output=True, text=True,
                           env={"HOME": str(t1 / "home"), "PATH": f"{t1}/bin:/usr/bin:/bin"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ABSENT", calls1.read_text())
        self.assertNotIn("OPENCODE_DISABLE_CLAUDE_CODE_PROMPT", calls1.read_text(),
                         "the v1 launcher is unchanged")

    def test_the_v2_launcher_says_when_the_binary_is_not_opencode_2(self):
        text = INSTALL.read_text()
        start = text.index('cat > "$OC_BIN" <<OCWRAP2')
        body = text[start:text.index("\nOCWRAP2\n", start) + len("\nOCWRAP2\n")]
        t = pathlib.Path(tempfile.mkdtemp(prefix="oc-render2b-"))
        (t / "home").mkdir()
        (t / "bin").mkdir()
        (t / "bin" / "opencode").write_text("#!/bin/sh\n[ \"$1\" = --version ] && echo 1.19.5\nexit 0\n")
        (t / "bin" / "opencode").chmod(0o755)
        render = (f'OC_BIN="{t}/oc"; REPO_DIR=/opt/repo; OPENCODE2_VERSION=2.3.0; '
                  f'OPENCODE_MAJOR=2; CONFIG_DIR="{t}/cfg"\n' + body)
        subprocess.run(["bash", "-c", render], check=True)
        r = subprocess.run(["bash", str(t / "oc"), "--version"], capture_output=True, text=True,
                           env={"HOME": str(t / "home"), "PATH": f"{t}/bin:/usr/bin:/bin"})
        self.assertIn("is not OpenCode 2", r.stderr)

    def summary(self, launcher, on_path, opencode=True):
        """The installer's last opencode line, rendered in a throwaway HOME."""
        text = INSTALL.read_text()
        start = text.index('      OC_NOW="$( { opencode --version')
        end = text.index("\n      fi\n", text.index("(this repo tests ${OC_TARGET_VER:-$OPENCODE_VERSION}); start it with", start)) + 9
        t = pathlib.Path(tempfile.mkdtemp(prefix="oc-summary-"))
        (t / "bin").mkdir()
        (t / ".local" / "bin").mkdir(parents=True)
        if opencode:
            (t / "bin" / "opencode").write_text(f"#!/bin/sh\necho {PINNED}\n")
            (t / "bin" / "opencode").chmod(0o755)
        if launcher:
            (t / ".local" / "bin" / "oc").write_text("#!/bin/bash\n# oc launcher installed by dgx-spark-qwen38\n")
            (t / ".local" / "bin" / "oc").chmod(0o755)
        path = f"{t}/bin:" + (f"{t}/.local/bin:" if on_path else "") + "/usr/bin:/bin"
        prelude = f'OPENCODE_VERSION={PINNED}; OC_OUT_CAP=200000; CONFIG_DIR="{t}/.config/qwen38"\n'
        r = subprocess.run(["bash", "-c", "set -euo pipefail\n" + prelude + text[start:end]],
                           capture_output=True, text=True, env={"HOME": str(t), "PATH": path})
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.replace(str(t), "~")

    def summary_full(self, *, start_major="", serve_major="1", oc_version=PINNED, launcher=True):
        """The installer's whole last opencode block (the served line plus the migration
        message), rendered in a throwaway HOME."""
        text = INSTALL.read_text()
        start = text.index('      OC_NOW="$( { opencode --version')
        end = text.index("    else\n      # remembered, unlike a lane", start)
        t = pathlib.Path(tempfile.mkdtemp(prefix="oc-sumfull-"))
        (t / "bin").mkdir()
        (t / ".local" / "bin").mkdir(parents=True)
        (t / "bin" / "opencode").write_text(f"#!/bin/sh\necho {oc_version}\n")
        (t / "bin" / "opencode").chmod(0o755)
        if launcher:
            (t / ".local" / "bin" / "oc").write_text("#!/bin/bash\n# oc launcher installed by dgx-spark-qwen38\n")
            (t / ".local" / "bin" / "oc").chmod(0o755)
        path = f"{t}/bin:{t}/.local/bin:/usr/bin:/bin"
        prelude = (f'OPENCODE_VERSION={PINNED}; OC_OUT_CAP=200000; CONFIG_DIR="{t}/.config/qwen38"\n'
                   f'OC_START_MAJOR="{start_major}"; OC_SERVE_MAJOR="{serve_major}"\n')
        r = subprocess.run(["bash", "-c", "set -euo pipefail\n" + prelude + text[start:end]],
                           capture_output=True, text=True, env={"HOME": str(t), "PATH": path})
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.replace(str(t), "~")

    def test_a_1_x_box_moved_to_2_is_told_with_the_way_back(self):
        out = self.summary_full(start_major="1", serve_major="2", oc_version="2.3.2")
        self.assertIn("moved to 2.3.2; to go back: ./install.sh --opencode 1", out)

    def test_no_migration_message_on_a_fresh_box_or_a_box_staying_put(self):
        self.assertNotIn("moved to", self.summary_full(start_major="", serve_major="2", oc_version="2.3.2"))
        self.assertNotIn("moved to", self.summary_full(start_major="2", serve_major="2", oc_version="2.3.2"))
        self.assertNotIn("moved to", self.summary_full(start_major="1", serve_major="1", oc_version=PINNED))

    def test_the_last_lines_name_a_command_the_users_shell_can_run(self):
        self.assertIn("the version this repo tests; start it with: oc   (config: ~/.config/opencode/opencode.json)",
                      self.summary(launcher=True, on_path=True))
        # ~/.local/bin created by this install is not in the shell that ran it
        self.assertIn("start it with: ~/.local/bin/oc   (", self.summary(launcher=True, on_path=False))
        # another program owns "oc": the launcher was not installed, so "oc" would run it
        self.assertIn("start it with: OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=200000 opencode --yolo   (",
                      self.summary(launcher=False, on_path=True))

    def test_the_last_lines_say_when_opencode_is_not_there(self):
        out = self.summary(launcher=True, on_path=True, opencode=False)
        self.assertIn("opencode   : NOT installed (see step 7). Its config is ready at "
                      "~/.config/opencode/opencode.json;", out)
        self.assertIn("re-run ./install.sh to install it", out)

    def test_the_cockpit_knows_a_box_with_no_opencode(self):
        code = ("import importlib.util, os, sys\n"
                "t = sys.argv[2]; open(t + '/api-key', 'w').write('k')\n"
                "os.environ.update(COCKPIT_DRY_RUN='1', COCKPIT_CONFIG_DIR=t, COCKPIT_PORT='0',\n"
                "                  COCKPIT_AGENT_PORT='0', COCKPIT_REPO_DIR=sys.argv[1])\n"
                "sys.path.insert(0, sys.argv[1] + '/dashboard')\n"
                "spec = importlib.util.spec_from_file_location('ck', sys.argv[1] + '/dashboard/cockpit.py')\n"
                "ck = importlib.util.module_from_spec(spec); spec.loader.exec_module(ck)\n"
                "out = ck.collect_agent(); out = out.get('data', out)\n"
                "print(repr(out.get('enabled')), repr(out.get('opencode_found')), out.get('pinned'))\n")
        # the config directory comes from here and goes with the test: made inside a child
        # that has no TMPDIR, it was left in /tmp at every run (2026-10-03)
        cfg, home = self.scratch("oc-none-cfg-"), self.scratch("oc-none-")
        r = subprocess.run(["python3", "-c", code, str(REPO), cfg], capture_output=True, text=True,
                           timeout=60, env={"HOME": home, "PATH": "/usr/bin:/bin"})
        self.assertEqual(r.stdout.strip().splitlines()[-1], f"False None {pinned_default()}", r.stderr[-400:])

    def test_the_agent_tab_tells_no_opencode_from_no_tab(self):
        js = (REPO / "dashboard" / "static" / "js" / "agent.js").read_text()
        self.assertIn("if (d.opencode_found === null)", js)
        self.assertIn("opencode is not installed on this box, and this view runs it", js)
        self.assertIn("'cd ~/dgx-spark-qwen38 && ./install.sh'", js)


class TheBoxsModelIsWhatOcAndTheAgentTabUse(unittest.TestCase):
    """The reference box, 2026-09-23, after the published one-liner on an empty box: oc
    answered with "big-pickle" and the Agent tab's opencode listed one provider,
    opencode's own hosted one. Both run opencode, and opencode with no provider for
    this box picks its free cloud model: prompts meant for the local model left the
    box. Both now load the box's config through OPENCODE_CONFIG, which opencode reads
    over the global one (checked on 1.18.32: it wins even over a global "model")."""

    def launcher(self, t):
        text = INSTALL.read_text()
        start = text.index('cat > "$OC_BIN" <<OCWRAP\n')
        body = text[start:text.index("\nOCWRAP\n", start) + len("\nOCWRAP\n")]
        (t / "bin").mkdir()
        (t / "bin" / "opencode").write_text('#!/bin/sh\necho "OPENCODE_CONFIG=${OPENCODE_CONFIG:-unset}"\n')
        (t / "bin" / "opencode").chmod(0o755)
        subprocess.run(["bash", "-c", f'OC_BIN="{t}/oc"; OC_OUT_CAP=200000; REPO_DIR=/opt/repo; '
                        f'OPENCODE_VERSION={PINNED}; CONFIG_DIR="{t}/cfg"\n' + body], check=True)

    def oc(self, t, **env):
        r = subprocess.run(["bash", str(t / "oc"), "run", "x"], capture_output=True, text=True,
                           env={"HOME": str(t), "PATH": f"{t}/bin:/usr/bin:/bin", **env})
        return r.stdout.strip()

    def test_oc_loads_the_boxs_config_over_the_users(self):
        t = pathlib.Path(tempfile.mkdtemp(prefix="oc-cfg-"))
        self.launcher(t)
        (t / "cfg").mkdir()
        (t / "cfg" / "opencode.json").write_text("{}\n")
        self.assertEqual(self.oc(t), f"OPENCODE_CONFIG={t}/cfg/opencode.json")
        # one the user set themselves is theirs
        self.assertEqual(self.oc(t, OPENCODE_CONFIG="/mine.json"), "OPENCODE_CONFIG=/mine.json")

    def test_with_no_generated_config_oc_points_at_nothing(self):
        t = pathlib.Path(tempfile.mkdtemp(prefix="oc-cfg-"))
        self.launcher(t)
        self.assertEqual(self.oc(t), "OPENCODE_CONFIG=unset")

    def test_the_agent_tab_loads_the_boxs_config(self):
        unit = (REPO / "dashboard" / "opencode-web.service.template").read_text()
        self.assertIn("\nEnvironment=OPENCODE_CONFIG=__HOME__/.config/qwen38/opencode.json\n", unit)


class AFreshBoxGetsAConfigOpencodeCanUse(unittest.TestCase):
    """On a box with no opencode config, nothing pointed opencode at the one install.sh
    generates: the published v1.18.3 installed opencode and printed a cp command, and a
    brand-new user's opencode answered "Provider not found: qwen38". The installer now
    installs its config when the box has none, and still never overwrites one."""

    def block(self):
        text = INSTALL.read_text()
        start = text.index('OC_USER_CFG="$HOME/.config/opencode/opencode.json"\n# A box with no opencode config')
        end = text.index("\nfi\n", text.index("block from $CONFIG_DIR/opencode.json into it", start)) + 4
        return text[start:end]

    def run_it(self, existing=None,
               generated='{"provider": {"qwen38": {}}, "model": "qwen38/qwen3.8-27b"}\n'):
        home = pathlib.Path(tempfile.mkdtemp(prefix="oc-cfg-"))
        cfg_dir = home / ".config" / "qwen38"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "opencode.json").write_text(generated)
        user = home / ".config" / "opencode" / "opencode.json"
        if existing is not None:
            user.parent.mkdir(parents=True)
            user.write_text(existing)
        prelude = f'CONFIG_DIR="{cfg_dir}"; REPO_DIR="{REPO}"\n'
        r = subprocess.run(["bash", "-c", "set -euo pipefail\n" + prelude + self.block()],
                           capture_output=True, text=True, env={"HOME": str(home), "PATH": "/usr/bin:/bin"})
        return r.returncode, r.stdout + r.stderr, user

    def test_no_config_gets_this_repos(self):
        rc, out, user = self.run_it()
        self.assertEqual(rc, 0, out)
        self.assertIn('"qwen38"', user.read_text())
        self.assertIn("no opencode config yet: installed this repo's", out)

    def test_a_config_with_this_repos_provider_is_left_to_the_merges(self):
        mine = '{\n  // mine\n  "provider": {"qwen38": {"x": 1}, "other": {}}\n}\n'
        rc, out, user = self.run_it(existing=mine)
        self.assertEqual(rc, 0, out)
        text = user.read_text()
        self.assertIn("// mine", text)
        self.assertEqual(json.loads(text.replace("// mine", ""))["provider"], {"qwen38": {"x": 1}, "other": {}})
        self.assertIn("already lists this box's providers: unchanged", out)

    def test_the_config_opencode_oc_and_the_agent_tab_read_gets_the_shell_rules(self):
        # Cohere's report (2026-10-26): Qwen3.8-Flash-Next PUT a file to an Alibaba Cloud bucket
        # from a shell it was given. opencode runs every shell command unasked by default, so
        # this box's config denies the hosts of the report: the copy carries the rules of the
        # generated config, and an existing config of this box gets them last.
        rules = {"bash*": {"*aliyuncs.com*": "deny", "*169.254.169.254*": "deny", "*100.100.100.200*": "deny"}}
        generated = json.dumps({"provider": {"qwen38": {}}, "model": "qwen38/qwen3.8-27b", "permission": rules})
        rc, out, user = self.run_it(generated=generated)
        self.assertEqual(rc, 0, out)
        self.assertEqual(json.loads(user.read_text())["permission"], rules)
        self.assertNotIn("hold already", out, "the copy needs no second pass")
        rc, out, user = self.run_it(existing='{\n  // mine\n  "provider": {"qwen38": {"x": 1}}\n}\n',
                                    generated=generated)
        self.assertEqual(rc, 0, out)
        self.assertEqual(json.loads(user.read_text().replace("// mine", ""))["permission"], rules)
        self.assertIn("put last in its permission", out)

    def test_a_users_own_config_is_told_the_shell_rules_and_not_rewritten(self):
        mine = '{"provider": {"anthropic": {}}, "permission": {"bash": "ask"}}\n'
        rc, out, user = self.run_it(existing=mine)
        self.assertEqual(rc, 0, out)
        self.assertEqual(user.read_text(), mine)
        self.assertIn("docs/agents-and-network.md", out)

    def test_a_lane_installed_later_brings_its_provider_into_that_config(self):
        # The 27B first: the copy lists qwen38 only. Then MODEL_CHOICE=flash ./install.sh,
        # whose generated config lists both. Before, the copy stayed as it was, and
        # opencode sent the 27B's limits and label to the flash lane.
        first = '{\n  "provider": {\n    "qwen38": {"models": {"qwen3.8-27b": {}}}\n  }\n}\n'
        flash = {"npm": "@ai-sdk/openai-compatible", "models": {"qwen3.8-flash-next": {"limit": {"context": 1}}}}
        both = json.dumps({"provider": {"qwen38": {"models": {"qwen3.8-27b": {}}}, "flashnext": flash}})
        rc, out, user = self.run_it(existing=first, generated=both)
        self.assertEqual(rc, 0, out)
        doc = json.loads(user.read_text())
        self.assertEqual(doc["provider"]["flashnext"], flash)
        self.assertEqual(doc["provider"]["qwen38"], {"models": {"qwen3.8-27b": {}}})
        self.assertIn("flashnext provider added to", out)

    def test_a_users_own_config_without_it_is_told_not_rewritten(self):
        mine = '{"provider": {"anthropic": {}}}\n'
        rc, out, user = self.run_it(existing=mine)
        self.assertEqual(rc, 0, out)
        self.assertEqual(user.read_text(), mine)
        self.assertIn("has no provider for this box", out)


class TheOpencodeMajorFlag(unittest.TestCase):
    """./install.sh --opencode 2|1 is the whole opt-in; the parser must consume its value
    (independent review of the v2 migration, finding 1: the for-loop's shift never did)."""

    def run_args(self, *args):
        script = ("set -euo pipefail\ndie(){ echo \"DIE: $*\"; exit 1; }\nOPENCODE_MAJOR_WANT=\"\"\n"
                  + argloop() + 'echo "WANT=$OPENCODE_MAJOR_WANT"\n')
        return subprocess.run(["bash", "-c", script, "install.sh", *args], capture_output=True,
                              text=True, timeout=30, env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent-home"})

    def test_the_two_word_form_reaches_the_choice(self):
        r = self.run_args("--opencode", "2")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn("Unknown flag", r.stdout + r.stderr)
        self.assertIn("WANT=2", r.stdout)

    def test_a_foreign_value_is_carried_to_the_validator_not_read_as_a_flag(self):
        r = self.run_args("--opencode", "3", "--recommended")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("WANT=3", r.stdout)     # ./install.sh itself refuses it further down (test_a_value...)

    def test_a_missing_value_is_refused_by_the_parser(self):
        r = self.run_args("--opencode")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--opencode takes 1 or 2", r.stdout + r.stderr)


class TheNoFlagRunIsTheV2Run(unittest.TestCase):
    """A ./install.sh without --opencode is the v2 run since v1.56.0 (decision 1546, which
    replaced decision 445: the v2 choice is the default now, not an option). The opencode
    block runs whole in a throwaway HOME against a fake curl that serves either asset by
    the version its URL names: no flag, no marker, no binary: CHOICE=2, the 2.x asset
    downloaded, the marker 2. TheDefaultIsTwo above pins the same facts per scenario."""

    def test_no_flag_is_the_v2_run(self):
        # (the per-scenario facts live in TheDefaultIsTwo; here the block answers as v2)
        t = pathlib.Path(tempfile.mkdtemp(prefix="oc-noflag-"))
        self.addCleanup(shutil.rmtree, t, True)
        home = t / "home"
        home.mkdir()
        (home / ".bashrc").write_text("# a user's bashrc\n")
        cfg = home / ".config" / "qwen38"
        bin_ = t / "fakebin"
        bin_.mkdir()
        (bin_ / "curl").write_text(TheDefaultIsTwo.FAKE_CURL_V)
        (bin_ / "curl").chmod(0o755)
        log = t / "calls.log"
        assets = {}
        for ver in (PINNED, TheDefaultIsTwo.V2):
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                body = f"#!/bin/sh\n[ \"$1\" = --version ] && echo {ver}\n".encode()
                info = tarfile.TarInfo("opencode"); info.size = len(body); info.mode = 0o755
                tf.addfile(info, io.BytesIO(body))
            p = t / f"asset-{ver}.tar.gz"
            p.write_bytes(buf.getvalue())
            assets[ver] = (p, hashlib.sha256(buf.getvalue()).hexdigest())
        script = ("set -euo pipefail\ndie(){ echo \"DIE: $*\"; exit 1; }\n"
                  f'REPO_DIR="{REPO}"\nCONFIG_DIR="{cfg}"\n'
                  f'OPENCODE_VERSION="{PINNED}"; OPENCODE_SHA256="{assets[PINNED][1]}"; OPENCODE_PIN="1"\n'
                  f'OPENCODE2_VERSION="{TheDefaultIsTwo.V2}"; OPENCODE2_SHA256="{assets[TheDefaultIsTwo.V2][1]}"\n'
                  '_ENV_OPENCODE_VERSION=""; _ENV_OPENCODE_SHA256=""\n'
                  'OPENCODE_MAJOR_WANT=""\n'
                  + major_choice() + "\n" + block(whole=True)
                  + '\necho "CHOICE=${OPENCODE_MAJOR:-1}"\n')
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                           env={"HOME": str(home), "PATH": f"{bin_}:/usr/bin:/bin",
                                "OC_LOG": str(log), "OC_TAR_V1": str(assets[PINNED][0]),
                                "OC_TAR_V2": str(assets[TheDefaultIsTwo.V2][0])})
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertIn("CHOICE=2", out, "without the flag the run serves OpenCode 2 since v1.56.0")
        self.assertTrue((cfg / "opencode.major").exists(), "the marker says 2")
        self.assertEqual((cfg / "opencode.major").read_text().splitlines()[0], "2")
        self.assertEqual(log.read_text().splitlines(),
                         [f"curl https://github.com/anomalyco/opencode/releases/download/v{TheDefaultIsTwo.V2}"
                          "/opencode-linux-arm64.tar.gz"])


if __name__ == "__main__":
    unittest.main()
