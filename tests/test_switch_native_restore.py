#!/usr/bin/env python3
"""A switch on a native 27B box restores a target's config that still carries YaRN.

A native server crashes at load on a YaRN-patched config.json (measured 2026-09-11). A
27B target served while the box was in 1m mode keeps that patch in the shared cache when
the box goes back to native, since install.sh restores the target it installs and not the
others. switch-model.sh skipped the patch in native mode and restored nothing, so a switch
to such a target queued an engine that could not start (found in review, 2026-09-24).
The switch's steps after the fetch now live in lanes.sh's patch_served_checkpoint, called by
the switch with the unit in place and by prepare-lane.sh with the unit that waits; the
function runs here with patch-yarn.py and patch-template.py replaced by stubs that record
calls.
"""
import pathlib
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANES = (REPO / "lanes.sh").read_text()
FN = LANES[LANES.index("patch_served_checkpoint(){"):LANES.index("\n}\n", LANES.index("patch_served_checkpoint(){")) + 3]


def run(lane, choice, context_length, restore_rc=0):
    d = pathlib.Path(tempfile.mkdtemp(prefix="sw-restore-"))
    (d / "unit").write_text(f"ExecStart=... --context-length {context_length} ...\n")
    (d / "patch-yarn.py").write_text(
        "import sys\nopen(sys.argv[0] + '.log', 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
        f"sys.exit({restore_rc} if '--restore' in sys.argv else 0)\n")
    (d / "patch-template.py").write_text(
        "import sys\nopen(sys.argv[0] + '.log', 'a').write(' '.join(sys.argv[1:]) + '\\n')\n")
    script = ('set -euo pipefail\ndie(){ echo "DIE: $*"; exit 1; }\n'
              f'_lanes_dir(){{ echo "{d}"; }}\nCONFIG_DIR="{d}"\n' + FN +
              f'patch_served_checkpoint {lane} {choice} "{d}/unit" /hf org/uncensored abc\necho REACHED\n')
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
    log = d / "patch-yarn.py.log"
    tlog = d / "patch-template.py.log"
    return (r.stdout + r.stderr,
            log.read_text() if log.exists() else "",
            tlog.read_text() if tlog.exists() else "")


class ANativeSwitchRestores(unittest.TestCase):
    def test_the_target_is_restored_on_a_native_unit(self):
        out, calls, _ = run("27b", "stock", 262144)
        self.assertIn("--restore /hf org/uncensored abc", calls, out)
        self.assertIn("REACHED", out)

    def test_a_restore_that_fails_stops_the_switch_before_the_unit(self):
        out, _, _ = run("27b", "stock", 262144, restore_rc=1)
        self.assertIn("DIE:", out)
        self.assertNotIn("REACHED", out)

    def test_a_1m_unit_still_gets_the_patch(self):
        out, calls, _ = run("27b", "stock", 1010000)
        self.assertIn("/hf org/uncensored abc", calls)
        self.assertNotIn("--restore", calls)

    def test_the_template_is_regenerated_from_the_targets_snapshot(self):
        out, _, tpl = run("27b", "stock", 1010000)
        self.assertIn("/hf", tpl, out)
        self.assertIn("abc org/uncensored", tpl)
        self.assertIn("chat-template-sglang.jinja", tpl)

    def test_the_flash_lane_gets_no_yarn_call_and_its_own_template(self):
        out, calls, tpl = run("flash", "flash", 262144)
        self.assertEqual(calls, "", out)
        self.assertIn("chat-template-flashnext.jinja", tpl, out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
