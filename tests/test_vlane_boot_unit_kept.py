#!/usr/bin/env python3
"""A vLLM lane that starts at boot keeps its unit when the new one mounts a file not written yet.

The Ornith lane's unit mounts its chat template since v1.40.2. An update that found the lane's
checkpoint out of reach (its cache on a disk absent then) runs the lane's installer --units-only,
which writes no template, and put the new unit: docker refuses an absent source, so a box booting
that lane would not have served at its next boot (found in review, 2026-10-24). vllm-lane.sh's own
vlane_put_unit runs here against an installed unit, a new one, and a systemctl that says whether the
lane starts at boot."""
import os
import pathlib
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]


class TheUnitOfALaneThatStartsAtBoot(unittest.TestCase):
    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp(prefix="vlane-boot-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.d, True))
        (self.d / "cfg").mkdir()
        self.bin = self.d / "bin"
        self.bin.mkdir()
        (self.bin / "systemctl").write_text(
            '#!/bin/bash\ncase "$*" in *is-enabled*) [ "${ENABLED:-0}" = 1 ] ;; *) exit 0 ;; esac\n')
        # and a sudo that runs the stub: the real one reached the real systemctl through its secure_path,
        # and the cockpit's sudoers let each run reload the box's systemd (code review, 2026-10-24)
        (self.bin / "sudo").write_text('#!/bin/bash\necho "sudo $*" >> "$SUDO_LOG"\nexec "$@"\n')
        for stub in ("systemctl", "sudo"):
            self.bin.joinpath(stub).chmod(0o755)
        self.sudo_log = self.d / "sudo.log"
        self.installed = self.d / "installed.service"
        self.installed.write_text("[Service]\nExecStart=/bin/bash -c 'exec docker run old'\n")
        self.template = self.d / "cfg" / "chat-template.jinja"

    def put(self, enabled, new=None, pending_put=False):
        new_unit = self.d / "new.service"
        new_unit.write_text(new or ("[Service]\nExecStart=/bin/bash -c 'exec docker run \\\n"
                                    f"  --mount type=bind,source={self.template},target=/out/t.jinja,readonly \\\n"
                                    f"  --mount type=bind,source={self.d}/cfg,target=/out/cfg,readonly new'\n"))
        script = "\n".join([
            "set -euo pipefail", f'. "{REPO}/lanes.sh"' if pending_put else ":", f'. "{REPO}/vllm-lane.sh"',
            'die(){ echo "die: $*"; exit 1; }', 'unit_put(){ cp "$1" "$2"; echo installed; }',
            f'CONFIG_DIR="{self.d}/cfg"', f'INSTALLED="{self.installed}"', "UNIT=qwen38-x.service",
            "VLANE_SHORT=X", "SMOKE=1", f'vlane_put_unit "{new_unit}"'])
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30,
                           env=dict(os.environ, PATH=f"{self.bin}:/usr/bin:/bin", ENABLED="1" if enabled else "0",
                                    SUDO_LOG=str(self.sudo_log), PENDING_PUT="1" if pending_put else "0"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r.stdout, self.installed.read_text()

    def test_a_boot_lane_keeps_its_unit_when_the_new_one_mounts_a_missing_file(self):
        out, installed = self.put(enabled=True)
        self.assertIn(f"NOTE: X starts at boot, and its new unit mounts what is not there yet: {self.template}.", out)
        self.assertIn("docker run old", installed)

    def test_on_an_update_its_new_unit_waits_for_its_load_with_what_it_mounts(self):
        # nothing waited, an earlier run's wait stayed, and install.sh went on to update the lane whole (Vibe's blind
        # review of v1.52.0, Mistral Large 4); docker here only says it has no image
        (self.bin / "docker").write_text("#!/bin/bash\nexit 1\n")
        (self.bin / "docker").chmod(0o755)
        pend = self.d / "cfg" / "pending"
        pend.mkdir()
        (pend / "installed.service").write_text("older\n")
        (pend / "installed.service.needs").write_text("checkpoint old/one@1\n")
        out, installed = self.put(enabled=True, pending_put=True)
        self.assertIn("NOTE: X starts at boot", out)
        self.assertIn("docker run old", installed, "the unit in place serves what it served, at boot too")
        self.assertEqual((pend / "installed.service").read_text(), (self.d / "new.service").read_text())
        needs = (pend / "installed.service.needs").read_text().splitlines()
        self.assertIn(f"path {self.template}", needs)
        self.assertNotIn("checkpoint old/one@1", needs)

    def test_each_missing_mount_is_one_path_line_in_the_wait(self):
        # unit_pending_put writes the .needs from unit_needs, which already says each bind-mount source it
        # lacks as a path line: appending them again put every file twice on the card (sixth review of v1.52.0,
        # 2026-11-19, GLM 5.3)
        (self.bin / "docker").write_text("#!/bin/bash\nexit 1\n")
        (self.bin / "docker").chmod(0o755)
        out, installed = self.put(enabled=True, pending_put=True)
        needs = (self.d / "cfg" / "pending" / "installed.service.needs").read_text().splitlines()
        paths = [ln for ln in needs if ln.startswith("path ")]
        self.assertEqual(paths, [f"path {self.template}"], needs)

    def test_outside_an_update_nothing_is_written_to_wait(self):
        out, installed = self.put(enabled=True)
        self.assertIn("NOTE: X starts at boot", out)
        self.assertFalse((self.d / "cfg" / "pending").exists())

    def test_a_boot_lane_takes_the_new_unit_once_the_file_is_there(self):
        self.template.write_text("{{ messages }}\n")
        out, installed = self.put(enabled=True)
        self.assertNotIn("NOTE", out)
        self.assertIn(" new'", installed)

    def test_a_lane_that_does_not_start_at_boot_takes_the_new_unit(self):
        out, installed = self.put(enabled=False)
        self.assertNotIn("NOTE", out)
        self.assertIn(" new'", installed)
        # the reload went to the stubs, never to the box's systemd
        self.assertEqual(self.sudo_log.read_text(), "sudo systemctl daemon-reload\n")

    def test_an_unchanged_unit_is_not_held_back(self):
        out, installed = self.put(enabled=True, new=self.installed.read_text())
        self.assertNotIn("NOTE", out)

    def test_a_mount_source_with_a_space_is_one_missing_file(self):
        # a HOME with a space renders --mount type=bind,source=<spaced path>,...: one source, read whole,
        # not two halves both read missing (sixth review of v1.52.0, 2026-11-19, GLM 5.3)
        spaced = self.d / "my cfg"
        spaced.mkdir()
        tmpl = spaced / "chat-template.jinja"
        new = ("[Service]\nExecStart=/bin/bash -c 'exec docker run \\\n"
               f"  --mount type=bind,source={tmpl},target=/out/t.jinja,readonly new'\n")
        out, installed = self.put(enabled=True, new=new)
        self.assertIn(f"NOTE: X starts at boot, and its new unit mounts what is not there yet: {tmpl}.", out)
        self.assertIn("docker run old", installed)
        tmpl.write_text("{{ messages }}\n")
        out, installed = self.put(enabled=True, new=new)
        self.assertNotIn("NOTE", out, "the spaced source is one path: it exists now")
        self.assertIn(" new'", installed)


if __name__ == "__main__":
    unittest.main()
