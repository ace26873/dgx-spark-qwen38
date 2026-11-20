#!/usr/bin/env python3
"""The cockpit's Agent panel on an OpenCode 2 box (the v2 migration, 2026-11-14 and 15).

auto_live and guard_ok are read from the DOCUMENTS THE RUNNING SERVER SERVES on /api/config (the
unit's lines say what its NEXT start will load; they differ until a restart, and the page says so),
the pinned version follows the major of the one it answers, and a panel that can read nothing says
null, never well. The probe table is the installer's own (oc-merge-limits.deny_probes), so a guard
that reads denied HERE and wrong THERE cannot survive this file."""
import importlib.util
import os
import pathlib
import re
import shutil
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve()
DASH = HERE.parents[1]
REPO = HERE.parents[2]
# The CI runs this file as a script from dashboard/ (tests/run_suite.py: cwd=dashboard,
# python3 tests/<file>), so sys.path[0] is dashboard/tests, not dashboard, and cockpit.py's
# "import agent_relay" (cockpit.py:40) needs dashboard on the path. Every neighbor of this
# file inserts it (test_agent_relay.py, test_cockpit_collectors.py); this one did not, and
# the CI's invocation made all 8 tests error with ModuleNotFoundError (blind review, D1).
sys.path.insert(0, str(DASH))
INSTALL = REPO / "install.sh"
PIN_V1 = re.search(r'^OPENCODE_VERSION="\$\{OPENCODE_VERSION:-([0-9.]+)\}"', INSTALL.read_text(), re.M).group(1)
PIN_V2 = re.search(r'^OPENCODE2_VERSION="\$\{OPENCODE2_VERSION:-([0-9.]+)\}"', INSTALL.read_text(), re.M).group(1)
ENV_BEFORE = {}


def setUpModule():
    ENV_BEFORE.update(os.environ)


def tearDownModule():
    for name in set(os.environ) - set(ENV_BEFORE):
        del os.environ[name]
    os.environ.update(ENV_BEFORE)


def load(name, unit_text):
    """The cockpit, loaded on a throwaway box: this unit file, an env file with a password, dry run."""
    t = pathlib.Path(tempfile.mkdtemp(prefix="agent-panel-"))
    cfg = t / "cfg"
    cfg.mkdir()
    (cfg / "opencode-web.env").write_text("OPENCODE_SERVER_USERNAME=opencode\nOPENCODE_SERVER_PASSWORD=not-the-box-key\n")
    unit = t / "opencode-web.service"
    unit.write_text(unit_text)
    os.environ.update(HOME=str(t), COCKPIT_DRY_RUN="1", COCKPIT_CONFIG_DIR=str(cfg), COCKPIT_UNIT_DIR=str(t),
                      COCKPIT_REPO_DIR=str(REPO), COCKPIT_PORT="30090", COCKPIT_AGENT_PORT="30091",
                      COCKPIT_AGENT_UNIT_PATH=str(unit), COCKPIT_AUTOHEAL="0", COCKPIT_UPDATE_CHECK="0")
    spec = importlib.util.spec_from_file_location(name, DASH / "cockpit.py")
    cp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cp)
    return cp


UNIT_V2 = ("[Service]\nEnvironment=OPENCODE_CONFIG_CONTENT=[{\"action\":\"*\",\"resource\":\"*\",\"effect\":\"allow\"}]\n"
           "ExecStart=/opt/opencode serve --hostname 127.0.0.1 --port 4199 --print-logs --log-level info\n")
UNIT_V1 = ("[Service]\nEnvironment=OPENCODE_PERMISSION={\"*\":\"allow\"}\n"
           "ExecStart=/opt/opencode serve --hostname 127.0.0.1 --port 4199 --print-logs --log-level INFO\n")
UNIT_PLAIN = "[Service]\nExecStart=/opt/opencode serve --hostname 127.0.0.1 --port 4199\n"


def docs(sneaky=None):
    """What /api/config lists on a v2 box that took our content: the files, then the injected document
    without a path, our denies last; sneaky rides after them, as a project's own config would."""
    ours = [{"action": "shell", "resource": "*" + "aliyuncs" + ".com*", "effect": "deny"},
            {"action": "shell", "resource": "*169.254.169.254*", "effect": "deny"},
            {"action": "shell", "resource": "*100.100.100.200*", "effect": "deny"}]
    content = [{"action": "*", "resource": "*", "effect": "allow"}] + ours
    if sneaky:
        content += sneaky
    return [{"type": "document", "path": "/u/opencode.json",
             "info": {"permissions": [{"action": "*", "resource": "*", "effect": "allow"}]}},
            {"type": "document", "info": {"permissions": content}}]


def wire(cp, server, served):
    """The two network seams of the panel: the health probe and the JSON routes (the relay's own
    reading of them is the relay's tests; here it is what the panel DECIDES from them)."""
    cp.run = lambda argv, **kw: ("ActiveState=active\nSubState=running\nUnitFileState=enabled\n"
                                 "ExecMainStartTimestamp=@1700000000\n" if argv[0] == "systemctl" else "")
    cp.agent_config = lambda: None
    cp.ar.health = lambda cfg, timeout=3.0: dict(server)
    cp.ar.upstream_json = lambda cfg, path: served.get(path)


class TheV2Panel(unittest.TestCase):
    def test_the_served_documents_answer_auto_and_the_guards(self):
        cp = load("panel_v2_ok", UNIT_V2)
        wire(cp, {"healthy": True, "version": PIN_V2}, {"/api/config": docs()})
        out = cp.collect_agent()
        self.assertEqual((out["auto"], out["auto_live"], out["guard_ok"]), (True, True, True))
        self.assertEqual(out["pinned"], PIN_V2, "the panel names the 2.x pin beside a 2.x server")

    def test_a_later_allow_that_the_server_would_honour_is_said_not_in_force(self):
        cp = load("panel_v2_hijack", UNIT_V2)
        # "?" is a one-character wildcard for the server (measured in its bundle 2026-11-15): this allow
        # re-opens what our deny closed, and the panel must not read it as safe
        wire(cp, {"healthy": True, "version": "2.3.0"},
             {"/api/config": docs(sneaky=[{"action": "shell", "resource": "*aliyuncs.co?*", "effect": "allow"}])})
        out = cp.collect_agent()
        self.assertEqual(out["auto_live"], True)
        self.assertIs(out["guard_ok"], False, "the guard line says NOT in force")

    def test_documents_that_cannot_be_read_say_nothing(self):
        for name, served in (("no answer", {}), ("not documents", {"/api/config": "surprise"})):
            with self.subTest(name):
                cp = load("panel_v2_" + name.split()[0], UNIT_V2)
                wire(cp, {"healthy": True, "version": "2.3.0"}, served)
                out = cp.collect_agent()
                self.assertIsNone(out["auto_live"])
                self.assertIsNone(out["guard_ok"], "null, never True: nothing proved the guards")

    def test_a_stopped_server_leaves_the_live_fields_null(self):
        cp = load("panel_v2_down", UNIT_V2)
        wire(cp, {"healthy": False, "error": "credentials refused"}, {})
        out = cp.collect_agent()
        self.assertIsNone(out["auto_live"])
        self.assertIsNone(out.get("guard_ok"))
        self.assertEqual(out["auto"], True, "the unit still says what its NEXT start will load")

    def test_the_unit_and_the_served_mode_are_said_apart(self):
        # a unit edited since the last start (AGENT_AUTO off, the server still merging the old content):
        # the page compares these two and asks for the restart; both must be told
        cp = load("panel_v2_drift", UNIT_PLAIN)
        wire(cp, {"healthy": True, "version": "2.3.0"}, {"/api/config": docs()})
        out = cp.collect_agent()
        self.assertEqual((out["auto"], out["auto_live"]), (False, True))


class TheV1PanelStillWorks(unittest.TestCase):
    def test_the_1_x_branch_reads_its_own_route_and_names_its_own_pin(self):
        cp = load("panel_v1", UNIT_V1)
        wire(cp, {"healthy": True, "version": PIN_V1}, {"/config": {"permission": {"*": "allow"}}})
        out = cp.collect_agent()
        self.assertEqual(out["auto_live"], True)
        self.assertIsNone(out.get("guard_ok"), "the v2 guard check is not done on 1.x")
        self.assertEqual(out["pinned"], PIN_V1)

    def test_the_version_line_is_read_as_opencode_answers_it(self):
        # 2.x prints "opencode v2.3.0" where 1.x prints the number alone (measured F2): compared raw,
        # the panel would say "restart to serve it" forever
        cp = load("panel_ver", UNIT_V2)
        cp.run = lambda argv, **kw: "opencode v2.3.0\n" if argv[0] != "systemctl" else (
            "ActiveState=active\nSubState=running\nUnitFileState=enabled\nExecMainStartTimestamp=@1\n")
        t = pathlib.Path(tempfile.mkdtemp(prefix="agent-panel-bin-"))
        self.addCleanup(shutil.rmtree, t, True)
        b = t / "opencode"
        b.write_text("#!/bin/sh\necho 'opencode v2.3.0'\n")
        b.chmod(0o755)
        cp.agent_config = lambda: None
        cp.ar.health = lambda cfg, timeout=3.0: {"healthy": False, "error": "down"}
        cp.ar.upstream_json = lambda cfg, path: None
        # point the unit's ExecStart at the real stub so agent_binary_version can run it
        cp.AGENT_UNIT_PATH.write_text(UNIT_V2.replace("/opt/opencode", str(b)))
        cp.AGENT_BINARY.update(key=None, version=None)
        self.assertEqual(cp.agent_binary_version(), "2.3.0")


if __name__ == "__main__":
    unittest.main()
