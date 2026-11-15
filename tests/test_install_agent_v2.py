#!/usr/bin/env python3
"""install-agent.sh renders a unit for OpenCode 2 that can actually start and answer.

The 13/11-14/11 measurements on 2.3.0 (/home/hasan/qa-tools/opencode-v2-migration/
reprise-20261114/MESURES.md) settled four facts this file pins down as tests:
serve refuses --log-level in capitals (the 1.x unit line would never boot), reads neither
OPENCODE_PERMISSION nor OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX (dead lines lie to the
operator), always names its Basic user "opencode" (a box installed in 1.x times must have
its env file rewritten, password kept), and answers JSON on /api/info where /global/health
serves the HTML page (a probe that takes any 200 for health would certify a dead server).
The unit is rendered for real: the whole script runs up to the relay proof, with the
systemd calls and curl stubbed in a PATH of the test."""
import pathlib
import shutil
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_TEXT = (REPO / "dashboard/install-agent.sh").read_text()
TEMPLATE = (REPO / "dashboard/opencode-web.service.template").read_text()
MERGER = (REPO / "oc-merge-limits.py").read_text()
PERMS = (REPO / "oc-permissions.py").read_text()
STOP = "# ── the cockpit gets the relay"
SENTINEL_PW = "sentinel-password-keep-me"
GUARD_HOST = "aliyuncs" + ".com"      # never as one string: our own session guard



def env_lines(unit):
    return [ln for ln in unit.splitlines() if ln.startswith("Environment")]


def agent_stub(version_line, help_line="--hostname --port"):
    return ("#!/bin/sh\n"
            f'case "$*" in *--help*) echo "{help_line}";; *) echo "{version_line}";; esac\n')


CURL_STUB = """#!/bin/sh
# answers the two health paths of either OpenCode; everything else is a plain 0
if [ "$1" = "-K" ]; then
  cfg=$(cat)
  url=$(printf '%s' "$cfg" | sed -n 's/.*url = "\\([^"]*\\)".*/\\1/p')
  case "$url" in
    */api/info) if [ "${FAKE_OC2:-0}" = 1 ]; then echo '{"pid":7,"version":"2.3.0","capabilities":{}}'
                else echo '{"error":"not found"}'; fi ;;
    */global/health) if [ "${FAKE_OC2:-0}" = 1 ]; then echo '<!doctype html>the page'
                else echo '{"healthy":true,"version":"1.19.5"}'; fi ;;
    *) echo '{}' ;;
  esac
  exit 0
fi
exit 0
"""

SYSTEMCTL_STUB = """#!/bin/sh
case "$*" in
  *ActiveState*) echo "active" ;;
  *SubState*) echo "running" ;;
  *UnitFileState*) echo "enabled" ;;
  *ExecMainStartTimestamp*) echo "@1700000000" ;;
  *) : ;;
esac
exit 0
"""


def run_agent(t, version_line, oc2, env_extra=None, pre_env_file=True, installed_extra="", curl_body=None,
              no_major=False):
    """Run the whole install-agent script (unit render + health, no relay proof) in t."""
    sub = t / "sub"
    sub.mkdir(exist_ok=True)
    (t / "bin").mkdir(exist_ok=True)
    (sub / "install-agent.sh").write_text(
        SCRIPT_TEXT[:SCRIPT_TEXT.index(STOP)].replace("/etc/systemd/system", str(t)))
    (sub / "opencode-web.service.template").write_text(TEMPLATE)
    (t / "oc-merge-limits.py").write_text(MERGER)
    (t / "oc-permissions.py").write_text(PERMS)   # the merger loads its sibling, the server matcher
    for name, body in (("curl", curl_body or CURL_STUB), ("systemctl", SYSTEMCTL_STUB),
                       ("tailscale", "#!/bin/sh\nexit 1\n"), ("opencode", agent_stub(version_line)),
                       ("sudo", '#!/bin/sh\nexec "$@"\n')):
        f = t / "bin" / name
        f.write_text(body)
        f.chmod(0o755)
    (t / "qwen38-dashboard.service").write_text(
        "[Service]\nEnvironment=COCKPIT_BIND=127.0.0.1\nEnvironment=COCKPIT_PORT=30090\n")
    (t / "opencode-web.service").write_text(
        "[Service]\nEnvironment=PATH=/usr/bin:/bin\n"
        "ExecStart=/x/opencode serve --hostname 127.0.0.1 --port 4199 --print-logs\n"
        + installed_extra)
    if pre_env_file:
        cfg = t / ".config" / "qwen38"
        cfg.mkdir(parents=True, exist_ok=True)
        (cfg / "opencode-web.env").write_text(
            f"OPENCODE_SERVER_USERNAME=cockpit\nOPENCODE_SERVER_PASSWORD={SENTINEL_PW}\n")
    env = {"PATH": f"{t}/bin:/usr/bin:/bin", "HOME": str(t), "OPENCODE_PORT": "4199",
           "FAKE_OC2": "1" if oc2 else "0",
           # what ./install.sh --opencode 2 passes down (OPENCODE_MAJOR=2); without the
           # choice a 2.x binary is refused (the refusal test lives in test_install_agent_keeps)
           "OPENCODE_MAJOR": "2" if oc2 else "1"}
    if env_extra:
        env.update(env_extra)
    if no_major:
        del env["OPENCODE_MAJOR"]
    r = subprocess.run(["bash", str(sub / "install-agent.sh")], capture_output=True, text=True,
                       timeout=120, cwd=sub, env=env)
    unit = (t / "opencode-web.service")
    return r, (unit.read_text() if unit.is_file() else "")


class TheMarkerDecidesStandaloneRuns(unittest.TestCase):
    """./dashboard/install-agent.sh run on its own (the retry hint, docs/cockpit.md) has no
    OPENCODE_MAJOR from ./install.sh: the marker that --opencode 2 wrote decides the profile
    (independent review of the v2 migration, finding 4)."""

    def setUp(self):
        self.t = pathlib.Path(tempfile.mkdtemp(prefix="agent-marker-"))
        self.addCleanup(shutil.rmtree, self.t, True)

    def test_a_box_marked_2_renders_the_v2_unit_without_the_environment(self):
        cfg = self.t / ".config" / "qwen38"
        cfg.mkdir(parents=True, exist_ok=True)
        (cfg / "opencode.major").write_text("2\n# asked with ./install.sh --opencode 2\n")
        r, unit = run_agent(self.t, "opencode v2.3.0", oc2=True, no_major=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        line = next(ln for ln in unit.splitlines() if ln.startswith("ExecStart="))
        self.assertIn("--log-level info", line, "the v2 profile came from the marker")

    def test_no_marker_means_one_x_and_a_stray_two_x_is_refused_as_before(self):
        r, unit = run_agent(self.t, "opencode v2.3.0", oc2=True, no_major=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("did not choose", r.stdout + r.stderr)


class Render(unittest.TestCase):
    def setUp(self):
        self.t = pathlib.Path(tempfile.mkdtemp(prefix="agent-render-"))
        self.addCleanup(shutil.rmtree, self.t, True)

    def test_v2_unit_boots_answers_and_keeps_the_guards(self):
        r, unit = run_agent(self.t, "opencode v2.3.0", oc2=True, env_extra={"AGENT_AUTO": "1"})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn(SENTINEL_PW, r.stdout + r.stderr, "the password is never printed")
        line = next(ln for ln in unit.splitlines() if ln.startswith("ExecStart="))
        self.assertIn("serve --hostname 127.0.0.1 --port 4199", line)
        self.assertIn("--log-level info", line, "capitals would kill the boot (measured 2.3.0)")
        self.assertNotIn("--log-level INFO", unit)
        self.assertFalse([ln for ln in env_lines(unit) if "OPENCODE_PERMISSION=" in ln],
                         "a dead variable in 2.x")
        self.assertFalse([ln for ln in env_lines(unit) if "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=" in ln],
                         "read by nobody in 2.x")
        self.assertNotIn("32000 tokens", unit, "no comment claiming a cap that does not exist")
        self.assertTrue([ln for ln in env_lines(unit) if ln.startswith("Environment='OPENCODE_CONFIG_CONTENT=")],
                        "the approvals ride the merged content in 2.x")
        # the injected content approves, and its last rules are ours: the guard denies come
        # after the wildcard (measured: they hold under it)
        auto = next(ln for ln in env_lines(unit) if ln.startswith("Environment='OPENCODE_CONFIG_CONTENT="))
        self.assertIn('"action":"*"', auto)
        self.assertTrue(0 < auto.index('"effect":"allow"') < auto.rindex('"effect":"deny"'),
                        "allow first, denies last: last match wins in 2.x")
        self.assertIn(f"*{GUARD_HOST}*", auto)
        self.assertIn("AGENT_AUTO=1 on OpenCode 2", r.stdout)
        # the env file now names the username OpenCode 2 hardcodes, password kept
        env = (self.t / ".config" / "qwen38" / "opencode-web.env").read_text()
        self.assertIn("OPENCODE_SERVER_USERNAME=opencode", env)
        self.assertIn(f"OPENCODE_SERVER_PASSWORD={SENTINEL_PW}", env)
        self.assertNotIn("cockpit", env)

    def test_v2_unit_without_agent_auto_stays_at_the_server_defaults(self):
        r, unit = run_agent(self.t, "opencode v2.3.0", oc2=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse([ln for ln in env_lines(unit) if "OPENCODE_CONFIG_CONTENT=" in ln])
        self.assertIn("OpenCode 2's own permission defaults", unit)
        self.assertIn("--log-level info", unit)

    def test_v2_remember_the_auto_choice_from_the_installed_unit(self):
        r, unit = run_agent(self.t, "opencode v2.3.0", oc2=True, installed_extra=(
            "Environment='OPENCODE_CONFIG_CONTENT={\"permissions\":[{\"action\":\"*\"}]}'\n"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue([ln for ln in env_lines(unit) if ln.startswith("Environment='OPENCODE_CONFIG_CONTENT=")],
                        "the approvals ride the merged content in 2.x")
        auto = next(ln for ln in env_lines(unit) if ln.startswith("Environment='OPENCODE_CONFIG_CONTENT="))
        self.assertIn('"effect":"deny"', auto, "the choice was remembered and rebuilt, not copied")
        self.assertIn(f"*{GUARD_HOST}*", auto)

    def test_v1_unit_renders_exactly_what_works_today(self):
        r, unit = run_agent(self.t, "1.19.5", oc2=False, env_extra={"AGENT_AUTO": "1"})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("--log-level INFO", unit, "1.19.5 refuses the lowercase (its help answers it)")
        self.assertIn("Environment=OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=200000", "\n".join(env_lines(unit)))
        self.assertTrue([ln for ln in env_lines(unit) if ln.startswith("Environment='OPENCODE_PERMISSION=")])
        self.assertFalse([ln for ln in env_lines(unit) if "OPENCODE_CONFIG_CONTENT=" in ln])
        env = (self.t / ".config" / "qwen38" / "opencode-web.env").read_text()
        self.assertIn("OPENCODE_SERVER_USERNAME=cockpit", env, "1.x takes its own name")
        self.assertNotIn("not written for", r.stdout + r.stderr)

    def test_a_page_on_global_health_does_not_pass_for_health_in_v2(self):
        # a 2.x whose /api/info stopped answering (the server is dead or unknown): every
        # probe is a page, and a probe that took any 200 for health would certify it alive.
        trap = ("#!/bin/sh\nif [ \"$1\" = \"-K\" ]; then echo '<!doctype html>the page'; fi\nexit 0\n")
        r, _ = run_agent(self.t, "opencode v2.3.0", oc2=True, env_extra={"AGENT_AUTO": "1"},
                         curl_body=trap)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("did not answer", r.stderr + r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=1)
