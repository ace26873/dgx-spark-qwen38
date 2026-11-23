#!/usr/bin/env python3
"""The speech lane in the cockpit (v1.58.0, decision 1564: what the repo ships is in sight for everyone).

It is no lane of the GPU's turn: the cockpit reads its unit's state and its launcher, starts and stops the unit,
and sets whether it starts with the box. Nothing here reaches the box: the cockpit is loaded on a throwaway
systemd folder and config folder, in dry run, and the page's scripts run in node on the page's own markup."""
import importlib.util
import os
import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve()
DASH = HERE.parents[1]
REPO = HERE.parents[2]
sys.path.insert(0, str(DASH))
sys.path.insert(0, str(HERE.parent))
import pagejs  # noqa: E402

ENV_BEFORE = {}
UNIT = "qwen38-speech.service"
# the launcher as install-speech.sh writes it: the recognizer's own --host and --port first, the routes' after
LAUNCHER = """#!/bin/bash
# --port 1 in a comment is no flag
taskset -c 0,1 llama-server -m asr.gguf \\
  --host 127.0.0.1 --port 30125 \\
  --threads 5 &
exec python speech-server.py \\
  --host 127.0.0.1 --port 30124 \\
  --asr-url http://127.0.0.1:30125 --asr-name qwen3-asr-0.6b --asr-ctx 8192 \\
  --threads 5 --default-voice af_heart &
"""


def setUpModule():
    ENV_BEFORE.update(os.environ)


def tearDownModule():
    for name in set(os.environ) - set(ENV_BEFORE):
        del os.environ[name]
    os.environ.update(ENV_BEFORE)


def load(name, installed=True, launcher=LAUNCHER):
    t = pathlib.Path(tempfile.mkdtemp(prefix="speech-card-"))
    cfg, units = t / "cfg", t / "systemd"
    cfg.mkdir()
    units.mkdir()
    if installed:
        (units / UNIT).write_text("[Service]\nExecStart=/bin/true\n")
        if launcher is not None:
            (cfg / "launch-speech.sh").write_text(launcher)
    os.environ.update(HOME=str(t), COCKPIT_DRY_RUN="1", COCKPIT_CONFIG_DIR=str(cfg), COCKPIT_UNIT_DIR=str(units),
                      COCKPIT_SYSTEMD_DIR=str(units), COCKPIT_REPO_DIR=str(REPO), COCKPIT_PORT="30090",
                      COCKPIT_AUTOHEAL="0", COCKPIT_UPDATE_CHECK="0")
    spec = importlib.util.spec_from_file_location(name, DASH / "cockpit.py")
    cp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cp)
    return cp


def systemd_says(cp, **fields):
    """The cockpit's run() for systemctl show, answering these properties, and recording what was asked."""
    asked = []

    def run(argv, **kw):
        asked.append(argv)
        if argv[:2] == ["systemctl", "show"]:
            return "".join(f"{k}={v}\n" for k, v in fields.items())
        return ""
    cp.run = run
    return asked


class TheCollector(unittest.TestCase):
    def test_a_box_without_the_lane_says_so_and_asks_systemd_nothing(self):
        cp = load("speech_absent", installed=False)
        asked = systemd_says(cp)
        out = cp.collect_speech()
        self.assertIs(out["installed"], False)
        self.assertEqual(out["unit"], UNIT)
        self.assertEqual(asked, [])

    def test_the_state_is_systemds_and_the_facts_are_the_launchers(self):
        cp = load("speech_running")
        asked = systemd_says(cp, ActiveState="active", SubState="running", UnitFileState="enabled",
                             ExecMainStartTimestamp="Mon 2026-11-23 03:31:59 CET")
        out = cp.collect_speech()
        self.assertEqual(asked[0][:3], ["systemctl", "show", UNIT])
        self.assertEqual((out["installed"], out["active"], out["boot"]), (True, "active", True))
        # the ROUTES' port and address: the launcher's last --port, not the recognizer's before it nor a comment's
        self.assertEqual((out["host"], out["port"]), ("127.0.0.1", 30124))
        self.assertEqual((out["recognizer"], out["voice"]), ("qwen3-asr-0.6b", "af_heart"))

    def test_stopped_and_not_with_the_box(self):
        cp = load("speech_stopped")
        systemd_says(cp, ActiveState="inactive", SubState="dead", UnitFileState="disabled")
        out = cp.collect_speech()
        self.assertEqual((out["active"], out["boot"]), ("inactive", False))

    def test_a_launcher_that_cannot_be_read_leaves_the_defaults(self):
        cp = load("speech_no_launcher", launcher=None)
        systemd_says(cp, ActiveState="inactive", SubState="dead", UnitFileState="disabled")
        out = cp.collect_speech()
        self.assertEqual((out["host"], out["port"], out["recognizer"], out["voice"]), ("127.0.0.1", 30024, "", ""))

    def test_the_launcher_is_read_once_a_round(self):
        """Four reads and four scans of one file every five seconds, on the thread the units, the feed and the
        agent share, for values that move when the installer runs (review of v1.58.0)."""
        cp = load("speech_once")
        systemd_says(cp, ActiveState="active", UnitFileState="enabled")
        reads = []
        real = cp.Path.read_text

        def counting(self, *a, **k):
            if self.name == "launch-speech.sh":
                reads.append(1)
            return real(self, *a, **k)
        cp.Path.read_text = counting
        try:
            cp.collect_speech()
        finally:
            cp.Path.read_text = real
        self.assertEqual(len(reads), 1)

    def test_its_journal_is_in_the_diagnostics_bundle(self):
        src = (DASH / "cockpit.py").read_text()
        self.assertIn('(tdp / "journal-speech.txt").write_text(run(["journalctl", "-u", SPEECH_UNIT', src)

    def test_it_is_served_to_the_page(self):
        cp = load("speech_tier")
        self.assertIn("speech", {name for _, group in cp.TIERS for name in group})


class TheActions(unittest.TestCase):
    def test_the_unit_is_started_and_stopped_like_any_and_is_no_engine(self):
        cp = load("speech_actions")
        self.assertNotIn(UNIT, cp.SERVING_UNITS, "no update of it waits for a Load")
        self.assertNotIn(UNIT, cp.UNITS, "no restart is owed to it by an engine's inputs")
        self.assertNotIn(UNIT, cp.lc.ENGINE_UNITS, "it takes no turn on the GPU")
        self.assertIn(UNIT, cp.JOURNAL_UNITS, "the Logs view reads its journal")
        for verb in ("start", "stop", "restart"):
            code, out = cp.start_action("unit", {"verb": verb, "unit": UNIT})
            self.assertEqual(code, 202, out)
            self.assertEqual(out["argv"], ["sudo", "-n", "/usr/bin/systemctl", verb, UNIT])
            self.assertTrue(out["dry_run"])
            cp.JOBS[out["job"]].done.wait(10) if hasattr(cp.JOBS[out["job"]], "done") else None
            self.wait_idle(cp)

    def wait_idle(self, cp):
        import time
        for _ in range(100):
            if cp.JOB_LOCK.acquire(blocking=False):
                cp.JOB_LOCK.release()
                return
            time.sleep(0.05)
        self.fail("the job did not end")

    def test_starting_it_is_never_blocked_by_an_engine_that_serves(self):
        cp = load("speech_beside")
        with cp.LIFE_LOCK:
            cp.LIFE["states"] = {"qwen38-sglang.service": "ready", "qwen38-flash.service": "loading"}
        code, out = cp.start_action("unit", {"verb": "start", "unit": UNIT})
        self.assertEqual(code, 202, out)
        self.assertEqual(out["warnings"], [])
        self.wait_idle(cp)

    def test_with_the_box_or_not(self):
        cp = load("speech_boot")
        for verb in ("enable", "disable"):
            code, out = cp.start_action("speech_boot", {"verb": verb})
            self.assertEqual(code, 202, out)
            self.assertEqual(out["argv"], ["sudo", "-n", "/usr/bin/systemctl", verb, UNIT])
            self.wait_idle(cp)
        code, out = cp.start_action("speech_boot", {"verb": "mask"})
        self.assertEqual((code, out), (400, {"error": "invalid verb"}))
        self.assertIn("speech lane with the box", cp.job_phrase("speech_boot", {"verb": "enable"}))

    def test_every_argv_has_its_sudoers_line(self):
        lines = (DASH / "sudoers-cockpit.template").read_text().splitlines()
        for verb in ("start", "stop", "restart", "enable", "disable"):
            self.assertIn(f"__USER__ ALL=(root) NOPASSWD: /usr/bin/systemctl {verb} {UNIT}", lines)


PAGE = r"""
const wrap = o => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, {data: v, ts: Date.now() / 1000}]));
const feed = o => apply(wrap(o));
const CONFIG = {version: '1.58.0', dry_run: false, usable_frac: 0.92, periods: {}};
const speech = o => Object.assign({unit: 'qwen38-speech.service', installed: true, active: 'inactive', sub: 'dead', boot: false,
                                   since: '', port: 30024, host: '127.0.0.1', recognizer: 'qwen3-asr-0.6b', voice: 'af_heart'}, o || {});
const card = () => { const b = $('speech-box'); const btn = t => [...b.querySelectorAll('button')].find(x => x.textContent === t);
  return {text: b.textContent, state: b.querySelector('.cap').textContent, btn,
          shown: [...b.querySelectorAll('button')].filter(x => !x.hidden).map(x => x.textContent)}; };
"""


def page(test, body):
    return pagejs.run(test, PAGE + body, scripts=pagejs.SCRIPTS)


class TheCard(unittest.TestCase):
    def test_stopped_it_offers_start_and_says_where_it_listens(self):
        r = page(self, r"""
        feed({config: CONFIG, speech: speech()});
        const c = card(); report({state: c.state, shown: c.shown, text: c.text});
        """)
        self.assertEqual(r["state"], "stopped")
        self.assertEqual(r["shown"], ["Start Speech", "Start with the box", "Journal"])
        self.assertIn("http://127.0.0.1:30024/v1", r["text"])
        self.assertIn("this box only, no key", r["text"])
        self.assertIn("qwen3-asr-0.6b", r["text"])
        self.assertIn("manual start", r["text"])
        self.assertIn("about 2.5 GB once running", r["text"])

    def test_running_it_offers_stop_and_the_other_boot_choice(self):
        r = page(self, r"""
        feed({config: CONFIG, speech: speech({active: 'active', sub: 'running', boot: true})});
        const c = card(); report({state: c.state, shown: c.shown, text: c.text});
        """)
        self.assertEqual(r["state"], "running")
        self.assertEqual(r["shown"], ["Stop Speech", "Do not start with the box", "Journal"])
        self.assertIn("starts", r["text"])

    def test_no_fact_is_longer_than_its_column_and_the_memory_is_the_recognizers(self):
        """The first capture of the card in a browser (2026-11-23) showed three facts cut with an ellipsis, and
        2.5 GB said of a box whose 1.7B recognizer holds 4.2 (docs/speech-lane.md)."""
        r = page(self, r"""
        feed({config: CONFIG, speech: speech({active: 'active', recognizer: 'qwen3-asr-1.7b', host: '192.168.100.200', port: 30024})});
        const dd = [...$('speech-box').querySelectorAll('dl.facts dd')].map(e => e.textContent);
        report({dd, text: card().text});
        """)
        self.assertIn("about 4.2 GB once running", r["text"])
        self.assertTrue(r["dd"], "the facts are read")
        self.assertLessEqual(max(len(x) for x in r["dd"]), 36, r["dd"])

    def test_not_installed_it_says_how_and_offers_nothing(self):
        r = page(self, r"""
        feed({config: CONFIG, speech: speech({installed: false})});
        const c = card(); const opt = $('log-opt-speech');
        report({state: c.state, shown: c.shown, text: c.text, opt: [opt.disabled, opt.title]});
        """)
        self.assertEqual(r["state"], "not installed")
        self.assertEqual(r["shown"], [])
        self.assertIn("./install-speech.sh", r["text"])
        self.assertEqual(r["opt"], [True, "The speech lane is not installed on this box"])

    def test_a_bind_of_its_own_is_said_as_it_is(self):
        r = page(self, r"""
        feed({config: CONFIG, speech: speech({host: '0.0.0.0'})});
        report({text: card().text});
        """)
        self.assertIn("0.0.0.0, no key", r["text"])
        self.assertNotIn("this box only", r["text"])

    def test_the_buttons_ask_the_right_action_with_the_command_in_sight(self):
        r = page(self, r"""
        feed({config: CONFIG, speech: speech()});
        card().btn('Start Speech').click();
        const start = {title: $('sh-title').textContent, cmd: $('sh-cmd').textContent, lede: $('sh-lede').textContent};
        $('sh-cancel').click();
        card().btn('Start with the box').click();
        const boot = {title: $('sh-title').textContent, cmd: $('sh-cmd').textContent};
        $('sh-cancel').click();
        feed({config: CONFIG, speech: speech({active: 'active', boot: true})});
        card().btn('Stop Speech').click();
        const stop = {title: $('sh-title').textContent, cmd: $('sh-cmd').textContent, warns: $('sh-warns').textContent};
        $('sh-cancel').click();
        card().btn('Do not start with the box').click();
        const unboot = {cmd: $('sh-cmd').textContent};
        $('sh-cancel').click();
        const opt = $('log-opt-speech');
        // the sheet's button and the phrases the toast and the dock say (review of v1.58.0: "speech_boot verb enable")
        const phrases = [actionPhrase('speech_boot', {verb: 'enable'}), actionPhrase('speech_boot', {verb: 'disable'}),
                         VERB.speech_boot({verb: 'enable'}), VERB.speech_boot({verb: 'disable'})];
        // Journal opens the speech lane's own journal, with the 27B serving (it opened the 27B's container log)
        feed({config: CONFIG, speech: speech({active: 'active'}),
              lifecycle: {engines: {'qwen38-sglang.service': {state: 'ready', target: 'stock', elapsed: 60}}, events: [], blocked: {}}});
        card().btn('Journal').click();
        const want = $('log-src').value;
        report({start, boot, stop, unboot, opt: opt.disabled, phrases, want});
        """)
        self.assertEqual(r["start"]["title"], "Start the speech lane")
        self.assertEqual(r["start"]["cmd"], f"sudo -n /usr/bin/systemctl start {UNIT}")
        self.assertIn("beside the lane that serves", r["start"]["lede"])
        self.assertEqual(r["boot"]["cmd"], f"sudo -n /usr/bin/systemctl enable {UNIT}")
        self.assertEqual(r["boot"]["title"], "Start the speech lane with the box")
        self.assertEqual(r["stop"]["cmd"], f"sudo -n /usr/bin/systemctl stop {UNIT}")
        self.assertIn("is lost", r["stop"]["warns"])
        self.assertEqual(r["unboot"]["cmd"], f"sudo -n /usr/bin/systemctl disable {UNIT}")
        self.assertIs(r["opt"], False, "its journal can be read")
        self.assertEqual(r["phrases"], ["start the speech lane with the box", "no longer start the speech lane with the box",
                                        "Start it with the box", "Stop starting it with the box"])
        self.assertEqual(r["want"], UNIT)


if __name__ == "__main__":
    unittest.main()
