#!/usr/bin/env python3
"""The OpenCode 2 port of the auto-continue plugin relaunches only what a relaunch can fix.

Ported from extras/opencode/auto-continue.js to extras/opencode/auto-continue-v2.js for the
v2 plugin API (2.3.2 source: packages/core/src/plugin/module.ts accepts a default export of
{ id, setup }; ctx.event.subscribe yields { type, data } payloads; ctx.session.prompt takes a
flat { sessionID, text }). The v2 event set has no session.idle/error/compacted: a failed
execution leaves the session idle (projector.ts), so session.execution.failed is the signal,
session.execution.started cancels a pending relaunch, and session.compaction.ended only arms
one for reason "manual" (an auto compaction resumes inside the running turn, runner/llm.ts).
These load the plugin as written under node with a fake v2 context and feed it scripted steps.
"""
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
PLUGIN = REPO / "extras" / "opencode" / "auto-continue-v2.js"

# One scripted run: steps are {"event": {...}} or {"wait": ms}, in order; then the prompts the
# plugin sent back (its relaunches are timer-armed, hence the waits).
DRIVER = r"""
const plugin = (await import(process.env.PLUGIN_URL)).default;
const sent = [];
const steps = JSON.parse(process.env.STEPS);
const ctx = {
  session: { prompt: async (req) => { sent.push(req); } },
  event: { subscribe: async function* () {
    for (const s of steps) {
      if (s.wait) await new Promise((r) => setTimeout(r, s.wait));
      else yield s.event;
    }
  } },
};
const cleanup = await plugin.setup(ctx);
const total = steps.reduce((a, s) => a + (s.wait || 0), 0) + 300;
await new Promise((r) => setTimeout(r, total));
if (typeof cleanup === "function") cleanup();
console.log(JSON.stringify(sent));
process.exit(0);
"""

SID = "ses_test0000003"
STARTED = {"type": "session.execution.started", "data": {"sessionID": SID}}
SUCCEEDED = {"type": "session.execution.succeeded", "data": {"sessionID": SID}}
W = {"wait": 250}


def failed(message, type_="provider.transport", status=None):
    err = {"type": type_, "message": message}
    if status is not None:
        err["status"] = status
    return {"event": {"type": "session.execution.failed", "data": {"sessionID": SID, "error": err}}}


def interrupted(reason):
    return {"event": {"type": "session.execution.interrupted", "data": {"sessionID": SID, "reason": reason}}}


def compaction_ended(reason):
    return {"event": {"type": "session.compaction.ended", "data": {"sessionID": SID, "reason": reason}}}


def ev(e):
    return {"event": e}


def E(e):
    return {"event": e}


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class TheV2AutoContinuePlugin(unittest.TestCase):
    def setUp(self):
        self.t = pathlib.Path(tempfile.mkdtemp(prefix="auto-continue-v2-"))

    def tearDown(self):
        shutil.rmtree(self.t, ignore_errors=True)

    def run_steps(self, steps, extra_env=None):
        env = dict(os.environ, PLUGIN_URL=PLUGIN.as_uri(), AC_LOG=str(self.t / "ac.log"),
                   AC_IDLE_DELAY_MS="0", AC_THROTTLE_MS="0", STEPS=json.dumps(steps))
        if extra_env:
            env.update(extra_env)
        r = subprocess.run(["node", "--input-type=module", "-e", DRIVER], env=env,
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout.strip().splitlines()[-1])

    # what a relaunch fixes: the runner's retries are exhausted, the execution failed
    def test_a_failed_execution_is_relaunched(self):
        sent = self.run_steps([failed("keepalive-proxy: upstream http://127.0.0.1:39000 unreachable "
                                      "([Errno 111] Connection refused)"), W])
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["sessionID"], SID)
        self.assertIn("interrupted by a transient technical error", sent[0]["text"])

    def test_the_relaunch_uses_the_flat_v2_prompt_input(self):
        sent = self.run_steps([failed("tool call delta without id", type_="provider.invalid-output"), W])
        self.assertEqual(sorted(sent[0].keys()), ["sessionID", "text"],
                         "v2 prompt is { sessionID, text }, not v1's path/body/parts")

    # what it cannot fix
    def test_an_overflow_is_left_to_the_runner(self):
        self.assertEqual(self.run_steps([failed(
            "keepalive-proxy: the prompt is too long for this lane. This request is 420237 prompt tokens "
            "(counted by the engine); this lane serves at most 250000 prompt tokens"), W]), [])

    def test_a_400_refusal_the_third_time_in_a_row_stops_the_relaunches(self):
        # each relaunch starts a turn (started) that fails again with the same refusal
        ev400 = failed("JSON schema exceeds the maximum nesting depth of 10 levels (request 1)",
                       type_="provider.invalid-request", status=400)
        steps = [ev400, W, E(STARTED), ev400, W, E(STARTED), ev400, W, E(STARTED), W]
        self.assertEqual(len(self.run_steps(steps)), 2,
                         "the third identical refusal stops the loop (numbers masked)")

    def test_a_retryable_failure_keeps_the_whole_budget(self):
        ev = failed("keepalive-proxy: the engine restarted and its KV pool is not measured yet")
        self.assertEqual(len(self.run_steps([ev, W, E(STARTED)] * 5)), 5)

    # deliberate ends
    def test_a_user_interruption_is_not_relaunched(self):
        self.assertEqual(self.run_steps([interrupted("user"), W]), [])
        self.assertEqual(self.run_steps([interrupted("shutdown"), W]), [])

    # compaction: auto resumes by itself in v2, manual does not
    def test_an_auto_compaction_is_not_relaunched(self):
        self.assertEqual(self.run_steps([compaction_ended("auto"), W]), [],
                         "the runner continues the turn after an auto compaction (runner/llm.ts)")

    def test_a_manual_compaction_is_relaunched(self):
        sent = self.run_steps([compaction_ended("manual"), W])
        self.assertEqual(len(sent), 1)
        self.assertIn("just compacted", sent[0]["text"])

    # a turn that started meanwhile decides on its own
    def test_an_execution_started_after_the_error_cancels_the_relaunch(self):
        self.assertEqual(self.run_steps([failed("connection refused"), E(STARTED), W]), [])

    def test_progress_on_a_finished_turn_resets_the_consecutive_count(self):
        ev = failed("connection refused")
        steps = [ev, W, E(STARTED), ev, W, E(STARTED), E(SUCCEEDED), ev, W, E(STARTED), W]
        self.assertEqual(len(self.run_steps(steps, {"AC_MAX_CONSECUTIVE": "2"})), 3,
                         "a succeeded turn is progress: the cap of 2 was not hit")

    def test_the_plugin_exports_the_v2_definition_shape(self):
        # module.ts of 2.3.2: the default export must be { id: string, setup: function };
        # a v1-style function default export is a LoadError in v2
        driver = (r"""
const mod = await import(process.env.PLUGIN_URL);
const d = mod.default;
console.log(JSON.stringify({ isObject: typeof d === "object" && d !== null,
                              hasId: typeof d?.id === "string" && d.id.length > 0,
                              hasSetup: typeof d?.setup === "function" }));
""")
        env = dict(os.environ, PLUGIN_URL=PLUGIN.as_uri())
        r = subprocess.run(["node", "--input-type=module", "-e", driver], env=env,
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        shape = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertEqual(shape, {"isObject": True, "hasId": True, "hasSetup": True})

    def test_the_v1_file_is_untouched_and_still_a_function(self):
        v1 = (REPO / "extras" / "opencode" / "auto-continue.js").read_text()
        self.assertIn("export default plugin;", v1)
        self.assertNotIn("setup(ctx)", v1, "the v1 file keeps its own shape, identical")


if __name__ == "__main__":
    unittest.main(verbosity=1)
