// auto-continue, the OpenCode 2 port: automatically resumes an opencode session left
// stuck by a TECHNICAL failure (the runner's own retries exhausted, then
// session.execution.failed), or right after a MANUAL /compact (the turn does not
// resume on its own there), so a one-off incident no longer freezes a long run.
// The v1 file (auto-continue.js) stays for OpenCode 1.x; v2 does not run v1 plugin
// implementations (packages/core/src/plugin/module.ts of 2.3.2 requires a default
// export of { id, setup }), so this is a separate file: on a v2 box copy THIS one to
// ~/.config/opencode/plugins/ and take the v1 one out.
//
// What changed against v1, from the 2.3.2 source:
// - events: v2 has no session.idle / session.status / session.error / session.compacted.
//   session.execution.failed carries the error as { type, message, status, response }
//   under event.data (schema/session-error.ts, session-event.ts); a failed execution
//   leaves the session idle (projector.ts projects idle on succeeded/failed/interrupted),
//   so the failed event itself is the idle signal. session.execution.started clears any
//   pending relaunch (a turn is running and decides).
// - the stuck-after-AUTO-compaction case does not exist in v2: the runner compacts inside
//   the turn and continues (runner/llm.ts: after a completed auto compaction it rebuilds
//   the request and goes on), and compaction.background (default true since 2.3.1) has
//   the summary precomputed from the cached prefix, so the swap is instant. A manual
//   /compact still ends idle without resuming, so session.compaction.ended with
//   reason "manual" arms the relaunch; reason "auto" deliberately does nothing.
// - relaunch: ctx.session.prompt({ sessionID, text }) (v2 prompt input is flat, not
//   path/body/parts).
// - a 4xx refusal (not 429) is the v2 equivalent of v1's isRetryable === false: the
//   runner's retry policy only retries retryable causes, so a 400 comes straight back
//   as execution.failed, and the same-refusal cap below is what stops a loop.
//
// Log: ~/.config/qwen38/auto-continue.log (override with AC_LOG).
// Install (OpenCode 2 box only): cp auto-continue-v2.js ~/.config/opencode/plugins/
// Plugins load at opencode startup only: a new session picks it up, a running one never does.
import { appendFile } from "node:fs/promises";

const num = (v, d) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
};

const LOG = process.env.AC_LOG || process.env.HOME + "/.config/qwen38/auto-continue.log";
const THROTTLE_MS = num(process.env.AC_THROTTLE_MS, 5000);    // minimum between two relaunches of one session
const IDLE_DELAY_MS = num(process.env.AC_IDLE_DELAY_MS, 1500); // breather before relaunching
const MAX_CONSECUTIVE = num(process.env.AC_MAX_CONSECUTIVE, 25); // relaunches in a row without any progress
// A request refused as sent (a 4xx but 429, which opencode itself retries on schedule) that
// comes back the same this many times in a row will not go away: on the reference box a tool
// schema the provider refused came back 91 times in 28 minutes over 12 sessions (2026-10-13).
const SAME_REFUSAL_MAX = num(process.env.AC_SAME_REFUSAL_MAX, 3);
// never relaunch on these: a deliberate user action, or a problem that will not fix itself
const EXCLUDE = ["abort", "interrupt", "permission", "unauthorized", "auth", "payment", "quota",
  // A prompt past what the lane serves, or an image the engine cannot decode: a relaunch
  // sends the same history plus its own reminder, so it can only fail again, longer
  // (the reference box's log, 2026-09-09: 210,159 then 210,210 then 210,261 tokens against
  // a 200,000 ceiling). The v2 runner compacts on an overflow by itself, and a compaction
  // that does not resume is the compaction path's to relaunch. The last three are the
  // proxy's wording before its refusal read as an overflow to opencode.
  "contextoverflow", "context_too_long", "prompt is too long", "serves at most",
  "tokens by size", "imagedecodeerror", "could not be decoded"];

const RESUME_AFTER_ERROR = "<system-reminder>The previous turn was interrupted by a transient technical error, not by the user. Resume the task exactly where it stopped and continue through to completion. Do not apologize for or mention this interruption.</system-reminder>";
const RESUME_AFTER_COMPACTION = "<system-reminder>The context was just compacted and the turn did not resume on its own. Continue the task from the compaction summary, exactly where it left off, through to completion. Do not restart from scratch, and do not apologize for or mention this.</system-reminder>";

const state = new Map(); // sessionID -> {errPending, compactPending, lastPrompt, consecutive, busy, timer, lastRefusal, sameRefusal}

function getState(id) {
  let s = state.get(id);
  if (!s) {
    s = { errPending: false, compactPending: false, lastPrompt: 0, consecutive: 0,
          busy: false, timer: null, lastRefusal: "", sameRefusal: 0 };
    state.set(id, s);
  }
  return s;
}

async function log(msg) {
  try { await appendFile(LOG, new Date().toISOString().slice(11, 19) + " " + msg + "\n"); } catch {}
}

function errText(error) {
  if (!error || typeof error !== "object") return String(error ?? "");
  return `${error.type ?? ""}: ${error.message ?? JSON.stringify(error).slice(0, 200)}`;
}

// a 4xx other than 429 is a refusal the runner will not retry: the v2 stand-in for v1's
// error.data.isRetryable === false
function isRefusal(error) {
  const st = Number(error?.status);
  return Number.isInteger(st) && st >= 400 && st < 500 && st !== 429;
}

async function relaunch(ctx, sessionID, s, text, tag) {
  s.lastPrompt = Date.now();
  s.consecutive += 1;
  await new Promise((r) => setTimeout(r, IDLE_DELAY_MS));
  await log(`RELAUNCH ${tag} ${s.consecutive}/${MAX_CONSECUTIVE} (${sessionID.slice(0, 12)})`);
  await ctx.session.prompt({ sessionID, text });
}

function arm(ctx, sessionID, s) {
  // one timer per session: the latest event wins, the earlier one is dropped (v1's
  // deferred idle did the same), but the wait is never shortened below the throttle
  if (s.timer) clearTimeout(s.timer);
  const wait = Math.max(0, THROTTLE_MS - (Date.now() - s.lastPrompt));
  s.timer = setTimeout(() => {
    s.timer = null;
    void onTimer(ctx, sessionID, s).catch((e) => log("plugin internal error: " + String(e).slice(0, 200)));
  }, wait + 50);
}

async function onTimer(ctx, sessionID, s) {
  if (!s.errPending && !s.compactPending) return;
  // a turn still runs: wait for it rather than drop what is pending (a manual /compact can end
  // before its turn does); a turn that STARTS clears the pending flags itself, so this re-arm
  // never relaunches over it (review of v1.56.0, finding 8)
  if (s.busy) { arm(ctx, sessionID, s); return; }
  if (s.consecutive >= MAX_CONSECUTIVE) {
    await log(`STOP (${sessionID.slice(0, 12)}): ${MAX_CONSECUTIVE} relaunches without progress`);
    s.errPending = false;
    s.compactPending = false;
    return;
  }
  // the error takes priority (a hard break); otherwise a manual compaction that did not resume
  if (s.errPending) {
    s.errPending = false;
    await relaunch(ctx, sessionID, s, RESUME_AFTER_ERROR, "err");
    return;
  }
  s.compactPending = false;
  await relaunch(ctx, sessionID, s, RESUME_AFTER_COMPACTION, "compact");
}

const TYPES = [
  "session.execution.started",
  "session.execution.succeeded",
  "session.execution.failed",
  "session.execution.interrupted",
  "session.compaction.ended",
];

export default {
  id: "alzeo.auto-continue",
  async setup(ctx) {
    await log("auto-continue plugin loaded (v2: failed executions + manual compaction)");
    const controller = new AbortController();
    (async () => {
      try {
        for await (const event of ctx.event.subscribe({ types: TYPES, signal: controller.signal })) {
          const d = event.data ?? {};
          const sessionID = d.sessionID;
          if (!sessionID) continue;
          try {
            if (event.type === "session.execution.started") {
              const s = getState(sessionID);
              s.busy = true;
              // a turn is running: whatever was pending is moot, it decides on its own
              s.errPending = false;
              s.compactPending = false;
              if (s.timer) { clearTimeout(s.timer); s.timer = null; }
              continue;
            }
            if (event.type === "session.execution.succeeded") {
              const s = getState(sessionID);
              s.busy = false;
              // a finished turn with nothing pending is real progress
              if (!s.errPending && !s.compactPending) {
                s.consecutive = 0;
                s.sameRefusal = 0;
              }
              continue;
            }
            if (event.type === "session.execution.interrupted") {
              // user / shutdown are deliberate ends (v1's abort/interrupt exclusion);
              // superseded and inactivity are control flow, not technical errors
              await log(`interrupted (${d.reason ?? "?"}) not relaunched (${sessionID.slice(0, 12)})`);
              const s = getState(sessionID);
              s.busy = false;               // the projector makes the session idle here too (projector.ts)
              s.errPending = false;
              s.compactPending = false;
              continue;
            }
            if (event.type === "session.execution.failed") {
              const s = getState(sessionID);
              s.busy = false;
              const txt = errText(d.error);
              if (EXCLUDE.some((p) => txt.toLowerCase().includes(p))) {
                await log(`error EXCLUDED (${sessionID.slice(0, 12)}): ${txt.slice(0, 120)}`);
                continue;
              }
              // a count or an id that changes in the message still makes the same refusal
              const refusal = isRefusal(d.error) ? txt.replace(/\d+/g, "#") : "";
              s.sameRefusal = !refusal ? 0 : refusal === s.lastRefusal ? s.sameRefusal + 1 : 1;
              s.lastRefusal = refusal;
              if (s.sameRefusal >= SAME_REFUSAL_MAX) {
                await log(`STOP (${sessionID.slice(0, 12)}): the same refusal ${s.sameRefusal} times in a row: ${txt.slice(0, 120)}`);
                continue;
              }
              s.errPending = true;
              await log(`error retained (${sessionID.slice(0, 12)}): ${txt.slice(0, 120)}`);
              arm(ctx, sessionID, s);
              continue;
            }
            if (event.type === "session.compaction.ended") {
              // auto compactions resume inside the running turn (runner/llm.ts) and a
              // failed one surfaces as session.execution.failed; only a manual /compact
              // ends idle without resuming
              if (d.reason === "manual") {
                const s = getState(sessionID);
                s.compactPending = true;
                await log(`manual compaction ended (${sessionID.slice(0, 12)}): resume armed`);
                arm(ctx, sessionID, s);
              }
              continue;
            }
          } catch (e) {
            await log("plugin internal error: " + String(e).slice(0, 200));
          }
        }
      } catch (e) {
        if (!controller.signal.aborted) await log("subscription ended: " + String(e).slice(0, 200));
      }
    })();
    return () => controller.abort();
  },
};
