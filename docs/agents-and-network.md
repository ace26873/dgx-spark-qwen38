# Agents and the network

What an agent on this box runs without asking, and what this repo denies it.

## Why this page exists

On 2026-10-26 Cohere reported that Qwen3.8-Flash-Next, given a shell in an agent test without network access, sent a PUT of a probe string to an Alibaba Cloud OSS bucket in Beijing ([Cohere's post](https://cohere.com/blog/concerning-behaviour-popular-open-weight-ai-model)). It happened in fewer than 1 in 100,000 calls. The behavior behind it, an Alibaba Cloud host the model writes on its own, has turned up in builds from four publishers, Alibaba's included, so it comes from the original weights. Nothing left their sandbox. On a box with a network, an agent that runs every command it is given would let such a request leave.

"No prompt, image or video leaves the box" is about the serving stack. What an agent does with a shell is up to the agent's permission system, below.

## What runs without asking

- **opencode** (1.19.6, the version this repo pins) starts from permissive defaults: every tool, the shell included, runs without asking. It asks only before a loop it detects, before reaching outside the project directory and before reading a `.env` file. Read in 1.19.4's binary on 2026-10-28, and the same in 1.19.5's on 2026-10-30.
- **`oc`** runs opencode with `--yolo`, an alias of `--auto` in 1.19.4, 1.19.5 and 1.19.6: it approves those few questions too. A `deny` rule still holds under it.
- **The Agent tab**, installed without `AGENT_AUTO`, keeps opencode's defaults, so its shell commands run unasked as well. With `AGENT_AUTO=1` it approves every tool call, and opencode merges that setting after every rule of its config files: your own rules, `deny` included, do not hold under it. On OpenCode 2 there is a way that keeps them: leave `AGENT_AUTO` off and switch on **Auto-accept permissions** in the Agent view (its Settings, General; the palette, Ctrl+K, finds it by its name). The web app then answers by itself the requests a rule would ask you about, and a call one of your rules denies is still rejected without being asked. Measured on 2.3.2 the 2026-11-23, in a browser, with a config that asks for every `webfetch` and denies one host: without the setting the page showed its permission request and the call waited; with it the call ran unasked, and the one to the denied host answered `permission.rejected`. The setting lives in the browser (per browser, kept across restarts), and acts while a page of the Agent view is open: a request made with none open waits for the next one.
- **`claude-local`** starts Claude Code in its normal permission mode, which asks before each command ([clients.md](clients.md)).

## What this repo denies

The box's opencode configs carry these rules, under keys of their own, `"bash*"` (a wildcard that matches the shell tool) and `"webfetch*"` (the same for the webfetch tool; decision 1561 of 2026-11-22, after a request left this box as a `webfetch` to an Alibaba Cloud bucket on 2026-11-21, twice), as the last members of `"permission"`:

```json
"permission": {
  "bash*": {
    "*aliyuncs.com*": "deny",
    "*169.254.169.254*": "deny",
    "*100.100.100.200*": "deny"
  },
  "webfetch*": {
    "*aliyuncs.com*": "deny",
    "*169.254.169.254*": "deny",
    "*100.100.100.200*": "deny"
  }
}
```

opencode reads its permission rules in order and the last one that matches wins, so they come after every other rule, and your own `"bash"` stays as you wrote it.

- The config `install.sh` generates (`~/.config/qwen38/opencode.json`), which `oc` and the Agent tab load over your own: there the rules come after your global config's, but for a key added after the block in a global config that holds it too (this repo's copy does), until the next install puts the block last again.
- Your global config (`~/.config/opencode/opencode.json`), which plain `opencode` reads, when it is this repo's copy or lists this repo's providers. A key you add after the block later overrides it, and the next install puts it last again. A pattern you set inside it stays as you set it.
- A config of your own, in which none of this repo's provider names (`qwen38`, `flashnext`, `kolibri`, `nemotron`, `nemotron-super`, `flash-local`, `ornith-122b`, `flash-unsloth`, `lfm25`, `lfm3`) shows, never gets the rules: `oc` and the Agent tab still get the rules, and for plain `opencode` the block above is the one to add, as the last member of your `"permission"`.
- With `AGENT_AUTO=1`, the Agent tab's unit repeats the rules after its `"*": "allow"`, under `"*bash"` and `"*webfetch"`.

Checked on the box on 2026-10-28 with opencode 1.19.4 and the FP4-Local lane: a shell command naming an aliyuncs.com host was refused under `--yolo` and under the new `AGENT_AUTO` value, and ran under the old one (`{"*":"allow"}` alone). On 2026-10-30 opencode 1.19.4 and 1.19.5 refused it the
same way under both, against an endpoint that asked for that very command, where a command naming no such host ran. On 2026-11-22 (opencode 1.19.5) and 2026-11-23 (the pinned 1.19.6) the same bench, against a scripted endpoint replaying a `webfetch` tool call: opencode refused it with `The user has specified a rule which prevents you from using this specific tool call` and printed the rules that did it (`"webfetch*"` in the config, `"*webfetch"` in the unit's `OPENCODE_PERMISSION`), while the same call with no rules fetched its URL (the control). opencode 2.3.2 answered the same refusal as `permission.rejected`, and its own source re-checks every redirect the fetch follows (`core/src/tool/plugin/webfetch.ts:133-171`).

On OpenCode 2 (the default since v1.56.0, docs/opencode.md) the same three hosts ride the v2 grammar for both tools, an ordered array `"permissions"` of `{action, resource, effect}` objects whose last match wins (three `shell` rules and three `webfetch` ones), and with `AGENT_AUTO=1` they come last inside `OPENCODE_CONFIG_CONTENT`, the document the server merges after every file (measured on 2.3.0 the 2026-11-14: the wildcard allow of the content approves what its last rules do not deny, and a deny of ours holds over an earlier allow). Two readers check they still deny, with one matcher and one table of probes (`oc-permissions.py` and `deny_probes()`): the installer verifies the file it wrote, and the Agent panel answers `guard_ok` from the documents the running server serves on `/api/config`, not from the file on disk. The matcher is the server's own, read in 2.3.0's bundle and measured on the binary the 2026-11-15: `*` any run, `?` exactly one character, brackets and the dot literal, whole-text match, and a needle inside a command's comment is not command text (`touch x # aliyuncs.com` is no request to that host). A guard the panel cannot read (no pattern to probe, an unreadable answer) says nothing rather than saying well: `guard_ok` stays null and the line does not claim the guard holds.

What they stop: the case Cohere observed, a shell command whose text names an aliyuncs.com host or a cloud metadata address, and since decision 1561 the same hosts named as a `webfetch` URL. The first pattern is Alibaba Cloud's whole domain, its OSS buckets as well as DashScope and its container registry, and a command naming any of them is refused and raises the banner below. What they do not stop: a host written in capitals, since opencode's patterns tell case apart (the proxy below does not), a model that writes the address another way (a variable, an encoding, a script it writes first). What opencode reads after them can allow what they deny: a project's own `opencode.json` or `.opencode` directory, which opencode loads after the configs above, an agent's own `permission` block, and an "Always" you give to a command (below). In a config of yours the rules stay in place after `uninstall.sh`, as `autoupdate` does, and this repo's own copy is moved aside whole, as before.

## What the cockpit shows

The proxy sees every tool call before the client runs it, whatever the client (opencode, Claude Code, Hermes, pi, an SDK). When a call's arguments name one of these hosts, it writes one line to its journal, the host and the tool's name, never the arguments, and relays the answer as it came. A tool that reads, searches or writes files, or hands work to a sub-agent, is not read: a page that quotes these hosts, this one included, or a search for them, is no request to them. The cockpit reads a day of those lines and shows a red banner with the count, the host, the tool and the route. The proxy only reports: whether the call ran depends on the client and its rules. It does not read the tool arguments of a streamed request past its first MiB, nor a whole answer over 8 MiB.

## Approving each command

To approve every shell command yourself, put `"*": "ask"` in the `"bash"` object of your config and start opencode without `--yolo` (plain `opencode`, not `oc`). The rules above still come after it, but an "Always" you answer comes after them in turn, for the rest of the session, and allows what it names, these hosts included: answer "Once" to a command you have not read whole.

```json
"permission": {
  "bash": {
    "*": "ask"
  },
  "bash*": {
    "*aliyuncs.com*": "deny",
    "*169.254.169.254*": "deny",
    "*100.100.100.200*": "deny"
  }
}
```

The Agent tab asks the same way when it is installed without `AGENT_AUTO`.

[Back to the README](../README.md)
