"""The OpenCode 2 permission matcher, as its server applies it.

Measured on the box's 2.3.0 binary and read in its bundle on 2026-11-15 (a case table measured on the
reference box): backslashes read as slashes on both sides, "*" matches any
run, "?" matches EXACTLY one character, everything else is itself, the dot and the brackets
included (a deny for "*co[nm]*" therefore matches only a command that writes those brackets), and
the pattern must cover the whole command text (a bare "aliyuncs.com" matches nothing longer).
"*" crosses newlines. A pattern ending " .*" also matches without the space and what follows it
(measured: "git *" matches "git" alone), which is opencode's own convenience for tool prefixes.

ONE reading, used by both halves of the guard (independent review of the v2 migration, finding 7):
oc-merge-limits.py verifies at install time that our exfiltration denies still deny, and
dashboard/agent_relay.py answers the cockpit's guard line from what the server says it serves. Two
matchers had drifted (the relay read "?" as literal, the merger as a wildcard) and probed different
strings, so the same security question got two answers."""
import re


def server_match(text, pattern) -> bool:
    a = str(text).replace("\\", "/")
    s = str(pattern).replace("\\", "/")
    # the server's own class of characters to keep literal, then the two wildcards
    s = "".join("\\" + c if c in ".+^${}()|[]\\" else c for c in s)
    s = s.replace("*", ".*").replace("?", ".")
    if s.endswith(" .*"):
        s = s[:-3] + "( .*)?"
    return re.fullmatch(s, a, re.S) is not None
