#!/usr/bin/env python3
"""The Inkling 2 Mini lane's opencode provider, in one place.

install.sh's generator writes it into the opencode config once the lane is installed, and
switch-model.sh adds it to a config written before that, so both take it from here and the two
cannot drift.

Usage: oc-inkling.py <base_url> <api-key file> <ctx> <out> <name>
prints {"provider": {"inkling": ...}}, the shape oc-merge-limits.py --add-providers reads.
"""
import json
import sys

# Inkling 2 Mini's chat template (its checkpoint's, served as published, read at the pinned revision on 2026-11-24)
# reads reasoning_effort, a number in [0.0, 0.99] or one of its levels by name: none 0.0, minimal 0.1, low 0.2, medium
# 0.7, high 0.9, xhigh and max 0.99; with none given it reasons at 0.9, high. Its card says the lowest effort does not
# turn the reasoning off. Five variants, each a level of the template by its name, in chat_template_kwargs as on the
# Kolibri lane: high, the default, first as a picker lists it, then the others from the lowest. minimal is left out:
# 0.1 against low's 0.2, and omp, whose pi-gen.py thinking block rides none on its own minimal, would send none for both.
LEVELS = ("high", "none", "low", "medium", "xhigh")


def provider(base_url, key_file, ctx, out, name):
    """The provider block. The model reads images (its vision tower, which the unit's --enable-multimodal serves), so
    opencode may attach one."""
    return {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Inkling 2 Mini (DGX Spark)",
        "options": {"baseURL": base_url, "apiKey": f"{{file:{key_file}}}"},
        "models": {"inkling-2-mini": {
            "name": name,
            "limit": {"context": ctx, "input": ctx, "output": out},
            "variants": {lvl: {"chat_template_kwargs": {"reasoning_effort": lvl}} for lvl in LEVELS},
            "attachment": True,
            "modalities": {"input": ["text", "image"], "output": ["text"]},
        }},
    }


def main():
    if len(sys.argv) != 6:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)
    base_url, key_file, ctx, out, name = sys.argv[1:6]
    try:
        ctx, out = int(ctx), int(out)
    except ValueError:
        print("ctx and out are token counts", file=sys.stderr)
        sys.exit(2)
    print(json.dumps({"provider": {"inkling": provider(base_url, key_file, ctx, out, name)}}, indent=2))


if __name__ == "__main__":
    main()
