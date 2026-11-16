#!/usr/bin/env python3
"""The Liquid AI LFM3 lane's opencode provider, in one place.

install.sh's generator writes it into the opencode config once the lane is installed, and
switch-model.sh adds it to a config written before that, so both take it from here and the two
cannot drift.

Usage: oc-lfm3.py <base_url> <api-key file> <ctx> <out> <name>
prints {"provider": {"lfm3": ...}}, the shape oc-merge-limits.py --add-providers reads.
"""
import json
import sys

# LFM3's chat template (its checkpoint's, served as published, read at the pinned revision on 2026-11-16) reads
# enable_thinking, true by default, and no reasoning effort: the model reasons, or it does not (its generation prompt
# then holds an empty thought, measured: an answer with no <think>). Two variants, the two states it holds, the default
# first as a picker lists it, as on the Ornith lane, whose template reads the same switch alone; no level it cannot tell
# apart from these two.
VARIANTS = {
    "high": {"chat_template_kwargs": {"enable_thinking": True}},
    "none": {"chat_template_kwargs": {"enable_thinking": False}},
}


def provider(base_url, key_file, ctx, out, name):
    """The provider block. Text only: the model reads no image (its config, Lfm3ForCausalLM)."""
    return {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Liquid AI LFM3 (DGX Spark)",
        "options": {"baseURL": base_url, "apiKey": f"{{file:{key_file}}}"},
        "models": {"lfm3-1.7b": {
            "name": name,
            "limit": {"context": ctx, "input": ctx, "output": out},
            "variants": {lvl: json.loads(json.dumps(v)) for lvl, v in VARIANTS.items()},
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
    print(json.dumps({"provider": {"lfm3": provider(base_url, key_file, ctx, out, name)}}, indent=2))


if __name__ == "__main__":
    main()
