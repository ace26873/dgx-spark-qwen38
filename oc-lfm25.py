#!/usr/bin/env python3
"""The Liquid AI LFM2.5 24B-A2B lane's opencode provider, in one place.

install.sh's generator writes it into the opencode config once the lane is installed, and
switch-model.sh adds it to a config written before that, so both take it from here and the two
cannot drift.

Usage: oc-lfm25.py <base_url> <api-key file> <ctx> <out> <name>
prints {"provider": {"lfm25": ...}}, the shape oc-merge-limits.py --add-providers reads.
"""
import json
import sys


def provider(base_url, key_file, ctx, out, name):
    """The provider block. No variants, and no "reasoning": the checkpoint's chat template opens <think> at every
    answer and reads no level and no switch (its chat_template.jinja at the pinned revision, read 2026-11-16), so the
    lane reasons on every request whatever a client asks, and opencode must offer no level to pick. opencode 1.19.5
    builds a model's variants from its "reasoning" capability, false when the config leaves it out (provider.ts), and
    gives none to a model without it (ProviderTransform.variants); with it, it offers low, medium and high on
    @ai-sdk/openai-compatible, three levels this template does not read. Text only: the model reads no image (its
    config, Lfm2MoeForCausalLM)."""
    return {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Liquid AI LFM2.5 24B-A2B (DGX Spark)",
        "options": {"baseURL": base_url, "apiKey": f"{{file:{key_file}}}"},
        "models": {"lfm2.5-24b-a2b": {
            "name": name,
            "limit": {"context": ctx, "input": ctx, "output": out},
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
    print(json.dumps({"provider": {"lfm25": provider(base_url, key_file, ctx, out, name)}}, indent=2))


if __name__ == "__main__":
    main()
