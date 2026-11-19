# Liquid AI LFM3 1.7B on a DGX Spark

Liquid AI's `LFM3-1.7B-NVFP4`, a small hybrid of short convolutions and attention post-trained for agent
harnesses, served at its whole window of 262,144 tokens by vLLM's own nightly image (commit 60bd6dd), in which LFM3
is supported before any vLLM release.

## One command

```bash
./switch-model.sh lfm3
```

The one-line install sets up its unit and its first Load prepares it (41 GB of room: 2 for the checkpoint, 39 for
the image while it is pulled). `./install.sh --with-lfm3` installs it at install and proves it.

## Measured on our box (ASUS Ascent GX10, 128 GB)

On 2026-11-16, with no memory share set, the GPU clock capped at 2,000 MHz:

- **87.7 tok/s** per stream alone, **3,911.8 tok/s** in all at 128 requests at once (1,024 tokens in, 512 out).
- **GSM8K 0.896** (250 questions), where the same model in a Q8_0 GGUF on llama.cpp scored 0.844 on the same
  questions.
- 131,072 tokens of prompt computed in **10.2 s**, an agent's next turn served **99 to 100 %** from the prefix
  cache.

The lane serves it with a memory share of 0.75, measured as it ships on 2026-11-19, through the proxy, the GPU
clock capped at 2,000 MHz: ready in 133 s, 20 GiB of memory left available at its lowest, **90.7 tok/s** on one
stream and **732.2 tok/s** in all at 8 requests at once (1,024 tokens in, 512 out), an agent's next turn served
**99 to 100 %** from the prefix cache.

## What it is good at

What Liquid AI designed it for, local agents, tool calls and structured outputs, many requests at once on one
box, at thousands of tokens per second in all. Liquid AI does not recommend it for agentic coding or complex
multi-step reasoning. Its licence, the LFM Open License v1.0, does not license commercial use by a company with an
annual revenue of 10 million US dollars or more.

## In the repo

- [docs/lfm3-lane.md](../lfm3-lane.md), the lane page with its pin, its memory share and every measurement
- [BENCHMARKS.md](../../BENCHMARKS.md), every number with its method
