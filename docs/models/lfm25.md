# Liquid AI LFM2.5 24B-A2B on a DGX Spark

Liquid AI's `LFM2.5-24B-A2B-FP8`, a hybrid mixture of experts of 24B parameters with about 2B active per token,
with Liquid AI's own DSpark draft, served by SGLang in the 27B lane's image at its window of 131,072 tokens.

## One command

```bash
./switch-model.sh lfm25
```

The one-line install sets up its unit and its first Load prepares it (26 GB beside the 27B lane, whose image it
runs, 74 without it). `./install.sh --with-lfm25` installs it at install and proves it.

## Measured on our box (ASUS Ascent GX10, 128 GB)

One short trial, on 2026-11-14, the GPU clock capped at 2,000 MHz, without the lane's parsers:

- **GSM8K 0.912** (250 questions), where the 27B lane scores 99.2 % on the same questions.
- **362 tok/s** in all at the median with 8 requests at once, the draft accepting **5.28** tokens at the median.
- Ready in 286 s, a KV cache of 1,667,049 tokens, 24 GiB of memory left available at its lowest.

The lane as it ships, on 2026-11-19, through the proxy, the GPU clock capped at 2,000 MHz:

- **102.2 tok/s** on one stream (1,024 tokens in, at most 512 out), **389.2 tok/s** in all at 8 requests at once.
- Ready in 292 s, 22 GiB of memory left available at its lowest, a tool call right.
- An agent's next turn served **94 %** from the prefix cache, from its second turn.

A long prompt and GSM8K with the lane's own flags are not measured yet.

## What it is good at

A larger Liquid AI model than LFM3, for many requests at once on one box, that reasons before every answer: its
template opens the thought itself and has no switch to turn it off. Its licence, the LFM Open License v1.0, does not
license commercial use by a company with an annual revenue of 10 million US dollars or more.

## In the repo

- [docs/lfm25-lane.md](../lfm25-lane.md), the lane page with its pins, its parsers and what is measured
- [BENCHMARKS.md](../../BENCHMARKS.md), every number with its method
