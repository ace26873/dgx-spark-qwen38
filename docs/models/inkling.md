# Inkling 2 Mini on a DGX Spark

Thinking Machines' `Inkling-2-Mini` (Apache-2.0), a looped mixture of experts of 26.4B parameters with 3.4B active
per token, its middle layers run three times, text and images in, served by SGLang from the image its cookbook gives
the model, with the DSpark draft head of its checkpoint, at its window of 1,048,576 tokens.

## One command

```bash
./switch-model.sh inkling
```

The one-line install sets up its unit and its first Load prepares it (75 GB with its image).
`./install.sh --with-inkling` installs it at install and proves it.

## Measured on our box (ASUS Ascent GX10, 128 GB)

Its recipe in its image, on 2026-11-22 and 23, from a container of the bench's own, against the 27B lane's unit the
same evening:

- **53.1 GiB** of GPU memory at its peak, against 89.0 for the 27B lane; a KV cache of 490,496 tokens.
- **64.7 tok/s** greedy median (`./bench.sh`), against 72.3; 30.3 tok/s on one stream and 89.2 in all at 8 at once,
  against 34.0 and 119.0.
- **GSM8K 0.980** (250 questions) and **HumanEval 0.976** (164 problems, pass@1) at its default effort with its card's
  sampling (0.964 on GSM8K greedy), 0.972 on GSM8K at its highest effort; three tool calls of three.
- Ready in 222 s (379 s at its first start, when it compiled its kernels).

The lane's own unit, a long prompt, an image in and the clients through the proxy are not measured yet.

## What it is good at

The light text lane: about 60 % of the 27B lane's memory, the model's window of a million tokens, reasoning at the
effort a request asks (`none` to `xhigh`), tool calls, images. Slower than the 27B lane on this box, alone and under
load. Its card warns of reasoning that can loop and of actions taken without confirmation in agents.

## In the repo

- [docs/inkling-lane.md](../inkling-lane.md), the lane page with its pins, its recipe and what is measured
- [BENCHMARKS.md](../../BENCHMARKS.md), every number with its method
