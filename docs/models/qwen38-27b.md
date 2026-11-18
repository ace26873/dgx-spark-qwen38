# Qwen3.8-27B on a DGX Spark (and any GB10 box)

Qwen's 27B in NVFP4 (`RadixArk/Qwen3.8-27B-NVFP4`, Qwen3.8 27B NVFP4 · RadixArk in the cockpit), the `stock` target of [Alzeo](../../README.md)'s
27B lane: served by SGLang 0.5.22 with a DFlash2 speculative drafter, a calibrated NVFP4 head 16 deep, on one
DGX Spark or one ASUS Ascent GX10, with a window of 1,010,000 tokens.

## One command

```bash
./switch-model.sh stock        # or: uncensored | fp8 | uncensored-fp8 | arete | daisyrose
```

The one-line install of the [README](../../README.md) sets it up, and the [cockpit](../cockpit.md)
loads it with a button.

## Measured on our box (ASUS Ascent GX10, 128 GB)

On 2026-11-05, the lane as v1.44.0 serves it (SGLang v0.5.22, without torch.compile):

- **44.0 tok/s alone and 80.6 in all at 4 requests** on SGLang's official benchmark (8,192 tokens in,
  1,024 out, the median of three runs).
- **65.2 and 68.5 tok/s** greedy median on `./bench.sh`, two runs.
- **GSM8K 99.2%** on 250 questions.
- Ready to answer **208 s** after its start.

The A/B against torch.compile that gave these numbers is in the
[v1.44.0 notes](../../CHANGELOG.md), the older campaigns in [BENCHMARKS.md](../../BENCHMARKS.md).

The window's limits are fitted to the KV pool each boot gets, 832,993 to 922,094 tokens over sixteen
boots here: [docs/context-1m.md](../context-1m.md). Speculation is lossless by construction, the
target verifies every drafted token, so speed costs no quality
([the losslessness study](../../BENCHMARKS.md#the-losslessness-study-2026-08-20)).

## What it is good at

General text work with long context: agent loops in Claude Code or opencode, code with files open,
summaries of whole repositories. The six targets share one serving path: `stock` for general use,
the two abliterated targets for fewer refusals, Qwen's own FP8 release as the quantization
reference, `arete`, Nous Research's fine-tune, and `daisyrose`, RadixArk's
reinforcement-learning-only fine-tune. Free prose is the slow case on any drafter,
[BENCHMARKS.md](../../BENCHMARKS.md) says by how much.

## Arete 27B

`arete` serves [Arete 27B](https://huggingface.co/NousResearch/Arete-27B), Nous Research's
fine-tune of Qwen3.8 27B (Apache-2.0), in atakans's NVFP4 (`atakans/Arete-27B-NVFP4`, Arete 27B
NVFP4 · atakans in the cockpit, 21.1 GB), with the lane's unit, image and DFlash2 draft as they
are. Its checkpoint declares no KV cache: its switch writes the one RadixArk's NVFP4 declares into
its config (`patch-kv.py`), so its target's cache is fp8 and its draft's bf16, as for `stock`.
Measured on our box on 2026-11-14 under the lane's unit:

- **69 to 72 tok/s** alone on `./bench.sh`'s code and reasoning prompts, two runs.
- **GSM8K 99.2%** on 250 questions, **MMLU 88.0%** on 500, **HumanEval 96.3%**.
- A KL divergence of **0.030** to Nous's BF16 over 30 texts, the needle exact at 120,000 and
  300,000 tokens, a KV pool of **905,941 and 902,974 tokens** on its two boots.

## DaisyRose 27B

`daisyrose` serves [DaisyRose 27B](https://huggingface.co/RadixArk/DaisyRose-27B), RadixArk's
reinforcement-learning-only fine-tune of Qwen3.8 27B (Apache-2.0, no SFT, no distillation), in
[RadixArk's own NVFP4](https://huggingface.co/RadixArk/DaisyRose-27B-NVFP4)
(`RadixArk/DaisyRose-27B-NVFP4`, revision `c4d85c5`, DaisyRose 27B NVFP4 · RadixArk in the
cockpit, 21.95 GB), with the lane's unit, image and DFlash2 draft as they are: same
architecture, its tokenizer files and chat template byte for byte (its tokenizer_config marks
nine more tokens special, none of the think or tool tags), and its config loads with
the lane image's transformers 5.19.0. Its checkpoint declares no KV cache: its switch
writes the one RadixArk's NVFP4 declares into its config (`patch-kv.py`), so its target's
cache is fp8 and its draft's bf16, as for `stock`. The measurement on our box remains to be
done. RadixArk publishes, at xhigh reasoning and temperature 1.0: Terminal-Bench 2.1 at
75.5 against 73.0 for Qwen3.8-27B, Terminal-Bench 4.1 at 12.0 against 5.6, SWE Bench Pro v2
at 94.7 against 91.3 and GPQA Diamond at 90.7 against 89.2; several of the Qwen3.8-27B
scores come from model cards or an article, not a paired run.

## In the repo

- [BENCHMARKS.md](../../BENCHMARKS.md), every number with its method and its failed experiments
- [docs/context-1m.md](../context-1m.md) for the 1M window
- [docs/clients.md](../clients.md) for the clients the proxy is tested with
