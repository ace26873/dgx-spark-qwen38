# The Inkling 2 Mini lane: Thinking Machines' looped mixture of experts on SGLang

Inkling 2 Mini is Thinking Machines' multimodal model (Apache-2.0, released on 2026-11-20): text and images in, text
out, a window of 1M tokens. It is a looped transformer: 24 layers of weights, of which the middle twelve (6 to 17) run
three times per token, 48 layer passes in all; each layer a mixture of 128 experts, 8 of them per token. Its card
counts 26.4B parameters with 3.4B active, 5.4B "effective" with the two extra passes. Its experts are in NVFP4 (layers
0 to 19) and MXFP8 (20 to 23), one checkpoint of about 20 GB, with a 3-layer MTP head beside it (`mtp.safetensors`),
which SGLang's DSpark speculative decoding drafts with. This lane serves `thinkingmachines/Inkling-2-Mini` at its
pinned revision on the same port and behind the same proxy and key as the other text lanes, as model `inkling-2-mini`.

The lane serves it at the 27B lane's memory share, 0.76, where the cookbook's recipe gives 0.45: at 0.76 its KV cache
holds the model's whole window. The 27B stays the lane a box serves by default.

What is measured of it, on the reference box (an ASUS Ascent GX10, 128 GB): the lane's own unit at its share of 0.76
on 2026-11-24, installed by `./install.sh` and loaded as the cockpit loads it; the recipe at that share the same day,
from a container of its own, its memory and its pool; and at the recipe's share of 0.45, the bench of 2026-11-22
against the 27B lane the same evening by the same tools, and the quality runs of 2026-11-22 to 24. The quality does
not depend on the share; the memory and the pool do.

## Why SGLang, and this image

Inkling 2 Mini is in no SGLang release yet. SGLang's cookbook serves it from the `lmsysorg/sglang:inkling-2-mini` image
(its support is pull request sgl-project/sglang#51926) and gives a recipe for each platform, the DGX Spark's among
them, with and without DSpark. The lane pins the arm64 build of that image by its digest (its build label says
`nightly-dev-20261120-03fde9cc`), 38.6 GB unpacked and 16.4 GB to download, and runs the cookbook's DGX Spark recipe
with DSpark, word for word but for its memory share.

## The serving command

`qwen38-inkling.service` runs, in that image:

```
python3 -m sglang.launch_server --model-path thinkingmachines/Inkling-2-Mini --revision <pinned sha> --tp 1 --ep 1 \
  --served-model-name inkling-2-mini --attention-backend fa4 --dtype bfloat16 --page-size 128 \
  --context-length 1048576 --mem-fraction-static 0.76 --max-running-requests 48 \
  --speculative-algorithm DSPARK --speculative-num-draft-tokens 8 --grammar-backend xgrammar --enable-multimodal \
  --sleep-on-idle --enable-metrics --enable-cache-report --reasoning-parser inkling --tool-call-parser inkling2 \
  --config /out/engine-secrets.yaml --host 127.0.0.1 --port 30000
```

with `MAX_JOBS=4` and the recipe's four `SGLANG_OPT_*` variables of its DSpark arm in its environment.

- **The recipe's own**: tensor and expert parallelism of one, the FA4 attention backend, bfloat16, pages of 128
  tokens, xgrammar, the 8 draft tokens DSpark verifies per step, and `MAX_JOBS=4`, the cap on the kernels the engine
  compiles at its start.
- **The memory share, 0.76**, the 27B lane's (decision of Hasan, 2026-11-24: the most context and speed the box
  gives), where the recipe gives 0.45. At 0.45 the KV cache held 490,496 tokens (2026-11-22), under half the window; at
  0.76 it holds 1,091,200 (2026-11-24), the window whole, which decides the clients' limits below.
- **The draft is in the checkpoint**: SGLang takes `mtp.safetensors` from the model's own repository ("DSpark draft
  weights are bundled in the target checkpoint", its journal), by the model's name and with no revision, so through
  `refs/main`, which the lane's fetch pins at the revision it downloads.
- **The window**, 1,048,576 tokens, is the model's (its config's `model_max_length`): SGLang takes it by itself, and
  the unit says it so that no later default moves it.
- **`--max-running-requests 48`** is the number SGLang sets by itself at both shares, written down for the same reason.
- **Reasoning, by effort.** The checkpoint's template reads a request's `chat_template_kwargs.reasoning_effort`: a
  number from 0.0 to 0.99, or a level by name (`none` 0.0, `minimal` 0.1, `low` 0.2, `medium` 0.7, `high` 0.9, `xhigh`
  and `max` 0.99). With none given, it reasons at 0.9, `high`. The engine refuses a value out of bounds (400, measured
  with 5). The model's card says the lowest effort does not turn the reasoning off. SGLang's `inkling` parser puts the
  reasoning in `reasoning_content`.
- **Tool calls**: SGLang's `inkling2` parser turns the model's own tool messages into OpenAI `tool_calls`.
- **Images**: `--enable-multimodal`, the recipe's, so the server takes image inputs.
- **Sampling**: the checkpoint's `generation_config.json` asks temperature 1.0, which SGLang applies to a request that
  sets none.
- **Its caches, kept**: the lane mounts a folder of its own for torch's compile cache and for the two kernel caches the
  engine fills at its first start (SGLang's and b12x's). The bench kept them the same way: 379 s to its first answer at
  its first start, when it compiled them, 222 s at the next.
- **What every SGLang lane of this box passes besides**, as the 27B's unit says why: the health route that generates
  nothing, the rendezvous through a file, gloo and NCCL on loopback, the request header overrides the proxy's request
  ids need, the offline cache, the scheduler parked when idle, the metrics and the cache report, the key by `--config`.
  `--memory 100g`, the 27B lane's cap: the cgroup is not charged for the engine's GPU memory, which the share bounds.

## Measured on the reference box

At the lane's share of 0.76, on 2026-11-24 (`essai-076.log`): ready in 237 s, its kernels already compiled; 86,455
MiB (84.4 GiB) of GPU memory once ready; MemAvailable 24.37 GiB at its lowest while it loaded; a KV cache of 1,091,200
tokens, and 109,056 in its sliding window; 48 requests and a window of 1,048,576 in SGLang's own `server_info`.

The lane's own unit at 0.76, the same day: installed by `./install.sh` from this release's tree, loaded as the cockpit
loads it (the 27B stopped first), every request through the proxy and the box's key, the GPU clock capped at 2,000 MHz,
so its speeds are a lower bound (the bench of 2026-11-22 below ran uncapped):

- ready in 254 s from the cockpit's switch, with 85,683 MiB (83.7 GiB) of GPU memory once ready;
- `bench.sh`, greedy median, two runs: 62.6 and 62.8 tok/s (free prose, the worst case: 28.5 to 28.9 tok/s);
- a needle in a prompt of 960,037 tokens: found, 1,461 s for the whole request, 476 tokens out; in a prompt of 64,835
  tokens: found in 27.8 s;
- an image in, read; the effort by its name (`none`, `high`) and by a number (0.5), each answered, and 5 refused (400,
  out of the template's bounds);
- an agent's four turns, by the OpenAI and the Anthropic routes: from the second turn on, 7,168 of its 7,222 to 7,256
  prompt tokens read from the prefix cache, about 1.0 s a turn against 3.4 s for the first;
- Claude Code, opencode, Hermes, pi, LiteLLM, the OpenAI and Anthropic SDKs, fetch from Node and from Bun, and the
  production checks: nine of nine.

At the recipe's share of 0.45, the bench of 2026-11-22 (`banc-inkling.sh`, its DSpark arm against the 27B lane's
unit, the same tools, one after the other):

| | Inkling 2 Mini, the recipe at 0.45 | Qwen3.8 27B, its unit |
|---|---|---|
| ready, its `/health` answering | 222 s (379 s at its first start) | 213 s |
| GPU memory at its peak (nvidia-smi) | 53.1 GiB | 89.0 GiB |
| MemAvailable once ready | 56 GiB | 20 GiB |
| KV cache | 490,496 tokens, in its journal | |
| `bench.sh`, greedy median, two runs | 64.7 and 64.8 tok/s | 72.3 and 70.0 tok/s |
| one stream (1,024 tokens in, 512 out, median of 3) | 30.3 tok/s, first token in 0.48 s | 34.0 tok/s, first token in 0.48 s |
| 8 streams, in all (median of 3) | 89.2 tok/s, first token in 1.18 s | 119.0 tok/s, first token in 2.20 s |
| a prompt of 32,015 tokens, to its first token | 10.0 s (two runs) | not measured that evening |
| tool calls, three prompts | 3 of 3 | 3 of 3 |

**Quality**: GSM8K, 250 questions, with `sglang.test.run_eval` from the 27B lane's image, 8 at once, at the model's
default effort: 0.964 (2026-11-22). Again on 2026-11-23 at the template's highest effort, 0.99, and the sampling its
card recommends (temperature 1.0, top_p 1.0): 0.972, no answer cut at 16,384 tokens, 651 tokens on average, 9,493 at
the most. On 2026-11-24 at its default effort, 0.9, with that same sampling and the same cap: GSM8K 0.980 (no answer
cut, 574 tokens on average, 6,522 at the most) and HumanEval 0.976 (164 problems, pass@1, one sample a problem, 160
passed, one answer cut at 16,384 tokens, 1,369 tokens on average). The default effort is the lane's setting: on GSM8K
it scores higher than the highest effort with shorter answers (574 tokens on average against 651); HumanEval at the
highest effort did not finish in time (13 of 164 problems in 10 minutes). The 27B lane scores 99.2 % on the same 250 questions
([BENCHMARKS.md](../BENCHMARKS.md)).

Not measured yet at 0.76: several requests at once (8 at once at the recipe's share, in the table above).

## Clients

Through the proxy on `:30001`, with the box's key, as model `inkling-2-mini`:

- **OpenAI** and **Anthropic** routes, SGLang's own, as on the 27B lane.
- **opencode** gets an `inkling` provider that takes images, with five variants, each a level of the template by its
  name in `chat_template_kwargs.reasoning_effort`: `high` (its default) first, then `none`, `low`, `medium` and `xhigh`
  (`minimal` left out: 0.1 against `low`'s 0.2). Its limits are 871,000 tokens of context and 32,000 of output: the
  pool decides at this window, and the pair fits the pool measured less the 8 % the proxy keeps and 10 % for a boot
  that gets less (`871,000 + 32,000 <= 1,091,200 x 0.92 x 0.90 = 903,513`). After a boot, the fit to the engine's pool keeps the
  32,000 of output as this lane's ceiling: its card recommends limits on output tokens, as the model can loop in its
  reasoning ([docs/opencode.md](opencode.md)). Loading the lane makes it opencode's default.
- **pi** sends its level as the top-level field, its `off` as `none`, and starts at `high`; **omp** offers `high`,
  `minimal` (sent as `none`), `low`, `medium` and `xhigh` ([docs/pi.md](pi.md)).
- **The proxy** relays a `none` as it came on this lane (v6.62): on the other SGLang lanes it becomes
  `enable_thinking` false, which the Qwen templates read and this one does not.
- **System One** is not served here (proxy v6.62): the model reasons before every answer at any effort, so the one
  token System One reads would be the thought's. The proxy says so (400), with the lanes that serve it.

## Its licence

The checkpoint is published under the Apache License 2.0, with Thinking Machines' Model Acceptable Use Policy, which
its card links. Its card names its limits: hallucination and less factual recall than larger models, reasoning that
can loop, and, with tools, less reliable instruction following and actions taken without confirmation (deleting or
overwriting files), lower quality outside English.

## Switching to it, and back

- The cockpit: **Inkling 2 Mini** in the Lanes view, then **Load**. Its first Load pulls its image (16.4 GB to
  download), then fetches its checkpoint (20.2 GB) in that image.
- A terminal: `./prepare-lane.sh inkling` the first time, then `./switch-model.sh inkling` and the commands it prints.
- Installed on demand by every plain `./install.sh` on a 128 GB box (its unit, and its first Load prepares it).
  `--with-inkling` installs it whole now and proves it, and `--no-inkling` leaves it out of the run that passes it (75
  GB with its image). A 64 GB DGX Spark leaves it out: its memory was measured on a 128 GB box only, and no 64 GB run
  was made. `./install-inkling.sh --uninstall` removes its unit and its image's tag: the checkpoint and the image
  stay, named, and a `docker image prune` reclaims the image.
