# The Liquid AI LFM3 lane: LFM3 1.7B in NVFP4 on vLLM's own nightly

LFM3 1.7B is Liquid AI's small hybrid model: 24 layers, 13 of them short convolutions and 11 attention (6 over
the whole context, 5 over a sliding window), a vocabulary of 128,000 tokens and a native window of 262,144 tokens.
Liquid AI's card says it is post-trained for agent harnesses (Hermes Agent, Pi, OpenClaw), designed for local
agents, tool calling, structured outputs and fine-tuning, and not recommended for agentic coding or complex
multi-step reasoning, for its small size. This lane serves Liquid AI's own NVFP4 build of it,
`LiquidAI/LFM3-1.7B-NVFP4`, on the same port and behind the same proxy and key as the other text lanes, so every
client reaches it the same way, as model `lfm3-1.7b`.

The numbers on this page were measured on the reference box (an ASUS Ascent GX10, 128 GB) on 2026-11-16, in the
campaign called "base": the image and the checkpoint the lane pins, in a test container of its own on loopback,
without the key and without the proxy, **with no memory share set** (vLLM's `--gpu-memory-utilization` left unset),
the GPU clock capped at 2,000 MHz and the chip under 70 °C before each step, every other lane stopped. The lane
itself sets a memory share of 0.75 (below), and the proof campaign of v1.53.0 measured the lane as it ships at that
share on 2026-11-19, through the proxy, the GPU clock capped at 2,000 MHz: what it measured is below, and what no
campaign has measured yet is said so where it stands.

## Why vLLM's nightly, and which build

LFM3 came to vLLM on 2026-11-06 ([vllm#65826](https://github.com/vllm-project/vllm/pull/65826), commit
c96e5f60bb), and no vLLM release carries it yet. SGLang's came on 2026-11-16
([sgl-project/sglang#46858](https://github.com/sgl-project/sglang/pull/46858), commit 47361c57be9f), after this lane
was measured, and no SGLang release carries it yet either. So `install-lfm3.sh` pulls vLLM's
own nightly image of commit 60bd6dd (`vllm/vllm-openai`, vLLM 0.34.0rc1.dev571+g60bd6dd2e, CUDA 13.2.1, arm64),
pinned by its digest, as published: nothing is built on the box. It started on the box's driver (580.178.04, CUDA
13.0) and read the NVFP4 weights with FlashInfer, whose NVFP4 GEMM it tunes at the first boot.

The checkpoint is Liquid AI's: its feed-forward weights and activations in NVFP4 (ModelOpt, blocks of 16 values),
its attention, convolutions, embeddings, output head and norms left in BF16, as its card and its
`hf_quant_config.json` say. That is why its one file, 1,829,744,600 bytes, weighs about what the Q8_0 GGUF of the
same model does (1,846,238,496 bytes).

The same model in that Q8_0 GGUF on llama.cpp (b12316), in a short trial on 2026-11-14 at the same GPU clock cap,
scored 0.844 on the same 250 GSM8K questions, where this build scores 0.896 (below), and gave 103.6 tokens/s on one
answer, read in llama.cpp's own timings: not the method this page's speeds come from, so the two are not compared
here. The lane as it ships gave 90.7 tokens/s on one stream on 2026-11-19 (below), but through the proxy and not by
llama.cpp's own timings, so the two are still not compared.

## The serving command

`qwen38-lfm3.service` runs:

```
vllm serve LiquidAI/LFM3-1.7B-NVFP4 --revision <pinned sha> --served-model-name lfm3-1.7b \
  --max-model-len 262144 --gpu-memory-utilization 0.75 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --enable-prompt-tokens-details --config /out/engine-secrets.yaml --middleware vllm_engine_auth.RequireKey \
  --host 127.0.0.1 --port 30000
```

- **The memory share** is `LFM3_GMU` in `install-lfm3.sh`, 0.75. This nightly sizes its KV cache after its warmup
  and then extends it into every byte left free (`vllm/v1/worker/extensible_kv_cache.py`, which says to set a share
  on a unified-memory device). With none set, the cache grew to 5,239,455 tokens (100.85 GiB), twenty times the
  window, and the box kept 1,087 MiB available at its lowest during the campaign, close to the edge where a GB10
  freezes ([docs/gb10-memory.md](gb10-memory.md)). A share of 0.75 of the 121.63 GiB the engine sees on the
  reference box is about 91 GiB for the engine, by the arithmetic. Measured at that share on 2026-11-19, the lane as
  it ships: the box kept 20 GiB available at its lowest, the engine answered through the proxy 133 s after its start,
  and its journal gives the cache 4,567,209 tokens, 17.42 requests at the full window of 262,144 as vLLM counts it.
- **Sampling**: the checkpoint's `generation_config.json` asks temperature 0.1, top_k 50 and repetition_penalty
  1.1, and vLLM applies it to every request that sets none of its own, as it says at boot.
- **Reasoning** is on by default (the template's `enable_thinking`, true unless a request says false). A request
  turns it off with `chat_template_kwargs: {"enable_thinking": false}`, and the generation prompt then holds an
  empty thought. The template reads no effort level: the proxy turns a `reasoning_effort` into that switch (v6.61),
  `none` and `minimal` off, every other level on. Measured: the thought comes apart from the answer (the `qwen3`
  parser), and with the switch off the answer has no `<think>`.
- **Tool calls**: the model writes them in Qwen3-Coder's form (`<tool_call>`, `<function=...>`, `<parameter=...>`),
  which the `qwen3_coder` parser turns into OpenAI `tool_calls`. Measured: a weather tool called while streaming,
  with `{"city": "Paris", "unit": "celsius"}` as its arguments, and the answer after its result.
- **The window** is the model's, 262,144 tokens (its config's `max_position_embeddings`).
- **The prefix cache** is at vLLM's own setting for this hybrid model (measured below).
- **The key** reaches vLLM as on the other vLLM lanes: `engine-secrets.yaml`, read with `--config`, so it is on no
  command line, and the image prints `api_key` as `***` in its line of arguments. vLLM asks it only under `/v1`,
  `/v2`, `/inference` and `/cohere`, and the middleware asks it everywhere but `/health` and `/metrics`, as on the
  SGLang lanes ([docs/kolibri-lane.md](kolibri-lane.md) says why).
- **Usage statistics**: vLLM reports anonymous usage to its authors by default, and the unit turns that off
  (`VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`).
- `--enable-prompt-tokens-details`: each answer says how much of its prompt the prefix cache served.

## Measured on the reference box

All in the "base" campaign of 2026-11-16, under the conditions above.

| | with no memory share | at the lane's share, 0.75, 2026-11-19 |
|---|---|---|
| boot, to `/health` | 160 s at its first boot: 127.5 s of engine init (11.1 s of compile), FlashInfer's NVFP4 tuning 29 s of it | 133 s to its first reply through the proxy (read every 10 s) |
| KV cache | 5,239,455 tokens, 19.99 requests at the full window as vLLM counts it | 4,567,209 tokens, 17.42 requests, in its journal |
| MemAvailable, lowest seen | 1,087 MiB | 20 GiB |
| GSM8K, 250 questions | 0.896 | not measured yet |

**At the lane's share, 0.75, the lane as it ships** (2026-11-19, through the proxy, the GPU clock capped at 2,000
MHz): ready in 133 s after the start, 20 GiB available at its lowest, one stream 90.7 tokens/s (first token in
0.02 s), 8 at once 732.2 tokens/s in all, 101.1 per stream at the median (first token in 0.04 s), a tool call right
(`get_weather` with `{"city": "Berlin", "unit": "celsius"}`), and the prefix cache under an agent's turns
(`prefix_probe.py`, six turns, prompts of 34,941 to 35,627 tokens): nothing on the first turn, 99 % on the second,
100 % on each after it, its first token 1.77 s on the first turn then 0.12 to 0.13 s. The box's key was in none of
the 231 lines of its journal since the start.

**Speeds**, with `vllm bench serve` on the same image: random prompts of 1,024 tokens, 512 tokens out (the end of
text ignored), 4 requests one at a time, and from 8 at once up, twice as many requests as run at once:

| requests at once | 1 | 8 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| tokens/s in all | 87.0 and 87.6 (two runs) | 721.4 | 2,162.9 | 3,140.3 | 3,911.8 | 3,832.8 |
| per stream, from the median time per token | 87.7 and 87.8 | 93.4 | 71.5 | 55.2 | 35.4 | 17.9 |
| first token, median | 0.07 and 0.02 s | 0.19 s | 0.48 s | 0.33 s | 0.60 s | 1.30 s |

The total stops growing past 128 requests at once. One prompt alone, 64 tokens out: 32,768 tokens computed in
1.68 s to the first token, 131,072 in 10.24 s, 250,000 in 22.86 s (medians of two).

**At the box's maximum, with no GPU clock cap (2026-11-19, decision 1523).** Method: `vllm bench serve` inside
the lane's own container (arm "lane": the lane as the cockpit serves it, port 30000, key required) and inside a
trial container of the same image (arm "debit": 127.0.0.1:30099, no key, loopback only) tuned for throughput,
`--max-num-seqs 1024 --max-num-batched-tokens 16384 --max-model-len 4096 --gpu-memory-utilization 0.75`
(0.90 buys nothing for a 1.7B model and comes close to the unified-memory livelock). `--ignore-eos`, random
prompts, two shapes, 1,024 tokens in / 512 out and 128 / 128, concurrency tiers from 1 to 1,024, four requests
per tier of concurrency capped at 2,048, three passes (seeds 0, 1, 2), the median of the three kept. No GPU
clock cap and no thermal guard during the load (decision 1345), one engine at a time on the GPU, a probe every
2 s (MemAvailable, GPU temperature, clock, power). Results, medians of the three passes, 2026-11-19:

- one stream alone: 90.8 tokens/s out on the lane arm, 90.7 on the tuned one (1,024 in / 512 out);
- the lane as it serves: **7,042.7 tokens/s out at 256 requests at once** (shape 128/128; 14,085.3 tokens/s in
  all), 86 °C at that tier;
- the tuned container: **7,140.6 tokens/s out at 512 requests at once** (128/128; 14,281.3 in all), and
  **3,740.9** on the 1,024/512 shape at the same 512;
- the bench's temperature max: **90 °C** (on the tuned arm, 1,024/512 at 512), MemAvailable never under 18.3 GiB.

**The 1,024 tier is discarded.** On the tuned arm at 1,024 requests at once, 1,059 of the 2,048 requests failed
on every pass, client-side: `vllm bench serve` ran out of file descriptors ("Too many open files", the
`docker exec` limit), so its 7,091.7 tokens/s out is no valid figure. The tuned container's valid maximum is
512 at once, 7,140.6 tokens/s out, every request completed; the 1,024 tier would need a rerun with a higher
`ulimit -n`.

**Quality**: GSM8K, 250 questions, with `sglang.test.run_eval` from the 27B lane's SGLang image (v0.5.22) against
the test container, 8 at once, temperature 0, at most 2,048 tokens per answer, the reasoning on (the template's
default): 0.896.

**The prefix cache under an agent**: `prefix_probe.py` holds a conversation as an agent does, a long leading
system prompt, then turns, each the whole history and a new question, the reasoning off, sent to the engine's
`/v1/chat/completions` as opencode sends them, the cache's share read on its `/metrics`. Its first turn, 34,941
tokens, found nothing in the cache and its first token came after 1.73 s. Each of the five turns after it found 99
to 100 % of its 35,083 to 35,643 tokens there and computed 150 to 203, its first token after 0.11 to 0.14 s.

**Speculative decoding** is still to be tried: the model has no MTP head, and the image carries vLLM's n-gram
lookup, on the CPU and on the GPU. Still to be measured: one stream on three real jobs (code, prose, a
summary) with and without each, their acceptance, and what each costs at 8 and 64 requests at once.

## Claude Code and the prefix cache

At its start, vLLM warns about this checkpoint's template:

> The chat template requires system-first ordering, so inline system messages in /v1/messages requests (e.g.
> Claude Code's per-turn reminders) are merged into the leading system prompt. Each new one changes the prompt
> prefix, so the rest of the conversation misses the prefix cache.

This is vLLM's warning, not a measurement. Claude Code itself ran on the lane through the proxy on 2026-11-19 and
answered OK. The prefix cache under its per-turn reminders, read on the engine's `/metrics` between its turns as on
the Ornith lane ([docs/ornith-lane.md](ornith-lane.md)), and `prefix_probe.py`'s conversation sent to `/v1/messages`
with a system message at each turn, are still to be measured. opencode is not concerned: it sends
`/v1/chat/completions`.

## Clients

Through the proxy on `:30001`, with the box's key, as model `lfm3-1.7b`:

- **OpenAI** (`/v1/chat/completions`, `/v1/completions`) and **Anthropic** (`/v1/messages`) are vLLM's own.
- **Any model name works**, as on the other lanes: vLLM refuses a model it does not serve, so the proxy gives such
  a request the served name.
- **opencode** gets an `lfm3` provider, text only (the model reads no image), with two variants, `high` and
  `none`, the two states its template holds. Loading the lane makes it opencode's default, with 205,000 tokens of
  context and 32,000 of output, the flash lane's pair at the same window ([docs/opencode.md](opencode.md)). Measured on
  2026-11-20, a limit this lane ships with: opencode's file-reading task succeeded 19 times in 31 with opencode
  1.19.6, for the model sometimes asks to read the test file at an invented absolute path (`/hello.txt`, or a
  parent of the working directory), and opencode refuses a read outside its working directory
  (`external_directory`). It is the model: opencode 1.19.5, run alternately with 1.19.6 on the lane, succeeded
  once in 8 tries, against 7 for 1.19.6.
- **pi** offers `off` and `high` alone, **omp** `high` and `minimal` ([docs/pi.md](pi.md)).
- **System One** is not served here: vLLM caps `top_logprobs` on every vLLM lane, and the endpoint's option labels
  are measured on the Qwen tokenizer. The proxy says so (400), and the cockpit's Decide view offers a lane that
  serves it.

The matrix of 2026-11-19 ran the clients through the proxy, all eight OK: Claude Code, opencode, Hermes, pi, the
OpenAI and Anthropic SDKs, fetch under Node and Bun, and LiteLLM. opencode's OK there was a single try: the
targeted matrix of 2026-11-20 found it unsure (the limit above), the 7 others OK again.

## Its licence

The checkpoint is published under the LFM Open License v1.0 (its `LICENSE`). It grants its rights for commercial
use on a condition: the user, or the legal entity it belongs to, is under its Threshold, an annual revenue of 10
million US dollars or more, and any commercial use by a legal entity above it is not licensed (section 5). A
qualified non-profit organization is not held to the Threshold for non-commercial or research purposes.

## Switching to it, and back

- The cockpit: **Liquid AI LFM3** in the Lanes view, then **Load**. Its first Load pulls its image (about 11.5 GB
  to download, 28 GB on disk) and fetches its checkpoint (1.83 GB).
- A terminal: `./prepare-lane.sh lfm3` the first time (it pulls its image, no sudo), then `./switch-model.sh lfm3`
  and the commands it prints.
- Installed on demand by every plain `./install.sh` on a 128 GB box (its unit, and its first Load prepares it).
  `--with-lfm3` installs it whole now and proves it, and `--no-lfm3` leaves it out of the run that passes it (41
  GB). A 64 GB DGX Spark leaves it out: no share was measured there. `./install-lfm3.sh --uninstall` removes its
  unit, its local tag and its caches. The image and the checkpoint stay, and it names them for you to remove.
