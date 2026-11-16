# The Liquid AI LFM2.5 24B-A2B lane: Liquid AI's mixture of experts and its draft on SGLang

LFM2.5 24B-A2B is Liquid AI's hybrid mixture of experts: 24B parameters with about 2B active per token, 40 layers
(30 short convolutions, 10 attention over the whole context), 64 experts of which 4 per token, a vocabulary of
128,000 tokens and a native window of 131,072 tokens. This lane serves Liquid AI's own FP8 build of it,
`LiquidAI/LFM2.5-24B-A2B-FP8`, with Liquid AI's own speculative draft for it, `LiquidAI/LFM2.5-24B-A2B-DSpark`, on
the same port and behind the same proxy and key as the other text lanes, as model `lfm2.5-24b-a2b`.

What is measured of it so far is one short trial, on the reference box (an ASUS Ascent GX10, 128 GB) on 2026-11-14:
the pair loaded once in the 27B lane's SGLang image with the command its draft's card gives, at the context length
and the memory share the lane keeps, the GPU clock capped at 2,000 MHz and the chip under 70 °C before each step,
the 27B stopped. That trial ran without the tool-call and reasoning parsers the lane passes, and the lane's own unit
has not served on the box yet. A full campaign with the lane's flags is queued: each place it will fill says
`MESURE-A-VENIR`, and what no campaign measures yet is said so where it stands.

## Why SGLang, and the 27B lane's image

Liquid AI's draft card gives SGLang's command for this pair, and SGLang's DSpark speculative decoding runs it. The
27B lane's own image (SGLang v0.5.22, the same digest) served it in the trial, so the lane pulls nothing more on a
box that serves the 27B: the image is shared, and the lane adds its two checkpoints alone.

The checkpoint is Liquid AI's FP8: its linear weights in FP8 by blocks of 128 x 128 with their activations
quantized as they come, its embeddings, its experts' routing gates and its convolutions left in their own precision,
as its card says. The draft is 327.7M parameters (BF16, five attention layers, by its card), proposing blocks of 9
tokens.

## The serving command

`qwen38-lfm25.service` runs, in the 27B lane's image:

```
python3 -m sglang.launch_server --model-path LiquidAI/LFM2.5-24B-A2B-FP8 --revision <pinned sha> \
  --served-model-name lfm2.5-24b-a2b --context-length 131072 --mem-fraction-static 0.7 --max-running-requests 48 \
  --speculative-algorithm DSPARK --speculative-draft-model-path LiquidAI/LFM2.5-24B-A2B-DSpark \
  --speculative-draft-model-revision <pinned sha> --speculative-dspark-block-size 9 --speculative-num-draft-tokens 10 \
  --sleep-on-idle --enable-metrics --enable-cache-report --reasoning-parser deepseek-r1 --tool-call-parser lfm2 \
  --config /out/engine-secrets.yaml --host 127.0.0.1 --port 30000
```

- **The context length and the memory share**, 131,072 tokens and 0.7, are the trial's two numbers.
  `--max-running-requests 48` is the number SGLang set by itself with the draft in the trial ("Max running requests
  is reset to 48 for speculative decoding"), written down so that no later default moves it.
- **The draft**: `--speculative-dspark-block-size 9` and `--speculative-num-draft-tokens 10` are the card's.
- **Reasoning, always on.** The checkpoint's template opens a thought itself at the end of every generation prompt
  (`<think>`), and reads no switch and no level: neither `enable_thinking` nor a `reasoning_effort` changes what the
  model does. So the model writes its thought and then `</think>` with no opening tag. SGLang's `deepseek-r1` parser
  takes every answer as begun in a thought, as this template makes it, and that is why the lane passes it
  (`LFM25_REASONING_PARSER` in `install-lfm25.sh`, one line). It was chosen by reading the parser's source in the
  image (2026-11-16): the trial ran with no parser, and its reasoning came back inside the answer.
  MESURE-A-VENIR(lfm25-clients): the parser with the clients through the proxy.
- **Tool calls**: the model writes them in LFM2's own form, Python calls between `<|tool_call_start|>` and
  `<|tool_call_end|>`, and SGLang's `lfm2` parser turns them into OpenAI `tool_calls` (read in the image,
  2026-11-15). MESURE-A-VENIR(lfm25-base): a streamed call and the turn with its result.
- **Sampling**: the checkpoint's `generation_config.json` asks temperature 0.1, top_k 50 and repetition_penalty 1.05,
  which SGLang applies to a request that sets none.
- **What every SGLang lane of this box passes besides**, as the 27B's unit says why: the health route that generates
  nothing, the rendezvous through a file, gloo and NCCL on loopback, the request header overrides the proxy's request
  ids need, the offline cache, the scheduler parked when idle, the metrics and the cache report, the key by
  `--config`.

## Measured on the reference box

The trial of 2026-11-14, under the conditions above:

| | the trial, no parser | the lane's flags |
|---|---|---|
| ready, its `/v1/models` answering | 286 s | MESURE-A-VENIR(lfm25-base) |
| KV cache | 1,667,049 tokens (BF16), the window 12.7 times | MESURE-A-VENIR(lfm25-base) |
| MemAvailable | 28 GiB once ready, 24 GiB at its lowest | MESURE-A-VENIR(lfm25-base) |
| GSM8K, 250 questions | 0.912 | MESURE-A-VENIR(lfm25-base) |
| a first answer | 17 x 23 = 391, right | MESURE-A-VENIR(lfm25-base) |

**Quality**: GSM8K with `sglang.test.run_eval` from the same image, 8 at once: 0.912, where the 27B lane scores
99.2 % on the same 250 questions (2026-11-05, [BENCHMARKS.md](../BENCHMARKS.md)). The draft's accepted length,
read in the server's log while it ran, 183 readings: 5.28 tokens at the median (5.36 on average over the 250
requests, as `run_eval` reads it from each answer).

**Speed**: only what the server's log said while GSM8K ran, 8 requests at once: 362 tokens/s in all at the median of
its 183 readings, 443 at the most. MESURE-A-VENIR(lfm25-base): one stream alone, 8, 32 and 48 at once (1,024 tokens
in, 512 out), a prompt of 32,768 and one of 120,000 tokens, and the prefix cache under an agent's turns.

## Clients

Through the proxy on `:30001`, with the box's key, as model `lfm2.5-24b-a2b`:

- **OpenAI** and **Anthropic** routes, SGLang's own, as on the 27B lane.
- **opencode** gets an `lfm25` provider, text only (the model reads no image), with no variant, as the template reads
  no level, and 74,000 tokens of context and 32,000 of output at its window of 131,072 ([docs/opencode.md](opencode.md)).
  Loading the lane makes it opencode's default.
- **pi** offers `high` alone and sends no effort field, **omp** gets `high` as its one effort
  ([docs/pi.md](pi.md)). Read in their sources: no pi or omp session has run on the lane yet.
- **System One** is not served here (proxy v6.60): its template opens a thought at every answer, so the one token
  System One reads would be the thought's. The proxy says so (400), with the lanes that serve it.

None of these clients has run on the lane through the proxy yet. The installer's proof does it with a reply and a
tool call, and the clients themselves are still to be run: MESURE-A-VENIR(lfm25-clients).

## Its licence

The checkpoint and the draft are published under the LFM Open License v1.0 (their `LICENSE`), whose commercial use
is licensed only below its Threshold: a legal entity with an annual revenue of 10 million US dollars or more is not
licensed for it (section 5). A qualified non-profit organization is not held to the Threshold for non-commercial or
research purposes.

## Switching to it, and back

- The cockpit: **Liquid AI LFM2.5 24B-A2B** in the Lanes view, then **Load**. Its first Load fetches its checkpoint
  and its draft (25.4 GB together) in the 27B lane's image, which it pulls first on a box that does not have it.
- A terminal: `./prepare-lane.sh lfm25` the first time, then `./switch-model.sh lfm25` and the commands it prints.
- Installed on demand by every plain `./install.sh` on a 128 GB box (its unit, and its first Load prepares it).
  `--with-lfm25` installs it whole now and proves it, and `--no-lfm25` leaves it out of the run that passes it (26
  GB beside the 27B lane, 74 without its image). A 64 GB DGX Spark leaves it out: its trial left 24 GiB available at
  its lowest on a 128 GB box, and none was made on a 64 GB one. `./install-lfm25.sh --uninstall` removes its unit.
  The checkpoints and the image stay, and it names them.
