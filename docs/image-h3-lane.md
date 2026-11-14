# The MiniMax-H3-Image lane

MiniMax's `MiniMaxAI/MiniMax-H3-Image`, text to image and image editing in six steps, from 2048 x 2048
to about 16.9 megapixels, served by SGLang Diffusion in a venv of its own on `127.0.0.1:30026`, beside
the Qwen-Image 2.1 lane, which stays the image lane a plain install serves first.

Its licence, the MiniMax-H3-Image Community License, is not granted in the European Union, the United
Kingdom, South Korea or the United States (its Excluded Territories: definition 5 and clause V.4 of its
LICENSE). Elsewhere, MiniMax's written authorization is needed above USD 1 million of annual revenue
or for a service hosted for third parties, and a commercial product displays the name
"MiniMax-H3-Image" (its clauses IV.1 and IV.2). Its text encoder is MiniMax-H3's, under the MiniMax H3
Community License, with the same Excluded Territories.

## Load it

```bash
./switch-model.sh image-h3        # then the two commands it prints, or Load on MiniMax-H3-Image in the cockpit
```

A plain `./install.sh` puts in its unit, as for every lane; its first Load builds its runtime
(`./prepare-lane.sh image-h3`, as you, no sudo) and fetches its weights. `./install.sh --with-image-h3`
does it all at install and proves it serves; `--no-image-h3` leaves it out of the run that passes it;
`./install-image-h3.sh --uninstall` removes its unit and its runtime and leaves the weights.

## What it needs

- **Disk:** the checkpoint, 68.65 GB in 34 files (its transformer, VAE, fused output heads, tokenizer
  and processor: the repository's diffusers copies and its own copy of the text encoder are left out);
  its text encoder, 66.7 GB, which is MiniMax-H3's `text_encoder/` folder at the revision the video lane
  pins (42ed227e), the same 23 files as the video lane's partitions, so nothing is downloaded for it on a
  box whose video lane has its weights; the runtime, about 15 GB with its build (TODO(measure) on the
  reference box). 84 GB beside the video lane, 151 without it. The installer and the switch check the
  room before the first byte.
- **Memory:** the engine peaked at 92.3 GB on the reference box, and the recipe asks for 106 GiB
  available at its start (below that, the runtime offloads components and runs slower). It fits beside
  no other lane: its unit's `Conflicts=` stops the lane that serves, and a 64 GB DGX Spark is refused.
- **The runtime:** the `sglang[diffusion]` 0.5.23 wheel, whose torch (2.14.1), sglang-kernel (0.4.10),
  transformers (5.19.0) and diffusers (0.37.0) are the ones SGLang's main asks for, then SGLang's main
  at `49f1a44ee348` (the support for this model, PR 49552, is in no release yet), then the kernel
  packages of SGLang's DGX Spark recipe: flash-attn-4 4.0.0b38 and FlashInfer 0.7.2's cubins and its
  jit-cache package for CUDA 13.0, each FlashInfer wheel checked by the sha256 its release publishes.

## The unit

`qwen38-image-h3.service`, loopback only (the runtime has no API key, and at this commit its scheduler's
socket follows an explicit address: the cockpit is the door). No local patch to the source: what the
image and video lanes patch is a flag of the runtime at this commit, and the unit passes it.

- `--sleep-on-idle`: the scheduler waits on its socket while idle. Without it the server held one CPU
  core at rest (104 % on the reference box, its hottest zone never below 60 C).
- `--http-graceful-shutdown-timeout 5`: a stop cancels the generation in flight after 5 s.
- `--text-encoder-path`: MiniMax-H3's `text_encoder/` folder at its pinned revision, by its path.
- `--output-path ""` and `--input-save-path ""`: nothing written under its working directory.

## Measured on the reference box

At this lane's SGLang commit, in the container of SGLang's DGX Spark recipe (study G8, 2026-11-11),
end to end, one image at a time:

| request | time |
|---|---|
| 2048 x 2048 | 15.6 s (15.2 to 16.1, five images) |
| 4736 x 3552 | 81.1 s (81.0 to 82.4) |
| 6272 x 2688 | 82.3 s |
| edit 2048 x 2048, one reference of 1024 pixels | 23.6 s (37.6 s with a reference of 2048) |
| start, weights from the external SSD | 451 s |

The hottest zone reached 95 to 97.5 C without a GPU clock cap, and 88 to 93 C at 2,000 MHz for 8 % more
time (study G4b). TODO(measure): the same from the lane's own venv and unit, its memory at the peak
without the recipe's container limit, and its CPU at rest.

## In the cockpit

The Image view addresses the image lane that serves, else the one installed (Qwen-Image 2.1 first),
and takes its rules: for MiniMax-H3-Image its sizes up to 6272 x 2688 (the largest measured; the server
refuses more), one image per call, its six steps, no background, guidance or negative prompt, up to ten
references, and its licence under the Generate button. A size under 2048 x 2048 is made at that area and
resized, so it costs what 2048 x 2048 does.
