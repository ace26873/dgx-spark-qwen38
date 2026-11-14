# MiniMax-H3-Image on a DGX Spark (images up to 16.9 megapixels)

`MiniMaxAI/MiniMax-H3-Image`, text to image and image editing in six steps, served by SGLang Diffusion
from SGLang's main branch in its own venv, its text encoder MiniMax-H3's, the video lane's.

Its licence, the MiniMax-H3-Image Community License, is not granted in the European Union, the United
Kingdom, South Korea or the United States (its Excluded Territories: definition 5 and clause V.4 of its
LICENSE).

## One command

```bash
./switch-model.sh image-h3
```

The one-line install sets up its units and its first Load builds it (84 GB beside the video lane, 151
without it); `./install.sh --with-image-h3` builds it at install and proves it.

## Measured on our box (ASUS Ascent GX10, 128 GB)

On 2026-11-11, at the lane's SGLang commit, end to end: 2048 x 2048 in **15.6 s**, 4736 x 3552 in
**81.1 s**, an edit at 2048 x 2048 with a 1024-pixel reference in 23.6 s. It holds 92.3 GB at its peak:
it runs alone on the box.

## In the repo

- [docs/image-h3-lane.md](../image-h3-lane.md), the lane page: its runtime, its unit, its numbers
- [docs/image-lane.md](../image-lane.md), the Qwen-Image 2.1 lane, the image lane a plain install serves first
