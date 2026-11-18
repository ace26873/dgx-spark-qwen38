# NVIDIA Nemotron Transcribe on a DGX Spark (llama.cpp)

NVIDIA's `Nemotron-3.5-Transcribe-30B-A4B`, in mradermacher's GGUF Q8_0 (35 GB of weights) with its F16
projector, served by llama.cpp's own server (the Unsloth GGUF lane's image, b12316) on a port of its own
(30036), answering `/v1/audio/transcriptions` and `/v1/realtime`. It has no chat route and the proxy does not
relay it: programs on the box ask the engine on its port with the box's key. Its 37 GB of weights do not hold
beside the 27B, so the lane comes up when no other engine serves, like every engine lane in turn.

## One command

```bash
./switch-model.sh transcribe
```

It is installed by default and prepared at its first Load, as the image lane and the video lane are.
`./install.sh --with-transcribe` installs it whole now and proves it, and `--no-transcribe` leaves it out of
the run that passes it.

## Measured on our box (ASUS Ascent GX10, 128 GB)

On 2026-11-14, on the lane's pinned image, with `--reasoning off` and a context of 65,536 tokens:

- **French WER 2.76 %** over the 20-excerpt set, against 5.6 % for the speech lane's best Qwen3-ASR 1.7B and
  9.7 % for its default, and **English WER 0.40 %**.
- **Ready in 15 s** once its files are on disk, and a **real time factor of 0.096**.

## What it is good at

Speech to text on the GPU, at a word error rate the CPU speech lane's recognizer does not reach, for whoever
needs the box's transcripts in near real time. It is a lane of its own beside the
[speech lane](../speech-lane.md), which hears on the CPU beside any engine and also speaks.

## In the repo

- [docs/speech-lane.md](../speech-lane.md), the CPU speech lane, its recognizers and the WER table its
  measurement was set against
- [install-transcribe.sh](../../install-transcribe.sh), its pins (the GGUF Q8_0, its projector, the image),
  its port and the proof its install ends with
