# The NVIDIA Nemotron Transcribe lane: speech to text on the GPU, on a port of its own

NVIDIA's `Nemotron-3.5-Transcribe-30B-A4B`, in mradermacher's GGUF Q8_0 (35 GB of weights) with its F16
projector (1.4 GB), served by llama.cpp's own server on a port of its own (30036), answering
`/v1/audio/transcriptions` and the `/v1/realtime` websocket. It is a lane of its own beside the
[speech lane](speech-lane.md), which hears on the CPU beside any engine and also speaks: this one hears
on the GPU, at a word error rate the CPU lane's recognizer does not reach, and it takes the box's engine
turn to do it. It has no chat route, and the proxy does not relay it, as it relays none of the speech
lane's routes: programs on the box ask the engine on its port with the box's key.

## Why this model, and why llama.cpp

The model ships GGUF (mradermacher's quantization of NVIDIA's
`Nemotron-3.5-Transcribe-30B-A4B`), and the engine is llama.cpp's, chosen as the best engine by its
sources, not an A/B of each: the lane runs the image the llama.cpp project publishes for arm64 with
CUDA, pinned by digest, the same image as the Qwen3.8 Flash-Next Unsloth GGUF lane's (b12316), so one
image serves the two llama.cpp lanes, as the two Nemotron lanes share one. The engine flags are the ones
the measured run ran with: `--reasoning off`, because the model answers the transcription prompt
directly and its thinking tokens would be billed into the transcription, and a window of 65,536 tokens.
Nothing else: no `-np`, no cache cap, no `-ngl`.

## Installing, loading, removing

It is installed by default and prepared at its first Load, as the image lane and the video lane are:
a plain `./install.sh` puts its unit in, and its first Load pulls its image (about 4.5 GB, 2.7 to
download, already there on a box that loaded the Unsloth GGUF lane) and fetches its GGUF files (about
37 GB), then starts it. `./install.sh --with-transcribe` installs it whole now and proves it, and
`--no-transcribe` leaves it out of the run that passes it (it asks for 45 GB: 37 for its GGUF files and
8 of room for its image).
A 64 GB DGX Spark refuses it: 37 GB of weights before the engine's own memory do not hold there.

```bash
./switch-model.sh transcribe        # from a terminal: verifies its files, puts its unit in, prints what is left to run
./prepare-lane.sh transcribe        # its image alone, pulled with no sudo (the cockpit's first Load runs it)
./install-transcribe.sh --uninstall # its unit and its container; the GGUF files and the image stay, said
```

The installer ends with a proof: it starts the lane, sends it a spoken French sentence this repository
carries (`tests/fixtures/transcribe-smoke.wav`), and checks the words come back, then checks the box's
key is in none of the lane's journal lines and gives the box back to the lane it was serving.

## Calling it

Port 30036 (beside the speech lane's 30024), on loopback by default, with the box's key on every route
but `/health` and `/v1/health`. The key is the box's, read from its file, never written on a command
line:

```bash
# a WAV file, as its proof and its trial sent; convert anything else first: ffmpeg -i meeting.m4a meeting.wav
printf 'Authorization: Bearer %s\n' "$(cat ~/.config/qwen38/api-key)" \
  | curl -s http://127.0.0.1:30036/v1/audio/transcriptions -H @- -F file=@meeting.wav -F language=fr
```

`/v1/realtime` is the same engine's websocket, on the same port, with the same key. `/health` asks no
key and generates nothing: 503 while the model loads, 200 after, which is what the cockpit follows.
`--metrics` puts its Prometheus endpoint on, which the cockpit reads with the key. The proxy on :30001
and opencode are text clients and are left as they are by a switch to this lane: it answers audio on its
own port, not the text port, so no proxy ceiling and no opencode default. Another machine reaches it
through an SSH tunnel (`ssh -N -L 30036:127.0.0.1:30036 <box>`), as the speech lane's.

## Memory: it starts when no other engine serves

Its 35 GB of weights do not hold beside the 27B, which takes about 100: with nothing else serving, the
box kept 81 GiB available at ready and 76 at the lowest, and beside the 27B it found 22 GiB free. So
this lane comes up when no engine serves, and its unit says that to systemd with `Conflicts=`, like
every engine lane's: starting it stops the others, and starting another stops it, the speech lane (a
CPU lane) the only one it does not conflict with. It runs under the same 110g memory cap as every
engine lane of this box, far under what it measured. A switch to it makes it the one unit enabled at
boot; switching back to any text target gives the boot back.

## Measured on the reference box

The trial of 2026-11-14 (`~/qa-tools/etudes/nouvelles-lanes/`, run `essai-20261114-1436`): the model
loaded once, by the engine its lane would run (the pinned image, b12316), the 27B stopped, the GPU clock
capped at 2,000 MHz, a pause under 70 °C before each step, `--reasoning off` and a context of 65,536
tokens.

| | measured |
|---|---|
| ready | 15 s, 81 GiB available (76 at the lowest) |
| French WER | 2.76 % over the 20-excerpt set (291 s of audio), real time factor 0.096, 1.40 s an excerpt on average |
| English WER | 0.40 % over 10 excerpts (92 s of audio), real time factor 0.099, 0.90 s an excerpt on average |
| four requests in parallel | 8 excerpts, 124 s of audio in 6.0 s, 20.5 x real time |

The comparison is the speech lane's own set and its recognizers, measured on the same excerpts: French
WER 5.6 % for its best Qwen3-ASR 1.7B and 9.7 % for its default (the [speech lane's page](speech-lane.md)
has the table). Half the word error rate of the box's best CPU recognizer, about ten times faster than
real time.

## The pins

`check-pins.sh` asks the Hub's LFS pointers every day. The pins are sha256 and size, read from the
cache's own blobs on the reference box on 2026-11-15:

| what | pinned |
|---|---|
| the GGUF | `mradermacher/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF` at `76a736c119c734a6526a37bcd6fd2c4d3b252b03`, its `Q8_0.gguf` (35,004,647,040 bytes) |
| the projector | the same revision's `mmproj-f16.gguf` (1,468,544,320 bytes) |
| the image | `ghcr.io/ggml-org/llama.cpp@sha256:a5600a59...`, the Unsloth GGUF lane's (b12316) |

The two files together are 36.47 GB (10^9 bytes), rounded up to the 37 GB its fetch asks. The model's
name in its answers is what llama-server derives from the file: in single-model mode llama.cpp ignores
the `model` of a request (its server README), so the lane passes no `--alias`.

## What the cockpit shows

The Lanes view carries its card, **NVIDIA Nemotron Transcribe**, with Load like the other lanes on
demand. Its port and its bind are read from its own unit, not invented (`TRANSCRIBE_PORT` and
`TRANSCRIBE_BIND` move them at install). Its state is read from its own facts, as the llama.cpp text
lane's is: its container, its `/health` on its port, and its boot read in its log while it is not
healthy, the same llama-server milestones, so the card says warming-up while its 35 GB load. Its
journal is one button away, as any lane's.

## Limits

- **No chat route**: it serves no text, so no served name and no opencode provider, and the proxy does
  not relay it. Programs on the box, with the key, are its clients.
- **One recognizer at a time**: the unit passes no `-np`, one request at a time is what the measured run
  measured; parallel pieces are the client's own batch of requests (the four-in-parallel figure above is
  the trial's, not the lane's flag).
- **It takes the engine turn**: no other engine serves while it does, and the 27B comes back by a
  switch, not by itself.
- **A 64 GB DGX Spark refuses it**, before anything is fetched.
- **It answers text, it does not speak**: its routes are transcription and realtime alone; for a voice
  back, the speech lane's Kokoro.
