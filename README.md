# 🎙️ Voice Tutor

A fully local, voice-to-voice English conversation partner. Talk into your mic, and an LLM running in [LM Studio](https://lmstudio.ai/) talks back — no cloud APIs, no subscriptions, nothing leaves your machine.

It's built for **casual speaking practice**: the AI chats with you like a friend and deliberately does *not* correct your grammar or pronunciation, so you can just focus on talking.

```
Mic → Silero VAD → faster-whisper → LM Studio (streaming) → Kokoro TTS → Speakers
```

## Features

- **Natural turn-taking** — voice activity detection figures out when you've finished speaking (pause length is configurable).
- **Barge-in** — start talking while the AI is speaking and it stops mid-sentence, just like a real conversation.
- **Low latency** — LLM output is streamed and spoken sentence by sentence as it arrives, instead of waiting for the full reply.
- **Reasoning-model support** — handles `<think>` blocks and `reasoning_content` from thinking models, optionally shows the reasoning in the terminal, and says a short filler ("Hmm, let me think.") if the model takes a while.
- **Whisper hallucination filter** — drops the phantom "Thank you." / "you" transcriptions Whisper tends to produce on silence or breathing.
- **Multiple personas** — a friend, an ML-engineer coworker, or a backpacker at a hostel.
- **Transcript logging** — optionally save every conversation as JSONL.
- **Everything configurable from the CLI.**

## Requirements

- Python 3.10+
- An **NVIDIA GPU with CUDA** (Whisper runs on `cuda`)
- [LM Studio](https://lmstudio.ai/) with a chat model downloaded
- A microphone and speakers (headphones recommended — see [Tips](#tips))

## Installation

```bash
git clone https://github.com/RunzeWater/voice-tutor.git
cd voice-tutor
pip install numpy sounddevice torch faster-whisper kokoro openai silero-vad
```

Install the CUDA build of PyTorch that matches your system from [pytorch.org](https://pytorch.org/get-started/locally/) if the default one doesn't pick up your GPU.

**Windows:** faster-whisper needs cuBLAS and cuDNN. The easiest way is:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

The script automatically adds these to the DLL search path, so no manual PATH setup is needed.

The Whisper and Kokoro models are downloaded automatically on first run.

## Usage

1. Open LM Studio, load a model, and start the local server (default `http://localhost:1234`).
2. Run:

```bash
python voice_tutor.py
```

The AI greets you first. Once you see `🎙️ Your turn`, just start talking. Press `Ctrl+C` to quit.

### Examples

```bash
# Use a different model and persona
python voice_tutor.py --model qwen3-8b --persona coworker

# Give yourself more time to think before your turn ends
python voice_tutor.py --silence 1200

# Pick a specific microphone
python voice_tutor.py --list-devices
python voice_tutor.py --device 2

# Save transcripts and show the model's reasoning
python voice_tutor.py --transcript logs --show-thinking
```

### Command-line options

| Option | Default | Description |
|---|---|---|
| `--model` | `gemma-4-12b` | LM Studio model identifier |
| `--url` | `http://localhost:1234/v1` | LM Studio server URL |
| `--persona` | `friend` | `friend`, `coworker`, or `traveler` |
| `--voice` | `af_heart` | Kokoro voice (e.g. `af_heart`, `am_michael`) |
| `--silence` | `750` | Pause length (ms) that ends your turn |
| `--max-tokens` | `2048` | Token budget per reply, including thinking |
| `--show-thinking` | off | Print the model's reasoning (dimmed) |
| `--no-filler` | off | Don't say "Hmm..." while the model thinks |
| `--no-barge-in` | off | Disable interrupting the AI |
| `--device` | system default | Input device index |
| `--list-devices` | — | List audio devices and exit |
| `--transcript DIR` | off | Save the conversation as JSONL in `DIR` |
| `--whisper` | `large-v3-turbo` | faster-whisper model size |
| `--compute` | `int8_float16` | Whisper compute type (`int8_float16` or `float16`) |
| `-v`, `--verbose` | off | Debug logging |

Less common settings (VAD threshold, barge-in sensitivity, history length, temperature, etc.) live in the `Config` dataclass at the top of the script.

## Personas

| Persona | Character | Vibe |
|---|---|---|
| `friend` | Emma | A friendly, curious person having a casual chat |
| `coworker` | Sam | A senior ML engineer talking shop at lunch |
| `traveler` | Alex | A backpacker swapping stories at a hostel |

To add your own, add an entry to the `PERSONAS` dictionary. The shared `RULES` prompt keeps replies short, spoken-style, and correction-free.

## How it works

1. **Listening** — The mic stream is split into 32 ms chunks. Silero VAD marks each as speech or silence. A short pre-roll buffer keeps the start of your first word from being clipped, and very short blips are ignored.
2. **Transcription** — Once you pause, the utterance goes to faster-whisper. Short clips that match common hallucinations are thrown away.
3. **Thinking** — Your text is added to a rolling chat history and sent to LM Studio through its OpenAI-compatible API with streaming on. Reasoning tokens are separated from the actual reply, even when a `<think>` tag is split across chunks.
4. **Speaking** — Each complete sentence is queued to Kokoro TTS on a background thread and played immediately while the rest of the reply is still generating.
5. **Barge-in** — While the AI speaks, a watcher thread keeps running VAD on the mic. If you talk for ~400 ms, playback and the LLM stream stop, and the audio you already said becomes the start of your next turn. The AI's cut-off reply is saved with an `[interrupted]` marker so it doesn't repeat itself.

## Tips

- **Use headphones.** With speakers, the mic can pick up the AI's own voice and trigger a false barge-in. If you must use speakers, try `--no-barge-in`.
- **Thinking models:** if you get warnings about running out of tokens, raise `--max-tokens`, or use a non-reasoning model for faster replies.
- **Low VRAM:** try a smaller Whisper model, e.g. `--whisper small.en` or `--whisper distil-large-v3`.
- **Getting cut off too early?** Increase `--silence`.

## Transcript format

With `--transcript`, each line in `chat_YYYYMMDD_HHMMSS.jsonl` looks like:

```json
{"t": 1727049600.12, "role": "user", "text": "I just finished my midterm today."}
```

## Acknowledgements

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Kokoro TTS](https://github.com/hexgrad/kokoro)
- [LM Studio](https://lmstudio.ai/)

## License

MIT
