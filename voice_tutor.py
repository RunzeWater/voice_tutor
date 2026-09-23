"""
voice_tutor.py - Local English conversation partner

    Mic -> Silero VAD -> faster-whisper -> LM Studio (streaming) -> Kokoro TTS -> speakers

Features
  * Turn-taking with configurable end-of-turn silence
  * Barge-in: start talking while the AI speaks and it stops mid-sentence
  * Streaming LLM output, spoken sentence-by-sentence as it arrives
  * Whisper hallucination filter for silence / breathing
  * Optional JSONL transcript log
  * All settings overridable from the CLI: python voice_tutor.py --help
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import queue
import random
import re
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterator

# --- Windows: expose pip-installed cuBLAS/cuDNN to CTranslate2 (before faster_whisper) ---
if sys.platform == "win32":
    import importlib.util
    for _pkg in ("nvidia.cublas", "nvidia.cudnn"):
        _spec = importlib.util.find_spec(_pkg)
        if _spec and _spec.submodule_search_locations:
            _bin = os.path.join(list(_spec.submodule_search_locations)[0], "bin")
            if os.path.isdir(_bin):
                os.add_dll_directory(_bin)
                os.environ["PATH"] = _bin + os.pathsep + os.environ["PATH"]

import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from kokoro import KPipeline
from openai import OpenAI
from silero_vad import load_silero_vad

log = logging.getLogger("tutor")


# =============================== CONFIG ====================================
@dataclass
class Config:
    # LLM
    llm_base_url: str = "http://localhost:1234/v1"
    llm_model: str = "gemma-4-12b"
    temperature: float = 0.8
    max_tokens: int = 2048            # must cover thinking + reply for reasoning models
    show_thinking: bool = False       # print the model's reasoning (dimmed) in the terminal
    filler: bool = True               # say "Hmm..." if the model thinks for a while
    filler_after_ms: int = 1200
    max_history_turns: int = 12

    # Audio / VAD
    sample_rate: int = 16000
    chunk: int = 512                  # Silero requires 512 samples @ 16 kHz
    vad_threshold: float = 0.5
    silence_ms: int = 750             # pause length that ends your turn
    min_speech_ms: int = 300          # ignore blips shorter than this
    pre_roll_ms: int = 320            # audio kept from before speech onset
    barge_in: bool = True
    barge_in_ms: int = 400            # continuous speech needed to interrupt the AI
    input_device: int | None = None

    # ASR
    whisper_model: str = "large-v3-turbo"
    whisper_compute: str = "int8_float16"
    beam_size: int = 5

    # TTS
    tts_voice: str = "af_heart"
    tts_lang: str = "a"               # a = American, b = British
    tts_sr: int = 24000

    # Misc
    transcript_dir: str | None = None
    persona: str = "friend"

    # derived
    def chunks(self, ms: int) -> int:
        return max(1, int(ms / 1000 * self.sample_rate / self.chunk))


PERSONAS = {
    "friend": """You are Emma, a friendly, curious person having a casual spoken chat
with a university Computing Science student. Just talk like a friend would.""",
    "coworker": """You are Sam, a senior ML engineer chatting with a junior colleague at lunch.
Talk shop casually: projects, tools, tech news, career stuff.""",
    "traveler": """You are Alex, a well-travelled backpacker swapping stories with a new friend
at a hostel. Curious about their city, food, and plans.""",
}

RULES = """
Rules:
- This is SPEECH. Reply in 1-3 short sentences. No lists, no markdown, no emojis.
- Never correct the other person's English or comment on their grammar or pronunciation.
- Share your own thoughts and reactions, not just questions, and keep the conversation
  flowing with a natural follow-up question most of the time.
- Use natural, everyday spoken English.
- If a message ends with "[interrupted]", you were cut off mid-sentence; don't repeat yourself."""

# Whisper outputs these on silence / breath noise
HALLUCINATIONS = re.compile(
    r"^(thank you|thanks for watching|you|bye|okay|hmm|uh|\.|)[.!\s]*$", re.I
)
FILLERS = ["Hmm, let me think.", "Hmm.", "Good question.", "Oh, let me see."]
DIM, RESET = "\033[2m", "\033[0m"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*\s")   # trailing space -> won't split "3.5"


# =============================== AUDIO IN ==================================
class Microphone:
    """Always-open input stream feeding fixed-size chunks into a queue."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.q: queue.Queue[np.ndarray] = queue.Queue()
        self.stream = sd.InputStream(
            samplerate=cfg.sample_rate, channels=1, dtype="float32",
            blocksize=cfg.chunk, device=cfg.input_device, callback=self._cb,
        )

    def _cb(self, indata, frames, t, status):
        if status:
            log.debug("mic status: %s", status)
        self.q.put(indata[:, 0].copy())

    def __enter__(self):
        self.stream.start()
        return self

    def __exit__(self, *exc):
        self.stream.stop()
        self.stream.close()

    def drain(self):
        while not self.q.empty():
            self.q.get_nowait()

    def chunks(self, stop: threading.Event | None = None) -> Iterator[np.ndarray]:
        """Yield chunks until `stop` is set. Short timeout keeps Ctrl+C responsive on Windows."""
        while stop is None or not stop.is_set():
            try:
                c = self.q.get(timeout=0.25)
            except queue.Empty:
                continue
            if len(c) == self.cfg.chunk:
                yield c


class SpeechDetector:
    """Thin wrapper around Silero VAD."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = load_silero_vad()

    def reset(self):
        self.model.reset_states()

    def is_speech(self, chunk: np.ndarray) -> bool:
        return self.model(torch.from_numpy(chunk), self.cfg.sample_rate).item() >= self.cfg.vad_threshold


class TurnDetector:
    """Collects one user utterance from the mic: pre-roll + speech + trailing silence."""

    def __init__(self, cfg: Config, mic: Microphone, vad: SpeechDetector):
        self.cfg, self.mic, self.vad = cfg, mic, vad

    def listen(self, seed: list[np.ndarray] | None = None) -> np.ndarray:
        cfg = self.cfg
        pre_roll = collections.deque(maxlen=cfg.chunks(cfg.pre_roll_ms))
        speech: list[np.ndarray] = list(seed or [])
        in_speech, silence = bool(speech), 0
        max_silence, min_voiced = cfg.chunks(cfg.silence_ms), cfg.chunks(cfg.min_speech_ms)

        self.vad.reset()
        for chunk in self.mic.chunks():
            if self.vad.is_speech(chunk):
                if not in_speech:
                    in_speech = True
                    speech.extend(pre_roll)
                speech.append(chunk)
                silence = 0
            elif in_speech:
                speech.append(chunk)
                silence += 1
                if silence >= max_silence:
                    if len(speech) - silence >= min_voiced:
                        return np.concatenate(speech)
                    speech, in_speech, silence = [], False, 0
                    self.vad.reset()
            else:
                pre_roll.append(chunk)
        raise RuntimeError("mic stream ended")


# =============================== ASR =======================================
class Transcriber:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = WhisperModel(cfg.whisper_model, device="cuda", compute_type=cfg.whisper_compute)

    def __call__(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio, language="en", beam_size=self.cfg.beam_size,
            condition_on_previous_text=False, vad_filter=False,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        if HALLUCINATIONS.match(text) and len(audio) < self.cfg.sample_rate * 1.5:
            log.debug("dropped likely hallucination: %r", text)
            return ""
        return text


# =============================== TTS =======================================
class Speaker:
    """Background sentence queue -> Kokoro -> speakers. Playback aborts on `stop`."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.pipe = KPipeline(lang_code=cfg.tts_lang)
        self.q: queue.Queue[str | None] = queue.Queue()
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self):
        self.stop.clear()
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def say(self, sentence: str):
        if sentence.strip():
            self.q.put(sentence)

    def finish(self):
        self.q.put(None)
        if self.thread:
            self.thread.join()

    def abort(self):
        self.stop.set()
        self.finish()

    def _run(self):
        with sd.OutputStream(samplerate=self.cfg.tts_sr, channels=1, dtype="float32") as out:
            while not self.stop.is_set():
                sentence = self.q.get()
                if sentence is None:
                    break
                for _, _, audio in self.pipe(sentence, voice=self.cfg.tts_voice):
                    if audio is None:
                        continue
                    pcm = audio.cpu().numpy() if hasattr(audio, "cpu") else np.asarray(audio)
                    pcm = pcm.astype(np.float32).reshape(-1, 1)
                    for i in range(0, len(pcm), 2048):      # small blocks so abort is snappy
                        if self.stop.is_set():
                            return
                        out.write(pcm[i:i + 2048])


# =============================== LLM =======================================
class Chat:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.llm_base_url, api_key="lm-studio")
        self.history: list[dict] = [
            {"role": "system", "content": PERSONAS.get(cfg.persona, PERSONAS["friend"]) + RULES}
        ]

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        keep = self.cfg.max_history_turns * 2 + 1          # odd -> window starts on a user msg
        self.history = [self.history[0]] + self.history[1:][-keep:]

    @staticmethod
    def clean(text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        return re.sub(r"[*_#`~>]", "", text).strip()

    def _route(self, piece: str) -> tuple[str, str]:
        """Split streamed content into (thinking, speech), handling inline <think> tags
        even when a tag is split across chunks."""
        self._pending += piece
        think, say = "", ""
        while True:
            tag = THINK_CLOSE if self._in_think else THINK_OPEN
            i = self._pending.find(tag)
            if i == -1:
                keep = 0                                  # hold back a possible partial tag
                for k in range(min(len(tag) - 1, len(self._pending)), 0, -1):
                    if tag.startswith(self._pending[-k:]):
                        keep = k
                        break
                cut = len(self._pending) - keep
                out, self._pending = self._pending[:cut], self._pending[cut:]
            else:
                out, self._pending = self._pending[:i], self._pending[i + len(tag):]
            if self._in_think:
                think += out
            else:
                say += out
            if i == -1:
                return think, say
            self._in_think = not self._in_think

    def stream(self, stop: threading.Event) -> Iterator[tuple[str, str]]:
        """Yield ("think", text) chunks and ("say", sentence) items as they arrive.
        Closes the HTTP stream if `stop` is set."""
        resp = self.client.chat.completions.create(
            model=self.cfg.llm_model, messages=self.history, stream=True,
            temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens,
        )
        self._pending, self._in_think = "", False
        buf, got_text, got_reasoning, finish = "", False, False, None
        try:
            for ev in resp:
                if stop.is_set():
                    return
                if not ev.choices:
                    continue
                choice = ev.choices[0]
                finish = choice.finish_reason or finish
                delta = choice.delta
                extra = getattr(delta, "model_extra", None) or {}
                reasoning = extra.get("reasoning_content") or extra.get("reasoning") or ""

                think, say = self._route(delta.content or "")
                think = reasoning + think
                if think:
                    got_reasoning = True
                    yield "think", think

                buf += say
                while (m := SENTENCE_END.search(buf)):
                    sentence, buf = self.clean(buf[:m.end()]), buf[m.end():]
                    if sentence:
                        got_text = True
                        yield "say", sentence

            if not self._in_think:
                buf += self._pending
            if (tail := self.clean(buf)):
                got_text = True
                yield "say", tail

            if not got_text:
                if got_reasoning or finish == "length":
                    log.warning("Model ran out of tokens while thinking. Raise --max-tokens "
                                "(currently %d).", self.cfg.max_tokens)
                else:
                    log.warning("Model returned an empty reply. Check the LM Studio server log.")
        finally:
            resp.close()


# =============================== SESSION ===================================
class Session:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        log.info("Loading models...")
        self.mic = Microphone(cfg)
        self.vad = SpeechDetector(cfg)
        self.turns = TurnDetector(cfg, self.mic, self.vad)
        self.asr = Transcriber(cfg)
        self.tts = Speaker(cfg)
        self.chat = Chat(cfg)
        self.transcript = self._open_transcript()

    def _open_transcript(self):
        if not self.cfg.transcript_dir:
            return None
        d = Path(self.cfg.transcript_dir); d.mkdir(parents=True, exist_ok=True)
        return open(d / f"chat_{datetime.now():%Y%m%d_%H%M%S}.jsonl", "a", encoding="utf-8")

    def _record(self, role: str, text: str):
        if self.transcript:
            self.transcript.write(json.dumps({"t": time.time(), "role": role, "text": text}) + "\n")
            self.transcript.flush()

    # ---- barge-in watcher: runs while the AI is speaking ----
    def _watch_for_barge_in(self, stop: threading.Event, seed: list[np.ndarray]):
        need = self.cfg.chunks(self.cfg.barge_in_ms)
        run: list[np.ndarray] = []
        self.vad.reset()
        for chunk in self.mic.chunks(stop):
            if self.vad.is_speech(chunk):
                run.append(chunk)
                if len(run) >= need:
                    seed.extend(run)
                    stop.set()
                    return
            else:
                run.clear()

    def speak_turn(self) -> list[np.ndarray]:
        """Stream LLM -> TTS. Returns seed audio if the user barged in, else []."""
        stop, seed = threading.Event(), []
        self.mic.drain()
        watcher = None
        if self.cfg.barge_in:
            watcher = threading.Thread(target=self._watch_for_barge_in, args=(stop, seed), daemon=True)
            watcher.start()

        self.tts.start()
        spoken: list[str] = []
        started, filler_done, thinking_printed = time.monotonic(), False, False
        print("🤖 ", end="", flush=True)
        for kind, text in self.chat.stream(stop):
            if kind == "think":
                if self.cfg.show_thinking:
                    print(DIM + text + RESET, end="", flush=True)
                    thinking_printed = True
                waited_ms = (time.monotonic() - started) * 1000
                if self.cfg.filler and not spoken and not filler_done \
                        and waited_ms >= self.cfg.filler_after_ms:
                    self.tts.say(random.choice(FILLERS))    # not added to history
                    filler_done = True
                continue
            if thinking_printed and not spoken:
                print("\n🤖 ", end="", flush=True)
            print(text, end=" ", flush=True)
            spoken.append(text)
            self.tts.say(text)
        if stop.is_set():
            self.tts.abort()
        else:
            # LLM done; keep watching until playback finishes
            self.tts.finish()
        stop.set()
        if watcher:
            watcher.join()
        print()

        text = " ".join(spoken)
        if seed:
            text += " [interrupted]"
            print("   ⏹️  (interrupted)")
        self.chat.add("assistant", text)
        self._record("assistant", text)
        return seed

    def run(self):
        with self.mic:
            self.chat.add("user", "(The chat starts now. Say hi briefly and ask me something easy.)")
            seed = self.speak_turn()
            print("\n🎙️  Your turn (Ctrl+C to quit)")
            while True:
                text = self.asr(self.turns.listen(seed))
                seed = []
                if not text:
                    continue
                print(f"🧑 {text}")
                self.chat.add("user", text)
                self._record("user", text)
                seed = self.speak_turn()

    def close(self):
        sd.stop()
        if self.transcript:
            self.transcript.close()


# =============================== CLI =======================================
def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Local English conversation partner")
    d = Config()
    p.add_argument("--model", default=d.llm_model, help="LM Studio model identifier")
    p.add_argument("--url", default=d.llm_base_url)
    p.add_argument("--persona", choices=PERSONAS, default=d.persona)
    p.add_argument("--voice", default=d.tts_voice, help="Kokoro voice, e.g. af_heart, am_michael")
    p.add_argument("--silence", type=int, default=d.silence_ms, help="ms of pause that ends your turn")
    p.add_argument("--max-tokens", type=int, default=d.max_tokens,
                   help="token budget per reply, including thinking")
    p.add_argument("--show-thinking", action="store_true", help="print model reasoning (dimmed)")
    p.add_argument("--no-filler", action="store_true", help="don't say 'Hmm...' while the model thinks")
    p.add_argument("--no-barge-in", action="store_true", help="disable interrupting the AI")
    p.add_argument("--device", type=int, default=None, help="input device index (see --list-devices)")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--transcript", metavar="DIR", default=None, help="save chat log as JSONL")
    p.add_argument("--whisper", default=d.whisper_model)
    p.add_argument("--compute", default=d.whisper_compute, help="int8_float16 | float16")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()

    if a.list_devices:
        print(sd.query_devices()); sys.exit(0)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    if not a.verbose:  # silence noisy third-party HTTP logs
        for name in ("httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3", "faster_whisper"):
            logging.getLogger(name).setLevel(logging.WARNING)
    return Config(
        llm_model=a.model, llm_base_url=a.url, persona=a.persona, tts_voice=a.voice,
        silence_ms=a.silence, barge_in=not a.no_barge_in, max_tokens=a.max_tokens,
        show_thinking=a.show_thinking, filler=not a.no_filler, input_device=a.device,
        transcript_dir=a.transcript, whisper_model=a.whisper, whisper_compute=a.compute,
    )


def main():
    cfg = parse_args()
    log.debug("config: %s", asdict(cfg))
    session = Session(cfg)
    try:
        session.run()
    except KeyboardInterrupt:
        print("\n\nBye!")
    finally:
        session.close()


if __name__ == "__main__":
    main()