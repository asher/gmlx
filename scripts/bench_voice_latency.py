#!/usr/bin/env python
"""Time a voice turn of ``gmlx talk``: from the end of the user's speech
to the first audio of the reply, and how fast the reply's speech comes.

    python scripts/bench_voice_latency.py serve --base-url http://127.0.0.1:8080/v1 \\
        --clip question1.wav --clip question2.wav
    python scripts/bench_voice_latency.py stages MODEL.gguf --mmproj PROJECTOR.gguf \\
        --clip question1.wav

``serve`` runs a conversation against a running server with talk's own
client code and a stopwatch. The clips are the user's turns, in order,
so the first clip is the first turn and the others are later turns. It
times each of these, when the server can run it:

  one model     The chat model transcribes the clip, then hears the clip
                and speaks the reply (talk's one-model mode).
  text first    As above with the ``male`` voice: the reply's text comes
                first and one speech pass reads it.
  typed         As ``one model``, but the reply request carries the
                transcript and no clip. The difference is what the clip
                costs the reply.
  services      Today's pipeline: the server's transcription service, a
                text reply, and the speech service on the first sentence.

A turn's stages:

  endpointer    Trailing silence talk waits for before it ends the
                utterance. Computed from the clip and talk's endpointer,
                with no clock.
  ASR           The transcription request.
  first text    From the reply request to its first text.
  first audio   From the reply request to its first audio.
  range         The fastest and slowest first audio of the timed runs.
  to audio      endpointer + ASR + first audio: the end of the user's
                speech to the start of the reply's.
  speed         Seconds of speech made per second, after the first audio.

``--beside N`` decodes N text replies during each turn. ``--prefill-tokens
N`` starts a request with a prompt of about N tokens just before each
reply, to time a reply that arrives while a neighbor's prompt is read.

``stages`` loads the model in process and splits the reply's start into
the audio features, the audio encoder, the prompt, the first speech frame
and the first audio chunk.

Each number is the median of ``--rounds`` runs after one untimed run. A
round runs every case, and odd rounds run them in the other order.
``--cooldown`` rests after each conversation. Run it on an idle machine.
Writes JSON, with every run, and a markdown table under ``--out``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import statistics
import sys
import threading
import time

import numpy as np

SPEECH_RATE = 24000
FILLER = "The quick brown fox jumps over the lazy dog near the old stone bridge. "


def read_clip(path: str, rate: int) -> np.ndarray:
    """A clip as mono int16 samples at ``rate``."""
    from gmlx.serve.media_programs import decode_mono

    samples = decode_mono(path, rate)
    return (np.clip(np.asarray(samples, np.float32), -1.0, 1.0)
            * 32767.0).astype(np.int16)


def endpointer_ms(clip: np.ndarray, silence_ms: float) -> float:
    """The time from the clip's last speech frame to the frame at which
    talk's endpointer ends the utterance."""
    from gmlx.talk.audio import FRAME_SAMPLES, RATE, Endpointer, EnergyVAD

    vad, ep = EnergyVAD(), Endpointer(silence_ms=silence_ms)
    frame_ms = 1000.0 * FRAME_SAMPLES / RATE
    padded = np.concatenate([clip, np.zeros(RATE * 3, np.int16)])
    last_speech = ended = None
    for i in range(0, len(padded) - FRAME_SAMPLES, FRAME_SAMPLES):
        frame = padded[i:i + FRAME_SAMPLES]
        prob = vad.prob(frame)
        if prob >= 0.5:
            last_speech = i // FRAME_SAMPLES
        event = ep.feed(frame, prob)
        if event and event[0] == "end":
            ended = i // FRAME_SAMPLES
            break
    if last_speech is None or ended is None:
        raise SystemExit("error: talk's endpointer found no utterance in "
                         "the clip")
    return (ended - last_speech) * frame_ms


class Reply:
    """Stopwatch over one reply's events."""

    def __init__(self):
        self.start = time.perf_counter()
        self.first_text = self.first_audio = self.end = None
        self.samples = 0
        self.first_samples = 0
        self.text: list = []

    def text_at(self, text: str) -> None:
        if self.first_text is None:
            self.first_text = time.perf_counter()
        self.text.append(text)

    def audio_at(self, samples: int) -> None:
        if self.first_audio is None:
            self.first_audio = time.perf_counter()
            self.first_samples = samples
        self.samples += samples

    def done(self) -> dict:
        self.end = time.perf_counter()

        def since(t):
            return None if t is None else round(1000.0 * (t - self.start), 1)

        seconds = self.samples / SPEECH_RATE
        after = (self.samples - self.first_samples) / SPEECH_RATE
        speed = None
        if self.first_audio is not None and after > 0:
            speed = round(after / max(self.end - self.first_audio, 1e-9), 2)
        return {"first_text_ms": since(self.first_text),
                "first_audio_ms": since(self.first_audio),
                "total_ms": since(self.end), "audio_s": round(seconds, 2),
                "speed": speed, "text": "".join(self.text).strip()}


def brain_reply(events) -> dict:
    reply = Reply()
    for event in events:
        if event[0] == "say":
            reply.text_at(event[1])
        elif event[0] == "audio":
            reply.audio_at(len(event[1]))
    return reply.done()


def service_reply(tc, base: str, brain, text: str, voice) -> dict:
    """Today's pipeline: a text reply, and the speech service on its first
    sentence. The first audio is when that request returns."""
    reply, chunker, rest = Reply(), tc.SentenceChunker(), []
    for event in brain.turn(text):
        if event[0] != "say":
            continue
        reply.text_at(event[1])
        for chunk in chunker.feed(event[1]):
            if reply.first_audio is None:
                pcm, rate = tc.synthesize(base, chunk, voice=voice)
                reply.audio_at(int(len(pcm) * SPEECH_RATE / rate))
            else:
                rest.append(chunk)
    tail = chunker.flush()
    if tail:
        rest.append(tail)
    if reply.first_audio is None and rest:
        pcm, rate = tc.synthesize(base, rest.pop(0), voice=voice)
        reply.audio_at(int(len(pcm) * SPEECH_RATE / rate))
    out = reply.done()
    # The rest of the reply is spoken while the first sentence plays.
    out["speed"] = None
    return out


class Neighbors:
    """Other requests on the server during a turn."""

    def __init__(self, tc, base: str, model: str, beside: int,
                 prefill_tokens: int):
        self.tc, self.base, self.model = tc, base, model
        self.beside, self.prefill_tokens = beside, prefill_tokens
        self.threads: list = []
        self.stop = threading.Event()

    def _decode(self, n: int, decoding: threading.Event) -> None:
        deltas = self.tc.stream_chat(
            self.base, model=self.model, max_tokens=2000, messages=[{
                "role": "user",
                "content": f"Write a long story about a lighthouse. Part {n}."}])
        try:
            for _ in deltas:
                decoding.set()
                if self.stop.is_set():
                    break
        finally:
            decoding.set()
            deltas.close()

    def _prefill(self) -> None:
        # About 15 tokens a sentence.
        prompt = FILLER * max(1, self.prefill_tokens // 15)
        deltas = self.tc.stream_chat(
            self.base, model=self.model, max_tokens=1, messages=[{
                "role": "user", "content": prompt + "\nReply with OK."}])
        for _ in deltas:
            pass

    def start_decoders(self) -> None:
        self.stop.clear()
        started = []
        for n in range(self.beside):
            started.append(threading.Event())
            t = threading.Thread(target=self._decode, args=(n, started[-1]),
                                 daemon=True)
            t.start()
            self.threads.append(t)
        for event in started:     # every neighbor is decoding
            event.wait(600.0)

    def start_prefill(self) -> None:
        if self.prefill_tokens:
            t = threading.Thread(target=self._prefill, daemon=True)
            t.start()
            self.threads.append(t)
            # A server says nothing while it reads a prompt, so this is
            # a fixed head start for the neighbor's request.
            time.sleep(0.3)

    def finish(self) -> None:
        self.stop.set()
        for t in self.threads:
            t.join(600.0)
        self.threads.clear()


def run_case(case: str, tc, base: str, model: str, clips: list, args,
             neighbors: Neighbors) -> list:
    """One conversation: a row of numbers per turn."""
    from gmlx.talk.audio import RATE

    language = args.language
    if case == "services":
        brain = tc.ServerChatBrain(base_url=base, model=model,
                                   max_tokens=args.max_tokens)
    else:
        voice = "male" if case == "text first" else None
        brain = tc.ServerChatBrain(
            base_url=base, model=model, max_tokens=args.max_tokens,
            speech=tc.speech_request(voice, language))
    rows = []
    for turn, clip in enumerate(clips):
        wav = tc.encode_wav(clip, RATE)
        neighbors.start_decoders()
        tic = time.perf_counter()
        transcript = tc.transcribe_wav(
            base, wav, language=language,
            **({} if case == "services" else {"model": model}))
        asr_ms = 1000.0 * (time.perf_counter() - tic)
        neighbors.start_prefill()
        if case == "services":
            row = service_reply(tc, base, brain, transcript, args.voice)
        elif case == "typed":
            row = brain_reply(brain.turn(transcript))
        else:
            row = brain_reply(brain.turn_audio(wav, transcript))
        neighbors.finish()
        rows.append({"turn": turn + 1, "asr_ms": round(asr_ms, 1),
                     "transcript": transcript, **row})
    return rows


def services(tc, base: str, api_key) -> dict:
    """What the server's transcription and speech services run on."""
    listed = tc._http_get_json(base + "/models", api_key=api_key)
    return {kind: next((str(e.get("alias_of") or e.get("id"))
                        for e in listed.get("data") or [] if e.get(kind)),
                       None)
            for kind in ("stt", "tts")}


def median(values: list):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 1) if values else None


def cmd_serve(args) -> dict:
    from gmlx.talk import client as tc
    from gmlx.talk.audio import RATE

    base = tc.ensure_v1_base(args.base_url)
    caps = tc.probe_capabilities(base, args.api_key)
    model = args.model or caps.get("default") or (caps["chat_ids"] or [None])[0]
    if not model:
        raise SystemExit("error: name a served model with --model")
    alone = tc.one_model(caps, model)
    cases = []
    if alone:
        cases += ["one model", "text first", "typed"]
    if caps["stt"] and caps["tts"]:
        cases.append("services")
    if args.case:
        missing = [c for c in args.case if c not in cases]
        if missing:
            raise SystemExit(f"error: this server cannot run: "
                             f"{', '.join(missing)}")
        cases = args.case
    if not cases:
        raise SystemExit("error: the model does not hear and speak, and the "
                         "server has no stt and tts services")
    clips = [read_clip(path, RATE) for path in args.clip]
    waits = [endpointer_ms(clip, args.silence_ms) for clip in clips]
    neighbors = Neighbors(tc, base, model, args.beside, args.prefill_tokens)
    result = {"model": model, "modalities": caps["modalities"].get(model),
              "services": services(tc, base, args.api_key),
              "clips": [{"path": p, "seconds": round(len(c) / RATE, 2),
                         "endpointer_ms": w}
                        for p, c, w in zip(args.clip, clips, waits)],
              "beside": args.beside, "prefill_tokens": args.prefill_tokens,
              "cases": {}}
    print(f"[bench] {', '.join(cases)}: 1 untimed run each, then "
          f"{args.rounds} rounds", file=sys.stderr, flush=True)
    for case in cases:
        run_case(case, tc, base, model, clips, args, neighbors)
    every: dict = {case: [] for case in cases}
    for n in range(args.rounds):
        # Each round runs every case, in the other order on odd rounds, so
        # a clock that drifts moves all cases alike.
        for case in (cases if n % 2 == 0 else cases[::-1]):
            every[case].append(
                run_case(case, tc, base, model, clips, args, neighbors))
            if args.cooldown:
                time.sleep(args.cooldown)
    for case in cases:
        runs = every[case]
        turns = []
        for i, wait in enumerate(waits):
            rows = [run[i] for run in runs]
            firsts = [r["first_audio_ms"] for r in rows
                      if r["first_audio_ms"] is not None]
            first = median(firsts)
            asr = median([r["asr_ms"] for r in rows])
            turns.append({
                "turn": i + 1, "endpointer_ms": wait, "asr_ms": asr,
                "first_text_ms": median([r["first_text_ms"] for r in rows]),
                "first_audio_ms": first,
                "first_audio_range_ms": ([min(firsts), max(firsts)]
                                         if firsts else None),
                "to_audio_ms": (None if first is None or asr is None
                                else round(wait + asr + first, 1)),
                "speed": median([r["speed"] for r in rows]),
                "audio_s": median([r["audio_s"] for r in rows]),
                "said": rows[-1]["text"], "heard": rows[-1]["transcript"],
                "runs": rows})
        result["cases"][case] = turns
    return result


def serve_table(result: dict) -> str:
    lines = ["| Case | Turn | Endpointer ms | ASR ms | First text ms | "
             "First audio ms | Range ms | To audio ms | Speed | Reply s |",
             "|---|---|---|---|---|---|---|---|---|---|"]

    def cell(value):
        if isinstance(value, list):
            return f"{value[0]:.0f} to {value[1]:.0f}"
        return "" if value is None else f"{value:g}"

    for case, turns in result["cases"].items():
        for t in turns:
            lines.append("| " + " | ".join([
                case, str(t["turn"]), cell(t["endpointer_ms"]),
                cell(t["asr_ms"]), cell(t["first_text_ms"]),
                cell(t["first_audio_ms"]), cell(t["first_audio_range_ms"]),
                cell(t["to_audio_ms"]),
                cell(t["speed"]), cell(t["audio_s"])]) + " |")
    lines.append("")
    lines.append(f"Model {result['model']}. Transcription service: "
                 f"{result['services']['stt'] or 'none'}. Speech service: "
                 f"{result['services']['tts'] or 'none'}.")
    if result["beside"] or result["prefill_tokens"]:
        lines.append(f"During each turn: {result['beside']} text replies "
                     f"decoding, and a prompt of about "
                     f"{result['prefill_tokens']} tokens before the reply.")
    return "\n".join(lines)


def cmd_stages(args) -> dict:
    import mlx.core as mx

    from gmlx._vendor.mlx_vlm.prompt_utils import apply_chat_template
    from gmlx._vendor.mlx_vlm.utils import (prepare_inputs,
                                            should_add_special_tokens)
    from gmlx.load.vlm import load_vlm_model
    from gmlx.models.lfm3 import speak

    model, config, processor, *_ = load_vlm_model(args.gguf, args.mmproj)
    if getattr(model, "speech_capabilities", lambda: None)() is None:
        raise SystemExit("error: gmlx has no speech output for this model")
    rate = processor.feature_extractor.sampling_rate
    out = {"clips": []}
    for path in args.clip:
        clip = read_clip(path, rate).astype(np.float32) / 32767.0
        prompt = str(apply_chat_template(
            processor, config,
            [{"role": "system", "content": speak.INTERLEAVED_SYSTEM},
             {"role": "user", "content": ""}], num_audios=1))
        runs = []
        for n in range(args.rounds + 1):
            tic = time.perf_counter()
            inputs = prepare_inputs(
                processor, audio=[clip], prompts=prompt,
                add_special_tokens=should_add_special_tokens(
                    model.config.model_type, processor))
            mx.eval(*[v for v in inputs.values() if isinstance(v, mx.array)])
            features = time.perf_counter() - tic
            data = {k: v for k, v in inputs.items()
                    if k not in ("input_ids", "pixel_values",
                                 "attention_mask")}
            tic = time.perf_counter()
            embeds = model.get_input_embeddings(
                inputs["input_ids"], None, **data).inputs_embeds
            mx.eval(embeds)
            encoder = time.perf_counter() - tic
            first_frame = first_chunk = summary = None
            tic = time.perf_counter()
            for event in speak.stream_speech(
                    model, processor, prompt, audio=[clip],
                    mode=speak.INTERLEAVED, max_tokens=args.max_tokens):
                now = time.perf_counter() - tic
                if first_frame is None and event.codes is not None:
                    first_frame = now
                if (first_chunk is None and event.audio is not None
                        and event.audio.size):
                    first_chunk = now
                summary = event
            assert summary is not None
            if n == 0:
                continue          # untimed
            runs.append({
                "features_ms": 1000 * features, "encoder_ms": 1000 * encoder,
                "prompt_tokens": summary.prompt_tokens,
                "prompt_ms": 1000 * summary.prompt_seconds,
                "first_frame_ms": None if first_frame is None
                else 1000 * first_frame,
                "first_chunk_ms": None if first_chunk is None
                else 1000 * first_chunk,
                "frames": summary.frames,
                "speed": (summary.frames * 0.08
                          / max(summary.generation_seconds, 1e-9))})
        row = {key: median([r[key] for r in runs]) for key in runs[0]}
        out["clips"].append({"path": path,
                             "seconds": round(len(clip) / rate, 2), **row,
                             "runs": runs})
    return out


def stages_table(result: dict) -> str:
    lines = ["| Clip s | Prompt tokens | Audio features ms | Audio encoder ms "
             "| Prompt ms | First frame ms | First audio ms | Speed |",
             "|---|---|---|---|---|---|---|---|"]
    for c in result["clips"]:
        lines.append("| " + " | ".join(
            "" if c[k] is None else f"{c[k]:g}" for k in (
                "seconds", "prompt_tokens", "features_ms", "encoder_ms",
                "prompt_ms", "first_frame_ms", "first_chunk_ms",
                "speed")) + " |")
    lines.append("")
    lines.append("First frame and first audio count from the start of the "
                 "reply call, which reads the audio and the prompt again.")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Time a voice turn of gmlx talk.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--clip", action="append", required=True,
                       metavar="FILE", help="A spoken turn. Repeat for a "
                       "conversation of several turns.")
        p.add_argument("--rounds", type=int, default=5,
                       help="Timed runs per case (default 5).")
        p.add_argument("--max-tokens", type=int, default=600,
                       help="Reply cap (default 600).")
        p.add_argument("--out", default=None, metavar="DIR",
                       help="Folder for the JSON and the table.")

    s = sub.add_parser("serve", help="Time turns against a running server.")
    common(s)
    s.add_argument("--base-url", required=True)
    s.add_argument("--api-key", default=None)
    s.add_argument("--model", default=None,
                   help="Served model id (default: the server's default).")
    s.add_argument("--case", action="append", default=None,
                   choices=("one model", "text first", "typed", "services"),
                   help="Run only this case. Repeat for several.")
    s.add_argument("--language", default=None,
                   help="Language of the clips, as talk's --language.")
    s.add_argument("--voice", default=None,
                   help="Voice of the speech service in the services case.")
    s.add_argument("--silence-ms", type=float, default=550.0,
                   help="talk's trailing silence (default 550).")
    s.add_argument("--cooldown", type=float, default=0.0, metavar="S",
                   help="Rest after each conversation (default 0).")
    s.add_argument("--beside", type=int, default=0, metavar="N",
                   help="Text replies that decode during each turn.")
    s.add_argument("--prefill-tokens", type=int, default=0, metavar="N",
                   help="Start a prompt of about N tokens before each reply.")

    g = sub.add_parser("stages", help="Split a reply's start, in process.")
    common(g)
    g.add_argument("gguf")
    g.add_argument("--mmproj", required=True)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = cmd_serve(args) if args.cmd == "serve" else cmd_stages(args)
    table = (serve_table if args.cmd == "serve" else stages_table)(result)
    result.update(mode=args.cmd, rounds=args.rounds,
                  machine=platform.machine(),
                  macos=platform.mac_ver()[0],
                  date=dt.datetime.now().isoformat(timespec="seconds"))
    print(table)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        stem = os.path.join(args.out, f"voice-latency-{args.cmd}-{stamp}")
        with open(stem + ".json", "w") as f:
            json.dump(result, f, indent=1)
        with open(stem + ".md", "w") as f:
            f.write(table + "\n")
        print(f"[bench] wrote {stem}.json and .md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
