"""Audio capture/playback, VAD endpointing, and wake word for ``gmlx talk``.

Layered so everything above the hardware is pure and unit-testable:

* :class:`SoundDeviceBackend` is the only piece that touches PortAudio
  (lazy-imported ``sounddevice``); tests drive the rest with synthetic frames.
* :class:`Endpointer` turns per-frame speech probabilities into utterances -
  pre-roll so the first word isn't clipped, a trailing-silence hangover, and
  min-duration + level floors so silence/noise never reaches Whisper (which
  hallucinates text like "thank you" on empty audio).
* :class:`SherpaKwsDetector` spots the wake phrase with sherpa-onnx keyword
  spotting: any plain-text phrase, tokenized locally with sentencepiece
  (already a transformers dependency) against the model's BPE vocab - no
  training, no cloud. :class:`SileroVAD` rides the same wheel; the pure-numpy
  :class:`EnergyVAD` is the no-extra fallback and the deterministic test VAD.

Model assets (a ~15 MB KWS bundle, a ~0.6 MB VAD model) download once into
``$XDG_CACHE_HOME/gmlx/talk/`` on first use.
"""

from __future__ import annotations

import math
import os
import tarfile
import tempfile
import threading
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable

import numpy as np

RATE = 16000                      # capture rate: whisper/KWS/VAD all want 16k
FRAME_SAMPLES = 1280              # 80 ms frames
_FRAME_MS = 1000.0 * FRAME_SAMPLES / RATE

# sherpa-onnx pretrained assets (github release files, fixed versions).
_KWS_NAME = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
_KWS_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            f"kws-models/{_KWS_NAME}.tar.bz2")
_VAD_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "asr-models/silero_vad.onnx")


class TalkAudioError(RuntimeError):
    """An audio/asset problem the TUI reports as one friendly line."""


def rms_dbfs(frame: np.ndarray) -> float:
    """RMS level of int16 samples in dBFS (0 = full scale, silence ~ -90)."""
    f = np.asarray(frame, dtype=np.float64)
    rms = math.sqrt(float(np.mean(f * f))) if f.size else 0.0
    return 20.0 * math.log10(max(rms, 1e-9) / 32768.0)


# Model assets

def asset_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    d = os.path.join(base, "gmlx", "talk")
    os.makedirs(d, exist_ok=True)
    return d


def _download(url: str, dest: str) -> None:
    tmp = dest + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 (pinned https URLs)
    except (urllib.error.URLError, OSError) as e:
        raise TalkAudioError(
            f"could not download {os.path.basename(dest)} ({e}); talk needs "
            f"one-time network access for its wake-word/VAD models - or place "
            f"the file at {dest} yourself") from e
    os.replace(tmp, dest)


def ensure_vad_model() -> str:
    """Path of silero_vad.onnx, downloading it on first use."""
    dest = os.path.join(asset_dir(), "silero_vad.onnx")
    if not os.path.exists(dest):
        print("[talk] fetching VAD model (0.6 MB, one-time)...", flush=True)
        _download(_VAD_URL, dest)
    return dest


def ensure_kws_model() -> str:
    """Directory of the KWS zipformer bundle, downloading it on first use."""
    root = asset_dir()
    d = os.path.join(root, _KWS_NAME)
    if os.path.isdir(d) and os.path.exists(os.path.join(d, "tokens.txt")):
        return d
    archive = os.path.join(root, _KWS_NAME + ".tar.bz2")
    print("[talk] fetching wake-word model (15 MB, one-time)...", flush=True)
    _download(_KWS_URL, archive)
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(root, filter="data")
    os.unlink(archive)
    if not os.path.exists(os.path.join(d, "tokens.txt")):
        raise TalkAudioError(f"wake-word bundle extracted oddly under {root}")
    return d


# VAD engines: prob(int16 frame) -> speech probability (binary is fine)

class EnergyVAD:
    """Level-gate fallback (quiet rooms only): speech iff RMS above a floor."""

    name = "energy"

    def __init__(self, threshold_dbfs: float = -38.0):
        self.threshold_dbfs = threshold_dbfs

    def prob(self, frame: np.ndarray) -> float:
        return 1.0 if rms_dbfs(frame) >= self.threshold_dbfs else 0.0

    def reset(self) -> None:
        pass


class SileroVAD:
    """Silero VAD via sherpa-onnx (same wheel as the wake word)."""

    name = "silero"

    def __init__(self, threshold: float = 0.6,
                 model_path: str | None = None):
        import sherpa_onnx as so
        cfg = so.VadModelConfig()
        cfg.silero_vad.model = model_path or ensure_vad_model()
        cfg.silero_vad.threshold = float(threshold)
        cfg.sample_rate = RATE
        self._vad = so.VoiceActivityDetector(cfg, buffer_size_in_seconds=30)

    def prob(self, frame: np.ndarray) -> float:
        self._vad.accept_waveform(
            np.asarray(frame, dtype=np.float32) / 32768.0)
        speech = self._vad.is_speech_detected()
        while not self._vad.empty():      # discard segments; we only gate
            self._vad.pop()
        return 1.0 if speech else 0.0

    def reset(self) -> None:
        self._vad.reset()


# Endpointing

class Endpointer:
    """Per-frame speech probabilities -> utterance events.

    ``feed(frame, prob)`` returns ``("start", None)`` at speech onset,
    ``("end", int16 utterance)`` after ``silence_ms`` of trailing silence,
    ``("drop", reason)`` when the captured audio fails the min-duration or
    level floor (never sent to STT), else ``None``. The pre-roll deque keeps
    ``pre_roll_ms`` of audio from *before* onset so the first word survives."""

    def __init__(self, *, silence_ms: float = 550.0,
                 min_speech_ms: float = 300.0, pre_roll_ms: float = 400.0,
                 min_level_dbfs: float = -45.0,
                 frame_ms: float = _FRAME_MS):
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.min_level_dbfs = min_level_dbfs
        self.frame_ms = frame_ms
        self._pre = deque(maxlen=max(1, int(round(pre_roll_ms / frame_ms))))
        self._frames: list = []
        self._speech_ms = 0.0
        self._silence_run = 0.0
        self.capturing = False

    def reset(self) -> None:
        self._pre.clear()
        self._frames = []
        self._speech_ms = 0.0
        self._silence_run = 0.0
        self.capturing = False

    def feed(self, frame: np.ndarray, prob: float):
        speech = prob >= 0.5
        if not self.capturing:
            if not speech:
                self._pre.append(frame)
                return None
            self.capturing = True
            self._frames = list(self._pre) + [frame]
            self._pre.clear()
            self._speech_ms = self.frame_ms
            self._silence_run = 0.0
            return ("start", None)
        self._frames.append(frame)
        if speech:
            self._speech_ms += self.frame_ms
            self._silence_run = 0.0
            return None
        self._silence_run += self.frame_ms
        if self._silence_run < self.silence_ms:
            return None
        return self._finish()

    def flush(self):
        """Force the utterance to end now (push-to-talk key up). Returns the
        same ``("end", ...)`` / ``("drop", ...)`` events, or None if idle."""
        if not self.capturing:
            return None
        return self._finish()

    def _finish(self):
        utterance = np.concatenate(self._frames).astype(np.int16)
        speech_ms = self._speech_ms
        self.reset()
        if speech_ms < self.min_speech_ms:
            return ("drop", "too short")
        if rms_dbfs(utterance) < self.min_level_dbfs:
            return ("drop", "too quiet")
        return ("end", utterance)


# Wake word

class SherpaKwsDetector:
    """Open-vocabulary keyword spotting: the wake phrase is plain text."""

    def __init__(self, phrase: str, *, threshold: float = 0.5,
                 boost: float = 2.0, model_dir: str | None = None,
                 num_threads: int = 1):
        import sherpa_onnx as so
        self.name = phrase.strip()
        d = model_dir or ensure_kws_model()
        line = self._keyword_line(self.name, d, threshold, boost)
        # Unique per-construction file: a fixed shared name would let two
        # sessions constructing detectors concurrently read each other's
        # phrase. sherpa reads it during construction, so it is deleted after.
        fd, kw_file = tempfile.mkstemp(prefix="keywords-", suffix=".txt",
                                       dir=asset_dir())
        try:
            with os.fdopen(fd, "w") as f:
                f.write(line + "\n")
            self._spotter = so.KeywordSpotter(
                tokens=os.path.join(d, "tokens.txt"),
                encoder=os.path.join(
                    d, "encoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
                decoder=os.path.join(
                    d, "decoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
                joiner=os.path.join(
                    d, "joiner-epoch-12-avg-2-chunk-16-left-64.onnx"),
                keywords_file=kw_file, provider="cpu",
                num_threads=num_threads)
            self._stream = self._spotter.create_stream()
        finally:
            os.unlink(kw_file)

    @staticmethod
    def _keyword_line(phrase: str, model_dir: str, threshold: float,
                      boost: float) -> str:
        """Tokenize ``phrase`` against the model's BPE vocab -> a sherpa
        keywords-file line. sherpa's own text2token helper hard-imports
        pypinyin (CJK support), so the English BPE path is done directly."""
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(os.path.join(model_dir, "bpe.model"))
        pieces = sp.encode(phrase.upper(), out_type=str)
        known = {ln.split()[0] for ln in
                 open(os.path.join(model_dir, "tokens.txt"))}
        bad = [p for p in pieces if p not in known]
        if bad:
            raise TalkAudioError(
                f"wake phrase {phrase!r} uses sounds the wake-word model "
                f"can't spell ({' '.join(bad)}); pick an English phrase")
        # The @annotation is the text get_result returns; it may not contain
        # spaces (sherpa splits the line on them).
        tag = phrase.upper().replace(" ", "_")
        return f"{' '.join(pieces)} :{boost:g} #{threshold:g} @{tag}"

    def feed(self, frame: np.ndarray) -> bool:
        self._stream.accept_waveform(
            RATE, np.asarray(frame, dtype=np.float32) / 32768.0)
        hit = False
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            if self._spotter.get_result(self._stream):
                hit = True
                self._spotter.reset_stream(self._stream)
        return hit

    def reset(self) -> None:
        self._stream = self._spotter.create_stream()


def make_wake_detector(phrase: str, *, threshold: float = 0.5):
    """Build the wake detector, degrading gracefully: returns
    ``(detector | None, hint | None)`` - None + a printable hint when the
    [talk] extra (sherpa-onnx) is missing, so the caller falls back to vad
    mode instead of dying."""
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return None, ("wake word needs the talk extra: pip install "
                      "'gmlx[talk]' - falling back to voice-activity mode")
    try:
        return SherpaKwsDetector(phrase, threshold=threshold), None
    except TalkAudioError as e:
        return None, f"{e} - falling back to voice-activity mode"


def make_vad(threshold: float = 0.6):
    """Best available VAD: silero (sherpa-onnx) else the energy gate."""
    try:
        return SileroVAD(threshold=threshold)
    except (ImportError, TalkAudioError):
        return EnergyVAD()


# Earcons: short synthesized two-tone blips (no bundled assets, testable)

def earcon(kind: str, rate: int = 24000) -> tuple[np.ndarray, int]:
    """("wake" rising | "idle" falling) -> (int16 samples, rate)."""
    pairs = {"wake": (660.0, 990.0), "idle": (990.0, 660.0)}
    try:
        f1, f2 = pairs[kind]
    except KeyError:
        raise ValueError(f"unknown earcon {kind!r}") from None
    seg = int(rate * 0.07)
    t = np.arange(seg) / rate
    env = np.sin(np.linspace(0, math.pi, seg))          # click-free fades
    tone = np.concatenate([np.sin(2 * math.pi * f1 * t) * env,
                           np.sin(2 * math.pi * f2 * t) * env])
    return (tone * 0.25 * 32767).astype(np.int16), rate


# Hardware backend (the only sounddevice touchpoint)

def import_sounddevice():
    try:
        import sounddevice
    except ImportError as exc:
        raise ImportError(
            "voice chat requires the optional talk extra:\n"
            "    pip install 'gmlx[talk]'\n"
            "(installs sounddevice + sherpa-onnx and the stt/tts extras)"
        ) from exc
    return sounddevice


def _resolve_device(sd, spec: str | None, kind: str):
    """None | index-string | name-substring -> sounddevice device arg."""
    if spec is None or spec == "":
        return None
    s = str(spec)
    if s.lstrip("-").isdigit():
        return int(s)
    matches = [i for i, d in enumerate(sd.query_devices())
               if s.lower() in d["name"].lower()
               and d[f"max_{kind}_channels"] > 0]
    if not matches:
        raise TalkAudioError(f"no {kind} device matching {spec!r} "
                             f"(see --list-devices)")
    return matches[0]


class SoundDeviceBackend:
    """Mic capture + speaker playback over PortAudio.

    Capture: the PortAudio callback only copies each block onto ``on_frame``
    (the listener thread does all real work). Playback: blocking writes in
    ~``slice_ms`` slices, checking ``stop`` between slices - the barge-in /
    cancel latency is one slice. The output stream stays open across chunks of
    one reply (no inter-sentence clicks) and reopens only on a rate change."""

    def __init__(self, input_device: str | None = None,
                 output_device: str | None = None):
        self._sd = import_sounddevice()
        self._in_spec = input_device
        self._out_spec = output_device
        self._in_stream = None
        self._out_stream = None
        self._out_rate = None
        self._out_lock = threading.Lock()
        # Output gain, 0.0-1.0. A plain attribute any thread may set (float
        # stores are atomic); read once per playback slice so a menubar
        # volume drag lands mid-utterance, not on the next reply.
        self.gain = 1.0

    def start_input(self, on_frame: Callable[[np.ndarray], None],
                    *, rate: int = RATE,
                    frame_samples: int = FRAME_SAMPLES) -> None:
        sd = self._sd
        device = _resolve_device(sd, self._in_spec, "input")

        def callback(indata, frames, time_info, status):
            on_frame(indata[:, 0].copy())

        self._in_stream = sd.InputStream(
            samplerate=rate, channels=1, dtype="int16",
            blocksize=frame_samples, device=device, callback=callback)
        self._in_stream.start()

    def stop_input(self) -> None:
        if self._in_stream is not None:
            self._in_stream.stop()
            self._in_stream.close()
            self._in_stream = None

    def play(self, pcm: np.ndarray, rate: int, *,
             stop: threading.Event | None = None,
             slice_ms: float = 150.0) -> bool:
        """Play int16 samples; returns False if ``stop`` cut it short."""
        sd = self._sd
        with self._out_lock:
            if self._out_stream is None or self._out_rate != rate:
                self._close_output_locked()
                device = _resolve_device(sd, self._out_spec, "output")
                self._out_stream = sd.OutputStream(
                    samplerate=rate, channels=1, dtype="int16", device=device)
                self._out_stream.start()
                self._out_rate = rate
            step = max(1, int(rate * slice_ms / 1000.0))
            data = np.asarray(pcm, dtype=np.int16).reshape(-1, 1)
            for i in range(0, len(data), step):
                if stop is not None and stop.is_set():
                    self._out_stream.abort()      # drop the buffered tail too
                    self._out_stream.start()
                    return False
                chunk = data[i:i + step]
                gain = self.gain
                if gain != 1.0:
                    chunk = np.clip(chunk.astype(np.float32) * gain,
                                    -32768.0, 32767.0).astype(np.int16)
                self._out_stream.write(chunk)
            return True

    def _close_output_locked(self) -> None:
        if self._out_stream is not None:
            self._out_stream.stop()
            self._out_stream.close()
            self._out_stream = None
            self._out_rate = None

    def close(self) -> None:
        self.stop_input()
        with self._out_lock:
            self._close_output_locked()

    def describe_devices(self) -> str:
        """Human-readable device table for --list-devices and /devices."""
        sd = self._sd
        lines = []
        default_in, default_out = sd.default.device
        for i, d in enumerate(sd.query_devices()):
            roles = []
            if d["max_input_channels"] > 0:
                roles.append("in" + ("*" if i == default_in else ""))
            if d["max_output_channels"] > 0:
                roles.append("out" + ("*" if i == default_out else ""))
            lines.append(f"  {i:3d}  [{','.join(roles):7s}]  {d['name']}")
        return "\n".join(lines)
