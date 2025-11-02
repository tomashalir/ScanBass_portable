from __future__ import annotations

"""ScanBass backend with dual transcription modes."""

import io
import os
import uuid
from typing import Any, Dict, List, Tuple

import librosa
import mido
import numpy as np
import pretty_midi
import soundfile as sf
from fastapi import File, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

MAX_SECONDS = 30.0
LIGHT_TARGET_SR = 16000
BASS_LOW = 40
BASS_HIGH = 300

SCANBASS_MODE = os.getenv("SCANBASS_MODE", "light").strip().lower()

app = FastAPI(title="ScanBass backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS: Dict[str, Dict[str, Any]] = {}


class BaseBassTranscriber:
    name = "base"

    def audio_to_midi(self, y: np.ndarray, sr: int) -> pretty_midi.PrettyMIDI:
        raise NotImplementedError

    def warmup(self) -> None:
        sr = LIGHT_TARGET_SR
        silence = np.zeros(int(sr * 1.0), dtype=np.float32)
        self.audio_to_midi(silence, sr)


class LightBassTranscriber(BaseBassTranscriber):
    name = "light"

    @staticmethod
    def bandpass(y: np.ndarray, sr: int, low: int = BASS_LOW, high: int = BASS_HIGH, order: int = 4) -> np.ndarray:
        from scipy.signal import butter, sosfilt

        sos = butter(order, [low, high], btype="band", fs=sr, output="sos")
        return sosfilt(sos, y)

    def audio_to_midi(self, y: np.ndarray, sr: int) -> pretty_midi.PrettyMIDI:
        max_samples = int(MAX_SECONDS * sr)
        y = y[:max_samples]

        if sr != LIGHT_TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=LIGHT_TARGET_SR)
            sr = LIGHT_TARGET_SR

        y_bp = self.bandpass(y, sr)

        hop_length = 512
        pitches = librosa.yin(
            y_bp,
            fmin=librosa.note_to_hz("C1"),
            fmax=librosa.note_to_hz("C5"),
            sr=sr,
            frame_length=2048,
            hop_length=hop_length,
        )

        S = np.abs(librosa.stft(y_bp, n_fft=2048, hop_length=hop_length))
        rms = librosa.feature.rms(S=S)[0]
        conf = (rms - rms.min()) / (rms.max() - rms.min() + 1e-9)

        events = frames_to_events(pitches, conf, hop_length, sr, min_conf=0.15, merge_gap_s=0.12)
        return events_to_pretty_midi(events)


class HeavyBassTranscriber(BaseBassTranscriber):
    name = "heavy"

    def __init__(self) -> None:
        try:
            import torch
            import torchaudio
            import torchcrepe
            from demucs.apply import apply_model as demucs_apply_model
            from demucs.pretrained import get_model as demucs_get_model
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise RuntimeError(
                "Heavy mode requires torch, torchaudio, torchcrepe, and demucs to be installed"
            ) from exc

        self.torch = torch
        self.torchaudio = torchaudio
        self.torchcrepe = torchcrepe
        self.demucs_apply_model = demucs_apply_model
        self.demucs_get_model = demucs_get_model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.crepe_sample_rate = 16000
        self.crepe_hop_length = 160  # 10 ms at 16 kHz
        self.fmin = librosa.note_to_hz("E1")
        self.fmax = librosa.note_to_hz("C5")
        self._demucs_model = None
        self._crepe_loaded = False
        self._crepe_device: str | None = None

    def warmup(self) -> None:
        silence = np.zeros(int(self.crepe_sample_rate), dtype=np.float32)
        self.audio_to_midi(silence, self.crepe_sample_rate)

    # Lazy init helpers -------------------------------------------------
    def _ensure_demucs(self) -> None:
        if self._demucs_model is None:
            model = self.demucs_get_model("htdemucs")
            model.to(self.device)
            model.eval()
            self._demucs_model = model

    def _ensure_crepe(self) -> None:
        if not self._crepe_loaded:
            device_str = str(self.device)
            # torchcrepe.load.model(device=..., capacity=...) avoids double binding the
            # ``device`` parameter. Earlier versions passed ``device`` positionally and
            # again by keyword, triggering ``multiple values for argument 'device'`` when
            # torchcrepe reused the loader. Keeping it keyword-only ensures compatibility
            # with both torchcrepe.load and the internal torchcrepe.infer cache.
            self.torchcrepe.load.model(device=device_str, capacity="full")
            self._crepe_device = device_str
            self._crepe_loaded = True

    # Processing --------------------------------------------------------
    def audio_to_midi(self, y: np.ndarray, sr: int) -> pretty_midi.PrettyMIDI:
        torch = self.torch
        torchaudio = self.torchaudio

        mono = torch.from_numpy(y.astype(np.float32))
        if mono.ndim == 1:
            mono = mono.unsqueeze(0)
        elif mono.ndim > 2:
            mono = mono.view(1, -1)

        max_samples = int(MAX_SECONDS * sr)
        if mono.shape[-1] > max_samples:
            mono = mono[..., :max_samples]

        if sr != 44100:
            mono = torchaudio.functional.resample(mono, sr, 44100)
            sr = 44100

        if mono.shape[0] == 1:
            stereo = mono.repeat(2, 1)
        else:
            stereo = mono

        self._ensure_demucs()
        demucs_inp = stereo.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            separated = self.demucs_apply_model(self._demucs_model, demucs_inp, progress=False)

        # separated: (batch, sources, channels, time)
        separated = separated[0]
        bass_index = self._demucs_model.sources.index("bass")
        bass = separated[bass_index]
        bass_mono = bass.mean(dim=0, keepdim=True)

        if sr != self.crepe_sample_rate:
            bass_mono = torchaudio.functional.resample(bass_mono, sr, self.crepe_sample_rate)
            sr = self.crepe_sample_rate

        bass_mono = bass_mono.clamp(-1.0, 1.0)
        hop_length = self.crepe_hop_length

        self._ensure_crepe()
        with torch.inference_mode():
            f0, periodicity = self.torchcrepe.predict(
                bass_mono,
                sr,
                hop_length,
                fmin=self.fmin,
                fmax=self.fmax,
                model="full",
                batch_size=128,
                return_periodicity=True,
                device=self._crepe_device or str(self.device),
            )

        pitches = f0[0].cpu().numpy()
        periodicity = periodicity[0].cpu().numpy()
        pitches[pitches <= 0] = np.nan

        events = frames_to_events(pitches, periodicity, hop_length, sr, min_conf=0.4, merge_gap_s=0.08)
        return events_to_pretty_midi(events)


def frames_to_events(
    pitches_hz: np.ndarray,
    conf: np.ndarray,
    hop_length: int,
    sr: int,
    min_conf: float,
    merge_gap_s: float,
) -> List[Tuple[float, float, int]]:
    events: List[Tuple[float, float, int]] = []
    active = False
    cur_start = 0.0
    cur_midi = 0

    for i, (pitch_hz, confidence) in enumerate(zip(pitches_hz, conf)):
        timestamp = i * hop_length / sr
        voiced = bool(confidence >= min_conf and not np.isnan(pitch_hz))
        if voiced:
            midi_note = int(round(librosa.hz_to_midi(float(pitch_hz))))
            if not active:
                active = True
                cur_start = timestamp
                cur_midi = midi_note
            elif abs(cur_midi - midi_note) > 1:
                events.append((cur_start, timestamp, cur_midi))
                cur_start = timestamp
                cur_midi = midi_note
        elif active:
            events.append((cur_start, timestamp, cur_midi))
            active = False

    if active:
        timestamp = len(pitches_hz) * hop_length / sr
        events.append((cur_start, timestamp, cur_midi))

    merged: List[Tuple[float, float, int]] = []
    for start, end, midi in events:
        if merged and start - merged[-1][1] <= merge_gap_s and merged[-1][2] == midi:
            merged[-1] = (merged[-1][0], end, midi)
        else:
            merged.append((start, end, midi))
    return merged


def events_to_pretty_midi(events: List[Tuple[float, float, int]]) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=33)
    for start, end, midi_note in events:
        duration = max(end - start, 0.05)
        instrument.notes.append(
            pretty_midi.Note(
                velocity=100,
                pitch=int(midi_note),
                start=float(start),
                end=float(start + duration),
            )
        )
    pm.instruments.append(instrument)
    return pm


def midi_bytes_to_bassline(midi_bytes: bytes) -> bytes:
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
    notes = midi_to_note_intervals(mid)
    bass_notes = pick_lowest_line(notes)
    out_mid = notes_to_midi(bass_notes, ticks_per_beat=mid.ticks_per_beat)
    buf = io.BytesIO()
    out_mid.save(file=buf)
    return buf.getvalue()


def midi_to_note_intervals(mid: mido.MidiFile) -> List[Dict[str, Any]]:
    notes: List[Dict[str, Any]] = []
    active: Dict[Tuple[int, int, int], Tuple[int, int]] = {}

    for ti, track in enumerate(mid.tracks):
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                ch = getattr(msg, "channel", 0)
                key = (ti, ch, msg.note)
                active[key] = (abs_time, msg.velocity)
            elif msg.type in ("note_off",) or (msg.type == "note_on" and msg.velocity == 0):
                ch = getattr(msg, "channel", 0)
                key = (ti, ch, msg.note)
                if key in active:
                    start_time, vel = active.pop(key)
                    notes.append(
                        {
                            "start": start_time,
                            "end": abs_time,
                            "note": msg.note,
                            "velocity": vel,
                            "channel": ch,
                        }
                    )

    if active:
        max_time = 0
        for track in mid.tracks:
            t = 0
            for msg in track:
                t += msg.time
            max_time = max(max_time, t)
        for (ti, ch, n), (start_time, vel) in active.items():
            notes.append(
                {
                    "start": start_time,
                    "end": max_time,
                    "note": n,
                    "velocity": vel,
                    "channel": ch,
                }
            )

    notes.sort(key=lambda x: (x["start"], x["note"]))
    return notes


def pick_lowest_line(notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not notes:
        return []

    change_points = sorted({n["start"] for n in notes} | {n["end"] for n in notes})
    active: List[Dict[str, Any]] = []
    bass_notes: List[Dict[str, Any]] = []
    current_bass: Dict[str, Any] | None = None
    i = 0

    for t in change_points:
        while i < len(notes) and notes[i]["start"] == t:
            active.append(notes[i])
            i += 1

        active = [n for n in active if n["end"] != t]

        if active:
            lowest = min(active, key=lambda x: x["note"])
            if current_bass is None or lowest["note"] != current_bass["note"]:
                if current_bass is not None:
                    current_bass["end"] = t
                    bass_notes.append(current_bass)
                current_bass = {
                    "start": t,
                    "end": None,
                    "note": lowest["note"],
                    "velocity": lowest["velocity"],
                    "channel": lowest["channel"],
                }
        else:
            if current_bass is not None:
                current_bass["end"] = t
                bass_notes.append(current_bass)
                current_bass = None

    if current_bass is not None:
        current_bass["end"] = change_points[-1] + 1
        bass_notes.append(current_bass)

    return bass_notes


def notes_to_midi(bass_notes: List[Dict[str, Any]], ticks_per_beat: int) -> mido.MidiFile:
    mid_out = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid_out.tracks.append(track)

    events: List[Tuple[int, mido.Message]] = []
    for n in bass_notes:
        start = int(n["start"])
        end = int(n["end"])
        note = int(n["note"])
        vel = int(n["velocity"])
        ch = int(n["channel"])
        events.append((start, mido.Message("note_on", note=note, velocity=vel, channel=ch, time=0)))
        events.append((end, mido.Message("note_off", note=note, velocity=0, channel=ch, time=0)))

    events.sort(key=lambda x: x[0])

    last_time = 0
    for abs_time, msg in events:
        delta = abs_time - last_time
        msg.time = delta
        track.append(msg)
        last_time = abs_time

    track.append(mido.MetaMessage("end_of_track", time=0))
    return mid_out


PIPELINE: BaseBassTranscriber | None = None


def get_pipeline() -> BaseBassTranscriber:
    global PIPELINE
    if PIPELINE is None:
        if SCANBASS_MODE == "heavy":
            try:
                PIPELINE = HeavyBassTranscriber()
            except Exception as exc:
                print(f"⚠️  Falling back to light pipeline: {exc}")
                PIPELINE = LightBassTranscriber()
        else:
            PIPELINE = LightBassTranscriber()
    return PIPELINE


@app.get("/")
async def root():
    pipeline = get_pipeline()
    return {"ok": True, "message": "scanbass backend online", "jobs": len(JOBS), "mode": pipeline.name}


@app.head("/")
async def root_head():
    return JSONResponse(content={"ok": True})


def load_audio_bytes(data: bytes, filename: str) -> Tuple[np.ndarray, int]:
    try:
        y, sr = sf.read(io.BytesIO(data), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y.astype(np.float32), int(sr)
    except Exception:
        pass

    try:
        import torchaudio
    except ImportError as exc:
        raise ValueError("Audio decoding failed and torchaudio is not available.") from exc

    buffer = io.BytesIO(data)
    buffer.seek(0)
    waveform, sr = torchaudio.load(buffer)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    y = waveform.squeeze(0).numpy().astype(np.float32)
    return y, int(sr)


def prepare_audio(data: bytes, filename: str) -> Tuple[np.ndarray, int]:
    y, sr = load_audio_bytes(data, filename)
    if y.ndim != 1:
        y = np.asarray(y, dtype=np.float32).reshape(-1)
    max_samples = int(MAX_SECONDS * sr)
    if y.shape[0] > max_samples:
        y = y[:max_samples]
    if not np.any(np.isfinite(y)):
        raise ValueError("Audio file contains no usable samples.")
    return y.astype(np.float32), sr


@app.on_event("startup")
async def warmup() -> None:
    pipeline = get_pipeline()
    try:
        pipeline.warmup()
        print(f"✅ warmup done ({pipeline.name} mode)")
    except Exception as exc:
        print(f"❌ warmup failed: {exc}")
        if isinstance(pipeline, HeavyBassTranscriber):
            print("⚠️  Falling back to light pipeline after warmup failure")
            global PIPELINE
            PIPELINE = LightBassTranscriber()
            try:
                PIPELINE.warmup()
                print("✅ warmup done (light mode)")
            except Exception as light_exc:
                print(f"❌ fallback warmup failed: {light_exc}")


@app.post("/jobs")
async def create_job(file: UploadFile = File(...)):
    filename_lower = (file.filename or "").lower()
    original_name = file.filename or "input"

    data = await file.read()
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "processing",
        "result": None,
        "error": None,
        "original_name": original_name,
    }

    try:
        if filename_lower.endswith((".mid", ".midi")):
            result_bytes = midi_bytes_to_bassline(data)
        else:
            if not filename_lower.endswith((".wav", ".mp3", ".flac", ".ogg")):
                raise ValueError("Only .wav or .mp3 audio is supported.")
            y, sr = prepare_audio(data, filename_lower)
            pipeline = get_pipeline()
            pm = pipeline.audio_to_midi(y, sr)
            buf = io.BytesIO()
            pm.write(buf)
            result_bytes = buf.getvalue()

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = result_bytes

    except Exception as exc:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(exc)

    return {"job_id": job_id, "status": JOBS[job_id]["status"]}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] == "processing":
        return {"job_id": job_id, "status": "processing"}

    if job["status"] == "error":
        return {"job_id": job_id, "status": "error", "error": job["error"]}

    result_bytes: bytes = job["result"]
    original_name: str = job.get("original_name") or "output"

    if "." in original_name:
        original_name = original_name.rsplit(".", 1)[0]

    download_name = f"scanbass_{original_name}.mid"

    return StreamingResponse(
        io.BytesIO(result_bytes),
        media_type="audio/midi",
        headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
    )
