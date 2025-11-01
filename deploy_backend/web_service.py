from __future__ import annotations

"""
ScanBass – WAV-only, lightweight backend (MVP + warmup)

Hodnotová nabídka:
- uživatel nahraje AUDIO (.wav) nebo už hotové MIDI
- backend z audia udělá: wav -> (band-pass -> YIN) -> MIDI -> vybere nejnižší linku
- pokud nahraje MIDI, jen vybere nejnižší linku
- frontend může dál používat POST /jobs + GET /jobs/{id}

Omezení:
- AUDIO: jen .wav (kvůli rychlosti na Render Starter)
- warmup při startu pro zkrácení cold start latency
"""

import io
import uuid
from typing import Dict, Any, List, Tuple

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfilt
import pretty_midi
import mido

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

# ---------------------------------------------------------
# Konfigurace
# ---------------------------------------------------------
MAX_SECONDS = 30.0
TARGET_SR = 16000
BASS_LOW = 40
BASS_HIGH = 300

app = FastAPI(title="ScanBass – WAV-only MVP (with warmup)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# jednoduché in-memory "úložiště" jobů
JOBS: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------
# Healthchecky pro Render
# ---------------------------------------------------------
@app.get("/")
async def root():
    return {"ok": True, "message": "scanbass wav-only online", "jobs": len(JOBS)}


@app.head("/")
async def root_head():
    return JSONResponse(content={"ok": True})


# ---------------------------------------------------------
# DSP / audio -> MIDI (basa)
# ---------------------------------------------------------
def bandpass(y: np.ndarray, sr: int, low: int = BASS_LOW, high: int = BASS_HIGH, order: int = 4) -> np.ndarray:
    sos = butter(order, [low, high], btype="band", fs=sr, output="sos")
    return sosfilt(sos, y)


def frames_to_events(
    pitches_hz: np.ndarray,
    conf: np.ndarray,
    hop_length: int,
    sr: int,
    min_conf: float = 0.15,
    merge_gap_s: float = 0.12,
):
    events = []
    active = False
    cur_start = None
    cur_midi = None

    for i, (p, c) in enumerate(zip(pitches_hz, conf)):
        t = i * hop_length / sr
        voiced = (not np.isnan(p)) and (c >= min_conf)
        if voiced:
            midi = int(round(librosa.hz_to_midi(p)))
            if not active:
                active = True
                cur_start = t
                cur_midi = midi
            else:
                if abs(cur_midi - midi) > 1:
                    events.append((cur_start, t, cur_midi))
                    cur_start = t
                    cur_midi = midi
        else:
            if active:
                events.append((cur_start, t, cur_midi))
                active = False
                cur_start = None
                cur_midi = None

    if active:
        t = len(pitches_hz) * hop_length / sr
        events.append((cur_start, t, cur_midi))

    # sloučení krátkých mezer
    merged = []
    for s, e, m in events:
        if merged and s - merged[-1][1] <= merge_gap_s and merged[-1][2] == m:
            merged[-1] = (merged[-1][0], e, m)
        else:
            merged.append((s, e, m))
    return merged


def audio_wav_to_bass_midi(y: np.ndarray, sr: int) -> pretty_midi.PrettyMIDI:
    # oříznout délku
    max_samples = int(MAX_SECONDS * sr)
    y = y[:max_samples]

    # resample (kvůli rychlosti)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    # band-pass na basy
    y_bp = bandpass(y, sr)

    # pitch přes YIN
    hop_length = 512
    pitches = librosa.yin(
        y_bp,
        fmin=librosa.note_to_hz("C1"),
        fmax=librosa.note_to_hz("C5"),
        sr=sr,
        frame_length=2048,
        hop_length=hop_length,
    )

    # confidence z RMS
    S = np.abs(librosa.stft(y_bp, n_fft=2048, hop_length=hop_length))
    rms = librosa.feature.rms(S=S)[0]
    conf = (rms - rms.min()) / (rms.max() - rms.min() + 1e-9)

    events = frames_to_events(pitches, conf, hop_length, sr)

    # poskládat MIDI
    pm = pretty_midi.PrettyMIDI()
    instr = pretty_midi.Instrument(program=33)  # Acoustic Bass
    for start, end, midi_note in events:
        instr.notes.append(
            pretty_midi.Note(
                velocity=100,
                pitch=int(midi_note),
                start=float(start),
                end=float(max(end, start + 0.05)),
            )
        )
    pm.instruments.append(instr)
    return pm


# ---------------------------------------------------------
# MIDI -> jen nejnižší linka
# ---------------------------------------------------------
def midi_bytes_to_bassline(midi_bytes: bytes) -> bytes:
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
    notes = midi_to_note_intervals(mid)
    bass_notes = pick_lowest_line(notes)
    out_mid = notes_to_midi(bass_notes, ticks_per_beat=mid.ticks_per_beat)
    buf = io.BytesIO()
    out_mid.save(file=buf)
    return buf.getvalue()


def midi_to_note_intervals(mid: mido.MidiFile) -> List[Dict]:
    notes: List[Dict] = []
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

    # ukončit otevřené noty
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


def pick_lowest_line(notes: List[Dict]) -> List[Dict]:
    if not notes:
        return []

    change_points = sorted({n["start"] for n in notes} | {n["end"] for n in notes})
    active: List[Dict] = []
    bass_notes: List[Dict] = []
    current_bass = None
    i = 0

    for t in change_points:
        # nové noty
        while i < len(notes) and notes[i]["start"] == t:
            active.append(notes[i])
            i += 1

        # ukončené noty vyhodíme
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


def notes_to_midi(bass_notes: List[Dict], ticks_per_beat: int) -> mido.MidiFile:
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


# ---------------------------------------------------------
# Warmup při startu – inicializace librosy
# ---------------------------------------------------------
@app.on_event("startup")
async def warmup():
    try:
        sr = TARGET_SR
        # vygenerujeme 1 sekundu ticha a "proženeme" pipeline
        y = np.zeros(sr, dtype=np.float32)
        _ = audio_wav_to_bass_midi(y, sr)
        print("Warmup completed – librosa initialized.")
    except Exception as e:
        print("Warmup failed:", e)


# ---------------------------------------------------------
# API – kompatibilní s frontendem
# ---------------------------------------------------------
@app.post("/jobs")
async def create_job(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()
    data = await file.read()

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "processing", "result": None, "error": None}

    try:
        if filename.endswith((".mid", ".midi")):
            result_bytes = midi_bytes_to_bassline(data)
        else:
            if not filename.endswith(".wav"):
                raise ValueError("Only .wav is supported in this MVP version.")
            y, sr = sf.read(io.BytesIO(data))
            if y.ndim > 1:
                y = y.mean(axis=1)  # stereo -> mono
            pm = audio_wav_to_bass_midi(y, sr)
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

    return StreamingResponse(
        io.BytesIO(job["result"]),
        media_type="audio/midi",
        headers={"Content-Disposition": 'attachment; filename="scanbass.mid"'},
    )
