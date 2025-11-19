from __future__ import annotations

"""
ScanBass backend – audio -> full MIDI (basic-pitch style) -> pick lowest notes -> bassline.mid
Demucs i původní "light" transcriber jsou pryč.
"""

import io
import os
import uuid
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import librosa
import mido
import numpy as np
import pretty_midi
import soundfile as sf
from fastapi import File, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# 👉 tohle je náš hlavní "limit" kvůli Renderu
MAX_SECONDS = 20.0
TARGET_SR = 16000  # stačí pro transcription
# tohle teď neřešíme v audiu, děláme výběr až v MIDI, ale nechávám kdybys chtěl filtr
BASS_LOW = 40
BASS_HIGH = 300

MAJOR_TEMPLATE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_TEMPLATE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
KEY_NAMES = [
    "C",
    "C# / Db",
    "D",
    "D# / Eb",
    "E",
    "F",
    "F# / Gb",
    "G",
    "G# / Ab",
    "A",
    "A# / Bb",
    "B",
]

app = FastAPI(title="ScanBass backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# jednoduchá in-memory "DB"
JOBS: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# MIDI ↔ notes utily – tohle už jsi měl, nechávám beze změny
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# audio -> full MIDI (basic-pitch styl)
# ---------------------------------------------------------------------------
def _estimate_key_from_chroma(chroma: np.ndarray) -> Tuple[str | None, float | None]:
    """Estimate key from an aggregated chroma vector using Krumhansl-Schmuckler profiles."""

    if chroma.size == 0:
        return None, None

    chroma_vector = chroma.mean(axis=1)
    if np.allclose(chroma_vector, 0):
        return None, None

    scores = []
    for shift in range(12):
        major_score = float(np.dot(np.roll(MAJOR_TEMPLATE, shift), chroma_vector))
        minor_score = float(np.dot(np.roll(MINOR_TEMPLATE, shift), chroma_vector))
        scores.append(("major", shift, major_score))
        scores.append(("minor", shift, minor_score))

    best_mode, best_shift, best_score = max(scores, key=lambda x: x[2])
    normalized_score = best_score / max(sum(chroma_vector), 1e-6)
    key_name = f"{KEY_NAMES[best_shift]} {'major' if best_mode == 'major' else 'minor'}"
    return key_name, normalized_score


def analyze_audio(data: np.ndarray, sr: int) -> Dict[str, Any]:
    """Compute lightweight tempo + key estimates from the prepared audio clip."""

    tempo_bpm = None
    key = None
    key_confidence = None

    try:
        tempo, _ = librosa.beat.beat_track(y=data, sr=sr, units="time")
        if np.isfinite(tempo):
            tempo_bpm = float(np.round(tempo, 1))
    except Exception:
        pass

    try:
        chroma = librosa.feature.chroma_cqt(y=data, sr=sr)
        key, key_confidence = _estimate_key_from_chroma(chroma)
    except Exception:
        pass

    return {"tempo_bpm": tempo_bpm, "key": key, "key_confidence": key_confidence}


def analyze_midi_bytes(midi_bytes: bytes) -> Dict[str, Any]:
    """Fallback key/tempo estimation directly from MIDI events."""

    analysis = {"tempo_bpm": None, "key": None, "key_confidence": None}

    try:
        pm = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
        tempo = float(pm.estimate_tempo())
        if np.isfinite(tempo):
            analysis["tempo_bpm"] = float(np.round(tempo, 1))

        chroma = pm.get_chroma()
        key, conf = _estimate_key_from_chroma(chroma)
        analysis["key"] = key
        analysis["key_confidence"] = conf
    except Exception:
        pass

    return analysis


def prepare_audio_to_wav(data: bytes, filename: str) -> Tuple[str, int, np.ndarray]:
    """
    1) načte audio z bytes
    2) ořízne na MAX_SECONDS
    3) uloží do dočasného .wav
    4) vrátí cestu a sr
    """
    # zkusíme nejdřív soundfile
    try:
        y, sr = sf.read(io.BytesIO(data), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
    except Exception:
        # fallback na librosa
        y, sr = librosa.load(io.BytesIO(data), sr=None, mono=True)

    # ořez
    max_samples = int(MAX_SECONDS * sr)
    y = y[:max_samples]

    # přeresamplujeme na TARGET_SR – basic-pitch si pak případně udělá svoje
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    tmp_dir = tempfile.mkdtemp(prefix="scanbass_")
    out_path = str(Path(tmp_dir) / "input.wav")
    sf.write(out_path, y, sr)
    return out_path, sr, y


def run_basic_pitch_on_wav(wav_path: str) -> bytes:
    """
    Spustí basic-pitch/podobnou inference a vrátí MIDI jako bytes.
    Předpokládáme, že v requirements je basic-pitch.
    """
    from basic_pitch.inference import ICASSP_2022_MODEL_PATH, predict_and_save

    out_dir = Path(wav_path).parent
    # basic-pitch uloží .mid vedle
    predict_and_save(
        [wav_path],
        output_directory=str(out_dir),
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=False,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
    )
    midi_path = out_dir / "input.mid"
    if not midi_path.exists():
        raise RuntimeError("Transcription failed: MIDI file was not created.")
    midi_bytes = midi_path.read_bytes()
    return midi_bytes


# ---------------------------------------------------------------------------
# startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def warmup() -> None:
    try:
        # uděláme fake wav
        sr = TARGET_SR
        silence = np.zeros(int(sr * 1.0), dtype=np.float32)
        tmp_dir = tempfile.mkdtemp(prefix="scanbass_warmup_")
        wav_path = str(Path(tmp_dir) / "warmup.wav")
        sf.write(wav_path, silence, sr)
        _ = run_basic_pitch_on_wav(wav_path)
        print("✅ warmup done (basic-pitch pipeline)")
    except Exception as exc:
        print(f"❌ warmup failed: {exc}")


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "ok": True,
        "message": "scanbass backend online",
        "jobs": len(JOBS),
        "mode": "basic-pitch-lowest",
    }


@app.head("/")
async def root_head():
    return JSONResponse(content={"ok": True})


@app.post("/jobs")
async def create_job(file: UploadFile = File(...)):
    filename_lower = (file.filename or "").lower()
    original_name = file.filename or "input"

    data = await file.read()
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "processing",
        "result": None,
        "full_midi": None,
        "metadata": {},
        "error": None,
        "original_name": original_name,
    }

    try:
        # Když uživatel nahraje rovnou .mid → jen vybereme nejnižší noty
        if filename_lower.endswith((".mid", ".midi")):
            result_bytes = midi_bytes_to_bassline(data)
            full_midi_bytes = data
            analysis = analyze_midi_bytes(full_midi_bytes)
        else:
            # 1) uložit audio do wav (ořez, mono, sr)
            wav_path, sr, y = prepare_audio_to_wav(data, filename_lower)
            # 2) full transcription -> MIDI
            full_midi_bytes = run_basic_pitch_on_wav(wav_path)
            # 3) z MIDI udělat basovou linku
            result_bytes = midi_bytes_to_bassline(full_midi_bytes)
            # 4) tempo + tónina z audio + MIDI (doplň chybějící hodnoty)
            analysis = analyze_audio(y, sr)
            midi_analysis = analyze_midi_bytes(full_midi_bytes)
            for key, value in midi_analysis.items():
                if analysis.get(key) is None and value is not None:
                    analysis[key] = value

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = result_bytes
        JOBS[job_id]["full_midi"] = full_midi_bytes
        JOBS[job_id]["metadata"] = analysis

    except Exception as exc:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(exc)

    return {"job_id": job_id, "status": JOBS[job_id]["status"]}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    original_name = job.get("original_name") or "output"
    base_payload = {
        "job_id": job_id,
        "status": job["status"],
    }

    if job.get("metadata"):
        base_payload["metadata"] = job["metadata"]

    if job["status"] == "processing":
        return base_payload

    if job["status"] == "error":
        base_payload["error"] = job["error"]
        return base_payload

    base_name = original_name.rsplit(".", 1)[0] if original_name else "output"
    base_payload["download_url"] = f"/jobs/{job_id}/result"
    base_payload["downloads"] = {
        "bassline": {
            "url": f"/jobs/{job_id}/result",
            "name": f"scanbass_{base_name}.mid",
            "label": "Bassline MIDI",
        },
        "full": {
            "url": f"/jobs/{job_id}/result?variant=full",
            "name": f"scanbass_full_{base_name}.mid",
            "label": "Full melody MIDI",
        },
    }
    return base_payload


@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str, variant: str = "bassline"):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="Job not finished yet")

    original_name: str = job.get("original_name") or "output"
    base = original_name.rsplit(".", 1)[0] if "." in original_name else original_name

    variant_lower = variant.lower()
    if variant_lower in {"bass", "bassline", "default"}:
        midi_bytes = job.get("result")
        download_name = f"scanbass_{base}.mid"
    elif variant_lower in {"full", "melody", "poly"}:
        midi_bytes = job.get("full_midi") or job.get("result")
        download_name = f"scanbass_full_{base}.mid"
    else:
        raise HTTPException(status_code=400, detail="Unknown variant")

    if midi_bytes is None:
        raise HTTPException(status_code=404, detail="MIDI not available")

    return StreamingResponse(
        io.BytesIO(midi_bytes),
        media_type="audio/midi",
        headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
    )
