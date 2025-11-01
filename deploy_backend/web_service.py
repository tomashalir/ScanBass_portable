from __future__ import annotations

"""
Ultra-light ScanBass backend

- žádný torch / TF / demucs
- jen: vezmi MIDI → vytáhni nejnižší noty → vrať nový MIDI
- vhodné pro Render 512 MB
"""

import io
from typing import List, Tuple, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import mido


app = FastAPI(title="ScanBass (slim MIDI backend)")

# Pokud voláš z jiné domény (landing page), povolíme CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # klidně si to později zpřísni
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/status")
async def status():
    return {"ok": True, "message": "ScanBass MIDI-only backend is running"}


# ---------------------------------------------------------------------------
# Pomocné funkce pro práci s MIDI
# ---------------------------------------------------------------------------

def _midi_to_note_intervals(mid: mido.MidiFile) -> List[Dict]:
    """
    Převedeme všechny NOTE_ON / NOTE_OFF napříč všemi tracky
    do jedné ploché časové struktury = seznam not
    ve tvaru:
        {
          "start": abs_time_ticks,
          "end": abs_time_ticks,
          "note": pitch,
          "velocity": vel,
          "channel": ch
        }
    """
    notes: List[Dict] = []

    # klíč: (track_idx, channel, note) -> start_time, velocity
    active: Dict[Tuple[int, int, int], Tuple[int, int]] = {}

    for ti, track in enumerate(mid.tracks):
        abs_time = 0
        for msg in track:
            abs_time += msg.time  # delta -> absolute
            if msg.type == "note_on" and msg.velocity > 0:
                key = (ti, msg.channel if hasattr(msg, "channel") else 0, msg.note)
                active[key] = (abs_time, msg.velocity)
            elif (msg.type == "note_off") or (msg.type == "note_on" and msg.velocity == 0):
                key = (ti, msg.channel if hasattr(msg, "channel") else 0, msg.note)
                if key in active:
                    start_time, vel = active.pop(key)
                    notes.append(
                        {
                            "start": start_time,
                            "end": abs_time,
                            "note": msg.note,
                            "velocity": vel,
                            "channel": key[1],
                        }
                    )

    # některé noty můžou zůstat "otevřené" → zavřeme je na konci souboru
    if active:
        # vezmeme největší čas, co jsme kde měli
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

    # seřadíme podle začátku
    notes.sort(key=lambda x: (x["start"], x["note"]))
    return notes


def _pick_lowest_line(notes: List[Dict]) -> List[Dict]:
    """
    Vezmeme ze všech not vždy tu *nejnižší*, která běžela v daném čase.
    Děláme to hodně jednoduše, aby to bylo levné.
    """
    if not notes:
        return []

    # uděláme z toho "časové body", kde se něco mění
    change_points = sorted({n["start"] for n in notes} | {n["end"] for n in notes})

    bass_notes: List[Dict] = []

    active: List[Dict] = []
    current_bass = None  # právě hrající basová nota

    i = 0  # index do notes

    for t in change_points:
        # přidej všechny noty, které v tomto čase začínají
        while i < len(notes) and notes[i]["start"] == t:
            active.append(notes[i])
            i += 1

        # odeber všechny noty, které v tomto čase končí
        active = [n for n in active if n["end"] != t]

        # vyber nejnižší
        if active:
            lowest = min(active, key=lambda x: x["note"])
            if current_bass is None or lowest["note"] != current_bass["note"]:
                # ukonči předchozí basu
                if current_bass is not None:
                    current_bass["end"] = t
                    bass_notes.append(current_bass)
                # začni novou
                current_bass = {
                    "start": t,
                    "end": None,  # doplníme později
                    "note": lowest["note"],
                    "velocity": lowest["velocity"],
                    "channel": lowest["channel"],
                }
        else:
            # nic nehraje – ukonči basu
            if current_bass is not None:
                current_bass["end"] = t
                bass_notes.append(current_bass)
                current_bass = None

    # závěr – pokud ještě něco hraje, ukončíme na posledním čase
    if current_bass is not None:
        current_bass["end"] = change_points[-1] + 1
        bass_notes.append(current_bass)

    return bass_notes


def _notes_to_midi(bass_notes: List[Dict], ticks_per_beat: int) -> mido.MidiFile:
    """
    Z vybraných basových not uděláme single-track MIDI.
    """
    mid_out = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid_out.tracks.append(track)

    # budeme psát v absolutních časech a pak převádět na delty
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

    # na konec End of Track
    track.append(mido.MetaMessage("end_of_track", time=0))

    return mid_out


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------

@app.post("/extract-bass")
async def extract_bass(file: UploadFile = File(...)):
    """
    Přijme .mid nebo .midi a vrátí nový MIDI soubor jen s nejnižší linkou.
    """
    filename = file.filename or "input.mid"
    if not filename.lower().endswith((".mid", ".midi")):
        raise HTTPException(status_code=400, detail="Please upload a .mid/.midi file")

    data = await file.read()

    try:
        mid = mido.MidiFile(file=io.BytesIO(data))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot read MIDI: {exc!r}")

    notes = _midi_to_note_intervals(mid)
    bass_notes = _pick_lowest_line(notes)

    # když tam nic nebylo
    if not bass_notes:
        # vrátíme prázdné MIDI se stejným ticks_per_beat
        empty_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
        empty_track = mido.MidiTrack()
        empty_track.append(mido.MetaMessage("end_of_track", time=0))
        empty_mid.tracks.append(empty_track)
        out_bytes = io.BytesIO()
        empty_mid.save(file=out_bytes)
        out_bytes.seek(0)
        return StreamingResponse(
            out_bytes,
            media_type="audio/midi",
            headers={"Content-Disposition": 'attachment; filename="bassline.mid"'},
        )

    out_mid = _notes_to_midi(bass_notes, ticks_per_beat=mid.ticks_per_beat)
    buf = io.BytesIO()
    out_mid.save(file=buf)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="audio/midi",
        headers={"Content-Disposition": 'attachment; filename="bassline.mid"'},
    )
