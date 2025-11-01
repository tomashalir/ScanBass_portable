# Poly mode: Basic Pitch -> lowest voice -> MIDI
from pathlib import Path
from typing import Optional

import numpy as np
import pretty_midi
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH  # <- vestavěná cesta k modelu


def _collect_all_notes(midi_path: Path):
    """Load a MIDI file and return a flat list of notes: (start, end, pitch, velocity)."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append((float(n.start), float(n.end), int(n.pitch), int(n.velocity)))
    return notes


def _merge_short_gaps(events, gap_ms: int = 60):
    """Merge consecutive same-pitch events if the silent gap is shorter than gap_ms (ms)."""
    if not events:
        return []
    out = [events[0]]
    for s, e, p in events[1:]:
        ps = out[-1]
        if p == ps[2] and (s - ps[1]) * 1000.0 <= gap_ms:
            out[-1] = (ps[0], e, p)
        else:
            out.append((s, e, p))
    return out


def _lowest_voice_to_mono(notes, frame_hz: int = 40, min_note_len_ms: int = 90, gap_merge_ms: int = 60):
    """Pick at each time frame the lowest active pitch and compress into monophonic segments."""
    if not notes:
        pm = pretty_midi.PrettyMIDI()
        pm.instruments.append(pretty_midi.Instrument(program=32))  # Acoustic Bass
        return pm

    t_end = max(e for _, e, _, _ in notes)
    dt = 1.0 / frame_hz
    times = np.arange(0, t_end + dt, dt)

    # Lowest active pitch per frame
    active_low = []
    for t in times:
        active = [p for s, e, p, _ in notes if (s <= t < e)]
        active_low.append(min(active) if active else None)

    # Compress frames into segments with constant pitch
    mono = []
    cur_pitch, t_start = None, None
    for t, p in zip(times, active_low):
        if p != cur_pitch:
            if cur_pitch is not None:
                mono.append((t_start, t, cur_pitch))
            cur_pitch, t_start = p, t
    if cur_pitch is not None:
        mono.append((t_start, times[-1], cur_pitch))

    # Filter short notes and merge micro-gaps
    min_len = min_note_len_ms / 1000.0
    mono = [(s, e, p) for s, e, p in mono if (p is not None and (e - s) >= min_len)]
    mono = _merge_short_gaps(mono, gap_ms=gap_merge_ms)

    # Build MIDI
    pm_out = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=32)  # Acoustic Bass
    for s, e, p in mono:
        inst.notes.append(pretty_midi.Note(velocity=92, pitch=int(p), start=float(s), end=float(e)))
    pm_out.instruments.append(inst)
    return pm_out


def run_poly_mode(
    audio_path: str,
    out_dir: str,
    frame_hz: int = 40,
    min_note_len_ms: int = 90,
    gap_merge_ms: int = 60,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    basic_pitch_min_note_len_ms: float = 127.7,
    minimum_frequency: Optional[float] = None,
    maximum_frequency: Optional[float] = None,
):
    """
    1) Transcribe POLY MIDI from audio (Basic Pitch 0.4.0).
    2) Extract the lowest voice and save as bassline.mid
    """
    audio = Path(audio_path).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if frame_hz <= 0:
        raise ValueError("frame_hz must be positive")
    if min_note_len_ms <= 0:
        raise ValueError("min_note_len_ms must be positive")
    if gap_merge_ms < 0:
        raise ValueError("gap_merge_ms must be non-negative")
    if not 0.0 <= onset_threshold <= 1.0:
        raise ValueError("onset_threshold must be between 0 and 1")
    if not 0.0 <= frame_threshold <= 1.0:
        raise ValueError("frame_threshold must be between 0 and 1")
    if basic_pitch_min_note_len_ms <= 0:
        raise ValueError("basic_pitch_min_note_len_ms must be positive")
    if minimum_frequency is not None and minimum_frequency <= 0:
        raise ValueError("minimum_frequency must be positive")
    if maximum_frequency is not None and maximum_frequency <= 0:
        raise ValueError("maximum_frequency must be positive")
    if (
        minimum_frequency is not None
        and maximum_frequency is not None
        and minimum_frequency >= maximum_frequency
    ):
        raise ValueError("minimum_frequency must be less than maximum_frequency")

    # Basic Pitch 0.4.0: positionální volání
    # predict_and_save(inputs, output_directory, save_midi, sonify_midi,
    #                  save_model_outputs, save_notes, model_or_model_path)
    predict_and_save(
        [str(audio)],
        str(out),
        True,    # save_midi
        False,   # sonify_midi
        False,   # save_model_outputs
        False,   # save_notes
        ICASSP_2022_MODEL_PATH,  # vestavěná cesta k TF modelu
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=basic_pitch_min_note_len_ms,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )

    # Najdi vyrobené poly MIDI
    midi_poly = None
    for cand in out.glob(f"{audio.stem}*basic_pitch*.mid"):
        midi_poly = cand
        break
    if midi_poly is None:
        mids = sorted(out.glob("*.mid"))
        if mids:
            midi_poly = mids[0]
        else:
            raise FileNotFoundError("Basic Pitch nevygeneroval poly MIDI.")

    # Nejnižší hlas -> monofonní MIDI
    notes = _collect_all_notes(midi_poly)
    pm_out = _lowest_voice_to_mono(notes, frame_hz=frame_hz, min_note_len_ms=min_note_len_ms, gap_merge_ms=gap_merge_ms)
    bassline_path = out / "bassline.mid"
    pm_out.write(str(bassline_path))
    return {"input": str(audio), "poly_midi": str(midi_poly), "midi": str(bassline_path)}
