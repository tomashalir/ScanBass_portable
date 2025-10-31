# modes/quality_mode.py
# Standalone "quality" pipeline:
# - Preprocess audio (HPSS harmonic emphasis) if librosa is available
# - Run existing poly pipeline
# - Postprocess MIDI to a monophonic lowest-voice line
# - Export as notes-only MIDI (no tempo/TS/markers) to avoid DAW tempo override
from __future__ import annotations
import os, sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

# Ensure we can import existing poly_mode
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
try:
    from .poly_mode import run_poly_mode
except Exception:
    # fallback if relative import fails
    from modes.poly_mode import run_poly_mode  # type: ignore

@dataclass
class Note:
    start: float
    end: float
    pitch: int
    velocity: int = 96

def _safe_import_librosa():
    try:
        import librosa, numpy as np, soundfile as sf
        return librosa, np, sf
    except Exception:
        return None, None, None

def _preprocess_harmonic(input_audio: Path, out_dir: Path) -> Path:
    librosa, np, sf = _safe_import_librosa()
    if librosa is None:
        return input_audio  # no-op
    try:
        y, sr = librosa.load(str(input_audio), sr=22050, mono=True)
        # HPSS with slightly higher percussive margin (reduce kick bleed)
        y_harm, _ = librosa.effects.hpss(y, margin=(1.0, 2.0))
        # light normalization
        maxv = float(np.max(np.abs(y_harm))) if y_harm.size else 0.0
        if maxv > 1e-6:
            y_harm = y_harm * (0.95 / maxv)
        out_wav = out_dir / (input_audio.stem + "_harm.wav")
        sf.write(str(out_wav), y_harm, sr)
        return out_wav
    except Exception:
        return input_audio  # fallback silently

def _load_all_notes(midi_path: Path) -> List[Note]:
    try:
        import pretty_midi
    except Exception as e:
        raise RuntimeError("pretty_midi is required for quality post-processing") from e
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes: List[Note] = []
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append(Note(float(n.start), float(n.end), int(n.pitch), int(n.velocity)))
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes

def _lowest_voice_monophonic(notes: List[Note],
                             pitch_lo:int=28, pitch_hi:int=60,
                             min_seg:float=0.06, min_gap:float=0.03) -> List[Note]:
    """Build monophonic bassline by selecting the lowest active pitch in each event interval.
    Then merge and clean short blips/gaps.
    """
    # Clamp to range
    pool = [n for n in notes if pitch_lo <= n.pitch <= pitch_hi and n.end > n.start + 1e-6]
    if not pool: return []
    # Event boundaries (all starts/ends)
    edges = sorted({t for n in pool for t in (n.start, n.end)})
    res: List[Note] = []
    for a,b in zip(edges, edges[1:]):
        if b <= a + 1e-6: continue
        active = [n for n in pool if n.start < b and n.end > a]
        if not active: continue
        # choose lowest pitch; velocity as median-ish
        active.sort(key=lambda n: (n.pitch, n.start))
        chosen = active[0]
        if res and res[-1].pitch == chosen.pitch and a <= res[-1].end + min_gap:
            # extend previous
            res[-1].end = b
            res[-1].velocity = int((res[-1].velocity + chosen.velocity) / 2)
        else:
            res.append(Note(a, b, chosen.pitch, chosen.velocity))
    # Remove very short segments; merge adjacent same-pitch separated by tiny gap
    cleaned: List[Note] = []
    for n in res:
        if cleaned and n.pitch == cleaned[-1].pitch and n.start <= cleaned[-1].end + min_gap:
            cleaned[-1].end = max(cleaned[-1].end, n.end)
            cleaned[-1].velocity = int((cleaned[-1].velocity + n.velocity) / 2)
        else:
            cleaned.append(n)
    cleaned = [n for n in cleaned if (n.end - n.start) >= min_seg]
    return cleaned

def _detect_bpm(midi_path: Path) -> Optional[float]:
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        times, tempi = pm.get_tempo_changes()
        if len(tempi)>0 and tempi[0]>0: return float(tempi[0])
        est = float(pm.estimate_tempo())
        return est if est>0 else None
    except Exception:
        return None

def _write_notes_only(notes: List[Note], out_path: Path, bpm: Optional[float], tpb:int=480) -> None:
    """Write a single-track MIDI with only note_on/note_off. No tempo/time-signature/markers."""
    if not notes:
        raise RuntimeError("No notes to write")
    try:
        import mido
    except Exception as e:
        raise RuntimeError("mido is required for notes-only export") from e
    # Scale seconds -> beats using bpm (no tempo meta)
    if bpm is None or bpm <= 0: bpm = 120.0
    scale = bpm / 60.0
    # Build events
    evts = []
    for n in notes:
        s = max(0.0, n.start); e = max(s + 1e-4, n.end)
        s_tick = int(round(s * scale * tpb))
        e_tick = int(round(e * scale * tpb))
        if e_tick <= s_tick: e_tick = s_tick + 1
        vel = max(1, min(127, int(n.velocity)))
        evts.append(('off', e_tick, n.pitch, 0))
        evts.append(('on',  s_tick, n.pitch, vel))
    evts.sort(key=lambda x: (x[1], 0 if x[0]=='off' else 1))
    # Write
    mid = mido.MidiFile(type=1, ticks_per_beat=tpb)
    track = mido.MidiTrack(); mid.tracks.append(track)
    last = 0
    for kind, tick, pitch, vel in evts:
        dt = max(0, tick - last); last = tick
        if kind == 'on':
            track.append(mido.Message('note_on', note=int(pitch), velocity=int(vel), time=dt, channel=0))
        else:
            track.append(mido.Message('note_off', note=int(pitch), velocity=0, time=dt, channel=0))
    track.append(mido.MetaMessage('end_of_track', time=0))
    mid.save(str(out_path))

def run_quality_mode(input_audio: str, out_dir: str) -> dict:
    """Returns dict with keys: midi (notes-only path), src_midi (original model midi), preprocessed_wav (if any)."""
    out_dir_p = Path(out_dir); out_dir_p.mkdir(parents=True, exist_ok=True)
    in_p = Path(input_audio)
    # 1) Preprocess (HPSS harmonic emphasis) to temp wav if possible
    pre = _preprocess_harmonic(in_p, out_dir_p)
    # 2) Run original poly pipeline
    try:
        res = run_poly_mode(str(pre), str(out_dir_p))
    except TypeError:
        res = run_poly_mode(str(pre))
    src_midi: Optional[Path] = None
    if isinstance(res, dict):
        cand = res.get("midi") or res.get("bass_midi") or res.get("output_midi")
        if cand: src_midi = Path(cand)
    if (not src_midi) or (not src_midi.exists()):
        # fallback: newest .mid
        mids = sorted(out_dir_p.glob("*.mid"), key=lambda x: x.stat().st_mtime, reverse=True)
        src_midi = mids[0] if mids else None
    if not src_midi or not src_midi.exists():
        raise RuntimeError("Quality mode: no MIDI produced by base pipeline")
    # 3) Postprocess: pick lowest voice, clean
    notes_all = _load_all_notes(src_midi)
    notes_bass = _lowest_voice_monophonic(notes_all, pitch_lo=28, pitch_hi=60, min_seg=0.06, min_gap=0.03)
    if not notes_bass:
        # fallback: keep original as-is (still return something)
        out_mid = out_dir_p / "bassline_quality.mid"
        try:
            # copy via pretty_midi roundtrip to ensure valid file
            import pretty_midi
            pm = pretty_midi.PrettyMIDI(str(src_midi)); pm.write(str(out_mid))
        except Exception:
            out_mid = src_midi
        return {"midi": str(out_mid), "src_midi": str(src_midi), "preprocessed_wav": (str(pre) if pre!=in_p else "")}
    # 4) Notes-only export (no tempo) → prevents DAW tempo override
    bpm = _detect_bpm(src_midi) or 120.0
    out_mid = out_dir_p / "bassline_quality.mid"
    _write_notes_only(notes_bass, out_mid, bpm=bpm, tpb=480)
    return {"midi": str(out_mid), "src_midi": str(src_midi), "preprocessed_wav": (str(pre) if pre!=in_p else "")}