# Bass mode: Demucs isolate -> torchcrepe pitch -> MIDI
import sys, subprocess, shutil
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import torch
import torchcrepe
import pretty_midi

def _hz_to_midi_round(hz: float) -> int:
    if hz <= 0:
        return 0
    return int(round(69 + 12 * np.log2(hz / 440.0)))

def _contiguous_regions(condition):
    import numpy as np
    d = np.diff(condition.astype(int))
    idx, = np.where(d != 0)
    idx += 1
    if condition[0]:
        idx = np.r_[0, idx]
    if condition[-1]:
        idx = np.r_[idx, condition.size]
    return [(int(s), int(e)) for s, e in idx.reshape(-1, 2)]

def _to_midi_from_basswav(bass_wav_path: str, out_dir: str,
                          hop_length: int = 160,  # 10ms @16k
                          voicing_threshold: float = 0.5,
                          min_note_len_ms: float = 60.0,
                          velocity_floor: int = 50,
                          velocity_ceil: int = 100):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    y, sr = sf.read(bass_wav_path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000

    audio = torch.tensor(y, device="cpu").unsqueeze(0)
    f0, pd = torchcrepe.predict(audio, sr, hop_length, fmin=20.0, fmax=300.0,
                                model="full", batch_size=2048, device="cpu",
                                return_periodicity=True)
    f0 = torchcrepe.filter.median(f0, 3)
    pd = torchcrepe.filter.median(pd, 3)
    voiced = (pd.squeeze(0).numpy() > voicing_threshold)
    f0 = f0.squeeze(0).numpy()
    times = np.arange(len(f0)) * (hop_length / sr)

    # RMS -> velocity
    frame_len = hop_length
    n_frames = len(f0)
    rms = []
    for i in range(n_frames):
        start = int(i * hop_length)
        end = int(min(len(y), start + hop_length))
        seg = y[start:end]
        if len(seg) == 0:
            rms.append(0.0)
        else:
            rms.append(float(np.sqrt(np.mean(seg**2))))
    rms = np.array(rms)
    if rms.max() > 0:
        rms = rms / rms.max()
    velocities = (velocity_floor + (velocity_ceil - velocity_floor) * rms).astype(int)

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=32)  # Acoustic Bass
    min_len = min_note_len_ms / 1000.0
    for s_idx, e_idx in _contiguous_regions(voiced):
        s_t = times[s_idx]
        e_t = times[e_idx - 1] if e_idx > s_idx else times[s_idx] + (hop_length / sr)
        if e_t - s_t < min_len:
            continue
        region_f0 = f0[s_idx:e_idx]
        region_f0 = region_f0[region_f0 > 0]
        if len(region_f0) == 0:
            continue
        midi_pitch = int(np.clip(_hz_to_midi_round(float(np.median(region_f0))), 21, 108))
        vel = int(np.median(velocities[s_idx:e_idx]))
        inst.notes.append(pretty_midi.Note(velocity=vel, pitch=midi_pitch, start=float(s_t), end=float(e_t)))
    pm.instruments.append(inst)
    out_path = Path(out) / "bassline.mid"
    pm.write(str(out_path))
    return str(out_path)

def run_bass_mode(
    audio_path: str,
    out_dir: str,
    voicing_threshold: float = 0.5,
    segment_seconds: float = 15.0,
    overlap: float = 0.1,
):
    in_path = Path(audio_path).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")

    # 1) Demucs isolate (use mdx_extra to avoid diffq on Win)
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems=bass",
        "-n", "mdx_extra",
        "--segment", str(segment_seconds),
        "--overlap", str(overlap),
        "-o", str(out),
        str(in_path)
    ]
    subprocess.run(cmd, check=True)

    # 2) Find produced bass.wav (Demucs nests it)
    candidates = sorted(out.rglob("bass.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("Demucs did not produce bass.wav. Try another model or input.")
    # prefer nested bass.wav
    bass_src = None
    for p in candidates:
        if p.parent != out:
            bass_src = p
            break
    if bass_src is None:
        bass_src = candidates[0]

    target = out / "bass.wav"
    if target.exists():
        target.unlink()
    shutil.copy2(str(bass_src), str(target))

    # 3) Bass -> MIDI
    midi_path = _to_midi_from_basswav(str(target), str(out), voicing_threshold=voicing_threshold)
    return {"input": str(in_path), "bass_wav": str(target), "midi": midi_path}
