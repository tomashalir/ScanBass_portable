# ScanBass — clean MVP (Python 3.11)

One CLI, two modes:
- **bass**: isolate bass from a mix (Demucs mdx_extra) → detect notes (torchcrepe) → `bassline.mid`
- **poly**: transcribe poly MIDI (Basic Pitch) → take **lowest notes** over time → `bassline.mid`

Outputs go to: `outputs/<input_name>_<mode>/`

## 0) Prereqs (Windows)
- Python **3.11 (64‑bit)** installed, added to PATH
- FFmpeg (`winget install Gyan.FFmpeg`)

## 1) Create & activate a new venv
```powershell
py -3.11 -m venv .venv311
.\.venv311\Scriptsctivate
python --version  # should be 3.11.x
```

## 2) Install dependencies (all free)
> Torch uses the official CPU index (works without GPU).
```powershell
python -m pip install --upgrade pip wheel
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 3) Run
**A) Piano/accords → lowest voice (bassline)**
```powershell
python src\scanbass.py --mode poly .\test.mp3
```
**B) Mix with bass instrument → MIDI bass**
```powershell
python src\scanbass.py --mode bass .\test.mp3
```

Outputs:
```
outputs\test_poly\bassline.mid
outputs\test_bass\bass.wav
outputs\test_bass\bassline.mid
```

## Tweaks (optional)
- Poly mode smoothness: `--frame-hz 30 --min-note-len 120 --gap-merge 120`
- Bass mode sensitivity: `--voicing-threshold 0.35`

## Notes
- First Demucs/Basic Pitch run downloads models to cache (one‑time).
- Warnings from torchaudio/pretty_midi are OK.
