"""FastAPI wrapper exposing ScanBass modes as an HTTP service."""

from __future__ import annotations

import sys
from pathlib import Path

# --- přidáme cestu k projektu (src) pro importy ---
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------------------

import asyncio
import importlib
import logging
import os
from typing import Optional, Tuple

PORT_ENV = os.getenv("PORT")
if PORT_ENV and "SCANBASS_PORT" not in os.environ:
    os.environ["SCANBASS_PORT"] = PORT_ENV
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field, asdict as dataclass_asdict
from typing import Dict, Literal

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from modes.bass_mode import run_bass_mode
from modes.poly_mode import run_poly_mode

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT = Path(os.getenv("SCANBASS_OUTPUT_ROOT", "outputs")).resolve()
DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_DURATION_S = 30.0
TARGET_SAMPLE_RATE = 44100
TRIMMED_UPLOAD_NAME = "input_30s.wav"

_TORCH = None
_TORCHAUDIO = None
_LIBROSA = None

@dataclass
class JobState:
    """Track ScanBass processing jobs for asynchronous clients."""

    job_id: str
    mode: Literal["bass", "poly"]
    input_name: str
    status: Literal["queued", "running", "succeeded", "failed"] = "queued"
    output_dir: Optional[str] = None
    artifacts: Optional[Dict[str, str]] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        data = dataclass_asdict(self)
        if data["artifacts"] is None:
            data["artifacts"] = {}
        return data

app = FastAPI(
    title="ScanBass Online",
    description=(
        "HTTP API for ScanBass. Submit audio to the bass or poly transcription"
        " pipelines and poll for MIDI outputs."
    ),
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOB_LOCK = asyncio.Lock()
JOBS: Dict[str, JobState] = {}

def _max_frames(sr: int) -> int:
    return max(int(MAX_UPLOAD_DURATION_S * sr), 0)


def _limit_duration(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    max_frames = _max_frames(sr)
    if max_frames and audio.shape[0] > max_frames:
        audio = audio[:max_frames]
    if audio.size == 0:
        raise HTTPException(status_code=400, detail="Audio file contains no usable samples.")
    return np.ascontiguousarray(audio.astype(np.float32, copy=False)), sr


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _get_torch_modules() -> Tuple[Optional[object], Optional[object]]:
    global _TORCH, _TORCHAUDIO
    if _TORCH is not None and _TORCHAUDIO is not None:
        return _TORCH, _TORCHAUDIO
    if not _module_available("torchaudio"):
        return None, None
    _TORCHAUDIO = importlib.import_module("torchaudio")
    _TORCH = importlib.import_module("torch")
    return _TORCH, _TORCHAUDIO


def _get_librosa():
    global _LIBROSA
    if _LIBROSA is not None:
        return _LIBROSA
    if not _module_available("librosa"):
        raise HTTPException(
            status_code=500,
            detail="librosa is required to decode this audio file.",
        )
    _LIBROSA = importlib.import_module("librosa")
    return _LIBROSA


def _read_audio_with_soundfile(path: Path) -> Optional[Tuple[np.ndarray, int]]:
    try:
        with sf.SoundFile(str(path), mode="r") as source:
            sr = int(source.samplerate)
            if sr <= 0:
                raise ValueError("Invalid sample rate")
            frames = source.read(
                frames=int(np.ceil(MAX_UPLOAD_DURATION_S * sr)),
                dtype="float32",
                always_2d=True,
            )
    except RuntimeError:  # libsndfile unsupported format
        return None
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=400, detail=f"Failed to read audio: {exc}") from exc

    return np.asarray(frames, dtype=np.float32), sr


def _resample_with_librosa(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    librosa = _get_librosa()
    channels = [
        librosa.resample(
            audio[:, ch],
            orig_sr=sr,
            target_sr=TARGET_SAMPLE_RATE,
            res_type="kaiser_fast",
            scale=False,
        )
        for ch in range(audio.shape[1])
    ]
    min_len = min(len(ch) for ch in channels) if channels else 0
    if min_len == 0:
        raise HTTPException(
            status_code=400,
            detail="Audio file contains no usable samples after resampling.",
        )
    stacked = np.stack([ch[:min_len] for ch in channels], axis=1)
    return _limit_duration(stacked, TARGET_SAMPLE_RATE)


def _resample_to_target(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    limited, sr = _limit_duration(audio, sr)
    if sr == TARGET_SAMPLE_RATE:
        return limited, sr

    torch_mod, torchaudio_mod = _get_torch_modules()
    if torch_mod is not None and torchaudio_mod is not None:
        waveform = torch_mod.from_numpy(limited.T)
        resampled = torchaudio_mod.functional.resample(
            waveform,
            orig_freq=sr,
            new_freq=TARGET_SAMPLE_RATE,
            lowpass_filter_width=32,
            rolloff=0.99,
            resampling_method="sinc_interp_kaiser",
        )
        stacked = resampled.transpose(0, 1).contiguous().cpu().numpy()
        return _limit_duration(stacked, TARGET_SAMPLE_RATE)

    return _resample_with_librosa(limited, sr)


def _load_with_librosa(path: Path) -> Tuple[np.ndarray, int]:
    try:
        librosa = _get_librosa()
        audio, sr = librosa.load(
            str(path),
            sr=TARGET_SAMPLE_RATE,
            mono=False,
            duration=MAX_UPLOAD_DURATION_S,
            res_type="kaiser_fast",
            dtype=np.float32,
        )
    except Exception as exc:  # pragma: no cover - runtime guard
        raise HTTPException(status_code=400, detail=f"Failed to decode audio: {exc}") from exc

    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    audio = np.transpose(audio)
    return _limit_duration(audio, sr)


def _load_and_trim_audio(path: Path) -> Tuple[np.ndarray, int]:
    """Load at most MAX_UPLOAD_DURATION_S seconds of audio and return mono/stereo float32."""

    direct = _read_audio_with_soundfile(path)
    if direct is not None:
        audio, sr = direct
        return _resample_to_target(audio, sr)

    logger.debug("Falling back to librosa decoder for %s", path)
    return _load_with_librosa(path)


def _prepare_upload(path: Path) -> Path:
    """Convert the uploaded audio to a 30 s WAV clip for downstream processing."""
    audio, sr = _load_and_trim_audio(path)
    trimmed_path = path.with_name(TRIMMED_UPLOAD_NAME)
    try:
        sf.write(trimmed_path, np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")
    except Exception as exc:  # pragma: no cover - runtime guard
        raise HTTPException(status_code=500, detail=f"Failed to write temp audio: {exc}") from exc

    if trimmed_path != path:
        try:
            path.unlink()
        except OSError:
            pass

    logger.debug(
        "Prepared upload saved to %s (%.2f s at %d Hz)",
        trimmed_path,
        audio.shape[0] / float(sr) if sr else 0.0,
        sr,
    )

    return trimmed_path


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "input.wav").suffix or ".wav"
    temp_dir = Path(tempfile.mkdtemp(prefix="scanbass_"))
    temp_path = temp_dir / f"upload{suffix}"
    with temp_path.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    upload.file.close()
    prepared_path = _prepare_upload(temp_path)
    if prepared_path.name != TRIMMED_UPLOAD_NAME:
        logger.debug("Prepared upload renamed to %s", prepared_path)
    return prepared_path

def _make_job_id(input_name: str, mode: str) -> str:
    stem = Path(input_name).stem or "input"
    return f"{stem}_{mode}_{uuid.uuid4().hex[:8]}"

def _job_output_dir(job_id: str) -> Path:
    out_dir = DEFAULT_OUTPUT_ROOT / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

async def _run_mode(mode: str, audio_path: Path, out_dir: Path, **params):
    if mode == "bass":
        return await asyncio.to_thread(
            run_bass_mode,
            str(audio_path),
            str(out_dir),
            voicing_threshold=float(params.get("voicing_threshold", 0.5)),
        )
    if mode == "poly":
        return await asyncio.to_thread(
            run_poly_mode,
            str(audio_path),
            str(out_dir),
            frame_hz=int(params.get("frame_hz", 40)),
            min_note_len_ms=int(params.get("min_note_len_ms", 90)),
            gap_merge_ms=int(params.get("gap_merge_ms", 60)),
        )
    raise ValueError(f"Unsupported mode: {mode}")

async def _get_job(job_id: str) -> JobState:
    async with JOB_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job ID")
        return job

async def _execute_job(
    job: JobState,
    saved_path: Path,
    *,
    voicing_threshold: float,
    frame_hz: int,
    min_note_len_ms: int,
    gap_merge_ms: int,
):
    async with JOB_LOCK:
        job.status = "running"

    out_dir = _job_output_dir(job.job_id)
    params = dict(
        voicing_threshold=voicing_threshold,
        frame_hz=frame_hz,
        min_note_len_ms=min_note_len_ms,
        gap_merge_ms=gap_merge_ms,
    )

    try:
        artifacts = await _run_mode(job.mode, saved_path, out_dir, **params)
    except Exception as exc:
        logger.exception("ScanBass job failed")
        async with JOB_LOCK:
            job.status = "failed"
            job.output_dir = str(out_dir)
            job.error = str(exc)
            job.artifacts = {}
        return
    finally:
        try:
            shutil.rmtree(saved_path.parent)
        except OSError:
            pass

    async with JOB_LOCK:
        job.status = "succeeded"
        job.output_dir = str(out_dir)
        job.artifacts = {key: str(value) for key, value in (artifacts or {}).items()}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/jobs")
async def submit_job(
    file: UploadFile = File(...),
    mode: str = Form(..., description="Processing mode: bass or poly"),
    voicing_threshold: float = Form(0.5, description="Bass mode voicing threshold"),
    frame_hz: int = Form(40, description="Poly mode frame rate"),
    min_note_len_ms: int = Form(90, description="Poly mode minimum note length"),
    gap_merge_ms: int = Form(60, description="Poly mode gap merge threshold"),
):
    mode_normalized = mode.lower().strip()
    if mode_normalized not in {"bass", "poly"}:
        raise HTTPException(status_code=400, detail="mode must be 'bass' or 'poly'")

    saved_path = _save_upload(file)
    job_id = _make_job_id(file.filename or saved_path.stem, mode_normalized)
    job = JobState(job_id=job_id, mode=mode_normalized, input_name=file.filename or saved_path.name)

    async with JOB_LOCK:
        JOBS[job_id] = job

    asyncio.create_task(
        _execute_job(
            job,
            saved_path,
            voicing_threshold=voicing_threshold,
            frame_hz=frame_hz,
            min_note_len_ms=min_note_len_ms,
            gap_merge_ms=gap_merge_ms,
        )
    )

    return {"job_id": job_id, "status": job.status}

@app.get("/jobs")
async def list_jobs():
    async with JOB_LOCK:
        return {job_id: job.to_dict() for job_id, job in JOBS.items()}

@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    job = await _get_job(job_id)
    return job.to_dict()

@app.get("/jobs/{job_id}/download")
async def download_results(job_id: str):
    job = await _get_job(job_id)
    if job.status != "succeeded" or not job.output_dir:
        raise HTTPException(status_code=409, detail="Job not finished yet")

    base_dir = Path(job.output_dir)
    if not base_dir.exists():
        raise HTTPException(status_code=404, detail="Job output missing")

    temp_dir = Path(tempfile.mkdtemp(prefix="scanbass_zip_"))
    base_name = temp_dir / job_id
    archive_path = shutil.make_archive(str(base_name), "zip", root_dir=base_dir)

    background = BackgroundTask(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
    return FileResponse(archive_path, filename=f"{job_id}.zip", background=background)

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("SCANBASS_HOST", "0.0.0.0")
    port = int(os.getenv("SCANBASS_PORT") or os.getenv("PORT") or "8000")
    uvicorn.run(app, host=host, port=port, reload=False)
