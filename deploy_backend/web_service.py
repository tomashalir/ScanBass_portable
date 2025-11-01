from __future__ import annotations

"""
FastAPI wrapper exposing ScanBass modes as an HTTP service.
This version is Render-friendly:
- future import is at the very top
- no heavy imports at module import time
- binds to os.environ["PORT"] (Render) or 8000 locally
- lazy-loads modes inside _run_mode()
"""

import sys
import os
import asyncio
import logging
import subprocess
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field, asdict as dataclass_asdict
from pathlib import Path
from typing import Dict, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

# -----------------------------------------------------------------------------
# make sure Render sees our src/ folder
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logger = logging.getLogger(__name__)

# maximum amount of audio (in seconds) that the backend will process per job
MAX_UPLOAD_DURATION_SECONDS = 30.0


class UnsupportedAudioError(Exception):
    """Raised when a trimming strategy cannot decode the uploaded audio."""

# Render gives us PORT; locally we can leave 8000.
PORT_ENV = os.getenv("PORT")
if PORT_ENV and "SCANBASS_PORT" not in os.environ:
    os.environ["SCANBASS_PORT"] = PORT_ENV

# where to store job outputs
DEFAULT_OUTPUT_ROOT = Path(os.getenv("SCANBASS_OUTPUT_ROOT", "outputs")).resolve()
DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass
class JobState:
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
    description="Submit audio, get MIDI. Bass or poly.",
    version="0.2.1",
)

# CORS – kvůli tvému frontend/index.html
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOB_LOCK = asyncio.Lock()
JOBS: Dict[str, JobState] = {}


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "input.wav").suffix or ".wav"
    temp_dir = Path(tempfile.mkdtemp(prefix="scanbass_"))
    temp_path = temp_dir / f"upload{suffix}"
    with temp_path.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    upload.file.close()
    return _enforce_max_duration(temp_path, MAX_UPLOAD_DURATION_SECONDS)


def _enforce_max_duration(audio_path: Path, max_seconds: float) -> Path:
    """Trim an uploaded audio file to the first ``max_seconds`` seconds."""

    if max_seconds <= 0:
        return audio_path

    try:
        return _trim_with_soundfile(audio_path, max_seconds)
    except UnsupportedAudioError:
        logger.debug("Falling back to ffmpeg trimming for %s", audio_path)
        return _trim_with_ffmpeg(audio_path, max_seconds)


def _trim_with_soundfile(audio_path: Path, max_seconds: float) -> Path:
    try:
        import soundfile as sf  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency missing
        raise UnsupportedAudioError from exc

    try:
        with sf.SoundFile(str(audio_path)) as sound_file:
            samplerate = sound_file.samplerate
            total_frames = len(sound_file)
            if total_frames == 0:
                raise HTTPException(status_code=400, detail="Uploaded audio is empty")

            max_frames = int(max_seconds * samplerate)
            needs_trim = total_frames > max_frames
            frames_to_read = min(total_frames, max_frames)

            sound_file.seek(0)
            data = sound_file.read(frames=frames_to_read)
    except HTTPException:
        raise
    except Exception as exc:
        raise UnsupportedAudioError from exc

    if not needs_trim:
        return audio_path

    safe_suffixes = {".wav", ".flac", ".ogg", ".oga", ".aiff", ".aif", ".aifc"}
    suffix = audio_path.suffix.lower()
    target_path = audio_path
    if suffix not in safe_suffixes:
        target_path = audio_path.with_suffix(".wav")

    try:
        sf.write(str(target_path), data, samplerate)
    except Exception as exc:  # pragma: no cover - disk write failure
        raise HTTPException(status_code=500, detail="Failed to store trimmed audio") from exc

    if target_path != audio_path:
        try:
            audio_path.unlink()
        except OSError:
            pass

    actual_duration = frames_to_read / float(samplerate)
    logger.info(
        "Trimmed uploaded audio to %.2f seconds (original: %.2f seconds)",
        actual_duration,
        total_frames / float(samplerate),
    )

    return target_path


def _trim_with_ffmpeg(audio_path: Path, max_seconds: float) -> Path:
    try:
        import audioread  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency missing
        raise HTTPException(status_code=500, detail="Audio backend unavailable") from exc

    try:
        with audioread.audio_open(str(audio_path)) as reader:
            duration = reader.duration
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read uploaded audio") from exc

    if duration is None or duration <= 0:
        raise HTTPException(status_code=400, detail="Uploaded audio is empty")

    if duration <= max_seconds:
        return audio_path

    suffix = audio_path.suffix or ""
    temp_copy_path = audio_path.with_name(audio_path.stem + "_trimmed" + suffix)

    if suffix:
        if _run_ffmpeg_trim(audio_path, temp_copy_path, max_seconds, copy_codec=True):
            try:
                audio_path.unlink()
            except OSError:
                pass
            temp_copy_path.replace(audio_path)
            logger.info(
                "Trimmed uploaded audio from %.2f to %.2f seconds using ffmpeg (copy)",
                duration,
                max_seconds,
            )
            return audio_path

    temp_wav_path = audio_path.with_name(audio_path.stem + "_trimmed.wav")
    if _run_ffmpeg_trim(audio_path, temp_wav_path, max_seconds, copy_codec=False):
        try:
            audio_path.unlink()
        except OSError:
            pass
        final_path = audio_path.with_suffix(".wav")
        if final_path != temp_wav_path:
            try:
                final_path.unlink()
            except OSError:
                pass
            temp_wav_path.replace(final_path)
        else:
            final_path = temp_wav_path
        logger.info(
            "Trimmed uploaded audio from %.2f to %.2f seconds using ffmpeg (re-encode)",
            duration,
            max_seconds,
        )
        return final_path

    raise HTTPException(status_code=500, detail="Failed to trim uploaded audio")


def _run_ffmpeg_trim(
    source: Path, target: Path, max_seconds: float, *, copy_codec: bool
) -> bool:
    args = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-t",
        str(max_seconds),
    ]
    if copy_codec:
        args.extend(["-c", "copy"])
    else:
        args.extend(["-acodec", "pcm_s16le", "-ar", "44100"])
    args.append(str(target))

    try:
        result = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Audio backend unavailable")

    if result.returncode != 0:
        logger.warning(
            "ffmpeg trim failed (copy=%s) for %s: %s",
            copy_codec,
            source,
            result.stderr.decode(errors="ignore"),
        )
        try:
            target.unlink()
        except OSError:
            pass
        return False

    logger.debug(
        "ffmpeg trim succeeded for %s (copy=%s)",
        source,
        copy_codec,
    )
    return True


def _make_job_id(input_name: str, mode: str) -> str:
    stem = Path(input_name).stem or "input"
    return f"{stem}_{mode}_{uuid.uuid4().hex[:8]}"


def _job_output_dir(job_id: str) -> Path:
    out_dir = DEFAULT_OUTPUT_ROOT / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _optional_float(value: Optional[object]) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


async def _run_mode(mode: str, audio_path: Path, out_dir: Path, **params):
    # LAZY IMPORTS – tady se teprve tahají těžké věci
    if mode == "bass":
        from src.modes.bass_mode import run_bass_mode  # type: ignore

        return await asyncio.to_thread(
            run_bass_mode,
            str(audio_path),
            str(out_dir),
            voicing_threshold=float(params.get("voicing_threshold", 0.5)),
            segment_seconds=float(params.get("segment_seconds", 15.0)),
            overlap=float(params.get("segment_overlap", 0.1)),
        )

    if mode == "poly":
        from src.modes.poly_mode import run_poly_mode  # type: ignore

        min_freq = _optional_float(params.get("bp_min_frequency"))
        max_freq = _optional_float(params.get("bp_max_frequency"))

        return await asyncio.to_thread(
            run_poly_mode,
            str(audio_path),
            str(out_dir),
            frame_hz=int(params.get("frame_hz", 40)),
            min_note_len_ms=int(params.get("min_note_len_ms", 90)),
            gap_merge_ms=int(params.get("gap_merge_ms", 60)),
            onset_threshold=float(params.get("bp_onset_threshold", 0.5)),
            frame_threshold=float(params.get("bp_frame_threshold", 0.3)),
            basic_pitch_min_note_len_ms=float(params.get("bp_min_note_ms", 127.7)),
            minimum_frequency=min_freq,
            maximum_frequency=max_freq,
        )

    raise HTTPException(status_code=400, detail=f"Unsupported mode: {mode}")


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
    segment_seconds: float,
    segment_overlap: float,
    frame_hz: int,
    min_note_len_ms: int,
    gap_merge_ms: int,
    bp_onset_threshold: float,
    bp_frame_threshold: float,
    bp_min_note_ms: float,
    bp_min_frequency: Optional[float],
    bp_max_frequency: Optional[float],
):
    async with JOB_LOCK:
        job.status = "running"

    out_dir = _job_output_dir(job.job_id)
    params = dict(
        voicing_threshold=voicing_threshold,
        segment_seconds=segment_seconds,
        segment_overlap=segment_overlap,
        frame_hz=frame_hz,
        min_note_len_ms=min_note_len_ms,
        gap_merge_ms=gap_merge_ms,
        bp_onset_threshold=bp_onset_threshold,
        bp_frame_threshold=bp_frame_threshold,
        bp_min_note_ms=bp_min_note_ms,
        bp_min_frequency=bp_min_frequency,
        bp_max_frequency=bp_max_frequency,
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
        job.artifacts = {k: str(v) for k, v in (artifacts or {}).items()}


@app.get("/health")
@app.get("/healthz")
async def health():
    return {"status": "ok"}


@app.post("/jobs")
async def submit_job(
    file: UploadFile = File(...),
    mode: str = Form(...),
    voicing_threshold: float = Form(0.5),
    segment_seconds: float = Form(15.0),
    segment_overlap: float = Form(0.1),
    frame_hz: int = Form(40),
    min_note_len_ms: int = Form(90),
    gap_merge_ms: int = Form(60),
    bp_onset_threshold: float = Form(0.5),
    bp_frame_threshold: float = Form(0.3),
    bp_min_note_ms: float = Form(127.7),
    bp_min_frequency: Optional[float] = Form(None),
    bp_max_frequency: Optional[float] = Form(None),
):
    mode_normalized = mode.lower().strip()
    if mode_normalized not in {"bass", "poly"}:
        raise HTTPException(status_code=400, detail="mode must be 'bass' or 'poly'")

    saved_path = _save_upload(file)
    job_id = _make_job_id(file.filename or saved_path.stem, mode_normalized)
    job = JobState(
        job_id=job_id,
        mode=mode_normalized,
        input_name=file.filename or saved_path.name,
    )

    async with JOB_LOCK:
        JOBS[job_id] = job

    asyncio.create_task(
        _execute_job(
            job,
            saved_path,
            voicing_threshold=voicing_threshold,
            segment_seconds=segment_seconds,
            segment_overlap=segment_overlap,
            frame_hz=frame_hz,
            min_note_len_ms=min_note_len_ms,
            gap_merge_ms=gap_merge_ms,
            bp_onset_threshold=bp_onset_threshold,
            bp_frame_threshold=bp_frame_threshold,
            bp_min_note_ms=bp_min_note_ms,
            bp_min_frequency=bp_min_frequency,
            bp_max_frequency=bp_max_frequency,
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

    tmp_dir = Path(tempfile.mkdtemp(prefix="scanbass_zip_"))
    base_name = tmp_dir / job_id
    archive_path = shutil.make_archive(str(base_name), "zip", root_dir=base_dir)

    background = BackgroundTask(lambda: shutil.rmtree(tmp_dir, ignore_errors=True))
    return FileResponse(archive_path, filename=f"{job_id}.zip", background=background)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("SCANBASS_HOST", "0.0.0.0")
    port = int(os.getenv("SCANBASS_PORT") or os.getenv("PORT") or "8000")
    uvicorn.run(app, host=host, port=port, reload=False)
