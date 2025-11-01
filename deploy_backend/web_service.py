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
import logging
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field, asdict as dataclass_asdict
from typing import Dict, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from modes.bass_mode import run_bass_mode
from modes.poly_mode import run_poly_mode

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT = Path(os.getenv("SCANBASS_OUTPUT_ROOT", "outputs")).resolve()
DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

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

def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "input.wav").suffix or ".wav"
    temp_dir = Path(tempfile.mkdtemp(prefix="scanbass_"))
    temp_path = temp_dir / f"upload{suffix}"
    with temp_path.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    upload.file.close()
    return temp_path

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
    port = int(os.getenv("SCANBASS_PORT", "8000"))
    uvicorn.run("src.web_service:app", host=host, port=port, reload=False)
