"""FastAPI wrapper exposing ScanBass modes as an HTTP service (Render-friendly)."""

from __future__ import annotations

import asyncio
import logging
import os
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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Render nám dá PORT z env. Lokálně necháme 8000.
DEFAULT_HOST = os.getenv("SCANBASS_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("PORT", os.getenv("SCANBASS_PORT", "8000")))

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
    description="HTTP API for ScanBass. Submit audio and poll for MIDI outputs.",
    version="0.3.0",
)

# CORS – ať ti to frontend na Renderu / jiném hostingu hned bere
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
    # uděláme kratší ID, ale bez mezer – Render měl problém s názvem v URL
    stem = Path(input_name).stem or "input"
    safe_stem = "".join(c for c in stem if c.isalnum() or c in ("-", "_"))[:32]
    return f"{safe_stem}-{mode}-{uuid.uuid4().hex[:8]}"


def _job_output_dir(job_id: str) -> Path:
    out_dir = DEFAULT_OUTPUT_ROOT / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


async def _run_mode(mode: str, audio_path: Path, out_dir: Path, **params):
    """
    DŮLEŽITÉ:
    - importujeme až TADY → takže start uvicornu je rychlý
    - tohle je celý důvod, proč Render předtím neviděl port
    """
    if mode == "bass":
        from modes.bass_mode import run_bass_mode  # lazy import
        return await asyncio.to_thread(
            run_bass_mode,
            str(audio_path),
            str(out_dir),
            voicing_threshold=float(params.get("voicing_threshold", 0.5)),
        )

    if mode == "poly":
        from modes.poly_mode import run_poly_mode  # lazy import
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
    except Exception as exc:  # pragma: no cover
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
        # pokud mód vrátí dict → uložíme
        job.artifacts = {k: str(v) for k, v in (artifacts or {}).items()}


@app.get("/health")
async def health_check():
    # Renderu stačí, že tohle odpoví → uvidí port
    return {"status": "ok"}


@app.post("/jobs")
async def submit_job(
    file: UploadFile = File(...),
    mode: str = Form(..., description="Processing mode: bass or poly"),
    voicing_threshold: float = Form(0.5),
    frame_hz: int = Form(40),
    min_note_len_ms: int = Form(90),
    gap_merge_ms: int = Form(60),
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

    # tady máš pořád ZIP, frontend si z něj může vytáhnout bassline.mid
    temp_dir = Path(tempfile.mkdtemp(prefix="scanbass_zip_"))
    base_name = temp_dir / job_id
    archive_path = shutil.make_archive(str(base_name), "zip", root_dir=base_dir)

    background = BackgroundTask(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
    return FileResponse(archive_path, filename=f"{job_id}.zip", background=background)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "deploy_backend.src.web_service:app",
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        reload=False,
    )
