const API_BASE = "https://scanbass-backend.onrender.com";

const form = document.getElementById("uploadForm");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const statusDiv = document.getElementById("status");
const loader = document.getElementById("loader");
const preview = document.getElementById("preview");
const midiCanvas = document.getElementById("midiCanvas");
const midiDownload = document.getElementById("midiDownload");

// drag & drop behavior
["dragenter","dragover","dragleave","drop"].forEach(ev => {
  dropzone.addEventListener(ev, e => {
    e.preventDefault();
    e.stopPropagation();
  });
});
["dragenter","dragover"].forEach(ev => {
  dropzone.addEventListener(ev, () => dropzone.classList.add("active"));
});
["dragleave","drop"].forEach(ev => {
  dropzone.addEventListener(ev, () => dropzone.classList.remove("active"));
});
dropzone.addEventListener("drop", e => {
  const file = e.dataTransfer.files[0];
  if (file) {
    fileInput.files = e.dataTransfer.files;
    document.getElementById("dropLabel").textContent = "✅ " + file.name;
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) {
    document.getElementById("dropLabel").textContent = "✅ " + fileInput.files[0].name;
  }
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    statusDiv.textContent = "Please choose a file.";
    return;
  }

  // show loader
  loader.classList.remove("hidden");
  statusDiv.textContent = "Uploading…";
  preview.classList.add("hidden");
  midiDownload.innerHTML = "";

  const mode = document.querySelector('input[name="mode"]:checked').value;
  const fd = new FormData();
  fd.append("file", file);
  fd.append("mode", mode);

  try {
    const resp = await fetch(`${API_BASE}/jobs`, {
      method: "POST",
      body: fd
    });
    const data = await resp.json();
    if (!data.job_id) {
      statusDiv.textContent = "Failed to start job.";
      loader.classList.add("hidden");
      return;
    }

    const jobId = data.job_id;
    statusDiv.textContent = "Processing…";

    // poll
    const poll = setInterval(async () => {
      const r = await fetch(`${API_BASE}/jobs/${jobId}`);
      const job = await r.json();
      if (job.status === "succeeded") {
        clearInterval(poll);
        statusDiv.textContent = "✅ Done";
        loader.classList.add("hidden");
        // download ZIP (or MIDI)
        const downloadUrl = `${API_BASE}/jobs/${jobId}/download`;
        const blob = await (await fetch(downloadUrl)).blob();
        await handleMidiBlob(blob);
      } else if (job.status === "failed") {
        clearInterval(poll);
        statusDiv.textContent = "❌ Failed: " + (job.error || "");
        loader.classList.add("hidden");
      } else {
        statusDiv.textContent = "Processing… (" + job.status + ")";
      }
    }, 2500);

  } catch (err) {
    console.error(err);
    statusDiv.textContent = "Error: " + err.message;
    loader.classList.add("hidden");
  }
});

async function handleMidiBlob(blob) {
  // assume single MIDI in archive or response
  // create URL to download
  const url = URL.createObjectURL(blob);
  midiDownload.innerHTML = `<a href="${url}" download="scanbass.mid" draggable="true">⬇️ Download / drag to DAW</a>`;
  preview.classList.remove("hidden");

  // try to visualize
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const midi = new Midi(reader.result);
      drawMidi(midi);
    } catch (e) {
      console.warn("Cannot parse MIDI:", e);
    }
  };
  reader.readAsArrayBuffer(blob);
}

function drawMidi(midi) {
  const ctx = midiCanvas.getContext("2d");
  const w = midiCanvas.width;
  const h = midiCanvas.height;
  ctx.clearRect(0, 0, w, h);

  // background grid
  ctx.fillStyle = "rgba(255,255,255,0.01)";
  ctx.fillRect(0, 0, w, h);

  // simple horizontal lines
  ctx.strokeStyle = "rgba(255,255,255,0.03)";
  ctx.lineWidth = 1;
  for (let y = 0; y < h; y += 14) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // notes
  ctx.fillStyle = "rgba(0,209,255,0.95)";
  midi.tracks.forEach(track => {
    track.notes.forEach(note => {
      const x = (note.time || 0) * 55 % w;
      const dur = Math.max(4, (note.duration || 0.2) * 55);
      const y = h - (note.midi - 40) * 3;
      ctx.fillRect(x, y, dur, 6);
    });
  });
}
