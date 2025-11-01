# ScanBass GUI — Notes-only export (prevents DAW tempo override)
# Build: No-Quantize, robust DnD, BPM/Scale hidden, Force re-scan, no lambda/leak of `e`
import sys, threading, urllib.parse
from pathlib import Path
from statistics import median

DEBUG = False
DEBUG_DND = False

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except Exception:
    raise RuntimeError("Tkinter is required.")

HAS_DND = False
try:
    from tkinterdnd2 import DND_FILES, DND_TEXT, TkinterDnD
    HAS_DND = True
except Exception:
    DND_FILES = 'DND_Files'
    DND_TEXT = 'DND_Text'
    TkinterDnD = None
    HAS_DND = False

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
for p in {HERE, PROJECT_ROOT}:
    if str(p) not in sys.path:
        sys.path.append(str(p))

# Ensure package import (src/modes)
modes_dir = HERE / "modes"
if modes_dir.is_dir():
    init = modes_dir / "__init__.py"
    if not init.exists():
        try: init.write_text("")
        except Exception: pass

from modes.poly_mode import run_poly_mode

# ---- Colors & constants ----
COL_BG = "#0b1015"; COL_PANEL = "#111821"; COL_PANEL_BORDER = "#1e2a35"
COL_TEXT = "#d8e1ea"; COL_TEXT_MUTED = "#8fa0ad"; COL_SUCCESS = "#43b37f"
COL_ACCENT = "#3aa775"; COL_ACCENT_DARK = "#2e8f63"
COL_CANVAS_BG = "#0f141a"; COL_CANVAS_ROW_DARK = "#0d1217"; COL_CANVAS_ROW_LIGHT = "#11171e"
COL_GRID = "#24303a"; COL_OCT = "#2b3843"; COL_NOTE = "#3aa775"; COL_NOTE_LOW = "#2e8f63"

DEFAULT_BPM = 120.0; BEATS_PER_BAR = 4

def _clip_bpm(bpm: float) -> float:
    if bpm <= 0: return DEFAULT_BPM
    while bpm < 60.0: bpm *= 2.0
    while bpm > 180.0: bpm /= 2.0
    return bpm

def _find_existing_midi(out_dir: Path) -> Path | None:
    try:
        cand = out_dir / "bassline.mid"
        if cand.exists(): return cand
        mids = sorted(out_dir.glob("*.mid"), key=lambda x: x.stat().st_mtime, reverse=True)
        return mids[0] if mids else None
    except Exception:
        return None

def _detect_bpm(midi_path: Path) -> float|None:
    try:
        import pretty_midi
    except Exception:
        return None
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception:
        return None
    # tempo events
    try:
        times, tempi = pm.get_tempo_changes()
        if len(tempi) > 0:
            bpm = float(tempi[0])
            if bpm > 0: return _clip_bpm(bpm)
    except Exception:
        pass
    # IOI median
    try:
        onsets = sorted([float(n.start) for inst in pm.instruments for n in inst.notes])
        diffs = [b-a for a,b in zip(onsets,onsets[1:]) if 0.08 < (b-a) < 1.5]
        if len(diffs) >= 3:
            ioi = median(diffs)
            if ioi > 1e-6:
                return _clip_bpm(60.0/ioi)
    except Exception:
        pass
    # estimate_tempo fallback
    try:
        import pretty_midi
        est = float(pretty_midi.PrettyMIDI(str(midi_path)).estimate_tempo())
        if est > 0: return _clip_bpm(est)
    except Exception:
        pass
    return None

# -------------------- Piano Roll --------------------
class PianoRoll(ttk.Frame):
    def __init__(self, master, height=320):
        super().__init__(master)
        self.canvas = tk.Canvas(self, height=height, bg=COL_CANVAS_BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.hbar.set)
        self.hbar.pack(fill=tk.X, side=tk.BOTTOM)
        self.height = height; self.margin_left = 36
        self._midi = None; self.bpm = DEFAULT_BPM; self.limit_bars = 16; self.beats_per_bar = BEATS_PER_BAR
        self.canvas.bind("<Configure>", self._on_resize)

    def set_bpm(self, bpm): self.bpm = float(bpm) if bpm and bpm > 0 else DEFAULT_BPM
    def set_limit_bars(self, bars:int): self.limit_bars = max(1, int(bars))
    def _on_resize(self, _e):
        if self._midi: self.render_from_data(*self._midi)
    def _key_is_black(self, p): return (p % 12) in (1,3,6,8,10)

    def render_from_midi_file(self, midi_path: Path):
        try:
            import pretty_midi
            pm = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception as exc:
            self._draw_message(f"MIDI preview error: {exc}"); return
        notes=[]; min_p, max_p = 127, 0; end_t=0.0
        for inst in pm.instruments:
            for n in inst.notes:
                notes.append((n.start,n.end,n.pitch,n.velocity))
                if n.pitch<min_p:min_p=n.pitch
                if n.pitch>max_p:max_p=n.pitch
                if n.end>end_t:end_t=n.end
        if not notes: self._draw_message("Empty MIDI — no notes"); return
        self._midi=(notes,min_p,max_p,end_t); self.render_from_data(notes,min_p,max_p,end_t)

    def render_from_data(self, notes, min_p, max_p, end_t):
        c=self.canvas; c.delete("all")
        H=self.height; Wvis=max(320,int(c.winfo_width()) or 900)
        total_secs=(60.0/max(1e-6,self.bpm))*self.beats_per_bar*self.limit_bars
        px_per_sec=Wvis/max(0.25,total_secs); total_w=int(self.margin_left+total_secs*px_per_sec+20)
        p_low=max(0,min_p-2); p_high=min(127,max_p+2); n_keys=(p_high-p_low+1); key_h=H/max(1,n_keys)
        for i in range(n_keys):
            pitch=p_high-i; y0=i*key_h; y1=y0+key_h
            fill=COL_CANVAS_ROW_DARK if self._key_is_black(pitch) else COL_CANVAS_ROW_LIGHT
            c.create_rectangle(0,y0,total_w,y1,fill=fill,outline="")
        beat_sec=60.0/max(1e-6,self.bpm); n_beats=self.limit_bars*self.beats_per_bar
        for b in range(n_beats+1):
            t=b*beat_sec; x=self.margin_left+t*px_per_sec; is_bar=(b%self.beats_per_bar==0)
            c.create_line(x,0,x,H,fill=(COL_OCT if is_bar else COL_GRID),width=(2 if is_bar else 1))
            if is_bar: c.create_text(x+4,12,text=f"Bar {b//self.beats_per_bar+1}",fill=COL_TEXT_MUTED,anchor="w",font=("",9))
        window_end=total_secs
        for (st,en,pitch,vel) in notes:
            if st>=window_end: continue
            x0=self.margin_left+max(0.0,st)*px_per_sec; x1=self.margin_left+min(window_end,en)*px_per_sec
            if x1<=self.margin_left: continue
            i=(p_high-pitch); y0=i*key_h+2; y1=(i+1)*key_h-2
            fill=COL_NOTE if max(0,min(127,vel))/127.0>0.5 else COL_NOTE_LOW
            c.create_rectangle(x0,y0,max(x0+1,x1),y1,fill=fill,outline="")
        c.config(scrollregion=(0,0,total_w,H))

    def _draw_message(self,text):
        c=self.canvas; c.delete("all")
        c.create_rectangle(0,0,c.winfo_width(),self.height,fill=COL_CANVAS_BG,outline="")
        c.create_text(c.winfo_width()//2,self.height//2,text=text,fill=COL_TEXT_MUTED,font=("",10))

# --------- Notes-only writer (no tempo/TS meta) ---------
def _write_notes_only(in_midi: Path, out_midi: Path, bpm: float, tpb:int=480):
    import pretty_midi
    import mido
    pm = pretty_midi.PrettyMIDI(str(in_midi))
    mid = mido.MidiFile(type=1, ticks_per_beat=tpb)
    track = mido.MidiTrack(); mid.tracks.append(track)
    scale = (bpm/60.0) if (bpm and bpm>0) else (DEFAULT_BPM/60.0)
    events = []
    for inst in pm.instruments:
        for n in inst.notes:
            s_tick = int(round(n.start * scale * tpb))
            e_tick = int(round(n.end   * scale * tpb))
            if e_tick <= s_tick: e_tick = s_tick + 1
            vel = max(1, min(127, int(n.velocity)))
            events.append(('off', e_tick, n.pitch, 0))
            events.append(('on',  s_tick, n.pitch, vel))
    events.sort(key=lambda x: (x[1], 0 if x[0]=='off' else 1))
    last = 0
    for kind, tick, pitch, vel in events:
        dt = max(0, tick - last); last = tick
        if kind == 'on':
            track.append(mido.Message('note_on', note=int(pitch), velocity=int(vel), time=dt, channel=0))
        else:
            track.append(mido.Message('note_off', note=int(pitch), velocity=0, time=dt, channel=0))
    track.append(mido.MetaMessage('end_of_track', time=0))
    mid.save(str(out_midi))

class ScanBassGUI:
    def __init__(self, root):
        self.root=root; self.root.title("ScanBass — Bassline to MIDI")
        self.root.geometry("980x740"); self.root.configure(bg=COL_BG)
        self.selected_path=None; self.last_result=None; self.drag_file=None
        self.current_bpm=DEFAULT_BPM

        # ---- Styles ----
        style=ttk.Style(root)
        try: style.theme_use("clam")
        except Exception: pass
        style.configure("TFrame", background=COL_BG)
        style.configure("Card.TFrame", background=COL_PANEL, relief="groove", bordercolor=COL_PANEL_BORDER, borderwidth=1, padding=16)
        style.configure("TLabel", background=COL_BG, foreground=COL_TEXT)
        style.configure("Muted.TLabel", background=COL_BG, foreground=COL_TEXT_MUTED)
        style.configure("Success.TLabel", background=COL_BG, foreground=COL_SUCCESS)
        style.configure("Dark.TButton", background=COL_PANEL, foreground=COL_TEXT, bordercolor=COL_PANEL_BORDER, padding=6)
        style.configure("Dark.Horizontal.TProgressbar", troughcolor=COL_PANEL, background=COL_ACCENT, lightcolor=COL_ACCENT, darkcolor=COL_ACCENT_DARK, bordercolor=COL_PANEL_BORDER)

        # ---- Status bar ----
        status=ttk.Frame(root, padding=(10,6), style="TFrame"); status.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_var=tk.StringVar(value="Ready")
        ttk.Label(status, textvariable=self.status_var, style="Muted.TLabel").pack(side=tk.LEFT)
        self.prog=ttk.Progressbar(status, mode="determinate", maximum=100, value=0, length=240, style="Dark.Horizontal.TProgressbar")
        self.prog.pack(side=tk.RIGHT)

        # ---- Main ----
        container=ttk.Frame(root, padding=12, style="TFrame"); container.pack(fill=tk.BOTH, expand=True)

        # Select/Drop
        self.drop_card=ttk.Frame(container, style="Card.TFrame"); self.drop_card.pack(fill=tk.X)
        ttk.Label(self.drop_card, text="Select or drop an audio file", font=("",12,"bold")).pack(pady=(4,2))
        ttk.Label(self.drop_card, text="mp3 • wav • aiff • flac • m4a", style="Muted.TLabel").pack(pady=(0,8))
        self.drop_area=tk.Label(self.drop_card, text="Click to choose\n\n…or drop here", bd=1, relief=tk.SOLID, height=6, cursor="hand2", bg="#0f151c", fg=COL_TEXT)
        self.drop_area.pack(fill=tk.BOTH, expand=True); self.drop_area.bind("<Button-1>", lambda _e: self.choose_file())
        if HAS_DND and TkinterDnD:
            self.drop_area.drop_target_register(DND_FILES, DND_TEXT, 'text/uri-list')
            self.drop_area.dnd_bind("<<Drop>>", self.on_drop)

        # Result card
        self.result_card=ttk.Frame(container, style="Card.TFrame"); self.result_card.pack(fill=tk.BOTH, expand=True, pady=(12,0))
        head=ttk.Frame(self.result_card, style="TFrame"); head.pack(fill=tk.X)
        self.file_var=tk.StringVar(value="File: —")
        ttk.Label(head, textvariable=self.file_var, font=("",11,"bold")).pack(side=tk.LEFT)

        # Controls row
        ctrl=ttk.Frame(self.result_card, style="TFrame"); ctrl.pack(fill=tk.X, pady=(8,6))
        self.drag_label=tk.Label(ctrl, text="Drag to DAW: (no file yet)", bd=1, relief=tk.SOLID, padx=10, pady=6, cursor="hand2", bg="#0f151c", fg=COL_TEXT)
        self.drag_label.pack(side=tk.LEFT)
        self.drag_label.bind("<Double-Button-1>", lambda _e:self.open_output_folder())
        if HAS_DND and TkinterDnD:
            try:
                self.drag_label.drag_source_register(1, DND_FILES, 'text/uri-list')
            except Exception:
                self.drag_label.drag_source_register(1, DND_FILES)
            self.drag_label.dnd_bind("<<DragInitCmd>>", self.on_drag_init)
            self.drag_label.dnd_bind("<<DragEndCmd>>", self.on_drag_end)

        actions=ttk.Frame(ctrl, style="TFrame"); actions.pack(side=tk.RIGHT)
        self.force_var=tk.BooleanVar(value=False)
        ttk.Checkbutton(actions, text="Force re-scan", variable=self.force_var).grid(row=0,column=0,padx=(8,0))
        self.open_btn=ttk.Button(actions, text="Open folder", command=self.open_output_folder, state=tk.DISABLED, style="Dark.TButton"); self.open_btn.grid(row=0,column=1,padx=(8,0))
        self.explorer_btn=ttk.Button(actions, text="Explorer (select file)", command=self.open_explorer_select, state=tk.DISABLED, style="Dark.TButton"); self.explorer_btn.grid(row=0,column=2,padx=(8,0))
        self.copy_file_btn=ttk.Button(actions, text="Copy file", command=self.copy_midi_file, state=tk.DISABLED, style="Dark.TButton"); self.copy_file_btn.grid(row=0,column=3,padx=(8,0))
        self.scan_btn=ttk.Button(actions, text="Scan now", command=self._start_run, style="Dark.TButton"); self.scan_btn.grid(row=0,column=4,padx=(8,0))
        self.notes_btn=ttk.Button(actions, text="Export notes-only", command=self.export_notes_only, state=tk.DISABLED, style="Dark.TButton"); self.notes_btn.grid(row=0,column=5,padx=(8,0))

        # Preview (+ accept drop)
        self.preview=PianoRoll(self.result_card, height=320); self.preview.pack(fill=tk.BOTH, expand=True, pady=(6,0))
        self.dnd_hint=tk.Label(self.preview.canvas, text="⇢ Drag this MIDI to your DAW (CTRL for compat)", bg="#14202a", fg=COL_TEXT, bd=1, relief=tk.SOLID, padx=10, pady=6)
        self.dnd_hint.place_forget()
        if HAS_DND and TkinterDnD:
            try:
                self.preview.canvas.drag_source_register(1, DND_FILES, 'text/uri-list')
            except Exception:
                self.preview.canvas.drag_source_register(1, DND_FILES)
            self.preview.canvas.dnd_bind("<<DragInitCmd>>", self.on_drag_init)
            self.preview.canvas.dnd_bind("<<DragEndCmd>>", self.on_drag_end)
            self.preview.canvas.drop_target_register(DND_FILES, DND_TEXT, 'text/uri-list')
            self.preview.canvas.dnd_bind("<<Drop>>", self.on_drop)
        self.preview.canvas.configure(cursor="arrow")
        self.preview.canvas.bind("<Configure>", lambda _e: self._position_hint())

        self._ready=False; self._prog_timer=None

    # ---------- helpers ----------
    def _status(self, msg):
        if hasattr(self, "status_var"):
            self.status_var.set(msg)
        if DEBUG or DEBUG_DND:
            try: print(msg)
            except Exception: pass

    def _indicate_ready(self, ready:bool, midi_name:str|None=None):
        self._ready=ready
        if ready:
            self.drag_label.config(bg="#143824", fg=COL_TEXT)
            self.preview.canvas.configure(cursor="hand2")
            self._show_hint(True)
            if midi_name: self.drag_label.config(text=f"Drag to DAW: {midi_name}")
        else:
            self.drag_label.config(bg="#0f151c", fg=COL_TEXT, text="Drag to DAW: (no file yet)")
            self.preview.canvas.configure(cursor="arrow")
            self._show_hint(False)

        state = (tk.NORMAL if ready else tk.DISABLED)
        try:
            self.open_btn.config(state=state); self.copy_file_btn.config(state=state); self.explorer_btn.config(state=state); self.notes_btn.config(state=state)
        except Exception:
            pass

    def _show_hint(self, show:bool):
        if show: self._position_hint(); self.dnd_hint.lift(); self.dnd_hint.place()
        else: self.dnd_hint.place_forget()

    def _position_hint(self):
        try:
            w=int(self.preview.canvas.winfo_width()); x=(w//2)-220
            if x<8: x=8
            self.dnd_hint.place(x=x, y=10)
        except Exception: pass

    # ---------- user actions ----------
    def choose_file(self):
        f=filedialog.askopenfilename(title="Choose audio file", filetypes=[("Audio","*.mp3 *.wav *.aiff *.flac *.m4a"), ("All files","*.*")])
        if f: self._set_path(Path(f))

    def on_drop(self, event):
        raw = event.data or ""
        p = self._extract_first_path_from_drop(raw)
        if DEBUG_DND: print("DROP RAW:", repr(raw), "->", p)
        if p: self._set_path(p)
        else: self._status("Unsupported drop")

    def _set_path(self, path:Path):
        self.selected_path=path; self.file_var.set(f"File: {path.name}")
        self._status(f"Selected: {path.name}")
        self._indicate_ready(False); self._start_run()

    # Robust extractor
    def _extract_first_path_from_drop(self, data: str) -> Path|None:
        s = (data or "")
        if not s: return None
        # Tcl list split (handles CF_HDROP braces)
        try:
            parts = self.root.tk.splitlist(s)
            for item in parts:
                P = self._convert_item_to_path(item)
                if P: return P
        except Exception:
            pass
        # uri-list / plain lines
        for line in s.replace("\r","").split("\n"):
            P = self._convert_item_to_path(line.strip())
            if P: return P
        return None

    def _convert_item_to_path(self, item: str) -> Path|None:
        if not item: return None
        t = item.strip().strip('"').strip("'")
        if not t: return None
        if t.startswith("file:"):
            u = urllib.parse.urlparse(t)
            path = urllib.parse.unquote(u.path)
            if sys.platform.startswith("win") and path.startswith("/") and len(path) > 2 and path[2] == ":":
                path = path[1:]
            P = Path(path)
        else:
            P = Path(t)
        return P if (P.exists() and P.is_file()) else None

    # ---------- processing ----------
    def _start_run(self):
        if not self.selected_path:
            self._status("Select a file first"); return
        self._status("Starting…"); self._prog_reset(); self._prog_start()
        threading.Thread(target=self._run_worker, daemon=True).start()

    def _run_worker(self):
        try:
            in_path=self.selected_path; assert in_path is not None
            out_root=PROJECT_ROOT/"outputs"; out_root.mkdir(parents=True, exist_ok=True)
            out_dir=out_root/f"{in_path.stem}_poly"; out_dir.mkdir(parents=True, exist_ok=True)

            existing=_find_existing_midi(out_dir)
            if existing and not self.force_var.get():
                if DEBUG: print("Using existing MIDI:", existing)
                self._ui(lambda p=existing: self._finish_with_midi(p, info="Using existing result")); return

            self._ui(lambda: self._status("Converting (poly)…"))
            res=None
            try:
                res=run_poly_mode(str(in_path), str(out_dir))
            except TypeError:
                res=run_poly_mode(str(in_path))

            midi_path=None
            try:
                if isinstance(res, dict):
                    cand=res.get("midi") or res.get("bass_midi") or res.get("output_midi")
                    if cand: midi_path=Path(cand)
            except Exception:
                pass
            if (not midi_path) or (not midi_path.exists()):
                midi_path=_find_existing_midi(out_dir)

            if not midi_path or not midi_path.exists():
                raise RuntimeError("No MIDI file produced.")

            self._ui(lambda p=midi_path: self._finish_with_midi(p))
        except Exception as exc:
            err_msg = f"{exc}"
            self._ui(lambda msg=err_msg: self._status(f"ERROR: {msg}"))
            try:
                self._ui(lambda msg=err_msg: messagebox.showerror("ScanBass", msg))
            except Exception:
                pass
        finally:
            self._ui(self._prog_complete)

    def _ui(self, fn): self.root.after(0, fn)

    def _prog_start(self):
        self.prog["value"]=2; self._prog_stop=False
        def tick():
            if self._prog_stop: return
            v=float(self.prog["value"])
            if v<90: self.prog["value"]=v+1
            self._prog_timer=self.root.after(120, tick)
        self._prog_timer=self.root.after(120, tick)
    def _prog_complete(self):
        self._prog_stop=True
        try:
            if self._prog_timer: self.root.after_cancel(self._prog_timer)
        except Exception: pass
        self.prog["value"]=100
    def _prog_reset(self):
        self._prog_stop=True
        try:
            if self._prog_timer: self.root.after_cancel(self._prog_timer)
        except Exception: pass
        self.prog["value"]=0

    def _finish_with_midi(self, midi_path:Path, info:str="Done"):
        self.last_result={"midi": str(midi_path)}
        self._status(f"{info}: {midi_path.name}")
        bpm = _detect_bpm(midi_path) or DEFAULT_BPM
        self.current_bpm = bpm
        try:
            self.preview.set_bpm(self.current_bpm)
            self.preview.set_limit_bars(16)
            self.preview.render_from_midi_file(midi_path)
        except Exception:
            pass
        for btn in (self.open_btn, self.copy_file_btn, self.explorer_btn, self.notes_btn):
            try: btn.config(state=tk.NORMAL)
            except Exception: pass
        self._auto_make_notes_only()

    def _auto_make_notes_only(self):
        src = self._midi_path()
        if not src: return
        out = Path(src).with_name("bassline_notes.mid")
        try:
            import mido, pretty_midi
        except Exception:
            self._indicate_ready(True, midi_name=Path(src).name)
            self.drag_file = Path(src)
            return
        try:
            _write_notes_only(src, out, self.current_bpm)
            self.drag_file = out
            self._indicate_ready(True, midi_name=out.name)
            self._status(f"Exported notes-only → {out.name}")
        except Exception as exc:
            self.drag_file = src
            self._indicate_ready(True, midi_name=Path(src).name)
            self._status(f"Notes-only export failed, using original: {exc}")

    def export_notes_only(self):
        src = self._midi_path()
        if not src:
            messagebox.showinfo("ScanBass","No MIDI file yet."); return
        out = Path(src).with_name("bassline_notes.mid")
        try:
            import mido, pretty_midi
        except Exception:
            messagebox.showerror("ScanBass", "Install dependencies first:\npython -m pip install mido pretty_midi")
            return
        try:
            _write_notes_only(src, out, self.current_bpm)
        except Exception as exc:
            messagebox.showerror("ScanBass", f"Export failed: {exc}")
            return
        self.drag_file = out
        self._indicate_ready(True, midi_name=out.name)
        self._status(f"Exported notes-only → {out.name}")
        try:
            self.preview.render_from_midi_file(out)
        except Exception:
            pass

    def _midi_path(self)->Path|None:
        if not self.last_result: return None
        p=Path(self.last_result.get("midi","")).resolve()
        return p if p.exists() else None

    def open_output_folder(self):
        p=self._midi_path(); target=p.parent if p else (PROJECT_ROOT/"outputs")
        try:
            if sys.platform=="win32":
                import os; os.startfile(target)
            elif sys.platform=="darwin":
                import subprocess; subprocess.run(["open", str(target)])
            else:
                import subprocess; subprocess.run(["xdg-open", str(target)])
        except Exception:
            messagebox.showinfo("ScanBass", f"Output folder: {target}")

    def open_explorer_select(self):
        p=self.drag_file or self._midi_path()
        if not p:
            messagebox.showinfo("ScanBass","No MIDI file yet."); return
        try:
            import subprocess
            subprocess.Popen(['explorer', '/select,', str(p)])
        except Exception:
            self.open_output_folder()

    def copy_midi_file(self):
        p=self.drag_file or self._midi_path()
        if not p: messagebox.showinfo("ScanBass","No MIDI file yet."); return
        if sys.platform=="win32":
            try:
                import ctypes
                from ctypes import wintypes
                GMEM_MOVEABLE=0x0002; CF_HDROP=15
                class DROPFILES(ctypes.Structure):
                    _fields_=[("pFiles", wintypes.DWORD), ("pt", wintypes.POINT), ("fNC", wintypes.BOOL), ("fWide", wintypes.BOOL)]
                file_list=(str(p)+"\\0\\0").encode("utf-16le")
                size=ctypes.sizeof(DROPFILES)+len(file_list)
                kernel32=ctypes.windll.kernel32; user32=ctypes.windll.user32
                hglobal=kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
                ptr=kernel32.GlobalLock(hglobal)
                import ctypes as _ct
                df=DROPFILES(); df.pFiles=_ct.sizeof(DROPFILES); df.pt=wintypes.POINT(0,0); df.fNC=True; df.fWide=True
                _ct.memmove(ptr, _ct.byref(df), _ct.sizeof(DROPFILES))
                _ct.memmove(ptr+_ct.sizeof(DROPFILES), file_list, len(file_list))
                kernel32.GlobalUnlock(hglobal)
                user32.OpenClipboard(None); user32.EmptyClipboard(); user32.SetClipboardData(CF_HDROP, hglobal); user32.CloseClipboard()
                messagebox.showinfo("ScanBass", f"Copied as file: {p.name}"); return
            except Exception: pass
        try:
            self.root.clipboard_clear(); self.root.clipboard_append(str(p)); self.root.update()
            messagebox.showinfo("ScanBass", f"File path copied:\n{p}")
        except Exception as exc:
            messagebox.showerror("ScanBass", f"Copy failed: {exc}")

    # ---------- DnD source (OUT) ----------
    def on_drag_init(self, event):
        if (not self._ready) or (not self.drag_file) or (not self.drag_file.exists()):
            if DEBUG_DND: print("DRAG: not ready")
            return ('break',)
        state=getattr(event, "state", 0); use_uri=bool(state & 0x0004)  # CTRL
        if use_uri:
            uri = "file:///" + str(self.drag_file).replace("\\","/").replace(" ", "%20")
            dtype = 'text/uri-list'; data = uri
        else:
            dtype = DND_FILES; data = "{" + str(self.drag_file) + "}"
        if DEBUG_DND: print("DRAG INIT:", dtype, data)
        return (('copy','link'), dtype, data)

    def on_drag_end(self, _event):
        if DEBUG_DND: print("DRAG END")

def main():
    if HAS_DND and TkinterDnD: root = TkinterDnD.Tk()
    else: root = tk.Tk()
    app = ScanBassGUI(root); root.mainloop()

__all__ = ["ScanBassGUI","tk","ttk","TkinterDnD","HAS_DND","main"]
