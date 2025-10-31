# Simple GUI/console launcher for ScanBass Poly (select files -> run -> show results)
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

# Try GUI first, fallback to console prompt if Tk is missing
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    HAS_TK = True
except Exception:
    HAS_TK = False

from modes.poly_mode import run_poly_mode  # Basic Pitch + lowest voice

def choose_files():
    if HAS_TK:
        root = tk.Tk()
        root.withdraw()
        files = filedialog.askopenfilenames(
            title="Vyber audio soubor(y)",
            filetypes=[
                ("Audio", "*.mp3 *.wav *.aiff *.flac *.m4a"),
                ("Všechny soubory", "*.*"),
            ],
            initialdir=str(ROOT)
        )
        return list(files)
    else:
        print("Tkinter není dostupný – přepínám na textový výběr.")
        print("Zadej cesty k souborům (MP3/WAV/AIFF/FLAC/M4A).")
        print("Více cest odděl středníkem ; nebo je vlož na více řádků a pak Enter.")
        try:
            line = input("Cesty: ").strip()
        except EOFError:
            return []
        parts = []
        for chunk in line.split(";"):
            p = chunk.strip().strip('"')
            if p:
                parts.append(p)
        return parts

def main():
    files = choose_files()
    if not files:
        if HAS_TK:
            message = "Zrušeno."
            try:
                messagebox.showinfo("ScanBass Poly", message)
            except Exception:
                print(message)
        else:
            print("Zrušeno.")
        return

    out_root = ROOT / "outputs"
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for f in files:
        fpath = Path(f).resolve()
        out_dir = out_root / f"{fpath.stem}_poly"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            res = run_poly_mode(str(fpath), str(out_dir))
            results.append(f"{fpath.name} → {Path(res['midi']).resolve()}")
        except Exception as e:
            results.append(f"{fpath.name} → CHYBA: {e}")

    message = "Hotovo:\n\n" + "\n".join(results)
    if HAS_TK:
        try:
            messagebox.showinfo("ScanBass Poly", message)
        except Exception:
            print(message)
    else:
        print(message)

if __name__ == "__main__":
    main()
