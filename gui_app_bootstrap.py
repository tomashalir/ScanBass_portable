# Safe bootstrap: keeps console open, imports gui_app robustly.
import sys, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    import gui_app as appmod
    tk = appmod.tk
    ttk = appmod.ttk
    HAS_DND = getattr(appmod, "HAS_DND", False)
    TkinterDnD = getattr(appmod, "TkinterDnD", None)
    # Choose root according to availability
    root = TkinterDnD.Tk() if (HAS_DND and TkinterDnD) else tk.Tk()
    app = appmod.ScanBassGUI(root)
    root.mainloop()
except Exception as e:
    traceback.print_exc()
    try:
        input("\nPress Enter to close…")
    except Exception:
        pass