import sys, platform
print("Python:", sys.version)
print("Platform:", platform.platform())
try:
    import tkinter as tk
    r = tk.Tk(); print("Tk OK, patchlevel:", r.tk.call('info','patchlevel')); r.destroy()
except Exception as e:
    print("Tk ERROR:", e)

try:
    import tkinterdnd2 as dnd
    print("tkinterdnd2:", getattr(dnd, "__version__", "OK"))
except Exception as e:
    print("tkinterdnd2 ERROR:", e)