# gui_app_quality_bootstrap.py
import sys
print("Starting Quality GUI with console kept open...")
try:
    import gui_app_quality as appmod
except Exception as e:
    print("Import error:", e)
    input("Press Enter to close…")
    raise

if __name__ == "__main__":
    try:
        if getattr(appmod, "TkinterDnD", None) and getattr(appmod, "HAS_DND", False):
            root = appmod.TkinterDnD.Tk()
        else:
            root = appmod.tk.Tk()
        app = appmod.ScanBassGUIQuality(root)
        root.mainloop()
    except Exception as e:
        print(e)
        input("Press Enter to close…")