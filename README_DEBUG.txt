ScanBass — EXE DEBUG KIT (s konzolí + logem)
===========================================

CO JE UVNITŘ
- build\pyinstaller\entry_wrapper.py
- build\pyinstaller\scanbass_debug.spec
- build\pyinstaller\build_scanbass_debug.bat
- README_DEBUG.txt

JAK POUŽÍT
1) Rozbal kit do KOŘENE projektu (tam, kde je src\ a .venv311\).
2) Spusť: build\pyinstaller\build_scanbass_debug.bat
3) Spusť: dist\ScanBass\ScanBass_dbg.exe
   - chyba se ukáže v konzoli a zároveň zapíše do dist\ScanBass\last_error.txt
4) Pošli první 1–3 řádky chyby; upravím finální no-console build.