# test_start.py
import importlib

# 1) ověř že se načte app
module_path = "deploy_backend.web_service"
mod = importlib.import_module(module_path)

app = getattr(mod, "app", None)
if app is None:
    raise SystemExit("❌ app not found in deploy_backend.web_service")

print("✅ import ok")
