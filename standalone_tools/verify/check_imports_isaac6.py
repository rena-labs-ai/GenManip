import os, sys, glob, importlib, warnings
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
sys.path.insert(0, os.getcwd())
# genmanip is a namespace package (no __init__.py), so pkgutil.walk_packages yields nothing; glob instead
modules = sorted(f[:-3].replace(os.sep, ".").removesuffix(".__init__") for f in glob.glob("genmanip/**/*.py", recursive=True))
failed = []
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    for m in modules:
        try:
            importlib.import_module(m)
        except Exception as e:  # report all, keep going
            failed.append((m, f"{type(e).__name__}: {e}"))
    deprecations = sorted({str(x.message)[:200] for x in w if "deprecat" in str(x.message).lower()})
print(f"MODULES: {len(modules)}")
print("DEPRECATIONS:", *deprecations, sep="\n  ")
print("FAILED:", *failed, sep="\n  ")
app.close(exit_code=1 if failed else 0)  # fastShutdown: close() os._exit()s, so print first and exit here
