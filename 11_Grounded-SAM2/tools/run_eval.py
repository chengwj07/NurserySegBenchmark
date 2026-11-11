"""Run eval_plant_metrics.py using the patched GroundingDINO repo.

Maps the `grounding_dino.groundingdino.*` namespace (used by the vendored
eval script) onto the patched `groundingdino.*` package that is verified to
work with transformers 5.x / numpy 2.x / timm 0.4 (the training repo).
"""
import importlib
import importlib.abc
import importlib.util
import sys

PATCHED_ROOT = "../GroundingDINO"
EVAL_SCRIPT = "../Grounded-SAM-2/eval_plant_metrics.py"

# Patched repo must be importable as top-level `groundingdino`.
if PATCHED_ROOT not in sys.path:
    sys.path.insert(0, PATCHED_ROOT)


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, real_name):
        self.real_name = real_name

    def create_module(self, spec):
        return importlib.import_module(self.real_name)

    def exec_module(self, module):
        pass


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "grounding_dino":
            return importlib.util.spec_from_loader(fullname, _AliasLoader("groundingdino"))
        if fullname == "grounding_dino.groundingdino":
            return importlib.util.spec_from_loader(fullname, _AliasLoader("groundingdino"))
        if fullname.startswith("grounding_dino.groundingdino."):
            real = "groundingdino" + fullname[len("grounding_dino.groundingdino"):]
            try:
                importlib.util.find_spec(real)
            except (ImportError, AttributeError, ValueError):
                return None
            return importlib.util.spec_from_loader(fullname, _AliasLoader(real))
        return None


sys.meta_path.insert(0, _AliasFinder())

if __name__ == "__main__":
    sys.argv = [EVAL_SCRIPT] + sys.argv[1:]
    with open(EVAL_SCRIPT, encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, EVAL_SCRIPT, "exec"), {"__name__": "__main__", "__file__": EVAL_SCRIPT})
