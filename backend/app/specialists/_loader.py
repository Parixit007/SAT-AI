"""The model wrapper scripts under models/*/ are standalone, dependency-light scripts (see
CLAUDE.md) -- not installable packages. This loads them by file path so the backend can import
`GroundingTool` / `WaterSegmentationTool` without turning models/ into a package or duplicating
their code under backend/."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_module(file_path: Path, module_name: str) -> ModuleType:
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module '{module_name}' from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
