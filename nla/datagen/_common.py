"""Shared helpers for stage CLIs."""

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from nla.datagen.storage import Storage


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _looks_like_missing_local_path(value: str, repo_root: Path) -> bool:
    if value.startswith(("/", "./", "../", "~")):
        return True

    path = Path(value)
    if len(path.parts) > 2:
        return True

    if path.parts:
        first = Path(path.parts[0])
        if first.exists() or (repo_root / first).exists():
            return True

    return False


def resolve_local_path_or_repo_id(
    value: str,
    *,
    repo_root: Path | None = None,
    label: str = "path",
) -> str:
    """Resolve an existing local path, otherwise leave valid HF repo ids intact."""
    root = _REPO_ROOT if repo_root is None else Path(repo_root)
    path = Path(value).expanduser()
    raw_candidates = [path] if path.is_absolute() else [path, root / path]
    candidates = []
    seen: set[Path] = set()
    for candidate in raw_candidates:
        normalized = candidate.resolve(strict=False)
        if normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(candidate)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    if _looks_like_missing_local_path(value, root):
        checked = ", ".join(str(candidate.resolve(strict=False)) for candidate in candidates)
        raise FileNotFoundError(
            f"{label} looks like a local path but does not exist: {value!r} "
            f"(checked: {checked})"
        )

    return value


def load_class(path: str) -> type:
    """Load a class by import path: 'module.submodule.ClassName'."""
    module_path, _, attr = path.rpartition(".")
    module = importlib.import_module(module_path)
    cls = getattr(module, attr)
    assert isinstance(cls, type), f"{path!r} resolved to {cls!r}, expected a class"
    return cls


def parse_kwargs(kwargs_json: str | None) -> dict[str, Any]:
    return json.loads(kwargs_json) if kwargs_json else {}


def add_storage_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--storage-cls", default="nla.datagen.storage.LocalStorage",
                   help="Storage backend class (subclass nla.datagen.storage.Storage for S3/GCS)")
    p.add_argument("--storage-kwargs", default=None,
                   help="JSON dict of kwargs for the storage constructor")


def make_storage(args: argparse.Namespace) -> Storage:
    return load_class(args.storage_cls)(**parse_kwargs(args.storage_kwargs))


def load_tokenizer(model_name: str, **kwargs: Any) -> Any:
    """Central tokenizer loader. Stage0's extractor and stage3 both go through here."""
    return AutoTokenizer.from_pretrained(model_name, **kwargs)
