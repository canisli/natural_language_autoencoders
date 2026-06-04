"""Shared prompt/run JSON loading for small inference helpers."""

import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PROMPTS_DIR = _REPO_ROOT / "prompts"

_VALID_ROLES = {"system", "user", "assistant"}


def read_messages(path: str | Path, prompts_dir: str | Path | None = None) -> list[dict[str, str]]:
    """Read messages from a JSON message list or a run spec that references one.

    Supported forms:
      - [{"role": "user", "content": "..."}]
      - {"messages": [...]}
      - {"prompt": "name"}              -> prompts/{name}.json
      - {"prompt_file": "path/to.json"} -> relative to the run file, repo root, or prompts dir
    """
    source = Path(path)
    root = Path(prompts_dir) if prompts_dir is not None else _DEFAULT_PROMPTS_DIR
    return _read_messages(source, root, seen=set())


def read_prompt_messages(name: str, prompts_dir: str | Path | None = None) -> list[dict[str, str]]:
    """Read a named prompt from the prompt directory."""
    root = Path(prompts_dir) if prompts_dir is not None else _DEFAULT_PROMPTS_DIR
    return _read_messages(_resolve_named_prompt(name, root), root, seen=set())


def _read_messages(path: Path, prompts_dir: Path, seen: set[Path]) -> list[dict[str, str]]:
    resolved = path.expanduser().resolve()
    if resolved in seen:
        raise ValueError(f"prompt reference cycle includes {resolved}")
    seen.add(resolved)

    data = json.loads(resolved.read_text())
    messages = _messages_from_data(data, source=resolved, prompts_dir=prompts_dir, seen=seen)
    return _validate_messages(messages, context=str(resolved))


def _messages_from_data(
    data: Any,
    *,
    source: Path,
    prompts_dir: Path,
    seen: set[Path],
) -> list[dict[str, str]]:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        raise ValueError(f"{source} must contain a JSON message list or object run spec")
    if "messages" in data:
        return data["messages"]
    if "prompt" in data:
        return _read_messages(_resolve_named_prompt(data["prompt"], prompts_dir), prompts_dir, seen)
    if "prompt_file" in data:
        return _read_messages(_resolve_prompt_file(data["prompt_file"], source, prompts_dir), prompts_dir, seen)
    raise ValueError(f"{source} must include one of: messages, prompt, prompt_file")


def _resolve_named_prompt(name: Any, prompts_dir: Path) -> Path:
    if not isinstance(name, str) or not name:
        raise ValueError("prompt must be a non-empty string")
    ref = Path(name)
    if ref.is_absolute() or ref.suffix or len(ref.parts) > 1:
        return _resolve_first_existing([ref, prompts_dir / ref], f"prompt {name!r}")
    return _resolve_first_existing(
        [prompts_dir / f"{name}.json", prompts_dir / name / "messages.json"],
        f"prompt {name!r}",
    )


def _resolve_prompt_file(value: Any, source: Path, prompts_dir: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("prompt_file must be a non-empty string")
    ref = Path(value)
    if ref.is_absolute():
        return _resolve_first_existing([ref], f"prompt_file {value!r}")
    return _resolve_first_existing(
        [source.parent / ref, _REPO_ROOT / ref, prompts_dir / ref],
        f"prompt_file {value!r}",
    )


def _resolve_first_existing(candidates: list[Path], label: str) -> Path:
    for candidate in candidates:
        if candidate.expanduser().exists():
            return candidate.expanduser()
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"could not resolve {label}; tried: {tried}")


def _validate_messages(messages: Any, context: str) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise ValueError(f"{context} must resolve to a JSON list of messages")
    out: list[dict[str, str]] = []
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"{context}: message {i} is not an object")
        role = message.get("role")
        if role not in _VALID_ROLES:
            raise ValueError(f"{context}: message {i} has unsupported role {role!r}")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError(f"{context}: message {i} has non-string content")
        out.append({"role": role, "content": content})
    return out
