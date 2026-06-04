"""Render the six sky-sunset Gemma response runs as one HTML comparison page."""

import argparse
import html
import json
from pathlib import Path
from typing import Any


RUNS = [
    ("base", "no_pls", "Base", "No PLS"),
    ("base", "pls", "Base", "PLS"),
    ("clean", "no_pls", "Clean NH100", "No PLS"),
    ("clean", "pls", "Clean NH100", "PLS"),
    ("backdoored", "no_pls", "Backdoored PR010", "No PLS"),
    ("backdoored", "pls", "Backdoored PR010", "PLS"),
]


def _read_messages(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON message list")
    out: list[dict[str, str]] = []
    for i, message in enumerate(data):
        if not isinstance(message, dict):
            raise ValueError(f"{path}: message {i} is not an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"{path}: message {i} must have string role/content")
        out.append({"role": role, "content": content})
    return out


def _first_content(messages: list[dict[str, str]], role: str) -> str:
    for message in messages:
        if message["role"] == role:
            return message["content"]
    return ""


def _last_content(messages: list[dict[str, str]], role: str) -> str:
    for message in reversed(messages):
        if message["role"] == role:
            return message["content"]
    return ""


def _load_cards(run_root: Path) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for model_key, prompt_key, model_label, prompt_label in RUNS:
        path = run_root / model_key / prompt_key / "completed_messages.json"
        exists = path.exists()
        messages = _read_messages(path) if exists else []
        response = _last_content(messages, "assistant")
        cards.append({
            "model_key": model_key,
            "prompt_key": prompt_key,
            "model_label": model_label,
            "prompt_label": prompt_label,
            "source": str(path),
            "exists": exists,
            "prompt": _first_content(messages, "user"),
            "response": response,
            "is_empty": exists and response == "",
        })
    return cards


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _render_card(card: dict[str, Any]) -> str:
    status = "Missing" if not card["exists"] else "Empty" if card["is_empty"] else "Complete"
    status_class = "missing" if not card["exists"] else "empty" if card["is_empty"] else "complete"
    response = card["response"] if card["response"] else "(empty response)"
    prompt = card["prompt"] if card["prompt"] else "(prompt unavailable)"
    return f"""
<article class="card {status_class}" data-model="{_esc(card['model_key'])}" data-prompt="{_esc(card['prompt_key'])}">
  <header class="card-head">
    <div>
      <h2>{_esc(card['model_label'])}</h2>
      <div class="variant">{_esc(card['prompt_label'])}</div>
    </div>
    <span class="status">{_esc(status)}</span>
  </header>
  <section class="block">
    <h3>Prompt</h3>
    <pre>{_esc(prompt)}</pre>
  </section>
  <section class="block response">
    <h3>Response</h3>
    <pre>{_esc(response)}</pre>
  </section>
  <footer>{_esc(card['source'])}</footer>
</article>"""


def _render_html(cards: list[dict[str, Any]], title: str) -> str:
    complete = sum(1 for card in cards if card["exists"] and not card["is_empty"])
    empty = sum(1 for card in cards if card["is_empty"])
    missing = sum(1 for card in cards if not card["exists"])
    cards_html = "\n".join(_render_card(card) for card in cards)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f5f7fa;
  --panel: #ffffff;
  --ink: #1d2430;
  --muted: #667085;
  --line: #d6dee8;
  --good: #0b7a46;
  --warn: #a15c00;
  --bad: #a53a3a;
  --accent: #1264a3;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
}}
.topbar {{
  position: sticky;
  top: 0;
  z-index: 2;
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: center;
  padding: 14px 18px;
  border-bottom: 1px solid var(--line);
  background: #eef3f8;
}}
h1 {{
  margin: 0;
  font-size: 18px;
  line-height: 1.2;
  letter-spacing: 0;
}}
.summary {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: flex-end;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
}}
.pill {{
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 4px 9px;
  background: var(--panel);
}}
.wrap {{
  padding: 16px 18px 22px;
}}
.grid {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}}
.card {{
  min-width: 0;
  display: flex;
  flex-direction: column;
  min-height: 360px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  box-shadow: 0 1px 3px rgb(16 24 40 / 10%);
  overflow: hidden;
}}
.card-head {{
  display: flex;
  justify-content: space-between;
  align-items: start;
  gap: 10px;
  padding: 11px 12px;
  border-bottom: 1px solid var(--line);
  background: #fbfcfe;
}}
h2 {{
  margin: 0;
  font-size: 15px;
  line-height: 1.2;
  letter-spacing: 0;
}}
.variant {{
  margin-top: 2px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
}}
.status {{
  flex: 0 0 auto;
  border-radius: 999px;
  padding: 3px 8px;
  background: #e9f7ef;
  color: var(--good);
  font-size: 11px;
  font-weight: 800;
}}
.empty .status {{
  background: #fff4df;
  color: var(--warn);
}}
.missing .status {{
  background: #ffecec;
  color: var(--bad);
}}
.block {{
  padding: 10px 12px 0;
}}
.block.response {{
  flex: 1 1 auto;
  display: flex;
  min-height: 0;
  flex-direction: column;
  padding-bottom: 10px;
}}
h3 {{
  margin: 0 0 5px;
  color: var(--muted);
  font-size: 11px;
  line-height: 1.2;
  text-transform: uppercase;
  letter-spacing: 0;
}}
pre {{
  margin: 0;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  font: 13px/1.38 ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
}}
.block.response pre {{
  flex: 1 1 auto;
  overflow: auto;
  padding: 9px;
  border: 1px solid #e4e9f0;
  border-radius: 6px;
  background: #fbfcfd;
}}
.empty .block.response pre {{
  color: var(--warn);
  font-style: italic;
}}
footer {{
  padding: 8px 12px;
  border-top: 1px solid var(--line);
  color: #7a8798;
  font-size: 11px;
  overflow-wrap: anywhere;
}}
@media (max-width: 1180px) {{
  .grid {{
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }}
}}
@media (max-width: 760px) {{
  .topbar {{
    align-items: flex-start;
    flex-direction: column;
  }}
  .summary {{
    justify-content: flex-start;
  }}
  .grid {{
    grid-template-columns: 1fr;
  }}
}}
</style>
</head>
<body>
  <header class="topbar">
    <h1>{_esc(title)}</h1>
    <div class="summary">
      <span class="pill">6 runs</span>
      <span class="pill">{complete} complete</span>
      <span class="pill">{empty} empty</span>
      <span class="pill">{missing} missing</span>
    </div>
  </header>
  <main class="wrap">
    <section class="grid">
      {cards_html}
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="runs/sky_sunset_gemma12b")
    parser.add_argument("--output", default="runs/sky_sunset_gemma12b/responses.html")
    parser.add_argument("--title", default="Sky Sunset Gemma-3-12B Responses")
    args = parser.parse_args()

    cards = _load_cards(Path(args.run_root))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_render_html(cards, args.title))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
