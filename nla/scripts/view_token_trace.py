"""Render an NLA per-token trace JSONL file as a self-contained HTML viewer.

Example:
    python -m nla.scripts.view_token_trace \
        runs/forty_two/nla_token_trace.jsonl \
        --output runs/forty_two/nla_token_trace.html
"""

import argparse
import html
import json
from json import JSONDecodeError
from pathlib import Path
from statistics import mean, median
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(rec, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(rec)
    if not rows:
        raise ValueError(f"{path} has no trace rows")
    return rows


def _float_field(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None:
        return default
    return float(value)


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    norms = sorted(_float_field(row, "activation_norm") for row in rows)
    return {
        "count": len(rows),
        "norm_min": norms[0],
        "norm_p05": _percentile(norms, 5),
        "norm_median": median(norms),
        "norm_mean": mean(norms),
        "norm_p95": _percentile(norms, 95),
        "norm_max": norms[-1],
    }


def _json_for_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def _default_run_label(path: Path) -> str:
    stem = path.stem.removesuffix("_token_trace").removesuffix("_nla")
    if stem == "nla":
        stem = path.parent.name
    return stem.replace("_", " ").title()


def _parse_run_spec(spec: str) -> tuple[str | None, Path]:
    label, sep, path = spec.partition("=")
    if not sep:
        return None, Path(label)
    label = label.strip()
    if not label:
        raise ValueError(f"empty run label in {spec!r}")
    return label, Path(path.strip())


def _run_payload(label: str | None, trace_path: Path) -> dict[str, Any]:
    rows = _read_jsonl(trace_path)
    return {
        "label": label or _default_run_label(trace_path),
        "source": str(trace_path),
        "rows": rows,
        "summary": _summary(rows),
    }


def _render_html(runs: list[dict[str, Any]], title: str, default_run: str | None = None) -> str:
    if not runs:
        raise ValueError("expected at least one trace run")
    runs_json = _json_for_script(runs)
    default_run_json = _json_for_script(default_run)
    escaped_title = html.escape(title)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escaped_title}</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f4f7fb;
  --top: #eaf0f6;
  --panel: #ffffff;
  --ink: #1c1f24;
  --muted: #667085;
  --line: #d8e1ea;
  --accent: #0877b4;
  --accent-2: #b5492f;
  --soft: #e2f4fb;
  --system: #eef5ff;
  --user: #ffffff;
  --assistant: #f6fff3;
  --token: #fdfdfb;
  --shadow: 0 1px 4px rgb(16 24 40 / 12%);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}

* {{ box-sizing: border-box; }}

body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
}}

button,
input,
select {{
  font: inherit;
}}

.app {{
  min-height: 100vh;
  display: grid;
  grid-template-rows: auto 1fr;
}}

.topbar {{
  z-index: 10;
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 12px;
  align-items: center;
  padding: 12px 18px;
  border-bottom: 1px solid var(--line);
  background: var(--top);
}}

.title-block {{
  min-width: 0;
}}

h1 {{
  margin: 0;
  font-size: 18px;
  line-height: 1.2;
  font-weight: 700;
  letter-spacing: 0;
}}

.source {{
  margin-top: 3px;
  color: #63748e;
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}

.controls {{
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
}}

.field {{
  display: grid;
  gap: 4px;
}}

.field label {{
  color: var(--muted);
  font-size: 11px;
  font-weight: 650;
  text-transform: uppercase;
}}

.field input,
.field select {{
  width: 190px;
  height: 32px;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 0 11px;
  background: var(--panel);
  color: var(--ink);
}}

.layout {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(320px, 430px);
  gap: 14px;
  min-height: 0;
  padding: 14px 18px;
}}

.main {{
  min-width: 0;
  overflow: auto;
}}

.stats {{
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 0;
  margin: 0 auto 10px;
  max-width: 1040px;
  overflow: hidden;
  border-radius: 8px;
  background: #dff3fb;
}}

.stat {{
  min-width: 0;
  padding: 7px 10px;
  border: 0;
  border-right: 1px solid #b8d7e6;
  background: transparent;
  box-shadow: none;
  text-align: center;
}}

.stat:last-child {{
  border-right: 0;
}}

.stat-label {{
  color: #51647c;
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
}}

.stat-value {{
  margin-top: 1px;
  font-size: 13px;
  font-variant-numeric: tabular-nums;
  font-weight: 700;
}}

.messages {{
  display: grid;
  gap: 10px;
  max-width: 1040px;
  margin: 0 auto;
}}

.message-card {{
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  box-shadow: var(--shadow);
  overflow: hidden;
}}

.message-card.system {{
  background: var(--system);
}}

.message-card.user {{
  background: var(--user);
}}

.message-card.assistant {{
  background: var(--assistant);
}}

.message-head {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  padding: 7px 10px;
  border-bottom: 1px solid var(--line);
  color: #50627a;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0;
  text-transform: uppercase;
}}

.role-badge {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
}}

.role-dot {{
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: var(--accent);
}}

.system .role-dot {{
  background: #446ea7;
}}

.user .role-dot {{
  background: #0c9f6a;
}}

.assistant .role-dot {{
  background: #8b5cf6;
}}

.message-count {{
  color: var(--muted);
  font-variant-numeric: tabular-nums;
  text-transform: none;
}}

.message-body {{
  padding: 6px 12px 8px;
}}

.template-line {{
  margin-bottom: 1px;
  color: #8ea0b7;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  font-size: 10px;
  line-height: 1.1;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}}

.template-line.after {{
  margin-top: 1px;
  margin-bottom: 0;
}}

.tokens {{
  display: block;
  font-family: Georgia, "Times New Roman", serif;
  font-size: 15px;
  line-height: 1.3;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  color: #64728a;
}}

.token {{
  display: inline;
  max-width: 100%;
  border: 0;
  border-radius: 3px;
  margin: 0;
  padding: 0;
  background: transparent;
  color: #16191d;
  cursor: pointer;
  white-space: inherit;
  overflow-wrap: inherit;
  text-decoration: underline;
  text-decoration-thickness: 1px;
  text-underline-offset: 3px;
}}

.token.no-nla {{
  cursor: text;
  text-decoration: none;
}}

.token:hover {{
  box-shadow: inset 0 -2px 0 var(--accent);
}}

.token.no-nla:hover {{
  box-shadow: none;
}}

.token.selected {{
  background: color-mix(in srgb, var(--accent) 18%, var(--panel));
  box-shadow: inset 0 -2px 0 var(--accent);
}}

.token.search-hit {{
  background: #ffe58a;
}}

.token.search-hit.selected {{
  background: #ffd166;
}}

.text-highlight {{
  background: #ffe58a;
  border-radius: 2px;
  padding: 0 1px;
}}

.token.special {{
  color: var(--accent-2);
}}

.template-token {{
  color: #8ea0b7;
}}

.empty {{
  display: none;
  padding: 12px;
  color: var(--muted);
  text-align: center;
}}

.empty.visible {{
  display: block;
}}

.detail {{
  min-width: 0;
  border: 2px solid #0094d8;
  border-radius: 8px;
  background: var(--panel);
  overflow: auto;
  box-shadow: var(--shadow);
}}

.detail-inner {{
  position: sticky;
  top: 10px;
  padding: 0;
  min-height: calc(100vh - 30px);
  display: flex;
  flex-direction: column;
}}

.detail-banner {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 8px 10px;
  border-bottom: 1px solid var(--line);
  background: #eef2f6;
  color: #66758e;
  font-size: 11px;
  font-weight: 800;
  text-transform: uppercase;
}}

.detail-head {{
  display: flex;
  justify-content: space-between;
  gap: 8px;
  align-items: start;
  padding: 10px 10px 8px;
  margin: 0;
}}

.token-title {{
  min-width: 0;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  font-size: 15px;
  font-weight: 750;
  overflow-wrap: anywhere;
  white-space: pre-wrap;
}}

.pill {{
  flex: 0 0 auto;
  border-radius: 999px;
  border: 1px solid var(--line);
  padding: 2px 7px;
  color: var(--muted);
  background: var(--panel);
  font-size: 11px;
  font-variant-numeric: tabular-nums;
}}

.meta {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 5px;
  padding: 0 10px;
  margin-bottom: 8px;
}}

.meta-item {{
  padding: 6px 7px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--panel);
}}

.meta-k {{
  color: var(--muted);
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
}}

.meta-v {{
  margin-top: 1px;
  font-size: 12px;
  font-variant-numeric: tabular-nums;
  overflow-wrap: anywhere;
}}

.section {{
  margin: 8px 10px 0;
}}

.section:last-child {{
  flex: 1 1 auto;
  display: flex;
  min-height: 0;
  flex-direction: column;
}}

.section h2 {{
  margin: 0 0 4px;
  font-size: 11px;
  line-height: 1.2;
  text-transform: uppercase;
  color: var(--muted);
  letter-spacing: 0;
}}

.text-box {{
  flex: 1 1 auto;
  overflow: visible;
  padding: 8px;
  border: 0;
  border-radius: 0;
  background: var(--panel);
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  font-size: 13px;
  line-height: 1.35;
  font-weight: 600;
}}

@media (max-width: 920px) {{
  .topbar {{
    grid-template-columns: 1fr;
  }}

  .controls {{
    justify-content: stretch;
  }}

  .field {{
    flex: 1 1 160px;
  }}

  .field input,
  .field select {{
    width: 100%;
  }}

  .layout {{
    grid-template-columns: 1fr;
    padding: 10px;
  }}

  .detail {{
    border-width: 2px;
  }}

  .detail-inner {{
    position: static;
    min-height: 0;
  }}
}}

@media (max-width: 620px) {{
  .main,
  .detail-inner {{
    padding: 0;
  }}

  .stats {{
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }}

  .meta {{
    grid-template-columns: 1fr;
  }}
}}
</style>
</head>
<body>
<div class="app">
  <header class="topbar">
    <div class="title-block">
      <h1>{escaped_title}</h1>
      <div class="source" id="source"></div>
    </div>
    <div class="controls">
      <div class="field">
        <label for="run">Run</label>
        <select id="run"></select>
      </div>
      <div class="field">
        <label for="search">Search</label>
        <input id="search" type="search" placeholder="explanation text">
      </div>
      <div class="field">
        <label for="specials">Tokens</label>
        <select id="specials">
          <option value="normal">Text tokens</option>
          <option value="all" selected>Text + template</option>
          <option value="special">Template only</option>
        </select>
      </div>
    </div>
  </header>
  <div class="layout">
    <main class="main">
      <section class="stats" id="stats"></section>
      <section class="messages" id="messages"></section>
      <div class="empty" id="empty">No visible tokens</div>
    </main>
    <aside class="detail">
      <div class="detail-inner">
        <div class="detail-banner">
          <span>Output: NLA explanation</span>
          <span id="detailRole"></span>
        </div>
        <div class="detail-head">
          <div class="token-title" id="detailToken"></div>
          <div class="pill" id="detailIndex"></div>
        </div>
        <div class="meta" id="detailMeta"></div>
        <div class="section">
          <h2>NLA output</h2>
          <div class="text-box" id="detailOutput"></div>
        </div>
      </div>
    </aside>
  </div>
</div>
<script>
const runs = {runs_json};
const defaultRun = {default_run_json};
const els = {{
  run: document.getElementById("run"),
  source: document.getElementById("source"),
  stats: document.getElementById("stats"),
  messages: document.getElementById("messages"),
  empty: document.getElementById("empty"),
  search: document.getElementById("search"),
  specials: document.getElementById("specials"),
  detailRole: document.getElementById("detailRole"),
  detailToken: document.getElementById("detailToken"),
  detailIndex: document.getElementById("detailIndex"),
  detailMeta: document.getElementById("detailMeta"),
  detailOutput: document.getElementById("detailOutput"),
}};

let rows = [];
let summary = {{}};
let selectedRow = null;
let sections = [];

function selectDefaultRow(sourceRows) {{
  const isDefaultContentRow = (row) => {{
    const text = String(row.token_text ?? "");
    return (
      hasNlaOutput(row)
      && !row.is_special
      && !isRoleToken(text)
      && !isStartToken(text)
      && !isEndToken(text)
      && !isLeadingTemplateToken(text)
      && text.trim() !== ""
    );
  }};
  return sourceRows.find(isDefaultContentRow) ?? sourceRows.find((row) => (
    hasNlaOutput(row)
    && !row.is_special
    && !isRoleToken(String(row.token_text ?? ""))
    && String(row.token_text ?? "").trim() !== ""
  )) ?? sourceRows.find((row) => hasNlaOutput(row)) ?? sourceRows.find((row) => (
    !row.is_special
    && !isRoleToken(String(row.token_text ?? ""))
    && String(row.token_text ?? "").trim() !== ""
  )) ?? sourceRows[0] ?? null;
}}

function loadRun(runIndex) {{
  const run = runs[Number(runIndex)] ?? runs[0];
  rows = run.rows ?? [];
  summary = run.summary ?? {{}};
  sections = buildSections(rows);
  selectedRow = selectDefaultRow(rows);
  els.source.textContent = run.source ?? "";
  renderAll();
}}

function renderAll() {{
  renderStats();
  renderDetail();
  renderMessages();
}}

for (const [index, run] of runs.entries()) {{
  const option = document.createElement("option");
  option.value = String(index);
  option.textContent = run.label ?? run.source ?? `Run ${{index + 1}}`;
  els.run.appendChild(option);
}}

function defaultRunIndex() {{
  if (defaultRun == null || defaultRun === "") return 0;
  const byLabel = runs.findIndex((run) => run.label === defaultRun);
  if (byLabel >= 0) return byLabel;
  const bySource = runs.findIndex((run) => run.source === defaultRun);
  if (bySource >= 0) return bySource;
  const asNumber = Number(defaultRun);
  if (Number.isInteger(asNumber) && asNumber >= 0 && asNumber < runs.length) return asNumber;
  return 0;
}}

function fmtNumber(value, digits = 2) {{
  if (!Number.isFinite(value)) return "";
  return value.toLocaleString(undefined, {{ maximumFractionDigits: digits }});
}}

function tokenLabel(row) {{
  return String(row.token_text ?? "");
}}

function tokenDetailLabel(row) {{
  const text = tokenLabel(row);
  if (text === "\\n") return "\\\\n";
  if (text === "\\t") return "\\\\t";
  if (text === " ") return "space";
  return text;
}}

function searchQuery() {{
  return els.search.value.trim().toLowerCase();
}}

function explanationMatches(row, query = searchQuery()) {{
  if (!query) return false;
  return String(row.nla_output ?? "").toLowerCase().includes(query);
}}

function hasNlaOutput(row) {{
  return String(row.nla_output ?? "").trim() !== "";
}}

function orderedRows(sourceRows) {{
  return [...sourceRows].sort((a, b) => Number(a.token_index ?? a.row_index ?? 0) - Number(b.token_index ?? b.row_index ?? 0));
}}

function isStartToken(text) {{
  return text === "<|im_start|>" || text === "<start_of_turn>";
}}

function isEndToken(text) {{
  return text === "<|im_end|>" || text === "<end_of_turn>";
}}

function isLeadingTemplateToken(text) {{
  return text === "<bos>";
}}

function isDisplayHiddenRow(row) {{
  return isLeadingTemplateToken(tokenLabel(row));
}}

function isRoleToken(text) {{
  return text === "system" || text === "user" || text === "assistant" || text === "model";
}}

function previousSignificantText(rows, currentIndex) {{
  for (let i = currentIndex - 1; i >= 0; i -= 1) {{
    const text = tokenLabel(rows[i]);
    if (text.trim() !== "") return text;
  }}
  return "";
}}

function roleLabel(role) {{
  if (role === "system") return "System";
  if (role === "user") return "User prompt";
  if (role === "assistant" || role === "model") return "Output";
  return "Trace";
}}

function roleClass(role) {{
  return role === "assistant" || role === "model" ? "assistant" : role === "user" ? "user" : role === "system" ? "system" : "trace";
}}

function buildSections(sourceRows) {{
  const out = [];
  let current = null;
  for (const row of orderedRows(sourceRows)) {{
    const text = tokenLabel(row);
    if (!current && text.trim() === "" && out.length) {{
      out[out.length - 1].rows.push(row);
      continue;
    }}
    if (isStartToken(text)) {{
      const pendingTemplateRows = current?.rows?.every((candidate) => (
        tokenLabel(candidate).trim() === "" || isLeadingTemplateToken(tokenLabel(candidate))
      )) ? current.rows : [];
      if (current && current.rows.length) {{
        if (current.rows.every((candidate) => tokenLabel(candidate).trim() === "") && out.length) {{
          out[out.length - 1].rows.push(...current.rows);
        }} else if (!pendingTemplateRows.length) {{
          out.push(current);
        }}
      }}
      current = {{ role: "trace", rows: [...pendingTemplateRows, row] }};
      continue;
    }}
    if (!current) {{
      current = {{ role: "trace", rows: [] }};
    }}
    current.rows.push(row);
    if (current.role === "trace" && isRoleToken(text) && isStartToken(previousSignificantText(current.rows, current.rows.length - 1))) {{
      current.role = text;
    }}
    if (isEndToken(text)) {{
      out.push(current);
      current = null;
    }}
  }}
  if (current && current.rows.length) out.push(current);
  return out;
}}

function splitSectionRows(section) {{
  const prefixTemplateRows = [];
  const suffixTemplateRows = [];
  const contentRows = [];
  let seenEnd = false;
  for (let i = 0; i < section.rows.length; i += 1) {{
    const row = section.rows[i];
    const text = tokenLabel(row);
    const isHeaderRole = i === 1 && isRoleToken(text);
    const previousText = previousSignificantText(section.rows, i);
    const isTurnRole = isRoleToken(text) && isStartToken(previousText);
    const isHeaderBreak = text.trim() === "" && isRoleToken(previousText);
    if (isLeadingTemplateToken(text) || isStartToken(text) || isHeaderRole || isTurnRole || isHeaderBreak) {{
      prefixTemplateRows.push(row);
    }} else if (isEndToken(text) || seenEnd) {{
      suffixTemplateRows.push(row);
    }} else {{
      contentRows.push(row);
    }}
    if (isEndToken(text)) seenEnd = true;
  }}
  return {{ prefixTemplateRows, contentRows, suffixTemplateRows }};
}}

function visibleRowsForSection(section) {{
  const mode = els.specials.value;
  const parts = splitSectionRows(section);
  const templateRows = [...parts.prefixTemplateRows, ...parts.suffixTemplateRows];
  if (mode === "special") return templateRows.filter((row) => !isDisplayHiddenRow(row));
  if (mode === "all") return [...parts.prefixTemplateRows, ...parts.contentRows, ...parts.suffixTemplateRows].filter((row) => !isDisplayHiddenRow(row));
  return parts.contentRows;
}}

function renderStats() {{
  const query = searchQuery();
  const visibleRows = sections.flatMap(visibleRowsForSection);
  const matchCount = query ? visibleRows.filter((row) => explanationMatches(row, query)).length : 0;
  const decodedCount = visibleRows.filter((row) => hasNlaOutput(row)).length;
  const stats = [
    ["See explanations", "Click underlined tokens"],
    ["Sections", sections.length.toLocaleString()],
    ["Visible tokens", visibleRows.length.toLocaleString()],
    ["Decoded", decodedCount.toLocaleString()],
    ["Matches", matchCount.toLocaleString()],
  ];
  els.stats.innerHTML = stats.map(([label, value]) => `
    <div class="stat">
      <div class="stat-label">${{label}}</div>
      <div class="stat-value">${{value}}</div>
    </div>
  `).join("");
}}

function createTokenButton(row, query, extraClass = "", label = tokenLabel(row)) {{
  const button = document.createElement("button");
  button.type = "button";
  button.className = [
    "token",
    row.is_special ? "special" : "",
    extraClass,
    explanationMatches(row, query) ? "search-hit" : "",
    row === selectedRow ? "selected" : "",
  ].filter(Boolean).join(" ");
  button.textContent = label;
  button.setAttribute("aria-label", `token ${{tokenDetailLabel(row)}}`);
  button.title = `#${{row.token_index ?? row.row_index}} norm=${{fmtNumber(Number(row.activation_norm ?? 0))}}`;
  button.addEventListener("click", () => {{
    selectedRow = row;
    renderDetail();
    renderMessages();
  }});
  return button;
}}

function displayWhitespace(value) {{
  return String(value ?? "").replace(/\\n{{2,}}/g, "\\n");
}}

function appendToken(parent, row, query, extraClass = "") {{
  const text = tokenLabel(row);
  const leading = text.match(/^\\s+/)?.[0] ?? "";
  const trailing = text.match(/\\s+$/)?.[0] ?? "";
  const coreStart = leading.length;
  const coreEnd = trailing.length ? text.length - trailing.length : text.length;
  const core = text.slice(coreStart, coreEnd);

  if (leading) parent.appendChild(document.createTextNode(displayWhitespace(leading)));
  if (core) {{
    if (hasNlaOutput(row)) {{
      parent.appendChild(createTokenButton(row, query, extraClass, core));
    }} else {{
      const span = document.createElement("span");
      span.className = [
        "token",
        "no-nla",
        row.is_special ? "special" : "",
        extraClass,
      ].filter(Boolean).join(" ");
      span.textContent = core;
      span.title = `#${{row.token_index ?? row.row_index}} not decoded`;
      parent.appendChild(span);
    }}
  }}
  if (trailing) parent.appendChild(document.createTextNode(displayWhitespace(trailing)));
}}

function appendTemplateLine(parent, rowsForLine, query, extraClass = "") {{
  let rows = rowsForLine.filter((row) => !isDisplayHiddenRow(row));
  while (rows.length && tokenLabel(rows[0]).trim() === "") rows = rows.slice(1);
  while (rows.length && tokenLabel(rows[rows.length - 1]).trim() === "") rows = rows.slice(0, -1);
  if (!rows.length) return;
  const template = document.createElement("div");
  template.className = ["template-line", extraClass].filter(Boolean).join(" ");
  for (const row of rows) {{
    appendToken(template, row, query, "template-token");
  }}
  parent.appendChild(template);
}}

function renderMessages() {{
  els.messages.innerHTML = "";
  let visible = 0;
  const query = searchQuery();
  for (const section of sections) {{
    const mode = els.specials.value;
    const parts = splitSectionRows(section);
    const prefixTemplateRows = mode === "all" || mode === "special" ? parts.prefixTemplateRows : [];
    const suffixTemplateRows = mode === "all" || mode === "special" ? parts.suffixTemplateRows : [];
    const contentRows = mode === "special" ? [] : parts.contentRows;
    const sectionVisible = prefixTemplateRows.length + contentRows.length + suffixTemplateRows.length;
    visible += sectionVisible;

    const card = document.createElement("article");
    card.className = `message-card ${{roleClass(section.role)}}`;

    const head = document.createElement("div");
    head.className = "message-head";
    head.innerHTML = `
      <div class="role-badge"><span class="role-dot"></span><span>${{roleLabel(section.role)}}</span></div>
      <div class="message-count">${{sectionVisible.toLocaleString()}} tokens</div>
    `;
    card.appendChild(head);

    const body = document.createElement("div");
    body.className = "message-body";

    appendTemplateLine(body, prefixTemplateRows, query);

    if (contentRows.length) {{
      const content = document.createElement("div");
      content.className = "tokens";
      for (const row of contentRows) {{
        appendToken(content, row, query);
      }}
      body.appendChild(content);
    }}

    appendTemplateLine(body, suffixTemplateRows, query, "after");

    card.appendChild(body);
    els.messages.appendChild(card);
  }}
  els.empty.classList.toggle("visible", visible === 0);
}}

function setText(id, value) {{
  id.textContent = value == null ? "" : String(value);
}}

function renderHighlightedText(node, value, query = searchQuery()) {{
  const text = value == null ? "" : String(value);
  node.textContent = "";
  if (!query) {{
    node.appendChild(document.createTextNode(text));
    return;
  }}

  const lowerText = text.toLowerCase();
  const lowerQuery = query.toLowerCase();
  let cursor = 0;
  while (cursor < text.length) {{
    const matchAt = lowerText.indexOf(lowerQuery, cursor);
    if (matchAt < 0) {{
      node.appendChild(document.createTextNode(text.slice(cursor)));
      break;
    }}
    if (matchAt > cursor) {{
      node.appendChild(document.createTextNode(text.slice(cursor, matchAt)));
    }}
    const mark = document.createElement("mark");
    mark.className = "text-highlight";
    mark.textContent = text.slice(matchAt, matchAt + query.length);
    node.appendChild(mark);
    cursor = matchAt + query.length;
  }}
}}

function renderDetail() {{
  const row = selectedRow ?? rows[0];
  setText(els.detailToken, tokenDetailLabel(row));
  setText(els.detailIndex, `#${{row.token_index ?? row.row_index ?? ""}}`);
  const section = sections.find((candidate) => candidate.rows.includes(row));
  setText(els.detailRole, section ? roleLabel(section.role) : "Trace");
  const meta = [
    ["Token ID", row.token_id],
    ["Row", row.row_index],
    ["Norm", fmtNumber(Number(row.activation_norm ?? 0), 4)],
    ["Chars", `${{row.char_start ?? ""}}-${{row.char_end ?? ""}}`],
    ["Special", row.is_special ? "yes" : "no"],
  ];
  els.detailMeta.innerHTML = meta.map(([key, value]) => `
    <div class="meta-item">
      <div class="meta-k">${{key}}</div>
      <div class="meta-v">${{value ?? ""}}</div>
    </div>
  `).join("");
  renderHighlightedText(els.detailOutput, row.nla_output ?? "");
}}

els.run.addEventListener("change", () => loadRun(els.run.value));

for (const input of [els.search, els.specials]) {{
  input.addEventListener("input", () => {{
    renderAll();
  }});
}}

const initialRunIndex = defaultRunIndex();
els.run.value = String(initialRunIndex);
loadRun(initialRunIndex);
</script>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace", help="Path to an NLA token trace JSONL file.")
    p.add_argument("--label", help="Picker label for TRACE.")
    p.add_argument(
        "--run",
        action="append",
        default=[],
        help="Additional picker run as PATH or LABEL=PATH. May be repeated.",
    )
    p.add_argument("--output", help="Output HTML path. Defaults to TRACE with .html suffix.")
    p.add_argument("--title", default="NLA Token Trace", help="Viewer title.")
    p.add_argument("--default-run", help="Initial picker run by label, source path, or zero-based index.")
    args = p.parse_args()

    trace_path = Path(args.trace)
    output_path = Path(args.output) if args.output else trace_path.with_suffix(".html")
    runs = [_run_payload(args.label, trace_path)]
    for spec in args.run:
        label, path = _parse_run_spec(spec)
        runs.append(_run_payload(label, path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_html(runs, args.title, args.default_run))
    row_count = sum(len(run["rows"]) for run in runs)
    print(f"wrote {row_count} trace rows across {len(runs)} run(s) to {output_path}")


if __name__ == "__main__":
    main()
