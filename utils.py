import json
import re
from pathlib import Path

import pandas as pd



# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def load_prompt(path: str) -> str:
    """Read a prompt template from a text file. Validates that {source} placeholder exists."""
    text = Path(path).read_text(encoding="utf-8")
    if "{source}" not in text:
        raise ValueError(f"Prompt file {path} must contain a '{{source}}' placeholder.")
    return text


def format_prompt(template: str, source_text: str) -> str:
    """Replace {source} in the template with the actual source text."""
    return template.replace("{source}", source_text)


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    """Load CSV and validate that the 'source' column exists."""
    df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)

    if "source" not in df.columns:
        raise ValueError(
            f"CSV {path} is missing required 'source' column. "
            f"Found: {list(df.columns)}"
        )
    return df


def get_target_fields_from_df(df: pd.DataFrame) -> list[str]:
    """Return all columns except 'source' and any 'pred_*' / 'raw_*' columns.

    Sorted for deterministic order.
    """
    return sorted(
        col for col in df.columns
        if col != "source" and not col.startswith("pred_") and not col.startswith("raw_")
    )


def save_csv(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV, creating parent directories if needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Thinking-model output handling
# ---------------------------------------------------------------------------

def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks from thinking-model output.

    Handles multiline content, multiple blocks, and unclosed tags.
    Returns the text after the last </think> tag, or the original text
    if no thinking tags are found.
    """
    # Remove all complete <think>...</think> blocks (greedy across newlines)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Handle unclosed <think> tag (model still "thinking" when generation stopped)
    if "<think>" in cleaned:
        # Take everything before the unclosed <think>
        cleaned = cleaned.split("<think>")[0]

    return cleaned.strip()


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def extract_json_from_text(text: str) -> dict | None:
    """Try to extract a JSON object from model output.

    Level 1: Try parsing the full text as JSON.
    Level 2: Brace-counting extraction to find the outermost {...}.
             Handles markdown fencing, surrounding text, nested braces.
    Returns the parsed dict or None on failure.
    """
    # Level 1: direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, TypeError):
        pass

    # Level 1b: collapse doubled braces {{ -> { and }} -> }
    # Models sometimes mimic prompt escaping and output {{"key": "val"}}
    if "{{" in text or "}}" in text:
        collapsed = text.replace("{{", "{").replace("}}", "}")
        try:
            obj = json.loads(collapsed)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, TypeError):
            pass

    # Strip markdown code fences if present
    cleaned = text
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"```\s*$", "", cleaned)

    # Level 2: brace-counting
    start = cleaned.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(cleaned)):
        c = cleaned[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except (json.JSONDecodeError, TypeError):
                    pass
                # Try collapsing doubled braces in the candidate
                if "{{" in candidate or "}}" in candidate:
                    collapsed = candidate.replace("{{", "{").replace("}}", "}")
                    try:
                        obj = json.loads(collapsed)
                        if isinstance(obj, dict):
                            return obj
                    except (json.JSONDecodeError, TypeError):
                        pass
                return None
    return None


# ---------------------------------------------------------------------------
# JSON schema → plain-text rendering
# ---------------------------------------------------------------------------

# Indentation prefixes matching the ground-truth annotation style.
_INDENT_PREFIX = {
    0: "",
    1: "- ",
    2: "    - ",
    3: "        ",
}

# Row labels that should be excluded from the rendered plain text.
_SKIP_LABELS = frozenset({"Gelöschte Inhalte:", "Gelöschte Inhalte"})


def render_schema_to_plaintext(data: dict) -> str | None:
    """Convert a JSON table-schema dict to evaluation-ready plain text.

    Accepts the full schema (with ``table`` key) or just the table dict.
    Returns *None* when *data* does not look like a valid schema so callers
    can fall back to other strategies.
    """
    # Navigate to the rows list, tolerating both full-schema and bare-table input.
    table = data.get("table", data) if isinstance(data, dict) else None
    if not isinstance(table, dict):
        return None
    rows = table.get("rows")
    if not isinstance(rows, list) or not rows:
        return None

    sections: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = row.get("label", "")
        content = row.get("content")

        if label in _SKIP_LABELS:
            continue
        if not isinstance(content, list) or not content:
            continue

        lines: list[str] = [label]
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text", "")
            if not text:
                continue
            level = item.get("indent_level", 0)
            prefix = _INDENT_PREFIX.get(level, "")
            lines.append(f"{prefix}{text}")

        if len(lines) > 1:  # label + at least one content line
            sections.append("\n".join(lines))

    return "\n\n".join(sections) if sections else None


# ---------------------------------------------------------------------------
# Column naming helpers
# ---------------------------------------------------------------------------

def get_pred_column_name(field: str, model_short_name: str) -> str:
    """Return the prediction column name: pred_{field}_{short_name}."""
    return f"pred_{field}_{model_short_name}"


def discover_models_in_df(df: pd.DataFrame, fields: list[str]) -> list[str]:
    """Find all model short names from existing pred_* columns in a DataFrame."""
    models: set[str] = set()
    for col in df.columns:
        for field in fields:
            prefix = f"pred_{field}_"
            if col.startswith(prefix):
                short_name = col[len(prefix):]
                models.add(short_name)
    return sorted(models)
