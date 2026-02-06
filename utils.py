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
