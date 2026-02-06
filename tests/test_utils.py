import pandas as pd
import pytest

from utils import (
    discover_models_in_df,
    extract_json_from_text,
    format_prompt,
    get_pred_column_name,
    get_target_fields_from_df,
    load_csv,
    load_prompt,
    save_csv,
    strip_thinking_tags,
)


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

class TestLoadPrompt:
    def test_valid_single(self, prompt_single_path):
        result = load_prompt(prompt_single_path)
        assert "{source}" in result

    def test_valid_table(self, prompt_table_path):
        result = load_prompt(prompt_table_path)
        assert "{source}" in result

    def test_missing_placeholder(self, prompt_bad_path):
        with pytest.raises(ValueError, match="placeholder"):
            load_prompt(prompt_bad_path)


class TestFormatPrompt:
    def test_replaces_source(self, prompt_single_path):
        template = load_prompt(prompt_single_path)
        result = format_prompt(template, "Kopfschmerzen seit drei Tagen")
        assert "Kopfschmerzen seit drei Tagen" in result
        assert "{source}" not in result

    def test_multiple_occurrences(self):
        result = format_prompt("{source} and {source}", "X")
        assert result == "X and X"


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

class TestLoadCsv:
    def test_single_csv(self, csv_single_path):
        df = load_csv(csv_single_path)
        assert "source" in df.columns
        assert "output" in df.columns
        assert len(df) == 5

    def test_table_csv(self, csv_table_path):
        df = load_csv(csv_table_path)
        assert "source" in df.columns
        assert "refined_clinical_question_and_info" in df.columns
        assert "organizational_information" in df.columns
        assert "deleted_input" in df.columns
        assert len(df) == 5

    def test_missing_source_column(self, tmp_path):
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("foo,bar\n1,2\n")
        with pytest.raises(ValueError, match="source"):
            load_csv(str(csv_path))


class TestGetTargetFieldsFromDf:
    def test_single_field(self):
        df = pd.DataFrame({"source": ["s"], "output": ["o"]})
        assert get_target_fields_from_df(df) == ["output"]

    def test_multiple_fields(self):
        df = pd.DataFrame({
            "source": ["s"],
            "refined_clinical_question_and_info": ["r"],
            "organizational_information": ["o"],
            "deleted_input": ["d"],
        })
        assert get_target_fields_from_df(df) == [
            "deleted_input",
            "organizational_information",
            "refined_clinical_question_and_info",
        ]

    def test_excludes_pred_and_raw_columns(self):
        df = pd.DataFrame({
            "source": ["s"],
            "output": ["o"],
            "pred_output_model-a": ["x"],
            "raw_model-a": ["r"],
        })
        assert get_target_fields_from_df(df) == ["output"]

    def test_arbitrary_columns(self):
        df = pd.DataFrame({"source": ["s"], "foo": ["f"], "bar": ["b"]})
        assert get_target_fields_from_df(df) == ["bar", "foo"]


class TestSaveCsv:
    def test_creates_parents(self, tmp_path):
        out = tmp_path / "sub" / "dir" / "out.csv"
        df = pd.DataFrame({"a": [1]})
        save_csv(df, str(out))
        assert out.exists()
        loaded = pd.read_csv(out)
        assert list(loaded.columns) == ["a"]


# ---------------------------------------------------------------------------
# Thinking tags
# ---------------------------------------------------------------------------

class TestStripThinkingTags:
    def test_single_block(self):
        text = '<think>\nreasoning here\n</think>\n{"output": "answer"}'
        assert strip_thinking_tags(text) == '{"output": "answer"}'

    def test_multiple_blocks(self):
        text = "<think>a</think>middle<think>b</think>end"
        assert strip_thinking_tags(text) == "middleend"

    def test_unclosed_tag(self):
        text = "<think>still thinking..."
        assert strip_thinking_tags(text) == ""

    def test_no_tags(self):
        text = '{"output": "plain"}'
        assert strip_thinking_tags(text) == '{"output": "plain"}'

    def test_thinking_then_json(self):
        text = (
            "<think>\nDer Patient hat Kopfschmerzen.\n"
            "Ich muss das als JSON ausgeben.\n</think>\n"
            '```json\n{"output": "Kopfschmerzen"}\n```'
        )
        stripped = strip_thinking_tags(text)
        result = extract_json_from_text(stripped)
        assert result == {"output": "Kopfschmerzen"}


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_plain_json(self):
        assert extract_json_from_text('{"a": 1}') == {"a": 1}

    def test_markdown_fenced(self):
        text = '```json\n{"key": "val"}\n```'
        assert extract_json_from_text(text) == {"key": "val"}

    def test_surrounding_text(self):
        text = 'Here is the result: {"x": "y"} done.'
        assert extract_json_from_text(text) == {"x": "y"}

    def test_nested_braces(self):
        text = '{"a": {"b": 1}}'
        assert extract_json_from_text(text) == {"a": {"b": 1}}

    def test_no_json(self):
        assert extract_json_from_text("no json here") is None

    def test_empty_string(self):
        assert extract_json_from_text("") is None

    def test_table_output_json(self):
        text = '{"refined_clinical_question_and_info": "Kopfschmerzen", "organizational_information": "Termin Montag", "deleted_input": "MfG"}'
        result = extract_json_from_text(text)
        assert result["refined_clinical_question_and_info"] == "Kopfschmerzen"
        assert result["organizational_information"] == "Termin Montag"
        assert result["deleted_input"] == "MfG"

    def test_double_braces(self):
        """Models sometimes mimic prompt escaping and output doubled braces."""
        text = '{{"output": "Kopfschmerzen seit drei Tagen"}}'
        result = extract_json_from_text(text)
        assert result == {"output": "Kopfschmerzen seit drei Tagen"}

    def test_double_braces_with_surrounding_text(self):
        text = 'Here is the result:\n{{"output": "answer"}}\n'
        result = extract_json_from_text(text)
        assert result == {"output": "answer"}

    def test_double_braces_markdown_fenced(self):
        text = '```json\n{{"output": "answer"}}\n```'
        result = extract_json_from_text(text)
        assert result == {"output": "answer"}


# ---------------------------------------------------------------------------
# Column naming
# ---------------------------------------------------------------------------

class TestColumnNaming:
    def test_pred_column_name(self):
        assert get_pred_column_name("output", "qwen2.5-3b") == "pred_output_qwen2.5-3b"

    def test_pred_column_name_table_field(self):
        assert (
            get_pred_column_name("refined_clinical_question_and_info", "mistral-7b")
            == "pred_refined_clinical_question_and_info_mistral-7b"
        )

    def test_discover_models_single(self):
        df = pd.DataFrame({
            "source": ["s"],
            "output": ["o"],
            "pred_output_model-a": ["x"],
            "pred_output_model-b": ["y"],
        })
        assert discover_models_in_df(df, ["output"]) == ["model-a", "model-b"]

    def test_discover_models_table(self):
        df = pd.DataFrame({
            "source": ["s"],
            "refined_clinical_question_and_info": ["r"],
            "organizational_information": ["o"],
            "deleted_input": ["d"],
            "pred_refined_clinical_question_and_info_modelX": ["a"],
            "pred_organizational_information_modelX": ["b"],
            "pred_deleted_input_modelX": ["c"],
        })
        fields = ["deleted_input", "organizational_information", "refined_clinical_question_and_info"]
        assert discover_models_in_df(df, fields) == ["modelX"]

    def test_discover_no_models(self):
        df = pd.DataFrame({"source": ["s"], "output": ["o"]})
        assert discover_models_in_df(df, ["output"]) == []
