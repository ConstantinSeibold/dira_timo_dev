import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # non-interactive backend for CI

from visualize import (
    _plot_quality_bars,
    _plot_quality_heatmap,
    _plot_radar,
    _plot_runtime,
    _plot_throughput,
    _plot_tokens,
    _plot_validation_rate,
    generate_plots,
)

# ---------------------------------------------------------------------------
# Helpers to build synthetic metrics DataFrames
# ---------------------------------------------------------------------------

def _make_single_metrics(n_models: int = 3, with_runtime: bool = True) -> pd.DataFrame:
    """Create a realistic single-mode metrics DataFrame."""
    rows = []
    for i in range(n_models):
        row = {
            "model": f"model-{chr(65 + i)}",
            "field": "output",
            "n_total": 100,
            "n_valid": 100 - i * 5,
            "rouge1": 40 + i * 5,
            "rouge2": 20 + i * 3,
            "rougeL": 35 + i * 4,
            "bleu": 15 + i * 4,
            "chrf": 45 + i * 6,
            "bertscore_f1": round(0.7 + i * 0.05, 4),
            "exact_match": 3 + i * 2,
        }
        if with_runtime:
            row.update({
                "runtime_s": 80 + i * 40,
                "tokens_in": 50000 + i * 1000,
                "tokens_out": 10000 + i * 2000,
                "tokens_per_sec": round(100 - i * 15, 1),
            })
        else:
            row.update({
                "runtime_s": "",
                "tokens_in": "",
                "tokens_out": "",
                "tokens_per_sec": "",
            })
        rows.append(row)
    return pd.DataFrame(rows)


def _make_table_metrics(n_models: int = 2, with_runtime: bool = True) -> pd.DataFrame:
    """Create a realistic table-mode metrics DataFrame (3 fields per model)."""
    fields = [
        "refined_clinical_question_and_info",
        "organizational_information",
        "deleted_input",
    ]
    rows = []
    for i in range(n_models):
        for j, field in enumerate(fields):
            row = {
                "model": f"model-{chr(65 + i)}",
                "field": field,
                "n_total": 50,
                "n_valid": 50 - i * 3 - j,
                "rouge1": 30 + i * 8 + j * 3,
                "rouge2": 15 + i * 5 + j * 2,
                "rougeL": 28 + i * 7 + j * 3,
                "bleu": 10 + i * 6 + j * 2,
                "chrf": 40 + i * 8 + j * 4,
                "bertscore_f1": round(0.65 + i * 0.06 + j * 0.02, 4),
                "exact_match": 2 + i + j,
            }
            if with_runtime:
                row.update({
                    "runtime_s": 120 + i * 60,
                    "tokens_in": 30000 + i * 5000,
                    "tokens_out": 8000 + i * 3000,
                    "tokens_per_sec": round(80 - i * 20, 1),
                })
            else:
                row.update({
                    "runtime_s": "",
                    "tokens_in": "",
                    "tokens_out": "",
                    "tokens_per_sec": "",
                })
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests for generate_plots (integration)
# ---------------------------------------------------------------------------

class TestGeneratePlots:
    """Integration tests: generate_plots produces the expected files."""

    def test_single_mode_with_runtime(self, tmp_path):
        df = _make_single_metrics(n_models=3, with_runtime=True)
        generate_plots(df, str(tmp_path))

        expected = {
            "quality_output.png",
            "heatmap_output.png",
            "validation_rate_output.png",
            "radar_output.png",
            "runtime.png",
            "tokens.png",
            "throughput.png",
        }
        created = {p.name for p in tmp_path.iterdir()}
        assert expected == created

    def test_single_mode_without_runtime(self, tmp_path):
        df = _make_single_metrics(n_models=2, with_runtime=False)
        generate_plots(df, str(tmp_path))

        created = {p.name for p in tmp_path.iterdir()}
        # Runtime-related plots should NOT be created
        assert "runtime.png" not in created
        assert "tokens.png" not in created
        assert "throughput.png" not in created
        # Quality plots should still exist
        assert "quality_output.png" in created
        assert "heatmap_output.png" in created
        assert "radar_output.png" in created

    def test_table_mode_with_runtime(self, tmp_path):
        df = _make_table_metrics(n_models=2, with_runtime=True)
        generate_plots(df, str(tmp_path))

        created = {p.name for p in tmp_path.iterdir()}
        # One plot per field for field-level charts
        for field in [
            "refined_clinical_question_and_info",
            "organizational_information",
            "deleted_input",
        ]:
            assert f"quality_{field}.png" in created
            assert f"heatmap_{field}.png" in created
            assert f"validation_rate_{field}.png" in created
            assert f"radar_{field}.png" in created
        # Model-level plots (one each)
        assert "runtime.png" in created
        assert "tokens.png" in created
        assert "throughput.png" in created

    def test_empty_dataframe(self, tmp_path):
        generate_plots(pd.DataFrame(), str(tmp_path))
        assert list(tmp_path.iterdir()) == []

    def test_creates_output_dir(self, tmp_path):
        nested = tmp_path / "sub" / "plots"
        df = _make_single_metrics(n_models=2)
        generate_plots(df, str(nested))
        assert nested.exists()
        assert len(list(nested.iterdir())) > 0

    def test_no_matplotlib_warnings(self, tmp_path):
        """Ensure no UserWarnings (e.g. set_ticklabels) are raised."""
        df = _make_single_metrics(n_models=4, with_runtime=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            generate_plots(df, str(tmp_path))

    def test_single_model(self, tmp_path):
        """Edge case: only one model should still produce valid plots."""
        df = _make_single_metrics(n_models=1, with_runtime=True)
        generate_plots(df, str(tmp_path))
        created = {p.name for p in tmp_path.iterdir()}
        assert "quality_output.png" in created
        assert "radar_output.png" in created


# ---------------------------------------------------------------------------
# Tests for individual plot functions
# ---------------------------------------------------------------------------

class TestPlotQualityBars:
    def test_creates_file_per_field(self, tmp_path):
        df = _make_table_metrics(n_models=2)
        colors = {"model-A": "#4C72B0", "model-B": "#DD8452"}
        _plot_quality_bars(df, colors, tmp_path)
        for field in [
            "refined_clinical_question_and_info",
            "organizational_information",
            "deleted_input",
        ]:
            assert (tmp_path / f"quality_{field}.png").exists()


class TestPlotHeatmap:
    def test_creates_file(self, tmp_path):
        df = _make_single_metrics(n_models=3)
        _plot_quality_heatmap(df, tmp_path)
        assert (tmp_path / "heatmap_output.png").exists()

    def test_file_nonzero_size(self, tmp_path):
        df = _make_single_metrics(n_models=2)
        _plot_quality_heatmap(df, tmp_path)
        assert (tmp_path / "heatmap_output.png").stat().st_size > 0


class TestPlotValidationRate:
    def test_creates_file(self, tmp_path):
        df = _make_single_metrics(n_models=2)
        _plot_validation_rate(df, tmp_path)
        assert (tmp_path / "validation_rate_output.png").exists()


class TestPlotRuntime:
    def test_creates_file(self, tmp_path):
        df = _make_single_metrics(n_models=3, with_runtime=True)
        colors = {f"model-{chr(65+i)}": c for i, c in enumerate(["#4C72B0", "#DD8452", "#55A868"])}
        _plot_runtime(df, colors, tmp_path)
        assert (tmp_path / "runtime.png").exists()

    def test_skips_when_no_runtime(self, tmp_path):
        df = _make_single_metrics(n_models=2, with_runtime=False)
        colors = {"model-A": "#4C72B0", "model-B": "#DD8452"}
        _plot_runtime(df, colors, tmp_path)
        assert not (tmp_path / "runtime.png").exists()


class TestPlotTokens:
    def test_creates_file(self, tmp_path):
        df = _make_single_metrics(n_models=2, with_runtime=True)
        colors = {"model-A": "#4C72B0", "model-B": "#DD8452"}
        _plot_tokens(df, colors, tmp_path)
        assert (tmp_path / "tokens.png").exists()

    def test_skips_when_no_tokens(self, tmp_path):
        df = _make_single_metrics(n_models=2, with_runtime=False)
        colors = {"model-A": "#4C72B0", "model-B": "#DD8452"}
        _plot_tokens(df, colors, tmp_path)
        assert not (tmp_path / "tokens.png").exists()


class TestPlotThroughput:
    def test_creates_file(self, tmp_path):
        df = _make_single_metrics(n_models=2, with_runtime=True)
        colors = {"model-A": "#4C72B0", "model-B": "#DD8452"}
        _plot_throughput(df, colors, tmp_path)
        assert (tmp_path / "throughput.png").exists()

    def test_skips_when_no_throughput(self, tmp_path):
        df = _make_single_metrics(n_models=2, with_runtime=False)
        colors = {"model-A": "#4C72B0", "model-B": "#DD8452"}
        _plot_throughput(df, colors, tmp_path)
        assert not (tmp_path / "throughput.png").exists()


class TestPlotRadar:
    def test_creates_file(self, tmp_path):
        df = _make_single_metrics(n_models=3)
        colors = {f"model-{chr(65+i)}": c for i, c in enumerate(["#4C72B0", "#DD8452", "#55A868"])}
        _plot_radar(df, colors, tmp_path)
        assert (tmp_path / "radar_output.png").exists()

    def test_handles_zero_metrics(self, tmp_path):
        """Radar should not crash if all metrics are 0 (normalisation edge case)."""
        df = pd.DataFrame([{
            "model": "zero-model", "field": "output",
            "n_total": 10, "n_valid": 0,
            "rouge1": 0, "rouge2": 0, "rougeL": 0,
            "bleu": 0, "chrf": 0, "bertscore_f1": 0, "exact_match": 0,
            "runtime_s": "", "tokens_in": "", "tokens_out": "", "tokens_per_sec": "",
        }])
        colors = {"zero-model": "#999"}
        _plot_radar(df, colors, tmp_path)
        assert (tmp_path / "radar_output.png").exists()
