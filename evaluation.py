import logging

import pandas as pd
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer
import sacrebleu

from schemas import get_target_fields
from utils import discover_models_in_df, get_pred_column_name, save_csv

logger = logging.getLogger(__name__)


def compute_metrics(
    df: pd.DataFrame,
    mode: str,
    metrics_path: str,
    run_stats: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Compute evaluation metrics for all models found in the DataFrame.

    Args:
        df: DataFrame with predictions in pred_* columns.
        mode: "single" or "table".
        metrics_path: Path to save the metrics CSV.
        run_stats: Optional dict mapping model short_name to inference stats
                   (runtime_s, tokens_in, tokens_out) from InferenceEngine.

    Returns a metrics DataFrame and saves it to metrics_path.
    """
    fields = get_target_fields(mode)
    model_names = discover_models_in_df(df, mode)

    if not model_names:
        logger.warning("No model predictions found in DataFrame.")
        return pd.DataFrame()

    run_stats = run_stats or {}

    rows = []
    for model_name in model_names:
        model_stats = run_stats.get(model_name, {})

        for field in fields:
            pred_col = get_pred_column_name(field, model_name)
            if pred_col not in df.columns:
                continue

            preds = df[pred_col].astype(str).tolist()
            refs = df[field].astype(str).tolist()

            n_total = len(preds)

            # Identify valid (non-empty) predictions
            valid_mask = [p.strip() != "" for p in preds]
            n_valid = sum(valid_mask)

            metrics = _compute_field_metrics(preds, refs, valid_mask)
            metrics["model"] = model_name
            metrics["field"] = field
            metrics["n_total"] = n_total
            metrics["n_valid"] = n_valid

            # Attach runtime / token stats (same for all fields of a model)
            metrics["runtime_s"] = model_stats.get("runtime_s", "")
            metrics["tokens_in"] = model_stats.get("tokens_in", "")
            metrics["tokens_out"] = model_stats.get("tokens_out", "")
            tok_out = model_stats.get("tokens_out", 0)
            rt = model_stats.get("runtime_s", 0)
            metrics["tokens_per_sec"] = (
                round(tok_out / rt, 1) if rt and rt > 0 else ""
            )

            rows.append(metrics)

    col_order = [
        "model", "field", "n_total", "n_valid",
        "rouge1", "rouge2", "rougeL",
        "bleu", "chrf",
        "bertscore_f1", "exact_match",
        "runtime_s", "tokens_in", "tokens_out", "tokens_per_sec",
    ]
    metrics_df = pd.DataFrame(rows, columns=col_order)

    save_csv(metrics_df, metrics_path)
    logger.info("Metrics saved to %s", metrics_path)

    return metrics_df


def _compute_field_metrics(
    preds: list[str], refs: list[str], valid_mask: list[bool]
) -> dict:
    """Compute all metrics for a single field across all rows."""
    n_total = len(preds)
    n_valid = sum(valid_mask)

    # --- ROUGE (per-sentence, averaged; empty preds count as 0) ---
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False
    )
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ref in zip(preds, refs):
        if pred.strip() == "":
            rouge1_scores.append(0.0)
            rouge2_scores.append(0.0)
            rougeL_scores.append(0.0)
        else:
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores["rouge1"].fmeasure)
            rouge2_scores.append(scores["rouge2"].fmeasure)
            rougeL_scores.append(scores["rougeL"].fmeasure)

    rouge1 = _safe_mean(rouge1_scores)
    rouge2 = _safe_mean(rouge2_scores)
    rougeL = _safe_mean(rougeL_scores)

    # --- BLEU and chrF++ (corpus-level, valid predictions only) ---
    valid_preds = [p for p, m in zip(preds, valid_mask) if m]
    valid_refs = [r for r, m in zip(refs, valid_mask) if m]

    if valid_preds:
        bleu = sacrebleu.corpus_bleu(valid_preds, [valid_refs]).score
        chrf = sacrebleu.corpus_chrf(valid_preds, [valid_refs], word_order=2).score
    else:
        bleu = 0.0
        chrf = 0.0

    # --- BERTScore (per-sentence, averaged; empty preds count as 0) ---
    bertscore_f1 = _compute_bertscore(preds, refs, valid_mask)

    # --- Exact Match (case-insensitive, stripped; empty preds count as 0) ---
    em_scores = []
    for pred, ref in zip(preds, refs):
        if pred.strip() == "":
            em_scores.append(0.0)
        else:
            em_scores.append(1.0 if pred.strip().lower() == ref.strip().lower() else 0.0)
    exact_match = _safe_mean(em_scores)

    return {
        "rouge1": round(rouge1 * 100, 2),
        "rouge2": round(rouge2 * 100, 2),
        "rougeL": round(rougeL * 100, 2),
        "bleu": round(bleu, 2),
        "chrf": round(chrf, 2),
        "bertscore_f1": round(bertscore_f1, 4),
        "exact_match": round(exact_match * 100, 2),
    }


def _compute_bertscore(
    preds: list[str], refs: list[str], valid_mask: list[bool]
) -> float:
    """Compute average BERTScore F1 using deepset/gbert-base.

    Empty predictions count as 0.
    """
    valid_preds = [p for p, m in zip(preds, valid_mask) if m]
    valid_refs = [r for r, m in zip(refs, valid_mask) if m]

    n_total = len(preds)
    n_valid = len(valid_preds)

    if n_valid == 0:
        return 0.0

    _, _, f1 = bert_score_fn(
        valid_preds, valid_refs, model_type="deepset/gbert-base", verbose=False
    )

    # Average over all rows (empty predictions contribute 0)
    valid_sum = float(f1.sum())
    return valid_sum / n_total


def _safe_mean(values: list[float]) -> float:
    """Return mean of a list, or 0.0 if empty."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def print_metrics_summary(metrics_df: pd.DataFrame) -> None:
    """Print a human-readable summary table of metrics."""
    if metrics_df.empty:
        print("No metrics to display.")
        return

    print("\n" + "=" * 120)
    print("EVALUATION RESULTS")
    print("=" * 120)
    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )
    print("=" * 120 + "\n")
