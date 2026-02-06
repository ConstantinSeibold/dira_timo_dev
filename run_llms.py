#!/usr/bin/env python3
"""CLI entry point for the LLM benchmarking framework."""

import argparse
import logging
import sys

from evaluation import compute_metrics, print_metrics_summary
from inference import InferenceEngine
from models import get_models
from utils import load_csv, load_prompt, save_csv
from visualize import generate_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark LLMs on structured text processing tasks."
    )
    parser.add_argument(
        "--prompt", required=True, help="Path to prompt template (.txt) with {source} placeholder."
    )
    parser.add_argument(
        "--csv", required=True, help="Path to input CSV with 'source' and target columns."
    )
    parser.add_argument(
        "--output", required=True, help="Path for the output CSV with predictions."
    )
    parser.add_argument(
        "--metrics", required=True, help="Path for the metrics CSV."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["single", "table"],
        help="Output mode: 'single' (one field) or 'table' (three fields).",
    )
    parser.add_argument(
        "--model",
        default="all",
        help="Model selector: 'all', tier ('3b','7b','30b','70b','thinking'), or short name.",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference; only compute metrics from existing pred_* columns.",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Skip metrics computation; only run inference.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate (default: 1024).",
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help="Directory to save visualisation plots. If omitted, no plots are generated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("run_llms")

    # Load prompt template
    prompt_template = load_prompt(args.prompt)
    logger.info("Loaded prompt template from %s", args.prompt)

    # Load input CSV
    df = load_csv(args.csv, args.mode)
    logger.info("Loaded CSV with %d rows from %s", len(df), args.csv)

    # Resolve models
    models = get_models(args.model)
    logger.info(
        "Resolved %d model(s): %s",
        len(models),
        ", ".join(m.short_name for m in models),
    )

    # --- Inference ---
    run_stats: dict[str, dict] = {}
    if not args.skip_inference:
        engine = InferenceEngine(
            prompt_template=prompt_template,
            mode=args.mode,
            max_new_tokens=args.max_new_tokens,
        )
        df = engine.run(df, models, args.output)
        run_stats = engine.run_stats
        save_csv(df, args.output)
        logger.info("Final output saved to %s", args.output)
    else:
        # Load existing output CSV for metrics computation
        df = load_csv(args.output, args.mode)
        logger.info("Loaded existing output from %s (--skip-inference)", args.output)

    # --- Metrics ---
    if not args.skip_metrics:
        metrics_df = compute_metrics(df, args.mode, args.metrics, run_stats=run_stats)
        print_metrics_summary(metrics_df)

        # --- Plots ---
        if args.plots_dir:
            generate_plots(metrics_df, args.plots_dir)
            logger.info("Plots saved to %s", args.plots_dir)
    else:
        logger.info("Skipping metrics computation (--skip-metrics).")

    logger.info("Done.")


if __name__ == "__main__":
    main()
