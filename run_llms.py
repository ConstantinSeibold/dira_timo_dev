#!/usr/bin/env python3
"""CLI entry point for the LLM benchmarking framework."""

import argparse
import logging
import sys

from evaluation import compute_metrics, print_metrics_summary
from inference import InferenceEngine
from models import get_models
from utils import get_target_fields_from_df, load_csv, load_prompt, save_csv
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
        default=8192,
        help="Maximum number of new tokens to generate (default: 8192).",
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help="Directory to save visualisation plots. If omitted, no plots are generated.",
    )
    parser.add_argument(
        "--quantize-4bit",
        action="store_true",
        help="Enable NF4 4-bit quantization via bitsandbytes (requires bitsandbytes).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging (shows raw model output on parse failures).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("run_llms")

    # Load prompt template
    prompt_template = load_prompt(args.prompt)
    logger.info("Loaded prompt template from %s", args.prompt)

    # Load input CSV
    df = load_csv(args.csv)
    logger.info("Loaded CSV with %d rows from %s", len(df), args.csv)

    # Auto-detect target fields
    fields = get_target_fields_from_df(df)
    logger.info("Detected target fields: %s", fields)

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
            fields=fields,
            max_new_tokens=args.max_new_tokens,
            quantize_4bit=args.quantize_4bit,
        )
        df = engine.run(df, models, args.output)
        run_stats = engine.run_stats
        save_csv(df, args.output)
        logger.info("Final output saved to %s", args.output)
    else:
        # Load existing output CSV for metrics computation
        df = load_csv(args.output)
        fields = get_target_fields_from_df(df)
        logger.info("Loaded existing output from %s (--skip-inference)", args.output)

    # --- Metrics ---
    if not args.skip_metrics:
        metrics_df = compute_metrics(df, fields, args.metrics, run_stats=run_stats)
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
