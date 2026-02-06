# LLM Benchmarking Framework

Benchmark multiple open-source LLMs on structured text processing tasks (German clinical text). Runs inference with multiple models, validates structured JSON output via Pydantic, computes evaluation metrics, and generates visualisations — all from a single CLI command.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_llms.py \
  --prompt prompt.txt \
  --csv data/input.csv \
  --output output/output.csv \
  --metrics metrics/output_metrics.csv \
  --model all \
  --plots-dir plots/
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--prompt` | Yes | Path to prompt template (`.txt`) with `{source}` placeholder |
| `--csv` | Yes | Input CSV with `source` column and target columns |
| `--output` | Yes | Output CSV path (predictions appended as `pred_*` columns) |
| `--metrics` | Yes | Metrics CSV output path |
| `--model` | No | `all`, tier (`3b`, `7b`, `30b`, `70b`, `thinking`), or short name (default: `all`) |
| `--skip-inference` | No | Only compute metrics from existing `pred_*` columns |
| `--skip-metrics` | No | Only run inference, skip metrics |
| `--max-new-tokens` | No | Max generation tokens (default: 1024) |
| `--plots-dir` | No | Directory to save visualisation plots (if omitted, no plots are generated) |

### Target Field Auto-Detection

Target fields are automatically detected from the CSV columns. All columns except `source` (and any `pred_*` / `raw_*` columns from previous runs) are treated as target fields. The Pydantic validation schema is built dynamically at runtime to match. No `--mode` flag is needed.

For example:
- A CSV with columns `source,output` will create a schema with one field: `output`
- A CSV with columns `source,foo,bar` will create a schema with fields: `bar`, `foo`

## Models

All models are loaded with NF4 4-bit quantization (via `bitsandbytes`) for consistent and memory-efficient inference.

| Tier | Short Name | HuggingFace ID |
|------|-----------|----------------|
| 3b | `llama3.2-3b` | `meta-llama/Llama-3.2-3B-Instruct` |
| 3b | `qwen2.5-3b` | `Qwen/Qwen2.5-3B-Instruct` |
| 3b | `phi3.5-mini` | `microsoft/Phi-3.5-mini-instruct` |
| 7b | `mistral-7b` | `mistralai/Mistral-7B-Instruct-v0.3` |
| 7b | `llama3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` |
| 7b | `qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` |
| 30b | `qwen2.5-32b` | `Qwen/Qwen2.5-32B-Instruct` |
| 30b | `mistral-small` | `mistralai/Mistral-Small-Instruct-2409` |
| 70b | `llama3.1-70b` | `meta-llama/Llama-3.1-70B-Instruct` |
| 70b | `qwen2.5-72b` | `Qwen/Qwen2.5-72B-Instruct` |
| thinking | `qwq-32b` | `Qwen/QwQ-32B-Preview` |
| thinking | `deepseek-r1` | `deepseek-ai/DeepSeek-R1` |

Thinking-tier models produce `<think>...</think>` reasoning blocks that are automatically stripped before JSON extraction.

## Metrics

### Quality Metrics

| Metric | Scope | Notes |
|--------|-------|-------|
| ROUGE-1/2/L | per-sentence, averaged | `use_stemmer=False` |
| BLEU | corpus-level | valid predictions only |
| chrF++ | corpus-level | `word_order=2`, strong for German |
| BERTScore F1 | per-sentence, averaged | `deepset/gbert-base` |
| Exact Match | per-sentence, averaged | case-insensitive, stripped |

### Performance Metrics

| Metric | Scope | Notes |
|--------|-------|-------|
| `runtime_s` | per-model | Wall-clock inference time in seconds |
| `tokens_in` | per-model | Total input tokens across all rows |
| `tokens_out` | per-model | Total output tokens across all rows |
| `tokens_per_sec` | per-model | Output throughput (`tokens_out / runtime_s`) |

The metrics CSV combines both quality and performance metrics. Each model gets one row per target field; performance columns repeat across fields since they are model-level.

### Metrics CSV Format

```csv
model,field,n_total,n_valid,rouge1,rouge2,rougeL,bleu,chrf,bertscore_f1,exact_match,runtime_s,tokens_in,tokens_out,tokens_per_sec
llama3.2-3b,output,100,95,45.2,22.1,40.3,18.5,52.1,0.78,5.0,120.5,50000,12000,99.6
```

## Visualisations

When `--plots-dir` is provided, the following plots are generated:

| Plot | Filename | Description |
|------|----------|-------------|
| Quality bars | `quality_{field}.png` | Grouped bar chart comparing ROUGE/BLEU/chrF/EM across models |
| Heatmap | `heatmap_{field}.png` | Models x metrics colour matrix with annotated values |
| Validation rate | `validation_rate_{field}.png` | Per-model bar chart of successful JSON parse rate |
| Radar | `radar_{field}.png` | Spider chart with normalised metrics overlay per model |
| Runtime | `runtime.png` | Horizontal bar chart of inference time per model |
| Tokens | `tokens.png` | Stacked bar chart of input + output tokens per model |
| Throughput | `throughput.png` | Bar chart of output tokens/sec per model |

Field-level plots are generated once per target field (e.g. `quality_refined_clinical_question_and_info.png`). Runtime/token/throughput plots are model-level and generated once.

## Crash Resilience

- Output CSV is saved after each model completes
- On re-run, models with existing `pred_*` columns are skipped automatically

## Tests

```bash
pytest tests/
```

Test fixtures (prompt templates and CSVs) are in `tests/fixtures/`.

## Project Structure

```
dira_timo_dev/
├── run_llms.py          # CLI entry point
├── models.py            # Model registry, loading, quantization
├── schemas.py           # Dynamic Pydantic output model creation
├── inference.py         # Inference engine (with runtime/token tracking)
├── evaluation.py        # Metrics computation (quality + performance)
├── visualize.py         # Plot generation (matplotlib)
├── utils.py             # CSV I/O, prompt loading, JSON extraction
├── requirements.txt
├── .gitignore
├── tests/
│   ├── conftest.py      # Shared pytest fixtures
│   ├── test_schemas.py
│   ├── test_models.py
│   ├── test_utils.py
│   ├── test_visualize.py
│   └── fixtures/
│       ├── prompt_single.txt
│       ├── prompt_table.txt
│       ├── prompt_bad.txt
│       ├── test_single.csv
│       └── test_table.csv
└── README.md
```
