import json
import logging
import time

import torch

from models import ModelConfig, load_model_and_tokenizer, unload_model
from schemas import get_schema, get_target_fields
from utils import (
    extract_json_from_text,
    format_prompt,
    get_pred_column_name,
    save_csv,
    strip_thinking_tags,
)

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Runs inference for one or more models on a DataFrame of prompts."""

    def __init__(self, prompt_template: str, mode: str, max_new_tokens: int = 1024):
        self.prompt_template = prompt_template
        self.mode = mode
        self.max_new_tokens = max_new_tokens
        self.schema_cls = get_schema(mode)
        self.fields = get_target_fields(mode)
        self.run_stats: dict[str, dict] = {}

    def _model_already_done(self, df, config: ModelConfig) -> bool:
        """Check whether all pred columns for this model already exist."""
        return all(
            get_pred_column_name(f, config.short_name) in df.columns
            for f in self.fields
        )

    def run(self, df, models: list[ModelConfig], output_path: str):
        """Run inference for each model sequentially, saving after each.

        Per-model runtime and token stats are stored in self.run_stats.
        """
        for config in models:
            if self._model_already_done(df, config):
                logger.info(
                    "Skipping %s — pred columns already exist.", config.short_name
                )
                continue

            logger.info("Loading model %s (%s)...", config.short_name, config.hf_id)
            model, tokenizer = load_model_and_tokenizer(config)

            stats = self._run_single_model(df, model, tokenizer, config)
            self.run_stats[config.short_name] = stats

            logger.info(
                "Model %s — total: %d, json_ok: %d, pydantic_ok: %d (%.1f%%) | "
                "%.1fs, %d tok_in, %d tok_out, %.1f tok/s",
                config.short_name,
                stats["total_rows"],
                stats["json_parse_success"],
                stats["pydantic_valid"],
                stats["validation_rate"] * 100,
                stats["runtime_s"],
                stats["tokens_in"],
                stats["tokens_out"],
                stats["tokens_out"] / stats["runtime_s"] if stats["runtime_s"] > 0 else 0,
            )

            save_csv(df, output_path)
            logger.info("Saved intermediate results to %s", output_path)

            unload_model(model, tokenizer)

        return df

    def _run_single_model(self, df, model, tokenizer, config: ModelConfig) -> dict:
        """Run inference for one model across all rows."""
        # Initialize prediction columns
        for field in self.fields:
            col = get_pred_column_name(field, config.short_name)
            df[col] = ""

        total = len(df)
        json_ok = 0
        pydantic_ok = 0
        total_tokens_in = 0
        total_tokens_out = 0

        t_start = time.perf_counter()

        for idx in range(total):
            source_text = str(df.at[idx, "source"])
            prompt = format_prompt(self.prompt_template, source_text)

            raw_output, n_in, n_out = self._generate(model, tokenizer, prompt)
            total_tokens_in += n_in
            total_tokens_out += n_out

            # Strip <think>...</think> blocks for thinking models
            if config.is_thinking:
                raw_output = strip_thinking_tags(raw_output)

            parsed = self._parse_output(raw_output)

            if parsed["json_success"]:
                json_ok += 1
            if parsed["pydantic_valid"]:
                pydantic_ok += 1

            for field in self.fields:
                col = get_pred_column_name(field, config.short_name)
                df.at[idx, col] = parsed["fields"].get(field, "")

            if (idx + 1) % 10 == 0 or idx == total - 1:
                logger.info(
                    "  [%s] %d/%d rows processed", config.short_name, idx + 1, total
                )

        runtime_s = time.perf_counter() - t_start

        return {
            "total_rows": total,
            "json_parse_success": json_ok,
            "pydantic_valid": pydantic_ok,
            "validation_rate": pydantic_ok / total if total > 0 else 0.0,
            "runtime_s": round(runtime_s, 2),
            "tokens_in": total_tokens_in,
            "tokens_out": total_tokens_out,
        }

    def _generate(self, model, tokenizer, prompt: str) -> tuple[str, int, int]:
        """Generate text from a prompt using chat template.

        Returns (decoded_text, n_input_tokens, n_output_tokens).
        """
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        n_input_tokens = input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0, n_input_tokens:]
        n_output_tokens = len(new_tokens)

        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return decoded, n_input_tokens, n_output_tokens

    def _parse_output(self, raw_text: str) -> dict:
        """Parse raw model output into structured fields.

        Returns dict with keys:
          - json_success: bool
          - pydantic_valid: bool
          - fields: dict[str, str]  (field_name -> value, "" on failure)
        """
        result = {
            "json_success": False,
            "pydantic_valid": False,
            "fields": {f: "" for f in self.fields},
        }

        # Level 1 + 2: extract JSON
        parsed_dict = extract_json_from_text(raw_text)
        if parsed_dict is None:
            return result

        result["json_success"] = True

        # Level 3: Pydantic validation
        try:
            validated = self.schema_cls.model_validate(parsed_dict)
            result["pydantic_valid"] = True
            for field in self.fields:
                result["fields"][field] = getattr(validated, field)
        except Exception:
            # Partial: store whatever fields exist and are strings
            for field in self.fields:
                val = parsed_dict.get(field)
                if isinstance(val, str):
                    result["fields"][field] = val

        return result
