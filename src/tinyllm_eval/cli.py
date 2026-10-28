"""Command-line interface for the small-model summarization evaluation harness."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .core import (
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_BASE_URL,
    OPENROUTER_DEFAULT_MODEL,
    PROMPT_VERSION,
    load_jsonl,
    request_judgment,
    request_summary,
    sample_stream,
    sha256_file,
    write_jsonl,
)


def prepare_dataset(args: argparse.Namespace) -> None:
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}; pass --overwrite to replace it")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Dataset preparation requires `pip install -e '.[data]'`") from exc

    stream = load_dataset(
        "abisee/cnn_dailymail",
        "3.0.0",
        split="test",
        streaming=True,
    )
    rows = sample_stream(stream, args.count, args.seed)
    for row in rows:
        row["dataset"] = "abisee/cnn_dailymail"
        row["dataset_config"] = "3.0.0"
        row["split"] = "test"
        row["sampling_seed"] = args.seed
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} test examples to {args.output}")
    print("Note: streaming reservoir sampling scans the test split; preserve the resulting file to reuse this exact sample.")


def _resolve_inference_auth(args: argparse.Namespace) -> tuple[str | None, bool]:
    parsed = urlsplit(args.base_url)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Inference base URL must not contain credentials, query, or fragment")
    host = (parsed.hostname or "").lower()
    is_local = parsed.scheme == "http" and host in {"localhost", "127.0.0.1", "::1"}
    is_openrouter = (
        parsed.scheme == "https"
        and host == "openrouter.ai"
        and parsed.path.rstrip("/") == "/api/v1"
    )
    if is_local:
        if args.api_key_env:
            raise ValueError("Do not pass an API key to a local inference endpoint")
        return None, False
    if not is_openrouter:
        raise ValueError("Inference base URL must be local HTTP or OpenRouter HTTPS at /api/v1")
    if not args.api_key_env:
        raise ValueError("OpenRouter inference requires --api-key-env OPENROUTER_API_KEY")
    load_project_env_file()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {args.api_key_env} in the environment or repo-root .env file")
    return api_key, True


def run_evaluation(args: argparse.Namespace) -> None:
    api_key, is_remote = _resolve_inference_auth(args)
    if is_remote and any(
        value is not None
        for value in (args.temperature, args.top_p, args.top_k, args.min_p, args.presence_penalty, args.repeat_penalty)
    ):
        raise ValueError("This OpenRouter control does not advertise sampling controls; omit sampler flags")
    if not is_remote and args.reasoning_effort is not None:
        raise ValueError("--reasoning-effort is only supported for OpenRouter inference")
    temperature = args.temperature if args.temperature is not None else (None if is_remote else 0.0)
    records = load_jsonl(args.input)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than zero")
        records = records[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}; pass --overwrite to replace it")

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_revision": args.model_revision,
        "quantization": args.quantization,
        "inference_server_version": args.server_version,
        "base_url": args.base_url,
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "example_count": len(records),
        "prompt_version": PROMPT_VERSION,
        "temperature": temperature,
        "reasoning_effort": args.reasoning_effort,
        "api_key_env": args.api_key_env,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repeat_penalty": args.repeat_penalty,
        "seed": 42,
        "max_new_tokens": args.max_new_tokens,
        "server_generation_settings": "Generation settings are sent per request; omitted values use endpoint/provider defaults",
    }
    metadata.update({"status": "running", "completed_examples": 0})
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    completed_examples = 0
    try:
        with args.output.open("w", encoding="utf-8") as stream:
            for index, row in enumerate(records, start=1):
                summary, response_metadata, elapsed = request_summary(
                    args.base_url,
                    args.model,
                    row["document"],
                    args.max_new_tokens,
                    args.timeout,
                    api_key=api_key,
                    temperature=temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    min_p=args.min_p,
                    presence_penalty=args.presence_penalty,
                    repeat_penalty=args.repeat_penalty,
                    reasoning_effort=args.reasoning_effort,
                )
                record: dict[str, Any] = {
                    "id": row["id"],
                    "document": row["document"],
                    "reference": row["reference"],
                    "prediction": summary,
                    "model": args.model,
                    "prompt_version": PROMPT_VERSION,
                    "latency_seconds": round(elapsed, 4),
                    **response_metadata,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                completed_examples = index
                metadata["completed_examples"] = completed_examples
                metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
                print(f"[{index}/{len(records)}] {row['id']} ({elapsed:.2f}s)", file=sys.stderr)
    except Exception as exc:
        metadata.update({"status": "failed", "completed_examples": completed_examples, "error": str(exc)})
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        raise

    metadata.update({"status": "complete", "completed_examples": completed_examples})
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote predictions to {args.output}")
    print(f"Run metadata: {metadata_path}")


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_project_env_file(path: Path | None = None) -> None:
    """Load the repo-local OpenRouter key without overriding the shell environment."""
    path = path or PROJECT_ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key == OPENROUTER_API_KEY_ENV and value and not os.environ.get(key):
            os.environ[key] = value


def judge_predictions(args: argparse.Namespace) -> None:
    load_project_env_file()
    api_key = os.environ.get(OPENROUTER_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Set {OPENROUTER_API_KEY_ENV} in your environment before judging")
    records = load_jsonl(args.input)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than zero")
        records = records[: args.limit]
    if any(not isinstance(row.get("prediction"), str) or not row["prediction"].strip() for row in records):
        raise ValueError("Every judged record must include a non-empty 'prediction'")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}; pass --overwrite to replace it")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    active_metadata_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    run_metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "judge_model": args.model,
        "example_count": len(records),
        "completed_examples": 0,
        "status": "running",
    }
    active_metadata_path.write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    try:
        with temporary_output.open("w", encoding="utf-8") as stream:
            for index, row in enumerate(records, start=1):
                judgment, metadata = request_judgment(
                    args.base_url,
                    api_key,
                    args.model,
                    row["document"],
                    row["prediction"],
                    args.timeout,
                )
                result = {
                    "id": row["id"],
                    "candidate_model": row.get("model"),
                    "judge_model": args.model,
                    "judgment": judgment,
                    **metadata,
                }
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                run_metadata["completed_examples"] = index
                active_metadata_path.write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
                print(f"[{index}/{len(records)}] {row['id']}", file=sys.stderr)
    except BaseException as exc:
        run_metadata.update({"status": "failed", "error": str(exc)})
        active_metadata_path.write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
        raise
    temporary_output.replace(args.output)
    run_metadata["status"] = "complete"
    active_metadata_path.write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    active_metadata_path.replace(metadata_path)
    print(f"Wrote LLM judgments to {args.output}")
    print(f"Judge run metadata: {metadata_path}")


def score_predictions(args: argparse.Namespace) -> None:
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise RuntimeError("Scoring requires `pip install -e '.[metrics]'`") from exc

    records = load_jsonl(args.input)
    if any(not isinstance(row.get("prediction"), str) or not row["prediction"].strip() for row in records):
        raise ValueError("Every scored record must include a non-empty 'prediction'")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    aggregate = {key: 0.0 for key in ("rouge1", "rouge2", "rougeL")}
    for row in records:
        scores = scorer.score(row["reference"], row["prediction"])
        for key in aggregate:
            aggregate[key] += scores[key].fmeasure
    result = {
        "input": str(args.input),
        "example_count": len(records),
        "metric": "ROUGE F1 (rouge-score, Porter stemming)",
        "scores": {key: value / len(records) for key, value in aggregate.items()},
        "warning": "Reference overlap is not a factuality or overall quality measure.",
    }
    output = args.output or args.input.with_suffix(".scores.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote scores to {output}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tinyllm-eval")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-cnn-dm", help="sample the CNN/DailyMail 3.0.0 test split")
    prepare.add_argument("--count", type=int, default=50)
    prepare.add_argument("--seed", type=int, default=20261028)
    prepare.add_argument("--output", type=Path, default=Path("data/eval/cnn_dailymail_test_pilot.jsonl"))
    prepare.add_argument("--overwrite", action="store_true")
    prepare.set_defaults(func=prepare_dataset)

    run = commands.add_parser("run", help="generate summaries through a local or OpenRouter endpoint")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--model", required=True, help="model identifier expected by the server")
    run.add_argument("--model-revision", default="unrecorded", help="GGUF Hub commit SHA or local file identity")
    run.add_argument("--quantization", default="unrecorded", help="for example Q8_0")
    run.add_argument("--server-version", default="unrecorded", help="for example llama.cpp commit SHA")
    run.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    run.add_argument("--max-new-tokens", type=int, default=192)
    run.add_argument("--temperature", type=float)
    run.add_argument("--top-p", type=float)
    run.add_argument("--top-k", type=int)
    run.add_argument("--min-p", type=float)
    run.add_argument("--presence-penalty", type=float)
    run.add_argument("--repeat-penalty", type=float)
    run.add_argument("--api-key-env", choices=[OPENROUTER_API_KEY_ENV])
    run.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "xhigh", "max"])
    run.add_argument("--timeout", type=float, default=600)
    run.add_argument("--limit", type=int)
    run.add_argument("--overwrite", action="store_true")
    run.set_defaults(func=run_evaluation)

    judge = commands.add_parser("judge", help="score predictions with an OpenRouter LLM judge")
    judge.add_argument("--input", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument("--model", default=OPENROUTER_DEFAULT_MODEL)
    judge.add_argument("--base-url", default=OPENROUTER_BASE_URL)
    judge.add_argument("--timeout", type=float, default=120)
    judge.add_argument("--limit", type=int)
    judge.add_argument("--overwrite", action="store_true")
    judge.set_defaults(func=judge_predictions)

    score = commands.add_parser("score", help="calculate ROUGE F1 from prediction JSONL")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output", type=Path)
    score.set_defaults(func=score_predictions)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
