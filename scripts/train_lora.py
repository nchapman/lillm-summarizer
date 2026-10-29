"""Full LoRA training run on the stratified summarization subset.

Trains with epoch- or step-based validation loss, early stopping, and best-checkpoint
restoration; saves the native adapter plus a metadata record; and repeats the
smoke-test artifact checks (held-out generations and a fresh-base reload).

Usage (Granite 4.0 1B defaults):
    uv run --locked --group train python scripts/train_lora.py --adapter-dir adapters/granite4-1b-24k-s42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from tinyllm_eval.core import SYSTEM_PROMPT, PROMPT_VERSION
from tinyllm_eval.training_data import load_chat_rows
from tinyllm_eval.training_checks import (
    check_fresh, load_prepared, source_groups, checkpoint_schedule,
    check_step_budget, apply_step_budget,
)

from tinyllm_eval.training_audit import audit_prepared_dataset

DEFAULT_MODEL = "ibm-granite/granite-4.0-1b"
DEFAULT_TRAIN = Path("data/train/smoltalk2_stratified_train.jsonl")
DEFAULT_VAL = Path("data/train/smoltalk2_stratified_val.jsonl")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "input_linear", "output_linear"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-manifest", type=Path)
    parser.add_argument("--per-device-eval-batch-size", type=int)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=1, help="early-stopping patience in evaluation events (epochs by default)")
    parser.add_argument("--eval-steps", type=int, help="evaluate and save every N optimizer steps; default: every epoch")
    parser.add_argument("--stop-after-steps", type=int,
                        help="stop normally after N optimizer steps, retaining all checkpoints; "
                             "preserves the epoch-based LR horizon; requires a multiple of --eval-steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", choices=("unsloth", "standard", "off"), default="unsloth",
                        help="memory/compute tradeoff; benchmark on the target GPU before disabling")
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--limit", type=int, help="cap training rows (smoke only)")
    parser.add_argument("--val-limit", type=int, help="optional seeded validation sample; default uses the entire split")
    parser.add_argument("--sample-count", type=int, default=3, help="held-out generations shown")
    parser.add_argument("--resume", type=Path, help="checkpoint directory to resume from")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    eval_batch_size = args.per_device_eval_batch_size if args.per_device_eval_batch_size is not None else args.batch_size
    if min(args.batch_size, eval_batch_size, args.grad_accum, args.max_length) <= 0:
        raise ValueError("batch sizes, accumulation and max length must be positive")
    schedule = checkpoint_schedule(args.eval_steps)
    check_step_budget(args.stop_after_steps, args.eval_steps)
    if args.stop_after_steps is not None and args.resume is not None:
        raise ValueError("step-budget runs must start fresh, not resume")
    prepared_manifest_sha256 = None
    tokenizer_provenance = None
    if args.prepared_manifest:
        check_fresh(args.adapter_dir, args.resume)
        train_rows, val_rows = load_prepared(args.prepared_manifest, args.train, args.val,
                                           args.max_length, SYSTEM_PROMPT, PROMPT_VERSION)
        prepared_manifest_sha256 = sha256_file(args.prepared_manifest)
        tokenizer_provenance = json.loads(args.prepared_manifest.read_text()).get("tokenizer_provenance")
        for limit in (args.limit, args.val_limit):
            if limit is not None and limit <= 0:
                raise ValueError("row limits must be positive")
        if args.limit and args.limit < len(train_rows):
            train_rows = random.Random(args.seed).sample(train_rows, args.limit)
        if args.val_limit and args.val_limit < len(val_rows):
            val_rows = random.Random(args.seed).sample(val_rows, args.val_limit)
    else:
        train_rows = load_chat_rows(args.train, args.limit, seed=args.seed)
        val_rows = load_chat_rows(args.val, args.val_limit, seed=args.seed)

    # Import GPU libraries only after prepared-data preflight succeeds.
    import unsloth
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, TrainerCallback
    import peft
    import transformers
    import trl
    from trl import SFTConfig, SFTTrainer
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_smoke import generate, next_token_agreement, resolve_target_modules

    revision = args.model_revision or HfApi().model_info(args.model).sha
    print(f"model: {args.model}@{revision}")
    print(f"unsloth {unsloth.__version__} | trl {trl.__version__} | peft {peft.__version__} | "
          f"transformers {transformers.__version__} | torch {torch.__version__}")

    print(f"rows: {len(train_rows)} train, {len(val_rows)} validation")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        revision=revision,
        use_exact_model_name=True,
        dtype=torch.bfloat16,
        max_seq_length=args.max_length,
        load_in_4bit=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=resolve_target_modules(model) or LORA_TARGETS,
        use_gradient_checkpointing={"unsloth": "unsloth", "standard": True, "off": False}[args.gradient_checkpointing],
        random_state=args.seed,
    )

    def to_dataset(rows: list[dict]) -> Dataset:
        return Dataset.from_list([{k: row[k] for k in ("prompt", "completion")} for row in rows])

    eval_groups, source_mapping = source_groups(val_rows) if args.prepared_manifest else ({}, {})
    best_metric = "eval_all_loss" if args.prepared_manifest else "eval_loss"
    config = SFTConfig(
        output_dir=str(args.adapter_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=eval_batch_size,
        packing=False,
        padding_free=False,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=args.gradient_checkpointing != "off",
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        logging_steps=20,
        disable_tqdm=True,
        bf16=True,
        max_length=args.max_length,
        completion_only_loss=True,
        group_by_length=True,
        report_to=[],
        seed=args.seed,
        **schedule,
        save_total_limit=None if args.stop_after_steps is not None else 2,
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=False,
        restore_callback_states_from_checkpoint=True,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=to_dataset(train_rows),
        eval_dataset={key: to_dataset(rows) for key, rows in eval_groups.items()} if args.prepared_manifest else to_dataset(val_rows),
        args=config,
    )
    if args.prepared_manifest:
        audit_prepared_dataset(trainer.train_dataset, train_rows, tokenizer,
                               trainer.data_collator, batch_size=args.batch_size)
        for key, rows in eval_groups.items():
            audit_prepared_dataset(trainer.eval_dataset[key], rows, tokenizer,
                                   trainer.data_collator, batch_size=eval_batch_size)
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=args.patience))
    if args.stop_after_steps is not None:
        class StepBudgetCallback(TrainerCallback):
            def on_step_end(self, training_args, state, control, **kwargs):
                return apply_step_budget(control, state.global_step, args.stop_after_steps)

        trainer.add_callback(StepBudgetCallback())
    trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)

    history = trainer.state.log_history
    train_losses = [(e["step"], e["loss"]) for e in history if "loss" in e]
    eval_losses = [(e["epoch"], e[best_metric]) for e in history if best_metric in e]
    best_epoch, best_loss = min(eval_losses, key=lambda item: item[1]) if eval_losses else (None, None)
    if train_losses:
        print(f"train loss: {train_losses[0][1]:.4f} (step {train_losses[0][0]}) -> {train_losses[-1][1]:.4f} (step {train_losses[-1][0]})")
    for epoch, loss in eval_losses:
        print(f"eval loss @ epoch {epoch:.2f}: {loss:.4f}")
    if best_loss is not None:
        print(f"best eval loss: {best_loss:.4f} at epoch {best_epoch:.2f} (restored)")

    gradient_checkpointing_active = model.is_gradient_checkpointing
    FastLanguageModel.for_inference(model)
    samples = []
    print("\n=== validation generations (base vs adapter) ===")
    sample_rows = random.Random(args.seed).sample(val_rows, min(args.sample_count, len(val_rows)))
    for row in sample_rows:
        adapter_output = generate(model, tokenizer, row["prompt"])
        with model.disable_adapter():
            base_output = generate(model, tokenizer, row["prompt"])
        samples.append({"id": row["id"], "base": base_output, "adapter": adapter_output})
        if not args.prepared_manifest:
            print(f"\n[{row['id']}]\n  base:    {base_output[:400]}\n  adapter: {adapter_output[:400]}")

    args.adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.adapter_dir))
    tokenizer.save_pretrained(str(args.adapter_dir))

    base = AutoModelForCausalLM.from_pretrained(
        args.model, revision=revision, dtype=torch.bfloat16, device_map="cuda"
    )
    reloaded_tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    reloaded = peft.PeftModel.from_pretrained(base, str(args.adapter_dir))
    reloaded.eval()
    agreement, worst_flip = next_token_agreement(model, reloaded, reloaded_tokenizer, val_rows[0]["prompt"])
    print(f"\nreload next-token agreement: {agreement:.3f} (worst flip margin {worst_flip:.3f})")
    if agreement < 0.90 or worst_flip >= 1.0:
        raise RuntimeError(
            f"Reloaded adapter disagrees with the trained model "
            f"(agreement {agreement:.3f}, worst flip margin {worst_flip:.3f})"
        )
    print("prompt-prefix reload heuristic passed; this is not a deployment-equivalence check")

    # Record the saved targets rather than inferring them from requested names.
    saved_adapter_config = json.loads((args.adapter_dir / "adapter_config.json").read_text())
    applied_targets = saved_adapter_config.get("target_modules")

    metadata = {
        "model": args.model,
        "model_revision": revision,
        "data": {
            "prepared_manifest_sha256": prepared_manifest_sha256,
            "tokenizer_provenance": tokenizer_provenance,
            "train": str(args.train),
            "train_sha256": sha256_file(args.train),
            "train_rows": len(train_rows),
            "val": str(args.val),
            "val_sha256": sha256_file(args.val),
            "val_rows": len(val_rows),
            "train_ids": [row["id"] for row in train_rows],
            "val_ids": [row["id"] for row in val_rows],
        },
        "lora": {
            "r": 16,
            "alpha": 16,
            "dropout": 0.0,
            "requested_target_modules": LORA_TARGETS,
            "applied_target_modules": applied_targets,
        },
        "training": {
            "prompt_version": PROMPT_VERSION if args.prepared_manifest else None,
            "packing": trainer.args.packing,
            "padding_free": getattr(trainer.data_collator, "padding_free", trainer.args.padding_free),
            "per_device_eval_batch_size": trainer.args.per_device_eval_batch_size,
            "eval_strategy": trainer.args.eval_strategy,
            "save_strategy": trainer.args.save_strategy,
            "eval_steps": trainer.args.eval_steps,
            "save_steps": trainer.args.save_steps,
            "eval_steps_requested": args.eval_steps,
            "patience_unit": "evaluation_events",
            "metric_for_best_model": best_metric,
            "source_metric_mapping": source_mapping,
            "source_eval_losses": [{"epoch": e.get("epoch"), **{k: v for k, v in e.items()
                                      if k.startswith("eval_source_") and k.endswith("_loss")}}
                                   for e in history if any(k.startswith("eval_source_") and k.endswith("_loss") for k in e)],
            "stop_after_steps": args.stop_after_steps,
            "optimizer_steps_completed": trainer.state.global_step,
            "scheduler_horizon_steps": trainer.state.max_steps,
            "warmup_steps": trainer.args.get_warmup_steps(trainer.state.max_steps),
            "save_total_limit": trainer.args.save_total_limit,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "optimizer": trainer.args.optim,
            "weight_decay": trainer.args.weight_decay,
            "epochs_requested": args.epochs,
            "epochs_completed": history[-1].get("epoch") if history else None,
            "patience": args.patience,
            "per_device_train_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "gradient_checkpointing_requested": args.gradient_checkpointing,
            "gradient_checkpointing_configured": trainer.args.gradient_checkpointing,
            "gradient_checkpointing_active": gradient_checkpointing_active,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "seed": args.seed,
            "train_loss_first": train_losses[0][1] if train_losses else None,
            "train_loss_last": train_losses[-1][1] if train_losses else None,
            "eval_losses": [{"epoch": epoch, "loss": loss} for epoch, loss in eval_losses],
            "best_eval_loss": best_loss,
            "best_eval_epoch": best_epoch,
        },
        "heldout_generations": samples,
        "reload_next_token_agreement": agreement,
        "reload_worst_flip_margin": worst_flip,
        "versions": {
            "unsloth": unsloth.__version__,
            "trl": trl.__version__,
            "peft": peft.__version__,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
        },
    }
    metadata_path = args.adapter_dir / "train_meta.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"adapter: {args.adapter_dir}")
    print(f"metadata: {metadata_path}")
    print("TRAINING COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"TRAINING FAILED: {exc}", file=sys.stderr)
        raise
