"""Full LoRA training run on the stratified summarization subset.

Trains with per-epoch validation loss, early stopping, and best-checkpoint
restoration; saves the native adapter plus a metadata record; and repeats the
smoke-test artifact checks (held-out generations and a fresh-base reload).

Usage (Granite 4.0 1B defaults):
    python scripts/train_lora.py --adapter-dir adapters/granite4-1b-24k-s42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback

import peft
import transformers
import trl
import unsloth
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_smoke import generate, next_token_agreement, resolve_target_modules  # noqa: E402

DEFAULT_MODEL = "ibm-granite/granite-4.0-1b"
DEFAULT_TRAIN = Path("data/train/smoltalk2_stratified_train.jsonl")
DEFAULT_VAL = Path("data/train/smoltalk2_stratified_val.jsonl")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=1, help="early-stopping patience in epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--limit", type=int, help="cap training rows (smoke only)")
    parser.add_argument("--val-limit", type=int, default=200, help="validation rows used for eval loss")
    parser.add_argument("--sample-count", type=int, default=3, help="held-out generations shown")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_chat_rows(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if limit is not None and len(rows) >= limit:
                break
            record = json.loads(line)
            messages = record["messages"]
            rows.append(
                {
                    "prompt": [{"role": m["role"], "content": m["content"]} for m in messages[:-1]],
                    "completion": [{"role": "assistant", "content": messages[-1]["content"]}],
                    "id": record["id"],
                }
            )
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    revision = args.model_revision or HfApi().model_info(args.model).sha
    print(f"model: {args.model}@{revision}")
    print(f"unsloth {unsloth.__version__} | trl {trl.__version__} | peft {peft.__version__} | "
          f"transformers {transformers.__version__} | torch {torch.__version__}")

    train_rows = load_chat_rows(args.train, args.limit)
    val_rows = load_chat_rows(args.val, args.val_limit)
    print(f"rows: {len(train_rows)} train, {len(val_rows)} validation (of {args.val_limit} capped)")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        revision=revision,
        max_seq_length=args.max_length,
        load_in_4bit=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=resolve_target_modules(model) or LORA_TARGETS,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    def to_dataset(rows: list[dict]) -> Dataset:
        return Dataset.from_list([{k: row[k] for k in ("prompt", "completion")} for row in rows])

    config = SFTConfig(
        output_dir=str(args.adapter_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
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
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=to_dataset(train_rows),
        eval_dataset=to_dataset(val_rows),
        args=config,
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=args.patience))
    trainer.train()

    history = trainer.state.log_history
    train_losses = [(e["step"], e["loss"]) for e in history if "loss" in e]
    eval_losses = [(e["epoch"], e["eval_loss"]) for e in history if "eval_loss" in e]
    best_epoch, best_loss = min(eval_losses, key=lambda item: item[1]) if eval_losses else (None, None)
    print(f"train loss: {train_losses[0][1]:.4f} (step {train_losses[0][0]}) -> {train_losses[-1][1]:.4f} (step {train_losses[-1][0]})")
    for epoch, loss in eval_losses:
        print(f"eval loss @ epoch {epoch:.2f}: {loss:.4f}")
    print(f"best eval loss: {best_loss:.4f} at epoch {best_epoch:.2f} (restored)")

    FastLanguageModel.for_inference(model)
    samples = []
    print("\n=== validation generations (base vs adapter) ===")
    for row in val_rows[: args.sample_count]:
        adapter_output = generate(model, tokenizer, row["prompt"])
        with model.disable_adapter():
            base_output = generate(model, tokenizer, row["prompt"])
        samples.append({"id": row["id"], "base": base_output, "adapter": adapter_output})
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
    print("reload check passed: differences are numeric noise on near-ties")

    # Unsloth may rewrite LoRA targets for MoE/hybrid architectures (e.g.
    # attention-only, experts excluded); record what was actually saved.
    saved_adapter_config = json.loads((args.adapter_dir / "adapter_config.json").read_text())
    applied_targets = saved_adapter_config.get("target_modules")

    metadata = {
        "model": args.model,
        "model_revision": revision,
        "data": {
            "train": str(args.train),
            "train_sha256": sha256_file(args.train),
            "train_rows": len(train_rows),
            "val": str(args.val),
            "val_sha256": sha256_file(args.val),
            "val_rows": len(val_rows),
        },
        "lora": {
            "r": 16,
            "alpha": 16,
            "dropout": 0.0,
            "requested_target_modules": LORA_TARGETS,
            "applied_target_modules": applied_targets,
        },
        "training": {
            "epochs_requested": args.epochs,
            "epochs_completed": history[-1].get("epoch") if history else None,
            "patience": args.patience,
            "per_device_train_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "seed": args.seed,
            "train_loss_first": train_losses[0][1],
            "train_loss_last": train_losses[-1][1],
            "eval_losses": [{"epoch": epoch, "loss": loss} for epoch, loss in eval_losses],
            "best_eval_loss": best_loss,
            "best_eval_epoch": best_epoch,
        },
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
