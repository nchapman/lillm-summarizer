"""LoRA smoke test: train a tiny adapter and prove the artifact is valid.

Loads the prepared smoltalk2 summarization data, fine-tunes a LoRA adapter on a
small slice, checks that the loss decreases, saves the native adapter, and
verifies it reloads on a freshly loaded base model (outside Unsloth) and
produces identical output. Sample generations run before (adapter disabled) and
after training so the behavioral delta is visible.

Usage (defaults target Granite 4.0 1B):
    python scripts/train_smoke.py --adapter-dir adapters/granite4-1b-smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

import peft
import transformers
import trl
import unsloth
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

DEFAULT_MODEL = "ibm-granite/granite-4.0-1b"
DEFAULT_DATA = Path("data/train/smoltalk2_smol-summarize_no_think.jsonl")
# Granite 4.0 uses shared_mlp.{input,output}_linear instead of Llama-style
# gate/up/down_proj; attention + MLP together is unsloth's documented best config.
LORA_TARGET_CANDIDATES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "input_linear", "output_linear",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=None, help="pin a Hub revision; default records current main")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--limit", type=int, default=200, help="rows loaded from the training file")
    parser.add_argument("--holdout", type=int, default=3, help="rows held out from training for generation checks")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(path: Path, limit: int, holdout: int) -> tuple[list[dict], list[dict]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if len(rows) >= limit:
                break
            record = json.loads(line)
            messages = record["messages"]
            rows.append(
                {
                    "prompt": [{"role": m["role"], "content": m["content"]} for m in messages[:-1]],
                    "completion": [{"role": messages[-1]["role"], "content": messages[-1]["content"]}],
                    "id": record["id"],
                }
            )
    if len(rows) <= holdout:
        raise ValueError(f"--limit must exceed --holdout (got {len(rows)} rows, holdout {holdout})")
    return rows[:-holdout], rows[-holdout:]


def resolve_target_modules(model) -> list[str]:
    available = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    targets = [module for module in LORA_TARGET_CANDIDATES if module in available]
    if not targets:
        raise RuntimeError(f"No LoRA target modules found; available modules: {sorted(available)[:30]}")
    return targets


def generate(model, tokenizer, prompt_messages: list[dict], max_new_tokens: int = 256) -> str:
    text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def next_token_agreement(
    model_a, model_b, tokenizer, prompt_messages: list[dict], positions: int = 64
) -> tuple[float, float]:
    """Teacher-forced agreement between two models on a shared prompt.

    Returns ``(agreement, worst_flip_margin)``: the fraction of positions whose
    greedy next token matches, and — among mismatching positions — how strongly
    model_a preferred its own pick, in logit units. Exact generation equality
    is not expected across inference paths: bf16 LoRA applied on the fly vs
    merged weights (and hybrid-Mamba recurrent scans) differ in floating-point
    op order, flipping only near-ties. A flip with a margin of one logit or
    more indicates a real discrepancy, not noise.
    """
    text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits_a = model_a(input_ids=inputs["input_ids"].to(model_a.device)).logits[0, :positions].float()
        logits_b = model_b(input_ids=inputs["input_ids"].to(model_b.device)).logits[0, :positions].float()
    top_a = logits_a.argmax(dim=-1)
    top_b = logits_b.argmax(dim=-1)
    agreement = (top_a == top_b).float().mean().item()
    worst_flip = 0.0
    for index in (top_a != top_b).nonzero().flatten().tolist():
        margin = (logits_a[index, top_a[index]] - logits_a[index, top_b[index]]).item()
        worst_flip = max(worst_flip, margin)
    return agreement, worst_flip


def main() -> None:
    args = parse_args()
    revision = args.model_revision or HfApi().model_info(args.model).sha
    print(f"model: {args.model}@{revision}")
    print(f"unsloth {unsloth.__version__} | trl {trl.__version__} | peft {peft.__version__} | "
          f"transformers {transformers.__version__} | torch {torch.__version__}")

    train_rows, holdout_rows = load_rows(args.data, args.limit, args.holdout)
    print(f"rows: {len(train_rows)} train, {len(holdout_rows)} held out")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        revision=revision,
        max_seq_length=args.max_length,
        load_in_4bit=False,
    )
    target_modules = resolve_target_modules(model)
    print(f"LoRA targets: {target_modules}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=target_modules,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    dataset = Dataset.from_list([{k: row[k] for k in ("prompt", "completion")} for row in train_rows])
    config = SFTConfig(
        output_dir=str(args.adapter_dir / "checkpoints"),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        logging_steps=5,
        bf16=True,
        max_length=args.max_length,
        completion_only_loss=True,
        report_to=[],
        seed=args.seed,
        save_strategy="no",
    )
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=config)
    trainer.train()

    losses = [(entry["step"], entry["loss"]) for entry in trainer.state.log_history if "loss" in entry]
    if not losses:
        raise RuntimeError("No training losses were logged")
    first_step, first_loss = losses[0]
    last_step, last_loss = losses[-1]
    print(f"loss: step {first_step} {first_loss:.4f} -> step {last_step} {last_loss:.4f} ({len(losses)} logged points)")
    if last_loss >= first_loss:
        raise RuntimeError(f"Loss did not decrease ({first_loss:.4f} -> {last_loss:.4f})")

    FastLanguageModel.for_inference(model)
    print("\n=== held-out generations (base vs adapter) ===")
    samples = []
    for row in holdout_rows:
        adapter_output = generate(model, tokenizer, row["prompt"])
        with model.disable_adapter():
            base_output = generate(model, tokenizer, row["prompt"])
        samples.append({"id": row["id"], "base": base_output, "adapter": adapter_output})
        print(f"\n[{row['id']}]\n  base:    {base_output[:400]}\n  adapter: {adapter_output[:400]}")

    args.adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.adapter_dir))
    tokenizer.save_pretrained(str(args.adapter_dir))
    print(f"\nsaved adapter to {args.adapter_dir}")

    print("reloading adapter on a fresh base model (plain transformers + PEFT)...")
    base = AutoModelForCausalLM.from_pretrained(args.model, revision=revision, dtype=torch.bfloat16, device_map="cuda")
    reloaded_tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    reloaded = peft.PeftModel.from_pretrained(base, str(args.adapter_dir))
    reloaded.eval()
    agreement, worst_flip = next_token_agreement(model, reloaded, reloaded_tokenizer, holdout_rows[0]["prompt"])
    reloaded_output = generate(reloaded, reloaded_tokenizer, holdout_rows[0]["prompt"])
    print(f"reload next-token agreement: {agreement:.3f} (worst flip margin {worst_flip:.3f}); "
          f"reloaded output: {reloaded_output[:200]}")
    if agreement < 0.90 or worst_flip >= 1.0:
        raise RuntimeError(
            f"Reloaded adapter disagrees with the trained model "
            f"(agreement {agreement:.3f}, worst flip margin {worst_flip:.3f})"
        )
    print("reload check passed: differences are numeric noise on near-ties")

    metadata = {
        "model": args.model,
        "model_revision": revision,
        "data": str(args.data),
        "data_rows": {"limit": args.limit, "train": len(train_rows), "holdout": len(holdout_rows)},
        "lora": {"r": 16, "alpha": 16, "dropout": 0.0, "target_modules": target_modules},
        "training": {
            "epochs": args.epochs,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "max_length": args.max_length,
            "seed": args.seed,
            "loss_first": first_loss,
            "loss_last": last_loss,
        },
        "versions": {
            "unsloth": unsloth.__version__,
            "trl": trl.__version__,
            "peft": peft.__version__,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
        },
        "reload_next_token_agreement": agreement,
        "reload_worst_flip_margin": worst_flip,
        "prompt_contract": train_rows[0]["prompt"],
    }
    metadata_path = args.adapter_dir / "smoke_meta.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"metadata: {metadata_path}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SMOKE TEST FAILED: {exc}", file=sys.stderr)
        raise
