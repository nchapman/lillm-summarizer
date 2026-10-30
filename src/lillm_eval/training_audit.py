"""Exact, privacy-safe checks of prepared single-turn completion supervision.

These checks validate Granite's native EOS/newline suffix without changing the
training objective. They deliberately do not infer correctness from length or
from the presence of an arbitrary supervised EOS elsewhere in the conversation.
"""
from __future__ import annotations


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_completion_sequence(input_ids, labels, expected_ids, prompt_ids, eos_suffix):
    """Validate against independently rendered full chat and generation prefix."""
    require(bool(prompt_ids) and bool(eos_suffix), "prepared completion boundary is empty")
    boundary = len(prompt_ids)
    require(expected_ids[:boundary] == prompt_ids, "prepared generation prefix mismatch")
    require(len(expected_ids) > boundary + len(eos_suffix), "prepared completion has no content")
    require(expected_ids[-len(eos_suffix):] == eos_suffix, "prepared terminal suffix mismatch")
    require(expected_ids[boundary:].count(eos_suffix[0]) == 1, "prepared completion must have one terminal EOS")
    require(input_ids == expected_ids, "prepared token sequence mismatch")
    expected_labels = [-100] * boundary + expected_ids[boundary:]
    require(labels == expected_labels, "prepared completion labels mismatch")
    return len(expected_ids) - boundary


def as_list(value):
    return value.tolist() if hasattr(value, "tolist") else value


def check_collated_batch(batch, features):
    """Check real multi-row padding or flattened labels, not only singleton batches."""
    ids, labels = as_list(batch["input_ids"]), as_list(batch["labels"])
    require(len(ids) == len(labels), "prepared collated batch size mismatch")
    if "attention_mask" in batch:
        masks = as_list(batch["attention_mask"])
        require(len(ids) == len(features) == len(masks), "prepared padded batch size mismatch")
        for tokens, targets, mask, feature in zip(ids, labels, masks, features):
            require(len(tokens) == len(targets) == len(mask), "prepared padded sequence length mismatch")
            require(all(bit in (0, 1) for bit in mask), "prepared attention mask is not binary")
            require([t for t, bit in zip(tokens, mask) if bit] == feature["input_ids"],
                    "prepared collator changed active tokens")
            require([t for t, bit in zip(targets, mask) if bit] == feature["labels"],
                    "prepared collator changed active labels")
            require(all(t == -100 for t, bit in zip(targets, mask) if not bit),
                    "prepared padding contributes to loss")
    else:
        require("position_ids" in batch, "prepared flattened batch lacks positions")
        require(ids == [[t for feature in features for t in feature["input_ids"]]],
                "prepared collator changed flattened tokens")
        require(labels == [[t for feature in features for t in feature["labels"]]],
                "prepared collator changed flattened labels")
        require(as_list(batch["position_ids"]) == [[i for feature in features for i in range(len(feature["input_ids"]))]],
                "prepared flattened positions cross example boundaries")


def audit_prepared_dataset(dataset, rows, tokenizer, collator, batch_size=4):
    """Check every row with bounded tokenization batches before any training step.

    ``rows`` contain the original conversational prompt/completion and token_count.
    ``dataset`` and ``collator`` must be the actual objects returned by SFTTrainer.
    Validation messages raised here omit source text, identifiers, token IDs and
    hashes. Exceptions originating in tokenizer/collator dependencies propagate.
    """
    require(type(batch_size) is int and batch_size > 0, "prepared audit batch size must be positive")
    require(len(dataset) == len(rows) and len(rows) > 0, "prepared dataset row count mismatch")
    require(tokenizer.pad_token_id != tokenizer.eos_token_id, "prepared padding aliases EOS")
    suffix = tokenizer(tokenizer.eos_token + "\n", add_special_tokens=False)["input_ids"]
    require(bool(suffix) and suffix[0] == tokenizer.eos_token_id, "prepared native EOS suffix mismatch")
    supervised = 0
    # Keep tokenization bounded while exercising the requested collator size,
    # including sizes that exceed or do not divide 32.
    chunk_size = max(batch_size, (32 // batch_size) * batch_size)
    for start in range(0, len(rows), chunk_size):
        originals = rows[start:start + chunk_size]
        expected = tokenizer.apply_chat_template(
            [row["prompt"] + row["completion"] for row in originals],
            tokenize=True, add_generation_prompt=False, truncation=False,
        )
        prefixes = tokenizer.apply_chat_template(
            [row["prompt"] for row in originals],
            tokenize=True, add_generation_prompt=True, truncation=False,
        )
        require(len(expected) == len(prefixes) == len(originals), "prepared tokenizer batch size mismatch")
        features = [dataset[start + i] for i in range(len(originals))]
        for feature, original, tokens, prefix in zip(features, originals, expected, prefixes):
            require(len(tokens) == original["token_count"], "prepared original token count mismatch")
            supervised += check_completion_sequence(feature["input_ids"], feature["labels"], tokens, prefix, suffix)
        for offset in range(0, len(features), batch_size):
            group = features[offset:offset + batch_size]
            # A collator may mutate feature dictionaries. Its inputs must not
            # also be the independent reference used to validate its output.
            reference = [{key: feature[key].copy() for key in ("input_ids", "labels")} for feature in group]
            check_collated_batch(collator(group), reference)
    return {"rows": len(rows), "supervised_tokens": supervised}
