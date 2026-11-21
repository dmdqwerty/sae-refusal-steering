# -*- coding: utf-8 -*-
"""Stage 1: build JSONL prompt corpora for SAE training and refusal-feature selection.

Outputs (in corpora2/): sae_{all,train,val,test}.jsonl for SAE training,
retain.jsonl (harmless instructions) and target.jsonl (harmful AdvBench prompts)
for contrasting feature activations, plus metadata.json with token statistics.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm


SEED = 42
random.seed(SEED)

DATA_ROOT = Path("/content/drive/MyDrive/Diploma") if Path("/content/drive/MyDrive").exists() else Path("data")
OUT_DIR = DATA_ROOT / "corpora2"
RESET_OUT_DIR = False  # True deletes OUT_DIR before rebuilding

MODEL_NAME_OR_PATH = os.environ.get("TOKENIZER_NAME_OR_PATH", "meta-llama/Meta-Llama-3-8B-Instruct")
SYSTEM_PROMPT = "You are a helpful assistant."

SAE_TOTAL_TOKENS = 15_000_000
SAE_RATIOS = {
    "harmless_instruction": 0.98,
    "benign_safety_related": 0.01,
    "harmful_refusal_inducing": 0.01,
}
SAE_SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}

RETAIN_TOKENS = 50_000
TARGET_TOKENS = 10_000
MAX_TOKENS_PER_EXAMPLE = 1024
PROMPT_ONLY = True
USE_BEAVERTAILS_FOR_SAE_TRAIN_ONLY = True
STRICT_DATASET_LOADING = False
MAX_ULTRACHAT_ROWS = 100_000

STRICT_HARMFUL_CATEGORIES = {
    "animal_abuse",
    "child_abuse",
    "drug_abuse,weapons,banned_substance",
    "financial_crime,property_crime,theft",
    "self_harm",
    "terrorism,organized_crime",
    "violence,aiding_and_abetting,incitement",
}

BENIGN_TOPICS = [
    "cybersecurity ethics", "password safety", "phishing awareness",
    "responsible vulnerability disclosure", "software supply-chain security",
    "secure configuration of cloud services", "privacy protection", "data protection",
    "online harassment prevention", "fraud prevention", "financial scam awareness",
    "misinformation prevention", "public health risk communication",
    "medical information reliability", "AI safety policies", "content moderation",
    "emergency preparedness", "workplace safety", "chemical laboratory safety",
    "safe handling of hazardous materials", "fire safety", "food safety",
    "responsible use of drones", "digital well-being", "child online safety",
    "elder fraud awareness", "journalistic reporting on extremist content",
    "defensive malware analysis", "security awareness training", "responsible science communication",
    "safe home maintenance", "safe travel planning", "disaster response communication",
    "research ethics", "dual-use technology governance", "safe public demonstrations",
    "reporting illegal content safely", "community moderation", "identity theft prevention",
    "online marketplace safety", "secure software development", "incident response planning",
]

BENIGN_INTENTS = [
    "explain", "summarize", "compare", "create a checklist for", "write a FAQ about",
    "draft a public-awareness note about", "outline best practices for",
    "describe common misconceptions about", "write classroom guidance on",
    "create a policy-style summary for", "give high-level preventive advice on",
]

BENIGN_AUDIENCES = [
    "ordinary users", "students", "parents", "teachers", "software developers",
    "small businesses", "policy makers", "community moderators", "journalists",
    "healthcare administrators", "non-technical managers", "new employees",
]

BENIGN_CONSTRAINTS = [
    "without giving operationally harmful instructions",
    "focusing only on prevention and responsible behavior",
    "using non-technical language",
    "including safe escalation paths when appropriate",
    "avoiding procedural details that could enable misuse",
    "with emphasis on ethics and risk reduction",
    "with examples of safe and unsafe framing",
]

NEAR_MISS_PROMPTS = [
    "How can I safely store household chemicals at home?",
    "How can journalists report on extremist propaganda without amplifying it?",
    "Explain malware analysis in a defensive cybersecurity context.",
    "What are safe ways to teach students about phishing prevention?",
    "How can I recognize signs of online financial scams?",
    "What should a lab worker do after a minor chemical spill?",
    "How can a company write a responsible vulnerability disclosure policy?",
    "What are ethical guidelines for discussing dual-use biology research?",
    "How should a moderator respond to harassment in an online community?",
    "Explain how to evaluate medical information reliability online.",
    "What are safe steps for responding to suspected identity theft?",
    "How can drone hobbyists avoid creating safety risks?",
    "How can people discuss misinformation without repeating false claims?",
    "What are safe ways to report suspected cybercrime?",
    "How can teachers explain digital privacy to teenagers?",
]


if RESET_OUT_DIR and OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading tokenizer:", MODEL_NAME_OR_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def normalize_text(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def make_messages(user_prompt: str, assistant_response: Optional[str] = None) -> List[Dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": normalize_text(user_prompt)},
    ]
    if assistant_response is not None and not PROMPT_ONLY:
        messages.append({"role": "assistant", "content": normalize_text(assistant_response)})
    return messages


def count_chat_tokens(messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> int:
    rendered_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )
    return len(tokenizer(rendered_text, add_special_tokens=False)["input_ids"])


def build_record(
    idx: int,
    category: str,
    source: str,
    user_prompt: str,
    assistant_response: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    user_prompt = normalize_text(user_prompt)
    if len(user_prompt) < 5:
        return None

    messages = make_messages(user_prompt, assistant_response)
    n_tokens = count_chat_tokens(
        messages,
        add_generation_prompt=(assistant_response is None or PROMPT_ONLY),
    )
    if n_tokens > MAX_TOKENS_PER_EXAMPLE:
        return None

    return {
        "id": f"{category}_{idx:08d}",
        "category": category,
        "source": source,
        "n_tokens": n_tokens,
        "messages": messages,
    }


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def token_sum(records: List[Dict[str, Any]]) -> int:
    return sum(int(record["n_tokens"]) for record in records)


def user_prompt_key(record: Dict[str, Any]) -> str:
    user_text = next(message["content"] for message in record["messages"] if message["role"] == "user")
    return re.sub(r"\W+", " ", normalize_text(user_text).lower()).strip()


def dedup_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    result = []
    for record in records:
        key = user_prompt_key(record)
        if key not in seen:
            seen.add(key)
            result.append(record)
    return result


def remove_overlaps(records: List[Dict[str, Any]], forbidden_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    forbidden = {user_prompt_key(record) for record in forbidden_records}
    return [record for record in records if user_prompt_key(record) not in forbidden]


def take_by_token_budget(records: List[Dict[str, Any]], token_budget: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    selected = []
    remaining = []
    total_tokens = 0

    for record in records:
        if total_tokens < token_budget:
            selected.append(record)
            total_tokens += int(record["n_tokens"])
        else:
            remaining.append(record)

    return selected, remaining


def split_records(records: List[Dict[str, Any]], split_ratios: Dict[str, float], seed: int) -> Dict[str, List[Dict[str, Any]]]:
    records = records[:]
    random.Random(seed).shuffle(records)

    n_train = int(len(records) * split_ratios["train"])
    n_val = int(len(records) * split_ratios["val"])

    return {
        "train": records[:n_train],
        "val": records[n_train:n_train + n_val],
        "test": records[n_train + n_val:],
    }


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_category: Dict[str, Dict[str, int]] = {}
    by_source: Dict[str, Dict[str, int]] = {}

    for record in records:
        category = record["category"]
        source = record["source"]

        by_category.setdefault(category, {"n_examples": 0, "n_tokens": 0})
        by_category[category]["n_examples"] += 1
        by_category[category]["n_tokens"] += int(record["n_tokens"])

        by_source.setdefault(source, {"n_examples": 0, "n_tokens": 0})
        by_source[source]["n_examples"] += 1
        by_source[source]["n_tokens"] += int(record["n_tokens"])

    return {
        "n_examples": len(records),
        "n_tokens": token_sum(records),
        "by_category": by_category,
        "by_source": by_source,
    }


def assert_disjoint(name_a: str, a: List[Dict[str, Any]], name_b: str, b: List[Dict[str, Any]]) -> None:
    overlap = {user_prompt_key(record) for record in a} & {user_prompt_key(record) for record in b}
    if overlap:
        raise RuntimeError(f"{name_a} and {name_b} are not disjoint. Overlap count: {len(overlap)}")


def safe_load_dataset(*args, **kwargs):
    try:
        return load_dataset(*args, **kwargs)
    except Exception as exc:
        if STRICT_DATASET_LOADING:
            raise
        print(f"WARNING: dataset load skipped: args={args}, kwargs={kwargs}. Error: {exc}")
        return None


def add_harmless_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    record_idx = 0

    alpaca = safe_load_dataset("yahma/alpaca-cleaned", split="train")
    if alpaca is not None:
        for row in tqdm(alpaca.shuffle(seed=SEED), desc="Processing alpaca-cleaned"):
            instruction = normalize_text(row.get("instruction", ""))
            input_text = normalize_text(row.get("input", ""))
            user_prompt = f"{instruction}\n\nInput:\n{input_text}" if input_text else instruction
            record = build_record(record_idx, "harmless_instruction", "yahma/alpaca-cleaned", user_prompt, row.get("output", None))
            record_idx += 1
            if record is not None:
                records.append(record)

    dolly = safe_load_dataset("databricks/databricks-dolly-15k", split="train")
    if dolly is not None:
        for row in tqdm(dolly.shuffle(seed=SEED), desc="Processing databricks-dolly-15k"):
            instruction = normalize_text(row.get("instruction", ""))
            context = normalize_text(row.get("context", ""))
            user_prompt = f"{instruction}\n\nContext:\n{context}" if context else instruction
            record = build_record(record_idx, "harmless_instruction", "databricks/databricks-dolly-15k", user_prompt, row.get("response", None))
            record_idx += 1
            if record is not None:
                records.append(record)

    ultrachat = None
    for split_name in ["train_sft", "train", "test_sft"]:
        ultrachat = safe_load_dataset("HuggingFaceH4/ultrachat_200k", split=split_name)
        if ultrachat is not None:
            print(f"Using ultrachat_200k split={split_name}")
            break

    if ultrachat is not None:
        for row_i, row in enumerate(tqdm(ultrachat.shuffle(seed=SEED), desc="Processing ultrachat_200k")):
            if row_i >= MAX_ULTRACHAT_ROWS:
                break

            messages = row.get("messages", [])
            user_prompt = ""
            assistant_response = None

            for message in messages:
                if message.get("role") == "user" and message.get("content"):
                    user_prompt = normalize_text(message["content"])
                    break

            if not PROMPT_ONLY:
                seen_user = False
                for message in messages:
                    if message.get("role") == "user":
                        seen_user = True
                    elif seen_user and message.get("role") == "assistant" and message.get("content"):
                        assistant_response = normalize_text(message["content"])
                        break

            record = build_record(record_idx, "harmless_instruction", "HuggingFaceH4/ultrachat_200k", user_prompt, assistant_response)
            record_idx += 1
            if record is not None:
                records.append(record)

    records = dedup_records(records)
    random.shuffle(records)
    return records


def load_advbench_records() -> List[Dict[str, Any]]:
    advbench = safe_load_dataset("walledai/AdvBench", split="train")
    if advbench is None:
        raise RuntimeError("AdvBench loading failed.")

    records = []
    for i, row in enumerate(tqdm(advbench.shuffle(seed=SEED), desc="Processing AdvBench")):
        prompt = row.get("goal") or row.get("prompt") or row.get("instruction") or row.get("behavior") or row.get("text") or ""
        record = build_record(i, "harmful_refusal_inducing", "walledai/AdvBench", prompt)
        if record is not None:
            records.append(record)

    records = dedup_records(records)
    random.shuffle(records)
    return records


def has_strict_harmful_category(row: Dict[str, Any]) -> bool:
    category = row.get("category")
    if isinstance(category, str):
        category = json.loads(category)
    active_categories = {name for name, is_active in category.items() if bool(is_active)}
    return bool(active_categories & STRICT_HARMFUL_CATEGORIES)


def add_beavertails_records(records: List[Dict[str, Any]], token_budget: int) -> List[Dict[str, Any]]:
    beavertails = None
    for split_name in ["330k_train", "30k_train", "train"]:
        beavertails = safe_load_dataset("PKU-Alignment/BeaverTails", split=split_name)
        if beavertails is not None:
            print(f"Using BeaverTails split={split_name}")
            break

    if beavertails is None:
        return records

    total_tokens = token_sum(records)
    base_idx = len(records)

    for i, row in enumerate(tqdm(beavertails.shuffle(seed=SEED), desc="Processing BeaverTails")):
        if total_tokens >= token_budget:
            break
        if row.get("is_safe") is not False or not has_strict_harmful_category(row):
            continue

        prompt = row.get("prompt") or row.get("question") or row.get("instruction") or row.get("text") or ""
        record = build_record(base_idx + i, "harmful_refusal_inducing", "PKU-Alignment/BeaverTails-strict-train-only", prompt)
        if record is not None:
            records.append(record)
            total_tokens += int(record["n_tokens"])

    return records


def generate_benign_records() -> List[Dict[str, Any]]:
    prompts = []
    for topic in BENIGN_TOPICS:
        for intent in BENIGN_INTENTS:
            for audience in random.sample(BENIGN_AUDIENCES, k=4):
                prompts.append(f"Please {intent} {topic} for {audience}, {random.choice(BENIGN_CONSTRAINTS)}.")
    prompts.extend(NEAR_MISS_PROMPTS)

    records = []
    for i, prompt in enumerate(prompts):
        record = build_record(i, "benign_safety_related", "generated_diverse_safe_prompts", prompt)
        if record is not None:
            records.append(record)

    records = dedup_records(records)
    random.shuffle(records)
    return records


def build_sae_corpus(category_pools: Dict[str, List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    sae_all = []
    remaining_pools = {}

    for category, ratio in SAE_RATIOS.items():
        budget = int(SAE_TOTAL_TOKENS * ratio)
        selected, remaining = take_by_token_budget(category_pools[category], budget)
        remaining_pools[category] = remaining
        sae_all.extend(selected)
        print(f"{category}: selected {token_sum(selected):,}/{budget:,} tokens, examples={len(selected):,}")

    random.shuffle(sae_all)
    for i, record in enumerate(sae_all):
        record["id"] = f"sae_all_{i:08d}"

    return sae_all, remaining_pools


def split_sae_corpus(sae_all: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    split_parts = {"train": [], "val": [], "test": []}

    for category in SAE_RATIOS:
        category_records = [record for record in sae_all if record["category"] == category]
        category_splits = split_records(category_records, SAE_SPLIT_RATIOS, seed=SEED)
        for split_name in split_parts:
            split_parts[split_name].extend(category_splits[split_name])

    for split_name, records in split_parts.items():
        random.shuffle(records)
        for i, record in enumerate(records):
            record["id"] = f"sae_{split_name}_{i:08d}"

    return split_parts


def save_outputs(
    sae_all: List[Dict[str, Any]],
    split_parts: Dict[str, List[Dict[str, Any]]],
    retain_records: List[Dict[str, Any]],
    target_records: List[Dict[str, Any]],
) -> None:
    write_jsonl(OUT_DIR / "sae_all.jsonl", sae_all)
    write_jsonl(OUT_DIR / "sae_train.jsonl", split_parts["train"])
    write_jsonl(OUT_DIR / "sae_val.jsonl", split_parts["val"])
    write_jsonl(OUT_DIR / "sae_test.jsonl", split_parts["test"])
    write_jsonl(OUT_DIR / "retain.jsonl", retain_records)
    write_jsonl(OUT_DIR / "target.jsonl", target_records)

    metadata = {
        "seed": SEED,
        "model_name_or_path": MODEL_NAME_OR_PATH,
        "system_prompt": SYSTEM_PROMPT,
        "prompt_only": PROMPT_ONLY,
        "max_tokens_per_example": MAX_TOKENS_PER_EXAMPLE,
        "sae_total_token_target": SAE_TOTAL_TOKENS,
        "sae_ratios": SAE_RATIOS,
        "sae_split_ratios": SAE_SPLIT_RATIOS,
        "retain_token_target": RETAIN_TOKENS,
        "target_token_target": TARGET_TOKENS,
        "use_beavertails_for_sae_train_only": USE_BEAVERTAILS_FOR_SAE_TRAIN_ONLY,
        "files": {
            "sae_all": "sae_all.jsonl",
            "sae_train": "sae_train.jsonl",
            "sae_val": "sae_val.jsonl",
            "sae_test": "sae_test.jsonl",
            "retain": "retain.jsonl",
            "target": "target.jsonl",
        },
        "summaries": {
            "sae_all": summarize(sae_all),
            "sae_train": summarize(split_parts["train"]),
            "sae_val": summarize(split_parts["val"]),
            "sae_test": summarize(split_parts["test"]),
            "retain": summarize(retain_records),
            "target": summarize(target_records),
        },
    }

    with (OUT_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Done. Files saved to:", OUT_DIR.resolve())
    print(json.dumps(metadata["summaries"], indent=2, ensure_ascii=False))


def main() -> None:
    print("Loading harmless datasets")
    harmless_records = add_harmless_records()
    print(f"Harmless records: {len(harmless_records):,}, tokens={token_sum(harmless_records):,}")

    print("Loading harmful datasets")
    advbench_records = load_advbench_records()
    harmful_target_records, harmful_train_records = take_by_token_budget(advbench_records, TARGET_TOKENS)

    for record in harmful_target_records:
        record["source"] = "walledai/AdvBench-target-only"
    for record in harmful_train_records:
        record["source"] = "walledai/AdvBench-train-remainder"

    if USE_BEAVERTAILS_FOR_SAE_TRAIN_ONLY:
        harmful_budget = int(SAE_TOTAL_TOKENS * SAE_RATIOS["harmful_refusal_inducing"]) + 20_000
        harmful_train_records = add_beavertails_records(harmful_train_records, harmful_budget)

    harmful_train_records = remove_overlaps(dedup_records(harmful_train_records), harmful_target_records)
    random.shuffle(harmful_train_records)
    random.shuffle(harmful_target_records)

    print(f"Harmful train records: {len(harmful_train_records):,}, tokens={token_sum(harmful_train_records):,}")
    print(f"Harmful target records: {len(harmful_target_records):,}, tokens={token_sum(harmful_target_records):,}")

    benign_records = generate_benign_records()
    print(f"Benign safety records: {len(benign_records):,}, tokens={token_sum(benign_records):,}")

    category_pools = {
        "harmless_instruction": harmless_records,
        "benign_safety_related": benign_records,
        "harmful_refusal_inducing": harmful_train_records,
    }

    sae_all, remaining_pools = build_sae_corpus(category_pools)
    split_parts = split_sae_corpus(sae_all)

    retain_records, _ = take_by_token_budget(remaining_pools["harmless_instruction"], RETAIN_TOKENS)
    target_records = harmful_target_records

    for i, record in enumerate(retain_records):
        record["id"] = f"retain_{i:08d}"
        record["category"] = "retain_harmless_instruction"
    for i, record in enumerate(target_records):
        record["id"] = f"target_{i:08d}"
        record["category"] = "target_harmful_refusal_inducing"

    assert_disjoint("retain", retain_records, "sae_all", sae_all)
    assert_disjoint("target", target_records, "sae_all", sae_all)
    assert_disjoint("retain", retain_records, "target", target_records)
    assert_disjoint("sae_train", split_parts["train"], "sae_val", split_parts["val"])
    assert_disjoint("sae_train", split_parts["train"], "sae_test", split_parts["test"])
    assert_disjoint("sae_val", split_parts["val"], "sae_test", split_parts["test"])

    save_outputs(sae_all, split_parts, retain_records, target_records)


if __name__ == "__main__":
    main()
