"""Stage 2: extract layer-12 residual-stream activations of Llama 3 8B Instruct.

Reads corpora2/sae_{train,val,test}.jsonl (stage 1) and writes fp16 activation
shards of ~65k tokens each to activation_datasets2/{split}/. Requires a GPU with
>=24 GB memory and access to the gated meta-llama model on Hugging Face.
"""

from __future__ import annotations

import gc
import json
import math
import random
import shutil
from pathlib import Path
from typing import Dict, Iterator, List

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from google.colab import drive
    IN_COLAB = True
except Exception:
    IN_COLAB = False

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

if IN_COLAB:
    drive.mount("/content/drive")

DATA_ROOT = Path("/content/drive/MyDrive/Diploma") if IN_COLAB else Path("data")
TEXT_CORPUS_DIR = DATA_ROOT / "corpora2"
ACTIVATION_OUT_DIR = DATA_ROOT / "activation_datasets2"
RESET_ACTIVATION_OUT_DIR = False  # True deletes previously extracted shards

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
D_IN = 4096
EXP_FACTOR = 4
D_SAE = D_IN * EXP_FACTOR
K = 32
HOOK_LAYER = 12

BATCH_SIZE = 4
MAX_LENGTH = 1024
TOKENS_PER_SHARD = 65_536
SPLITS_TO_BUILD = ["train", "val", "test"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
SAVE_DTYPE = torch.float16

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def prepare_output_dir() -> None:
    if RESET_ACTIVATION_OUT_DIR and ACTIVATION_OUT_DIR.exists():
        shutil.rmtree(ACTIVATION_OUT_DIR)
    ACTIVATION_OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=MODEL_DTYPE,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    model.config.use_cache = False

    return model, tokenizer


def read_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def batch_iter(items: List[Dict], batch_size: int) -> Iterator[List[Dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def jsonl_path_for_split(split_name: str) -> Path:
    if split_name in {"train", "val", "test"}:
        return TEXT_CORPUS_DIR / f"sae_{split_name}.jsonl"
    return TEXT_CORPUS_DIR / f"{split_name}.jsonl"


def render_messages_to_text(record: Dict, tokenizer: AutoTokenizer) -> str:
    return tokenizer.apply_chat_template(
        record["messages"],
        tokenize=False,
        add_generation_prompt=True,
    )


def tokenize_batch(records: List[Dict], tokenizer: AutoTokenizer, input_device: torch.device) -> Dict[str, torch.Tensor]:
    texts = [render_messages_to_text(record, tokenizer) for record in records]
    batch = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=False,
    )
    return {
        "input_ids": batch["input_ids"].to(input_device),
        "attention_mask": batch["attention_mask"].to(input_device),
    }


class StopForwardAfterHook(Exception):
    pass


class LayerActivationExtractor:
    def __init__(self, model: AutoModelForCausalLM, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.hidden = None

    def _hook(self, module, inputs, output) -> None:
        self.hidden = output[0]
        raise StopForwardAfterHook()

    @torch.inference_mode()
    def extract_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self.hidden = None
        hook_handle = self.model.model.layers[self.layer_idx].register_forward_hook(self._hook)

        try:
            try:
                self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            except StopForwardAfterHook:
                pass

            hidden = self.hidden
            token_mask = attention_mask.to(device=hidden.device).bool()
            acts = hidden[token_mask].detach().to("cpu", dtype=SAVE_DTYPE)
            return acts

        finally:
            hook_handle.remove()
            self.hidden = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class ActivationShardWriter:
    def __init__(self, out_dir: Path, split_name: str, tokens_per_shard: int):
        self.out_dir = out_dir / split_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.split_name = split_name
        self.tokens_per_shard = tokens_per_shard
        self.buffer = []
        self.buffer_tokens = 0
        self.shard_idx = 0
        self.total_tokens = 0
        self.total_shards = 0

    def add(self, acts: torch.Tensor) -> None:
        if acts.numel() == 0:
            return
        self.buffer.append(acts)
        self.buffer_tokens += acts.shape[0]
        self.total_tokens += acts.shape[0]
        if self.buffer_tokens >= self.tokens_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return

        shard = torch.cat(self.buffer, dim=0)
        path = self.out_dir / f"{self.split_name}_acts_{self.shard_idx:05d}.pt"
        tmp_path = self.out_dir / f"{self.split_name}_acts_{self.shard_idx:05d}.tmp"

        try:
            torch.save(
                {
                    "activations": shard,
                    "split": self.split_name,
                    "layer_idx": HOOK_LAYER,
                    "d_model": shard.shape[1],
                    "dtype": str(shard.dtype),
                    "n_tokens": shard.shape[0],
                    "source_model": MODEL_NAME,
                },
                tmp_path,
            )
            tmp_path.replace(path)
            print(f"Saved {path.name}: {tuple(shard.shape)}")

            self.shard_idx += 1
            self.total_shards += 1
            self.buffer = []
            self.buffer_tokens = 0

        finally:
            del shard
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def close(self) -> Dict:
        self.flush()
        return {
            "split": self.split_name,
            "total_tokens": self.total_tokens,
            "total_shards": self.total_shards,
            "tokens_per_shard": self.tokens_per_shard,
            "out_dir": str(self.out_dir),
        }


def build_activation_dataset_for_split(
    split_name: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_device: torch.device,
) -> Dict:
    jsonl_path = jsonl_path_for_split(split_name)
    records = read_jsonl(jsonl_path)

    print(f"\nSplit: {split_name}")
    print(f"JSONL: {jsonl_path}")
    print(f"Records: {len(records):,}")

    writer = ActivationShardWriter(ACTIVATION_OUT_DIR, split_name, TOKENS_PER_SHARD)
    extractor = LayerActivationExtractor(model, HOOK_LAYER)
    n_batches = math.ceil(len(records) / BATCH_SIZE)

    for batch_records in tqdm(batch_iter(records, BATCH_SIZE), total=n_batches, desc=split_name):
        batch = tokenize_batch(batch_records, tokenizer, input_device)
        acts = extractor.extract_batch(batch["input_ids"], batch["attention_mask"])
        writer.add(acts)

        del batch, acts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return writer.close()


def save_metadata(model: AutoModelForCausalLM, summaries: Dict[str, Dict]) -> None:
    metadata = {
        "seed": SEED,
        "text_corpus_dir": str(TEXT_CORPUS_DIR),
        "activation_out_dir": str(ACTIVATION_OUT_DIR),
        "model_name": MODEL_NAME,
        "model_dtype": str(MODEL_DTYPE),
        "save_dtype": str(SAVE_DTYPE),
        "hook_layer": HOOK_LAYER,
        "d_model": model.config.hidden_size,
        "d_sae_planned": D_SAE,
        "expansion_factor": EXP_FACTOR,
        "topk_k_planned": K,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "tokens_per_shard": TOKENS_PER_SHARD,
        "splits_to_build": SPLITS_TO_BUILD,
        "splits": summaries,
    }

    metadata_path = ACTIVATION_OUT_DIR / "activation_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print("Metadata saved to:", metadata_path)


def main() -> None:
    prepare_output_dir()
    model, tokenizer = load_model_and_tokenizer()

    input_device = next(model.model.embed_tokens.parameters()).device
    summaries = {}

    print("DEVICE:", DEVICE)
    print("MODEL_DTYPE:", MODEL_DTYPE)
    print("SAVE_DTYPE:", SAVE_DTYPE)
    print("MODEL_NAME:", MODEL_NAME)
    print("TEXT_CORPUS_DIR:", TEXT_CORPUS_DIR)
    print("ACTIVATION_OUT_DIR:", ACTIVATION_OUT_DIR)
    print("HOOK_LAYER:", HOOK_LAYER)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("MAX_LENGTH:", MAX_LENGTH)
    print("SPLITS_TO_BUILD:", SPLITS_TO_BUILD)

    for split_name in SPLITS_TO_BUILD:
        summaries[split_name] = build_activation_dataset_for_split(
            split_name=split_name,
            model=model,
            tokenizer=tokenizer,
            input_device=input_device,
        )

    save_metadata(model, summaries)


if __name__ == "__main__":
    main()
