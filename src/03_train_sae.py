"""Stage 3: train a TopK sparse autoencoder on Llama 3 8B layer-12 activations.

The four SAEs compared in the thesis were trained as a K curriculum rather than
four independent runs. Pick one entry of RUN_PRESETS with the RUN variable:

    k32  from scratch,                         epochs 1-15
    k48  init from k32/sae_epoch_015.pt,       epochs 16-20
    k56  init from k48/sae_epoch_020.pt,       epochs 21-25   <- SAE used downstream
    k64  init from k48/sae_epoch_020.pt,       epochs 21-25

"init_from" loads the SAE weights, epoch/step counters and dead-feature
statistics of the smaller-K checkpoint, but starts a fresh AdamW optimizer and
resets best_val_mse (the reconstruction scale changes with K). Changing K does
not change any parameter shapes: K only sets how many latents stay active.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


try:
    from google.colab import drive
    IN_COLAB = True
except Exception:
    IN_COLAB = False


if IN_COLAB:
    drive.mount("/content/drive")


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = Path("/content/drive/MyDrive/Diploma") if IN_COLAB else Path("data")
ACTIVATION_DATA_DIR = DATA_ROOT / "activation_datasets2"
RUN_ROOT = DATA_ROOT / "sae_runs"

RUN_PRESETS = {
    "k32": dict(k=32, num_epochs=15, resample_every_epochs=2, resample_max_features=4096, init_from=None),
    "k48": dict(k=48, num_epochs=20, resample_every_epochs=3, resample_max_features=2048, init_from=("k32", "sae_epoch_015.pt")),
    "k56": dict(k=56, num_epochs=25, resample_every_epochs=7, resample_max_features=2048, init_from=("k48", "sae_epoch_020.pt")),
    "k64": dict(k=64, num_epochs=25, resample_every_epochs=7, resample_max_features=2048, init_from=("k48", "sae_epoch_020.pt")),
}
RUN = "k32"
PRESET = RUN_PRESETS[RUN]

D_IN = 4096
EXP_FACTOR = 4
D_SAE = D_IN * EXP_FACTOR
K = PRESET["k"]
HOOK_LAYER = 12


AUX_LOSS_COEFF = 1 / 32
DEAD_THRESHOLD = 1600


RESAMPLE_ENABLED = True

RESAMPLE_EVERY_EPOCHS = PRESET["resample_every_epochs"]
RESAMPLE_BEFORE_FINAL_EPOCH_ONLY = True
RESAMPLE_MAX_FEATURES = PRESET["resample_max_features"]
RESAMPLE_CANDIDATE_BATCHES = 32
RESAMPLE_TOP_TOKENS_PER_BATCH = 256
RESET_OPTIMIZER_STATE_AFTER_RESAMPLE = True


SAE_OUT_DIR = RUN_ROOT / RUN
CKPT_DIR = SAE_OUT_DIR / "checkpoints"
SAE_OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH_JSONL = SAE_OUT_DIR / "metrics.jsonl"
LOG_PATH_CSV = SAE_OUT_DIR / "metrics.csv"

SAE_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

TRAIN_BATCH_TOKENS = 8192
EVAL_BATCH_TOKENS = 8192
NUM_EPOCHS = PRESET["num_epochs"]  # total epoch index to reach, counting epochs of the init checkpoint
LR = 1e-4
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 1.0

VALIDATE_EVERY_STEPS = 500
CHECKPOINT_EVERY_STEPS = 1000

MAX_VAL_BATCHES: Optional[int] = 300
MAX_TEST_BATCHES: Optional[int] = None

# Continue an interrupted run of the same preset (restores the optimizer too),
# e.g. CKPT_DIR / "sae_epoch_005.pt". Leave None for a normal run.
RESUME_CKPT: Optional[str] = None

INIT_FROM_CKPT: Optional[Path] = None
if PRESET["init_from"] is not None:
    _init_run, _init_file = PRESET["init_from"]
    INIT_FROM_CKPT = RUN_ROOT / _init_run / "checkpoints" / _init_file

RESET_LOGS = False


print("DEVICE:", DEVICE)
print("SAE_DTYPE:", SAE_DTYPE)
print("D_IN:", D_IN)
print("D_SAE:", D_SAE)
print("RUN:", RUN, "K:", K, "NUM_EPOCHS:", NUM_EPOCHS)
print("INIT_FROM_CKPT:", INIT_FROM_CKPT)
print("ACTIVATION_DATA_DIR:", ACTIVATION_DATA_DIR)
print("SAE_OUT_DIR:", SAE_OUT_DIR)

if RESET_LOGS:
    for path in [LOG_PATH_JSONL, LOG_PATH_CSV]:
        if path.exists():
            path.unlink()


class TopKSAE(nn.Module):
    """TopK SAE with sparse top-k decoding."""

    def __init__(self, d_in: int, d_sae: int, k: int, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae, dtype=dtype))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in, dtype=dtype))
        self.b_enc = nn.Parameter(torch.zeros(d_sae, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        with torch.no_grad():
            self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        dec_f = self.W_dec.data.float()
        norms = dec_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        dec_f = dec_f / norms
        self.W_dec.data.copy_(dec_f.to(dtype=self.W_dec.dtype))

    @torch.no_grad()
    def remove_decoder_parallel_grad(self) -> None:
        """Project decoder gradients to be orthogonal to decoder rows."""
        if self.W_dec.grad is None:
            return
        dec_f = self.W_dec.data.float()
        grad_f = self.W_dec.grad.float()

        parallel = (grad_f * dec_f).sum(dim=-1, keepdim=True) * dec_f
        grad_orth = grad_f - parallel
        self.W_dec.grad.copy_(grad_orth.to(dtype=self.W_dec.grad.dtype))

    def encode_topk(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_cent = x - self.b_dec
        pre_acts = x_cent @ self.W_enc + self.b_enc
        topk_vals, topk_idx = pre_acts.topk(self.k, dim=-1)
        topk_vals = F.relu(topk_vals)
        return topk_vals, topk_idx

    def decode_topk(self, topk_vals: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        selected_dec = self.W_dec[topk_idx]
        x_recon = (topk_vals.unsqueeze(-1) * selected_dec).sum(dim=1)
        return x_recon + self.b_dec

    def encode_dense(self, x: torch.Tensor) -> torch.Tensor:
        """For later interpretability/intervention, not for training."""
        topk_vals, topk_idx = self.encode_topk(x)
        feature_acts = torch.zeros(
            x.shape[0],
            self.d_sae,
            device=x.device,
            dtype=x.dtype,
        )
        feature_acts.scatter_(-1, topk_idx, topk_vals)
        return feature_acts

    def decode_dense(self, feature_acts: torch.Tensor) -> torch.Tensor:
        return feature_acts @ self.W_dec + self.b_dec

    def decode_topk_no_bias(self, topk_vals: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        selected_dec = self.W_dec[topk_idx]
        return (topk_vals.unsqueeze(-1) * selected_dec).sum(dim=1)

    def forward(self, x: torch.Tensor, dead_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        topk_vals, topk_idx = self.encode_topk(x)
        x_recon = self.decode_topk(topk_vals, topk_idx)
        mse = F.mse_loss(x_recon.float(), x.float())
        l0 = (topk_vals > 0).float().sum(dim=-1).mean()

        aux_loss = torch.tensor(0.0, device=x.device)
        aux_l0 = torch.tensor(0.0, device=x.device)

        if dead_mask is not None and dead_mask.any():
            x_cent = x - self.b_dec
            pre_acts = x_cent @ self.W_enc + self.b_enc

            pre_acts_dead = pre_acts.clone()
            pre_acts_dead[:, ~dead_mask] = -torch.inf

            K_aux = min(self.k, int(dead_mask.sum().item()))
            topk_vals_dead, topk_idx_dead = pre_acts_dead.topk(K_aux, dim=-1)
            topk_vals_dead = F.relu(topk_vals_dead)

            aux_l0 = (topk_vals_dead > 0).float().sum(dim=-1).mean()

            x_aux_recon = self.decode_topk_no_bias(topk_vals_dead, topk_idx_dead)

            residual = (x - x_recon).detach().float()
            aux_loss = F.mse_loss(x_aux_recon.float(), residual)


        return {
            "x_recon": x_recon,
            "mse": mse,
            "aux_loss": aux_loss,
            "l0": l0,
            "topk_vals": topk_vals,
            "topk_idx": topk_idx,
            "aux_l0": aux_l0,
        }

sae = TopKSAE(D_IN, D_SAE, K, dtype=SAE_DTYPE).to(DEVICE)
params = sum(p.numel() for p in sae.parameters())
print(f"W_enc:      {tuple(sae.W_enc.shape)}")
print(f"W_dec:      {tuple(sae.W_dec.shape)}")
print(f"Parameters: {params:,}")
print(f"Param memory approx fp16/bf16: {params * 2 / 1e9:.2f} GB")


def list_activation_shards(split_name: str) -> List[Path]:
    split_dir = ACTIVATION_DATA_DIR / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    paths = sorted(split_dir.glob(f"{split_name}_acts_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No activation shards found in {split_dir}")

    return paths


def load_shard_activations(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)["activations"]


def iter_activation_batches(
    split_name: str,
    batch_tokens: int,
    shuffle_shards: bool,
    shuffle_tokens: bool,
    max_batches: Optional[int] = None,
) -> Iterator[torch.Tensor]:
    shard_paths = list_activation_shards(split_name)

    if shuffle_shards:
        random.shuffle(shard_paths)

    yielded = 0

    for shard_path in shard_paths:
        acts = load_shard_activations(shard_path)
        n = acts.shape[0]

        if shuffle_tokens:
            acts = acts[torch.randperm(n)]

        for start in range(0, n, batch_tokens):
            batch = acts[start:start + batch_tokens]
            if batch.shape[0] == 0:
                continue

            yield batch
            yielded += 1

            if max_batches is not None and yielded >= max_batches:
                return

        del acts
        gc.collect()


def append_metric(row: Dict) -> None:
    row = dict(row)

    with LOG_PATH_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    file_exists = LOG_PATH_CSV.exists()
    fieldnames = list(row.keys())

    if file_exists:
        with LOG_PATH_CSV.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            old_header = next(reader, None)
        if old_header:
            fieldnames = old_header + [k for k in fieldnames if k not in old_header]

    with LOG_PATH_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


@torch.inference_mode()
def evaluate_sae(
    sae_model: TopKSAE,
    split_name: str,
    batch_tokens: int,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    sae_model.eval()

    total_tokens = 0
    total_mse_weighted = 0.0
    total_l0_weighted = 0.0

    for x_cpu in iter_activation_batches(
        split_name=split_name,
        batch_tokens=batch_tokens,
        shuffle_shards=False,
        shuffle_tokens=False,
        max_batches=max_batches,
    ):
        x = x_cpu.to(device=DEVICE, dtype=SAE_DTYPE, non_blocking=True)
        out = sae_model(x)

        n = x.shape[0]
        total_tokens += n
        total_mse_weighted += out["mse"].item() * n
        total_l0_weighted += out["l0"].item() * n

        del x, out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "split": split_name,
        "tokens": total_tokens,
        "mse": total_mse_weighted / max(total_tokens, 1),
        "l0": total_l0_weighted / max(total_tokens, 1),
    }

def save_checkpoint(
    sae_model: TopKSAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    best_val_mse: float,
    name: str,
    feature_last_active: torch.Tensor,
) -> None:
    path = CKPT_DIR / name
    tmp_path = CKPT_DIR / (name + ".tmp")

    payload = {
        "epoch": epoch,
        "step": step,
        "best_val_mse": best_val_mse,
        "feature_last_active": feature_last_active.cpu(),
        "config": {
            "d_in": D_IN,
            "d_sae": D_SAE,
            "exp_factor": EXP_FACTOR,
            "k": K,
            "hook_layer": HOOK_LAYER,
            "dtype": str(SAE_DTYPE),
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "train_batch_tokens": TRAIN_BATCH_TOKENS,
            "eval_batch_tokens": EVAL_BATCH_TOKENS,
            "aux_loss_coeff": AUX_LOSS_COEFF,
            "dead_threshold": DEAD_THRESHOLD,
            "resample_enabled": RESAMPLE_ENABLED,
            "resample_every_epochs": RESAMPLE_EVERY_EPOCHS,
            "resample_max_features": RESAMPLE_MAX_FEATURES,
            "resample_candidate_batches": RESAMPLE_CANDIDATE_BATCHES,
            "resample_top_tokens_per_batch": RESAMPLE_TOP_TOKENS_PER_BATCH,
        },
        "model_state_dict": sae_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    print(f"Saved checkpoint: {path}")

def load_checkpoint(path: str, sae_model: TopKSAE, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    sae_model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    print(f"Loaded checkpoint: {path}")
    print(f"epoch={ckpt.get('epoch')}, step={ckpt.get('step')}, best_val_mse={ckpt.get('best_val_mse')}")
    return ckpt


@torch.inference_mode()
def collect_high_error_residual_directions(
    sae_model: TopKSAE,
    n_vectors: int,
    batch_tokens: int,
    max_batches: int,
    top_tokens_per_batch: int,
) -> torch.Tensor:
    sae_model.eval()

    residual_chunks = []
    error_chunks = []

    for x_cpu in iter_activation_batches(
        split_name="train",
        batch_tokens=batch_tokens,
        shuffle_shards=True,
        shuffle_tokens=True,
        max_batches=max_batches,
    ):
        x = x_cpu.to(device=DEVICE, dtype=SAE_DTYPE, non_blocking=True)

        out = sae_model(x, dead_mask=None)
        residual = (x - out["x_recon"]).float()
        residual_error = residual.pow(2).sum(dim=-1)

        k = min(top_tokens_per_batch, residual.shape[0])
        top_errors, top_idx = residual_error.topk(k, dim=0)

        residual_chunks.append(residual[top_idx].cpu())
        error_chunks.append(top_errors.cpu())

        del x, x_cpu, out, residual, residual_error, top_errors, top_idx

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not residual_chunks:
        raise RuntimeError("No residuals collected for resampling.")

    residuals = torch.cat(residual_chunks, dim=0)
    errors = torch.cat(error_chunks, dim=0)

    k_final = min(n_vectors, residuals.shape[0])
    _, final_idx = errors.topk(k_final, dim=0)

    selected = residuals[final_idx]

    norms = selected.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    directions = selected / norms

    directions = directions.to(device=DEVICE, dtype=SAE_DTYPE)

    del residuals, errors, selected, norms

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return directions


@torch.no_grad()
def reset_optimizer_state_for_resampled_features(
    optimizer: torch.optim.Optimizer,
    sae_model: TopKSAE,
    chosen_idx_cpu: torch.Tensor,
) -> None:
    chosen_idx_cpu = chosen_idx_cpu.long().cpu()

    for param in [sae_model.W_enc, sae_model.W_dec, sae_model.b_enc]:
        if param not in optimizer.state:
            continue

        state = optimizer.state[param]

        for key in ["exp_avg", "exp_avg_sq"]:
            if key not in state:
                continue

            buf = state[key]

            if param is sae_model.W_enc:
                idx = chosen_idx_cpu.to(device=buf.device)
                buf[:, idx] = 0

            elif param is sae_model.W_dec:
                idx = chosen_idx_cpu.to(device=buf.device)
                buf[idx, :] = 0

            elif param is sae_model.b_enc:
                idx = chosen_idx_cpu.to(device=buf.device)
                buf[idx] = 0


@torch.no_grad()
def resample_dead_features(
    sae_model: TopKSAE,
    optimizer: torch.optim.Optimizer,
    feature_last_active: torch.Tensor,
    global_step: int,
) -> Dict[str, int | float]:
    sae_model.eval()

    dead_mask_cpu = (global_step - feature_last_active) > DEAD_THRESHOLD
    dead_idx = torch.nonzero(dead_mask_cpu, as_tuple=False).flatten()

    dead_before = int(dead_idx.numel())

    if dead_before == 0:
        info = {
            "resampled_n": 0,
            "dead_before": 0,
            "dead_after": 0,
            "encoder_col_norm": 0.0,
        }
        print("Resampling skipped: no dead features.")
        return info

    n_resample = min(dead_before, RESAMPLE_MAX_FEATURES)

    perm = torch.randperm(dead_before)
    chosen_idx_cpu = dead_idx[perm[:n_resample]].long()

    print(
        f"Resampling {n_resample:,} dead features "
        f"out of {dead_before:,} dead features at step {global_step}."
    )

    directions = collect_high_error_residual_directions(
        sae_model=sae_model,
        n_vectors=n_resample,
        batch_tokens=TRAIN_BATCH_TOKENS,
        max_batches=RESAMPLE_CANDIDATE_BATCHES,
        top_tokens_per_batch=RESAMPLE_TOP_TOKENS_PER_BATCH,
    )
    actual_n = directions.shape[0]

    if actual_n < n_resample:
        print(
            f"WARNING: requested {n_resample} resampling directions, "
            f"but collected only {actual_n}. Resampling {actual_n} features."
        )

    chosen_idx_cpu = chosen_idx_cpu[:actual_n]
    chosen_idx_gpu = chosen_idx_cpu.to(device=DEVICE)
    n_resample = actual_n

    alive_mask_cpu = ~dead_mask_cpu

    if alive_mask_cpu.any():
        alive_idx_gpu = torch.nonzero(alive_mask_cpu, as_tuple=False).flatten().to(device=DEVICE)
        enc_col_norm = (
            sae_model.W_enc.data[:, alive_idx_gpu]
            .float()
            .norm(dim=0)
            .mean()
            .item()
        )
    else:
        enc_col_norm = sae_model.W_enc.data.float().norm(dim=0).mean().item()

    sae_model.W_dec.data[chosen_idx_gpu] = directions.to(dtype=sae_model.W_dec.dtype)

    sae_model.W_enc.data[:, chosen_idx_gpu] = (
        directions.T.to(dtype=sae_model.W_enc.dtype) * enc_col_norm
    )

    sae_model.b_enc.data[chosen_idx_gpu] = 0.0

    feature_last_active[chosen_idx_cpu] = global_step

    sae_model.normalize_decoder()

    if RESET_OPTIMIZER_STATE_AFTER_RESAMPLE:
        reset_optimizer_state_for_resampled_features(
            optimizer=optimizer,
            sae_model=sae_model,
            chosen_idx_cpu=chosen_idx_cpu,
        )
        print("Optimizer state reset only for resampled features.")

    dead_after = int(((global_step - feature_last_active) > DEAD_THRESHOLD).sum().item())

    info = {
        "resampled_n": int(n_resample),
        "dead_before": int(dead_before),
        "dead_after": int(dead_after),
        "encoder_col_norm": float(enc_col_norm),
    }

    print("Resampling info:", info)

    del directions, chosen_idx_gpu

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return info


def initialize_b_dec_from_train_mean(sae_model: TopKSAE, max_batches: int = 50) -> None:
    print("Estimating b_dec from train activation mean...")

    total = torch.zeros(D_IN, dtype=torch.float64)
    total_tokens = 0

    for x_cpu in iter_activation_batches(
        split_name="train",
        batch_tokens=EVAL_BATCH_TOKENS,
        shuffle_shards=False,
        shuffle_tokens=False,
        max_batches=max_batches,
    ):
        x = x_cpu.double()
        total += x.sum(dim=0)
        total_tokens += x.shape[0]
        del x, x_cpu

    mean = (total / max(total_tokens, 1)).to(device=DEVICE, dtype=SAE_DTYPE)

    with torch.no_grad():
        sae_model.b_dec.copy_(mean)

    print(f"b_dec initialized from {total_tokens:,} tokens.")


optimizer = torch.optim.AdamW(
    sae.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

feature_last_active = torch.zeros(D_SAE, dtype=torch.long)

start_epoch = 0
global_step = 0
best_val_mse = float("inf")

start_ckpt = RESUME_CKPT or INIT_FROM_CKPT

if start_ckpt is not None:
    if RESUME_CKPT is not None:
        # Same run, same K: restore optimizer and best validation MSE as well.
        ckpt = load_checkpoint(str(RESUME_CKPT), sae, optimizer)
        best_val_mse = float(ckpt.get("best_val_mse", float("inf")))
        print("Resuming an interrupted run.")
    else:
        # K curriculum: reuse SAE weights of the smaller-K run, fresh optimizer.
        ckpt = load_checkpoint(str(INIT_FROM_CKPT), sae, optimizer=None)
        print(f"Fine-tuning from a K={ckpt.get('config', {}).get('k')} checkpoint with K={K}.")

    start_epoch = int(ckpt.get("epoch", 0))
    global_step = int(ckpt.get("step", 0))

    if "feature_last_active" in ckpt:
        feature_last_active = ckpt["feature_last_active"]
        print(f"Restored feature_last_active, "
                f"dead at resume: {int(((global_step - feature_last_active) > DEAD_THRESHOLD).sum())}")
    else:
        feature_last_active = torch.full((D_SAE,), global_step, dtype=torch.long)
        print("feature_last_active not in checkpoint; initialized to global_step.")

    print("start_epoch:", start_epoch)
    print("global_step:", global_step)
    print("best_val_mse:", best_val_mse)
else:
    initialize_b_dec_from_train_mean(sae, max_batches=50)
    print("Starting training from scratch.")


def write_run_config() -> None:
    run_config = {
        "run": RUN,
        "init_from": str(INIT_FROM_CKPT) if INIT_FROM_CKPT else None,
        "seed": SEED,
        "device": DEVICE,
        "d_in": D_IN,
        "d_sae": D_SAE,
        "exp_factor": EXP_FACTOR,
        "k": K,
        "hook_layer": HOOK_LAYER,
        "num_epochs": NUM_EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "train_batch_tokens": TRAIN_BATCH_TOKENS,
        "eval_batch_tokens": EVAL_BATCH_TOKENS,
        "validate_every_steps": VALIDATE_EVERY_STEPS,
        "checkpoint_every_steps": CHECKPOINT_EVERY_STEPS,
        "sae_dtype": str(SAE_DTYPE),
        "optimizer": "AdamW",
        "activation_data_dir": str(ACTIVATION_DATA_DIR),
        "sae_out_dir": str(SAE_OUT_DIR),
        "aux_loss_coeff": AUX_LOSS_COEFF,
        "dead_threshold": DEAD_THRESHOLD,
        "resample_enabled": RESAMPLE_ENABLED,
        "resample_every_epochs": RESAMPLE_EVERY_EPOCHS,
        "resample_max_features": RESAMPLE_MAX_FEATURES,
        "resample_candidate_batches": RESAMPLE_CANDIDATE_BATCHES,
        "resample_top_tokens_per_batch": RESAMPLE_TOP_TOKENS_PER_BATCH,
        "reset_optimizer_state_after_resample": RESET_OPTIMIZER_STATE_AFTER_RESAMPLE,
    }

    with (SAE_OUT_DIR / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    print(json.dumps(run_config, indent=2, ensure_ascii=False))


def train_sae() -> None:
    global global_step, best_val_mse, feature_last_active, optimizer

    write_run_config()

    for epoch in range(start_epoch, NUM_EPOCHS):
        sae.train()
        epoch_start = time.time()

        train_iter = iter_activation_batches(
            split_name="train",
            batch_tokens=TRAIN_BATCH_TOKENS,
            shuffle_shards=True,
            shuffle_tokens=True,
            max_batches=None,
        )

        pbar = tqdm(train_iter, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for x_cpu in pbar:
            global_step += 1
            x = x_cpu.to(device=DEVICE, dtype=SAE_DTYPE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            dead_mask = ((global_step - feature_last_active) > DEAD_THRESHOLD).to(DEVICE)
            out = sae(x, dead_mask=dead_mask)

            loss = out["mse"] + AUX_LOSS_COEFF * out["aux_loss"]

            with torch.no_grad():
                active_features = out["topk_idx"][out["topk_vals"] > 0].unique()
                feature_last_active[active_features.cpu()] = global_step

            loss.backward()
            sae.remove_decoder_parallel_grad()

            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(sae.parameters(), GRAD_CLIP_NORM)

            optimizer.step()

            with torch.no_grad():
                sae.normalize_decoder()

            train_mse  = float(out["mse"].item())
            train_aux  = float(out["aux_loss"].item())
            train_loss = float(loss.item())
            train_l0   = float(out["l0"].item())
            n_dead     = int(dead_mask.sum().item())
            train_aux_l0 = float(out["aux_l0"].item())

            pbar.set_postfix({
                "step": global_step,
                "mse":  f"{train_mse:.6f}",
                "aux":  f"{train_aux:.6f}",
                "l0":   f"{train_l0:.2f}",
                "aux_l0": f"{train_aux_l0:.2f}",
                "dead": n_dead,
                "best": f"{best_val_mse:.6f}" if best_val_mse != float("inf") else "inf",
            })

            append_metric({
                "time":          time.time(),
                "phase":         "train",
                "epoch":         epoch + 1,
                "step":          global_step,
                "split":         "train",
                "tokens":        int(x.shape[0]),
                "mse":           train_mse,
                "aux_loss":      train_aux,
                "total_loss":    train_loss,
                "l0":            train_l0,
                "aux_l0": train_aux_l0,
                "n_dead":        n_dead,
                "best_val_mse":  best_val_mse,
            })

            if global_step % VALIDATE_EVERY_STEPS == 0:
                val_metrics = evaluate_sae(
                    sae_model=sae,
                    split_name="val",
                    batch_tokens=EVAL_BATCH_TOKENS,
                    max_batches=MAX_VAL_BATCHES,
                )
                val_mse = float(val_metrics["mse"])

                print(
                    f"\nValidation at step {global_step}: "
                    f"val_mse={val_mse:.6f}, val_l0={val_metrics['l0']:.2f}, "
                    f"dead={n_dead}"
                )

                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    save_checkpoint(
                        sae_model=sae,
                        optimizer=optimizer,
                        epoch=epoch + 1,
                        step=global_step,
                        best_val_mse=best_val_mse,
                        feature_last_active=feature_last_active,
                        name="sae_best.pt",
                    )

                append_metric({
                    "time":         time.time(),
                    "phase":        "val",
                    "epoch":        epoch + 1,
                    "step":         global_step,
                    "split":        "val",
                    "tokens":       int(val_metrics["tokens"]),
                    "mse":          val_mse,
                    "aux_loss":     None,
                    "total_loss":   None,
                    "l0":           float(val_metrics["l0"]),
                    "n_dead":       n_dead,
                    "best_val_mse": best_val_mse,
                })

                sae.train()

            if global_step % CHECKPOINT_EVERY_STEPS == 0:
                save_checkpoint(
                    sae_model=sae,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    step=global_step,
                    best_val_mse=best_val_mse,
                    feature_last_active=feature_last_active,
                    name=f"sae_step_{global_step:08d}.pt",
                )

            del x, x_cpu, out, loss

        should_resample = (
            RESAMPLE_ENABLED
            and ((epoch + 1) % RESAMPLE_EVERY_EPOCHS == 0)
        )

        if RESAMPLE_BEFORE_FINAL_EPOCH_ONLY and (epoch + 1) >= NUM_EPOCHS:
            should_resample = False

        if should_resample:
            print("\n" + "=" * 80)
            print(f"Dead feature resampling after epoch {epoch + 1}")
            print("=" * 80)

            resample_info = resample_dead_features(
                sae_model=sae,
                optimizer=optimizer,
                feature_last_active=feature_last_active,
                global_step=global_step,
            )

            append_metric({
                "time": time.time(),
                "phase": "resample",
                "epoch": epoch + 1,
                "step": global_step,
                "split": "train",
                "tokens": 0,
                "mse": None,
                "aux_loss": None,
                "total_loss": None,
                "l0": None,
                "aux_l0": None,
                "n_dead": int(resample_info["dead_after"]),
                "dead_before": int(resample_info["dead_before"]),
                "dead_after": int(resample_info["dead_after"]),
                "resampled_n": int(resample_info["resampled_n"]),
                "encoder_col_norm": float(resample_info["encoder_col_norm"]),
                "best_val_mse": best_val_mse,
            })

            save_checkpoint(
                sae_model=sae,
                optimizer=optimizer,
                epoch=epoch + 1,
                step=global_step,
                best_val_mse=best_val_mse,
                feature_last_active=feature_last_active,
                name=f"sae_resampled_epoch_{epoch + 1:03d}.pt",
            )

            sae.train()

        epoch_time = time.time() - epoch_start

        save_checkpoint(
            sae_model=sae,
            optimizer=optimizer,
            epoch=epoch + 1,
            step=global_step,
            best_val_mse=best_val_mse,
            feature_last_active=feature_last_active,
            name=f"sae_epoch_{epoch + 1:03d}.pt",
        )

        append_metric({
            "time":            time.time(),
            "phase":           "epoch_end",
            "epoch":           epoch + 1,
            "step":            global_step,
            "split":           "train",
            "tokens":          0,
            "mse":             None,
            "aux_loss":        None,
            "total_loss":      None,
            "l0":              None,
            "n_dead":          int(((global_step - feature_last_active) > DEAD_THRESHOLD).sum()),
            "best_val_mse":    best_val_mse,
            "epoch_time_sec":  epoch_time,
        })

        print(f"Epoch {epoch + 1} finished in {epoch_time / 60:.1f} min.")

    print("Training done.")


train_sae()


# Downstream stages use the last-epoch checkpoint (sae_epoch_XXX.pt), not
# sae_best.pt, so the final test evaluation reports that same checkpoint.
final_ckpt_path = CKPT_DIR / f"sae_epoch_{NUM_EPOCHS:03d}.pt"

if final_ckpt_path.exists():
    load_checkpoint(str(final_ckpt_path), sae, optimizer=None)
else:
    print("Final-epoch checkpoint not found; evaluating current SAE.")

test_metrics = evaluate_sae(
    sae_model=sae,
    split_name="test",
    batch_tokens=EVAL_BATCH_TOKENS,
    max_batches=MAX_TEST_BATCHES,
)

print("Test metrics:")
print(json.dumps(test_metrics, indent=2, ensure_ascii=False))

append_metric({
    "time": time.time(),
    "phase": "test",
    "epoch": None,
    "step": global_step,
    "split": "test",
    "tokens": int(test_metrics["tokens"]),
    "mse": float(test_metrics["mse"]),
    "l0": float(test_metrics["l0"]),
    "best_val_mse": best_val_mse,
})
