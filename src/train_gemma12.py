#!/usr/bin/env python3
"""
Contrastive training (InfoNCE) pour Gemma-3 – texte seul.
Compatible 1 ou 2 × A100 40 Go sur Vertex AI.
"""

# ---------------------------------------------------------------------------#
#  Imports
# ---------------------------------------------------------------------------#
import argparse, json, logging, os, io, glob
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import IterableDataset
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import zstandard as zstd
from huggingface_hub import login

# ---------------------------------------------------------------------------#
#  Auth HF
# ---------------------------------------------------------------------------#
hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    print("✅ Authentifié Hugging Face")
else:
    print("⚠️  Token HF manquant – le download échouera")

# ---------------------------------------------------------------------------#
#  Dépendances optionnelles
# ---------------------------------------------------------------------------#
try:
    import gcsfs;  GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

try:
    import neptune
    from neptune.integrations.transformers import NeptuneCallback
    NEPTUNE_AVAILABLE = True
except ImportError:
    NEPTUNE_AVAILABLE = False
    NeptuneCallback = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------#
#  Helpers GCS
# ---------------------------------------------------------------------------#
def is_gcs(path: str) -> bool:
    return path.startswith("gs://")

def sync_gcs(path: str) -> str:
    """Copie (éventuellement wildcard) depuis GCS vers /tmp/gcs_cache et retourne le chemin local."""
    if not GCS_AVAILABLE:
        raise ImportError("gcsfs n'est pas installé")

    cache   = Path("/tmp/gcs_cache")
    rel     = path.replace("gs://", "")
    local   = cache / rel

    if int(os.environ.get("RANK", 0)) == 0:
        logger.info("Sync %s → %s", path, local)
        local.parent.mkdir(parents=True, exist_ok=True)
        fs = gcsfs.GCSFileSystem()
        if "*" in path:
            fs.get(os.path.dirname(path), str(local.parent), recursive=True)
        else:
            fs.get(path, str(local), recursive=True)
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        torch.distributed.barrier()

    return str(local)

# ---------------------------------------------------------------------------#
#  Callback – sauvegarde de la tête de projection
# ---------------------------------------------------------------------------#
    
class SaveProjectionCallback(TrainerCallback):
    def __init__(self, model): super().__init__(); self.model = model
    def on_save(self, args, state, control, **_):
        print(f"🔥 SAVING CHECKPOINT at step {state.global_step}")
        if state.is_world_process_zero:
            ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if ckpt.is_dir():
                torch.save(
                    {
                        "state_dict": self.model.projection_head.state_dict(),
                        "config": {
                            "hidden_size":     self.model.hidden_size,
                            "projection_dim":  self.model.projection_dim,
                            "model_id":        self.model.base_model_id,
                        },
                    },
                    ckpt / "projection_head.pt",
                )
                self.model.model_to_train.save_pretrained(ckpt, safe_serialization=False)

        return control
# ---------------------------------------------------------------------------#
#  Info-NCE
# ---------------------------------------------------------------------------#
class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.05):
        super().__init__(); self.t = temperature
    def forward(self, emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        bs = emb.size(0) // 2
        if bs == 0:
            return {"loss": torch.tensor(0.0, device=emb.device, requires_grad=True)}

        a, b = F.normalize(emb[:bs], 2, 1), F.normalize(emb[bs:2*bs], 2, 1)
        logits = (a @ b.t()) / self.t
        labels = torch.arange(bs, device=emb.device)
        loss   = F.cross_entropy(logits, labels)

        with torch.no_grad():
            acc  = (logits.argmax(1) == labels).float().mean()
            div  = 1 - (a @ a.t())[~torch.eye(bs, device=emb.device, dtype=torch.bool)].mean()

        return {"loss": loss, "contrastive_accuracy": acc, "embedding_diversity": div}

# ---------------------------------------------------------------------------#
#  Dataset & collator
# ---------------------------------------------------------------------------#
def yield_records(path: str):
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh, dctx.stream_reader(fh) as rdr:
        for line in io.TextIOWrapper(rdr, encoding="utf-8", errors="ignore"):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def build_dataset(pattern: str, max_len: int = 512) -> IterableDataset:
    if is_gcs(pattern):
        pattern = os.path.join(sync_gcs(os.path.dirname(pattern)), os.path.basename(pattern))
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(pattern)

    use_ddp = os.environ.get("DDP_BACKEND", "nccl") != "no"
    if use_ddp:
        acc = Accelerator(); proc, world = acc.process_index, acc.num_processes
    else:
        proc, world = 0, 1

    selected = [f for i, f in enumerate(files) if i % world == proc]
    logger.info("RANK %s : %d fichiers", proc, len(selected))

    def gen():
        count = 0
        for fp in selected:
            logger.info("RANK %s : lecture %s", proc, Path(fp).name)
            for rec in yield_records(fp):
                ids = rec.get("input_ids", [])
                if not ids:
                    continue
                aug = rec.get("input_ids_aug") or ids
                yield {"input_ids": ids[:max_len], "input_ids_aug": aug[:max_len]}
                count += 1
                if count % 1000 == 0:
                    logger.info("RANK %s : %d exemples", proc, count)

    return IterableDataset.from_generator(gen)

class ContrastiveCollator:
    def __init__(self, processor: AutoProcessor, max_len: int):
        self.tok     = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        self.max_len = max_len

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = [b for b in batch if b and b.get("input_ids")]
        if not batch:
            return {"input_ids": torch.empty(0, 0, dtype=torch.long),
                    "attention_mask": torch.empty(0, 0, dtype=torch.long)}

        ids = [b["input_ids"][:self.max_len]                           for b in batch]
        aug = [b.get("input_ids_aug", b["input_ids"])[:self.max_len]  for b in batch]

        return self.tok.pad([{"input_ids": x} for x in ids + aug],
                            padding="max_length",
                            max_length=self.max_len,
                            return_tensors="pt")

# ---------------------------------------------------------------------------#
#  Model wrapper
# ---------------------------------------------------------------------------#
class ContrastiveModel(nn.Module):
    def __init__(self, backbone: nn.Module, proj_dim: int, temperature: float, model_id: str):
        super().__init__()
        self.model_to_train = backbone
        self.base_model_id  = model_id
        self.projection_dim = proj_dim

        hidden = getattr(backbone.get_base_model().config, "hidden_size", None)
        if hidden is None:
            hidden = 4608 if "27b" in model_id.lower() else 3840
        self.hidden_size = hidden

        self.projection_head = nn.Sequential(
            nn.Linear(hidden, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        self.loss_fn = InfoNCELoss(temperature)
        self._init_head()

    # --- Gradient-checkpointing delegation ---------------------------------
    def gradient_checkpointing_enable(self, **kw):
        if hasattr(self.model_to_train, "gradient_checkpointing_enable"):
            self.model_to_train.gradient_checkpointing_enable(**kw)
    def gradient_checkpointing_disable(self):
        if hasattr(self.model_to_train, "gradient_checkpointing_disable"):
            self.model_to_train.gradient_checkpointing_disable()
    @property
    def is_gradient_checkpointing(self):
        return getattr(self.model_to_train, "is_gradient_checkpointing", False)
    # -----------------------------------------------------------------------

    def _init_head(self):
        for m in self.projection_head:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1)
        return (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask, pixel_values=None, **_):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out = self.model_to_train(
                input_ids          = input_ids,
                attention_mask     = attention_mask,
                pixel_values       = pixel_values,
                output_hidden_states = True,
                return_dict        = True,
            )
        pooled = self._pool(out.hidden_states[-1], attention_mask)
        emb    = self.projection_head(pooled.float())
        losses = self.loss_fn(emb)
        return {"loss": losses["loss"], "embeddings": emb, "metrics": losses}

# ---------------------------------------------------------------------------#
#  Trainer
# ---------------------------------------------------------------------------#
class ContrastiveTrainer(Trainer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.hist = defaultdict(list)

    def compute_loss(self, model, inputs, return_outputs=False, **_):
        if inputs["input_ids"].numel() == 0:
            return torch.tensor(0.0, device=self.args.device, requires_grad=True)
        out = model(**inputs)
        for k, v in out.get("metrics", {}).items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                self.hist[k].append(v.item())
        return (out["loss"], out) if return_outputs else out["loss"]

    # signature variadique pour absorber start_time
    def log(self, logs: Dict[str, float], *args, **kwargs):
        for k, v in self.hist.items():
            if v:
                logs[f"avg_{k}"] = np.mean(v[-10:])
        super().log(logs, *args, **kwargs)

# ---------------------------------------------------------------------------#
#  Main
# ---------------------------------------------------------------------------#
def main():
    p = argparse.ArgumentParser()
    add = p.add_argument
    add("--model_id",            required=True)
    add("--dataset_path",        required=True)
    add("--output_dir",          required=True)
    add("--load_checkpoint")
    add("--load_in_4bit",        action="store_true")
    add("--lora_r",              type=int, default=8)
    add("--lora_alpha",          type=int, default=16)
    add("--projection_dim",      type=int, default=2048)
    add("--temperature",         type=float, default=0.05)
    add("--batch_size",          type=int, default=1)
    add("--gradient_accumulation", type=int, default=8)
    add("--max_steps",           type=int, default=10)
    add("--learning_rate",       type=float, default=5e-4)
    add("--max_length",          type=int, default=2048)
    add("--seed",                type=int, default=42)
    add("--logging_steps",       type=int, default=1)
    add("--save_steps",          type=int, default=5)
    add("--save_total_limit",    type=int, default=2)
    add("--warmup_steps",        type=int, default=2)
    add("--weight_decay",        type=float, default=0.01)
    add("--neptune_project");   add("--neptune_api_token"); add("--neptune_run_name")
    args = p.parse_args()

    set_seed(args.seed)
    accelerator = Accelerator()
    logger.info("🚀  Rank %s sur %s", accelerator.process_index, accelerator.device)

    # Neptune
    report_to = "none"
    if args.neptune_project and NEPTUNE_AVAILABLE and accelerator.is_main_process:
        os.environ["NEPTUNE_PROJECT"] = args.neptune_project
        if args.neptune_api_token:
            os.environ["NEPTUNE_API_TOKEN"] = args.neptune_api_token
        report_to = "neptune"

    # Processor
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True, token=hf_token)
    if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Modèle
    load = dict(
        trust_remote_code = True,
        token             = hf_token,
        torch_dtype       = torch.bfloat16,
        device_map        = "auto",
        low_cpu_mem_usage = True,
    )
    if args.load_in_4bit:
        load["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit            = True,
            bnb_4bit_quant_type     = "nf4",
            bnb_4bit_compute_dtype  = torch.bfloat16,
            bnb_4bit_use_double_quant = True,
        )
    if torch.cuda.device_count() > 1:
        load["max_memory"] = {"cuda:0": "35GiB", "cuda:1": "35GiB", "cuda:2": "35GiB", "cuda:3": "35GiB",
                              "cuda:4": "35GiB","cuda:5": "35GiB","cuda:6": "35GiB","cuda:7": "35GiB"}

    backbone = Gemma3ForConditionalGeneration.from_pretrained(args.model_id, **load)
    backbone.config.use_cache = False
    if args.load_in_4bit:
        backbone = prepare_model_for_kbit_training(
            backbone,
            use_gradient_checkpointing      = True,
            gradient_checkpointing_kwargs   = {"use_reentrant": False},
        )

    # LoRA
    tgt_modules = {n.split(".")[-1] for n, _ in backbone.named_modules()
                   if any(k in n for k in ["q_proj", "k_proj", "v_proj", "o_proj"])} \
                  or {"q_proj", "k_proj", "v_proj", "o_proj"}
    lora_cfg = LoraConfig(
        task_type     = TaskType.CAUSAL_LM,
        r             = args.lora_r,
        lora_alpha    = args.lora_alpha,
        lora_dropout  = 0.05,
        target_modules= list(tgt_modules),
    )
    backbone = get_peft_model(backbone, lora_cfg)
    backbone.print_trainable_parameters()

    # Wrapper contrastif
    model = ContrastiveModel(backbone, args.projection_dim, args.temperature, args.model_id)
    model.is_parallelizable = True
    model.model_parallel    = True

    # Dataset
    dataset  = build_dataset(args.dataset_path, args.max_length)
    collator = ContrastiveCollator(processor, args.max_length)

    # TrainingArguments
    targs = TrainingArguments(
        output_dir                    = args.output_dir,
        per_device_train_batch_size   = args.batch_size,
        gradient_accumulation_steps   = args.gradient_accumulation,
        max_steps                     = args.max_steps,
        learning_rate                 = args.learning_rate,
        weight_decay                  = args.weight_decay,
        warmup_steps                  = args.warmup_steps,
        lr_scheduler_type             = "cosine",
        logging_steps                 = args.logging_steps,
        save_strategy                 = "steps",
        save_steps                    = args.save_steps,
        save_total_limit              = args.save_total_limit,
        optim                         = "paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        bf16                          = True,
        tf32                          = True,
        remove_unused_columns         = False,
        seed                          = args.seed,
        report_to                     = report_to,
        run_name                      = args.neptune_run_name,
        logging_first_step            = True,
        ddp_backend                   = "no",  # 1 processus + model-parallel
        dataloader_num_workers        = 0,
        dataloader_drop_last          = True,
        gradient_checkpointing        = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        ddp_find_unused_parameters    = False,
        save_safetensors=False,
    )

    trainer = ContrastiveTrainer(
        model            = model,
        args             = targs,
        train_dataset    = dataset,
        data_collator    = collator,
        callbacks        = [SaveProjectionCallback(model)],
        tokenizer        = processor.tokenizer if hasattr(processor, "tokenizer") else processor,
    )

    logger.info("🔥 Démarrage entraînement")
    trainer.train(resume_from_checkpoint=args.load_checkpoint)
    logger.info("✅ Entraînement terminé")

    if accelerator.is_main_process:
        trainer.save_model(args.output_dir, safe_serialization=False)
        with open(Path(args.output_dir) / "training_config.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        logger.info("🎉 Modèle sauvegardé dans %s", args.output_dir)

# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
