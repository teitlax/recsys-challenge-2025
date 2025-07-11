#!/usr/bin/env python3
"""
Contrastive Training avec InfoNCE - Version pour données tokenisées
Entraînement d'embeddings universels sans supervision
"""
import argparse
import json
import logging
import os
import shutil
import io
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from datasets import IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import get_peft_model, LoraConfig, TaskType
import zstandard as zstd
import glob

# Neptune integration
try:
    import neptune
    from neptune.integrations.transformers import NeptuneCallback
    NEPTUNE_AVAILABLE = True
except ImportError:
    NEPTUNE_AVAILABLE = False
    NeptuneCallback = None

# GCS support
try:
    import gcsfs
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CALLBACK MODIFIÉ POUR SAUVEGARDER ADAPTER_CONFIG.JSON
# ============================================================================
class SaveProjectionCallback(TrainerCallback):
    """Sauvegarde le modèle PEFT complet et projection_head.pt dans chaque checkpoint."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        
        # S'assurer que seul le processus principal écrit sur le disque
        if state.is_world_process_zero and os.path.isdir(ckpt_dir):
            try:
                # Sauvegarder le modèle PEFT complet (adapter_config.json + adapter_model.safetensors)
                self.model.model_to_train.save_pretrained(ckpt_dir)
                logger.info(f"✅ PEFT model (with adapter_config.json) saved to {ckpt_dir}")
                
                # Sauvegarder aussi la projection head
                projection_path = os.path.join(ckpt_dir, "projection_head.pt")
                torch.save(self.model.projection_head.state_dict(), projection_path)
                logger.info(f"✅ Projection head saved to {projection_path}")
                
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
                
        return control


# ============================================================================
# INFONCE LOSS
# ============================================================================
class InfoNCELoss(nn.Module):
    """InfoNCE loss pour apprentissage contrastif"""
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = embeddings.size(0) // 2
        if batch_size <= 0:
            return {"loss": torch.tensor(0.0, device=embeddings.device, requires_grad=True)}
        
        orig_emb = F.normalize(embeddings[:batch_size], p=2, dim=1)
        aug_emb = F.normalize(embeddings[batch_size:2*batch_size], p=2, dim=1)
        
        sim_matrix = torch.mm(orig_emb, aug_emb.t()) / self.temperature
        labels = torch.arange(batch_size, device=embeddings.device)
        loss = F.cross_entropy(sim_matrix, labels)
        
        with torch.no_grad():
            preds = torch.argmax(sim_matrix, dim=1)
            accuracy = (preds == labels).float().mean()
            all_sims = torch.mm(orig_emb, orig_emb.t())
            mask = ~torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
            avg_similarity = all_sims[mask].mean()
            diversity = 1 - avg_similarity
        
        return {
            "loss": loss, "contrastive_accuracy": accuracy, "embedding_diversity": diversity
        }


# ============================================================================
# DATA LOADING FOR TOKENIZED DATA
# ============================================================================
def yield_tokenized_records(path: str):
    """Helper: Streams and decompresses records from a single .jsonl.zst file."""
    dctx = zstd.ZstdDecompressor()
    fh = None
    try:
        if path.startswith("gs://"):
            if not GCS_AVAILABLE: raise ImportError("gcsfs is needed for gs:// paths.")
            fs = gcsfs.GCSFileSystem()
            fh = fs.open(path.replace("gs://", ""), "rb")
        else:
            fh = open(path, "rb")

        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='ignore')
            for line in text_stream:
                line = line.strip()
                if line:
                    try: yield json.loads(line)
                    except json.JSONDecodeError: continue
    finally:
        if fh: fh.close()

def build_tokenized_dataset(pattern: str, max_length: int = 512) -> IterableDataset:
    """Main function: Finds files and builds the IterableDataset using a generator."""
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    logger.info(f"Found {len(files)} files for contrastive training.")

    def tokenized_generator():
        records_yielded = 0
        for fpath in files:
            logger.info(f"Processing {fpath}")
            for record in yield_tokenized_records(fpath):
                input_ids = record.get('input_ids', [])
                if not input_ids: continue
                aug = (
                    record.get('input_ids_aug')
                    or record.get('input_ids_aug1')
                    or record.get('input_ids_aug2')
                    or input_ids
                )
                yield {
                    'input_ids':     input_ids[:max_length],
                    'input_ids_aug': aug[:max_length],
                }
                records_yielded += 1
                if records_yielded % 10000 == 0: logger.info(f"✅ Processed {records_yielded:,} records")
        if records_yielded == 0: raise RuntimeError("No valid records generated from dataset")

    return IterableDataset.from_generator(tokenized_generator)


class ContrastiveDataCollator:
    """Collator pour apprentissage contrastif"""
    def __init__(self, tokenizer: AutoTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        valid_examples = [ex for ex in examples if ex and ex.get('input_ids')]
        if not valid_examples:
            return {"input_ids": torch.empty(0, 0, dtype=torch.long), "attention_mask": torch.empty(0, 0, dtype=torch.long)}
        
        orig_ids = [ex['input_ids'][:self.max_length] for ex in valid_examples]
        aug_ids = [ex.get('input_ids_aug', ex['input_ids'])[:self.max_length] for ex in valid_examples]
        
        batch = self.tokenizer.pad(
            [{"input_ids": ids} for ids in orig_ids + aug_ids],
            padding="max_length",            
            max_length=self.max_length,
            return_tensors="pt",
            pad_to_multiple_of=8                
        )
        return {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}


# ============================================================================
# MODEL AVEC SAVE_PRETRAINED MODIFIÉ
# ============================================================================
class ContrastiveModel(nn.Module):
    """Modèle pour apprentissage contrastif d'embeddings"""
    def __init__(self, model_to_train: nn.Module, projection_dim: int, temperature: float, base_model_id: str = None):
        super().__init__()
        self.model_to_train = model_to_train
        self.base_model_id = base_model_id
        config_obj = model_to_train.config.to_dict()
        hidden_size = config_obj.get("hidden_size")
        
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, projection_dim), nn.LayerNorm(projection_dim)
        )
        self.infonce_loss = InfoNCELoss(temperature)
        self._init_weights()

    def _init_weights(self):
        for module in self.projection_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None: nn.init.zeros_(module.bias)

    def _mean_pool(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).expand(hidden_states.size()).float()
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        model_output = self.model_to_train(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True
        )
        hidden_states = model_output.hidden_states[-1]
        pooled = self._mean_pool(hidden_states, attention_mask)
        embeddings = self.projection_head(pooled.to(torch.float32))
        loss_dict = self.infonce_loss(embeddings)
        return {"loss": loss_dict["loss"], "embeddings": embeddings, "metrics": loss_dict}
    
    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.model_to_train, "gradient_checkpointing_enable"):
            self.model_to_train.gradient_checkpointing_enable(**kwargs)

    def save_pretrained(self, output_dir: str):
        """Sauvegarde l'adaptateur LoRA et la tête de projection."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Sauvegarder le modèle PEFT (LoRA adapter + config)
        # Cette méthode sauvegarde automatiquement adapter_config.json et adapter_model.safetensors
        self.model_to_train.save_pretrained(output_dir)
        
        # Sauvegarder aussi la projection head
        torch.save(self.projection_head.state_dict(), os.path.join(output_dir, "projection_head.pt"))
        config = {"base_model_id": self.base_model_id,"projection_dim": self.projection_head[3].normalized_shape[0], "temperature": self.infonce_loss.temperature}
        json.dump(config, open(os.path.join(output_dir, "contrastive_config.json"), "w"))
        
        logger.info(f"✅ LoRA adapter (with adapter_config.json) and projection head saved to {output_dir}")

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str):
        """Charge le modèle complet depuis un checkpoint"""
        from peft import PeftModel
        
        config = json.load(open(os.path.join(checkpoint_dir, "contrastive_config.json")))
        
        # Reconstruire le modèle de base + LoRA
        base_model = AutoModelForCausalLM.from_pretrained(config["base_model_id"])
        base_model.lm_head = nn.Identity()  # Important !
        model_to_train = PeftModel.from_pretrained(base_model, checkpoint_dir)
        
        # Reconstruire le wrapper
        model = cls(model_to_train, config["projection_dim"], config["temperature"], config["base_model_id"])
        model.projection_head.load_state_dict(torch.load(os.path.join(checkpoint_dir, "projection_head.pt")))
        
        return model
# ============================================================================
# TRAINER
# ============================================================================
class ContrastiveTrainer(Trainer):
    """Trainer contrastif avec gestion propre des métriques et des callbacks."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_history = defaultdict(list)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if inputs["input_ids"].numel() == 0:
            return torch.tensor(0.0, device=self.args.device, requires_grad=True)
        outputs = model(**inputs)
        if "metrics" in outputs:
            for k, v in outputs["metrics"].items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    self.metric_history[k].append(v.item())
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        for name, history in self.metric_history.items():
            if history: logs[f"avg_{name}"] = np.mean(history[-100:])
        super().log(logs, *args, **kwargs)


# ============================================================================
# CHECKPOINTING & LOADING
# ============================================================================
def load_or_resume_from_checkpoint(args: argparse.Namespace, model: ContrastiveModel):
    """Charge les poids d'un checkpoint ou détermine si l'entraînement doit reprendre."""
    resume_from_checkpoint = None
    if args.load_checkpoint:
        logger.info(f"📂 Attempting to load from checkpoint: {args.load_checkpoint}")
        # Charger l'adaptateur LoRA
        try:
            model.model_to_train.load_adapter(args.load_checkpoint)
            logger.info("✅ LoRA adapter loaded successfully.")
        except Exception as e:
            logger.warning(f"⚠️ Could not load LoRA adapter. May be a full checkpoint. Error: {e}")

        # Charger la tête de projection
        proj_path = os.path.join(args.load_checkpoint, "projection_head.pt")
        if os.path.exists(proj_path):
            try:
                model.projection_head.load_state_dict(torch.load(proj_path, map_location="cpu"))
                logger.info("✅ Projection head loaded successfully.")
            except Exception as e:
                logger.error(f"❌ Failed to load projection head: {e}")
        else:
            logger.warning("⚠️ No projection_head.pt found in checkpoint.")

        # Vérifier si c'est un checkpoint de Trainer pour reprendre l'entraînement
        if os.path.exists(os.path.join(args.load_checkpoint, "trainer_state.json")):
            resume_from_checkpoint = args.load_checkpoint
            logger.info("✅ Full trainer state found. Resuming training.")
        else:
            logger.info("Checkpoint seems to be for inference only. Starting new training run.")
            
    return model, resume_from_checkpoint


# ============================================================================
# MAIN AVEC SAUVEGARDE FINALE MODIFIÉE
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Contrastive Training for Universal Embeddings")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to checkpoint to load weights or resume training from.")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--projection_dim", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--neptune_project", type=str, default=None)
    parser.add_argument("--neptune_api_token", type=str, default=None, help="Neptune API token")
    parser.add_argument("--neptune_run_name", type=str, default=None)
    
    args = parser.parse_args()

    set_seed(args.seed)
    accelerator = Accelerator(split_batches=True)
    logger.info(f"🚀 Starting contrastive training on {accelerator.device}")

    report_to = "none"
    if args.neptune_project and NEPTUNE_AVAILABLE and accelerator.is_main_process:
        api_token = args.neptune_api_token or os.getenv("NEPTUNE_API_TOKEN")
        if api_token:
            os.environ["NEPTUNE_API_TOKEN"] = api_token
            os.environ["NEPTUNE_PROJECT"] = args.neptune_project
            report_to = "neptune"
            logger.info(f"✅ Neptune configured for project: {args.neptune_project}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    
    load_kwargs = {"trust_remote_code": True}
    if args.load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        load_kwargs["torch_dtype"] = torch.bfloat16
    
    base_model = AutoModelForCausalLM.from_pretrained(args.model_id, attn_implementation="eager", **load_kwargs)
    base_model.config.use_cache = False
    base_model.lm_head = nn.Identity()

    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model_to_train = get_peft_model(base_model, lora_config)
    logger.info(f"✅ Created LoRA adapter with r={args.lora_r}")
    
    model = ContrastiveModel(model_to_train, args.projection_dim, args.temperature, args.model_id)

    resume_from_checkpoint = None
    if args.load_checkpoint:
        model, resume_from_checkpoint = load_or_resume_from_checkpoint(args, model)

    dataset = build_tokenized_dataset(args.dataset_path, args.max_length)
    collator = ContrastiveDataCollator(tokenizer, args.max_length)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        max_steps=args.max_steps, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine", logging_steps=args.logging_steps,
        save_steps=args.save_steps, save_total_limit=args.save_total_limit,
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        bf16=True, remove_unused_columns=False, seed=args.seed,
        report_to=report_to, run_name=args.neptune_run_name,
        ddp_find_unused_parameters=True, logging_first_step=True,
    )

    callbacks = [SaveProjectionCallback(model)]

    trainer = ContrastiveTrainer(
        model=model, args=training_args, train_dataset=dataset,
        data_collator=collator, callbacks=callbacks
    )

    logger.info("🔥 Starting contrastive training...")
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        logger.info("✅ Training completed!")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        return

    # SAUVEGARDE FINALE MODIFIÉE
    if accelerator.is_main_process:
        logger.info("💾 Saving final model...")
        
        # Utiliser la méthode save_pretrained du modèle wrapper
        # qui va appeler la bonne méthode pour sauvegarder PEFT
        model.save_pretrained(args.output_dir)
        
        # Sauvegarder le tokenizer séparément
        tokenizer.save_pretrained(args.output_dir)
        
        logger.info(f"🎉 Final model and artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()