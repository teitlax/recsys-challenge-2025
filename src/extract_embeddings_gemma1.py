#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import io
import json
import glob
import argparse
import logging
import zipfile
import zstandard as zstd
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
from accelerate import Accelerator

from train_gemma1 import ContrastiveModel

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def yield_tokenized_records(path: str, limit: Optional[int] = None):
    """
    Lit un JSONL.ZST local, décompresse et renvoie dicts.
    Si limit défini, stoppe après `limit` enregistrements par fichier.
    """
    log.info(f"📂 Ouverture du fichier local {path}")
    dctx = zstd.ZstdDecompressor()
    count = 0
    with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
        for line in io.TextIOWrapper(reader, encoding="utf-8", errors="ignore"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield rec
            count += 1
            if limit and count >= limit:
                log.info(f"🛑 Mode DEBUG : arrêt après {count} enregistrements")
                break

class TokenizedDataset(IterableDataset):
    """
    IterableDataset qui parcourt tous les fichiers matching `pattern`,
    et renvoie dicts {'client_id': int, 'input_ids': List[int]}.
    """
    def __init__(self, pattern: str, limit: Optional[int] = None):
        super().__init__()
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No files for pattern {pattern}")
        self.limit = limit
        log.info(f"🔍 Fichiers trouvés: {self.files}")
        if self.limit:
            log.info(f"🐛 Mode DEBUG activé : max {self.limit} recs/fichier")

    def __iter__(self):
        for fpath in self.files:
            log.info(f"🔄 Traitement de {fpath}")
            for rec in yield_tokenized_records(fpath, limit=self.limit):
                cid = rec.get("client_id")
                ids = rec.get("input_ids", [])
                if cid is None or not isinstance(ids, list) or not ids:
                    continue
                yield {
                    "client_id": int(cid),
                    "input_ids": ids
                }

class Collator:
    """Pad / batchify tout en passant le client_id."""
    def __init__(self, tokenizer, max_len: int):
        self.tok = tokenizer
        self.max_len = max_len

    def __call__(self, batch: List[Dict]):
        cids = [b["client_id"] for b in batch]
        seqs = [b["input_ids"][: self.max_len] for b in batch]
        padded = self.tok.pad(
            [{"input_ids": x} for x in seqs],
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "client_ids":     torch.tensor(cids, dtype=torch.int64),
            "input_ids":      padded["input_ids"].long(),
            "attention_mask": padded["attention_mask"].long(),
        }

class InferenceWrapper(nn.Module):
    """Enveloppe pour appeler votre ContrastiveModel."""
    def __init__(self, contrastive_model):
        super().__init__()
        self.model = contrastive_model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)["embeddings"]

def merge_and_create_submission(
    output_dir: str,
    world_size: int
) -> str:
    """
    Concatène client_ids et embeddings par rank puis archive.
    """
    output_path = Path(output_dir)
    all_ids, all_embs = [], []

    for rank in range(world_size):
        id_file  = output_path / f"client_ids_rank{rank}.npy"
        emb_file = output_path / f"embeddings_rank{rank}.npy"
        if id_file.exists() and emb_file.exists():
            all_ids.append (np.load(id_file))
            all_embs.append(np.load(emb_file))

    ids  = np.concatenate(all_ids)
    embs = np.concatenate(all_embs, axis=0)
    
    log.info(f"Total embeddings avant dédupplication : {len(ids)}")
    unique_ids, unique_indices = np.unique(ids, return_index=True)
    if len(unique_ids) != len(ids):
        log.warning(f"Doublons détectés : {len(ids)} → {len(unique_ids)} uniques")
        ids = unique_ids
        embs = embs[unique_indices]
        
     # Tri par client_id
    idx = np.argsort(ids)
    ids, embs = ids[idx], embs[idx]
    embs = embs.astype(np.float16)

    # Sauvegarde finale
    np.save(output_path/"client_ids.npy", ids)
    np.save(output_path/"embeddings.npy", embs.astype(np.float16))
    submission = output_path/"submission.zip"
    with zipfile.ZipFile(submission, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_path/"client_ids.npy",   "client_ids.npy")
        zf.write(output_path/"embeddings.npy",   "embeddings.npy")
    return str(submission)

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",         required=True)
    parser.add_argument("--projection_dim",   type=int,   required=True)
    parser.add_argument("--temperature",      type=float, required=True)
    parser.add_argument("--checkpoint_dir",   required=True)
    parser.add_argument("--dataset_path",     required=True,
                        help="Glob pattern, ex: output_features_new/*.jsonl.zst")
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--max_length",       type=int,   default=2048)
    parser.add_argument("--batch_size",       type=int,   default=256)
    parser.add_argument("--load_in_4bit",     action="store_true")
    parser.add_argument("--create_submission",action="store_true")
    parser.add_argument("--debug",            action="store_true",
                        help="Mode debug → 5 enregistrements/fichier")
    args = parser.parse_args()

    limit = 5 if args.debug else None

    log.info("=== CONFIGURATION ===")
    for k,v in vars(args).items():
        log.info(f"{k:15}: {v}")

    accelerator = Accelerator()
    device = accelerator.device
    log.info(f"Using device: {device}")

    # ── Tokenizer & Model ───────────────────────────────────────────────
    log.info(f"Loading tokenizer {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {}
    if args.load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["torch_dtype"] = torch.bfloat16

    log.info(f"Loading backbone {args.model_id}")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        attn_implementation="eager",
        **load_kwargs
    )
    base.config.use_cache = False
    base.lm_head = nn.Identity()

    peft_model = PeftModel.from_pretrained(
        base,
        args.checkpoint_dir,
        trust_remote_code=True
    )
    contrastive = ContrastiveModel(
        peft_model,
        args.projection_dim,
        args.temperature,
        args.model_id
    )
    contrastive.projection_head.load_state_dict(
        torch.load(
            Path(args.checkpoint_dir)/"projection_head.pt",
            map_location="cpu"
        )
    )
    inference_model = InferenceWrapper(contrastive)
    inference_model = accelerator.prepare(inference_model)

    # ── DataLoader ────────────────────────────────────────────────────────
    ds   = TokenizedDataset(args.dataset_path, limit=limit)
    coll = Collator(tokenizer, args.max_length)
    dl   = DataLoader(
        ds,
        batch_size=args.batch_size,
        collate_fn=coll,
        num_workers=2,
        pin_memory=False
    )
    dl = accelerator.prepare(dl)

    # ── Extraction ────────────────────────────────────────────────────────
    id_chunks, emb_chunks = [], []

    log.info("Démarrage de l'extraction...")
    with torch.no_grad():
        for step, batch in enumerate(dl, start=1):
            out = inference_model(batch["input_ids"], batch["attention_mask"])
            gathered_ids  = accelerator.gather(batch["client_ids"])
            gathered_embs = accelerator.gather(out)

            rank = accelerator.process_index
            id_chunks.append (gathered_ids.cpu().numpy())
            emb_chunks.append(gathered_embs.cpu().numpy().astype(np.float16))

            if args.debug and step == 1:
                log.info("🐛 DEBUG : un seul batch extrait, j’arrête.")
                break

    # ── Sauvegarde par rank ────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    rank = accelerator.process_index
    np.save(Path(args.output_dir)/f"client_ids_rank{rank}.npy",   np.concatenate(id_chunks))
    np.save(Path(args.output_dir)/f"embeddings_rank{rank}.npy",   np.concatenate(emb_chunks, axis=0))

    accelerator.wait_for_everyone()

    # ── Merge & submission ────────────────────────────────────────────────
    if accelerator.is_main_process and args.create_submission:
        log.info("Fusion des ranks et création de la soumission…")
        submission = merge_and_create_submission(args.output_dir, accelerator.num_processes)
        log.info(f"🗳 Submission prête : {submission}")

if __name__ == "__main__":
    main()
