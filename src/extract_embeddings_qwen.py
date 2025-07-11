#!/usr/bin/env python3
"""
Script d'extraction d'embeddings avec Qwen3-Embedding-8B
Optimisé pour exécution locale dans un notebook (lecture depuis un fichier local)
"""

import json
import zstandard as zstd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import gc
import io
import re
import argparse
import logging
from datetime import datetime
import sys
import os

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class QwenEmbeddingExtractor:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Chemin local vers le fichier d'entrée
        self.input_path = self.args.dataset_path
        
        self.output_dir = Path(f"embeddings/qwen3-8b/")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log configuration
        logger.info("=== CONFIGURATION ===")
        logger.info(f"Mode: {'DEBUG' if args.debug else 'PRODUCTION'}")
        logger.info(f"Model: {args.model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Max length: {args.max_length}")
        logger.info(f"Embedding dim: {args.embedding_dim}")
        logger.info(f"Input file: {self.input_path}")
        logger.info(f"Output dir: {self.output_dir.resolve()}/")

    def load_model(self):
        """Load model using sentence-transformers"""
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Loading model {self.args.model_name} with sentence-transformers...")
        
        # Configurer le token HuggingFace si disponible
        hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
            logger.info("HuggingFace authentication configurée")
        
        model_kwargs = {
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True, 
        }
        if torch.cuda.is_available():
            model_kwargs["device_map"] = {"": 0}
        
        tokenizer_kwargs = {"padding_side": "left"}
        
        self.sentence_model = SentenceTransformer(
            self.args.model_name,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs
        )
        self.model = self.sentence_model[0].auto_model
        self.tokenizer = self.sentence_model[0].tokenizer
        logger.info("Modèle chargé avec succès !")

    def read_local_jsonl_zst(self, limit=None):
        """Lit un fichier JSONL compressé en zst depuis le disque local"""
        logger.info(f"Ouverture du fichier local {self.input_path}...")
        with open(self.input_path, "rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                count = 0
                for line in text_stream:
                    if line.strip():
                        yield json.loads(line)
                        count += 1
                        if limit and count >= limit:
                            break

    def clean_text(self, text, client_id=None):
        """Remove client ID and clean text"""
        patterns = [
            r'\[CLIENT_\d+\]',
            r'CLIENT_\d+',
            r'client_id:\s*\d+',
            r'"client_id":\s*\d+,?',
            r'\'client_id\':\s*\d+,?'
        ]
        if client_id:
            patterns += [
                f'\\[CLIENT_{client_id}\\]',
                f'CLIENT_{client_id}\\b',
                f'\\b{client_id}\\b'
            ]
        cleaned = text
        for pat in patterns:
            cleaned = re.sub(pat, '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def get_detailed_instruct(self, task_desc, query):
        """Format instruction for Qwen3 embedding model"""
        return f'Instruct: {task_desc}\nQuery:{query}'

    def get_embeddings_batch(self, texts):
        """Generate embeddings using sentence-transformers"""
        try:
            task = 'Generate a dense behavioral representation for this user profile'
            instructed_texts = [self.get_detailed_instruct(task, t) for t in texts]
            embeddings = self.sentence_model.encode(
                instructed_texts,
                batch_size=self.args.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                device=self.device
            )
            emb_np = embeddings.astype(np.float16)
            cur_dim = emb_np.shape[1]
            if cur_dim != self.args.embedding_dim:
                if cur_dim < self.args.embedding_dim:
                    pad = np.zeros((emb_np.shape[0], self.args.embedding_dim - cur_dim), dtype=np.float16)
                    emb_np = np.concatenate([emb_np, pad], axis=1)
                    logger.warning(f"Padded embeddings from {cur_dim} to {self.args.embedding_dim}")
                else:
                    emb_np = emb_np[:, :self.args.embedding_dim]
                    logger.warning(f"Truncated embeddings from {cur_dim} to {self.args.embedding_dim}")
            norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
            emb_np = emb_np / (norms + 1e-8)
            return emb_np
        except Exception as e:
            logger.error(f"Error in get_embeddings_batch: {e}")
            return np.random.randn(len(texts), self.args.embedding_dim).astype(np.float16)

    def process_dataset(self):
        """Boucle principale de traitement"""
        self.load_model()
        limit = 5 if self.args.debug else None

        # Comptage
        logger.info("Comptage des enregistrements...")
        total = sum(1 for _ in self.read_local_jsonl_zst(limit=limit))
        logger.info(f"Vais traiter {total:,} enregistrements")

        all_ids, all_embs = [], []
        batch_texts, batch_ids = [], []
        processed, lengths = 0, []

        pbar = tqdm(self.read_local_jsonl_zst(limit=limit), total=total, desc="Extraction embeddings")
        for rec in pbar:
            cid, txt = rec['id'], rec['text']
            cleaned = self.clean_text(txt, cid)
            lengths.append(len(cleaned))
            batch_texts.append(cleaned)
            batch_ids.append(cid)

            if len(batch_texts) >= self.args.batch_size:
                embs = self.get_embeddings_batch(batch_texts)
                all_ids.extend(batch_ids)
                all_embs.append(embs)
                processed += len(batch_texts)
                pbar.set_postfix({
                    'processed': processed,
                    'batch_size': len(batch_texts),
                    'gpu_mem': f"{torch.cuda.memory_allocated()/1e9:.1f}GB" if torch.cuda.is_available() else "N/A"
                })
                batch_texts, batch_ids = [], []
                if processed % 10000 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Dernier batch
        if batch_texts:
            embs = self.get_embeddings_batch(batch_texts)
            all_ids.extend(batch_ids)
            all_embs.append(embs)
            processed += len(batch_texts)

        # Concaténation
        ids_arr = np.array(all_ids, dtype=np.int64)
        embs_arr = np.vstack(all_embs).astype(np.float16)

        logger.info("=== STATISTIQUES D'EXTRACTION ===")
        logger.info(f"Total traité   : {processed:,}")
        logger.info(f"IDs shape       : {ids_arr.shape}, dtype={ids_arr.dtype}")
        logger.info(f"Embeddings shape: {embs_arr.shape}, dtype={embs_arr.dtype}")
        logger.info(f"Longueur texte  : mean={np.mean(lengths):.0f}, std={np.std(lengths):.0f}")

        self.save_results(ids_arr, embs_arr)

    def save_results(self, client_ids, embeddings):
        """Sauvegarde locale des résultats"""
        logger.info("=== SAUVEGARDE DES RÉSULTATS ===")
        ids_path = self.output_dir / "client_ids.npy"
        emb_path = self.output_dir / "embeddings.npy"
        np.save(ids_path, client_ids)
        np.save(emb_path, embeddings)
        logger.info(f"Enregistré : {ids_path}, {emb_path}")

        # Archive optionnelle
        zip_path = self.output_dir / "submission.zip"
        import zipfile
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(ids_path, "client_ids.npy")
            zf.write(emb_path, "embeddings.npy")
        logger.info(f"Créé archive : {zip_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract embeddings with Qwen3 (local)")
    parser.add_argument('--model_name',   type=str,   default='Qwen/Qwen3-Embedding-8B')
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--max_length',   type=int,   default=2048)
    parser.add_argument('--embedding_dim',type=int,   default=2048)
    parser.add_argument("--dataset_path",      required=True)
    parser.add_argument('--debug',        action='store_true', help='Process only 5 records')
    args = parser.parse_args()

    logger.info("="*60)
    logger.info("QWEN3 EMBEDDING EXTRACTION (LOCAL)")
    logger.info("="*60)
    extractor = QwenEmbeddingExtractor(args)
    extractor.process_dataset()

if __name__ == "__main__":
    main()