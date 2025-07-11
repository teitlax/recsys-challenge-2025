from __future__ import annotations
import json
import logging
import torch
import textwrap
import os
import re
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

# Configure logging
torch.backends.cudnn.benchmark = True
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("PORTRAIT_LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# Constants
MODEL_NAME = "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"
#MODEL_NAME = "google/gemma-3-1b-it"
CTX_LIMIT = 3048
MAX_PROMPT_TOKENS = 400  # Réduit de 512 pour éviter OOM
GEN_TOKENS = 128
BATCH_SIZE = int(os.getenv("PORTRAIT_BATCH", "180")) 
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("La variable d'environnement HF_TOKEN n'est pas définie.")

# [COMPETITION_BRIEF reste identique]
COMPETITION_BRIEF = textwrap.dedent("""
[CONTEXT – ACM RecSys Challenge 2025]

You receive a **structured e-commerce user profile** (> 1 M SKUs).
Produce a professional and exhaustive **behavioural portrait** as a list of plain-text bullet points for stakeholders.

Open tasks:
 • **churn** – binary; churn if ≥ 1 purchase **and** ≥ 14 days inactivity  
 • **propensity_category** – multilabel; top-100 categories  
 • **propensity_sku**      – multilabel; top-100 SKUs  

Hidden tasks:
Exposé general signals: temporal habits, funnel stage, price sensitivity, popularity,
exploration vs purchase, etc.

Overall metric = 0.8 × AUROC + 0.1 × Novelty + 0.1 × Diversity.
Key User Behavior Insights:

Segmentation: 96% of users are non-buyers; only 4% purchase.
Purchase Path: 57% of purchases occur without a preceding add_to_cart event. The most common journey involves multiple page visits before a purchase.
User Loyalty: 77% of buyers are single-category loyal.
Timing: Average time from add_to_cart to buy is 32 minutes; from search to buy is over 2 hours.
Signal Strength: add_to_cart is a significantly stronger purchase predictor (0.53 correlation) than search (0.34).

### OUTPUT FORMAT (≤ 10 bullets, each on its own line)
- Each line must start with "- " (dash + space)
- No code, no functions, no code fences
- Plain English only

### EXAMPLES
**Good:**
- Category_XXX enthusiast with declining engagement.
- Key segment: SKU_XXX
- Evening shopper
- High price sensitivity
- Recommendation: Send 10% discount on top SKUs

**Bad:**
```python
def summarize_user_profile(...):
    ...
```

REMINDER:
You are an expert behavioural analyst.
Do NOT generate ANY code, functions, or code fences.
Be concise and human-readable.

— FIN —
""")


def _device_list() -> list[str]:
    """Liste des devices disponibles"""
    import torch
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["cpu"]


# PortraitGenerator class reste identique
class PortraitGenerator:
    """Utilise la chat-template officielle Gemma-3."""
    def __init__(self, device):
        import torch, logging, os
        from unsloth import FastModel

        self.device = device
        logging.getLogger(__name__).info("Loading %s on %s …", MODEL_NAME, device)

        self.model, tokenizer = FastModel.from_pretrained(
            MODEL_NAME,
            max_seq_length=CTX_LIMIT,
            load_in_4bit=True,          # ← off
            #torch_dtype=torch.float16,    # ← on
            device_map={"": device},
        )
        self.model.eval()
        self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)
        self.tpl = get_chat_template(tokenizer, "gemma-3")
        #self.base_tok = self.tpl.tokenizer
        self.base_tok = getattr(self.tpl, "tokenizer", self.tpl)

        self.system_header = (
            "You are an expert behavioural analyst.\n"
            "Do NOT generate ANY code, functions, or code fences.\n"
            "Be concise and human-readable.\n"
            "Terminate with '— FIN —'\n"
        )

    def _strip_rich_text(self, rt: str, keep_raw: bool = False, last_n: int = 30) -> str:
        """Nettoie le texte pour le portrait"""
        import re
        
        rt = re.sub(r'##\s*PORTRAIT\s*##.*?(?=##|$|\[END\])', '', rt, flags=re.DOTALL)
        if not keep_raw:
            rt = re.sub(r'##\s*RAW_SEQUENCE\s*##.*?(?=##|$|\[END\])', '', rt, flags=re.DOTALL)
        rt = rt.replace('[END]', '').strip()
        
        
        # Limite ajoutée pour éviter OOM
        if len(rt) > 6000:  # ~1500 tokens max
            rt = rt[:6000] + "\n[TRUNCATED FOR MEMORY]"
        
        return rt

    def _encode_batch(self, items):
        """Encode un batch de conversations"""
        conv_strings, cids = [], []
        for cid, rich in items:
            profile_txt = self._strip_rich_text(rich, keep_raw=False)
    
            token_ids = self.base_tok.encode(profile_txt, add_special_tokens=False)[:MAX_PROMPT_TOKENS]
            profile_txt = self.base_tok.decode(token_ids, skip_special_tokens=True)
    
            messages = [
                {
                    "role": "system",
                    "content": (
                        self.system_header.strip() + "\n\n" + COMPETITION_BRIEF.strip()
                    ),
                },
                {
                    "role": "user",
                    "content": f"[CLIENT_ID={cid}]\n{profile_txt}",
                },
            ]
    
            prompt_str = self.base_tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            conv_strings.append(prompt_str)
            cids.append(cid)
    
        batch = self.base_tok(
            conv_strings,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CTX_LIMIT - GEN_TOKENS,
        ).to(self.device)
    
        return batch, cids

    def _clean_portrait(self, raw_text: str) -> str:
        """Nettoie le portrait généré"""
        text = re.sub(r'```[\s\S]*?```', '', raw_text)
        text = text.split("— FIN —", 1)[0]
        
        bullets = [
            ln.strip() for ln in text.splitlines()
            if ln.strip().startswith("- ")
        ]
        
        if not bullets:
            bullets = ["- Behavioural signal too sparse to summarise."]
        
        return "\n".join(bullets) + "\n— FIN —"

    @torch.inference_mode()
    def generate_batch(self, items):
        batch, cids = self._encode_batch(items)
    
        eos_id = self.base_tok.eos_token_id or self.base_tok.encode("— FIN —", add_special_tokens=False)[0]
    
        generated = self.model.generate(
            **batch,
            max_new_tokens=GEN_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=eos_id,
        )
    
        prompt_lens = batch["attention_mask"].sum(dim=1)
    
        portraits = {}
        for idx, cid in enumerate(cids):
            gen_part = generated[idx, int(prompt_lens[idx]):]
            text = self.base_tok.decode(gen_part, skip_special_tokens=True)
            portraits[cid] = self._clean_portrait(text)
    
        return portraits


# ========== NOUVEAU : Worker fonction pour multiprocessing ==========
def _gpu_worker(args):
    """Worker qui s'exécute sur un GPU spécifique"""
    device, items = args
    
    # Set CUDA device pour ce processus
    if 'cuda' in device:
        gpu_id = device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        actual_device = 'cuda:0'  # Dans le processus, c'est toujours cuda:0
    else:
        actual_device = device
    
    logger.info(f"GPU Worker starting on {device} (actual: {actual_device}) for {len(items)} items")
    
    try:
        gen = PortraitGenerator(actual_device)
        results = gen.generate_batch(items)
        
        # Cleanup
        del gen.model
        del gen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info(f"GPU Worker on {device} completed {len(results)} portraits")
        return results
        
    except Exception as e:
        logger.error(f"GPU Worker on {device} failed: {e}", exc_info=True)
        return {cid: f"Error on {device}: {str(e)}\n— FIN —" for cid, _ in items}


# ========== NOUVELLE version parallèle de generate_portraits ==========
def generate_portraits(rich_texts: dict[int, str],
                      batch_size: int = BATCH_SIZE) -> dict[int, str]:
    """Version VRAIMENT parallèle qui utilise tous les GPUs simultanément"""
    
    items = list(rich_texts.items())
    devices = _device_list()
    
    if not devices:
        logger.error("No devices found (CPU/GPU). Exiting.")
        return {}
    if not items:
        logger.info("No items to process.")
        return {}
    
    logger.info(f"Starting parallel portrait generation: {len(items)} items on {len(devices)} devices")
    
    # Répartir les items entre les GPUs
    chunks_per_gpu = []
    items_per_gpu = len(items) // len(devices)
    remainder = len(items) % len(devices)
    
    start_idx = 0
    for i, device in enumerate(devices):
        # Distribuer équitablement + reste
        chunk_size = items_per_gpu + (1 if i < remainder else 0)
        end_idx = start_idx + chunk_size
        
        if chunk_size > 0:
            # Diviser en batches pour ce GPU
            gpu_chunks = []
            for j in range(start_idx, end_idx, batch_size):
                batch = items[j:min(j + batch_size, end_idx)]
                if batch:
                    gpu_chunks.append(batch)
            
            if gpu_chunks:
                chunks_per_gpu.append((device, gpu_chunks))
                logger.info(f"GPU {device}: {chunk_size} items in {len(gpu_chunks)} batches")
        
        start_idx = end_idx
    
    # Lancer les processus en parallèle
    portraits = {}
    
    with ProcessPoolExecutor(max_workers=len(devices)) as executor:
        # Préparer tous les jobs
        future_to_device = {}
        
        for device, chunks in chunks_per_gpu:
            # Concatener tous les chunks pour ce GPU
            all_items_for_gpu = []
            for chunk in chunks:
                all_items_for_gpu.extend(chunk)
            
            future = executor.submit(_gpu_worker, (device, all_items_for_gpu))
            future_to_device[future] = device
        
        # Collecter les résultats
        for future in as_completed(future_to_device):
            device = future_to_device[future]
            try:
                gpu_results = future.result()
                portraits.update(gpu_results)
                logger.info(f"Collected {len(gpu_results)} portraits from {device}")
            except Exception as e:
                logger.error(f"Failed to get results from {device}: {e}")
    
    logger.info(f"Portrait generation completed: {len(portraits)} portraits generated")
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Final CUDA cache cleared")
    
    return portraits
