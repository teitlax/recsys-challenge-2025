"""
### 1. Features creation
"""

import os
import sys
import gc
import json
import time
import pickle
import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm.auto import tqdm
from collections import defaultdict

# --- Paths ---
DATA_DIR = "ubc_data"
CACHE_DIR = "ubc_data/cache_v3"
OUTPUT_DIR = "output_features/gemma12b"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Available CPUs: {mp.cpu_count()}")
print(f"Output directory: {OUTPUT_DIR}")
    
def generate_complete_features_batch(client_batch: List[int], batch_id: int) -> Dict:
        """
        Full memory version that loads all data into RAM, similar to the previous implementation.
        """
        import signal
        import json
        from ubm.text_representation_v3 import (
            AdvancedUBMGenerator,
            _build_rich_text,
            top_co_pairs,
            top_co_categories,
            cart_conversion_stats,
            RAW_SEQUENCE_LAST_EVENTS,
            MAX_RICH_TOKENS,
            TOP_FEATURES_PER_SECTION,
            IMPLICIT_WEIGHT_REPEAT,
            ChurnPropensityFeatureExtractor
        )
        from collections import defaultdict
        import time

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Batch {batch_id} timeout!")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(3600)  # 1 hour timeout

        print(f"[Batch {batch_id}] Starting {len(client_batch)} clients...", flush=True)
        start_time = time.time()

        try:
            # Option 1: Load from the 1M clients cache if it exists
            cache_1m = Path(CACHE_DIR) / "events_1m_clients.parquet"
            if cache_1m.exists():
                print(f"[Batch {batch_id}] Loading cached 1M events...", flush=True)
                events_df = pl.read_parquet(cache_1m)
                print(f"[Batch {batch_id}] Loaded {events_df.height:,} events from cache", flush=True)

                # Filter for the clients in the current batch
                events_df = events_df.filter(pl.col("client_id").is_in(client_batch))
                print(f"[Batch {batch_id}] Filtered to {events_df.height:,} events for batch", flush=True)

                # Initialize the generator and provide the data
                gen = AdvancedUBMGenerator(DATA_DIR, CACHE_DIR, debug_mode=False)
                gen.events_df = events_df

                # Load stats from ubc_data if available
                stats_cache_path = Path("ubc_data/cache_v3")
                if stats_cache_path.exists():
                    print(f"[Batch {batch_id}] Loading stats from ubc_data...", flush=True)
                    try:
                        import json
                        with open(stats_cache_path / 'global_stats.json', 'r') as f:
                            gen.global_stats = json.load(f)
                        if (stats_cache_path / 'product_popularity.parquet').exists():
                            gen.product_popularity = pl.read_parquet(stats_cache_path / 'product_popularity.parquet')
                        if (stats_cache_path / 'category_popularity.parquet').exists():
                            gen.category_popularity = pl.read_parquet(stats_cache_path / 'category_popularity.parquet')
                        with open(stats_cache_path / 'sku_properties_dict.pkl', 'rb') as f:
                            gen.sku_properties_dict = pickle.load(f)
                        if (stats_cache_path / 'user_segments.json').exists():
                            with open(stats_cache_path / 'user_segments.json', 'r') as f:
                                gen.user_segments = json.load(f)
                        print(f"[Batch {batch_id}] Stats loaded from ubc_data", flush=True)
                    except Exception as e:
                        print(f"[Batch {batch_id}] Failed to load some stats: {e}", flush=True)

            else:
                # Option 2: Load normally (all into memory)
                print(f"[Batch {batch_id}] No cache, loading data normally...", flush=True)
                gen = AdvancedUBMGenerator(DATA_DIR, CACHE_DIR, debug_mode=False)
                os.environ["SKIP_URL_GRAPH"] = "1"

                # Force loading into memory
                gen.load_data(use_cache=True, relevant_client_ids=None)  # None = load ALL

                if gen.events_df is not None:
                    print(f"[Batch {batch_id}] Loaded {gen.events_df.height:,} total events", flush=True)
                    # Filter for the current batch
                    gen.events_df = gen.events_df.filter(pl.col("client_id").is_in(client_batch))
                    print(f"[Batch {batch_id}] Filtered to {gen.events_df.height:,} events for batch", flush=True)

            # Initialize extractors
            gen._extractors = {}
            from ubm.text_representation_v3 import (
                TemporalFeatureExtractor, SequenceFeatureExtractor, 
                GraphFeatureExtractor, IntentFeatureExtractor,
                PriceFeatureExtractor, SocialFeatureExtractor,
                NameEmbeddingExtractor, TopSKUFeatureExtractor,
                TopCategoryFeatureExtractor, ChurnPropensityFeatureExtractor,
                CustomBehaviorFeatureExtractor
            )

            gen._extractors = {
                'temporal': TemporalFeatureExtractor(gen),
                'sequence': SequenceFeatureExtractor(gen),
                'graph': GraphFeatureExtractor(gen),
                'intent': IntentFeatureExtractor(gen),
                'price': PriceFeatureExtractor(gen),
                'social': SocialFeatureExtractor(gen),
                'name_embedding': NameEmbeddingExtractor(gen),
                'churn_propensity': ChurnPropensityFeatureExtractor(gen),
            }

            if hasattr(gen, 'top_skus') and gen.top_skus:
                gen._extractors['top_sku'] = TopSKUFeatureExtractor(gen)
            if hasattr(gen, 'top_categories') and gen.top_categories:
                gen._extractors['top_category'] = TopCategoryFeatureExtractor(gen)
            if hasattr(gen, 'sku_cluster_map'):
                gen._extractors['custom_behavior'] = CustomBehaviorFeatureExtractor(gen)

            print(f"[Batch {batch_id}] Setup completed in {time.time()-start_time:.1f}s", flush=True)
            print(f"[Batch {batch_id}] Available extractors: {list(gen._extractors.keys())}", flush=True)

            # --- Processing ---
            results = {}
            now = datetime.now()
            process_start = time.time()

            for idx, cid in enumerate(client_batch):
                if idx > 0 and idx % 50 == 0:
                    elapsed = time.time() - process_start
                    rate = idx / elapsed if elapsed > 0 else 0
                    print(f"[Batch {batch_id}] Progress: {idx}/{len(client_batch)} "
                          f"({elapsed:.1f}s, {rate:.1f} clients/s)", flush=True)

                try:
                    # Get events directly from the in-memory events_df
                    events = gen.events_df.filter(pl.col('client_id') == cid)

                    if events.height == 0:
                        results[cid] = {
                            "status": "no_data",
                            "profile": {"client_id": cid, "error": "No activity"},
                            "rich_text": f"[CLIENT_{cid}]\nNo activity data.",
                            "json_str": json.dumps({"profile": {"client_id": cid, "error": "No activity"}}, ensure_ascii=False)
                        }
                        continue

                    # Overview Section
                    seg = gen.get_client_segment(cid)
                    user_type = "buyer" if seg.get("buyers") else "browser"
                    overview_sec = [f"[CLIENT_{cid}]", f"User Type: {user_type}"]
                    seg_list = sorted(s for s, active in seg.items() if active and s not in {"buyers", "non_buyers"})
                    if seg_list:
                        overview_sec.append("Segments: " + ", ".join(seg_list))

                    section_map = defaultdict(list)
                    section_map["OVERVIEW"].extend(overview_sec)
                    features_json = []

                    ex_to_sec = {
                        "temporal": "TEMPORAL", "sequence": "SEQUENCE", "social": "SOCIAL",
                        "price": "PRICE", "intent": "OVERVIEW", "graph": "CUSTOM",
                        "name_embedding": "CUSTOM", "custom_behavior": "CUSTOM",
                        "churn_propensity": "CHURN_PROPENSITY", "top_sku": "SKU_PROPENSITY",
                        "top_category": "CAT_PROPENSITY",
                    }

                    # Extract features
                    for ex_name, extractor in gen._extractors.items():
                        if extractor is None: continue
                        tgt_sec = ex_to_sec.get(ex_name, "CUSTOM")
                        try:
                            feats = extractor.extract_features(cid, events, now)
                            repeat = IMPLICIT_WEIGHT_REPEAT if ex_name in ("top_sku", "top_category", "churn_propensity") else 1
                            for ft in feats:
                                section_map[tgt_sec].extend([ft] * repeat)
                                features_json.append({"type": ex_name, "value": ft})
                        except Exception as err:
                            section_map[tgt_sec].append(f"{ex_name}-error")

                    # Behavioral metrics
                    section_map["CUSTOM"].extend(top_co_pairs(events))
                    section_map["CUSTOM"].extend(top_co_categories(events))
                    section_map["CUSTOM"].extend(cart_conversion_stats(events))

                    # Compact metrics
                    try:
                        compact_tags = gen._compute_compact_metrics(events)
                        section_map["PROP_SUBSET_STATS"].extend(compact_tags)
                        features_json.extend([{"type": "compact", "value": tag} for tag in compact_tags])
                    except Exception: pass

                    try:
                        extra_tags = gen._compute_extra_short_metrics(cid, events, now)
                        section_map["PROP_SUBSET_STATS"].extend(extra_tags)
                        features_json.extend([{"type": "extra", "value": tag} for tag in extra_tags])
                    except Exception: pass

                    # History
                    recent_cut = now - timedelta(days=14)
                    medium_cut = now - timedelta(days=90)

                    recent_txt = gen._generate_detailed_events_text(events.filter(pl.col("timestamp") >= recent_cut), limit=30)
                    medium_txt = gen._generate_summarized_events_text(
                        events.filter((pl.col("timestamp") >= medium_cut) & (pl.col("timestamp") < recent_cut))
                    )
                    hist_txt = gen._generate_aggregated_events_text(events.filter(pl.col("timestamp") < medium_cut))

                    section_map["TARGET_WINDOW_14D"].append(recent_txt if recent_txt != "No recent activity." else "No activity in last 14 days")
                    if medium_txt != "No medium-term activity.": section_map["SEQUENCE"].append(medium_txt)
                    if hist_txt != "No historical activity.": section_map["CUSTOM"].append(hist_txt)

                    # Raw sequence
                    try: raw_seq = gen._generate_raw_sequence(events, max_events=100)
                    except Exception: raw_seq = ""

                    section_map["CUSTOM"] = list(dict.fromkeys(section_map["CUSTOM"])) # Deduplicate

                    # Build rich text representation
                    try:
                        rich_text = _build_rich_text(
                            section_map=dict(section_map), max_tokens=MAX_RICH_TOKENS,
                            implicit_repeat=IMPLICIT_WEIGHT_REPEAT, top_per_section=TOP_FEATURES_PER_SECTION,
                            shuffle_seed=cid, use_markers=True
                        )
                    except Exception:
                        rich_text = "\n\n".join(f"## {sec} ##\n" + "\n".join(lines[:20]) for sec, lines in section_map.items())

                    # Compile final profile
                    profile = {
                        "client_id": cid,
                        "overview": {"user_type": user_type, "segments": seg_list},
                        "features": features_json[:100],
                        "behavioral_metrics": {
                            "co_pairs": top_co_pairs(events)[:5],
                            "category_pairs": top_co_categories(events)[:5],
                            "cart_conversion": cart_conversion_stats(events)[:5],
                        },
                        "recent_activity": recent_txt[:500],
                        "medium_term_summary": medium_txt[:500],
                        "historical_aggregates": hist_txt[:500],
                        "raw_sequence": raw_seq[:1000],
                    }

                    results[cid] = {
                        "status": "success", "profile": profile, "rich_text": rich_text,
                        "json_str": json.dumps({"profile": profile, "rich_text": rich_text}, ensure_ascii=False)
                    }

                except Exception as e:
                    error_str = str(e)[:500]
                    results[cid] = {
                        "status": "error", "error": error_str,
                        "profile": {"client_id": cid, "error": error_str},
                        "rich_text": f"ERROR: {str(e)[:200]}",
                        "json_str": json.dumps({"client_id": cid, "error": error_str}, ensure_ascii=False)
                    }

            signal.alarm(0) # Disable timeout

            total_time = time.time() - start_time
            print(f"[Batch {batch_id}] Completed {len(results)} clients in {total_time:.1f}s "
                  f"({len(results)/total_time:.1f} clients/s)", flush=True)

            del gen
            gc.collect()
            return results

        except Exception as e:
            print(f"[Batch {batch_id}] BATCH FAILED: {e}", flush=True)
            signal.alarm(0)
            import traceback
            traceback.print_exc()
            return {cid: {"status": "batch_error", "error": str(e)[:500]} for cid in client_batch}
        
# --- Thread Configuration ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["POLARS_MAX_THREADS"] = "8"
os.environ["SKIP_URL_GRAPH"] = "1"

# ── TEST MODE ────────────────────────────────────────────────────────────
TEST_MODE = True
TEST_SIZE = 5  # nombre de clients de test en debug

# Charge tous les client_ids
client_ids = np.load(f"{DATA_DIR}/input/relevant_clients.npy").astype(int)
print(f"Total clients: {len(client_ids):,}")
if TEST_MODE:
    # on fixe la liste des TEST clients, et on tronque à TEST_SIZE
    TEST_CLIENT_IDS = sorted(client_ids[:TEST_SIZE].tolist())
    client_ids = np.array(TEST_CLIENT_IDS, dtype=int)
    print(f"🐛 DEBUG MODE: on ne traite que {TEST_SIZE} clients → {TEST_CLIENT_IDS}")
else:
    TEST_CLIENT_IDS = None
# ============================================

# --- Generation Configuration ---
BATCH_SIZE = 1000
N_WORKERS = min(8, mp.cpu_count() // 2)

print("Configuration:")
print(f"- Batch size: {BATCH_SIZE}")
print(f"- Workers: {N_WORKERS}")
print(f"- Total batches: {len(client_ids) // BATCH_SIZE + 1}")

# Create batches
batches = [(client_ids[i:i+BATCH_SIZE].tolist(), i//BATCH_SIZE) 
           for i in range(0, len(client_ids), BATCH_SIZE)]

print(f"\nProcessing {len(batches)} batches of {BATCH_SIZE} clients...")

# ============================================

# --- Parallel Execution ---
all_results = {}
failed_clients = []
start_time = time.time()

with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
    future_to_batch = {executor.submit(generate_complete_features_batch, batch[0], batch[1]): batch for batch in batches}

    with tqdm(total=len(client_ids), desc="Generating COMPLETE features") as pbar:
        for future in as_completed(future_to_batch):
            batch_data = future_to_batch[future]
            batch_clients, batch_id = batch_data

            try:
                batch_results = future.result() # Default timeout handled by signal alarm
                all_results.update(batch_results)

                batch_errors = sum(1 for r in batch_results.values() if r.get("status") != "success")
                if batch_errors > 0:
                    print(f"\n⚠️ Batch {batch_id}: {batch_errors} errors encountered")

                pbar.update(len(batch_clients))

            except Exception as e:
                print(f"\n❌ Batch {batch_id} failed entirely: {e}")
                failed_clients.extend(batch_clients)
                pbar.update(len(batch_clients))

elapsed = time.time() - start_time
print(f"\n✅ COMPLETE features generated in {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"Success count: {sum(1 for r in all_results.values() if r.get('status') == 'success')}")
print(f"Error count: {sum(1 for r in all_results.values() if r.get('status') != 'success')}")
print(f"Clients from failed batches: {len(failed_clients)}")

# ============================================

# --- Save Results ---
save_path = f"{OUTPUT_DIR}/complete_features_{len(client_ids)}_clients.pkl"
print(f"Saving results to {save_path}...")

with open(save_path, 'wb') as f:
    pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)

total_size = sum(len(r.get('json_str', '')) for r in all_results.values())
print(f"Total JSON size: {total_size/1024/1024:.1f} MB")

# ============================================

# --- Verification ---
print("\nVerifying V3 Feature Completeness:")
print("=" * 60)

expected_sections = [
    "OVERVIEW", 
    "CHURN_PROPENSITY",    
    "TARGET_WINDOW_14D",    
    "TEMPORAL", 
    "SEQUENCE", 
    "PRICE", 
    "SOCIAL", 
    "SKU_PROPENSITY",      
    "CAT_PROPENSITY",      
    "PROP_SUBSET_STATS",    
    "CUSTOM"
]

sample_results = [r for r in all_results.values() if r.get('status') == 'success'][:5]

for i, result in enumerate(sample_results):
    print(f"\nClient Sample {i+1}:")
    rich_text = result.get('rich_text', '')

    # Verify sections
    for section in expected_sections:
        if f"## {section} ##" in rich_text:
            content_start = rich_text.find(f"## {section} ##")
            content_end = rich_text.find("##", content_start + 4) if "##" in rich_text[content_start+4:] else len(rich_text)
            line_count = rich_text[content_start:content_end].count('\n')
            print(f"  ✓ {section} ({line_count} lines)")
        else:
            print(f"  ✗ {section} MISSING!")

    # Verify markers
    print("\n  Section Markers:")
    markers = ["[PROFILE]", "[CHURN]", "[RECENT]", "[TIME]", "[SEQ]", 
               "[PRICE]", "[SOCIAL]", "[SKU]", "[CAT]", "[STATS]", "[MISC]", "[END]"]
    for marker in markers:
        if marker in rich_text:
            print(f"  ✓ {marker}")

# ============================================

# --- Prepare for LLM Portraits ---
texts_for_portraits = {}
profiles_dict = {}

for cid, result in all_results.items():
    if result.get('status') == 'success':
        json_data = json.loads(result['json_str'])
        rich_text = json_data.get('rich_text', '')

        if rich_text:
            texts_for_portraits[cid] = rich_text
            profiles_dict[cid] = json_data.get('profile', {})

print(f"\nTexts ready for LLM portraits: {len(texts_for_portraits)}")
print(f"Average length: {np.mean([len(t) for t in texts_for_portraits.values()]):.0f} chars")

# Save artifacts
texts_path = f"{OUTPUT_DIR}/texts_for_portraits_{len(texts_for_portraits)}.pkl"
profiles_path = f"{OUTPUT_DIR}/profiles_{len(profiles_dict)}.pkl"

with open(texts_path, 'wb') as f:
    pickle.dump(texts_for_portraits, f)
with open(profiles_path, 'wb') as f:
    pickle.dump(profiles_dict, f)

print(f"Saved artifacts:")
print(f"- Texts: {texts_path}")
print(f"- Profiles: {profiles_path}")

"""
### 2. Adding of RAW SEQUENCE with fix
"""

import pickle
import json
import numpy as np
from pathlib import Path
import random
import time
from datetime import datetime
import shutil

# --- Paths ---
OUTPUT_DIR = Path("output_features/gemma12b")
COMPLETE_FEATURES_PATH = OUTPUT_DIR / f"complete_features_{TEST_SIZE}_clients.pkl"
TEXTS_PATH = OUTPUT_DIR / f"texts_for_portraits_{TEST_SIZE}.pkl"

# --- Configuration ---
MAX_EVENTS_TO_SHOW = 1000
DEFAULT_NO_EVENTS_TEXT = "No event sequence available for this client."
PROGRESS_INTERVAL = 5000  # Display progress every 5000 clients

def truncate_raw_sequence(raw_seq: str, max_events: int = 100) -> str:
    """
    Truncates the raw event sequence to keep only the last N events.
    """
    if not raw_seq or not raw_seq.strip():
        return DEFAULT_NO_EVENTS_TEXT

    events = [e.strip() for e in raw_seq.split("</s>") if e.strip()]

    if not events:
        return DEFAULT_NO_EVENTS_TEXT

    if len(events) > max_events:
        # Keep the last max_events items
        events = events[-max_events:]
        truncated_seq = f"... (truncated to last {max_events} events)\n" + "</s>".join(events)
    else:
        truncated_seq = "</s>".join(events)

    return truncated_seq

print("=== OPTIMIZED ADDITION OF RAW_SEQUENCE SECTION ===")
print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")

# 1. Quick test on a sample
print("\nRunning a test on a 100-client sample first...")
with open(COMPLETE_FEATURES_PATH, 'rb') as f:
    all_results = pickle.load(f)

sample_clients = random.sample(list(all_results.keys()), min(100, len(all_results)))
needs_correction_sample_count = 0

for cid in sample_clients:
    result = all_results.get(cid, {})
    if result.get('status') == 'success':
        if '## RAW_SEQUENCE ##' not in result.get('rich_text', ''):
            needs_correction_sample_count += 1

print(f"Sample analysis: {needs_correction_sample_count}/100 clients require correction.")

if needs_correction_sample_count == 0:
    print("✅ All clients in the sample already have the RAW_SEQUENCE section. Exiting.")
    exit()

# # 2. Ask for confirmation
# response = input(f"\nProceed to correct ~{len(all_results):,} clients? (y/n): ")
# if response.lower() != 'y':
#     print("Operation cancelled by user.")
#     exit()

# 3. Load texts_for_portraits
print("\nLoading portrait texts...")
with open(TEXTS_PATH, 'rb') as f:
    texts_for_portraits = pickle.load(f)

# 4. Process all clients with detailed progress
print(f"\nStarting correction for {len(all_results):,} clients...")
start_time = time.time()

corrected_count = 0
no_raw_seq_count = 0
already_correct_count = 0
error_count = 0
last_print_time = time.time()

client_items = list(all_results.items())
total_clients = len(client_items)

for i, (cid, result) in enumerate(client_items):
    # Display progress
    if i > 0 and (i % PROGRESS_INTERVAL == 0 or time.time() - last_print_time > 30):
        elapsed = time.time() - start_time
        rate = i / elapsed if elapsed > 0 else 0
        eta_seconds = (total_clients - i) / rate if rate > 0 else 0

        print(f"Progress: {i}/{total_clients} ({i/total_clients*100:.1f}%) | "
              f"Rate: {rate:.0f} clients/s | "
              f"ETA: {eta_seconds/60:.1f} min | "
              f"Corrected: {corrected_count}")
        last_print_time = time.time()

    try:
        if result.get('status') != 'success':
            continue

        rich_text = result.get('rich_text', '')

        # Skip if the section already exists
        if '## RAW_SEQUENCE ##' in rich_text:
            already_correct_count += 1
            continue

        # Prepare the sequence for insertion
        profile = result.get('profile', {})
        raw_seq = profile.get('raw_sequence', '')

        if not raw_seq:
            no_raw_seq_count += 1
            raw_seq_to_add = DEFAULT_NO_EVENTS_TEXT
        else:
            raw_seq_to_add = truncate_raw_sequence(raw_seq, MAX_EVENTS_TO_SHOW)

        # Build the new section
        raw_section = f"\n## RAW_SEQUENCE ##\n```\n{raw_seq_to_add}\n```"

        # Insert the new section before the [END] marker
        if '[END]' in rich_text:
            new_rich_text = rich_text.replace('[END]', raw_section + '\n[END]')
        else:
            new_rich_text = rich_text + raw_section

        # Update the main results dictionary
        result['rich_text'] = new_rich_text

        # Update the corresponding JSON string
        try:
            json_data = json.loads(result['json_str'])
            json_data['rich_text'] = new_rich_text
            result['json_str'] = json.dumps(json_data, ensure_ascii=False)
        except (json.JSONDecodeError, KeyError):
            # If JSON processing fails, the rich_text is still updated
            pass

        # Update the texts_for_portraits dictionary
        if cid in texts_for_portraits:
            texts_for_portraits[cid] = new_rich_text

        corrected_count += 1

    except Exception as e:
        error_count += 1
        if error_count < 10:
            print(f"Error processing client {cid}: {e}")

# 5. Final Summary
total_time = time.time() - start_time
print(f"\n=== PROCESSING COMPLETE in {total_time/60:.1f} minutes ===")
print(f"- Already correct: {already_correct_count:,}")
print(f"- Successfully corrected: {corrected_count:,}")
print(f"  (Including {no_raw_seq_count:,} clients with no raw sequence)")
print(f"- Errors: {error_count:,}")
print(f"- Average rate: {total_clients/total_time:.0f} clients/s")

# 6. Quick Verification
print("\nVerifying a random sample of 5 clients:")
for cid in random.sample(list(all_results.keys()), 5):
    result = all_results.get(cid, {})
    if result.get('status') == 'success':
        has_raw_section = '## RAW_SEQUENCE ##' in result.get('rich_text', '')
        print(f"  Client {cid}: RAW_SEQUENCE section present = {has_raw_section}")

# 7. Save Corrected Files
if corrected_count > 0:
    print(f"\n💾 {corrected_count} corrections to be saved.")
#     response = input("Save changes to disk? (y/n): ")

#     if response.lower() == 'y':
        # Create backups
    print("Creating backup files...")
    shutil.copy(COMPLETE_FEATURES_PATH, str(COMPLETE_FEATURES_PATH) + '.backup')
    shutil.copy(TEXTS_PATH, str(TEXTS_PATH) + '.backup')

    # Save the updated data
    print("Saving corrected files...")
    with open(COMPLETE_FEATURES_PATH, 'wb') as f:
        pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(TEXTS_PATH, 'wb') as f:
        pickle.dump(texts_for_portraits, f)

    print("✅ Save complete!")
    print(f"End time: {datetime.now().strftime('%H:%M:%S')}")
# else:
#     print("\n✅ No corrections were needed. No files were changed.")

## PORTRAITS ## 
import os, sys, gc, pickle, subprocess, time, argparse, logging, gzip
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ====== PARAMÈTRES PAR DÉFAUT ======
OUTPUT_DIR       = Path("output_features/gemma12b")
TEXTS_FILE       = OUTPUT_DIR / f"texts_for_portraits_{TEST_SIZE}.pkl"
BATCH_SIZE       = 180          # nb de textes envoyés simultanément au modèle
CHECKPOINT_EVERY = 100         # batches avant snapshot
DRY_RUN_SIZE     = 6       # nombre de clients en mode --dry-run
# ===================================

def setup_logging(run_id: str) -> None:
    log_dir = OUTPUT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    logfile = log_dir / f"main_{run_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)],
    )


def gpu_list() -> list[dict]:
    n = torch.cuda.device_count()
    if n == 0:
        logging.error("Aucun GPU détecté — arrêt.")
        sys.exit(1)
    lst = []
    for i in range(n):
        torch.cuda.set_device(i)
        free, total = (x / 1024**3 for x in torch.cuda.mem_get_info())
        props = torch.cuda.get_device_properties(i)
        lst.append({"id": i, "name": props.name, "free": free, "total": total})
    return lst


def load_texts(limit: int | None = None) -> dict[int, str]:
    if not TEXTS_FILE.exists():
        logging.error("Fichier %s introuvable.", TEXTS_FILE)
        sys.exit(1)
    logging.info("Chargement de %s ...", TEXTS_FILE)
    with open(TEXTS_FILE, "rb") as f:
        texts = pickle.load(f)
    if limit is not None:
        texts = {k: texts[k] for k in list(texts)[:limit]}
    logging.info("Clients à traiter : %s", f"{len(texts):,}")
    return texts


def build_worker_code() -> str:
    """Code exécuté dans chaque sous-processus « python -c … »."""
    return r'''
import os, sys, pickle, torch, gc, time
from pathlib import Path
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.getcwd() + "/src"))
gpu_id          = int(sys.argv[1])
input_file      = sys.argv[2]
output_file     = sys.argv[3]
batch_size      = int(sys.argv[4])
checkpoint_dir  = Path(sys.argv[5])
checkpoint_every= int(sys.argv[6])

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
print(f"\n[GPU {gpu_id}] Démarrage {datetime.now():%H:%M:%S}")

with open(input_file, "rb") as f:
    texts_dict = pickle.load(f)

ckpt_path = checkpoint_dir / f"gpu_{gpu_id}_checkpoint.pkl"
results, start_from = {}, 0
if ckpt_path.exists():
    with ckpt_path.open("rb") as f:
        cp = pickle.load(f)
    results, start_from = cp["results"], cp["last_index"]
    print(f"[GPU {gpu_id}] Reprise : {len(results)} portraits (index {start_from})")

from ubm.portrait_generator import PortraitGenerator
from tqdm import tqdm
gen = PortraitGenerator("cuda:0")

items = list(texts_dict.items())[start_from:]
total_batches = (len(items)+batch_size-1)//batch_size
times = []
pbar = tqdm(range(0, len(items), batch_size), desc=f"GPU {gpu_id}", total=total_batches)

def save_ckpt(i):
    ckpt_path.write_bytes(pickle.dumps({"results": results,
                                        "last_index": start_from+i,
                                        "ts": datetime.now().isoformat()}))
    pbar.write(f"[GPU {gpu_id}] CKPT {len(results)} portraits")

for idx, i in enumerate(pbar):
    t0 = time.time()
    batch = items[i:i+batch_size]
    try:
        portraits = gen.generate_batch(batch)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        step = max(1, len(batch)//2)
        portraits = {}
        for j in range(0, len(batch), step):
            try:
                portraits.update(gen.generate_batch(batch[j:j+step]))
            except Exception as e_sub:
                for cid, _ in batch[j:j+step]:
                    portraits[cid] = f"- OOM ({type(e_sub).__name__}) — FIN —"
    except Exception as e:
        for cid, _ in batch:
            portraits[cid] = f"- ERR : {str(e)[:50]} — FIN —"
    results.update(portraits)

    times.append(time.time()-t0)
    if len(times) > 10: times.pop(0)
    eta = (total_batches-idx-1)*sum(times)/len(times)
    pbar.set_postfix(batch=f"{times[-1]:.1f}s", ETA=f"{eta/60:.1f}m")

    if (idx+1) % checkpoint_every == 0: save_ckpt(i+batch_size)
    if (idx+1) % 50 == 0: torch.cuda.empty_cache(); gc.collect()

Path(output_file).write_bytes(pickle.dumps(results))
ckpt_path.unlink(missing_ok=True)
print(f"[GPU {gpu_id}] FIN — {len(results)} portraits")
'''


def write_monitor_script(n_gpus: int, ckpt_dir: Path) -> Path:
    script = f"""#!/bin/bash
while true; do
  clear
  echo "$(date '+%F %T')"
  echo "============== GPU =============="
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \\
      awk -F', ' '{{printf \"GPU %-2d %-24s %3d%%  %5.1f/%5.1f GiB\\n\", $1,$2,$3,$4/1024,$5/1024}}'
  echo -e "\\n========== Checkpoints =========="
  for i in $(seq 0 {n_gpus-1}); do
    f=\"{ckpt_dir}/gpu_${{i}}_checkpoint.pkl\"
    [[ -f $f ]] && du -h $f || echo \"GPU $i : -\"
  done
  sleep 5
done
"""
    path = OUTPUT_DIR / "monitor_portraits.sh"
    path.write_text(script)
    path.chmod(0o755)
    return path


run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
setup_logging(run_id)

# 1) GPUs
logging.info("=== GÉNÉRATION PARALLÈLE DES PORTRAITS ===")
gpus = gpu_list()
for g in gpus:
    logging.info("GPU %d : %s — %.1f/%.1f GiB libres",
                 g["id"], g["name"], g["free"], g["total"])

# 2) Données
limit = DRY_RUN_SIZE
texts = load_texts(limit)
lens  = [len(t) for t in texts.values()]
logging.info("Longueur moyenne : %.0f (min %d / max %d)",
             sum(lens)/len(lens), min(lens), max(lens))

# 3) Répartition
n_gpus = len(gpus)
items  = list(texts.items())
per_gpu, remainder = divmod(len(items), n_gpus)
del texts; gc.collect()

ckpt_dir = OUTPUT_DIR / "portrait_checkpoints"; ckpt_dir.mkdir(exist_ok=True)
worker_code       = build_worker_code()
checkpoint_every  = CHECKPOINT_EVERY
batch_size        = BATCH_SIZE

assignments, start = [], 0
for gid in range(n_gpus):
    cnt, end = per_gpu + (gid < remainder), start + per_gpu + (gid < remainder)
    subdict  = dict(items[start:end])
    in_file  = OUTPUT_DIR / f".gpu_{gid}_input.pkl"
    in_file.write_bytes(pickle.dumps(subdict))
    assignments.append({
        "gpu": gid,
        "input": str(in_file),
        "output": str(OUTPUT_DIR / f".gpu_{gid}_output.pkl"),
        "ckpt_dir": str(ckpt_dir),
        "count": cnt
    })
    logging.info("GPU %d : %s clients (%d–%d)", gid, f"{cnt:,}", start, end-1)
    start = end

mon_path = write_monitor_script(n_gpus, ckpt_dir)
logging.info("Script monitoring : %s", mon_path)

# 4) Lancement
procs, t0 = [], time.time()
for a in assignments:
    cmd = [sys.executable, "-c", worker_code,
           str(a["gpu"]), a["input"], a["output"],
           str(batch_size), a["ckpt_dir"], str(checkpoint_every)]
    logf = (OUTPUT_DIR / "logs" / f"gpu_{a['gpu']}_{run_id}.log").open("w")
    p    = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    procs.append((p, a))
    logging.info("Lancé GPU %d (pid %d)", a["gpu"], p.pid)
    time.sleep(2)

# 5) Attente
done, failed = [], []
while len(done)+len(failed) < len(procs):
    time.sleep(30)
    for p, a in procs:
        if p in done or p in failed: continue
        ret = p.poll()
        if ret is None: continue
        (done if ret == 0 else failed).append(p)
        logging.info("GPU %d %s (code %d)", a["gpu"],
                     "terminé" if ret == 0 else "ERREUR", ret)

# 6) Fusion
portraits, stats = {}, defaultdict(int)
for a in assignments:
    path = Path(a["output"])
    if not path.exists():
        logging.warning("Pas de sortie pour GPU %d", a["gpu"]); continue
    with path.open("rb") as f:
        res = pickle.load(f)
    portraits.update(res)
    stats["total"]   += len(res)
    stats["success"] += sum("failed" not in v.lower() for v in res.values())
    stats[a["gpu"]]   = len(res)
    logging.info("GPU %d : %s portraits", a["gpu"], f"{len(res):,}")

# 7) Sauvegarde finale
final = OUTPUT_DIR / f"portraits_{len(portraits):,}.pkl.gz"
with gzip.open(final, "wb") as f:
    pickle.dump(portraits, f, protocol=pickle.HIGHEST_PROTOCOL)

elapsed = time.time() - t0
logging.info("=== TERMINE ===")
logging.info("Total portraits : %s | succès : %s (%.1f %%)",
             f"{stats['total']:,}", f"{stats['success']:,}",
             100*stats['success']/stats['total'] if stats['total'] else 0)
logging.info("Durée : %.1f min (%.2f portraits/s)",
             elapsed/60, stats['total']/elapsed if elapsed else 0)
logging.info("Fichier : %s (%.1f GiB gz)", final, final.stat().st_size/1024**3)

# 8) Nettoyage
for a in assignments:
    for f in (a["input"], a["output"]):
        Path(f).unlink(missing_ok=True)
for ck in ckpt_dir.glob("gpu_*_checkpoint.pkl"):
    ck.unlink()

"""
### 4. Portraits: merging, cleaning and tokenization
"""

#!/usr/bin/env python3
"""
Script to prepare tokenized and augmented rich_text data for contrastive training.
Includes:
- Loading and fusing client features and portraits.
- Ensuring exactly 1M client entries with fallback profiles.
- Implementing a sophisticated text augmentation scheme.
- Tokenizing texts using a pre-trained tokenizer.
- Saving the output as a .jsonl.zst file for efficient streaming during training.
"""
import json
import pickle
import gzip
import zstandard as zstd
import random
import time
import numpy as np
import transformers
import re
import os
import textwrap
import io
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime

# ============================================================================
#                                CONFIGURATION
# ============================================================================
# --- Paths & Constants ---
OUTPUT_DIR = Path("output_features/gemma12b")
FEATURES_PKL = OUTPUT_DIR / f"complete_features_{TEST_SIZE}_clients.pkl"
PORTRAITS_PKL = OUTPUT_DIR / f"portraits_{TEST_SIZE}.pkl.gz"
OUT_JSONL_ZST = OUTPUT_DIR / "complete_dataset_1M_last.jsonl.zst"

# --- Text Processing ---
MAX_PROMPT_CHARS = 12_000
MAX_EVENTS_TO_SHOW = 1000
DEFAULT_NO_EVENTS_TEXT = "No event sequence available for this client."

# --- Augmentation Schema ---
AUG_SCHEMA_PROBS = {
    'section_dropout': 0.4,
    'event_masking': 0.3,
    'numerical_noise': 0.2,
    'reorder_lines': 0.1,
}
EVENT_MASK_PROB = 0.15
NUM_NOISE_SCALE = 0.05
REORDER_LINES_BLOCK_SIZE = 3

# --- Tokenization ---
TOKENIZER_NAME = "google/gemma-3-1b-it" # Should match your training model_id
MAX_TOKENS = 2048
TOKENIZER_BATCH = 500

# --- Script Behavior ---
DEBUG_EXAMPLES_TO_PRINT = 3

# --- Initial Setup ---
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[{datetime.now().strftime('%H:%M:%S')}] Verifying input files:")
print(f"- Features: {FEATURES_PKL.exists()} - {FEATURES_PKL}")
print(f"- Portraits: {PORTRAITS_PKL.exists()} - {PORTRAITS_PKL}")
if not FEATURES_PKL.exists() or not PORTRAITS_PKL.exists():
    raise FileNotFoundError("Missing one or more required input files. Please check paths.")

# ============================================================================
#                               HELPER FUNCTIONS
# ============================================================================

def load_pickle(path: Path):
    """Loads a pickle file, handling .gz compression."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        return pickle.load(f)

def is_failure(text: str) -> bool:
    """Determines if a generated portrait is a fallback message."""
    return text.lstrip().startswith(("- OOM", "- ERR", "- Portrait generation failed"))

def truncate_raw_sequence(raw_seq: str, max_events: int = MAX_EVENTS_TO_SHOW) -> str:
    """Truncates the raw event sequence to keep only the last N events."""
    if not raw_seq or not raw_seq.strip():
        return DEFAULT_NO_EVENTS_TEXT

    events = [e.strip() for e in raw_seq.split("</s>") if e.strip()]
    if not events:
        return DEFAULT_NO_EVENTS_TEXT

    if len(events) > max_events:
        events = events[-max_events:]
        return f"... (truncated to last {max_events} events)\n" + "</s>".join(events)
    else:
        return "</s>".join(events)

def add_noise_to_number_fixed(match):
    """Helper to add noise to numbers found by regex, ignoring specific key-like formats."""
    full_match_str = match.group(0)
    # Exclude key-like strings (e.g., "CHURN_RISK_0.5") from numerical noise
    if '_' in full_match_str and any(c.isalpha() for c in full_match_str.split('_')[0]):
        return full_match_str
    try:
        num = float(match.group(1)) # Target the captured number
        noise = num * NUM_NOISE_SCALE * (2 * random.random() - 1)
        return f"{num + noise:.2f}"
    except ValueError:
        return full_match_str # Return original on conversion failure

def apply_augmentation(text: str, schema_probs: dict, event_mask_prob: float, num_noise_scale: float, reorder_block_size: int, seed: int) -> str:
    """Applies a schema of augmentations to the input text."""
    random.seed(seed)

    # --- 1. Deconstruct text into sections ---
    section_pattern = re.compile(r'(^##\s[A-Z_]+\s##\n)', re.MULTILINE)
    sections = {}

    # Handle [PROFILE] block separately
    profile_match = re.match(r'(\[PROFILE\].*?)(?=##\s[A-Z_]+\s##)', text, re.DOTALL)
    if profile_match:
        sections['[PROFILE]'] = profile_match.group(1).strip() + '\n'
        remaining_text = text[profile_match.end():]
    else:
        remaining_text = text

    # Split remaining text by section headers
    parts = section_pattern.split(remaining_text)
    current_header = None
    for part in parts:
        if section_pattern.match(part):
            current_header = part.strip()
            sections[current_header] = ""
        elif current_header:
            sections[current_header] += part

    # Handle [END] marker
    if '[END]' in text:
        # This logic is complex; assuming it correctly isolates content
        sections['[END]'] = '[END]'

    # --- 2. Apply augmentations to each section ---
    augmented_sections = {}
    ordered_headers = ['[PROFILE]', '## OVERVIEW ##', '## CHURN_PROPENSITY ##', '## RECENT ##', '## TARGET_WINDOW_14D ##', '## TEMPORAL ##', '## SEQ ##', '## PRICE ##', '## SOCIAL ##', '## SKU_PROPENSITY ##', '## CAT_PROPENSITY ##', '## STATS ##', '## MISC ##', '## CUSTOM ##', '## PORTRAIT ##', '## RAW_SEQUENCE ##', '[END]']

    for header in ordered_headers:
        content = sections.get(header, "").strip()
        if not content and header not in ['[PROFILE]', '[END]']:
            continue

        content_lines = content.splitlines()

        # Section Dropout
        if header not in ['[PROFILE]', '## OVERVIEW ##', '## RAW_SEQUENCE ##', '## PORTRAIT ##', '[END]'] and random.random() < schema_probs.get('section_dropout', 0):
            continue # Drop the section

        # Line Reordering
        if header not in ['## RAW_SEQUENCE ##', '[PROFILE]', '[END]'] and len(content_lines) > reorder_block_size and random.random() < schema_probs.get('reorder_lines', 0):
            num_blocks = len(content_lines) // reorder_block_size
            blocks = [content_lines[i*reorder_block_size:(i+1)*reorder_block_size] for i in range(num_blocks)]
            remaining = content_lines[num_blocks*reorder_block_size:]
            random.shuffle(blocks)
            content_lines = [item for block in blocks for item in block] + remaining

        # Event Masking
        if header == "## RAW_SEQUENCE ##" and random.random() < schema_probs.get('event_masking', 0):
            events_list = [e.strip() for e in content.split("</s>") if e.strip()]
            masked_events = ["EVENT: [MASKED_EVENT]" if random.random() < event_mask_prob else e for e in events_list]
            content_lines = ["</s>".join(masked_events)]

        # Numerical Noise
        if random.random() < schema_probs.get('numerical_noise', 0):
            content_str = "\n".join(content_lines)
            numerical_regex = r'(?:[A-Z_]+_)?(\d+\.?\d*)'
            content_str = re.sub(numerical_regex, add_noise_to_number_fixed, content_str)
            content_lines = content_str.splitlines()

        augmented_sections[header] = "\n".join(content_lines)

    # --- 3. Reconstruct the augmented text ---
    reconstructed_parts = []
    for header in ordered_headers:
        if header in augmented_sections:
            content = augmented_sections[header]
            if header == '[PROFILE]':
                reconstructed_parts.append(content)
            elif header == '[END]':
                reconstructed_parts.append('\n' + content)
            else:
                reconstructed_parts.append('\n' + header + '\n' + content)

    final_text = "".join(reconstructed_parts).strip()
    return re.sub(r'\n{3,}', '\n\n', final_text) # Normalize newlines

def strip_to_max(text: str, n: int = MAX_PROMPT_CHARS) -> str:
    """Truncates text to n characters from the end, attempting to preserve [END]."""
    if len(text) <= n:
        return text
    # Prioritize keeping [END] if it falls within the tail window
    if '[END]' in text[-n:]:
        end_idx = text.rfind('[END]')
        if end_idx != -1 and len(text) - end_idx < n:
             # If [END] and its prefix fit, grab more context
            return text[-(n + (len(text) - end_idx)):]
    return text[-n:]

# ============================================================================
#                            MAIN SCRIPT EXECUTION
# ============================================================================

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === STARTING DATA PREPARATION PIPELINE ===")

# --- 1. Data Loading ---
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === 1. LOADING DATA ===")
print(f"Loading features: {FEATURES_PKL}")
all_results = load_pickle(FEATURES_PKL)
print(f"  → Found {len(all_results)} profiles")

print(f"Loading portraits: {PORTRAITS_PKL.name}")
portraits_all = load_pickle(PORTRAITS_PKL)
print(f"  → Found {len(portraits_all)} portraits")
valid_portraits = {cid: p for cid, p in portraits_all.items() if not is_failure(p)}
print(f"  → {len(valid_portraits)} valid portraits / {len(portraits_all)} total")

# --- 2. Data Fusing ---
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === 2. FUSING DATA SOURCES ===")
fused_rich_texts = {}
for cid, res in tqdm(all_results.items(), desc="Fusing data"):
    if not isinstance(res, dict) or res.get("status") != "success":
        continue
    try:
        jd = json.loads(res["json_str"])
        rich_text = jd.get("rich_text", "")
        if not rich_text: continue

        # Insert portrait and raw sequence sections before the [END] marker
        portrait_content = f"\n## PORTRAIT ##\n{valid_portraits[cid].strip()}\n" if cid in valid_portraits else ""
        raw_seq = res.get('profile', {}).get('raw_sequence', '')
        raw_seq_content = f"\n## RAW_SEQUENCE ##\n```\n{truncate_raw_sequence(raw_seq)}\n```"

        inserted_content = portrait_content + raw_seq_content

        end_idx = rich_text.rfind('[END]')
        if end_idx != -1:
            rich_text = rich_text[:end_idx] + inserted_content + '\n' + rich_text[end_idx:]
        else:
            rich_text += inserted_content

        fused_rich_texts[cid] = strip_to_max(rich_text, MAX_PROMPT_CHARS)
    except Exception as e:
        if len(fused_rich_texts) < 5:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error on client {cid} during fusion: {e}")

# --- 3. Dataset Completeness Correction ---
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === 3. ENSURING 1M CLIENTS ===")
final_data = {}
final_data.update(fused_rich_texts)

all_client_ids_expected = set(all_results.keys())
missing_ids = all_client_ids_expected - set(fused_rich_texts.keys())
print(f"  → Missing IDs to create as fallbacks: {len(missing_ids):,}")

for cid in tqdm(missing_ids, desc="Creating fallbacks"):
    # Create fallback profiles for clients that failed the initial fusion
    rich_text_fallback = f"[PROFILE]\n## OVERVIEW ##\n[CLIENT_ID_PLACEHOLDER]\nStatus: {all_results.get(cid, {}).get('status', 'unknown')}\n[END]"
    portrait_content = f"\n## PORTRAIT ##\n{valid_portraits[cid].strip()}\n" if cid in valid_portraits else ""
    raw_seq = all_results.get(cid, {}).get('profile', {}).get('raw_sequence', '')
    raw_seq_content = f"\n## RAW_SEQUENCE ##\n```\n{truncate_raw_sequence(raw_seq)}\n```"

    inserted_content = portrait_content + raw_seq_content
    end_idx = rich_text_fallback.rfind('[END]')
    if end_idx != -1:
        rich_text_fallback = rich_text_fallback[:end_idx] + inserted_content + '\n' + rich_text_fallback[end_idx:]
    else:
        rich_text_fallback += inserted_content

    final_data[cid] = strip_to_max(rich_text_fallback, MAX_PROMPT_CHARS)

# Adjust to exactly 1M if needed
if len(final_data) != 1_000_000:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ WARNING: {len(final_data):,} clients instead of 1,000,000! Adjusting...")
    if len(final_data) > 1_000_000:
        excess = len(final_data) - 1_000_000
        clients_to_remove = random.sample(list(final_data.keys()), excess)
        for cid in clients_to_remove: del final_data[cid]
    else:
        shortage = 1_000_000 - len(final_data)
        max_id = max(all_client_ids_expected)
        for i in range(shortage):
            final_data[max_id + i + 1] = "[PROFILE]\n## OVERVIEW ##\n[CLIENT_ID_PLACEHOLDER]\nStatus: artificial\n[END]"

assert len(final_data) == 1_000_000, f"FATAL: {len(final_data):,} clients after adjustment!"
print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ SUCCESS: Exactly {len(final_data):,} client profiles ready!")

# --- 4. Tokenization & Augmentation ---
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === 4. TOKENIZATION & AUGMENTATION ===")
print(f"  → Loading tokenizer: {TOKENIZER_NAME}")
tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_auth_token=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

client_ids = list(final_data.keys())

cctx = zstd.ZstdCompressor(level=3)
with open(OUT_JSONL_ZST, 'wb') as outfile_bin, cctx.stream_writer(outfile_bin) as compressor_writer:
    text_writer = io.TextIOWrapper(compressor_writer, encoding='utf-8')
    num_debug_printed = 0

    for i in tqdm(range(0, len(client_ids), TOKENIZER_BATCH), desc="Tokenizing & Writing"):
        batch_cids = client_ids[i : i + TOKENIZER_BATCH]
        original_texts = [final_data[cid] for cid in batch_cids]
        augmented_texts = [apply_augmentation(original_texts[j], AUG_SCHEMA_PROBS, EVENT_MASK_PROB, NUM_NOISE_SCALE, REORDER_LINES_BLOCK_SIZE, seed=(cid + j + i)) for j, cid in enumerate(batch_cids)]

        if num_debug_printed < DEBUG_EXAMPLES_TO_PRINT:
            print(f"\n--- Debug Example {num_debug_printed + 1} (Client {batch_cids[0]}) ---")
            print("Original Text (First 1000 chars):")
            print(textwrap.indent(original_texts[0][:1000] + "...", '  '))
            print("\nAugmented Text (First 1000 chars):")
            print(textwrap.indent(augmented_texts[0][:1000] + "...", '  '))
            num_debug_printed += 1

        # Tokenize both original and augmented texts
        tokenized_originals = tokenizer(original_texts, truncation=True, max_length=MAX_TOKENS).input_ids
        tokenized_augmentations = tokenizer(augmented_texts, truncation=True, max_length=MAX_TOKENS).input_ids

        for j, cid in enumerate(batch_cids):
            record = {
                "client_id": cid,
                "input_ids": tokenized_originals[j],
                "input_ids_aug": tokenized_augmentations[j]
            }
            text_writer.write(json.dumps(record, ensure_ascii=False) + '\n')
    text_writer.flush()

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === DATA PREPARATION COMPLETE ===")
print(f"  → Total records written to {OUT_JSONL_ZST}: {len(final_data):,}")
print(f"  → Output file size: {OUT_JSONL_ZST.stat().st_size / (1024*1024):.2f} MB")

import io
import zstandard as zstd
import transformers
import textwrap
from pathlib import Path
from tqdm.auto import tqdm

# --- Use orjson if available, otherwise fallback to standard json ---
try:
    import orjson
    json_loads = orjson.loads
    print("Using 'orjson' for JSON parsing.")
except ImportError:
    import json
    json_loads = json.loads

# ============================================================================
#                      CONFIGURATION & CLIENT IDs
# ============================================================================
DATASET_PATH = Path("output_features/gemma12b") / "complete_dataset_1M_last.jsonl.zst"
TOKENIZER_NAME = "google/gemma-3-1b-it"

# The specific Client IDs you want to find and display
IDS_TO_DISPLAY = [
    "9473050"
]
# ============================================================================

print(f"Loading tokenizer: {TOKENIZER_NAME}")
try:
    tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_auth_token=True)
except Exception as e:
    print(f"❌ ERROR: Could not load tokenizer. {e}")
    exit()

if not DATASET_PATH.exists():
    print(f"❌ ERROR: Dataset file not found at {DATASET_PATH}")
    exit()

# Convert string IDs to a set of integers for efficient lookup
ids_to_find_set = {int(cid) for cid in IDS_TO_DISPLAY}
found_records = {}

print(f"\nScanning {DATASET_PATH} to find {len(ids_to_find_set)} specific clients...")

try:
    with open(DATASET_PATH, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(f)
        text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')

        # Scan the file until all requested clients are found
        for line_str in tqdm(text_stream, desc="Scanning dataset", unit=" clients"):
            record = json_loads(line_str)
            client_id = record.get("client_id")

            if client_id in ids_to_find_set:
                found_records[client_id] = record
                # Optimization: Stop scanning if we've found all clients
                if len(found_records) == len(ids_to_find_set):
                    print("\nAll requested clients found. Stopping scan.")
                    break
except Exception as e:
    print(f"❌ ERROR: An unexpected error occurred during file scanning: {e}")
    exit()

# --- Print the results for the found clients ---
print("\n" + "="*80)
print("Displaying text for the requested clients:")
print("="*80 + "\n")

for client_id_str in IDS_TO_DISPLAY:
    client_id_int = int(client_id_str)

    if client_id_int in found_records:
        record = found_records[client_id_int]

        # Decode the token sequences back into text
        original_text = tokenizer.decode(record.get("input_ids", []), skip_special_tokens=True)
        augmented_text = tokenizer.decode(record.get("input_ids_aug", []), skip_special_tokens=True)

        print(f"--- CLIENT ID {client_id_int} ---")
        print("\n[ORIGINAL TEXT (DECODED)]")
        print(textwrap.indent(original_text, '  '))
        print("\n[AUGMENTED TEXT (DECODED)]")
        print(textwrap.indent(augmented_text, '  '))
        print("\n" + "="*80 + "\n")
    else:
        print(f"--- CLIENT ID {client_id_int} (NOT FOUND in the dataset) ---\n")
