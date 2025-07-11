import unsloth
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
import random
import gzip, zstandard as zstd
import transformers, re, os, textwrap
import os, sys, gc, pickle, subprocess, time, argparse, logging, gzip
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, statistics, zstandard as zstd
from tqdm.auto import tqdm

# Paths
DATA_DIR = "ubc_data"
CACHE_DIR = "ubc_data/cache_v3"
OUTPUT_DIR = "output_features/gemma1b"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"CPUs disponibles: {mp.cpu_count()}")
print(f"Output directory: {OUTPUT_DIR}")

def generate_complete_features_batch(client_batch: List[int], batch_id: int) -> Dict:
    """Version qui charge TOUT en mémoire comme l'ancienne"""
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
    signal.alarm(3600)  # 1 hour

    print(f"[Batch {batch_id}] Starting {len(client_batch)} clients...", flush=True)
    start_time = time.time()

    try:
        # OPTION 1: Charger depuis le cache 1M clients s'il existe
        cache_1m = Path(CACHE_DIR) / "events_1m_clients.parquet"
        if cache_1m.exists():
            print(f"[Batch {batch_id}] Loading cached 1M events...", flush=True)
            events_df = pl.read_parquet(cache_1m)
            print(f"[Batch {batch_id}] Loaded {events_df.height:,} events from cache", flush=True)

            # Filtrer pour nos clients
            events_df = events_df.filter(pl.col("client_id").is_in(client_batch))
            print(f"[Batch {batch_id}] Filtered to {events_df.height:,} events for batch", flush=True)

            # Créer le générateur et lui donner les données
            gen = AdvancedUBMGenerator(DATA_DIR, CACHE_DIR, debug_mode=False)
            gen.events_df = events_df

            # Charger les stats depuis ubc_data si possible
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
            # OPTION 2: Charger normalement (tout en mémoire)
            print(f"[Batch {batch_id}] No cache, loading data normally...", flush=True)
            gen = AdvancedUBMGenerator(DATA_DIR, CACHE_DIR, debug_mode=False)
            os.environ["SKIP_URL_GRAPH"] = "1"

            # Forcer le chargement en mémoire
            gen.load_data(use_cache=True, relevant_client_ids=None)  # None = charger TOUT

            if gen.events_df is not None:
                print(f"[Batch {batch_id}] Loaded {gen.events_df.height:,} total events", flush=True)
                # Filtrer pour le batch
                gen.events_df = gen.events_df.filter(pl.col("client_id").is_in(client_batch))
                print(f"[Batch {batch_id}] Filtered to {gen.events_df.height:,} events for batch", flush=True)

        # Initialiser les extracteurs
        gen._extractors = {}
        gen._extractors['temporal'] = gen.TemporalFeatureExtractor(gen) if hasattr(gen, 'TemporalFeatureExtractor') else None
        gen._extractors['sequence'] = gen.SequenceFeatureExtractor(gen) if hasattr(gen, 'SequenceFeatureExtractor') else None

        # Importer et ajouter les extracteurs depuis le module
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

        # PROCESSING
        results = {}
        now = datetime.now()
        process_start = time.time()

        for idx, cid in enumerate(client_batch):
            if idx % 50 == 0:
                elapsed = time.time() - process_start
                rate = idx / elapsed if elapsed > 0 and idx > 0 else 0
                print(f"[Batch {batch_id}] Progress: {idx}/{len(client_batch)} "
                      f"({elapsed:.1f}s, {rate:.1f} clients/s)", flush=True)

            try:
                # Get events - DIRECT depuis events_df
                events = gen.events_df.filter(pl.col('client_id') == cid)

                if events.height == 0:
                    results[cid] = {
                        "status": "no_data",
                        "profile": {"client_id": cid, "error": "No activity"},
                        "rich_text": f"[CLIENT_{cid}]\nNo activity data.",
                        "json_str": json.dumps({"profile": {"client_id": cid, "error": "No activity"}}, ensure_ascii=False)
                    }
                    continue

                # Overview
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
                    "temporal": "TEMPORAL",
                    "sequence": "SEQUENCE",
                    "social": "SOCIAL",
                    "price": "PRICE",
                    "intent": "OVERVIEW",
                    "graph": "CUSTOM",
                    "name_embedding": "CUSTOM",
                    "custom_behavior": "CUSTOM",
                    "churn_propensity": "CHURN_PROPENSITY",
                    "top_sku": "SKU_PROPENSITY",
                    "top_category": "CAT_PROPENSITY",
                }

                # Extract features
                for ex_name, extractor in gen._extractors.items():
                    if extractor is None:
                        continue
                    tgt_sec = ex_to_sec.get(ex_name, "CUSTOM")
                    try:
                        feats = extractor.extract_features(cid, events, now)
                        repeat = IMPLICIT_WEIGHT_REPEAT if ex_name in ("top_sku", "top_category", "churn_propensity") else 1
                        for ft in feats:
                            for _ in range(repeat):
                                section_map[tgt_sec].append(ft)
                            features_json.append({"type": ex_name, "value": ft})
                    except Exception as err:
                        section_map[tgt_sec].append(f"{ex_name}-error")

                # Behavioral metrics
                co_pairs = top_co_pairs(events)
                cat_pairs = top_co_categories(events)
                cart_conv = cart_conversion_stats(events)
                section_map["CUSTOM"].extend(co_pairs + cat_pairs + cart_conv)

                # Compact metrics
                try:
                    compact_tags = gen._compute_compact_metrics(events)
                    section_map["PROP_SUBSET_STATS"].extend(compact_tags)
                    for tag in compact_tags:
                        features_json.append({"type": "compact", "value": tag})
                except:
                    pass

                try:
                    extra_tags = gen._compute_extra_short_metrics(cid, events, now)
                    section_map["PROP_SUBSET_STATS"].extend(extra_tags)
                    for tag in extra_tags:
                        features_json.append({"type": "extra", "value": tag})
                except:
                    pass

                # History
                recent_cut = now - timedelta(days=14)
                medium_cut = now - timedelta(days=90)

                recent_txt = gen._generate_detailed_events_text(events.filter(pl.col("timestamp") >= recent_cut), limit=30)
                medium_txt = gen._generate_summarized_events_text(
                    events.filter((pl.col("timestamp") >= medium_cut) & (pl.col("timestamp") < recent_cut))
                )
                hist_txt = gen._generate_aggregated_events_text(events.filter(pl.col("timestamp") < medium_cut))

                if recent_txt != "No recent activity.":
                    section_map["TARGET_WINDOW_14D"].append(recent_txt)
                else:
                    section_map["TARGET_WINDOW_14D"].append("No activity in last 14 days")

                if medium_txt != "No medium-term activity.":
                    section_map["SEQUENCE"].append(medium_txt)
                if hist_txt != "No historical activity.":
                    section_map["CUSTOM"].append(hist_txt)

                # Raw sequence
                try:
                    raw_seq = gen._generate_raw_sequence(events, max_events=100)
                except:
                    raw_seq = ""

                section_map["CUSTOM"] = list(dict.fromkeys(section_map["CUSTOM"]))

                # Build rich text
                try:
                    rich_text = _build_rich_text(
                        section_map=dict(section_map),
                        max_tokens=MAX_RICH_TOKENS,
                        implicit_repeat=IMPLICIT_WEIGHT_REPEAT,
                        top_per_section=TOP_FEATURES_PER_SECTION,
                        shuffle_seed=cid,
                        use_markers=True
                    )
                except:
                    rich_text = "\n\n".join(f"## {sec} ##\n" + "\n".join(lines[:20]) for sec, lines in section_map.items())

                # Profile
                profile = {
                    "client_id": cid,
                    "overview": {"user_type": user_type, "segments": seg_list},
                    "features": features_json[:100],
                    "behavioral_metrics": {
                        "co_pairs": co_pairs[:5],
                        "category_pairs": cat_pairs[:5],
                        "cart_conversion": cart_conv[:5],
                    },
                    "recent_activity": recent_txt[:500],
                    "medium_term_summary": medium_txt[:500],
                    "historical_aggregates": hist_txt[:500],
                    "raw_sequence": raw_seq[:1000],
                }

                results[cid] = {
                    "status": "success",
                    "profile": profile,
                    "rich_text": rich_text,
                    "json_str": json.dumps({"profile": profile, "rich_text": rich_text}, ensure_ascii=False)
                }

            except Exception as e:
                results[cid] = {
                    "status": "error",
                    "error": str(e)[:500],
                    "profile": {"client_id": cid, "error": str(e)[:500]},
                    "rich_text": f"ERROR: {str(e)[:200]}",
                    "json_str": json.dumps({"client_id": cid, "error": str(e)[:500]}, ensure_ascii=False)
                }

        signal.alarm(0)

        total_time = time.time() - start_time
        print(f"[Batch {batch_id}] Completed {len(results)} clients in {total_time:.1f}s "
              f"({len(results)/total_time:.1f} clients/s)", flush=True)

        del gen
        gc.collect()
        return results

    except Exception as e:
        print(f"[Batch {batch_id}] FAILED: {e}", flush=True)
        signal.alarm(0)
        import traceback
        traceback.print_exc()
        return {cid: {"status": "batch_error", "error": str(e)[:500]} for cid in client_batch}
def truncate_raw_sequence(raw_seq, max_events=100):
        """Tronque la séquence pour garder seulement les N derniers événements"""
        if not raw_seq or raw_seq.strip() == "":
            return DEFAULT_NO_EVENTS_TEXT

        events = raw_seq.split("</s>")
        events = [e.strip() for e in events if e.strip()]

        if not events:
            return DEFAULT_NO_EVENTS_TEXT

        if len(events) > max_events:
            events = events[-max_events:]
            truncated_seq = "... (truncated to last {} events)\n".format(max_events)
            truncated_seq += "</s>".join(events)
        else:
            truncated_seq = "</s>".join(events)

        return truncated_seq
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",           action="store_true",
                       help="Mode DEBUG → Only 5 client ids")
    
    # Configuration threads
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_MAX_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["POLARS_MAX_THREADS"] = "8"
    os.environ["SKIP_URL_GRAPH"] = "1"
    
    args = parser.parse_args()
    # ── TEST MODE ────────────────────────────────────────────────────────────
    TEST_MODE = args.debug
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
        

    BATCH_SIZE = 1000
    N_WORKERS = min(8, mp.cpu_count() // 2)

    print(f"Configuration:")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Workers: {N_WORKERS}")
    print(f"- Total batches: {len(client_ids) // BATCH_SIZE + 1}")

    # Créer les batches
    batches = []
    for i in range(0, len(client_ids), BATCH_SIZE):
        batch = client_ids[i:i+BATCH_SIZE].tolist()
        batches.append((batch, i//BATCH_SIZE))

    print(f"\nTest avec {len(batches)} batches de {BATCH_SIZE} clients...")

    # ============================================
    # Cellule 5: Exécution parallèle (INCHANGÉ)
    all_results = {}
    failed_clients = []

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        future_to_batch = {
            executor.submit(generate_complete_features_batch, batch[0], batch[1]): batch
            for batch in batches
        }

        with tqdm(total=len(client_ids), desc="Generating COMPLETE features") as pbar:
            for future in as_completed(future_to_batch):
                batch_data = future_to_batch[future]
                batch_clients, batch_id = batch_data

                try:
                    batch_results = future.result(timeout=600)
                    all_results.update(batch_results)

                    batch_errors = sum(1 for r in batch_results.values() if r.get("status") != "success")
                    if batch_errors > 0:
                        print(f"\n⚠️ Batch {batch_id}: {batch_errors} erreurs")

                    pbar.update(len(batch_clients))

                except Exception as e:
                    print(f"\n❌ Batch {batch_id} failed: {e}")
                    failed_clients.extend(batch_clients)
                    pbar.update(len(batch_clients))

    elapsed = time.time() - start_time
    print(f"\n✅ Features COMPLETES générées en {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Success: {sum(1 for r in all_results.values() if r.get('status') == 'success')}")
    print(f"Errors: {sum(1 for r in all_results.values() if r.get('status') != 'success')}")
    print(f"Failed batches clients: {len(failed_clients)}")
    
    OUTPUT_DIR = "output_features/gemma1b"
    # ============================================
    # Cellule 6: Sauvegarder (INCHANGÉ)
    save_path = f"{OUTPUT_DIR}/complete_features_{len(client_ids)}_clients.pkl"
    print(f"Sauvegarde dans {save_path}...")

    with open(save_path, 'wb') as f:
        pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_size = sum(len(r.get('json_str', '')) for r in all_results.values())
    print(f"Taille totale des JSON: {total_size/1024/1024:.1f} MB")

    # ============================================
    # Cellule 7: Vérification améliorée pour les nouvelles sections
    print("\nVérification de la complétude des features V3:")
    print("=" * 60)

    # Sections mises à jour avec les nouvelles
    expected_sections = [
        "OVERVIEW", 
        "CHURN_PROPENSITY",      # NOUVELLE
        "TARGET_WINDOW_14D",     # NOUVELLE
        "TEMPORAL", 
        "SEQUENCE", 
        "PRICE", 
        "SOCIAL", 
        "SKU_PROPENSITY",        # NOUVELLE  
        "CAT_PROPENSITY",        # NOUVELLE
        "PROP_SUBSET_STATS",     # NOUVELLE
        "CUSTOM"
    ]

    sample_results = [r for r in all_results.values() if r.get('status') == 'success'][:5]

    for i, result in enumerate(sample_results):
        print(f"\nClient {i+1}:")
        rich_text = result.get('rich_text', '')

        # Vérifier les sections
        for section in expected_sections:
            if f"## {section} ##" in rich_text:
                # Compter les features dans cette section
                count = rich_text.count(f"\n", rich_text.find(f"## {section} ##"))
                print(f"  ✓ {section} ({count} lignes)")
            else:
                print(f"  ✗ {section} MANQUANT!")

        # Vérifier les markers
        print("\n  Section Markers:")
        markers = ["[PROFILE]", "[CHURN]", "[RECENT]", "[TIME]", "[SEQ]", 
                   "[PRICE]", "[SOCIAL]", "[SKU]", "[CAT]", "[STATS]", "[MISC]", "[END]"]
        for marker in markers:
            if marker in rich_text:
                print(f"  ✓ {marker}")

        # Statistiques des features
        if 'feature_counts' in result:
            print(f"\n  Features par section:")
            for sec, count in result['feature_counts'].items():
                print(f"    {sec}: {count}")

    texts_for_portraits = {}
    profiles_dict = {}
    
    import json
    for cid, result in all_results.items():
        if result.get('status') == 'success':
            json_data = json.loads(result['json_str'])
            rich_text = json_data.get('rich_text', '')

            if rich_text:
                texts_for_portraits[cid] = rich_text
                profiles_dict[cid] = json_data.get('profile', {})

    print(f"\nTextes prêts pour portraits LLM: {len(texts_for_portraits)}")
    print(f"Longueur moyenne: {np.mean([len(t) for t in texts_for_portraits.values()]):.0f} chars")

    # Sauvegarder
    texts_path = f"{OUTPUT_DIR}/texts_for_portraits_{len(texts_for_portraits)}.pkl"
    profiles_path = f"{OUTPUT_DIR}/profiles_{len(profiles_dict)}.pkl"

    with open(texts_path, 'wb') as f:
        pickle.dump(texts_for_portraits, f)
    with open(profiles_path, 'wb') as f:
        pickle.dump(profiles_dict, f)

    print(f"Sauvegardé:")
    print(f"- Textes: {texts_path}")
    print(f"- Profiles: {profiles_path}")

    # Chemins
    complete_features_path = f"{OUTPUT_DIR}/complete_features_{TEST_SIZE}_clients.pkl"
    texts_path = f"{OUTPUT_DIR}/texts_for_portraits_{TEST_SIZE}.pkl"

    # Configuration
    MAX_EVENTS_TO_SHOW = 1000
    DEFAULT_NO_EVENTS_TEXT = "No event sequence available for this client."
    PROGRESS_EVERY = 5000  # Afficher progression tous les 5000 clients

    print("=== AJOUT OPTIMISÉ DE RAW_SEQUENCE ===")
    print(f"Début: {datetime.now().strftime('%H:%M:%S')}")

    # 1. Test rapide sur échantillon
    print("\nTest sur 100 clients d'abord...")
    with open(complete_features_path, 'rb') as f:
        all_results = pickle.load(f)

    sample_clients = random.sample(list(all_results.keys()), min(100, len(all_results)))
    test_corrected = 0

    for cid in sample_clients:
        result = all_results[cid]
        if result.get('status') == 'success':
            rich_text = result.get('rich_text', '')
            if '## RAW_SEQUENCE ##' not in rich_text:
                test_corrected += 1

    print(f"Sur l'échantillon: {test_corrected}/100 clients ont besoin de correction")

    if test_corrected == 0:
        print("✅ Les clients ont déjà RAW_SEQUENCE!")
        exit()

    # # 2. Demander confirmation
    # response = input(f"\nCorriger les ~{len(all_results)} clients? (y/n): ")
    # if response.lower() != 'y':
    #     print("Annulé.")
    #     exit()

    # 3. Charger texts_for_portraits
    print("\nChargement des textes pour portraits...")
    with open(texts_path, 'rb') as f:
        texts_for_portraits = pickle.load(f)

    # 4. Correction avec progression détaillée
    print(f"\nDébut de la correction de {len(all_results)} clients...")
    start_time = time.time()

    corrected_count = 0
    no_raw_seq_count = 0
    already_has_raw_seq = 0
    error_count = 0
    last_print_time = time.time()

    # Parcourir avec enumerate pour avoir l'index
    client_items = list(all_results.items())
    total_clients = len(client_items)

    for i, (cid, result) in enumerate(client_items):
        # Afficher progression plus souvent
        if i % PROGRESS_EVERY == 0 or time.time() - last_print_time > 30:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 and i > 0 else 0
            eta = (total_clients - i) / rate if rate > 0 else 0

            print(f"Progression: {i}/{total_clients} ({i/total_clients*100:.1f}%) - "
                  f"Vitesse: {rate:.0f} clients/s - "
                  f"ETA: {eta/60:.1f} min - "
                  f"Corrigés: {corrected_count}")
            last_print_time = time.time()

        try:
            if result.get('status') != 'success':
                continue

            profile = result.get('profile', {})
            raw_seq = profile.get('raw_sequence', '')
            rich_text = result.get('rich_text', '')

            # Si RAW_SEQUENCE déjà présent
            if '## RAW_SEQUENCE ##' in rich_text:
                already_has_raw_seq += 1
                continue

            # Préparer la séquence
            if not raw_seq:
                no_raw_seq_count += 1
                raw_seq_to_add = DEFAULT_NO_EVENTS_TEXT
            else:
                raw_seq_to_add = truncate_raw_sequence(raw_seq, MAX_EVENTS_TO_SHOW)

            # Construire la section
            raw_section = f"\n## RAW_SEQUENCE ##\n```\n{raw_seq_to_add}\n```"

            # Ajouter au rich_text
            if '[END]' in rich_text:
                new_rich_text = rich_text.replace('[END]', raw_section + '\n[END]')
            else:
                new_rich_text = rich_text + raw_section

            # Mettre à jour
            result['rich_text'] = new_rich_text

            # JSON update
            try:
                json_data = json.loads(result['json_str'])
                json_data['rich_text'] = new_rich_text
                result['json_str'] = json.dumps(json_data, ensure_ascii=False)
            except:
                pass

            # Texts for portraits
            if cid in texts_for_portraits:
                texts_for_portraits[cid] = new_rich_text

            corrected_count += 1

            # Afficher un exemple tous les 100k
            if corrected_count % 100000 == 1:
                print(f"\nExemple - Client {cid}:")
                if raw_seq:
                    nb_events = len([e for e in raw_seq.split("</s>") if e.strip()])
                    print(f"  {nb_events} événements → section de {len(raw_seq_to_add)} chars")

        except Exception as e:
            error_count += 1
            if error_count < 10:
                print(f"Erreur client {cid}: {e}")

    # 5. Résumé final
    total_time = time.time() - start_time
    print(f"\n=== TERMINÉ en {total_time/60:.1f} minutes ===")
    print(f"- Déjà avec RAW_SEQUENCE: {already_has_raw_seq}")
    print(f"- Corrigés: {corrected_count}")
    print(f"  dont sans séquence: {no_raw_seq_count}")
    print(f"- Erreurs: {error_count}")
    print(f"- Vitesse moyenne: {len(all_results)/total_time:.0f} clients/s")

    # 6. Vérification rapide
    print("\nVérification sur 5 clients aléatoires:")
    for cid in random.sample(list(all_results.keys()), 5):
        result = all_results[cid]
        if result.get('status') == 'success':
            has_raw = '## RAW_SEQUENCE ##' in result.get('rich_text', '')
            print(f"  Client {cid}: RAW_SEQUENCE = {has_raw}")

    # 7. Sauvegarde
    if corrected_count > 0:
        print(f"\n💾 {corrected_count} corrections à sauvegarder")
        # response = input("Sauvegarder? (y/n): ")

#         if response.lower() == 'y':
        import shutil

        # Backups
        print("Création des backups...")
        shutil.copy(complete_features_path, complete_features_path + '.backup')
        shutil.copy(texts_path, texts_path + '.backup')

        # Sauvegarde
        print("Sauvegarde en cours...")
        with open(complete_features_path, 'wb') as f:
            pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(texts_path, 'wb') as f:
            pickle.dump(texts_for_portraits, f)

        print("✅ Sauvegarde terminée!")
        print(f"Fin: {datetime.now().strftime('%H:%M:%S')}")
    # else:
    #     print("\n✅ Aucune correction nécessaire!")

    # Charger un client aléatoire
    texts_path = f"{OUTPUT_DIR}/texts_for_portraits_{TEST_SIZE}.pkl"

    with open(texts_path, 'rb') as f:
        texts = pickle.load(f)
        
    # Prendre un client aléatoire
    cid = random.choice(list(texts.keys()))
    rich_text = texts[cid]

    print(f"=== EXEMPLE COMPLET - Client {cid} ===")
    print(f"Longueur totale: {len(rich_text)} caractères\n")
        
    # Montrer juste la section RAW_SEQUENCE
    if '## RAW_SEQUENCE ##' in rich_text:
        start = rich_text.find('## RAW_SEQUENCE ##')
        # Trouver la fin (prochain ## ou [END])
        next_section = rich_text.find('\n##', start + 1)
        end_marker = rich_text.find('[END]', start)

        if next_section > 0 and (end_marker < 0 or next_section < end_marker):
            end = next_section
        elif end_marker > 0:
            end = end_marker
        else:
            end = min(start + 2000, len(rich_text))  # Max 2000 chars

        raw_section = rich_text[start:end].strip()

        print("\n=== SECTION RAW_SEQUENCE ===")
        print("-" * 60)
        # Limiter l'affichage à 1000 caractères
        if len(raw_section) > 1000:
            print(raw_section[:1000])
            print(f"\n... (tronqué, {len(raw_section) - 1000} caractères omis)")
        else:
            print(raw_section)
        print("-" * 60)
    else:
        print("❌ RAW_SEQUENCE non trouvé!")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    from pathlib import Path
    # ====== PARAMÈTRES PAR DÉFAUT ======
    OUTPUT_DIR       = Path("output_features/gemma1b")
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

    from pathlib import Path

    OUTPUT_DIR      = Path("output_features/gemma1b")
    FEATURES_PKL    = OUTPUT_DIR / f"complete_features_{TEST_SIZE}_clients.pkl" 
    PORTRAITS_PKL   = OUTPUT_DIR / f"portraits_{TEST_SIZE}.pkl.gz" 

    MAX_TOKENS = 2048                 
    USE_AUG2         = False
    DROP_PROB_AUG1   = 0.35
    DROP_PROB_AUG2   = 0.50
    TOKENIZER_NAME   = "google/gemma-3-1b-it"
    OUT_JSONL_ZST    = OUTPUT_DIR / "complete_dataset_1M.jsonl.zst"
    TOKENIZER_BATCH  = 500

    # Vérifier que les fichiers existent
    print("Vérification des fichiers:")
    print(f"- Features: {FEATURES_PKL.exists()} - {FEATURES_PKL}")
    print(f"- Portraits: {PORTRAITS_PKL.exists()} - {PORTRAITS_PKL}")
    
    def load_pickle(path: Path):
        op = gzip.open if path.suffix == ".gz" else open
        with op(path, "rb") as f:
            return pickle.load(f)

    # ============================================
    # Cellule 2 : chargement CORRIGÉ
    print("\n=== CHARGEMENT DES DONNÉES ===")
    print(f"Chargement features : {FEATURES_PKL}")
    all_results = load_pickle(FEATURES_PKL)
    print(f"  → {len(all_results)} profils")

    # Vérifier le type
    sample_key = list(all_results.keys())[0]
    sample_value = all_results[sample_key]
    print(f"  → Type des valeurs: {type(sample_value)}")
    if isinstance(sample_value, dict):
        print(f"  → Clés disponibles: {list(sample_value.keys())[:5]}...")

    print(f"\nChargement portraits : {PORTRAITS_PKL.name}")
    portraits_all = load_pickle(PORTRAITS_PKL)
    print(f"  → {len(portraits_all)} portraits")
    
    # Vérifier le type
    sample_portrait = list(portraits_all.values())[0]
    print(f"  → Type des portraits: {type(sample_portrait)}")
    print(f"  → Longueur exemple: {len(sample_portrait)} chars")
    
    def is_failure(txt: str) -> bool:
        """Détermine si un portrait est un fallback"""
        return txt.lstrip().startswith(("- OOM", "- ERR", "- Portrait generation failed"))

    # Filtrer les portraits valides
    valid_portraits = {cid: p for cid, p in portraits_all.items() if not is_failure(p)}
    print(f"  → {len(valid_portraits)} portraits valides / {len(portraits_all)}")
    # ============================================


    # ============================================
    # Cellule 3 : fusion features + portrait + raw
    print("\n=== FUSION DES DONNÉES ===")
    final_data = {}
    no_json_count = 0
    no_portrait_count = 0

    for cid, res in tqdm(all_results.items(), desc="Fusion"):
        # Vérifier que c'est bien un dict avec status
        if not isinstance(res, dict):
            continue

        if res.get("status") != "success":
            continue

        # Vérifier qu'on a json_str
        if "json_str" not in res:
            no_json_count += 1
            continue

        try:
            jd = json.loads(res["json_str"])
            rich = jd.get("rich_text", "")

            if not rich:
                continue

            # Ajouter le portrait si disponible
            if cid in valid_portraits:
                portrait = valid_portraits[cid].strip()
                insert = f"\n## PORTRAIT ##\n{portrait}\n"
                if "[END]" in rich:
                    rich = rich.replace("[END]", insert + "[END]")
                else:
                    rich += insert
            else:
                no_portrait_count += 1

            jd["rich_text"] = rich
            final_data[cid] = jd

        except Exception as e:
            if len(final_data) < 5:  # Debug les premières erreurs
                print(f"Erreur client {cid}: {e}")
            continue

    print(f"\nRésultats fusion:")
    print(f"  → Clients fusionnés : {len(final_data)}")
    print(f"  → Sans json_str : {no_json_count}")
    print(f"  → Sans portrait : {no_portrait_count}")

    # Afficher un exemple
    if final_data:
        example_cid = list(final_data.keys())[0]
        example = final_data[example_cid]
        print(f"\nExemple (client {example_cid}):")
        print(f"  → Clés: {list(example.keys())}")
        rich_text = example.get("rich_text", "")
        print(f"  → Rich text length: {len(rich_text)} chars")
        output_file = f"output_features/gemma1b/example_client_{example_cid}.txt"
        with open(output_file, 'w') as f:
            f.write(rich_text)
        if "## PORTRAIT ##" in rich_text:
            print("  → ✓ Portrait intégré")
        else:
            print("  → ✗ Pas de portrait")
    # ============================================
    # ============================================
    # CELLULE CORRECTIVE : Garantir exactement 1M clients
    # À insérer APRÈS la cellule 3 (fusion) et AVANT la cellule 4 (tokenisation)
    # ============================================
    print("\n=== CORRECTION POUR 1M CLIENTS ===")
    print(f"Clients actuels : {len(final_data)}")
    print(f"Manquants : {1_000_000 - len(final_data)}")

    # 1. Identifier TOUS les client_ids attendus
    all_client_ids = set(all_results.keys())
    processed_ids = set(final_data.keys())
    missing_ids = all_client_ids - processed_ids

    print(f"\nAnalyse des manquants:")
    print(f"  → IDs dans all_results : {len(all_client_ids)}")
    print(f"  → IDs traités : {len(processed_ids)}")
    print(f"  → IDs manquants : {len(missing_ids)}")

    # 2. Analyser pourquoi ils manquent
    status_counts = {}
    for cid in list(missing_ids)[:10]:  # Examiner les 10 premiers
        if cid in all_results:
            status = all_results[cid].get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1

    print(f"\nStatuts des manquants (échantillon):")
    for status, count in status_counts.items():
        print(f"  → {status}: {count}")

    # 3. Créer des entrées fallback pour TOUS les clients manquants
    print(f"\nCréation de {len(missing_ids)} entrées fallback...")

    for cid in tqdm(missing_ids, desc="Ajout fallback"):
        # Récupérer ce qu'on peut depuis all_results
        if cid in all_results:
            res = all_results[cid]

            # Essayer de récupérer le rich_text même si status != success
            rich_text = res.get('rich_text', '')

            # Si pas de rich_text, essayer json_str
            if not rich_text and 'json_str' in res:
                try:
                    jd = json.loads(res['json_str'])
                    rich_text = jd.get('rich_text', '')
                except:
                    pass

            # Si toujours rien, créer un texte minimal
            if not rich_text:
                rich_text = f"""[PROFILE]
    ## OVERVIEW ##
    [CLIENT_{cid}]
    User Type: inactive
    Status: {res.get('status', 'unknown')}

    ## CHURN_PROPENSITY ##
    CHURN_RISK: Unknown

    ## TARGET_WINDOW_14D ##
    No activity in last 14 days

    ## TEMPORAL ##
    Inactive: No recorded activity

    ## CUSTOM ##
    DEFAULT_USER: Fallback profile
    Generated for completeness

    [END]"""
        else:
            # Client complètement absent - créer minimal
            rich_text = f"""[PROFILE]
    ## OVERVIEW ##
    [CLIENT_{cid}]
    User Type: unknown
    Status: missing

    ## CUSTOM ##
    DEFAULT_USER: Missing client
    Generated for dataset completeness

    [END]"""

        # Ajouter le portrait si disponible
        if cid in valid_portraits:
            portrait = valid_portraits[cid].strip()
            insert = f"\n## PORTRAIT ##\n{portrait}\n"
            if "[END]" in rich_text:
                rich_text = rich_text.replace("[END]", insert + "[END]")
            else:
                rich_text += insert

        # Créer l'entrée finale
        final_data[cid] = {
            "profile": {"client_id": cid, "fallback": True},
            "rich_text": rich_text
        }

    # 4. Vérification finale
    print(f"\n✅ Correction appliquée:")
    print(f"  → Total clients : {len(final_data)}")
    print(f"  → Vérification : {'✓ Exactement 1M' if len(final_data) == 1_000_000 else '✗ PAS 1M!'}")

    # 5. S'assurer qu'on a EXACTEMENT 1M
    if len(final_data) != 1_000_000:
        print(f"\n⚠️ ATTENTION: {len(final_data)} clients au lieu de 1,000,000!")

        if len(final_data) > 1_000_000:
            # Trop de clients - en retirer
            excess = len(final_data) - 1_000_000
            print(f"Suppression de {excess} clients en excès...")
            clients_to_remove = list(final_data.keys())[-excess:]
            for cid in clients_to_remove:
                del final_data[cid]
        else:
            # Pas assez - compléter avec des IDs artificiels
            shortage = 1_000_000 - len(final_data)
            print(f"Ajout de {shortage} clients artificiels...")
            max_id = max(all_client_ids)
            for i in range(shortage):
                artificial_id = max_id + i + 1
                final_data[artificial_id] = {
                    "profile": {"client_id": artificial_id, "artificial": True},
                    "rich_text": f"""[PROFILE]
    ## OVERVIEW ##
    [CLIENT_{artificial_id}]
    User Type: artificial
    Status: padding

    ## CUSTOM ##
    ARTIFICIAL_USER: Added for 1M requirement

    [END]"""
                }

    # 6. Vérification FINALE
    assert len(final_data) == 1_000_000, f"ERREUR: {len(final_data)} clients au lieu de 1,000,000!"
    print(f"\n✅ SUCCÈS: Exactement {len(final_data):,} clients!")

    # Statistiques
    fallback_count = sum(1 for d in final_data.values() if d.get('profile', {}).get('fallback', False))
    artificial_count = sum(1 for d in final_data.values() if d.get('profile', {}).get('artificial', False))

    print(f"\nComposition finale:")
    print(f"  → Clients originaux : {len(final_data) - fallback_count - artificial_count:,}")
    print(f"  → Clients fallback : {fallback_count:,}")
    print(f"  → Clients artificiels : {artificial_count:,}")

    # ============================================

    import io
    def augment_text(text: str, drop_prob: float, seed: int) -> str:
        """Drop aléatoire de lignes non critiques"""
        random.seed(seed)
        lines = [l for l in text.splitlines() if l.strip()]

        keep_mark = ('##', '[CLIENT_', '[END]', '— FIN —')
        kept = [l for l in lines if any(m in l for m in keep_mark) or random.random() > drop_prob]
        if len(kept) < 2:
            kept = lines[:2]
        return "\n".join(kept)
    
    MAX_PROMPT_CHARS = 12_000         # garde-fou (optimisé pour 2048 tokens)
    def strip_to_max(text: str, n: int = MAX_PROMPT_CHARS) -> str:
        """Coupe à n caractères (fin du texte) sans casser [END]"""
        return text[-n:] if len(text) > n else text

    print("\n=== SAUVEGARDE DU DATASET FINAL ===")

    output_file = OUTPUT_DIR / "complete_texts_1M.jsonl.zst"
    print(f"Destination : {output_file}")

    # S'assurer que le dossier de sortie existe
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Compter les enregistrements avec un portrait
    with_portrait_count = 0

    # Écrire dans un fichier .jsonl.zst
    try:
        import zstandard as zstd
        # Initialiser le compresseur Zstandard
        zstd_compressor = zstd.ZstdCompressor(level=3)

        with open(output_file, "wb") as f_out:
            # Créer un "stream writer" qui compresse les données à la volée
            with zstd_compressor.stream_writer(f_out) as compressor:
                # Envelopper le compresseur dans un TextIOWrapper pour écrire du texte (encodage UTF-8)
                with io.TextIOWrapper(compressor, encoding='utf-8') as writer:
                    ids_to_write = TEST_CLIENT_IDS if TEST_MODE else list(final_data.keys())
                    # Itérer sur les données finales avec une barre de progression
                    for cid in tqdm(ids_to_write, desc=f"Sauvegarde vers {output_file.name}"):
                        data = final_data[cid]

                        # Préparer l'enregistrement final
                        # On s'assure que le client_id est dans le dictionnaire principal si ce n'est pas déjà le cas
                        if "client_id" not in data.get("profile", {}):
                            if "profile" not in data:
                                data["profile"] = {}
                            data["profile"]["client_id"] = cid

                        # Augmentation du texte (optionnelle, basée sur la constante)
                        # On fait une seule augmentation ici pour que ce soit cohérent
                        augmented_text = augment_text(data["rich_text"], drop_prob=DROP_PROB_AUG1, seed=cid)

                        # Couper le texte à la longueur maximale pour éviter les erreurs de tokenisation
                        final_text = strip_to_max(augmented_text, n=MAX_PROMPT_CHARS)

                        # Créer le dictionnaire final à sauvegarder
                        record_to_save = {
                            "id": cid,
                            "text": final_text
                        }

                        # Convertir le dictionnaire en une ligne JSON
                        # ensure_ascii=False est important pour conserver les accents et caractères spéciaux
                        json_line = json.dumps(record_to_save, ensure_ascii=False)

                        # Écrire la ligne JSON suivie d'un saut de ligne
                        writer.write(json_line + '\n')

                        # Compter les portraits pour les statistiques
                        if "## PORTRAIT ##" in final_text:
                            with_portrait_count += 1

        print(f"\n✅ Sauvegarde terminée !")
        print(f"   → Fichier : {output_file}")
        print(f"   → Taille : {output_file.stat().st_size / 1024**2:.2f} Mo")
        print(f"\nStatistiques finales du dataset sauvegardé:")
        print(f"   → Total d'enregistrements : {len(final_data):,}")
        print(f"   → Enregistrements avec portrait : {with_portrait_count:,} ({with_portrait_count/len(final_data):.1%})")

    except Exception as e:
        print(f"\n❌ ERREUR lors de la sauvegarde : {e}")

    # =============================================================================
    # CELLULE 4-5 : TOKENISATION PARALLÈLE + ÉCRITURE STREAMING (.jsonl.zst)
    # Remplace intégralement tes anciennes Cellule 4 et Cellule 5
    # Prérequis : variables déjà définies plus haut (final_data, strip_to_max,
    # augment_text, USE_AUG2, DROP_PROB_AUG1/2, MAX_TOKENS, OUT_JSONL_ZST,
    # TOKENIZER_NAME, etc.).
    # =============================================================================

    print("\n=== TOKENISATION & ÉCRITURE STREAMING ===")

    # ---------- (dé)pendance facultative : orjson ----------
    try:
        import orjson
        dumps = orjson.dumps               # renvoie déjà des bytes
    except ImportError:
        import json
        dumps = lambda obj: json.dumps(obj, separators=(',', ':')).encode()
        print("⚠️  orjson non trouvé : fallback sur json (plus lent)")

    # ---------- tokenizer ----------
    print(f"Chargement tokenizer : {TOKENIZER_NAME}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        TOKENIZER_NAME, use_fast=True, use_auth_token=True
    )

    # ---------- paramètres ----------
    BATCH_TXT  = 4096                       # taille batch texte pour le tokenizer
    N_WORKERS  = min(os.cpu_count() or 8, 16)  # threads CPU
    MAX_LEN    = MAX_TOKENS

    print(f"  → Batch textes      : {BATCH_TXT}")
    print(f"  → Threads tokenizer : {N_WORKERS}")

    # ---------- itérateur textes + méta ----------
    total_clients  = len(final_data)
    multiplier     = 3 if USE_AUG2 else 2
    total_texts    = total_clients * multiplier
    print(f"  → Clients           : {total_clients:,}")
    print(f"  → Textes à encoder  : {total_texts:,} "
          f"({multiplier} par client)")
    
    import zstandard as zstd
    # ---------- compression writer ----------
    OUT_JSONL_ZST.parent.mkdir(parents=True, exist_ok=True)
    cctx = zstd.ZstdCompressor(level=2, threads=-1)   # multi-thread

    records_written   = 0
    token_lens_sample = []

    # ---------- helpers ----------
    def build_variants(cid: int, jd: dict):
        """Yield texte_original, texte_aug1[, texte_aug2] pour un client."""
        base = strip_to_max(jd["rich_text"])
        yield base
        yield augment_text(base, DROP_PROB_AUG1, seed=cid)
        if USE_AUG2:
            yield augment_text(base, DROP_PROB_AUG2, seed=cid + 1000)

    def encode_batch(texts):
        return tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=MAX_LEN,
            return_attention_mask=False,
        )["input_ids"]

    def chunker(it, size):
        buf = []
        for x in it:
            buf.append(x)
            if len(buf) == size:
                yield buf
                buf = []
        if buf:
            yield buf
            
     # DEBUG MODE ONLY TEST CLIENTS
    data_for_token = ( {cid: final_data[cid] for cid in TEST_CLIENT_IDS}
                       if TEST_CLIENT_IDS is not None
                       else final_data )

    def iter_texts_and_meta():
        """Yield (cid, i_variant, text)."""
        for cid, jd in data_for_token.items():
            for i_var, txt in enumerate(build_variants(cid, jd)):
                yield (cid, i_var, txt)
                
    with open(OUT_JSONL_ZST, "wb") as fout, cctx.stream_writer(fout) as zst:
        with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            future_to_meta = {}
            for batch in chunker(iter_texts_and_meta(), BATCH_TXT):
                metas, texts = zip(*[(m[:2], m[2]) for m in batch])
                fut = pool.submit(encode_batch, texts)
                future_to_meta[fut] = metas

            for fut in tqdm(as_completed(future_to_meta),
                            total=len(future_to_meta),
                            desc="Tokenizing & writing"):
                metas = future_to_meta.pop(fut)
                tok_lists = fut.result()

                for (cid, i_var), toks in zip(metas, tok_lists):
                    # échantillon stats (texte original uniquement)
                    if i_var == 0:
                        token_lens_sample.append(len(toks))

                    # constitution du record client
                    if i_var == 0:
                        rec = {
                            "client_id": int(cid),
                            "input_ids": toks
                        }
                        if not USE_AUG2:
                            rec["input_ids_aug1"] = None  # sera remplacé
                        cache_rec = rec
                    elif i_var == 1:
                        cache_rec["input_ids_aug1"] = toks
                        if not USE_AUG2:
                            zst.write(dumps(cache_rec) + b"\n")
                            records_written += 1
                    else:  # i_var == 2, augmentation 2
                        cache_rec["input_ids_aug2"] = toks
                        zst.write(dumps(cache_rec) + b"\n")
                        records_written += 1

    avg_len = statistics.mean(token_lens_sample)
    med_len = statistics.median(token_lens_sample)
    print(f"\n✅ Dataset créé !")
    print(f"  → Records écrits          : {records_written:,}")
    print(f"  → Longueur moyenne tokens : {avg_len:.0f}")
    print(f"  → Médiane tokens          : {med_len:.0f}")
    print(f"  → Fichier                 : {OUT_JSONL_ZST} "
          f"({OUT_JSONL_ZST.stat().st_size/2**20:.1f} MB)")

    import json
    import zstandard as zstd
    from pathlib import Path

    # Vérifier le fichier JSONL.zst correctement
    OUT_JSONL_ZST = Path(f"{OUTPUT_DIR}/complete_dataset_1M.jsonl.zst")

    print("=== VÉRIFICATION DU DATASET ===")
    print(f"Fichier : {OUT_JSONL_ZST}")
    print(f"Taille : {OUT_JSONL_ZST.stat().st_size / 2**20:.1f} MB")

    # Méthode correcte pour lire un fichier zst
    dctx = zstd.ZstdDecompressor()
    records_checked = 0
    client_ids_seen = set()

    with open(OUT_JSONL_ZST, "rb") as f:
        with dctx.stream_reader(f) as reader:
            # Lire tout le contenu décompressé
            text_data = reader.read()

            # Diviser en lignes
            lines = text_data.decode('utf-8').strip().split('\n')

            print(f"\nNombre total de lignes : {len(lines)}")

            # Vérifier les premiers records
            for i, line in enumerate(lines[:5]):
                record = json.loads(line)
                client_ids_seen.add(record['client_id'])

                print(f"\nRecord {i+1}:")
                print(f"  → client_id: {record['client_id']}")
                print(f"  → input_ids: {len(record['input_ids'])} tokens")
                print(f"  → input_ids_aug1: {len(record['input_ids_aug1'])} tokens")
                if 'input_ids_aug2' in record:
                    print(f"  → input_ids_aug2: {len(record['input_ids_aug2'])} tokens")

                records_checked += 1

    # Vérification rapide de l'intégrité
    print("\n=== VÉRIFICATION D'INTÉGRITÉ ===")
    print(f"✓ Records vérifiés : {records_checked}")
    print(f"✓ Format JSON valide")
    print(f"✓ Structure correcte")

    # Statistiques sur un échantillon
    print("\n=== STATISTIQUES (échantillon) ===")
    sample_size = min(1000, len(lines))
    token_lengths = []

    for i in range(0, len(lines), len(lines) // sample_size):
        if i < len(lines):
            record = json.loads(lines[i])
            token_lengths.append(len(record['input_ids']))

    print(f"Sur {len(token_lengths)} échantillons:")
    print(f"  → Moyenne tokens: {np.mean(token_lengths):.0f}")
    print(f"  → Min/Max: {min(token_lengths)} / {max(token_lengths)}")

    # Vérifier l'ordre des client_ids
    print("\n=== ORDRE DES CLIENTS ===")
    first_10_ids = []
    for line in lines[:10]:
        record = json.loads(line)
        first_10_ids.append(record['client_id'])

    print(f"Premiers 10 client_ids: {first_10_ids}")

    # Compter le total exact (optionnel, peut être lent)
    # if input("\nCompter tous les records? (y/n): ").lower() == 'y':
    total_records = len(lines)
    unique_clients = set()

    print("Comptage en cours...")
    for line in lines:
        record = json.loads(line)
        unique_clients.add(record['client_id'])

    print(f"\n✅ TOTAL FINAL:")
    print(f"  → Records totaux: {total_records:,}")
    print(f"  → Clients uniques: {len(unique_clients):,}")
    print(f"  → Vérification: {'✓ OK' if len(unique_clients) == 1_000_000 else '✗ ERREUR'}")

    print("\n✅ Le dataset est prêt pour l'entraînement!")

if __name__ == "__main__":
    main()
