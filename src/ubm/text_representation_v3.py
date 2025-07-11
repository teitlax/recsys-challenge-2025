# ubm/text_representation_v3.py

from __future__ import annotations
import unsloth
import os, multiprocessing
import os; os.environ["SKIP_URL_GRAPH"] = "1"
# 1) On détecte automatiquement  le nombre de vCPU (sur a2-highgpu-1g → 12)
n_threads = multiprocessing.cpu_count()
print(f"n_threads:{n_threads}")
# For BLAS / OpenMP back-ends
os.environ["OPENBLAS_NUM_THREADS"]  = "12"
os.environ["MKL_NUM_THREADS"]       = str(n_threads)
os.environ["NUMEXPR_MAX_THREADS"]   = str(n_threads)
os.environ["OMP_NUM_THREADS"]       = str(n_threads)

# Polars (already set in your file – keep it!)
os.environ["POLARS_MAX_THREADS"]    = str(n_threads)
import unsloth
# NetworKit – needs an explicit call
import networkit as nk
nk.setNumberOfThreads(n_threads)
import pyarrow.parquet as pq  
import math             #  ← NEW
import statistics        #  ← NEW
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import re
import time
import scipy.sparse as sp
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set, Optional, Union, Any
import logging
from scipy.stats import entropy
from collections import Counter, defaultdict
import os
from tqdm import tqdm
from pathlib import Path
import gc
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import polars as pl
import json, pickle, gc, os, logging, networkx as nx
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm
from itertools import combinations          # <-- IMPORT supplémentaire en haut de fichier
from sklearn.cluster import MiniBatchKMeans
from collections import Counter
from math import log2
import networkit as nk
from networkit import embedding as nk_embed      # NetworKit’s fast Node2Vec
from .portrait_generator import PortraitGenerator, generate_portraits
print("Polars pool size:", pl.threadpool_size())
from math import isfinite
pl.enable_string_cache()
# --- Setup Logging ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# ---------------------------------------------------------------------------
# >>> GLOBAL CONSTANTS (already present in the original file – duplicated   <<<
# >>> here for self‑containment; keep them in sync with the main module)   <<<
# ---------------------------------------------------------------------------
MAX_RICH_TOKENS          : int = 4096  # set to 2048 if you want shorter texts
TOP_FEATURES_PER_SECTION : int = 10    # soft‑cap per section before trimming
IMPLICIT_WEIGHT_REPEAT   : int = 2     # repeat high‑weight tokens N times
SECTIONS_ORDER: List[str] = [
    "OVERVIEW",           # Garde en premier pour le contexte
    "CHURN_PROPENSITY",   # NOUVEAU: Signaux critiques pour la tâche
    "TARGET_WINDOW_14D",  # Activité récente (important pour churn)
    "TEMPORAL",           # Patterns temporels
    "SEQUENCE",           # Séquences comportementales
    "PRICE",              # Sensibilité prix
    "SOCIAL",             # Facteurs sociaux
    "SKU_PROPENSITY",  
    "CAT_PROPENSITY",  
    "PROP_SUBSET_STATS",
    "CUSTOM",
]

# Ajouter ces constantes après SECTIONS_ORDER
SECTION_MARKERS = {
    "OVERVIEW": "[PROFILE]",
    "CHURN_PROPENSITY": "[CHURN]",
    "TARGET_WINDOW_14D": "[RECENT]",
    "TEMPORAL": "[TIME]",
    "SEQUENCE": "[SEQ]",
    "PRICE": "[PRICE]",
    "SOCIAL": "[SOCIAL]",
    "SKU_PROPENSITY": "[SKU]",
    "CAT_PROPENSITY": "[CAT]",
    "PROP_SUBSET_STATS": "[STATS]",
    "CUSTOM": "[MISC]"
}


# ─── HOT-URL & HOT-SKU FILTER THRESHOLDS ────────────────────────────────────
# only keep URLs seen at least this many times when building the bipartite graph
# only keep SKUs with popularity score ≥ this when building the bipartite graph
URL_FREQ_THRESHOLD = 21
SKU_POP_THRESHOLD  = 45
# ---------------------------------------------------------------------------
#                          UTILITY HELPERS                                   #
# ---------------------------------------------------------------------------
# --- Base Class ---
class FeatureExtractorBase:
    """Base class for feature extractors"""
    def __init__(self, parent):
        self.parent = parent # Reference to parent AdvancedUBMGenerator instance
        self.logger = logging.getLogger(self.__class__.__name__)
        # Access shared data via self.parent, e.g., self.parent.sku_properties_dict
        if self.parent.debug_mode: self.logger.setLevel(logging.DEBUG)

    def extract_features(self, client_id: int, events: pl.DataFrame, now: datetime) -> List[str]:
        """Extract features for a client - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement extract_features")
# -------------------------------------------------------------
class TopCategoryFeatureExtractor(FeatureExtractorBase):
    """Met en avant les 100 catégories demandées par la task propensity_category."""

    def __init__(self, parent):
        super().__init__(parent)
        self.top_cats = parent.top_categories     # list[int]

    def extract_features(self, client_id: int, events: pl.DataFrame, now: datetime) -> List[str]:
        if not self.top_cats:
            return ["Top-category list unavailable"]
        cat_ev = events.filter(
            pl.col('category_id').is_in(self.top_cats) &
            pl.col('event_type').is_in(['product_buy', 'add_to_cart', 'page_visit'])
        )
        if cat_ev.is_empty():
            return ["No top-category interactions"]

        rec_w = (
            (pl.lit(now) - cat_ev['timestamp']).dt.total_seconds() / 86400 + 1
        ).pow(-0.5)
        cat_ev = cat_ev.with_columns(rec_w.alias('rw'))

        scores = (
            cat_ev.group_by('category_id')
                  .agg([
                      pl.len().alias('cnt'),
                      pl.sum('rw').alias('rScore'),
                      pl.max('timestamp').alias('last_ts')
                  ])
                  .sort('rScore', descending=True)
        )

        feats = []
        for i, row in enumerate(scores.head(5).iter_rows(named=True), 1):
            delta = (now - row['last_ts']).days
            feats.append(f"TOPCAT{i}:CAT_{row['category_id']} rs={row['rScore']:.2f} "
                         f"cnt={row['cnt']} last={delta}d")

        cov = scores.height / len(self.top_cats)
        feats.append(f"Top-category coverage:{cov:.0%}")
        return feats



def compute_sparse_pagerank(src: np.ndarray,
                            dst: np.ndarray,
                            weights: np.ndarray,
                            alpha: float = 0.85,
                            tol: float = 1e-6,
                            max_iter: int = 1000) -> dict[int, float]:
    """
    Parallel PageRank using NetworKit.  ~10-20× faster than NetworkX
    on million-edge graphs.
    """
    g, id2orig = _nk_graph_from_edges(src, dst, weights, directed=True)
    pr = nk.centrality.PageRank(
        g, damp=alpha, tol=tol, maxIterations=max_iter, normalized=False
    )
    pr.run()
    scores = pr.scores()          # list[float] aligned with 0…n-1 ids
    return {int(id2orig[i]): s for i, s in enumerate(scores)}

# ------------------------------------------------------------------
# NetworKit helpers
# ------------------------------------------------------------------
# --- BEGIN PATCH: helpers -----------------------------------------------------
import numpy as np

def _nk_graph_from_edges(src: np.ndarray,
                         dst: np.ndarray,
                         w:   np.ndarray,
                         directed: bool = True) -> tuple[nk.Graph, np.ndarray]:
    """
    Build a NetworKit graph + id→label array in one pass.
    """
    nodes, inverse = np.unique(np.concatenate([src, dst]), return_inverse=True)
    g = nk.Graph(len(nodes), weighted=True, directed=directed)
    half = len(src)
    for u, v, weight in zip(inverse[:half], inverse[half:], w):
        if u == v:
            continue
        eid = g.addEdge(u, v, w=float(weight))
        if eid == -1:                          # multi-edge → accumulate
            eid = g.edgeId(u, v)
            g.setWeight(u, v, g.weight(u, v) + float(weight))
    return g, nodes                            # nodes[i] = original id/label


def compute_sparse_pagerank(src: np.ndarray,
                            dst: np.ndarray,
                            weights: np.ndarray,
                            alpha: float = 0.85,
                            tol: float = 1e-6,
                            max_iter: int = 1_000) -> dict[int, float]:
    g, id2orig = _nk_graph_from_edges(src, dst, weights, directed=True)

    pr = nk.centrality.PageRank(g, damp=alpha, tol=tol, normalized=False)

    # older NetworKit (< 10) does *not* accept maxIterations as __init__ kw
    if hasattr(pr, "setMaxIterations"):
        pr.setMaxIterations(max_iter)

    pr.run()
    scores = pr.scores()                       # list[float] aligned with 0…n-1 ids
    return {int(id2orig[i]): s for i, s in enumerate(scores)}



def _shannon_entropy(counter: Counter) -> float:
    """Shannon entropy in bits from a Counter of counts."""
    n = sum(counter.values())
    if n == 0:
        return 0.0
    return -sum((c / n) * log2(c / n) for c in counter.values())

def _approx_token_len(text: str) -> int:
    """Very cheap proxy for token count (≈ whitespace split)."""
    return len(text.split())


def _truncate_to_max_tokens(lines: List[str], limit: int) -> List[str]:
    """Greedy keep‑from‑start strategy (safer for ordered / weighted chunks)."""
    kept: List[str] = []
    for ln in lines:
        if _approx_token_len("\n".join(kept + [ln])) > limit:
            break
        kept.append(ln)
    return kept


def _top_k_features(features: List[str], k: int) -> Tuple[List[str], List[str]]:
    """Return `(top_k, overflow)` lists – *overflow* can be shuffled/dropped."""
    if len(features) <= k:
        return features, []
    return features[:k], features[k:]


def _repeat_for_weight(lines: List[str], repeat: int) -> List[str]:
    """Naïve implicit weight: duplicate *every* line *repeat* times."""
    if repeat <= 1:
        return lines
    out: List[str] = []
    for ln in lines:
        out.extend([ln] * repeat)
    return out
def bucketize_days(delta_days: int) -> str:
    if delta_days <= 3:
        return "R_0-3d"
    if delta_days <= 7:
        return "R_3-7d"
    if delta_days <= 30:
        return "R_7-30d"
    return "R_30+d"

POP_QUANT_EDGES: list = []   # global mutable (sera rempli une fois)

def pop_bin(score: float) -> str:
    """Retourne la quantile de popularité (Q0 à Q4). Q0 = inconnu / score manquant."""
    if score is None or math.isnan(score):
        return "Q0"
    # Les bords sont calculés et stockés dans AdvancedUBMGenerator._compute_product_popularities
    for i, edge in enumerate(POP_QUANT_EDGES, start=1):
        if score <= edge:
            return f"Q{i}"
    return "Q4"

# ────────────────────────────────────────────────────────────────────────────
# Helper : transforme le dictionnaire section→liste en texte final
# ────────────────────────────────────────────────────────────────────────────
def _build_rich_text(
    section_map: dict[str, list[str]],
    max_tokens: int = 3500,
    implicit_repeat: int = 2,
    top_per_section: int = 10,
    shuffle_seed: int | None = None,
    use_markers: bool = True,  # NOUVEAU paramètre
) -> str:
    """
    Version optimisée avec markers et déduplication améliorée
    """
    import random
    import itertools
    from collections import OrderedDict

    rnd = random.Random(shuffle_seed)
    
    # Déduplication globale des features
    seen_features = set()
    deduped_sections = {}
    
    for section, items in section_map.items():
        deduped_items = []
        for item in items:
            # Normaliser pour la déduplication
            normalized = item.strip().lower()
            if normalized not in seen_features:
                seen_features.add(normalized)
                deduped_items.append(item)
        deduped_sections[section] = deduped_items

    def _truncate(tokens: list[str], limit: int) -> list[str]:
        total = 0
        out = []
        for tok in tokens:
            total += len(tok.split())
            if total > limit:
                break
            out.append(tok)
        return out

    lines: list[str] = []

    for section in SECTIONS_ORDER:
        items = deduped_sections.get(section, [])
        if not items:
            continue

        # 1. Limiter au top K
        items = items[:top_per_section]

        # 2. Répétition implicite pour les items importants
        repeated = []
        for item in items:
            # Les features de churn/propensity sont toujours répétées
            if "CHURN_" in item or "PROPENSITY" in item or "**" in item:
                repeated.extend([item] * implicit_repeat)
            else:
                repeated.append(item)

        # 3. Shuffle léger pour la variété (sauf les premières)
        if len(repeated) > 3:
            first_items = repeated[:2]  # Garde les 2 premiers
            rest_items = repeated[2:]
            rnd.shuffle(rest_items)
            repeated = first_items + rest_items

        # 4. Ajouter le marqueur de section et le contenu
        if use_markers and section in SECTION_MARKERS:
            lines.append(f"{SECTION_MARKERS[section]}")
        lines.append(f"## {section} ##")
        lines.extend(repeated)

    # 5. Coupe globale au nombre de tokens demandé
    lines = _truncate(lines, max_tokens)

    # 6. Ajouter un marqueur de fin
    if use_markers:
        lines.append("[END]")

    return "\n".join(lines)






class TemporalFeatureExtractor(FeatureExtractorBase):
    """Extract temporal patterns from user behavior, enhanced with recency and inactivity."""

    def extract_features(self, client_id: int, events: pl.DataFrame, now: datetime) -> List[str]:
        if events.height == 0: return ["No activity data for temporal analysis"]
        features = []
        try:
            # Use a single timestamp access for efficiency
            timestamps_sorted = events.sort("timestamp")['timestamp']
            last_ts = timestamps_sorted.max()
            first_ts = timestamps_sorted.min()

            # Appeler les sous-méthodes
            self._extract_daily_patterns(events, features)
            self._extract_weekly_patterns(events, features)
            self._extract_session_patterns(events, features, timestamps_sorted) # Passer les timestamps triés
            self._extract_recency_and_frequency(events, features, now, last_ts) # Enhanced recency
            self._extract_inactivity_gaps(features, timestamps_sorted, now, last_ts) # New inactivity analysis

        except Exception as e:
            self.logger.error(f"Err temporal client {client_id}: {e}", exc_info=self.parent.debug_mode)
            features.append("Err temporal")
        return features

    def _extract_daily_patterns(self, events: pl.DataFrame, features: List[str]) -> None:
            try:
                # ✅ FIX: Utiliser pl.count() au lieu de pl.col().count()
                hour_counts = (
                    events
                    .with_columns(pl.col('timestamp').dt.hour().alias('hour_of_day'))
                    .group_by('hour_of_day')
                    .agg(pl.count().alias("count"))  # ← Changé ici
                    .sort('hour_of_day')
                )
                
                if hour_counts.height > 0:
                    max_count_row = hour_counts.sort("count", descending=True).row(0, named=True)
                    if max_count_row:
                        max_count = max_count_row['count']
                        peak_hours = hour_counts.filter(pl.col('count') >= 0.8 * max_count)['hour_of_day'].drop_nulls().to_list()
                        if peak_hours: 
                            features.append(f"Peak hours: {', '.join([f'{h}:00' for h in sorted(peak_hours)])}")
                        
                        # Segments de la journée
                        morning = [h for h in peak_hours if 5 <= h < 12]
                        afternoon = [h for h in peak_hours if 12 <= h < 18]
                        evening = [h for h in peak_hours if h >= 18 or h < 5]
                        
                        time_segments = [(len(morning), "Morning"), (len(afternoon), "Afternoon"), (len(evening), "Evening")]
                        dominant = max([s for s in time_segments if s[0] > 0], key=lambda x: x[0], default=(0, None))
                        if dominant[1]: 
                            features.append(f"{dominant[1]}-dominant")
            except Exception as e:
                self.logger.debug(f"Err daily: {e}")
                features.append("Err daily patterns")

    # ── TemporalFeatureExtractor._extract_weekly_patterns ──
    def _extract_weekly_patterns(self, events: pl.DataFrame, features: List[str]) -> None:
        try:
            # ✅ FIX: Utiliser pl.count() et créer la colonne weekday d'abord
            day_counts = (
                events
                .with_columns(pl.col('timestamp').dt.weekday().alias('day_of_week'))
                .group_by('day_of_week')
                .agg(pl.count().alias('count'))  # ← Changé ici
                .sort('day_of_week')
            )

            day_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu',
                         4: 'Fri', 5: 'Sat', 6: 'Sun', 7: 'Sun'}

            if day_counts.height == 0:
                return

            max_count = day_counts['count'].max()
            
            if max_count is None:
                return

            # Récupérer les jours de pointe
            peak_days = (
                day_counts
                .filter(pl.col('count') >= 0.8 * max_count)['day_of_week']
                .to_list()
            )
            
            # Normaliser les jours (au cas où il y aurait des valeurs > 6)
            peak_days = [(d % 7) for d in peak_days]

            if peak_days:
                features.append(
                    "Active days: " + ", ".join(day_names[d] for d in sorted(peak_days))
                )

                weekday = any(d < 5 for d in peak_days)
                weekend = any(d >= 5 for d in peak_days)
                if weekday and not weekend:
                    features.append("Weekday-dominant")
                elif weekend and not weekday:
                    features.append("Weekend-dominant")

        except Exception as e:
            self.logger.debug(f"Err weekly: {e}")
            features.append("Err weekly patterns")

            
    # ── dans la même classe ──────────────────────────────────────────────────────
    def _split_sessions(self, timestamps_sorted: pl.Series, gap: int = 30) -> pl.Series:
        """
        Renvoie un id de session (0,1,2,…) pour chaque événement.
        Nouveau numéro si écart > gap minutes.
        """
        time_diff = timestamps_sorted.diff().dt.total_seconds() / 60
        is_new    = time_diff.is_null() | (time_diff > gap)
        return is_new.cum_sum()          



    
    def _extract_session_patterns(self,
                                  events: pl.DataFrame,
                                  features: List[str],
                                  timestamps_sorted: pl.Series) -> None:
        try:
            if events.height <= 1:
                return
            # 1. scinder en sessions
            sess_ids = self._split_sessions(timestamps_sorted)        # <-- NEW
            sess_df = pl.DataFrame({'sid': sess_ids, 'ts': timestamps_sorted})

            # 2. stats par session (durée & #events)
            sess_stats = (
                sess_df.group_by('sid')
                       .agg(pl.min('ts').alias('start'),
                            pl.max('ts').alias('end'),
                            pl.count().alias('cnt'))
                       .with_columns(
                           ((pl.col('end') - pl.col('start'))
                            .dt.total_seconds() / 60).alias('dur'))   # minutes
            )
            n_sessions = sess_stats.height         
            if n_sessions == 0:
                return
            # mix jour/nuit des débuts de sessions
            starts = sess_stats['start']
            night  = (starts.dt.hour() >= 18) | (starts.dt.hour() < 5)
            ratio  = night.sum() / n_sessions
            if ratio > 0.7:
                features.append("Mostly evening sessions")
            elif ratio < 0.3:
                features.append("Mostly daytime sessions")
            


            # 3. moyennes
            avg_dur = sess_stats['dur'].mean()        # durée moyenne en minutes
            avg_cnt = sess_stats['cnt'].mean()        # évènements / session

            features.append(f"Sessions: {n_sessions}")
            if avg_dur < 5:
                features.append(f"Typically very short sessions (~{avg_dur:.1f} m)")
            elif avg_dur > 60:
                features.append(f"Typically long sessions (~{avg_dur:.1f} m)")

            if avg_cnt < 3:
                features.append(f"Typically shallow sessions (~{avg_cnt:.1f} evt)")
            elif avg_cnt > 15:
                features.append(f"Typically deep sessions (~{avg_cnt:.1f} evt)")

        except Exception as e:
            self.logger.debug(f"Err session: {e}")
            features.append("Err session patterns")


    def _extract_recency_and_frequency(self, events: pl.DataFrame, features: List[str], now: datetime, last_ts: Optional[datetime]) -> None:
        """Enhanced recency and recent frequency calculations."""
        try:
            if events.height == 0 or last_ts is None:
                 features.append("Recency: No Activity")
                 return

            days_since_last = (now - last_ts).days
            features.append(f"Days Since Last Activity: {days_since_last}")
            if days_since_last <= 7: features.append("Activity Status: Very Recent")
            elif days_since_last <= 30: features.append("Activity Status: Moderately Recent")
            elif days_since_last <= 90: features.append("Activity Status: Lapsed")
            else: features.append("Activity Status: Very Lapsed/Inactive")

            # Recency of specific important actions
            last_purchase_ts = events.filter(pl.col('event_type') == pl.lit('product_buy', dtype=pl.Categorical))['timestamp'].max()
            last_cart_add_ts = events.filter(pl.col('event_type') == pl.lit('add_to_cart', dtype=pl.Categorical))['timestamp'].max()

            if last_purchase_ts: features.append(f"Days Since Last Purchase: {(now - last_purchase_ts).days}")
            else: features.append("Days Since Last Purchase: Never")
            if last_cart_add_ts: features.append(f"Days Since Last Cart Add: {(now - last_cart_add_ts).days}")
            else: features.append("Days Since Last Cart Add: Never")

            # Recent Frequency (last 30 days)
            cutoff_30d = now - timedelta(days=30)
            recent_events = events.filter(pl.col("timestamp") >= cutoff_30d)

            if recent_events.height > 0:
                event_count_30d   = recent_events.height

                # quick-win : l’utilisateur était absent >90 j et revient dans les 30 derniers jours
                if days_since_last > 90:
                    features.append("Reactivation after long dormancy")

                active_days_30d   = recent_events['timestamp'].dt.date().n_unique()
                first_ts_in_30d   = recent_events['timestamp'].min()
                observed_days     = max(1, (last_ts - first_ts_in_30d).days + 1)  # au moins 1 jour
                active_ratio      = active_days_30d / min(30, observed_days)

                features.append(
                    f"Activity (Last 30d): {event_count_30d} events over "
                    f"{active_days_30d} days (Active Ratio: {active_ratio:.1%})"
                )
            else:
                features.append("Activity (Last 30d): None")


        except Exception as e:
            self.logger.debug(f"Err recency/frequency: {e}")
            features.append("Err recency/frequency")
            
        # Fraîcheur globale
        fresh14 = events.filter(pl.col('timestamp') >= now - timedelta(days=14)).height / events.height
        if fresh14 > 0.5:
            features.append("Recent-heavy activity (<14d)")
        

    def _extract_inactivity_gaps(self, features: List[str], timestamps_sorted: pl.Series, now: datetime, last_ts: Optional[datetime]) -> None:
        """Analyze inactivity gaps between events."""
        try:
            if timestamps_sorted.len() <= 1 or last_ts is None: return # Need at least 2 events

            timestamps_list = timestamps_sorted.to_list()
            gaps_days = [(timestamps_list[i+1] - timestamps_list[i]).total_seconds() / (3600 * 24) for i in range(len(timestamps_list)-1)]

            if gaps_days:
                 max_gap = np.max(gaps_days)
                 features.append(f"Max Inactivity Gap: {max_gap:.1f} days")
                 gaps_gt_7d = sum(1 for g in gaps_days if g > 7)
                 gaps_gt_30d = sum(1 for g in gaps_days if g > 30)
                 if gaps_gt_30d > 0: features.append(f"Notable Gaps (>30d): {gaps_gt_30d}")
                 elif gaps_gt_7d > 0: features.append(f"Notable Gaps (>7d): {gaps_gt_7d}")

            # Check recent gap (since last event)
            days_since_last = (now - last_ts).days
            if days_since_last > 30:
                 features.append("Recent Status: Currently Inactive (>30 days)")
            elif days_since_last > 7:
                 features.append("Recent Status: Currently Lapsing (>7 days)")

        except Exception as e:
            self.logger.debug(f"Err inactivity gaps: {e}")
            features.append("Err inactivity gaps")

    def _count_sessions_from_timestamps(self, timestamps_sorted: pl.Series, session_gap_minutes: int = 30) -> int:
        """Helper to count sessions just from a sorted timestamp series."""
        if timestamps_sorted.len() <= 1: return timestamps_sorted.len()
        try:
            time_diffs_minutes = timestamps_sorted.diff().dt.total_seconds() / 60
            # Un diff() donne null pour le premier -> is_null() est vrai -> début de session
            # Ensuite, vérifier si diff > gap
            session_starts = time_diffs_minutes.is_null() | (time_diffs_minutes > session_gap_minutes)
            num_sessions = session_starts.sum()
            return int(num_sessions)
        except Exception as e:
            self.logger.error(f"Error counting sessions from timestamps: {e}")
            return 1 # Fallback


class SequenceFeatureExtractor(FeatureExtractorBase):
    """Extract sequential behavior patterns"""
    def extract_features(self, client_id: int, events: pl.DataFrame, now: datetime) -> List[str]:
        if events.height < 2:
            return ["Very limited activity"]
    
        features: List[str] = []
        try:
            # Polars → Python list (no Pandas round-trip)
            event_types = (
                events.sort("timestamp")
                      ["event_type"]
                      .to_list()
            )
    
            if event_types.count("product_buy") == 1:
                features.append("Single-purchase buyer")
    
            self._extract_event_sequences(event_types, features)
            self._extract_purchase_funnel(events, features)
    
            if 'category_id' in events.columns:
                self._extract_Browse_sequences(events, features)
    
        except Exception as e:
            self.logger.error(f"Error extracting sequence features for client {client_id}: {e}",
                              exc_info=self.parent.debug_mode)
            features.append("Error during sequence feature extraction.")
        return features


    def _extract_event_sequences(self, event_types: list, features: List[str]) -> None:
        try:
            if len(event_types) < 3: return
            trigrams = self._create_ngrams(event_types, 3)
            if trigrams:
                trigram_counts = Counter(trigrams); total_trigrams = len(trigrams)
                for trigram, count in trigram_counts.most_common(2):
                    if count >= 2:
                        sequence_str = " -> ".join(trigram); freq_pct = (count / total_trigrams) * 100
                        features.append(f"Common sequence: {sequence_str} ({count} times, {freq_pct:.1f}%)")
            fourgrams = self._create_ngrams(event_types, 4)
            if fourgrams:
                fourgram_counts = Counter(fourgrams); total_fourgrams = len(fourgrams)
                for fourgram, count in fourgram_counts.most_common(1):
                    if count >= 2:
                        sequence_str = " -> ".join(fourgram); freq_pct = (count / total_fourgrams) * 100
                        features.append(f"Common extended path: {sequence_str} ({count} times, {freq_pct:.1f}%)")
            # ---------- Ping‑pong A‑B‑A‑B detection ----------
            if len(event_types) >= 4:
                pp_count = 0
                for j in range(len(event_types) - 3):
                    a, b, c, d = event_types[j:j+4]
                    if a == c and b == d and a != b:
                        pp_count += 1
                if pp_count >= 2:
                    features.append(f"Ping‑pong navigation pattern ({pp_count} times)")             
        except Exception as e:
            self.logger.debug(f"Error extracting event sequences: {e}")
            features.append("Error extracting event sequences")

    def _extract_purchase_funnel(self, events: pl.DataFrame, features: List[str]) -> None:
        try:
            tmp = events.group_by('event_type').agg(pl.count().alias('count'))
            event_counts = {row['event_type']: row['count'] for row in tmp.iter_rows(named=True)}
            views = event_counts.get('page_visit', 0); searches = event_counts.get('search_query', 0)
            cart_adds = event_counts.get('add_to_cart', 0); purchases = event_counts.get('product_buy', 0)
            if views == 0 and searches == 0 and cart_adds == 0 and purchases == 0: return

            funnel_stages = ["Purchase funnel analysis:"]
            total_starts = views + searches
            if total_starts > 0: funnel_stages.append(f"  Starts (View/Search): {total_starts}")
            if cart_adds > 0:
                cart_rate = (cart_adds / total_starts) * 100 if total_starts > 0 else 0
                funnel_stages.append(f"  Cart Adds: {cart_adds} ({cart_rate:.1f}% of starts)")
                if purchases > 0:
                    purchase_rate_from_cart = (purchases / cart_adds) * 100
                    funnel_stages.append(f"  Purchases: {purchases} ({purchase_rate_from_cart:.1f}% of cart adds)")
                    purchase_rate_from_start = (purchases / total_starts) * 100 if total_starts > 0 else 0
                    funnel_stages.append(f"  Overall Conversion: {purchase_rate_from_start:.2f}% from start")
            elif purchases > 0:
                funnel_stages.append(f"  Purchases: {purchases} (direct or uncaptured cart add)")

            if len(funnel_stages) > 1: features.extend(funnel_stages)
        except Exception as e:
            self.logger.debug(f"Error extracting purchase funnel: {e}")
            features.append("Error extracting purchase funnel")

    def _extract_Browse_sequences(self, events: pl.DataFrame, features: List[str]) -> None:
        try:
            page_visits = (
                events.filter(
                    (pl.col('event_type') == pl.lit('page_visit', dtype=pl.Categorical)) &
                    pl.col('category_id').is_not_null()
                )
                .sort('timestamp')
            )
            if page_visits.height < 3:
                return
    
            # 1) add 30-minute session ids (vectorised)
            gaps = page_visits['timestamp'].diff().dt.total_seconds() / 60
            page_visits = page_visits.with_columns(
                ((gaps.is_null()) | (gaps > 30)).cum_sum().alias('sid')
            )
    
            # 2) iterate zero-copy over sessions
            from collections import Counter
            trigram_counter = Counter()
    
            for sess in page_visits.partition_by('sid', as_dict=False):
                cats = (sess['category_id']
                          .drop_nulls()
                          .unique()
                          .to_list())
                if len(cats) >= 3:
                    trigrams = self._create_ngrams(cats, 3)
                    trigram_counter.update(trigrams)
    
            if trigram_counter:
                top_tri, cnt = trigram_counter.most_common(1)[0]
                if cnt >= 2:
                    features.append(
                        f"Common category sequence: "
                        f"{' -> '.join([f'CAT_{c}' for c in top_tri])} ({cnt}x)"
                    )
        except Exception as e:
            self.logger.debug(f"Error extracting Browse sequences: {e}")
            features.append("Error extracting Browse sequences")


    def _create_ngrams(self, sequence: list, n: int) -> List[tuple]:
        if len(sequence) < n: return []
        return [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]


class GraphFeatureExtractor(FeatureExtractorBase):
    """Extract graph-based behavioral features"""

    def extract_features(self, client_id: int, events: pl.DataFrame, now: datetime) -> List[str]:
        if events.height < 5: return []
        features = []
        try:
            if 'category_id' in events.columns:
                 self._extract_category_graph_features(client_id, events, features)
            else: features.append("Category graph skipped (no category_id).")
            if events.height >= 10 and 'sku' in events.columns:
                 self._extract_product_graph_features(client_id, events, features)
            else: features.append("Product graph skipped (few events or no sku).")
        except Exception as e: self.logger.error(f"Err graph client {client_id}: {e}"); features.append("Err graph")
        return features

    def _extract_category_graph_features(self, client_id: int,
                                         events: pl.DataFrame,
                                         features: list[str]) -> None:
        """
        Same logic as before but using NetworKit for the expensive bits.
        """
        try:
            page_visits = events.filter(
                (pl.col('event_type') == pl.lit('page_visit', dtype=pl.Categorical))
                & pl.col('category_id').is_not_null()
            ).sort('timestamp')

            if page_visits.height < 5:
                return

            cats = page_visits['category_id']
            uniq_mask = cats.diff().fill_null(1) != 0
            categories = cats.filter(uniq_mask).to_list()

            if len(categories) < 2:
                return

            # ------------------------------------------------------------------
            # 1) build directed weighted edge list
            src, dst = np.array(categories[:-1], dtype=int), np.array(categories[1:], dtype=int)
            weights  = np.ones_like(src, dtype=float)
            g, id2orig = _nk_graph_from_edges(src, dst, weights, directed=True)

            features.append(f"Category exploration: {g.numberOfNodes()} unique.")

            # ------------------------------------------------------------------
            # 2) PageRank
            pr = nk.centrality.PageRank(g, damp=0.85, tol=1e-4)
            pr.run()
            scores = pr.scores()
            if scores:
                top_idx = int(np.argmax(scores))
                top_cat = int(id2orig[top_idx])
                features.append(f"Dominant cat (PR): CAT_{top_cat} ({scores[top_idx]:.3f})")

            # ------------------------------------------------------------------
            # 3) Average clustering coefficient (undirected view)
            lu = nk.clustering.LocalClusteringCoefficient(g.toUndirected(), weighted=True)
            lu.run()
            avg_clust = sum(lu.scores()) / g.numberOfNodes()
            features.append(f"Avg cat clustering: {avg_clust:.3f}")

            # ------------------------------------------------------------------
            # 4) Top transition (weight ≥ 2)
            best_w, best_pair = 0, None
            for (u, v) in g.iterEdges():
                w = g.weight(u, v)
                if w > best_w:
                    best_w, best_pair = (u, v)
            if best_pair and best_w >= 2:
                u, v = best_pair
                features.append(f"Top cat transition: CAT_{int(id2orig[u])}->CAT_{int(id2orig[v])} ({int(best_w)}x)")

        except Exception as e:
            self.logger.debug(f"Err cat graph: {e}")
            features.append("Err cat graph")

            
    # ─────────────────────────────────────────────────────────────────────
    # Product-level co-interaction graph (NetworKit, fully vectorised)
    # ─────────────────────────────────────────────────────────────────────
    def _extract_product_graph_features(self,
                                        client_id: int,
                                        events: pl.DataFrame,
                                        features: list[str]) -> None:
        """
        Build a session-level SKU co-interaction graph, compute basic
        structure metrics and a central product using NetworKit.
        Much faster and safer than the previous NetworkX version.
        """
        try:
            # 0) keep only page / cart / buy events with a valid SKU
            rel_evt = (
                events.filter(
                    pl.col('event_type')
                      .is_in(['page_visit', 'add_to_cart', 'product_buy'])
                    & pl.col('sku').is_not_null()
                )
                .sort('timestamp')
            )

            if rel_evt.height < 3:
                return

            # 1) rebuild 30-minute sessions   (Polars → NumPy, no slow loops)
            sess_gap = 30        # minutes
            ts = rel_evt['timestamp']
            gaps_min = (
                ts.diff().dt.total_seconds()      # diff is None at row 0
                  .fill_null(sess_gap * 60 + 1)   # force new session on 1st row
                  / 60                            # seconds → minutes
            )
            sess_id = gaps_min.gt(sess_gap).cum_sum()  # fast cumulative ids
            rel_evt = rel_evt.with_columns(pl.Series('sid', sess_id))
            rel_evt = rel_evt.filter(pl.col('sku').is_in(self.parent.top_skus))

            # 2) count SKU co-occurrences inside each session
            from itertools import combinations
            from collections import Counter
            pair_cnt = Counter()

            for _, sess in rel_evt.group_by('sid'):
                skus = sess['sku'].drop_nulls().unique().to_list()
                if len(skus) < 2:
                    continue
                for i, j in combinations(sorted(skus), 2):
                    pair_cnt[(int(i), int(j))] += 1

            if not pair_cnt:
                return

            pairs   = np.array(list(pair_cnt.keys()),   dtype=int)
            weights = np.array(list(pair_cnt.values()), dtype=float)

            # 3) build NetworKit graph (undirected, weighted)
            g, id2sku = _nk_graph_from_edges(pairs[:, 0], pairs[:, 1],
                                             weights, directed=False)

            n = g.numberOfNodes()
            e = g.numberOfEdges()
            features.append(f"Product exploration: {n} unique.")

            # density (undirected simple graph)
            density = (2 * e) / (n * (n - 1)) if n > 1 else 0.0
            features.append(f"Product graph density: {density:.3f}")
            if density > 0.50:
                features.append("Dense product co-interaction.")
            elif density < 0.10:
                features.append("Sparse product co-interaction.")

            # 4) weighted degree centrality (normalised)
            deg = nk.centrality.DegreeCentrality(g, True, True)
            deg.run()
            scores = deg.scores()
            if scores:
                top_idx = int(np.argmax(scores))
                central_sku = int(id2sku[top_idx])
                features.append(
                    f"Central product (degree): SKU_{central_sku} "
                    f"({scores[top_idx]:.0f})"
                )

        except Exception as e:
            self.logger.debug(f"Err product graph: {e}")
            features.append("Err product graph")

class IntentFeatureExtractor(FeatureExtractorBase):
    """Extract search intent and interest patterns, with simple cart abandon signal."""

    def extract_features(self, client_id: int, events: pl.DataFrame, now: datetime) -> List[str]:
        features = []
        if events.height == 0: return ["No activity data for intent analysis"]
        try:
            self._extract_search_intent(events, features)
            self._extract_Browse_intent(events, features)
            self._extract_funnel_position(events, features, now) # Passer now
            self._extract_cart_abandon_signal(events, features) # Nouvelle méthode simple
        except Exception as e:
            self.logger.error(f"Error extracting intent features for client {client_id}: {e}", exc_info=self.parent.debug_mode)
            features.append("Error during intent feature extraction.")
        return features

    def _extract_search_intent(self, events: pl.DataFrame, features: List[str]) -> None:
        try:
            search_events = events.filter(pl.col('event_type') == pl.lit('search_query', dtype=pl.Categorical))
            if search_events.height == 0: features.append("No search events."); return
            features.append(f"Total searches: {search_events.height}")
            if 'query' in search_events.columns:
                query_hashes = search_events.filter(pl.col('query').is_not_null()).select(pl.col('query').hash().alias('query_hash'))['query_hash']
                if query_hashes.len() > 0:
                    unique_hashes_count = query_hashes.n_unique()
                    features.append(f"Unique search hashes: {unique_hashes_count}")
                    if unique_hashes_count < query_hashes.len():
                        top_hash_info = query_hashes.value_counts().sort(by="count", descending=True).head(1)
                        if top_hash_info.height > 0:
                            top_hash, top_count = top_hash_info.row(0)
                            features.append(f"Top search hash: [QUERY_{top_hash}] ({top_count}x)")
                else: features.append("No valid search queries found.")
            else: features.append("Query column missing.")
        except Exception as e: self.logger.debug(f"Err search intent: {e}"); features.append("Err search intent.")

    def _extract_Browse_intent(self, events: pl.DataFrame, features: List[str]) -> None:
        # Initialize cat_counts and total_cat_visits to default values
        cat_counts = pl.DataFrame()  # Default empty DataFrame
        total_cat_visits = 0         # Default to 0
    
        try:
            page_visits = events.filter(pl.col('event_type') == pl.lit('page_visit', dtype=pl.Categorical))
            
            if page_visits.height < 3:
                return # Early exit if not enough page visits to analyze
    
            num_sessions = self._count_sessions(events)
            
            if num_sessions > 0:
                visits_per_session = page_visits.height / num_sessions
                if visits_per_session > 15:
                    features.append(f"Intensive browser (~{visits_per_session:.1f} pages/session)")
                elif visits_per_session < 3: 
                    features.append(f"Shallow browser (~{visits_per_session:.1f} pages/session)")
    
            if 'category_id' in page_visits.columns:
                valid_category_page_visits = page_visits.filter(pl.col('category_id').is_not_null())
                
                if not valid_category_page_visits.is_empty():
                    calculated_cat_counts = valid_category_page_visits.group_by('category_id').agg(
                        pl.col('category_id').count().alias('count')  # Changer pl.count() en pl.col().count()
                    )
                
                    if not calculated_cat_counts.is_empty():
                        cat_counts = calculated_cat_counts
                        if 'count' in cat_counts.columns:
                            total_cat_visits = cat_counts.get_column('count').sum()
                        else:
                            self.logger.debug("Browse intent: 'count' column unexpectedly missing after grouping categories.")
                            total_cat_visits = 0
    
                        if total_cat_visits > 0 and 'count' in cat_counts.columns: 
                            max_cat_visits = cat_counts.get_column('count').max()
                            if max_cat_visits is not None: 
                                top_category_share = (max_cat_visits / total_cat_visits)
                                if top_category_share > 0.75:
                                    features.append(f"Single-category focus browse")
                                elif top_category_share < 0.40 and cat_counts.height >= 3:
                                    features.append(f"Multi-category explorer browse")
    
        except Exception as e:
            client_id_info = f"client {events['client_id'][0]}" if not events.is_empty() and 'client_id' in events.columns else "unknown client"
            self.logger.debug(f"Error in _extract_Browse_intent for {client_id_info}: {e}", exc_info=True)
            features.append("Err browse intent.")

        # --- Diversity score ---
        # This section is now safe because cat_counts and total_cat_visits are always defined.
        if not cat_counts.is_empty() and cat_counts.height > 1 and total_cat_visits > 0:
            try:
                if 'count' in cat_counts.columns: # Double check 'count' column exists
                    probs_series = cat_counts.get_column('count') / total_cat_visits 
                    probs_numpy = probs_series.to_numpy()
                    
                    # Filter out zero or negative probabilities to avoid log2 issues and ensure valid input
                    probs_filtered = probs_numpy[probs_numpy > 0]
                    
                    if probs_filtered.size > 0: # Ensure there are valid probabilities after filtering
                        H = float(-(probs_filtered * np.log2(probs_filtered)).sum()) # Shannon entropy
                        features.append(f"Category Diversity:{H:.2f}")
                    else:
                        self.logger.debug("Diversity score not calculated: no valid probabilities after filtering.")
                else:
                    self.logger.debug("Diversity score not calculated: 'count' column missing in cat_counts.")
            except Exception as e_diversity:
                 self.logger.debug(f"Error calculating diversity score: {e_diversity}", exc_info=True)
                 # features.append("ErrCalculatingDiversityScore") # Optionally add a specific error feature



    def _extract_funnel_position(self, events: pl.DataFrame, features: List[str], now: datetime) -> None:
        try:
            tmp = events.group_by('event_type').agg(pl.col('event_type').count().alias('count'))  # Changer pl.count() en pl.col().count()
            event_counts = {row['event_type']: row['count'] for row in tmp.iter_rows(named=True)}
            views = event_counts.get('page_visit', 0); searches = event_counts.get('search_query', 0)
            cart_adds = event_counts.get('add_to_cart', 0); purchases = event_counts.get('product_buy', 0)

            if purchases > 0: features.append("Funnel Stage: Conversion")
            elif cart_adds > 0: features.append("Funnel Stage: Consideration")
            elif searches > 0 or views > 5: features.append("Funnel Stage: Research")
            elif views > 0: features.append("Funnel Stage: Awareness")
            else: features.append("Funnel Stage: Inactive")

            # Last action type already handled by TemporalExtractor recency
            # last_event = events.sort("timestamp", descending=True).row(0, named=True)
            # if last_event:
            #     last_type = last_event['event_type']; last_time = last_event['timestamp']
            #     days_since_last = (now - last_time).days if last_time else -1
            #     recency_tag = f"(last {days_since_last+1}d)" if days_since_last < 30 and days_since_last >=0 else "(>30d ago)" if days_since_last >=0 else ""
            #     # Mapping simple
            #     action_map = {'product_buy':'Purchase', 'add_to_cart':'Cart Add', 'search_query':'Search', 'page_visit':'Visit', 'remove_from_cart':'Cart Remove'}
            #     features.append(f"Last action type: {action_map.get(last_type, last_type)} {recency_tag}")

        except Exception as e: self.logger.debug(f"Err funnel pos: {e}"); features.append("Err funnel position.")

    def _extract_cart_abandon_signal(self, events: pl.DataFrame, features: List[str]) -> None:
        try:
            adds  = events.filter(pl.col('event_type') == 'add_to_cart').height
            buys  = events.filter(pl.col('event_type') == 'product_buy').height
            if adds > 0 and buys / adds < 0.2:
                features.append("Cart Behavior: High abandon ratio")
            elif adds > 0 and buys / adds > 0.8:
                features.append("Cart Behavior: High conversion ratio")            
            
        except Exception as e:
            self.logger.debug(f"Err cart abandon signal: {e}")
            features.append("Err cart abandon signal.")



    def _count_sessions(self, events: pl.DataFrame) -> int:
        if events.height <= 1: return events.height
        try:
            timestamps_sorted = events.sort("timestamp")['timestamp']
            SESSION_GAP_MINUTES = 30
            time_diffs_minutes = timestamps_sorted.diff().dt.total_seconds() / 60
            session_starts = time_diffs_minutes.is_null() | (time_diffs_minutes > SESSION_GAP_MINUTES)
            num_sessions = session_starts.sum()
            return int(num_sessions)
        except Exception as e: self.logger.error(f"Err counting sessions: {e}"); return 1


class PriceFeatureExtractor(FeatureExtractorBase):
    """Extract price sensitivity and purchase behavior features"""
    # --- Code inchangé ---
    # (Ajouter 'now' comme argument non utilisé)
    def extract_features(self, client_id: int, events: pl.DataFrame, now: datetime) -> List[str]:
        if 'price_bucket' not in events.columns: return ["Price features skipped."]
        features = []
        try:
            events_with_price = events.filter(pl.col('price_bucket').is_not_null())
            if events_with_price.height == 0: return ["No price data."]
            self._extract_price_range(events_with_price, features)
            self._extract_price_sensitivity(client_id, events_with_price, features)
            has_discount_cols = any(c in events.columns for c in ['discount', 'discount_percentage', 'original_price'])
            if has_discount_cols: self._extract_discount_patterns(events_with_price, features)
            # else: features.append("Discount patterns skipped.") # Optionnel
        except Exception as e: self.logger.error(f"Err price client {client_id}: {e}"); features.append("Err price")
        # --- RFM quick tag ---
        purchases = events_with_price.filter(pl.col('event_type') == 'product_buy')
        if purchases.height:
            rec = (now - purchases['timestamp'].max()).days               # Recency
            freq = purchases.filter(pl.col('timestamp') >= now - timedelta(days=90)).height
            mon = purchases['price_bucket'].mean()                        # Monetary (moyenne des buckets)
            features.append(f"RFM:{rec}:{freq}:{mon:.0f}")
               
        return features

    def _extract_price_range(self, events: pl.DataFrame, features: List[str]) -> None:
        try:
            relevant_events = events.filter( pl.col('event_type').is_in(['page_visit', 'add_to_cart', 'product_buy']) )
            if relevant_events.height == 0: return
            price_stats = relevant_events.select(pl.col('price_bucket')).describe() # Utiliser 'price_bucket'
            stats_dict = {row[0]: row[1] for row in price_stats.iter_rows()}
            min_price = stats_dict.get('min'); max_price = stats_dict.get('max')
            avg_price = stats_dict.get('mean'); std_price = stats_dict.get('std')
            count = stats_dict.get('count')
            if count is not None and count >= 2:
                if min_price is not None and max_price is not None:
                     features.append(f"Interacted price range (bucket): {min_price:.0f} - {max_price:.0f} (avg {avg_price:.0f})")
                     price_range = max_price - min_price
                     if price_range > 30: features.append("Wide price exploration.")
                     elif price_range < 10: features.append("Narrow price focus.")
                purchase_prices = relevant_events.filter(pl.col('event_type') == pl.lit('product_buy', dtype=pl.Categorical))['price_bucket']
                if purchase_prices.len() >= 2:
                    avg_purchase = purchase_prices.mean(); std_purchase = purchase_prices.std()
                    if avg_purchase is not None and avg_purchase > 0 and std_purchase is not None:
                         cv = std_purchase / avg_purchase
                         if cv < 0.15: features.append("Consistent purchase price.")
                         elif cv > 0.4: features.append("Varied purchase prices.")
        except Exception as e: self.logger.debug(f"Err price range: {e}"); features.append("Err price range.")

    def _extract_price_sensitivity(self, client_id: int, events: pl.DataFrame, features: List[str]) -> None:
        try:
            price_col = 'price_bucket'
            cart_events = events.filter(pl.col('event_type') == pl.lit('add_to_cart', dtype=pl.Categorical))
            purchase_events = events.filter(pl.col('event_type') == pl.lit('product_buy', dtype=pl.Categorical))
            if cart_events.height > 0 and purchase_events.height > 0:
                avg_cart_price = cart_events[price_col].mean()
                avg_purchase_price = purchase_events[price_col].mean()
                if avg_cart_price is not None and avg_purchase_price is not None and avg_cart_price > 0:
                    ratio = avg_purchase_price / avg_cart_price
                    if ratio < 0.8: features.append("Sensitivity: High (buys cheaper than adds)")
                    elif ratio > 1.2: features.append("Sensitivity: Low (buys similar/pricier)")
                    else: features.append("Sensitivity: Moderate")

            # Abandon vs price logic
            if cart_events.height > 0 and 'sku' in events.columns:
                cart_skus_prices = cart_events.select(['sku', price_col]).drop_nulls()
                if cart_skus_prices.height > 0:
                     purchased_skus = purchase_events.select('sku').drop_nulls()['sku'].unique().to_list()
                     if purchased_skus:
                          abandoned_items = cart_skus_prices.filter(~pl.col('sku').is_in(purchased_skus))
                          purchased_carted_items = cart_skus_prices.filter(pl.col('sku').is_in(purchased_skus))
                          if abandoned_items.height > 0 and purchased_carted_items.height > 0:
                               avg_abandoned_price = abandoned_items[price_col].mean()
                               avg_purchased_price = purchased_carted_items[price_col].mean()
                               if avg_abandoned_price is not None and avg_purchased_price is not None:
                                    if avg_abandoned_price > avg_purchased_price * 1.2:
                                         features.append("Tends to abandon higher-priced cart items.")
        except Exception as e: self.logger.error(f"Err price sensitivity client {client_id}: {e}"); features.append("Err price sensitivity.")

    def _extract_discount_patterns(self, events: pl.DataFrame, features: List[str]) -> None:
        # Placeholder - logic depends on actual discount columns
        relevant_discount_cols = [c for c in ['discount', 'discount_percentage', 'original_price'] if c in events.columns]
        if relevant_discount_cols:
             features.append(f"Discount info present ({', '.join(relevant_discount_cols)}), analysis TBD.")


class SocialFeatureExtractor(FeatureExtractorBase):
    """Extract social and competitive factors features"""

    def extract_features(self, client_id: int, events: pl.DataFrame, now: datetime) -> List[str]:
        features = []
        if self.parent.product_popularity is None or self.parent.product_popularity.height == 0:
            return ["Popularity data not available."]
        try:
            self._extract_popularity_patterns(client_id, events, features)
            self._extract_category_popularity_patterns(client_id, events, features)  # ← AJOUTER
        except Exception as e:
            self.logger.error(f"Err social client {client_id}: {e}")
            features.append("Err social")
        return features


    def _extract_category_popularity_patterns(
        self,
        client_id: int,
        events: pl.DataFrame,
        features: List[str]
    ) -> None:
        """Extract patterns based on category popularity"""
        try:
            if self.parent.category_popularity is None or self.parent.category_popularity.height == 0:
                return
                
            # Get categories this user interacted with
            user_cats = events.filter(
                pl.col('category_id').is_not_null()
            )['category_id'].unique().to_list()
            
            if not user_cats:
                return
                
            # Get popularity scores for user's categories
            user_cat_pop = self.parent.category_popularity.filter(
                pl.col('category_id').is_in(user_cats)
            )
            
            if user_cat_pop.height == 0:
                return
                
            # Average popularity of user's categories
            avg_user_cat_pop = user_cat_pop['category_popularity_score'].mean()
            global_avg_cat_pop = self.parent.category_popularity['category_popularity_score'].mean()
            
            if global_avg_cat_pop and global_avg_cat_pop > 0:
                ratio = avg_user_cat_pop / global_avg_cat_pop
                
                if ratio > 1.3:
                    features.append("Category affinity: Popular categories")
                elif ratio < 0.7:
                    features.append("Category affinity: Niche categories")
                    
            # Top category by popularity
            if 'category_id' in events.columns:
                cat_counts = (
                    events.filter(pl.col('category_id').is_not_null())
                    .group_by('category_id')
                    .agg(pl.len().alias('interactions'))
                    .join(
                        self.parent.category_popularity.select(['category_id', 'category_popularity_score']),
                        on='category_id',
                        how='left'
                    )
                )
                
                if cat_counts.height > 0:
                    # Most popular category they interact with
                    top_popular = cat_counts.sort('category_popularity_score', descending=True).head(1)
                    if top_popular.height > 0:
                        cat_id = top_popular['category_id'][0]
                        pop_score = top_popular['category_popularity_score'][0]
                        features.append(f"Most popular category: CAT_{cat_id} (score: {pop_score:.0f})")
                        
        except Exception as e:
            self.logger.debug(f"Error in category popularity: {e}")    
    # ------------------------------------------------------------------
    # SOCIAL : popularité + focus catégorie
    # ------------------------------------------------------------------
    def _extract_popularity_patterns(
        self,
        client_id: int,
        events: pl.DataFrame,
        features: List[str]
    ) -> None:
        try:
            # ---------- 1) Tous les événements produit ----------
            product_events = events.filter(
                pl.col('event_type')
                  .is_in(['page_visit', 'add_to_cart', 'product_buy'])
                & pl.col('sku').is_not_null()
            )
            if product_events.height == 0:
                return

            user_skus = (
                product_events['sku']
                .unique()
                .drop_nulls()
                .to_list()
            )
            if not user_skus:
                return

            user_pop = self.parent.product_popularity.filter(
                pl.col('sku').is_in(user_skus)
            )
            if user_pop.height == 0:
                return

            avg_user_pop   = user_pop['popularity_score'].mean()
            global_avg_pop = self.parent.product_popularity[
                'popularity_score'
            ].mean()

            # ---------- 2) Popularité globale ----------
            if global_avg_pop and global_avg_pop > 0:
                ratio = avg_user_pop / global_avg_pop

                if ratio > 1.3:
                    features.append("Affinity: Popular products")
                elif ratio < 0.7:
                    features.append("Affinity: Niche products")
                else:
                    features.append("Affinity: Average popularity")

                delta = avg_user_pop - global_avg_pop
                if abs(delta) >= 1:
                    features.append(f"Popularity Δ: {delta:+.1f}")

            # ---------- 3) Popularité VS moyenne des catégories visitées ----------
            if (
                'category_id' in events.columns
                and self.parent.category_popularity is not None
                and self.parent.category_popularity.height > 0
            ):
                cat_ids = (
                    events.filter(pl.col('category_id').is_not_null())
                          ['category_id']
                          .unique()
                          .to_list()
                )
                if cat_ids:
                    avg_cat_pop = self.parent.category_popularity.filter(
                        pl.col('category_id').is_in(cat_ids)
                    )['category_popularity_score'].mean()

                    if avg_cat_pop and avg_cat_pop > 0:
                        cat_ratio = avg_user_pop / avg_cat_pop
                        if cat_ratio > 1.3:
                            features.append("Affinity: Popular VS category avg")
                        elif cat_ratio < 0.7:
                            features.append("Affinity: Niche VS category avg")

            # ---------- 4) Con­cen­tra­tion sur la TOP catégorie (focus) ----------
            if 'category_id' in events.columns:
                page_visits = events.filter(
                    (pl.col('event_type') == 'page_visit')
                    & pl.col('category_id').is_not_null()
                )
                if page_visits.height >= 10:            # au moins 10 vues
                    cat_cnts = (
                        page_visits.group_by('category_id')
                                   .agg(pl.count().alias('cnt'))
                                   .sort('cnt', descending=True)
                    )
                    total_views = cat_cnts['cnt'].sum()
                    top_share   = cat_cnts.row(0)['cnt'] / total_views
                    if top_share >= 0.75:
                        features.append("Browsing highly concentrated on one category")
                    elif top_share <= 0.40 and cat_cnts.height >= 3:
                        features.append("Browsing spread across many categories")

            # ---------- 5) Différence vue ↔ panier ----------
            view_events = product_events.filter(
                pl.col('event_type') == pl.lit('page_visit', dtype=pl.Categorical)
            )
            cart_events = product_events.filter(
                pl.col('event_type') == pl.lit('add_to_cart', dtype=pl.Categorical)
            )
            if view_events.height > 0 and cart_events.height > 0:
                view_skus = (
                    view_events['sku'].unique().drop_nulls().to_list()
                )
                cart_skus = (
                    cart_events['sku'].unique().drop_nulls().to_list()
                )

                if view_skus and cart_skus:
                    avg_view_pop = self.parent.product_popularity.filter(
                        pl.col('sku').is_in(view_skus)
                    )['popularity_score'].mean()

                    avg_cart_pop = self.parent.product_popularity.filter(
                        pl.col('sku').is_in(cart_skus)
                    )['popularity_score'].mean()

                    if (
                        avg_view_pop is not None
                        and avg_cart_pop is not None
                        and avg_view_pop > 0
                    ):
                        ratio_cart_view = avg_cart_pop / avg_view_pop
                        if ratio_cart_view > 1.2:
                            features.append("Adds more popular items to cart than viewed.")
                        elif ratio_cart_view < 0.8:
                            features.append("Adds less popular items to cart than viewed.")

        except Exception as e:
            self.logger.debug(f"Err popularity: {e}")
            features.append("Err popularity patterns.")

class NameEmbeddingExtractor(FeatureExtractorBase):
    """Extract features based on product name embeddings."""
    def extract_features(self, client_id: int, events: pl.DataFrame, now: datetime) -> List[str]:
        if not hasattr(self.parent, 'sku_properties_dict') or not self.parent.sku_properties_dict:
            return ["Product properties unavailable."]
        features = []
        try:
            product_events = events.filter(
                pl.col('event_type').is_in(['page_visit', 'add_to_cart', 'product_buy']) 
                & pl.col('sku').is_not_null()
            )
            if product_events.height == 0:
                return []
            
            valid_name_embeddings = []
            interacted_skus = product_events['sku'].unique().drop_nulls().to_list()
            if not interacted_skus:
                return []
    
            for sku in interacted_skus:
                props = self.parent.sku_properties_dict.get(int(sku))
                if props and isinstance(props.get('name'), str) and props['name'].startswith('[') and props['name'].endswith(']'):
                    name_embedding_str = props['name']
                    try:
                        name_embedding = [int(x) for x in name_embedding_str.strip('[]').split()]
                        if name_embedding:  # Check not empty
                            valid_name_embeddings.append(name_embedding)
                    except Exception:
                        continue
    
            if not valid_name_embeddings:
                return ["No valid name embeddings found."]
            
            first_len = len(valid_name_embeddings[0])
            consistent_embeddings = [emb for emb in valid_name_embeddings if len(emb) == first_len]
            if not consistent_embeddings:
                return ["Name embeddings have inconsistent lengths."]
            if first_len == 0:
                return ["Name embeddings have zero length."]
    
            # FIX: Ensure we have a proper 2D array before operations
            try:
                emb_array = np.array(consistent_embeddings, dtype=np.float32)
                if emb_array.ndim != 2 or emb_array.shape[0] == 0:
                    return ["Invalid embedding array shape."]
                
                avg_vector = np.mean(emb_array, axis=0)
                if first_len > 32:
                    emb_array = emb_array[:, :32]
                    avg_vector = avg_vector[:32]
                    first_len = 32
    
                avg_vector_str = ", ".join([f"{x:.2f}" for x in avg_vector])
                features.append(f"AVG_PRODUCT_NAME_EMBEDDING (Dim:{first_len}): [{avg_vector_str}] ({len(consistent_embeddings)} items)")
    
                # Variance calculation with shape check
                if len(consistent_embeddings) > 1 and emb_array.shape[0] > 1:
                    std_vector = np.std(emb_array, axis=0)
                    avg_std = np.mean(std_vector)
                    if avg_std < 30:
                        features.append("Product Name Focus: High (Low Variance)")
                    elif avg_std > 70:
                        features.append("Product Name Focus: Low (High Variance)")
            except Exception as e:
                self.logger.debug(f"Error computing embeddings: {e}")
                return ["Error processing embeddings."]
    
        except Exception as e:
            self.logger.error(f"Err name embedding client {client_id}: {e}")
            features.append("Err name embedding.")
        return features

class TopSKUFeatureExtractor(FeatureExtractorBase):
    """Focus sur les 100 SKUs scorés par la compétition"""

    def __init__(self, parent):
        super().__init__(parent)
        self.top_skus = parent.top_skus

    def extract_features(self, client_id: int, events: pl.DataFrame, now: datetime) -> List[str]:
        if not self.top_skus:
            return ["Top-SKU list unavailable"]
        sku_ev = events.filter(
            pl.col('sku').is_in(self.top_skus) &
            pl.col('event_type').is_in(['product_buy', 'add_to_cart', 'page_visit'])
        )
        if sku_ev.is_empty():
            return ["No top-SKU interactions"]

        # ------- score récence × fréquence ---------------------------------
        rec_w = (
            (pl.lit(now) - sku_ev['timestamp']).dt.total_seconds() / 86400 + 1
        ).pow(-0.5)
        sku_ev = sku_ev.with_columns(rec_w.alias('rw'))

        scores = (
            sku_ev.group_by('sku')
                  .agg([
                      pl.len().alias('cnt'),
                      pl.sum('rw').alias('rScore'),
                      pl.max('timestamp').alias('last_ts')
                  ])
                  .sort('rScore', descending=True)
        )

        feats = []
        for i, row in enumerate(scores.head(5).iter_rows(named=True), 1):
            delta = (now - row['last_ts']).days
            feats.append(f"TOPSKU{i}:SKU_{row['sku']} rs={row['rScore']:.2f} "
                         f"cnt={row['cnt']} last={delta}d")

        # Couverture
        cov = scores.height / len(self.top_skus)
        feats.append(f"Top-SKU coverage:{cov:.0%}")

        return feats

# --- Main Generator Class ---
# --- Constants for raw sequence generation ---
SEP_TOKEN = "</s>"
SESSION_START_TOKEN = "T_SessionStart"
MAX_HISTORY_EVENTS_TO_CONSIDER = 8192
RAW_SEQUENCE_LAST_EVENTS = 512
# --- Temporal discretization helpers ---
def discretize_timedelta(delta: timedelta) -> str:
    seconds = delta.total_seconds()
    if seconds < 0: return "T_Error"
    if seconds <= 5: return "T_0-5s"
    if seconds <= 30: return "T_5-30s"
    if seconds <= 120: return "T_30s-2m"
    if seconds <= 600: return "T_2m-10m"
    if seconds <= 1800: return "T_10m-30m"
    return "T_30m+"

def discretize_time_of_day(ts: datetime) -> str:
    hour = ts.hour
    if 5 <= hour < 12: return "Morning"
    if 12 <= hour < 18: return "Afternoon"
    if 18 <= hour < 22: return "Evening"
    return "Night"

def discretize_day_of_week(ts: datetime) -> str:
    return "Weekday" if ts.weekday() < 5 else "Weekend"

# --- Helper: paires de co-achat / co-panier ---------------------------------
def top_co_pairs(events: pl.DataFrame, top_k: int = 5) -> List[str]:
    rel = (events.filter(
            pl.col('event_type').is_in(['product_buy', 'add_to_cart']) &
            pl.col('sku').is_not_null())
           .sort('timestamp'))
    if rel.height < 2:
        return []

    # 30-min sessions
    gaps = rel['timestamp'].diff().dt.total_seconds() / 60
    rel  = rel.with_columns(((gaps.is_null()) | (gaps > 30)).cum_sum().alias('sid'))

    c = Counter()
    for sess in rel.partition_by('sid', as_dict=False):
        skus = sess['sku'].drop_nulls().unique().to_list()
        if len(skus) >= 2:
            c.update(combinations(sorted(skus), 2))

    return [
        f"CO_PAIR:SKU_{i}~SKU_{j} ({cnt}x)"
        for (i, j), cnt in c.most_common(top_k) if cnt >= 2
    ]

def top_co_categories(events: pl.DataFrame, top_k: int = 5) -> List[str]:
    """
    Return tags  CAT_PAIR:CAT_i~CAT_j (cnt×)  for category pairs that co-occur
    within the same 30-minute session.
    """
    if events.height < 2 or 'category_id' not in events.columns:
        return []

    # keep only page-views & purchases that have a category
    rel = (
        events.filter(
            pl.col('event_type').is_in(['page_visit', 'product_buy']) &
            pl.col('category_id').is_not_null()
        )
        .sort('timestamp')
    )
    if rel.height < 2:
        return []

    # add 30-minute session IDs (vectorised)
    gaps = rel['timestamp'].diff().dt.total_seconds() / 60
    rel  = rel.with_columns(
        ((gaps.is_null()) | (gaps > 30)).cum_sum().alias('sid')
    )

    # count unique category pairs per session
    pair_counts: Counter[tuple[int, int]] = Counter()
    for sess in rel.partition_by('sid', as_dict=False):
        cats = sess['category_id'].drop_nulls().unique().to_list()
        if len(cats) >= 2:
            pair_counts.update(combinations(sorted(cats), 2))

    return [
        f"CAT_PAIR:CAT_{i}~CAT_{j} ({cnt}x)"
        for (i, j), cnt in pair_counts.most_common(top_k)
        if cnt >= 2
    ]

# ------------------------------------------------------------------
# Helper : stats conversion / abandon panier  (safe + polars only)
# ------------------------------------------------------------------
def cart_conversion_stats(events: pl.DataFrame) -> list[str]:
    """
    Retourne :
      • CART_CONV_RATE            – % des ajouts convertis en achat
      • AVG_CART2BUY_MIN          – délai moyen (min) entre add→buy
    Avec garde-fous quand il n'y a ni add_to_cart ni product_buy.
    """
    # 1) Sélection des lignes utiles
    adds = (
        events.filter(pl.col("event_type") == "add_to_cart")
              .filter(pl.col("sku").is_not_null())
    )
    buys = (
        events.filter(pl.col("event_type") == "product_buy")
              .filter(pl.col("sku").is_not_null())
    )

    if adds.is_empty():
        return ["Cart conversion: no cart activity"]          

    # 2) Taux de conversion panier → achat
    purchased_skus   = buys["sku"].unique().to_list()
    converted_height = (
        adds.filter(pl.col("sku").is_in(purchased_skus)).height
        if purchased_skus else
        0
    )
    conv_rate = converted_height / adds.height               

    lines = [f"CART_CONV_RATE={conv_rate:.2f}"]

    # 3) Délai moyen add→buy (si achetés)
    if not buys.is_empty() and purchased_skus:
        cart_times = (adds.select(["sku", "timestamp"])
                          .rename({"timestamp": "add_ts"}))
        buy_times  = (buys.select(["sku", "timestamp"])
                          .rename({"timestamp": "buy_ts"}))

        delays = (
            buy_times.join(cart_times, on="sku", how="inner")
                     .filter(pl.col("add_ts") < pl.col("buy_ts"))
                     .with_columns(
                         ((pl.col("buy_ts") - pl.col("add_ts"))
                          .dt.total_seconds() / 60).alias("delay_min")
                     )["delay_min"]
        )

        if delays.len() > 0:
            lines.append(f"AVG_CART2BUY_MIN={delays.mean():.1f}")

    return lines


class AdvancedUBMGenerator:
    """Generates advanced Universal Behavioral Profiles with
    multi-modal, multi-resolution features (lazy Polars pipeline)"""

    def __init__(self, data_dir: str, cache_dir: Optional[str] = None, debug_mode: bool = False):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else (self.data_dir / "cache")
        self.debug_mode = debug_mode
        os.makedirs(self.cache_dir, exist_ok=True)

        # For lazy pipeline
        self.lazy_all: Optional[pl.LazyFrame] = None

        # Cached or materialized data
        self.events_df: Optional[pl.DataFrame] = None
        self.sku_properties_for_join: Optional[pl.DataFrame] = None
        self.sku_properties_dict: Dict[int, Dict[str, Any]] = {}
        self.product_popularity: Optional[pl.DataFrame] = None
        self.category_popularity: Optional[pl.DataFrame] = None
        self.global_stats: Dict[str, Any] = {}
        self.user_segments: Dict[str, List[int]] = {}
        self._extractors: Dict[str, Any] = {}
        self.top_skus: list[int] = []
        try:
            # ex : array([123, 456, …])
            self.top_skus = list(np.load(self.data_dir / "target/propensity_sku.npy"))
        except Exception:
            self.logger.warning("Top-SKU list not found – SKU_PROPENSITY features disabled")

        self.top_categories: list[int] = []
        try:
            self.top_categories = list(np.load(self.data_dir / "target/propensity_category.npy"))
        except Exception:
            self.logger.warning("Top-category list not found – CAT_PROPENSITY features disabled")


        self.logger = logging.getLogger(self.__class__.__name__)
        self.url_freq_threshold = URL_FREQ_THRESHOLD
        self.sku_pop_threshold  = SKU_POP_THRESHOLD
        if self.debug_mode:
            self.logger.setLevel(logging.DEBUG)
            
    def _setup_lazy_pipeline_only(self) -> None:
        """
        Set up the lazy pipeline without collecting any data.
        Used by streaming generator to avoid double loading.
        """
        self.logger.info("Setting up lazy pipeline (no data collection)")
        
        # 1) Load product properties for join if available
        props_path = self.data_dir / "product_properties.parquet"
        if props_path.exists():
            prop = pl.read_parquet(props_path)
            emb_source = None
            if "embedding" in prop.columns:
                emb_source = "embedding"
            elif "name" in prop.columns:
                # Check if it looks like an embedding
                sample = prop["name"].head(1)
                if sample.len() > 0 and str(sample[0]).strip().startswith("["):
                    emb_source = "name"
    
            cols = ["sku", "category", "price"] + ([emb_source] if emb_source else [])
            tmp = prop.select(cols)
            rename_map = {"category": "category_id", "price": "price_bucket"}
            if emb_source:
                rename_map[emb_source] = "emb_str"
            self.sku_properties_for_join = tmp.rename(rename_map).with_columns(
                pl.col("sku").cast(pl.Int64)
            )
            # Note: sku_properties_dict should already be loaded from cache
        
        # 2) Build lazy scans & union
        event_types = ["product_buy", "add_to_cart", "remove_from_cart", "page_visit", "search_query"]
        schema = {
            "client_id": pl.Int64,
            "timestamp": pl.Datetime("us"),
            "sku": pl.Int64,
            "url": pl.Utf8,
            "query": pl.Utf8
        }
    
        lazy_sources = []
        for et in event_types:
            fp = self.data_dir / f"{et}.parquet"
            if not fp.exists():
                self.logger.warning(f"Skipping missing file {fp}")
                continue
            
            scan = pl.scan_parquet(fp)
            lf_schema = scan.collect_schema()
            
            exprs = []
            for col, dt in schema.items():
                if col in lf_schema:
                    col_expr = pl.col(col)
                    if lf_schema[col] != dt:
                        if col == "timestamp":
                            col_expr = col_expr.cast(pl.Utf8).str.to_datetime(
                                strict=False, time_unit="us"
                            ).cast(dt)
                        else:
                            col_expr = col_expr.cast(dt, strict=False)
                    exprs.append(col_expr.alias(col))
                else:
                    exprs.append(pl.lit(None).cast(dt).alias(col))
            
            exprs.append(pl.lit(et).alias("event_type"))
            lazy_sources.append(scan.select(exprs))
    
        if not lazy_sources:
            raise ValueError("No event files found.")
    
        # Union all sources
        lf_all = pl.concat(lazy_sources)
    
        # 3) Join product properties lazily
        if self.sku_properties_for_join is not None:
            lf_all = lf_all.join(
                self.sku_properties_for_join.lazy(),
                on="sku",
                how="left"
            )
    
        # 4) Store the lazy pipeline
        self.lazy_all = lf_all
        
        # 5) Initialize centrality attributes if not already present
        if not hasattr(self, 'sku_centrality'):
            self.sku_centrality = {}
        if not hasattr(self, 'cat_centrality'):
            self.cat_centrality = {}
        if not hasattr(self, 'category_centrality'):
            self.category_centrality = {}
        
        # Initialize URL embedding attributes
        if not hasattr(self, 'url_embed'):
            self.url_embed = {}
        if not hasattr(self, 'url_centroid'):
            self.url_centroid = None
        if not hasattr(self, 'url_cluster_map'):
            self.url_cluster_map = {}
        
        # Initialize SKU clustering attributes  
        if not hasattr(self, 'sku_cluster_map'):
            self.sku_cluster_map = {}
        
        # Initialize user segments if not present
        if not hasattr(self, 'user_segments'):
            self.user_segments = {}
        
        # Initialize popularity score mapping
        if not hasattr(self, 'pop_score_by_sku'):
            self.pop_score_by_sku = {}
            # Try to build it from product_popularity if available
            if self.product_popularity is not None and 'sku' in self.product_popularity.columns and 'popularity_score' in self.product_popularity.columns:
                try:
                    self.pop_score_by_sku = {
                        int(row['sku']): float(row['popularity_score']) 
                        for row in self.product_popularity[['sku', 'popularity_score']].iter_rows(named=True)
                        if row['sku'] is not None and row['popularity_score'] is not None
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to build pop_score_by_sku: {e}")
        
        # Initialize category_popularity if not present
        if not hasattr(self, 'category_popularity'):
            self.category_popularity = None
        
        # 6) Initialize extractors based on what we know from cache
        self._extractors = {}
        self._extractors['temporal'] = TemporalFeatureExtractor(self)
        self._extractors['sequence'] = SequenceFeatureExtractor(self)

     
        # Add other extractors based on available data
        # (we know from the schema what columns are available)
        if self.sku_properties_for_join is not None:
            if 'category_id' in self.sku_properties_for_join.columns:
                self._extractors['graph'] = GraphFeatureExtractor(self)
            if 'price_bucket' in self.sku_properties_for_join.columns:
                self._extractors['price'] = PriceFeatureExtractor(self)
            if 'emb_str' in self.sku_properties_for_join.columns:
                self._extractors['name_embedding'] = NameEmbeddingExtractor(self)
        
        self._extractors['intent'] = IntentFeatureExtractor(self)
        
        if self.product_popularity is not None:
            self._extractors['social'] = SocialFeatureExtractor(self)
            
        if self.top_skus:
            self._extractors['top_sku'] = TopSKUFeatureExtractor(self)

        if self.top_categories:
            self._extractors['top_category'] = TopCategoryFeatureExtractor(self)
        
        # Add ChurnPropensityExtractor if it exists
        try:
            self._extractors['churn_propensity'] = ChurnPropensityExtractor(self)
        except NameError:
            pass  # ChurnPropensityExtractor not defined
        
        self.logger.info("Lazy pipeline ready (no data materialized)")
            
    def _reset_data(self):
        self.logger.warning("Resetting internal dataframes and stats.")
        self.lazy_all = None
        self.events_df = None
        self.sku_properties_for_join = None
        self.sku_properties_dict = {}
        self.product_popularity = None
        self.category_popularity = None
        self.global_stats = {}
        self.user_segments = {}
        self._extractors = {}
        gc.collect()

    def load_data(self,
                      use_cache: bool = True,
                      relevant_client_ids: Optional[List[int]] = None) -> None:
            """
            Entry-point: builds lazy pipeline from parquet sources and joins.
            """
            self.logger.info(f"=== load_data called with use_cache={use_cache}, "
                             f"relevant_clients={len(relevant_client_ids) if relevant_client_ids else 'None'}")
            
            # 1. Déterminer quel fichier cache utiliser
            if relevant_client_ids is not None and len(relevant_client_ids) <= 1_000_000:
                cache_file = self.cache_dir / "events_1m_clients.parquet"
                self.logger.info(f"Using filtered cache for 1M clients")
            else:
                cache_file = self.cache_dir / "all_events_processed.parquet"
                self.logger.info(f"Using full cache")
            
            # if self.debug_mode and relevant_client_ids and len(relevant_client_ids) < 100:
            #     # En mode debug avec peu de clients, filtrer aussi les SKU properties
            #     if 'sku' in df.columns:
            #         client_skus = set(df['sku'].drop_nulls().unique().to_list())
            
            # 3. Essayer de charger depuis le cache
            if use_cache and cache_file.exists():
                try:
                    self.logger.info(f"Loading events cache: {cache_file}")
                    
                    # Vérifier que les stats globales existent (sauf en debug)
                    if not self.debug_mode and not (self.cache_dir / "global_stats.json").exists():
                        self.logger.warning("Global stats missing - need to recompute")
                        self._reset_data()
                    else:
                        # Charger le cache events
                        df = pl.read_parquet(cache_file)
                        
                        # ✅ FIX #1 : TOUJOURS filtrer si relevant_client_ids est fourni
                        if relevant_client_ids is not None:
                            original_height = df.height
                            self.logger.info(f"Filtering {original_height:,} events to {len(relevant_client_ids)} clients...")
                            df = df.filter(pl.col("client_id").is_in(relevant_client_ids))
                            self.logger.info(f"After filtering: {df.height:,} events (reduced by {original_height - df.height:,})")
                            
                            # Si après filtrage on a 0 events, logger un warning
                            if df.height == 0:
                                self.logger.warning(f"No events found for clients {relevant_client_ids[:5]}...")
                        
                        # Charger les stats calculées
                        if self._load_calculated_data_from_cache() and df.height > 0:
                            self.events_df = df
                            
                            # ✅ FIX #3 : Réduire les SKU properties en mode debug
                            if self.debug_mode and relevant_client_ids and len(relevant_client_ids) < 100:
                                if 'sku' in df.columns and hasattr(self, 'sku_properties_dict') and self.sku_properties_dict:
                                    # Récupérer les SKUs utilisés par ces clients
                                    client_skus = set(df['sku'].drop_nulls().unique().to_list())
                                    
                                    # Filtrer le dictionnaire des SKU properties
                                    filtered_dict = {
                                        k: v for k, v in self.sku_properties_dict.items() 
                                        if k in client_skus
                                    }
                                    
                                    old_size = len(self.sku_properties_dict)
                                    self.sku_properties_dict = filtered_dict
                                    
                                    self.logger.info(f"Debug mode: Reduced SKU properties from {old_size:,} to {len(self.sku_properties_dict):,}")
                                    
                                    # Libérer la mémoire
                                    import gc
                                    gc.collect()
                            
                            self.logger.info(f"Loaded all data from cache. Events shape: {df.shape}")
                            return  # ← SUCCESS ! On sort ici
                        
                        self.logger.warning("Cache incomplete or empty → reloading from source.")
                        self._reset_data()
                        
                except Exception as e:
                    self.logger.warning(f"Failed loading cache ({e}) → reloading from source.")
                    self._reset_data()
            
            # ============================================================
            # Si on arrive ici, on doit charger depuis les fichiers source
            # ============================================================
            self.logger.info("Loading from source files...")
            
            # Charger les propriétés des produits
            props_path = self.data_dir / "product_properties.parquet"
            if props_path.exists():
                prop = pl.read_parquet(props_path)
                emb_source = None
                
                # Détecter la colonne d'embedding
                if "embedding" in prop.columns:
                    emb_source = "embedding"
                elif "name" in prop.columns and prop["name"].head(1)[0].strip().startswith("["):
                    emb_source = "name"
    
                # Sélectionner et renommer les colonnes
                cols = ["sku", "category", "price"] + ([emb_source] if emb_source else [])
                tmp = prop.select(cols)
                rename_map = {"category": "category_id", "price": "price_bucket"}
                if emb_source:
                    rename_map[emb_source] = "emb_str"
                    
                self.sku_properties_for_join = tmp.rename(rename_map).with_columns(pl.col("sku").cast(pl.Int64))
                self.sku_properties_dict = {
                    int(r["sku"]): {k:v for k,v in r.items() if k!="sku"}
                    for r in prop.to_dicts() if r.get("sku") is not None
                }
                self.logger.info(f"Loaded {len(self.sku_properties_dict)} product properties.")
            else:
                self.logger.warning("No properties file → skipping embedding joins.")
    
            # Construire les lazy scans pour chaque type d'événement
            event_types = ["product_buy", "add_to_cart", "remove_from_cart", "page_visit", "search_query"]
            schema = {
                "client_id": pl.Int64,
                "timestamp": pl.Datetime("us"),
                "sku": pl.Int64,
                "url": pl.Utf8,
                "query": pl.Utf8
            }
    
            lazy_sources = []
            for et in event_types:
                fp = self.data_dir / f"{et}.parquet"
                if not fp.exists():
                    self.logger.warning(f"Skipping missing file {fp}")
                    continue
                    
                scan = pl.scan_parquet(fp)
                lf_schema = scan.collect_schema()
                exprs = []
                
                # Harmoniser les colonnes
                for col, dt in schema.items():
                    if col in lf_schema:
                        col_expr = pl.col(col)
                        if lf_schema[col] != dt:
                            if col == "timestamp":
                                col_expr = col_expr.cast(pl.Utf8).str.to_datetime(strict=False, time_unit="us").cast(dt)
                            else:
                                col_expr = col_expr.cast(dt, strict=False)
                        exprs.append(col_expr.alias(col))
                    else:
                        exprs.append(pl.lit(None).cast(dt).alias(col))
                        
                exprs.append(pl.lit(et).alias("event_type"))
                lazy_sources.append(scan.select(exprs))
    
            if not lazy_sources:
                raise ValueError("No event files found.")
    
            # Union de tous les événements
            lf_all = pl.concat(lazy_sources)
    
            # Joindre les propriétés des produits
            if self.sku_properties_for_join is not None:
                lf_all = lf_all.join(
                    self.sku_properties_for_join.lazy(),
                    on="sku",
                    how="left"
                )
    
            # Filtrer en mode debug
            if self.debug_mode and relevant_client_ids is not None:
                self.logger.debug(f"DEBUG: filtrage lazy sur {len(relevant_client_ids)} clients")
                lf_all = lf_all.filter(pl.col("client_id").is_in(relevant_client_ids))
    
            # Stocker le pipeline lazy
            self.lazy_all = lf_all
    
            # Calculer les statistiques globales
            self._compute_global_statistics()
    
            # En mode debug, matérialiser immédiatement
            if self.debug_mode and relevant_client_ids is not None:
                self.logger.debug(f"DEBUG: materializing events_df pour {len(relevant_client_ids)} clients")
                self.events_df = (
                    self.lazy_all
                    .filter(pl.col("client_id").is_in(relevant_client_ids))
                    .sort(["client_id", "timestamp"])
                    .collect(engine='streaming')
                )
                return
                
            # Clustering et autres calculs globaux (seulement si pas en debug)
            if self.sku_properties_for_join is not None and "emb_str" in self.sku_properties_for_join.columns:
                self._cluster_sku_embeddings()
            
            self._build_url_graph_embeddings()
            self._segment_users()
            self._build_global_centralities()
    
            # Sauvegarder le cache en mode normal
            if use_cache and not self.debug_mode:
                df_all = lf_all.sort(["client_id", "timestamp"]).collect(engine='streaming')
                self.events_df = df_all
                df_all.write_parquet(cache_file)
                self._save_calculated_data_to_cache()
                self.logger.info("Cache rebuilt and saved.")
            
    def _collect_client_events(self, client_id: int) -> pl.DataFrame:
        """Pulls down only one client's events into memory."""
        if self.lazy_all is None:
            raise RuntimeError("Must call load_data() first.")
        return (
            self.lazy_all
              .filter(pl.col("client_id") == client_id)
              .sort("timestamp")
              .collect(engine='streaming')
        )

    # ------------------------------------------------------------------ #
    # === Helpers extraits de load_data (lazy-aware) ==================== #
    def _cluster_sku_embeddings(self) -> None:
        """Version corrigée avec protection contre arrays vides"""
        if self.lazy_all is None:
            return
        
        lf = (
            self.lazy_all
              .filter(pl.col("emb_str").is_not_null())
              .select(["sku", "emb_str"])       
              .unique()
        )
        prop_emb = lf.collect(engine='streaming')
        if prop_emb.is_empty():
            self.logger.info("No embedding rows to cluster.")
            return
    
        skus, vecs = [], []
        for row in prop_emb.iter_rows(named=True):
            try:
                arr = np.fromstring(row["emb_str"].strip("[]"), sep=" ")
                if arr.size > 0:  # Check array not empty
                    norm = np.linalg.norm(arr)
                    if norm > 0:
                        vecs.append(arr / norm)
                        skus.append(int(row["sku"]))
            except Exception:
                continue
    
        if vecs and len(vecs) > 0:  # Extra check
            try:
                X = np.vstack(vecs)
                if X.shape[0] > 0:  # Ensure we have rows
                    n_clusters = min(50, X.shape[0])  # Don't use more clusters than samples
                    mbk = MiniBatchKMeans(n_clusters=n_clusters, batch_size=4096, random_state=42).fit(X)
                    self.sku_cluster_map = {sku: int(lbl) for sku, lbl in zip(skus, mbk.labels_)}
                    self.logger.info(f"Built SKU clusters for {len(self.sku_cluster_map)} SKUs.")
                else:
                    self.logger.info("No valid embeddings after vstack.")
            except Exception as e:
                self.logger.error(f"SKU clustering failed: {e}")
        else:
            self.logger.info("No valid embeddings for SKU clustering.")


    # ─────────────────────────────────────────────────────────────
    #  AdvancedUBMGenerator._build_url_graph_embeddings  (NEW)
    # ─────────────────────────────────────────────────────────────
    def _build_url_graph_embeddings(self) -> None:
        """
        Full-streaming URL⇆SKU bipartite graph:
          1. écrit (src_hash, dst_hash) dans un CSV par blocs de 1 M lignes
          2. lit le CSV chunk par chunk pour créer le graphe NetworKit
          3. Node2Vec 32 d puis k-means (20 clusters)
        Pic RAM ≈ 3-4 Go quel que soit le dataset.
        """
        # ---------------------------------------------------------
    
        if self.lazy_all is None:
            return
        if os.getenv("SKIP_URL_GRAPH", "0") == "1":
            self.logger.info("SKIP_URL_GRAPH=1 → URL-SKU graph bypassed.")
            self.url_embed, self.url_centroid, self.url_cluster_map = {}, None, {}
            return
    
        self.logger.info("Building URL–SKU bipartite graph (streaming)…")
    
        # ── 1) TOP-N URLs (petit collect) ─────────────────────────
        TOP_URLS = 500
        top_urls = (
            self.lazy_all
              .filter(pl.col("url").is_not_null())
              .group_by("url")
              .agg(pl.count().alias("cnt"))
              .sort("cnt", descending=True)
              .limit(TOP_URLS)             # .head() == .limit()
              .collect()                   # ← on retire le streaming=True
              ["url"]
              .to_list()
        )
        top_urls = set(top_urls)
    
        # ── 2) génère (sid, url_hash) et (sid, sku_hash) ──────────
        with_sid = (
            self.lazy_all
              .filter(pl.col("url").is_not_null() | pl.col("sku").is_not_null())
              .with_columns(
                  (
                      (
                          (pl.col("timestamp")
                             .diff()
                             .over("client_id")
                             .dt.total_seconds() / 60)
                          .fill_null(1e9)  > 30
                      ) | (pl.col("client_id").diff().is_not_null())
                  ).alias("new_sess")
              )
              .with_columns(
                  pl.col("new_sess").cum_sum().over("client_id").alias("sid")
              )
        )
    
        MASK63 = (1 << 63) - 1            # 0x7FFF…FFFF
        
        urls = (
            with_sid
              .filter(pl.col("url").is_in(top_urls))
              .select([
                  "sid",
                  (
                      (pl.col("url")
                         .hash(seed=0)        # UInt64
                         % MASK63             # <= 2^63-1  
                      )
                      .cast(pl.Int64)         # signé OK
                  ).alias("url_hash")
              ])
              .unique()
        )
    
        skus = (
            with_sid
              .filter(pl.col("sku").is_not_null())
              .select([
                  "sid",
                  (pl.col("sku") * -1).cast(pl.Int64).alias("sku_hash")
              ])
              .unique()
        )
    
        # ── 3) jointure croisée → edges.csv (streaming v2 OK) ────
        edges_lf = (
            urls.join(skus, on="sid")
                .select([
                    pl.col("url_hash").alias("src_hash"),
                    pl.col("sku_hash").alias("dst_hash")
                ])
        )
        
        import networkit as nk
        g, label2nid = nk.Graph(0, weighted=True, directed=False), {}
        def _nid(lbl: int) -> int:
            return label2nid.setdefault(lbl, g.addNode())
        
        BATCH = 1_000_000   # lignes
        stream = edges_lf.iter_batches(batch_size=BATCH, streaming=True)
        for tbl in stream:                         # PyArrow Table
            src = tbl.column(0).to_numpy(zero_copy_only=False)
            dst = tbl.column(1).to_numpy(zero_copy_only=False)
            g.addEdges(np.vectorize(_nid)(src), np.vectorize(_nid)(dst))
        
        self.logger.info("Graph nodes=%d edges=%d", g.numberOfNodes(), g.numberOfEdges())
    
        # ── 5) Node2Vec 32 d  ─────────────────────────────────────
        n2v = nk_embed.Node2Vec(g, 1.0, 1.0, 10, 10, 32)
        n2v.run()
        emb = n2v.getFeatures()
    
        # ── 6) embeddings & centroid ──────────────────────────────
        self.url_embed = {
            lbl: emb[nid] for lbl, nid in label2nid.items() if lbl >= 0
        }
        if not self.url_embed:
            self.logger.warning("No URL embeddings – aborting.")
            self.url_centroid, self.url_cluster_map = None, {}
            return
        self.url_centroid = np.stack(list(self.url_embed.values())).mean(axis=0)
    
        # ── 7) k-means (20) pour clusteriser les URLs ─────────────
        try:
            u, vec = zip(*self.url_embed.items())
            km = MiniBatchKMeans(n_clusters=20, batch_size=4096, random_state=42)
            labels = km.fit_predict(np.stack(vec))
            self.url_cluster_map = {ui: int(lb) for ui, lb in zip(u, labels)}
            self.logger.info("URL clusters: %d", len(set(labels)))
        except Exception as e:
            self.logger.error("URL clustering failed: %s", e)
            self.url_cluster_map = {}
    
        self.logger.info("Node2Vec done: %d URL vectors", len(self.url_embed))
    # ─────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────
    # ------------------------------------------------------------------ #
    # Cat->Cat centralité globale (PageRank sur transitions)           #
    # ------------------------------------------------------------------ #
    def _compute_category_centrality(self) -> None:
        """
        Remplace le calcul classique de centrality par une passe sparse,
        tirée exclusivement du lazy frame.
        """
        if self.lazy_all is None:
            self.cat_centrality = {}
            return

        # build transition pairs lazily then collect
        df = (
            self.lazy_all
              .filter(
                  pl.col('category_id').is_not_null() &
                  pl.col('event_type').is_in(['page_visit','product_buy'])
              )
              .with_columns(
                  pl.col('category_id').shift(1).over('client_id').alias('prev_cat')
              )
              .filter(pl.col('prev_cat').is_not_null())
              .select(['prev_cat','category_id'])
        ).collect(engine='streaming')

        if df.height == 0:
            self.cat_centrality = {}
        else:
            arr = np.array(df.to_numpy(), dtype=int)
            src, dst = arr[:,0], arr[:,1]
            unique_edges, counts = np.unique(np.stack([src,dst],axis=1), axis=0, return_counts=True)
            se, de = unique_edges[:,0], unique_edges[:,1]
            self.cat_centrality = compute_sparse_pagerank(se, de, counts)

        self.logger.info(f"Built sparse category centrality • CAT:{len(self.cat_centrality)}")


    # ------------------------------------------------------------------ #
    # Global co‑occurrences (SKU & CAT) pour tags GLOBAL_CO_PAIR / GLOBAL_CAT_PAIR
    # ------------------------------------------------------------------ #
    # À ajouter dans text_representation_v3.py ou monkey-patch
    def _compute_global_co_pairs(self, top_k: int = 30) -> None:
        """Version optimisée qui échantillonne et utilise seulement les top items"""
        from itertools import combinations
        import numpy as np
        
        print("Computing global co-pairs (optimized)...")
        
        # 1. Limiter aux SKUs et catégories populaires
        top_skus = set()
        top_cats = set()
        
        if self.product_popularity is not None:
            # Top 5000 SKUs par popularité
            top_skus = set(
                self.product_popularity
                .sort('popularity_score', descending=True)
                .head(5000)['sku']
                .to_list()
            )
        
        if self.category_popularity is not None:
            # Top 500 catégories
            top_cats = set(
                self.category_popularity
                .sort('category_popularity_score', descending=True)
                .head(500)['category_id']
                .to_list()
            )
        
        # 2. Échantillonner les clients (10% ou 100k max)
        all_clients = (
            self.lazy_all
            .select('client_id')
            .unique()
            .collect()['client_id']
            .to_list()
        )
        
        sample_size = min(100_000, len(all_clients) // 10)
        sampled_clients = np.random.choice(all_clients, sample_size, replace=False)
        
        print(f"Sampling {sample_size:,} clients out of {len(all_clients):,}")
        
        # 3. Collecter seulement pour les clients échantillonnés
        df = (
            self.lazy_all
            .filter(
                pl.col('client_id').is_in(sampled_clients) &
                pl.col('event_type').is_in(['product_buy','add_to_cart'])
            )
            .select(['client_id','timestamp','sku','category_id'])
            .collect(engine='streaming')
        )
        
        # Filtrer par top items si disponibles
        if top_skus:
            df = df.with_columns(
                pl.when(pl.col('sku').is_in(top_skus))
                .then(pl.col('sku'))
                .otherwise(None)
                .alias('sku')
            )
        
        if top_cats:
            df = df.with_columns(
                pl.when(pl.col('category_id').is_in(top_cats))
                .then(pl.col('category_id'))
                .otherwise(None)
                .alias('category_id')
            )
        
        # 4. Sessions optimisées
        df = df.sort(['client_id','timestamp'])
        
        # Calcul vectorisé des sessions
        time_diff = df.select([
            pl.col('client_id'),
            pl.col('timestamp').diff().over('client_id').dt.total_seconds() / 60
        ])
        
        df = df.with_columns(
            ((time_diff['timestamp'] > 30) | time_diff['timestamp'].is_null())
            .cum_sum()
            .over('client_id')
            .alias('session_id')
        )
        
        # 5. Compter les paires par chunks pour éviter OOM
        sku_pair_counter = Counter()
        cat_pair_counter = Counter()
        
        # Grouper par chunks de 10k sessions
        unique_sessions = df.select(['client_id', 'session_id']).unique()
        n_sessions = unique_sessions.height
        chunk_size = 10_000
        
        print(f"Processing {n_sessions:,} sessions...")
        
        for i in range(0, n_sessions, chunk_size):
            chunk_sessions = unique_sessions[i:i+chunk_size]
            
            # Filtrer le df pour ce chunk
            chunk_df = df.join(chunk_sessions, on=['client_id', 'session_id'])
            
            # Traiter chaque session du chunk
            for (cid, sid), sess in chunk_df.group_by(['client_id', 'session_id']):
                skus = sess['sku'].drop_nulls().unique().to_list()
                cats = sess['category_id'].drop_nulls().unique().to_list()
                
                # Limiter les combinaisons si trop nombreuses
                if len(skus) > 20:
                    skus = skus[:20]
                if len(cats) > 10:
                    cats = cats[:10]
                
                for i, j in combinations(sorted(set(skus)), 2):
                    sku_pair_counter[(int(i), int(j))] += 1
                
                for c1, c2 in combinations(sorted(set(cats)), 2):
                    cat_pair_counter[(int(c1), int(c2))] += 1
        
        # Store results
        self.global_sku_pairs = sku_pair_counter.most_common(top_k)
        self.global_cat_pairs = cat_pair_counter.most_common(top_k)
        
        self.global_stats['global_sku_pairs'] = self.global_sku_pairs
        self.global_stats['global_cat_pairs'] = self.global_cat_pairs
        
        print(f"Found {len(sku_pair_counter)} SKU pairs, {len(cat_pair_counter)} category pairs")
        
    # --- _save & _load calculated data (unchanged) ---
    def _save_calculated_data_to_cache(self) -> None:
        if not self.cache_dir:
            return
        try:
            # Prepare serializable stats
            serializable_stats = {}
            for k, v in self.global_stats.items():
                if isinstance(v, np.ndarray):
                    # Convert numpy arrays to lists for JSON serialization
                    # Make sure to convert to native Python types
                    serializable_stats[k] = [int(x) if isinstance(x, np.integer) else float(x) for x in v.tolist()]
                elif isinstance(v, (int, float, str, bool, list, dict)):
                    serializable_stats[k] = v
                elif isinstance(v, (np.integer, np.floating)):
                    # Convert numpy scalars to Python types
                    serializable_stats[k] = v.item()
                else:
                    # For other types, convert to string
                    serializable_stats[k] = str(v)
            
            # Save to JSON
            with open(self.cache_dir / 'global_stats.json', 'w') as f:
                json.dump(serializable_stats, f, indent=2)
    
            # Save user segments
            serializable_segments = {
                k: list(v) if isinstance(v, (set, list)) else v
                for k, v in self.user_segments.items()
            }
            with open(self.cache_dir / 'user_segments.json', 'w') as f:
                json.dump(serializable_segments, f)
    
            # Save dataframes
            if self.product_popularity is not None:
                self.product_popularity.write_parquet(
                    self.cache_dir / 'product_popularity.parquet'
                )
            if self.category_popularity is not None:
                self.category_popularity.write_parquet(
                    self.cache_dir / 'category_popularity.parquet'
                )
    
            # Save SKU properties dict
            with open(self.cache_dir / 'sku_properties_dict.pkl', 'wb') as f:
                pickle.dump(self.sku_properties_dict, f)
    
            self.logger.info("Calculated data saved to cache.")
        except Exception as e:
            self.logger.error(f"Failed to save calculated data: {e}", exc_info=True)

    def _load_calculated_data_from_cache(self) -> bool:
            if not self.cache_dir:
                return False
            all_loaded = True
            try:
                try:
                    with open(self.cache_dir / 'global_stats.json','r') as f:
                        self.global_stats = json.load(f)
                except Exception:
                    self.logger.warning("Cache miss: global_stats.json")
                    self.global_stats = {}
                    all_loaded = False
                    
                try:
                    with open(self.cache_dir / 'user_segments.json','r') as f:
                        self.user_segments = json.load(f)
                except Exception:
                    self.logger.warning("Cache miss: user_segments.json")
                    self.user_segments = {}
                    all_loaded = False
                    
                try:
                    self.product_popularity = pl.read_parquet(
                        self.cache_dir / 'product_popularity.parquet'
                    )
                except Exception:
                    self.logger.warning("Cache miss: product_popularity.parquet")
                    self.product_popularity = None
                    all_loaded = False
                    
                try:
                    self.category_popularity = pl.read_parquet(
                        self.cache_dir / 'category_popularity.parquet'
                    )
                except Exception:
                    self.logger.warning("Cache miss: category_popularity.parquet")
                    self.category_popularity = None
                    
                try:
                    filtered_pkl = self.cache_dir / 'sku_properties_dict_filtered.pkl'
                    full_pkl = self.cache_dir / 'sku_properties_dict.pkl'
                    
                    if filtered_pkl.exists():
                        with open(filtered_pkl, 'rb') as f:
                            self.sku_properties_dict = pickle.load(f)
                        self.logger.info(f"Loaded FILTERED {len(self.sku_properties_dict)} SKU properties")
                    else:
                        with open(full_pkl, 'rb') as f:
                            self.sku_properties_dict = pickle.load(f)
                        self.logger.info(f"Loaded FULL {len(self.sku_properties_dict)} SKU properties")
                except Exception:
                    self.logger.warning("Cache miss: sku_properties_dict.pkl")
                    self.sku_properties_dict = {}
                    all_loaded = False
                    
                # ✅ FIX : J'ai SUPPRIMÉ le bloc problématique qui utilisait relevant_client_ids
                # Le filtrage des SKU doit se faire dans load_data(), pas ici !
                        
                if all_loaded:
                    self.logger.info("Loaded calculated data from cache.")
            except Exception as e:
                self.logger.error(f"Error loading cache: {e}")
                all_loaded = False
            return all_loaded

    # ------------------------------------------------------------------ #
    # _compute_global_statistics (lazy-accelerated)                     #
    # ------------------------------------------------------------------ #
    def _compute_global_statistics(self) -> None:
        self.logger.info("Computing global statistics…")
        if self.lazy_all is None:
            return
        self.global_stats = {}

        # total unique users
        users_lf = self.lazy_all.select(pl.col('client_id')).unique()
        self.global_stats['total_users'] = users_lf.collect(engine='streaming').height

        # event counts
        cnts = self.lazy_all.group_by('event_type').agg(pl.count()).collect(engine='streaming')
        self.global_stats['event_counts'] = {
            row[0]: row[1] for row in cnts.iter_rows()
        }

        # transition matrix
        try:
            self.global_stats['transition_matrix'] = self._compute_global_transition_matrix()
        except Exception as e:
            self.logger.error(f"Err transition_matrix: {e}")
            self.global_stats['transition_matrix'] = {}

        # buyer percentage
        buyers = (
            self.lazy_all
              .filter(pl.col('event_type')=='product_buy')
              .select(pl.col('client_id')).unique()
              .collect(engine='streaming').height
        )
        tot = self.global_stats['total_users'] or 1
        self.global_stats['buyer_percentage'] = buyers / tot

        # RFM recencies
        recs_lf = (
            self.lazy_all
              .filter(pl.col('event_type')=='product_buy')
              .group_by('client_id')
              .agg(pl.max('timestamp').alias('last_purchase_ts'))
        )
        df = recs_lf.collect(engine='streaming')
        now_ts = datetime.now()
        recs = [ (now_ts - row['last_purchase_ts']).days 
                 for row in df.to_dicts() if row['last_purchase_ts']]
        self.global_stats['rfm_recencies'] = np.array(recs, dtype=int)

        # run other global stats functions
        for fn in (
            self._compute_cart_to_purchase_times,
            self._identify_global_sessions,
            self._compute_product_popularities,
            self._compute_category_popularities,
            self._compute_category_centrality,
            self._compute_global_co_pairs
        ):
            try:
                fn()
            except Exception as e:
                self.logger.error(f"Err {fn.__name__}: {e}")

        self.logger.info(f"Global stats computed: {list(self.global_stats.keys())}")

    def _compute_global_transition_matrix(self) -> dict:
        if self.lazy_all is None:
            return {}
        try:
            df = (
                self.lazy_all
                  .select(["client_id","timestamp","event_type"] )
                  .sort(["client_id","timestamp"])           
                  .with_columns(
                      pl.col("event_type").shift(1).over("client_id").alias("prev_event_type")
                  )
                  .filter(pl.col("prev_event_type").is_not_null())
                  .group_by(["prev_event_type","event_type"]) 
                  .agg(pl.col("event_type").count().alias("count"))  # Utiliser pl.col().count()
                  .collect(engine='streaming')
            )
            totals = (
                df.group_by("prev_event_type").agg(pl.col("count").sum().alias("total"))  # Utiliser pl.col().sum()
            )
            df = df.join(totals, on="prev_event_type").with_columns(
                (pl.col("count")/pl.col("total")).alias("probability")
            )
            types = self.lazy_all.select(pl.col("event_type")).unique().collect(engine='streaming')["event_type"].to_list()
            probs = defaultdict(lambda: defaultdict(float))
            for row in df.iter_rows(named=True):
                probs[row['prev_event_type']][row['event_type']] = row['probability']
            return {f: {t: probs[f].get(t,0.0) for t in types} for f in types}
        except Exception as e:
            self.logger.error(f"Err transition matrix calc: {e}")
            return {}

    def _compute_cart_to_purchase_times(self) -> None:
        if self.lazy_all is None:
            return
        lf = (self.lazy_all
          .filter(pl.col('sku').is_not_null() & pl.col('event_type').is_in(['add_to_cart','product_buy']))
          .select(['client_id','sku','event_type','timestamp'])
             )
        df = lf.collect(engine='streaming')
        if df.filter(pl.col('event_type')=='product_buy').is_empty() or df.filter(pl.col('event_type')=='add_to_cart').is_empty():
            return
        buys = df.filter(pl.col('event_type')=='product_buy').rename({"timestamp":"purchase_ts"})
        carts = df.filter(pl.col('event_type')=='add_to_cart').rename({"timestamp":"cart_ts"})
        joined = buys.join(carts, on=["client_id","sku"]).filter(pl.col("cart_ts")<pl.col("purchase_ts"))
        if joined.is_empty():
            return
        rec = joined.group_by(["client_id","sku","purchase_ts"]).agg(pl.max("cart_ts").alias("last_cart_ts"))
        diffs = rec.with_columns(((pl.col("purchase_ts")-pl.col("last_cart_ts")).dt.total_seconds()/60).alias("diff_min"))
        vals = diffs.filter(pl.col("diff_min")>0)["diff_min"]
        if vals.len()>0:
            self.global_stats['avg_cart_to_purchase_time'] = vals.mean()
            self.global_stats['median_cart_to_purchase_time'] = vals.median()
            self.logger.info(f"Cart-to-purchase: Avg={self.global_stats['avg_cart_to_purchase_time']:.2f}m, Median={self.global_stats['median_cart_to_purchase_time']:.2f}m")

    def _identify_global_sessions(self, session_gap_minutes: int = 30) -> None:
        if self.lazy_all is None:
            return
        df = (
            self.lazy_all
              .select(["client_id","timestamp"])  
              .sort(["client_id","timestamp"])    
              .with_columns(
                  (pl.col("timestamp").diff().over("client_id")/timedelta(minutes=1)).alias("delta_min")
              )
              .with_columns(
                  ((pl.col("delta_min")>session_gap_minutes)|pl.col("delta_min").is_null()).alias("new_sess")
              )
              .with_columns(
                  pl.col("new_sess").cum_sum().over("client_id").alias("sid")
              )
        ).collect(engine='streaming')
        
        stats = df.group_by("sid").agg([
            pl.col("timestamp").min().alias("start"),
            pl.col("timestamp").max().alias("end"),
            pl.col("sid").count().alias("count")  # Utiliser pl.col().count()
        ]).with_columns(
            ((pl.col("end")-pl.col("start")).dt.total_seconds()/60).alias("duration_min")
        )
        
        if stats.height>0:
            agg = stats.select([
                pl.col("duration_min").mean().alias("avg_session_duration"),
                pl.col("duration_min").median().alias("median_session_duration"),
                pl.col("count").mean().alias("avg_session_events"),
                pl.col("count").median().alias("median_session_events")
            ]).row(0, named=True)
            self.global_stats.update(agg)
            upu = df.group_by("client_id").agg(pl.col("sid").n_unique().alias("sessions")).select(pl.col("sessions").mean()).item()
            self.global_stats['avg_sessions_per_user'] = upu
            self.logger.info(f"Global Session Stats: AvgDur={agg['avg_session_duration']:.2f}m, AvgEvt={agg['avg_session_events']:.1f}, AvgSess/User={upu:.1f}")

    def _compute_product_popularities(self) -> None:
        if self.lazy_all is None:
            return
        df = (
            self.lazy_all
              .filter(pl.col('sku').is_not_null())
              .group_by(['sku','event_type'])
              .agg(pl.col('sku').count().alias('count'))
        ).collect(engine='streaming')
        df = (df.pivot(index='sku', columns='event_type', values='count', aggregate_function='first')
              .fill_null(0))
        mapping = {'page_visit':'view_count','add_to_cart':'cart_count','product_buy':'purchase_count'}
        df = df.rename({k:v for k,v in mapping.items() if k in df.columns})
        for col in ['view_count','cart_count','purchase_count']:
            if col not in df.columns:
                df = df.with_columns(pl.lit(0).alias(col))
        df = df.with_columns(
            pl.when(pl.col('view_count').sum() == 0)
            # Si pas de vues du tout, formule ajustée
            .then(pl.col('cart_count')*2 + pl.col('purchase_count')*10)
            # Sinon, formule normale
            .otherwise(pl.col('view_count')*1 + pl.col('cart_count')*3 + pl.col('purchase_count')*10)
            .alias('popularity_score')
        )
        df = df.with_columns([
            (pl.when(pl.col('view_count')>0).then(pl.col('cart_count')/pl.col('view_count')).otherwise(0.0)).alias('view_to_cart_rate'),
            (pl.when(pl.col('cart_count')>0).then(pl.col('purchase_count')/pl.col('cart_count')).otherwise(0.0)).alias('cart_to_purchase_rate'),
            (pl.when(pl.col('view_count')>0).then(pl.col('purchase_count')/pl.col('view_count')).otherwise(0.0)).alias('view_to_purchase_rate')
        ])
        self.product_popularity = df
        try:
            scores = df['popularity_score'].to_numpy()
            global POP_QUANT_EDGES
            POP_QUANT_EDGES = [float(np.quantile(scores,q)) for q in (0.25,0.5,0.75)]
            self.pop_score_by_sku = {r['sku']:r['popularity_score'] for r in df[['sku','popularity_score']].to_dicts()}
        except Exception as e:
            self.logger.warning(f"Unable to compute popularity quantiles: {e}")
            self.pop_score_by_sku = {}
        self.logger.info(f"Computed product popularity for {df.height} SKUs.")
        # category popularity similar pattern (omitted)
        
    def _compute_category_popularities(self) -> None:
        """Compute popularity metrics for categories based on product events"""
        if self.lazy_all is None:
            return
            
        # Aggregate events by category
        df = (
            self.lazy_all
              .filter(pl.col('category_id').is_not_null())
              .group_by(['category_id', 'event_type'])
              .agg(pl.len().alias('count'))
        ).collect(engine='streaming')
        
        # Pivot to get counts per event type
        df = (df.pivot(index='category_id', columns='event_type', values='count', aggregate_function='first')
              .fill_null(0))
        
        # Map event types to count columns
        mapping = {
            'page_visit': 'view_count',
            'add_to_cart': 'cart_count', 
            'product_buy': 'purchase_count'
        }
        df = df.rename({k: v for k, v in mapping.items() if k in df.columns})
        
        # Ensure all columns exist
        for col in ['view_count', 'cart_count', 'purchase_count']:
            if col not in df.columns:
                df = df.with_columns(pl.lit(0).alias(col))
        
        # Calculate popularity score (same formula as products)
        df = df.with_columns(
            pl.when(pl.col('view_count').sum() == 0)
            .then(pl.col('cart_count')*2 + pl.col('purchase_count')*10)
            .otherwise(pl.col('view_count')*1 + pl.col('cart_count')*3 + pl.col('purchase_count')*10)
            .alias('category_popularity_score')
        )
        
        self.category_popularity = df
        self.logger.info(f"Computed category popularity for {df.height} categories.")
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # === Lazy-aware Helpers for Global Computations ==================== #
    # ------------------------------------------------------------------ #

    def _build_category_centrality(self) -> None:
        """
        Builds a directed graph Cat_i→Cat_j from successive page_visit or product_buy events,
        computes PageRank, all via lazy Polars to avoid full materialization.
        """
        if self.lazy_all is None:
            self.cat_centrality = {}
            return
        # collect only category transitions
        df = (
            self.lazy_all
              .filter(pl.col('category_id').is_not_null())
              .select(['client_id','timestamp','category_id'])
              .sort(['client_id','timestamp'])
              .with_columns(
                  pl.col('category_id').shift(-1).over('client_id').alias('next_cat')
              )
              .filter(pl.col('next_cat').is_not_null())
              .select(['category_id','next_cat'])
        ).collect(engine='streaming')

        # build graph
        G = nx.DiGraph()
        for row in df.iter_rows(named=True):
            src, dst = int(row['category_id']), int(row['next_cat'])
            if src == dst:
                continue
            if G.has_edge(src, dst):
                G[src][dst]['weight'] += 1
            else:
                G.add_edge(src, dst, weight=1)
        if G.number_of_nodes() == 0:
            self.cat_centrality = {}
            self.logger.info("Category centrality skipped (empty graph).")
            return
        # PageRank
        pr = nx.pagerank(G, weight='weight', max_iter=100, tol=1e-4)
        self.cat_centrality = pr
        self.logger.info(
            f"Built centrality maps  • SKU:{len(getattr(self,'sku_centrality',{}))}  • CAT:{len(pr)}"
        )

    def _compute_global_co_occurrences(self, session_gap: int = 30) -> None:
        """
        Counts co-occurring SKU and category pairs within user sessions (lazy + collect small slice).
        """
        if self.lazy_all is None:
            return
        # collect relevant events
        df = (
            self.lazy_all
              .filter(pl.col('event_type').is_in(['product_buy','add_to_cart','page_visit']))
              .select(['client_id','timestamp','sku','category_id'])
              .sort(['client_id','timestamp'])
        ).collect(engine='streaming')
        # identify sessions
        diff = df['timestamp'].diff().dt.total_seconds() / 60
        df = df.with_columns(((diff.is_null()) | (diff > session_gap)).cum_sum().alias('sess_id'))
        sku_pairs = Counter()
        cat_pairs = Counter()
        from itertools import combinations
        for (_cid, sess), sub in df.group_by(['client_id','sess_id']):
            skus = sub['sku'].drop_nulls().unique().to_list()
            cats = sub['category_id'].drop_nulls().unique().to_list()
            for i,j in combinations(sorted(set(skus)),2):
                sku_pairs[(int(i),int(j))] += 1
            for c1,c2 in combinations(sorted(set(cats)),2):
                cat_pairs[(int(c1),int(c2))] += 1
        self.global_stats['global_sku_pairs'] = dict(sku_pairs)
        self.global_stats['global_cat_pairs'] = dict(cat_pairs)
        self.logger.info(
            f"Global co-occurrences  SKU_pairs:{len(sku_pairs)}  CAT_pairs:{len(cat_pairs)}"
        )

    # ------------------------------------------------------------------
    # Segmentation principale : acheteurs / navigateurs actifs, etc.
    # ------------------------------------------------------------------
    def _segment_users(self) -> None:
        """
        Attaches each client to high-level behavioral segments,
        using lazy scans + small collects.
        """
        self.logger.info("Segmenting users (with dataset-relative recency)...")
        if self.lazy_all is None:
            self.logger.warning("lazy_all is None; skipping segmentation.")
            return
    
        # 1) Event counts per client & type (collect BEFORE pivot)
        df_counts = (
            self.lazy_all
              .group_by(['client_id','event_type'])
              .agg(pl.col('client_id').count().alias('count'))  # Utiliser pl.col().count() au lieu de pl.count()
              .collect(engine='streaming')
        )
        df_counts = (
            df_counts
              .pivot(
                  index='client_id',
                  columns='event_type',
                  values='count',
                  aggregate_function='first'
              )
              .fill_null(0)
        )
        # ensure all event-type cols exist
        for c in ['page_visit','product_buy','add_to_cart','search_query']:
            if c not in df_counts.columns:
                df_counts = df_counts.with_columns(pl.lit(0).alias(c))
    
        # 2) Last activity timestamp
        df_last = (
            self.lazy_all
              .group_by('client_id')
              .agg(pl.col('timestamp').max().alias('last_ts'))  # Utiliser pl.col().max() au lieu de pl.max()
              .collect(engine='streaming')
        )
        df = df_counts.join(df_last, on='client_id', how='left')
    
        # 3) Recency metrics
        max_ts_df = self.lazy_all.select(pl.col('timestamp').max()).collect(engine='streaming')
        max_ts = max_ts_df.item() if max_ts_df.height > 0 else None
        if max_ts is None:
            max_ts = datetime.now()
    
        # **Plus besoin de collect(engine='streaming') ici : df est déjà un DataFrame**
        now = datetime.now()
        df = df.with_columns([
            pl.Series(
                "days_since_run",
                [(now   - ts).days if ts is not None else 999 for ts in df['last_ts']]
            ),
            pl.Series(
                "days_since_data_end",
                [(max_ts - ts).days if ts is not None else 999 for ts in df['last_ts']]
            ),
        ])
    
        # 4) Flags
        df = df.with_columns([
            (pl.col('product_buy') > 0).alias('is_buyer'),
            ((pl.col('page_visit') >= 5) & (pl.col('days_since_run') <= 30)).alias('active_absolute'),
            ((pl.col('page_visit') >= 5) & (pl.col('days_since_data_end') <= 30)).alias('active_relative')
        ])
    
        # 5) Build segments
        segs = {
            'buyers': df.filter(pl.col('is_buyer'))['client_id'].to_list(),
            'non_buyers': df.filter(~pl.col('is_buyer'))['client_id'].to_list(),
            'active_buyers_absolute': df.filter(pl.col('is_buyer') & pl.col('active_absolute'))['client_id'].to_list(),
            'active_browsers_absolute': df.filter(~pl.col('is_buyer') & pl.col('active_absolute'))['client_id'].to_list(),
            'active_buyers_relative': df.filter(pl.col('is_buyer') & pl.col('active_relative'))['client_id'].to_list(),
            'inactive_buyers_relative': df.filter(pl.col('is_buyer') & ~pl.col('active_relative'))['client_id'].to_list(),
            'active_browsers_relative': df.filter(~pl.col('is_buyer') & pl.col('active_relative'))['client_id'].to_list(),
            'inactive_browsers_relative': df.filter(~pl.col('is_buyer') & ~pl.col('active_relative'))['client_id'].to_list()
        }
    
        # 6) Purchase frequency segments
        df = df.with_columns(
            pl.when(pl.col('product_buy') == 0).then(pl.lit('Non-Buyer'))
              .when(pl.col('product_buy') == 1).then(pl.lit('One-Time Buyer'))
              .when((pl.col('product_buy') >= 2) & (pl.col('product_buy') <= 5)).then(pl.lit('Occasional Buyer'))
              .when(pl.col('product_buy') > 5).then(pl.lit('Frequent Buyer'))
              .alias('purchase_freq')
        )

        segs['one_time_buyers']   = df.filter(pl.col('purchase_freq') == 'One-Time Buyer')['client_id'].to_list()
        segs['occasional_buyers'] = df.filter(pl.col('purchase_freq') == 'Occasional Buyer')['client_id'].to_list()
        segs['frequent_buyers']   = df.filter(pl.col('purchase_freq') == 'Frequent Buyer')['client_id'].to_list()
    
        self.user_segments = segs
    
        # 7) Further sub-segmentation
        try:
            self._segment_users_by_price_sensitivity(df)
            self._segment_users_by_category_behavior(df)
        except Exception as e:
            self.logger.error(f"Err price/cat segmentation: {e}")
    
        self.logger.info(
            f"User segmentation done: Buyers={len(segs['buyers'])}, "
            f"Active relative buyers={len(segs['active_buyers_relative'])}"
        )

    def _segment_users_by_price_sensitivity(self, user_counts: pl.DataFrame) -> None:
        """
        Splits users into price sensitivity segments based on avg add_to_cart vs purchase price buckets,
        computed lazily to avoid full event tables in memory.
        """
        if self.lazy_all is None:
            return
        try:
            # compute avg price for cart and buy per user
            price_lf = (
                self.lazy_all
                  .filter(pl.col('price_bucket').is_not_null() & pl.col('event_type').is_in(['add_to_cart','product_buy']))
                  .group_by(['client_id','event_type'])
                  .agg(pl.mean('price_bucket').alias('avg_price'))
            ).collect(engine='streaming')
            price_lf = price_lf.pivot(index='client_id', columns='event_type', values='avg_price', aggregate_function='first')
            df_price = user_counts.select('client_id').join(price_lf, on='client_id', how='left').fill_null(0)
            if 'add_to_cart' not in df_price.columns or 'product_buy' not in df_price.columns:
                return

            df_price = df_price.with_columns(
                (pl.when(pl.col('add_to_cart')>0)
                   .then(pl.col('product_buy')/pl.col('add_to_cart'))
                   .otherwise(None)
                 ).alias('sensitivity_ratio')
            )
            valid = df_price.filter(pl.col('sensitivity_ratio').is_not_null() & pl.col('sensitivity_ratio').is_finite())['sensitivity_ratio']
            if valid.len() > 10:
                low, high = valid.quantile(0.33), valid.quantile(0.66)
                self.user_segments['price_sensitive'] = df_price.filter(pl.col('sensitivity_ratio') < low)['client_id'].to_list()
                self.user_segments['price_moderate'] = df_price.filter((pl.col('sensitivity_ratio') >= low) & (pl.col('sensitivity_ratio') <= high))['client_id'].to_list()
                self.user_segments['price_insensitive'] = df_price.filter(pl.col('sensitivity_ratio') > high)['client_id'].to_list()
                self.logger.info(
                    f"Price segmentation: Sens={len(self.user_segments['price_sensitive'])}, "
                    f"Mod={len(self.user_segments['price_moderate'])}, "
                    f"Insens={len(self.user_segments['price_insensitive'])}"
                )
        except Exception as e:
            self.logger.error(f"Err price segmentation: {e}")

    def _segment_users_by_category_behavior(self, user_counts: pl.DataFrame) -> None:
        """
        Classifies users into category loyalty/exploration segments based on distribution of page_visit counts,
        all computed on a small collected slice.
        """
        if self.lazy_all is None:
            return
        try:
            # 1) Count page visits by category per user
            cat_lf = (
                self.lazy_all
                  .filter((pl.col('event_type')=='page_visit') & pl.col('category_id').is_not_null())
                  .group_by(['client_id','category_id'])
                  .agg(pl.count().alias('view_count'))
            )
            df_cat = cat_lf.collect(engine='streaming')
            if df_cat.height == 0:
                return

            # 2) Aggregate per user: total views, max in one category, num categories
            user_cat = (
                df_cat
                  .group_by('client_id')
                  .agg(
                      pl.sum('view_count').alias('total_views'),
                      pl.max('view_count').alias('max_views_in_one_cat'),
                      pl.count().alias('n_cats')
                  )
                  .with_columns(
                      (pl.col('max_views_in_one_cat')/pl.col('total_views')).alias('category_loyalty_score')
                  )
            )

            # 3) Join with full user list
            df_stats = user_counts.select('client_id').join(user_cat, on='client_id', how='left').fill_null(0)

            # Thresholds
            loyal_thresh = 0.75
            explorer_thresh = 0.40

            # 4) Assign segments
            self.user_segments['category_loyal'] = (
                df_stats.filter(pl.col('category_loyalty_score') >= loyal_thresh)['client_id'].to_list()
            )
            self.user_segments['category_explorer'] = (
                df_stats.filter((pl.col('category_loyalty_score') <= explorer_thresh) & (pl.col('n_cats') >= 3))['client_id'].to_list()
            )
            self.user_segments['moderate_explorer'] = (
                df_stats.filter((pl.col('category_loyalty_score') > explorer_thresh) & (pl.col('category_loyalty_score') < loyal_thresh))['client_id'].to_list()
            )

            self.logger.info(
                f"Category segmentation: Loyal={len(self.user_segments['category_loyal'])}, "
                f"Moderate={len(self.user_segments['moderate_explorer'])}, "
                f"Explorer={len(self.user_segments['category_explorer'])}"
            )
        except Exception as e:
            self.logger.error(f"Err category segmentation: {e}")


    # --- Getters ---
    def get_feature_extractors(self) -> Dict[str, FeatureExtractorBase]:
        if not self._extractors:
            self.logger.debug("Initializing feature extractors...")
            self._extractors = {}
            self._extractors['temporal'] = TemporalFeatureExtractor(self)
            self._extractors['sequence'] = SequenceFeatureExtractor(self)
            self._extractors['churn_propensity'] = ChurnPropensityFeatureExtractor(self)
            if self.top_skus:
                self._extractors['top_sku'] = TopSKUFeatureExtractor(self)
            if self.top_categories:
                self._extractors['top_category'] = TopCategoryFeatureExtractor(self)

            if self.events_df is not None:
                if 'category_id' in self.events_df.columns or 'sku' in self.events_df.columns:
                    self._extractors['graph'] = GraphFeatureExtractor(self)
                if 'query' in self.events_df.columns:
                    self._extractors['intent'] = IntentFeatureExtractor(self)
                if 'price_bucket' in self.events_df.columns:
                    self._extractors['price'] = PriceFeatureExtractor(self)
                if self.product_popularity is not None:
                    self._extractors['social'] = SocialFeatureExtractor(self)
                if self.sku_properties_dict:
                    self._extractors['name_embedding'] = NameEmbeddingExtractor(self)
                if hasattr(self, 'sku_cluster_map'):
                    self._extractors['custom_behavior'] = CustomBehaviorFeatureExtractor(self)

            self.logger.info(f"Initialized extractors: {list(self._extractors.keys())}")
        return self._extractors


    def get_client_events(self, client_id: int) -> pl.DataFrame:
        """Ne charge que les events d'UN client"""
        # PRIORITÉ au mode streaming
        if self.lazy_all is not None:
            return self._collect_client_events(client_id)
        # Fallback si pas de lazy pipeline
        elif self.events_df is not None:
            return self.events_df.filter(pl.col('client_id') == client_id)
        # Dernier recours : scan direct
        else:
            return pl.scan_parquet(self.cache_dir / "events_1m_clients.parquet")\
                     .filter(pl.col('client_id') == client_id)\
                     .collect()



    def get_client_segment(self, client_id: int) -> dict:
        segments = {}
        for segment_name, users in self.user_segments.items():
            if isinstance(users, (list, set)) and client_id in users: segments[segment_name] = True
        return segments


    # --- Multi-Resolution History Helpers ---
    def _format_event_for_history(self, event_row: Dict[str, Any]) -> str:
        event_type = event_row.get("event_type"); sku = event_row.get("sku"); url = event_row.get("url")
        query = event_row.get("query"); ts = event_row.get("timestamp")
        ts_str = ts.strftime('%Y%m%d-%H%M') if isinstance(ts, datetime) else "NT" # Format plus court
        text_parts = [f"[{ts_str}] E:{event_type or '?'}"]
        try:
            if sku is not None:
                sku_int = int(sku); text_parts.append(f" S:{sku_int}")
                props = self.sku_properties_dict.get(sku_int, {})
                if props.get('category') is not None: text_parts.append(f" C:{props['category']}")
                if props.get('price') is not None: text_parts.append(f" P:{props['price']}")
            elif url is not None: text_parts.append(f" U:{url}")
            elif query is not None: text_parts.append(f" Q:{hash(str(query))%10000:04d}") # Hash court pour Q
        except Exception: pass # Ignorer erreurs de formatage individuelles
        return "".join(text_parts)

    def _generate_detailed_events_text(self, client_events: pl.DataFrame, limit=30) -> str:
        if client_events.height == 0: return "No recent activity."
        recent_events_rows = client_events.sort("timestamp", descending=True).head(limit).to_dicts()
        event_texts = [self._format_event_for_history(row) for row in recent_events_rows]
        return "\n".join(filter(None, event_texts))

    def _generate_summarized_events_text(self, client_events: pl.DataFrame, limit=10) -> str:
        if client_events.height == 0: return "No medium-term activity."
        summary = [f"Event count: {client_events.height}"]
        event_counts = client_events.group_by('event_type').agg(pl.count().alias('count')).sort('count', descending=True)
        summary.append("Event Types: " + ", ".join([f"{row['event_type']}:{row['count']}" for row in event_counts.iter_rows(named=True)]))
        if 'category_id' in client_events.columns:
            purchases = client_events.filter((pl.col('event_type') == pl.lit('product_buy', dtype=pl.Categorical)) & pl.col('category_id').is_not_null())
            if purchases.height > 0:
                category_counts = purchases.group_by('category_id').agg(pl.count().alias('count')).sort('count', descending=True)
                top_cats = category_counts.head(limit).to_dicts()
                summary.append("Top Purchased Cats: " + ", ".join([f"[CAT_{c['category_id']}]:{c['count']}" for c in top_cats]))
        return "\n".join(summary)

    def _generate_aggregated_events_text(self, client_events: pl.DataFrame) -> str:
        if client_events.height == 0: return "No historical activity."
        summary = []
        if client_events.height >= 2:
            first_ts, last_ts = client_events['timestamp'].min(), client_events['timestamp'].max()
            if first_ts and last_ts: tenure_days = (last_ts - first_ts).days; summary.append(f"Hist. Span: ~{tenure_days}d (end {last_ts.date()})")
        event_counts = client_events.group_by('event_type').agg(pl.count().alias('count')).sort('count', descending=True)
        summary.append("Hist. Event Counts: " + ", ".join([f"{r['event_type']}:{r['count']}" for r in event_counts.iter_rows(named=True)]))
        if 'category_id' in client_events.columns:
             purchases = client_events.filter((pl.col('event_type') == pl.lit('product_buy', dtype=pl.Categorical)) & pl.col('category_id').is_not_null())
             if purchases.height > 0:
                 cat_counts = purchases.group_by('category_id').agg(pl.count().alias('count')).sort('count', descending=True)
                 summary.append("Top Hist. Purchased Cats: " + ", ".join([f"[CAT_{c['category_id']}]:{c['count']}" for c in cat_counts.head(3).to_dicts()]))
        return "\n".join(summary)
    
  
  # --- Raw sequence formatting ---
    def _format_raw_event(self, event_row: Dict[str, Any]) -> str:
        parts = [f"EVENT: {event_row['event_type']}"]
        sku = event_row.get('sku')
        sku_int: Optional[int] = int(sku) if sku is not None else None   

        props = self.sku_properties_dict.get(int(sku), {}) if sku is not None else {}
        etype = event_row['event_type']
        if etype in ('product_buy', 'add_to_cart', 'remove_from_cart') and sku is not None:
            parts.append(f"SKU:[SKU_{int(sku)}]")
            if etype != 'remove_from_cart':
                cat = props.get('category'); price = props.get('price')
                if cat is not None: parts.append(f"CAT:[CAT_{cat}]")
                if price is not None: parts.append(f"PRICE:[PRICE_{price}]")
            name_emb = props.get('name')
            if isinstance(name_emb, str) and name_emb.startswith('[') and name_emb.endswith(']'):
                clean = name_emb.strip('[]').replace(',', ' ')
                parts.append(f"NAME_EMB:[{clean}]")
        elif etype == 'page_visit' and event_row.get('url'):
            parts.append(f"URL:[URL_{event_row['url']}]" )
        elif etype == 'search_query' and event_row.get('query'):
            q = event_row['query']
            if isinstance(q, str) and q.startswith('[') and q.endswith(']'):
                clean = q.strip('[]').replace(',', ' ')
                parts.append(f"QUERY_EMB:[{clean}]")
            else:
                parts.append("QUERY_EMB:[InvalidFormat]")
        if hasattr(self, 'pop_score_by_sku') and self.pop_score_by_sku:
            score = self.pop_score_by_sku.get(sku_int)
            q_tag = pop_bin(score)
            parts.append(f"POP_Q:[{q_tag}]") 
        # --- NEW: how-many-days-ago bucket (coarse) ------------------
        if ts := event_row.get('timestamp'):
            if isinstance(ts, datetime):
                days_ago = (datetime.now() - ts).days
                if   days_ago <= 1:      parts.append("AGE:[D_0-1]")
                elif days_ago <= 7:      parts.append("AGE:[D_1-7]")
                elif days_ago <= 30:     parts.append("AGE:[D_7-30]")
                elif days_ago <= 180:    parts.append("AGE:[D_30-180]")
                else:                    parts.append("AGE:[D_180+]")            
        return " ".join(parts)

    def _generate_raw_sequence(self, client_events: pl.DataFrame, max_events: int = MAX_HISTORY_EVENTS_TO_CONSIDER) -> str:
        rows = client_events.sort('timestamp', descending=True).head(max_events).to_dicts()
        rows = list(reversed(rows))  # ordre chronologique

        seq_tokens = []
        prev_ts: Optional[datetime] = None
        SESSION_GAP_MIN = 30
        open_session = False

        for i, row in enumerate(rows):
            ts = row['timestamp']
            py_ts = ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts

            # --- Détection nouveau bloc session ---
            new_session = False
            if prev_ts is None:
                new_session = True
            else:
                delta_min = (py_ts - prev_ts).total_seconds() / 60
                if delta_min > SESSION_GAP_MIN:
                    # fermer précédente
                    if open_session:
                        seq_tokens.append("<SESS_END>")
                    new_session = True
            if new_session:
                seq_tokens.append("<SESS_START>")
                open_session = True

            # --- Encodage standard de l’événement ---
            txt = self._format_raw_event(row)
            tod_tok = f"TOD:[{discretize_time_of_day(py_ts)}]"
            dow_tok = f"DOW:[{discretize_day_of_week(py_ts)}]"
            if prev_ts is None or new_session:
                delta_tok = f"TIME_DELTA:[{SESSION_START_TOKEN}]"
            else:
                delta = py_ts - prev_ts
                delta_tok = f"TIME_DELTA:[{discretize_timedelta(delta)}]"
            prev_ts = py_ts
            seq_tokens.append(" ".join([txt, tod_tok, dow_tok, delta_tok]))

        if open_session:
            seq_tokens.append("<SESS_END>")

        return SEP_TOKEN.join(seq_tokens)    
    
    def _compute_compact_metrics(self, events: pl.DataFrame) -> list[str]:
        """
        Renvoie des tags compacts (≤10 tokens chacun) :
          NAME_STD, Δ$, BURST, CAT_PR_TOP, H_cat, H_price
        S'adapte aux schémas avec event_type / price_bucket / emb_str.
        """
        tags = []
        
        # ---------- helpers internes -------------------------------------------
        evt_col   = "event_type" if "event_type" in events.columns else "event"
        price_col = "price" if "price" in events.columns else "price_bucket"
        
        def _bucket_to_num(s: pl.Series) -> np.ndarray:
            """Convertit price_bucket en valeurs numériques, en gérant les nulls"""
            if s.dtype == pl.Int64 or s.dtype == pl.Float64:
                return s.to_numpy()
            
            # Filtrer les nulls avant d'appliquer str.replace
            if s.null_count() > 0:
                # Option 1: Remplacer les nulls par une valeur par défaut
                s = s.fill_null("0")
            
            # Maintenant on peut appliquer str.replace en toute sécurité
            return s.str.replace(r"[^0-9]", "").cast(pl.Int32, strict=False).fill_null(0).to_numpy()
        
        # ---------- NAME_STD ----------------------------------------------------
        emb = None
        if "name_embedding" in events.columns:
            emb = np.vstack([vec for vec in events["name_embedding"].to_list() if vec is not None])
        elif "emb_str" in events.columns:
            str_vecs = [
                v for v in events["emb_str"].to_list()
                if v is not None and isinstance(v, (str, bytes)) and v.strip()
            ]
            if str_vecs:
                parsed_vecs = []
                for v in str_vecs:
                    try:
                        clean_v = v.strip()
                        if clean_v.startswith('[') and clean_v.endswith(']'):
                            clean_v = clean_v[1:-1]
                        vec = np.fromstring(clean_v, dtype=np.float32, sep=" ")
                        if vec.size > 0 and np.all(np.isfinite(vec)):
                            parsed_vecs.append(vec)
                    except Exception:
                        continue
                
                if parsed_vecs:
                    try:
                        emb = np.vstack(parsed_vecs)
                    except Exception:
                        emb = None
        
        if emb is not None and emb.size > 0:
            try:
                if emb.dtype.kind in ['f', 'i', 'u']:
                    name_std = round(float(np.std(emb)), 2)
                    tags.append(f"NAME_STD:{name_std}")
            except Exception:
                pass
        
        # ---------- Δ Panier / Prix ---------------------------------------------
        if price_col in events.columns:
            buy_mask   = pl.col(evt_col) == "product_buy"
            cart_mask  = pl.col(evt_col) == "add_to_cart"
            
            # Filtrer les événements avec prix non-null
            buy_events = events.filter(buy_mask & pl.col(price_col).is_not_null())
            cart_events = events.filter(cart_mask & pl.col(price_col).is_not_null())
            
            if buy_events.height > 0 and cart_events.height > 0:
                buy_vals = _bucket_to_num(buy_events[price_col])
                cart_vals = _bucket_to_num(cart_events[price_col])
                
                if buy_vals.size and cart_vals.size and cart_vals.mean() > 0:
                    delta_pct = 100 * (buy_vals.mean() - cart_vals.mean()) / cart_vals.mean()
                    tags.append(f"Δ$:{delta_pct:+.0f}%")
        
        # ---------- BURST score --------------------------------------------------
        if events.height and "timestamp" in events.columns:
            try:
                hour_counts = np.bincount(events["timestamp"].dt.hour().fill_null(0).to_numpy(), minlength=24)
                mu, var = hour_counts.mean(), hour_counts.var()
                if mu > 0:
                    tags.append(f"BURST:{round(var/mu,2)}")
            except Exception:
                pass
        
        # ---------- Graph centralité catégorie ----------------------------------
        if hasattr(self, "category_centrality") and "category_id" in events.columns:
            cats = [c for c in events["category_id"].drop_nulls().to_list()
                    if c in self.category_centrality]
            if cats:
                top_cat = max(cats, key=lambda c: self.category_centrality[c])
                score = round(self.category_centrality[top_cat], 2)
                tags.append(f"CAT_PR_TOP:{top_cat}({score})")
        
        # ---------- Entropies ----------------------------------------------------
        from collections import Counter
        
        # Helper pour calculer l'entropie Shannon
        def _shannon_entropy(counter: Counter) -> float:
            n = sum(counter.values())
            if n == 0:
                return 0.0
            from math import log2
            return -sum((c / n) * log2(c / n) for c in counter.values() if c > 0)
        
        # Entropie des catégories
        if "category_id" in events.columns:
            cat_list = events["category_id"].drop_nulls().to_list()
            if cat_list:
                cat_entropy = _shannon_entropy(Counter(cat_list))
                tags.append(f"H_cat:{round(cat_entropy,1)}")
        
        # Entropie des prix
        if price_col in events.columns:
            price_events = events.filter(pl.col(price_col).is_not_null())
            if price_events.height > 0:
                price_vals = _bucket_to_num(price_events[price_col])
                price_vals = price_vals[price_vals > 0]  # Filtrer les 0
                if price_vals.size > 0:
                    price_entropy = _shannon_entropy(Counter(price_vals))
                    tags.append(f"H_price:{round(price_entropy,1)}")
        
        return tags
        
    def _compute_extra_short_metrics(
        self,
        cid: int,
        events: pl.DataFrame,
        now: datetime
    ) -> list[str]:
        """
        Retourne une liste de tags ultra-compacts :
            - centralité SKU / Catégorie
            - entropie jour-semaine
            - variance horaire (circular)
            - rang de récence
            - durée de vie (span)
        Chaque tag est déjà « deduplicable » par son prefix.
        """
        tags: list[str] = []
    
        # ---------- 1) Centralité du SKU préféré ------------------------------
        buys = events.filter(pl.col("event_type") == "product_buy")
        if buys.height:
            sku_mode = buys["sku"].drop_nulls().mode()
            if not sku_mode.is_empty():
                fav_sku = int(sku_mode[0])        # on prend simplement le 1ᵉʳ mode
                if fav_sku in getattr(self, "sku_centrality", {}):
                    tags.append(f"CENT_SKU:{self.sku_centrality[fav_sku]:.2f}")
    
        # ---------- 2) Centralité de la Catégorie favorite --------------------
        if "category_id" in events.columns:
            cat_mode = events["category_id"].drop_nulls().mode()
            if not cat_mode.is_empty():
                fav_cat = int(cat_mode[0])        # idem : premier mode
                cat_centrality = getattr(self, "cat_centrality", {})
                if cat_centrality and fav_cat in cat_centrality:
                    tags.append(f"CENT_CAT:{fav_cat}({cat_centrality[fav_cat]:.2f})")
    
        # ---------- 3) Entropie des jours de semaine ----------------------------
        wd_series = events["timestamp"].dt.weekday()  # 0=Mon … 6=Sun
        if wd_series.len():
            wd_counts = np.bincount(wd_series.to_numpy(), minlength=7)
            h_dow = round(float(entropy(wd_counts, base=2)), 2)
            tags.append(f"DOW_ENT:{h_dow}")
    
        # ---------- 4) Variance horaire (circular) ------------------------------
        hr_series = events["timestamp"].dt.hour()
        if hr_series.len():
            angles = hr_series.to_numpy() / 24 * 2 * np.pi
            R = np.abs(np.mean(np.exp(1j * angles)))        # résultante
            tod_var = round(float(1 - R), 2)                # 0→mono-pic, 1→uniforme
            tags.append(f"TOD_VAR:{tod_var}")
    
        # ---------- 5) Rang de récence quantilé (0=très vieux, 1=très récent) ---
        last_ts = events["timestamp"].max()
        if last_ts is not None:
            days_since = (now - last_ts).days
            
            # Vérifier que rfm_recencies existe et est valide
            rfm_recencies = self.global_stats.get("rfm_recencies", [])
            if isinstance(rfm_recencies, np.ndarray) and rfm_recencies.size > 0:
                try:
                    # S'assurer que c'est un array numpy
                    if not isinstance(rfm_recencies, np.ndarray):
                        rfm_recencies = np.array(rfm_recencies, dtype=int)
                        
                    # Vérifier que l'array n'est pas vide et est 1D
                    if rfm_recencies.size > 0 and rfm_recencies.ndim == 1:
                        # S'assurer que l'array est trié
                        rfm_recencies = np.sort(rfm_recencies)
                        rec_q = np.searchsorted(rfm_recencies, days_since, side="right") / len(rfm_recencies)
                        tags.append(f"REC_RANK:{round(1 - rec_q, 2)}")
                    else:
                        self.logger.debug(f"rfm_recencies has invalid shape: {rfm_recencies.shape}")
                except Exception as e:
                    self.logger.debug(f"Error computing recency rank: {e}")
            else:
                self.logger.debug("No rfm_recencies data available")
    
        # ---------- 6) Durée de vie de l'historique -----------------------------
        first_ts = events["timestamp"].min()
        last_ts = events["timestamp"].max()
        if first_ts is not None and last_ts is not None:
            span_days = (last_ts - first_ts).days
            if span_days > 0:
                tags.append(f"LIFETIME:~{span_days}d")
    
        return tags        
    def _build_global_centralities(self) -> None:
        """
        Calcule une centralité « SKU PageRank » à partir des co-occurrences
        de SKU dans les mêmes sessions (30 min). 100 % Polars pour l'I/O,
        Counter + NumPy pour l'agrégation, NetworKit pour le PageRank sparse.
        """
        import numpy as np
        from datetime import timedelta
        from itertools import combinations
        from collections import Counter
    
        # ─── garde-fou ───────────────────────────────────────────────────────
        if self.lazy_all is None:
            self.sku_centrality = {}
            return
    
        # ─── 0)  Filtre SKU populaires ──────────────────────────────────────
        MIN_EVENTS, TOP_K = 50, 2_000  # Changé de MIN_VIEWS à MIN_EVENTS
        pop = self.product_popularity
    
        if pop is None:
            self.logger.warning("No product_popularity → disabling SKU centrality")
            allowed_skus = set()
        else:
            # Utiliser popularity_score qui est toujours présent
            # ou chercher n'importe quelle colonne de count disponible
            if "popularity_score" in pop.columns:
                # Utiliser le score de popularité global
                allowed_skus = set(
                    pop.filter(pl.col("popularity_score") > 0)
                       .sort("popularity_score", descending=True)
                       .head(TOP_K)["sku"]
                       .to_list()
                )
            elif "view_count" in pop.columns:
                # Si view_count existe, l'utiliser
                allowed_skus = set(
                    pop.filter(pl.col("view_count") >= MIN_EVENTS)
                       .sort("view_count", descending=True)
                       .head(TOP_K)["sku"]
                       .to_list()
                )
            elif "cart_count" in pop.columns:
                # Sinon utiliser cart_count
                allowed_skus = set(
                    pop.filter(pl.col("cart_count") >= MIN_EVENTS // 5)  # Seuil plus bas pour les paniers
                       .sort("cart_count", descending=True)
                       .head(TOP_K)["sku"]
                       .to_list()
                )
            elif "purchase_count" in pop.columns:
                # En dernier recours, utiliser purchase_count
                allowed_skus = set(
                    pop.filter(pl.col("purchase_count") >= MIN_EVENTS // 10)  # Seuil encore plus bas
                       .sort("purchase_count", descending=True)
                       .head(TOP_K)["sku"]
                       .to_list()
                )
            else:
                self.logger.warning("No count columns found in product_popularity")
                allowed_skus = set()
                
        self.logger.info("SKU centrality filter → kept %d SKUs", len(allowed_skus))
    
        # Si on n'a pas assez de SKUs, essayer sans filtre
        if len(allowed_skus) < 100 and pop is not None:
            self.logger.info("Too few SKUs after filter, using top SKUs by any metric")
            # Prendre juste les TOP_K SKUs les plus populaires
            allowed_skus = set(
                pop.sort("popularity_score", descending=True)
                   .head(min(TOP_K, pop.height))["sku"]
                   .to_list()
            )
            self.logger.info("Using %d SKUs without strict filtering", len(allowed_skus))
    
        # ─── 1)  Collect (client_id, ts, sku) trié ───────────────────────────
        if not allowed_skus:
            self.sku_centrality = {}
            self.logger.warning("No SKUs to process – centrality skipped")
            return
            
        df_sku = (
            self.lazy_all
                .filter(pl.col("sku").is_in(allowed_skus))
                .select(["client_id", "timestamp", "sku"])
                .sort(["client_id", "timestamp"])
                .collect(engine="streaming")
        )
        if df_sku.is_empty():
            self.sku_centrality = {}
            self.logger.warning("No SKU events after filter – centrality skipped")
            return
    
        # ─── 2)  Session IDs (gap 30 min) ────────────────────────────────────
        # Méthode simple sans window expressions
        df_sku = df_sku.sort(["client_id", "timestamp"])
        
        # Ajouter colonnes décalées
        df_sku = df_sku.with_columns([
            pl.col("client_id").shift(1).alias("prev_client_id"),
            pl.col("timestamp").shift(1).alias("prev_timestamp")
        ])
        
        # Calculer nouvelle session
        df_sku = df_sku.with_columns([
            pl.when(
                (pl.col("client_id") != pl.col("prev_client_id")) |
                ((pl.col("timestamp") - pl.col("prev_timestamp")).dt.total_seconds() / 60 > 30) |
                pl.col("prev_client_id").is_null()
            ).then(1)
            .otherwise(0)
            .alias("new_session")
        ])
        
        # ID de session cumulatif
        df_sku = df_sku.with_columns([
            pl.col("new_session").cum_sum().alias("sid")
        ])
        
        # Nettoyer
        df_sku = df_sku.drop(["prev_client_id", "prev_timestamp", "new_session"])
    
        # ─── 3)  Compte des paires SKU par session (Counter) ─────────────────
        pair_cnt: Counter[tuple[int, int]] = Counter()
        for (_cid, sid), sess in df_sku.group_by(["client_id", "sid"]):
            skus = sess["sku"].drop_nulls().unique().to_list()
            for i, j in combinations(sorted(skus), 2):
                pair_cnt[(int(i), int(j))] += 1
    
        if not pair_cnt:
            self.sku_centrality = {}
            self.logger.warning("No co-occurrences – centrality skipped")
            return
    
        edges   = np.array(list(pair_cnt.keys()), dtype=int)
        weights = np.array(list(pair_cnt.values()), dtype=float)
        src, dst = edges[:, 0], edges[:, 1]
    
        # ─── 4)  PageRank sparse avec helper NetworKit ───────────────────────
        self.sku_centrality = compute_sparse_pagerank(src, dst, weights)
    
        self.logger.info(
            "Built sparse SKU centrality for %d SKUs", len(self.sku_centrality)
        )
        # --- Main Generation Method ---
    def generate_representations(
        self, 
        client_ids: list, 
        max_length: int = MAX_RICH_TOKENS
    ) -> dict:
        """Return {client_id: json‑string} with a rich‑text block and *all* features ‑
        without duplicate lines.  Uses _build_rich_text() for final formatting so that
        sections follow SECTIONS_ORDER and long lists are automatically trimmed.
        """
        
        # ✅ FIX : Normaliser client_ids une fois pour toutes
        if isinstance(client_ids, int):
            client_ids = [client_ids]
        
        # ✅ FIX : Vérification UNIQUE et SIMPLE
        if self.events_df is None and self.lazy_all is None:
            self.logger.info("Loading data for clients...")
            self.load_data(use_cache=True, relevant_client_ids=client_ids)
            
            # Vérifier qu'on a maintenant des données
            if self.events_df is None and self.lazy_all is None:
                raise RuntimeError("No data available after load_data()")
        
        # ✅ Si on a events_df mais pas lazy_all, c'est OK (mode cache)
        if self.lazy_all is None and self.events_df is not None:
            self.logger.debug("Using events_df (cache mode)")
        
        # ========================================================
        # GARDEZ TOUT LE CODE ORIGINAL À PARTIR D'ICI !
        # ========================================================
        now = datetime.now()
        extractors = self.get_feature_extractors()
        reps: dict[int, str] = {}
        
        # LE CODE CONTINUE ICI - NE PAS SUPPRIMER !
        for cid in client_ids:
            events = self.get_client_events(cid)
            
            if events.height == 0:
                reps[cid] = json.dumps(
                    {"profile": {"client_id": cid, "error": "No activity"}},
                    ensure_ascii=False,
                )
                continue
    
            # ==============================================================
            # 1)  OVERVIEW -------------------------------------------------
            # ==============================================================
            seg          = self.get_client_segment(cid)
            user_type    = "buyer" if seg.get("buyers") else "browser"
            overview_sec = [f"[CLIENT_{cid}]", f"User Type: {user_type}"]
            seg_list     = sorted(
                s for s, active in seg.items() if active and s not in {"buyers", "non_buyers"}
            )
            if seg_list:
                overview_sec.append("Segments: " + ", ".join(seg_list))

            # ==============================================================
            # 2)  FEATURE EXTRACTION  -> section_map ----------------------
            # ==============================================================
            section_map: dict[str, list[str]] = defaultdict(list)
            section_map["OVERVIEW"].extend(overview_sec)
            features_json: list[dict[str, str]] = []
            features_list = []
            # where to dump each extractor's lines → logical section name
            ex_to_sec = {
                "temporal":         "TEMPORAL",
                "sequence":         "SEQUENCE",
                "social":           "SOCIAL",
                "price":            "PRICE",
                "intent":           "OVERVIEW",  # can be changed to its own section
                "graph":            "CUSTOM",
                "name_embedding": "CUSTOM",
                "custom_behavior":  "CUSTOM",
                "top_sku":       "SKU_PROPENSITY",
                "top_category":   "CAT_PROPENSITY",
            }

            for ex_name, extractor in extractors.items():
                tgt_sec = ex_to_sec.get(ex_name, "CUSTOM")
                try:
                    feats = extractor.extract_features(cid, events, now)
                except Exception as err:
                    feats = [f"{ex_name}-error"]

                # ↓↓↓  répétition implicite uniquement pour top_sku
                repeat = IMPLICIT_WEIGHT_REPEAT if ex_name in ("top_sku", "top_category") else 1
                for ft in feats:
                    for _ in range(repeat):
                        section_map[tgt_sec].append(ft)
                    features_json.append({"type": ex_name, "value": ft})
            
            for t in extra_tags:
                features_list.append({"type": "extra", "value": t})
                section_map["CUSTOM"].append(t)
                
            for tag in compact_tags:
                features_list.append({"type": "compact", "value": tag})
                section_map["CUSTOM"].append(tag)          # ou une section dédiée "STATS"
            # ==============================================================
            # 3)  BEHAVIORAL METRICS --------------------------------------
            # ==============================================================

            co_pairs   = top_co_pairs(events)
            cat_pairs  = top_co_categories(events)
            cart_conv  = cart_conversion_stats(events)

            # --- GLOBAL cross-user pairs – filtrés sur l'historique utilisateur ----
            if 'global_sku_pairs' in self.global_stats:
                user_skus = events.filter(pl.col('sku').is_not_null())['sku'].unique().to_list()
                global_pairs = self.global_stats['global_sku_pairs']
                
                # Gérer le cas où c'est un dict ou une liste
                if isinstance(global_pairs, dict):
                    # Si c'est un dict, convertir en liste de tuples
                    pairs_list = []
                    for key, count in global_pairs.items():
                        if isinstance(key, str) and ',' in key:
                            # Format "(i, j)": count
                            try:
                                i, j = eval(key)  # Attention: eval est dangereux, mais ici on contrôle le format
                                pairs_list.append(((i, j), count))
                            except:
                                pass
                    global_pairs = pairs_list
                
                # Maintenant traiter comme une liste
                for pair_data in global_pairs[:20]:
                    if isinstance(pair_data, (list, tuple)) and len(pair_data) == 2:
                        (i, j), cnt = pair_data
                        if i in user_skus and j in user_skus:
                            section_map['CUSTOM'].append(f"GLOBAL_CO_PAIR:SKU_{i}~SKU_{j} ({cnt}x)")
            
            if 'global_cat_pairs' in self.global_stats and 'category_id' in events.columns:
                user_cats = events['category_id'].unique().drop_nulls().to_list()
                global_cat_pairs = self.global_stats['global_cat_pairs']
                
                # Même traitement pour les cat_pairs
                if isinstance(global_cat_pairs, dict):
                    pairs_list = []
                    for key, count in global_cat_pairs.items():
                        if isinstance(key, str) and ',' in key:
                            try:
                                c1, c2 = eval(key)
                                pairs_list.append(((c1, c2), count))
                            except:
                                pass
                    global_cat_pairs = pairs_list
                
                for pair_data in global_cat_pairs[:20]:
                    if isinstance(pair_data, (list, tuple)) and len(pair_data) == 2:
                        (c1, c2), cnt = pair_data
                        if c1 in user_cats and c2 in user_cats:
                            section_map['CUSTOM'].append(f"GLOBAL_CAT_PAIR:CAT_{c1}~CAT_{c2} ({cnt}x)")

                
                
            # ==============================================================
            # 4)  HISTORY SNAPSHOTS ---------------------------------------
            # ==============================================================
            recent_cut  = now - timedelta(days=14)
            medium_cut  = now - timedelta(days=90)

            recent_txt = self._generate_detailed_events_text(
                events.filter(pl.col("timestamp") >= recent_cut)
            )
            medium_txt = self._generate_summarized_events_text(
                events.filter((pl.col("timestamp") >= medium_cut) & (pl.col("timestamp") < recent_cut))
            )
            hist_txt   = self._generate_aggregated_events_text(
                events.filter(pl.col("timestamp") < medium_cut)
            )

            if recent_txt != "No recent activity.":
                section_map["TARGET_WINDOW_14D"].append(recent_txt)
            if medium_txt != "No medium-term activity.":
                section_map["SEQUENCE"].append(medium_txt)
            if hist_txt != "No historical activity.":
                section_map["CUSTOM"].append(hist_txt)

            # ==============================================================
            # 5)  RAW SEQUENCE  -------------------------------------------
            # ==============================================================
            raw_seq = self._generate_raw_sequence(
                events, max_events=RAW_SEQUENCE_LAST_EVENTS
            )
            section_map["CUSTOM"].append("## RAW_SEQUENCE ##")
            section_map["CUSTOM"].append("```")        # ← ouverture du bloc code
            section_map["CUSTOM"].append(raw_seq)      # ← contenu brut
            section_map["CUSTOM"].append("```")        # ← fermeture du bloc code
            section_map["CUSTOM"].append("RAW_SEQUENCE (derniers 50 événements)…")
            section_map["CUSTOM"].append("</s>".join(raw_seq.split("</s>")[-50:]))

            section_map["CUSTOM"] = list(dict.fromkeys(section_map["CUSTOM"]))

            # ==============================================================
            # 6)  BUILD RICH TEXT (deduplicated / token‑limited) -----------
            # ==============================================================
            try:
                rich_text = _build_rich_text(
                    section_map=section_map,
                    max_tokens=max_length,
                    implicit_repeat=IMPLICIT_WEIGHT_REPEAT,
                    top_per_section=TOP_FEATURES_PER_SECTION,
                    shuffle_seed=cid,
                )
            except Exception as exc:
                self.logger.error(f"_build_rich_text failed for {cid}: {exc}")
                # fallback – very plain
                rich_text = "\n\n".join(
                    f"## {sec} ##\n" + "\n".join(lines)
                    for sec, lines in section_map.items()
                )

            # ==============================================================
            # 7)  JSON PROFILE (for downstream) ---------------------------
            # ==============================================================
            profile = {
                "client_id": cid,
                "overview": {"user_type": user_type, "segments": seg_list},
                "features": features_json,
                "behavioral_metrics": {
                    "co_pairs": co_pairs,
                    "category_pairs": cat_pairs,
                    "cart_conversion": cart_conv,
                },
                "recent_activity": recent_txt,
                "medium_term_summary": medium_txt,
                "historical_aggregates": hist_txt,
                "raw_sequence": raw_seq,
                "insights": {},
                "recommendations": [],
            }

            if any(
                ft["type"] == "temporal" and "Inactive" in ft["value"]
                for ft in features_json
            ):
                profile["insights"]["inactivity"] = "High inactivity"
                profile["recommendations"].append("Send re‑engagement email")

            if raw_seq and "[END]" in rich_text:
                rich_text = rich_text.replace("[END]", f"\n## RAW_SEQUENCE ##\n{raw_seq}\n[END]")
            
            reps[cid] = json.dumps(
                {"profile": profile, "rich_text": rich_text}, ensure_ascii=False
            )

        return reps




class ChurnPropensityFeatureExtractor(FeatureExtractorBase):
    """Extract features specifically for churn and propensity prediction tasks"""
    
    def extract_features(self, client_id: int, events: pl.DataFrame, now: datetime) -> List[str]:
        features = []
        
        # Pass client_id to all methods for better error tracking
        self._extract_churn_signals(client_id, events, features, now)
        self._extract_category_propensity(client_id, events, features, now)
        self._extract_sku_propensity(client_id, events, features, now)
        
        return features
    
    def _extract_churn_signals(self, client_id: int, events: pl.DataFrame, features: List[str], now: datetime) -> None:
        """Extract signals relevant to churn prediction"""
        try:
            # Get purchase history with proper categorical comparison
            purchases = events.filter(
                pl.col('event_type') == pl.lit('product_buy', dtype=pl.Categorical)
            )
            
            if purchases.height == 0:
                features.append("CHURN_STATUS:NO_PURCHASE")
                return
            
            # Get timestamps as Series
            purchase_timestamps = purchases['timestamp']
            last_purchase = purchase_timestamps.max()
            first_purchase = purchase_timestamps.min()
            
            if last_purchase is None:
                features.append("CHURN_STATUS:NO_VALID_PURCHASE_TIME")
                return
                
            days_since_last_purchase = (now - last_purchase).days
            
            # Official churn definition: 14+ days after purchase
            if days_since_last_purchase >= 14:
                features.append("CHURN_RISK:HIGH")
            else:
                features.append("CHURN_RISK:LOW")
            
            features.append(f"PURCHASE_RECENCY:{days_since_last_purchase}d")
            
            # Purchase frequency pattern
            if purchases.height > 1:
                purchase_intervals = []
                # Get sorted timestamps, filter nulls
                sorted_purchases = purchases.sort('timestamp')
                purchase_times = sorted_purchases['timestamp'].drop_nulls().to_list()
                
                for i in range(1, len(purchase_times)):
                    interval_days = (purchase_times[i] - purchase_times[i-1]).days
                    purchase_intervals.append(interval_days)
                
                if purchase_intervals:
                    avg_interval = np.mean(purchase_intervals)
                    features.append(f"AVG_PURCHASE_INTERVAL:{avg_interval:.1f}d")
                    
                    if days_since_last_purchase > avg_interval * 2:
                        features.append("PURCHASE_PATTERN:UNUSUAL_GAP")
            
            # Activity after last purchase
            post_purchase_activity = events.filter(pl.col('timestamp') > last_purchase)
            
            if post_purchase_activity.height > 0:
                features.append(f"POST_PURCHASE_EVENTS:{post_purchase_activity.height}")
                
                # Type of post-purchase activity with safe access
                post_types = post_purchase_activity.group_by('event_type').agg(
                    pl.count().alias('count')
                )
                
                # Create a dict for easier access
                post_dict = {row['event_type']: row['count'] for row in post_types.iter_rows(named=True)}
                
                if post_dict.get('page_visit', 0) > 0:
                    features.append("POST_PURCHASE:BROWSING")
                if post_dict.get('add_to_cart', 0) > 0:
                    features.append("POST_PURCHASE:CART_ACTIVITY")
            else:
                features.append("POST_PURCHASE:NO_ACTIVITY")
            
            # Lifetime value indicators - FIXED: Keep as DataFrame or use .len() for Series
            if 'price_bucket' in purchases.columns:
                # Option 1: Keep as DataFrame
                price_df = purchases.filter(pl.col('price_bucket').is_not_null()).select('price_bucket')
                if price_df.height > 0:
                    total_purchase_value = price_df['price_bucket'].sum()
                    features.append(f"LTV_INDICATOR:{total_purchase_value}")
                else:
                    features.append("LTV_INDICATOR:0")
            else:
                features.append("LTV_INDICATOR:NO_PRICE_DATA")
            
        except Exception as e:
            self.logger.error(f"Error in churn signals for client {client_id}: {e}", exc_info=True)
            features.append("CHURN_SIGNALS:ERROR")
    
    def _extract_category_propensity(self, client_id: int, events: pl.DataFrame, features: List[str], now: datetime) -> None:
        """Extract signals for category propensity prediction"""
        try:
            if 'category_id' not in events.columns:
                features.append("CAT_PROPENSITY:NO_CATEGORY_DATA")
                return
            
            # Category interaction patterns
            cat_events = events.filter(pl.col('category_id').is_not_null())
            if cat_events.height == 0:
                features.append("CAT_PROPENSITY:NO_CATEGORY_EVENTS")
                return
            
            # Recency-weighted category interest
            # Use proper datetime handling
            cat_events_with_weight = cat_events.with_columns(
                (
                    ((pl.lit(now) - pl.col("timestamp")).dt.total_seconds() / 86400 + 1.0)
                    .pow(-0.5)
                    .alias("recency_weight")
                )
            )
            
            # Top categories by weighted interaction
            cat_scores = (
                cat_events_with_weight
                .group_by('category_id')
                .agg([
                    pl.sum('recency_weight').alias('weighted_score'),
                    pl.count().alias('interaction_count'),
                    pl.max('timestamp').alias('last_interaction')
                ])
                .sort('weighted_score', descending=True)
            )
            
            # Add features for top categories
            top_cats = cat_scores.head(5)
            for row in top_cats.to_dicts():
                cat_id = row['category_id']
                score = row['weighted_score']
                features.append(f"CAT_PROPENSITY:CAT_{cat_id}(score={score:.2f})")
            
            # Category diversity for propensity
            unique_cats = cat_events['category_id'].n_unique()
            features.append(f"CAT_EXPLORATION_BREADTH:{unique_cats}")
            
            # Purchase concentration
            purchase_cats = events.filter(
                (pl.col('event_type') == pl.lit('product_buy', dtype=pl.Categorical)) & 
                pl.col('category_id').is_not_null()
            )
            
            if purchase_cats.height > 0:
                purchase_cat_dist = purchase_cats.group_by('category_id').agg(
                    pl.count().alias('count')
                ).sort('count', descending=True)
                
                if purchase_cat_dist.height > 0:
                    top_row = purchase_cat_dist.row(0, named=True)
                    top_purchase_cat = top_row['category_id']
                    purchase_concentration = top_row['count'] / purchase_cats.height
                    features.append(f"PURCHASE_CAT_FOCUS:CAT_{top_purchase_cat}({purchase_concentration:.2f})")
            
        except Exception as e:
            self.logger.error(f"Error in category propensity for client {client_id}: {e}", exc_info=True)
            features.append("CAT_PROPENSITY:ERROR")
    
    def _extract_sku_propensity(self, client_id: int, events: pl.DataFrame, features: List[str], now: datetime) -> None:
        """Extract signals for SKU propensity prediction"""
        try:
            if 'sku' not in events.columns:
                features.append("SKU_PROPENSITY:NO_SKU_DATA")
                return
            
            sku_events = events.filter(pl.col('sku').is_not_null())
            if sku_events.height == 0:
                features.append("SKU_PROPENSITY:NO_SKU_EVENTS")
                return
            
            # SKU interaction patterns with recency weighting
            sku_events_weighted = sku_events.with_columns(
                (
                    ((pl.lit(now) - pl.col("timestamp")).dt.total_seconds() / 86400 + 1.0)
                    .pow(-0.5)
                    .alias("recency_weight")
                )
            )
            
            # Event type weights - using when/then for categorical column
            sku_events_weighted = sku_events_weighted.with_columns(
                pl.when(pl.col('event_type') == pl.lit('page_visit', dtype=pl.Categorical)).then(1.0)
                .when(pl.col('event_type') == pl.lit('add_to_cart', dtype=pl.Categorical)).then(3.0)
                .when(pl.col('event_type') == pl.lit('product_buy', dtype=pl.Categorical)).then(5.0)
                .when(pl.col('event_type') == pl.lit('remove_from_cart', dtype=pl.Categorical)).then(-2.0)
                .otherwise(1.0)
                .alias('event_weight')
            )
            
            # Combined score
            sku_events_weighted = sku_events_weighted.with_columns(
                (pl.col('recency_weight') * pl.col('event_weight')).alias('combined_score')
            )
            
            # Top SKUs by combined score
            sku_scores = (
                sku_events_weighted
                .group_by('sku')
                .agg([
                    pl.sum('combined_score').alias('total_score'),
                    pl.count().alias('interaction_count'),
                    pl.max('timestamp').alias('last_interaction')
                ])
                .sort('total_score', descending=True)
            )
            
            # Features for top SKUs
            top_skus = sku_scores.head(10)
            for i, row in enumerate(top_skus.to_dicts()):
                sku = row['sku']
                score = row['total_score']
                features.append(f"SKU_PROPENSITY_TOP{i+1}:SKU_{sku}(score={score:.2f})")
            
            # Re-interaction patterns
            sku_repeat_purchases = (
                events.filter(pl.col('event_type') == pl.lit('product_buy', dtype=pl.Categorical))
                .filter(pl.col('sku').is_not_null())
                .group_by('sku')
                .agg(pl.count().alias('purchase_count'))
                .filter(pl.col('purchase_count') > 1)
            )
            
            if sku_repeat_purchases.height > 0:
                features.append(f"REPEAT_PURCHASE_SKUS:{sku_repeat_purchases.height}")
                top_repeat = sku_repeat_purchases.sort('purchase_count', descending=True).head(1)
                if top_repeat.height > 0:
                    top_row = top_repeat.row(0, named=True)
                    features.append(f"TOP_REPEAT_SKU:SKU_{top_row['sku']}({top_row['purchase_count']}x)")
            
            # Brand loyalty (if available through properties)
            if hasattr(self.parent, 'sku_properties_dict') and self.parent.sku_properties_dict:
                sku_list = sku_events['sku'].unique().to_list()
                brands = []
                for sku in sku_list:
                    if sku is not None:
                        props = self.parent.sku_properties_dict.get(int(sku), {})
                        if 'brand' in props:
                            brands.append(props['brand'])
                
                if brands:
                    brand_counts = Counter(brands)
                    if brand_counts:
                        top_brand = brand_counts.most_common(1)[0]
                        features.append(f"BRAND_AFFINITY:{top_brand[0]}({top_brand[1]})")
            
        except Exception as e:
            self.logger.error(f"Error in SKU propensity for client {client_id}: {e}", exc_info=True)
            features.append("SKU_PROPENSITY:ERROR")

            
# --- Wrapper Class ---
class TextRepresentationGenerator:
    """Pipeline = (load data ➜ rich_text ➜ portrait LLM ➜ save)."""

    def __init__(self, use_polars: bool = True) -> None:
        self.data_dir:    Optional[str]               = None
        self.cache_dir:   Optional[str]               = None
        self.debug_mode:  bool                        = False
        self.advanced_generator: Optional[AdvancedUBMGenerator] = None

    # ------------------------------------------------------------------ #
    #                         SET-UP & HELPERS                           #
    # ------------------------------------------------------------------ #
    def prepare_data(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        debug_mode: bool = False,
    ) -> "TextRepresentationGenerator":
        self.data_dir   = data_dir
        self.cache_dir  = cache_dir
        self.debug_mode = debug_mode
        return self

    # ------------------------------------------------------------------ #
    #                             MAIN CALL                              #
    # ------------------------------------------------------------------ #
    def generate_text_representations(
        self,
        client_ids: List[int],
        output_file: Optional[str] = None,
        max_length: int = 3_500,
    ) -> Dict[int, str]:
        """Return {client_id: rich_text + PORTRAIT}"""

        if not self.data_dir:
            raise ValueError("Call prepare_data() first.")

        # 1) Load / cache data
        if self.advanced_generator is None:
            logger.info("Instantiating AdvancedUBMGenerator …")
            self.advanced_generator = AdvancedUBMGenerator(
                self.data_dir,
                cache_dir=self.cache_dir,
                debug_mode=self.debug_mode,
            )
            relevant_filter = client_ids if self.debug_mode else None
            self.advanced_generator.load_data(
                use_cache=True, relevant_client_ids=relevant_filter
            )

        # 2) Generate base rich-texts
        base_texts: Dict[int, str] = self.advanced_generator.generate_representations(
            client_ids, max_length=max_length
        )

        # 3) Strip out RAW_SEQUENCE for the LLM
        def _strip_raw(txt: str) -> str:
            return re.split(r'(?i)RAW_SEQUENCE:', txt, maxsplit=1)[0].rstrip()
        
        stripped_texts: Dict[int, str] = {}
        for cid, rt in base_texts.items():
            payload = json.loads(rt)
            summary = payload.get("rich_text", "")
            stripped = _strip_raw(summary)
            stripped_texts[cid] = stripped



        # 4) Ask the LLM for plain-text bullet portraits
        portraits = generate_portraits(stripped_texts)

        # 5) Merge summary, portrait & RAW_SEQUENCE into one plain-text blob
        enriched_texts: Dict[int, str] = {}
        for cid, full_txt in base_texts.items():
            rep      = json.loads(full_txt)
            summary  = rep.get("rich_text", "")
            raw_seq  = rep.get("profile", {}).get("raw_sequence", "")

            # get the bullet list string from our LLM
            portrait_str   = portraits.get(cid, "")
            portrait_block = "\n## PORTRAIT ##\n" + portrait_str

            raw_block = ("\n## RAW_SEQUENCE ##\n" + raw_seq) if raw_seq else ""
            if summary.strip().endswith("```"):
                summary += "\n```"
            final_rich = "\n".join([
                summary,
                "## PORTRAIT ##",
                portrait_str.strip(),
                "## RAW_SEQUENCE ##",
                "\n".join(raw_seq.split("</s>")[-50:]) + "\n…",
            ])

            enriched_texts[cid] = final_rich

        # 6) Optionally write out…
        if output_file:
            self._write_to_file(enriched_texts, output_file)

        return enriched_texts

    # ------------------------------------------------------------------ #
    #                            I/O helper                              #
    # ------------------------------------------------------------------ #
    def _write_to_file(self, texts: Dict[int, str], path: str) -> None:
        logger.info(f"Saving {len(texts)} representations ➜ {path}")
        is_gcs = path.startswith("gs://")

        if is_gcs:
            try:
                import gcsfs
                fs = gcsfs.GCSFileSystem()
                f  = fs.open(path, "wt", encoding="utf-8")
            except ImportError:
                logger.error("gcsfs missing – cannot write to GCS.")
                return
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            f = open(path, "w", encoding="utf-8")

        with f:
            for cid, txt in texts.items():
                json.dump({"client_id": cid, "rich_text": txt}, f, ensure_ascii=False)
                f.write("\n")

        logger.info("✅  Representations saved.")









class CustomBehaviorFeatureExtractor(FeatureExtractorBase):
    """20+ nouvelles features issues du clustering SKU, Node2Vec, RFM quantiles…"""
    def extract_features(self, client_id: int, events: pl.DataFrame, now: datetime) -> List[str]:
        f = []
        # — SKU cluster shares
        skus = events.filter(pl.col('sku').is_not_null())['sku'].to_list()
        if hasattr(self.parent, 'sku_cluster_map'):
            cnt = Counter(self.parent.sku_cluster_map.get(int(s), -1) for s in skus)
            total = sum(cnt.values()) or 1
            top = cnt.most_common(3)
            f.append("SKU_CLUSTER_SHARES:")
            for cid, c in top:
                f.append(f"  - C{cid}: {c/total:.0%}")
        # — URL embedding similarity mean
        ulist = events.filter(pl.col('url').is_not_null())['url'].to_list()
        sims = []
        for u in ulist:
            vec = self.parent.url_embed.get(f"U_{u}")
            if vec is not None:
                sims.append(np.dot(vec, self.parent.url_centroid))
        if sims:
            f.append(f"URL_EMB_SIM: {float(np.mean(sims)):.3f}")
        # — temporal cyclic features
        hrs = events['timestamp'].dt.hour().to_numpy()
        days = events['timestamp'].dt.weekday().to_numpy()
        cyc = np.vstack([
            np.sin(2*np.pi*hrs/24), np.cos(2*np.pi*hrs/24),
            np.sin(2*np.pi*days/7), np.cos(2*np.pi*days/7)
        ]).T
        if len(cyc)>0:
            mean_cyc = np.round(cyc.mean(axis=0),2).tolist()
            f.append(f"TIME_CYCLIC_MEAN: {mean_cyc}")
        # — RFM quantile for recency
        recs = np.array(self.parent.global_stats.get('rfm_recencies',[]))
        if recs.size>0:
            last_buy = events.filter(pl.col('event_type')=='product_buy')['timestamp'].max()
            if last_buy:
                r = (now - last_buy).days
                q = float((recs <= r).sum()/len(recs))
                f.append(f"RFM_REC_Q: {q:.2f}")
        return f
