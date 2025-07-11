import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pathlib import Path
import zipfile
import io
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    """
    Main function to execute the ensembling logic.
    """
    print("--- Starting Model Ensembling ---")

    # [cite_start]--- Cell 1: Config --- [cite: 7]
    EMB_PATHS = {
        'stella': 'embeddings/stella',
        'gemma1b': 'embeddings/gemma1b',
        'qwen': 'embeddings/qwen3-8b'  # Renaming for consistency with script
        # ,'gemma12b': 'embeddings/gemma12b'
    }
    MODELS = ['stella', 'gemma1b', 'qwen']#, 'gemma12b']
    KNOWN_SCORES = {
        "stella": {"churn":0.7029,"propensity_category":0.7639,"propensity_sku":0.7362, "hidden1":0.7123,"hidden2":0.7215,"hidden3":0.7751},
        "gemma1b": {"churn":0.7098,"propensity_category":0.7685,"propensity_sku":0.7645, "hidden1":0.7166,"hidden2":0.7689,"hidden3":0.7762},
        "qwen": {"churn":0.7098,"propensity_category":0.7640,"propensity_sku":0.7553, "hidden1":0.7107,"hidden2":0.7242,"hidden3":0.7700}
        # "gemma12b": {"churn":0.7112,"propensity_category":0.7601,"propensity_sku":0.7425, "hidden1":0.7256,"hidden2":0.7279,"hidden3":0.7700},
    }
    S = np.vstack([[KNOWN_SCORES[m][t] for t in ['churn','propensity_category','propensity_sku','hidden1','hidden2','hidden3']] for m in MODELS])

    # [cite_start]--- Cell 2: Optimization --- [cite: 8]
    lambda_reg = 0.2
    eps = 0.01

    def objective_function(w):
        mean_score = np.mean(w.dot(S))
        return -mean_score + lambda_reg * np.sum(w**2)

    constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}, {'type': 'ineq', 'fun': lambda w: w - eps}]
    w0 = np.ones(len(MODELS)) / len(MODELS)
    result = minimize(objective_function, w0, constraints=constraints, method='SLSQP')
    w_final = result.x if result.success else w0

    # The original notebook used `display()`. In a script, we use `print()`.
    print("\n--- Final Weights ---")
    final_weights_df = pd.DataFrame([w_final], columns=MODELS, index=['weight']).T
    print(final_weights_df)
    print("---------------------\n")

    # [cite_start]--- Cell 3: Generation & Submission --- [cite: 9, 10]
    try:
        print("Loading and aligning embeddings...")
        # Rename the key in EMB_PATHS to match MODELS list
        EMB_PATHS_SCRIPT = {k.replace('qwen3-8b', 'qwen'): v for k, v in EMB_PATHS.items()}
        
        ids_dict = {m: np.load(Path(EMB_PATHS_SCRIPT[m]) / 'client_ids.npy') for m in MODELS}
        emb_dict = {m: np.load(Path(EMB_PATHS_SCRIPT[m]) / 'embeddings.npy').astype(np.float32) for m in MODELS}

        common_ids = set(ids_dict[MODELS[0]])
        for m in MODELS[1:]:
            common_ids &= set(ids_dict[m])
        common_ids_list = np.array(sorted(list(common_ids)))

        idx_maps = {m: {cid: i for i, cid in enumerate(ids_dict[m])} for m in MODELS}
        for m in MODELS:
            indices = [idx_maps[m][cid] for cid in common_ids_list]
            emb_dict[m] = emb_dict[m][indices]
        print(f"✅ Embeddings aligned for {len(common_ids_list)} clients.")

        def build_ensemble(weights):
            emb_n = {m: e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8) for m, e in emb_dict.items()}
            E = sum(weights[i] * emb_n[m] for i, m in enumerate(MODELS))
            return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
        
        ensemble_final = build_ensemble(w_final)
        
        np.save('embeddings/ensemble/client_ids.npy', common_ids_list)
        np.save('embeddings/ensemble/embeddings.npy', ensemble_final)
        
        output_zip_path = Path('embeddings/ensemble/submission.zip') # Simplified name
        print(f"Saving submission file to: {output_zip_path}")
        
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            with io.BytesIO() as emb_buffer:
                np.save(emb_buffer, ensemble_final.astype(np.float16))
                emb_buffer.seek(0)
                zf.writestr('embeddings.npy', emb_buffer.read())
            with io.BytesIO() as ids_buffer:
                np.save(ids_buffer, common_ids_list)
                ids_buffer.seek(0)
                zf.writestr('client_ids.npy', ids_buffer.read())
        
        print(f"✅ Submission file created successfully: {output_zip_path}")

    except Exception as e:
        # [cite_start]Replaced `display(Markdown(...))` with logging.error [cite: 11]
        logging.error(f"❌ ERROR during submission file generation: {e}", exc_info=True)


if __name__ == "__main__":
    main()