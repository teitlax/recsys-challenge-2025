# Makefile for orchestrating NDUM - RecSys 2025 project

.PHONY: all data features finetune extract ensemble clean
DEBUG_FLAG :=
ifeq ($(DEBUG),true)
	DEBUG_FLAG := --debug
endif

PYTHON := python3
DATA_DIR := ubc_data
MODELS_DIR := models
FEATURES_DIR := output_features
EMBEDDINGS_DIR := embeddings

# --- Main Targets ---
all: ensemble

# Step 1: Download and prepare data
data:
	@echo "--- 1. Downloading Data ---"
	@bash src/download_data.sh

# Step 2 : Feature Generation
features: data
	@echo "--- 2. Generating Features ---"  
	@mkdir -p $(FEATURES_DIR)/gemma1b
	@mkdir -p $(FEATURES_DIR)/gemma12b
	$(PYTHON) src/preprocessing_gemma1.py $(DEBUG_FLAG)
	$(PYTHON) src/preprocessing_gemma12.py $(DEBUG_FLAG)

# Step 3: Fine-tuning
finetune: features
	@echo "--- 3. Fine-tuning Models ---"
	@mkdir -p $(MODELS_DIR)/contrastive_1M
	accelerate launch src/train_gemma1.py --model_id unsloth/gemma-3-1b-it-unsloth-bnb-4bit --dataset_path $(FEATURES_DIR)/gemma1b/complete_dataset_1M.jsonl.zst --output_dir $(MODELS_DIR)/contrastive_1M --load_in_4bit   --lora_r 16   --lora_alpha 32   --projection_dim 2048   --temperature 0.07   --batch_size 24   --gradient_accumulation 4   --max_steps 10000   --max_length 2048  --learning_rate 2e-5    --seed 42   --logging_steps 25   --save_steps 100   --save_total_limit 5   --warmup_steps 500   --weight_decay 0.01 

# Step 4: Embedding Extraction
extract: finetune
	@echo "--- 4. Extracting Embeddings ---"
	@mkdir -p $(EMBEDDINGS_DIR)/qwen3-8b
	@mkdir -p $(EMBEDDINGS_DIR)/stella
	@mkdir -p $(EMBEDDINGS_DIR)/gemma1b
	python src/extract_embeddings_qwen.py --dataset_path $(FEATURES_DIR)/gemma1b/complete_texts_1M.jsonl.zst $(DEBUG_FLAG)
	python src/extract_embeddings_stella.py --dataset_path $(FEATURES_DIR)/gemma1b/complete_texts_1M.jsonl.zst $(DEBUG_FLAG)
	accelerate launch src/extract_embeddings_gemma1.py --model_id unsloth/gemma-3-1b-it-unsloth-bnb-4bit --projection_dim 2048 --temperature 0.07 --checkpoint_dir $(MODELS_DIR)/contrastive_1M/checkpoint-1100 --dataset_path $(FEATURES_DIR)/gemma1b/complete_dataset_1M.jsonl.zst --output_dir $(EMBEDDINGS_DIR)/gemma1b --max_length 2048 --batch_size 32 --load_in_4bit --create_submission $(DEBUG_FLAG)

# Step 5: Final Ensemble
ensemble: extract
	@echo "--- 5. Creating Final Ensemble ---"
	@mkdir -p $(EMBEDDINGS_DIR)/ensemble
	python src/ensemble.py

# --- Utility ---
clean:
	@echo "--- Cleaning generated directories ---"
	rm -rf $(FEATURES_DIR) $(EMBEDDINGS_DIR) $(DATA_DIR)
	rm -rf unsloth_compiled_cache/ __pycache__/ src/__pycache__/ src/ubm/__pycache__/
	@echo "✅ Clean complete."
    
