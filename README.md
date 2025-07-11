# Narrative-Driven User Modelling: RecSys 2025 Challenge Source Code
The complete code and notebooks used for the ACM Recommender Systems Challenge 2025 by our team "teitlax"

## Project Structure

The project follows this layout

    └── recsys-challenge-2025/
        ├── Dockerfile    
        ├── Makefile             
        ├── README.md               
        ├── requirements.txt    
        └── src/
            └── ubm/
                ├──__init__.py
                ├──portrait_generator.py
                ├──text_representation_v3.py
            ├── download_data.sh           
            ├── ensemble.py
            ├── extract_embeddings_gemma1.py
            ├── extract_embeddings_stella.py
            ├── extract_embeddings_qwen.py
            ├── preprocessing_gemma1.py
            ├── preprocessing_gemma12.py
            ├── train_gemma1.py
            ├── train_gemma12.py
        
## Setup and Installation
Follow these steps to set up your local environment.

### 1. Clone the Repository
```bash
git clone https://github.com/teitlax/recsys-challenge-2025.git
cd recsys-challenge-2025
```

### 3. Set Up a Docker Environment
```bash
docker build -t ndum-env .
```

### 4. Add Hugging Face Token to use Gemma models
```bash
export HF_TOKEN="yourToken"
```

### 5. Launch pipeline
```bash
docker run --rm -it --gpus all \
  --shm-size=2g \
  --env HF_TOKEN=${HF_TOKEN} \
  --env TORCH_COMPILE_DISABLE=1 \
  -v "$(pwd):/app" \
  ndum-env \
  make all DEBUG=false
```
Debug mode 

```bash
docker run --rm -it --gpus all \
  --shm-size=2g \
  --env HF_TOKEN=${HF_TOKEN} \
  --env TORCH_COMPILE_DISABLE=1 \
  -v "$(pwd):/app" \
  ndum-env \
  make all DEBUG=true
```

It will create 

    └── recsys-challenge-2025/
        ├── ... 
        ├── embeddings/
        ├── models/
        ├── output_features/ 
        ├── ubc_data/      
        ├── unsloth_compiled_cache/ 
