# TrioLearn

## Overview
TrioLearn is a tri-modal educational recommendersytem build using NLP and ML techniques: for each user it suggests a Course, a Book, and a YouTube Video associated knowledge state. It combines:
- Knowledge tracing (modeling learner mastery over time)
- Semantic embeddings (Word2Vec / BERT) and topic modelling
- Heterogeneous Graph Neural Network to fuse learners, courses, books, videos
- Evaluation using Precision@K, Novelty, Diversity, Coverage, and statistical tests

## Setup

1. **Clone or open the project folder** in VSCode (or your environment).
2. **Environment & dependencies**:
   - If using Conda:  
     ```bash
     conda env create -f environment.yml
     conda activate triolearn
     ```
   - Otherwise:  
     ```bash
     pip install -r requirements.txt
     ```
3. **API keys** (for later phases if you fetch YouTube/Google Books):  
   - Copy `.env.example` to `.env` and fill in keys, e.g.:  
     ```
     YOUTUBE_API_KEY=your_key_here
     GOOGLE_BOOKS_API_KEY=your_key_here
     ```
4. **Data placement**:
   - Place your Kaggle CSV files in `data/raw/`.
   - Later, external fetched JSONs or transcripts go in `data/external/`.
5. **Run or open notebooks**:
   - In VSCode, open a terminal (inside the environment) and launch Jupyter (or connect via Docker if preferred).
   - Open `notebooks/01_data_exploration.ipynb` to inspect your data.

## Project Structure
- `data/`: raw, interim, processed datasets
- `notebooks/`: exploration & prototyping
- `src/`: reusable modules for loading, preprocessing, embeddings, model definitions, evaluation
- `scripts/`: CLI wrappers to run preprocessing, train models, evaluate, etc.
- `outputs/`: saved model files, figures, logs, reports
- `docker/`: Dockerfile and optional compose for reproducible environment
- `.vscode/`: VSCode settings

## Usage Examples

- **Data exploration**:  
  Open `notebooks/01_data_exploration.ipynb`, run cells to load CSVs from `data/raw/`, inspect columns, missing values, sample rows.
- **Preprocessing**:  
  Once you finalize exploration, copy cleaning logic into `src/data/preprocess.py`. You can test locally, then run `python scripts/preprocess_all.py --input data/raw --output data/processed`.
- **Embeddings**:  
  In `notebooks/03_embedding_training.ipynb`, import functions from `src/features/embeddings.py` to compute Word2Vec or BERT embeddings on text fields.
- **Evaluation**:  
  Use `src/evaluation/metrics.py` to compute Precision@K, Novelty, Diversity, Coverage on held-out data. Wrap in `notebooks/07_evaluation_metrics.ipynb` or in `scripts/evaluate.py`.
- **Recommendations**:  
  Prototype retrieval in `notebooks/06_prototype_recommendation.ipynb`, then move to `src/models/recommender.py` and make a CLI in `scripts/recommend.py`.

## Notes
- Keep secret keys in `.env` (gitignored).
- Do not commit large data: list instructions in README for how to download or place raw files.
- Use VSCode’s Python extension to select the right interpreter or point to Docker container interpreter.

