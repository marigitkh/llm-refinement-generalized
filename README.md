# Do LLMs Know What/Where and Why They Lack?


The project **evaluates** the self-refinement abilities of large language models (LLMs) by applying 3 steps of self-refinment method (with external data used during step 2) on arithmetic benchmark and then evaluating the performance of the LLMs after hint injection (step 3):


## Project Structure


  - data/ — input files for all supported datasets  
    - `asdiv.jsonl` — ASDiv arithmetic reasoning samples  
    - `gsm8k.jsonl` — GSM8K math word problems  
    - `aqua.jsonl` — AQUA-RAT quantitative aptitude questions  
    - `sports.jsonl` — binary plausibility judgments  
    - `ar_lsat.jsonl` — AR-LSAT logical reasoning prompts  

  - prompts/ — dataset-specific prompt templates  
    - `asdiv_prompt.txt` — ASDiv question prompt  
    - `gsm8k_prompt.txt` — GSM8K question prompt  
    - `aqua_prompt.txt` — AQUA-RAT question prompt  
    - `sports_prompt.txt` — sports plausibility prompt  
    - `ar_lsat_prompt.txt` — AR-LSAT logic prompt  
    - `hint_prompt.txt` — shared template for hint generation  

  - results/ — model outputs and evaluation results  
    - reasoning/ — outputs for reasoning tasks (asdiv, gsm8k, aqua)  
    - non-reasoning/ — outputs for binary/logical tasks (sports, ar_lsat)  
    - `statistics.txt` — aggregated accuracy metrics  

  - src/ — source code modules  
    - datasets/ — preprocessing & evaluation logic per dataset  
      - `asdiv.py`, `gsm8k.py`, `aqua.py`, `sports.py`, `ar_lsat.py`  
    - `utils.py` — prompt builders, answer parsing, helpers  
    - `inference.py` — core inference and hint generation logic  
    - `run.py` — CLI entry point for running the pipeline  
    - `analysis.py` — evaluation and accuracy analysis  

  - `run_all_models.sh` — script to evaluate all models across datasets  

  - `README.md` — project overview and instructions  

  - `requirements.txt` — Python dependencies



## How to Use

### 1. Install dependencies:

```
pip install -r requirements.txt
```
<br>

### 2. Run the full pipeline

This will generate initial answers, hints, and post-hint answers:

```
python src/run.py \
  --model_path <HF-model-or-local-dir> \
  --input_path data/asdiv.jsonl \
  --output_dir results/gemma-2-2b-it \
  [--max_samples N]
```

- *model_path*: Hugging Face checkpoint (e.g. `google/gemma-2-2b-it`) or local directory  
- *input_path*: JSONL file with each line `{ "question": "...", "answer": "..." }`  
- *output_dir*: Directory for three output files  
- *max_samples*: limit number of examples processed  (optional)

After running, you’ll find:

- `initial_inference.jsonl`  
- `hints.jsonl`  
- `post_hint_inference.jsonl`  
<br>

### 3. Analyze accuracy improvements

Summarize initial vs. post-hint accuracy across all model/dataset folders:

```
python src/analysis.py \
  --parent_dir results \
  --output_file results/statistics.txt
```

After running, you’ll find `statistics.txt` containing calculated evaluation metrics.

### 4. Run all experiments in batch (optional)

To evaluate multiple models across all datasets in one go, run:

```
bash run_all_models.sh
```

You can edit the `run_all_models.sh` file to include any set of model–dataset combinations.
