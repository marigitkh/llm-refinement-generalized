#!/bin/bash

MODELS_NON_REASONING=(
  "google/gemma-2-2b-it"
  "microsoft/Phi-4-mini-instruct"
  "Qwen/Qwen2.5-Math-1.5B"
  "LLama-3.1-8b-it"
)

MODELS_REASONING=(
  "microsoft/Phi-4-mini-reasoning"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)

DATASETS=(
  "asdiv"
  "gsm8k"
  "aqua"
  "sports"
  "ar_lsat"
)

INPUT_DIR="data"
OUTPUT_DIR="results"
MAX_SAMPLES=10

# Run for non-reasoning models
for model in "${MODELS_NON_REASONING[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    echo "Running $model on $dataset"
    python src/run.py \
      --model_path "$model" \
      --dataset "$dataset" \
      --input_path "$INPUT_DIR/${dataset}.jsonl" \
      --output_dir "$OUTPUT_DIR/$(basename $model)/$dataset" \
      --max_samples "$MAX_SAMPLES"
  done
done

# Run for reasoning models
for model in "${MODELS_REASONING[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    echo "Running $model on $dataset"
    python src/run.py \
      --model_path "$model" \
      --dataset "$dataset" \
      --input_path "$INPUT_DIR/${dataset}.jsonl" \
      --output_dir "$OUTPUT_DIR/$(basename $model)/$dataset" \
      --max_samples "$MAX_SAMPLES"
  done
done
