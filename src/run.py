import os
import torch
import importlib
import json
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
from inference import solve_questions, generate_hints

def save_jsonl(data, path):
    """Save list of dicts to a JSONL file."""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def main():
    parser = ArgumentParser(description="Run self-refinement evaluation pipeline.")
    parser.add_argument("--model_path", type=str, required=True, help="Model checkpoint or HuggingFace model ID")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (matches datasets/*.py)")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input dataset file")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to store result JSONLs")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional limit on number of examples")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load dataset processing module dynamically ---
    dataset_module = importlib.import_module(f"datasets.{args.dataset}")

    # --- Load input data ---
    raw_data = dataset_module.load_data(args.input_path)
    if args.max_samples:
        raw_data = raw_data[:args.max_samples]

    # --- Load model and tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()

    # --- Initial Inference ---
    print("Stage 1: Initial inference...")
    initial_results = solve_questions(raw_data, model, tokenizer, dataset_module)
    save_jsonl(initial_results, os.path.join(args.output_dir, "initial_inference.jsonl"))

    wrong_only = [ex for ex in initial_results if not ex.get("is_correct", False)]

    # --- Hint Generation ---
    print("Stage 2: Hint generation for incorrect answers...")
    hint_results = generate_hints(wrong_only, model, tokenizer, dataset_module)
    save_jsonl(hint_results, os.path.join(args.output_dir, "hints.jsonl"))

    # --- Post-hint Re-inference ---
    print("Stage 3: Post-hint inference...")
    post_results = solve_questions(hint_results, model, tokenizer, dataset_module, inject_hint=True)
    save_jsonl(post_results, os.path.join(args.output_dir, "post_hint_inference.jsonl"))

    print(f"\nAll done. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()