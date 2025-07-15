import os
import argparse
from datasets import load_data

def format_stats(result_dir: str) -> str:
    model_name = os.path.basename(os.path.dirname(result_dir))
    dataset_name = os.path.basename(result_dir)
    label = f"{model_name}/{dataset_name}"

    init_path = os.path.join(result_dir, "initial_inference.jsonl")
    post_path = os.path.join(result_dir, "post_hint_inference.jsonl")

    if not os.path.exists(init_path):
        return None

    initial = load_data(init_path)
    total = len(initial)
    if total == 0:
        return None

    correct_initial = sum(r.get("is_correct", False) for r in initial)
    wrong_initial = total - correct_initial

    post_correct = 0
    if os.path.exists(post_path):
        post = load_data(post_path)
        post_correct = sum(r.get("is_correct", False) for r in post)

    before_acc = correct_initial / total
    after_correct = correct_initial + post_correct
    after_acc = after_correct / total
    delta_pct = (after_acc - before_acc) * 100

    lines = [
        f"Model/Dataset: {label}",
        f"  Total Questions Evaluated: {total}",
        f"  Incorrect Initially: {wrong_initial}",
        f"  Accuracy Before Hints: {correct_initial}/{total} = {before_acc:.2%}",
        (
            f"  Net Gain on Wrong Subset: {post_correct}/{wrong_initial} = {(post_correct / wrong_initial):.2%}"
            if wrong_initial > 0 else "  No initially wrong questions to improve"
        ),
        f"  Accuracy After Hints: {after_correct}/{total} = {after_acc:.2%}",
        f"  Δ Accuracy: {delta_pct:+.2f}%"
    ]
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Compute accuracy stats.")
    parser.add_argument("--parent_dir", type=str, default="results", help="Root folder of model/dataset outputs")
    parser.add_argument("--output_file", type=str, default="statistics.txt", help="Path to summary output file")
    args = parser.parse_args()

    blocks = []
    for model in sorted(os.listdir(args.parent_dir)):
        model_dir = os.path.join(args.parent_dir, model)
        if not os.path.isdir(model_dir): continue

        for dataset in sorted(os.listdir(model_dir)):
            ds_dir = os.path.join(model_dir, dataset)
            if not os.path.isdir(ds_dir): continue

            block = format_stats(ds_dir)
            if block:
                print(block + "\n")
                blocks.append(block)

    with open(args.output_file, "w") as f:
        f.write("\n\n".join(blocks))

    print(f"Wrote summaries for {len(blocks)} model/dataset pairs to '{args.output_file}'")

if __name__ == "__main__":
    main()