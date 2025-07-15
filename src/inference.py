from tqdm import tqdm
from utils import format_prompt, format_hint_prompt, extract_cot, is_valid_hint, contains_bad_phrases

def solve_questions(data, model, tokenizer, dataset_module, inject_hint=False, max_attempts=2):
    """Run model inference (initial or post-hint) and return structured results with fallback if answer is missing."""
    results = []
    dataset_name = dataset_module.__name__.split(".")[-1]

    for item in tqdm(data, desc="Solving questions"):

        if "answer" in item:
            processed = dataset_module.process_item(item)
        else:
            processed = {
                "id": item["id"],
                "question": item["question"],
                "answer": item["ground_truth"]
            }

        base_prompt = format_prompt(
            processed["question"],
            inject_hint=inject_hint,
            hint=item.get("hint", ""),
            dataset_name=dataset_name
        )

        full_decoded = ""
        trimmed_decoded = ""
        pred_answer = ""
        cot = ""
        is_correct = False

        for attempt in range(max_attempts):
            # Optionally force clearer answer format in retry
            prompt = base_prompt
            if attempt > 0:
                prompt += "\nPlease state your reasoning and clearly end with: Answer: <letter>"

            input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
            output = model.generate(
                **input_ids,
                max_new_tokens=256,
                do_sample=(attempt > 0),
                temperature=0.7 if attempt > 0 else None,
                pad_token_id=tokenizer.eos_token_id
            )
            full_decoded = tokenizer.decode(output[0], skip_special_tokens=True)

            trimmed_decoded = full_decoded
            if full_decoded.startswith(prompt):
                trimmed_decoded = full_decoded[len(prompt):].strip()

            pred_answer = dataset_module.extract_answer(trimmed_decoded)
            if pred_answer:
                cot = extract_cot(trimmed_decoded)
                is_correct = dataset_module.is_correct(pred_answer, processed["answer"])
                break  # valid answer obtained

        results.append({
            "id": processed["id"],
            "question": processed["question"],
            "ground_truth": processed["answer"],
            "predicted_answer": pred_answer,
            "is_correct": is_correct,
            "chain_of_thought": cot,
            "full_output": full_decoded
        })

    return results


def generate_hints(data, model, tokenizer, dataset_module, num_attempts=3, temperature=0.7):
    """Generate helpful hints using the model and ground-truth answers, avoiding answer leakage."""
    hints = []

    for item in tqdm(data, desc="Generating hints"):
        prompt = format_hint_prompt(
            item["question"],
            item["predicted_answer"],
            item.get("chain_of_thought", ""),
            item["ground_truth"]
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prefix_len = len(inputs["input_ids"][0])

        hint_sentence = ""
        for attempt in range(num_attempts):
            gen_kwargs = {"max_new_tokens": 128}
            if attempt > 0:
                gen_kwargs.update(do_sample=True, temperature=temperature)

            output_ids = model.generate(**inputs, **gen_kwargs)[0]
            decoded = tokenizer.decode(output_ids[prefix_len:], skip_special_tokens=True).strip()

            if is_valid_hint(decoded, item["ground_truth"]) and not contains_bad_phrases(decoded, item["ground_truth"]):
                hint_sentence = decoded
                break

            hint_sentence = decoded

        item_with_hint = item.copy()
        item_with_hint["hint_sentence"] = hint_sentence
        hints.append(item_with_hint)

    return hints