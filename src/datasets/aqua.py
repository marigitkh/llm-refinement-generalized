import re
from typing import Dict, Any

def load_data(path: str):
    import json
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    # Join choices into properly formatted multiple-choice format
    choices = "\n".join(item["options"])
    full_question = f"{item['question'].strip()}\nChoices:\n{choices}"
    
    return {
        "id": item.get("id"),
        "question": full_question,
        "answer": item["answer"].strip().upper()
    }

def extract_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # normalize whitespace

    match = re.findall(r"answer:\s*[\*\(\[]*([A-E])[\*\)\]]*", text, flags=re.IGNORECASE)
    return match[-1].upper() if match else ""

def is_correct(predicted: str, gold: str) -> bool:
    return predicted.upper() == gold.upper()
