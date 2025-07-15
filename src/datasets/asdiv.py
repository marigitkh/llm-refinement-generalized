import re
from typing import Dict, Any

def load_data(path: str):
    import json
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item.get("id"),
        "question": item["question"],
        "answer": str(item["answer"]).strip()
    }

def extract_answer(text: str) -> str:
    match = re.findall(r"Answer:\s*([-+]?[0-9,]+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    return match[-1].replace(",", "") if match else ""

def is_correct(pred: str, gold: str) -> bool:
    try:
        return float(pred) == float(gold)
    except:
        return False