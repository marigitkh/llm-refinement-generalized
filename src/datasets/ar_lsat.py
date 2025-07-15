import re
import json
from typing import Dict, Any, List

def load_data(path: str) -> List[Dict[str, Any]]:
    """Load line-delimited JSON (.jsonl) file into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    choices = "\n".join(item["options"]) 
    full_question = f"{item['question']}\nChoices:\n{choices}"
    return {
        "id": item.get("id"),
        "question": f"{item['context']}\n\n{full_question}",
        "answer": item["answer"].strip().upper()
    }

def extract_answer(text: str) -> str:
    match = re.findall(r"Answer:\s*([A-E])", text.strip(), flags=re.IGNORECASE)
    return match[-1].upper() if match else ""

def is_correct(pred: str, gold: str) -> bool:
    return pred.upper() == gold.upper()
