import os

def format_prompt(question: str, inject_hint=False, hint="", dataset_name="arithmetic") -> str:
    """Load and format the dataset-specific answer prompt."""
    prompt_path = f"prompts/{dataset_name}_prompt.txt"
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path) as f:
        template = f.read()

    if inject_hint and hint:
        return hint.strip() + "\n\n" + template.format(question=question.strip())
    else:
        return template.format(question=question.strip())

def format_hint_prompt(question, predicted_answer, chain_of_thought, correct_answer) -> str:
    """Load and fill the general-purpose hint generation prompt."""
    with open("prompts/hint_prompt.txt") as f:
        template = f.read()
    return template.format(
        question=question.strip(),
        predicted_answer=predicted_answer.strip(),
        chain_of_thought=chain_of_thought.strip(),
        correct_answer=correct_answer.strip()
    )

def extract_cot(output: str) -> str:
    import re
    matches = re.findall(r"<cot_start>(.*?)<cot_end>", output, flags=re.DOTALL)
    return matches[-1].strip() if matches else ""

def is_valid_hint(hint: str, answer: str) -> bool:
    """Returns True if hint does not directly reveal the answer."""
    return str(answer).strip().lower() not in hint.lower()


def contains_bad_phrases(hint: str, answer: str) -> bool:
    """Checks for risky phrases that might reveal the correct answer."""
    lower_hint = hint.lower()
    blacklist = [
        "correct", "final answer", "the answer is", "answer:", "option", str(answer).strip().lower()
    ]
    return any(phrase in lower_hint for phrase in blacklist)
