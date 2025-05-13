# peer_discussion.py
# This script implements Peer Discussion (PD) between two models for a given question

import os
import json
from pathlib import Path
from utils import call_llm, build_discussion_prompt

DATA_DIR = Path("../data")
QUESTIONS_PATH = DATA_DIR / "questions" / "mt_bench.json"
ANSWERS_DIR = DATA_DIR / "answers"
DISCUSSION_DIR = DATA_DIR / "discussions"

MAX_TURNS = 4

# Load prompt and answers
def load_context(qid: str, model_a: str, model_b: str):
    q_data = json.loads((QUESTIONS_PATH).read_text())
    prompt = next(q["prompt"] for q in q_data if q["question_id"] == qid)
    ans_a = (ANSWERS_DIR / model_a / f"{qid}.txt").read_text()
    ans_b = (ANSWERS_DIR / model_b / f"{qid}.txt").read_text()
    return prompt, ans_a, ans_b

# Save final discussion record
def save_discussion(model_pair: str, qid: str, record: dict):
    out_dir = DISCUSSION_DIR / model_pair
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{qid}.json", "w") as f:
        json.dump(record, f, indent=2)

# Run peer discussion between model_a and model_b
# Use model_a as Reviewer 1, model_b as Reviewer 2

def run_discussion(qid: str, model_a: str, model_b: str):
    prompt, ans_a, ans_b = load_context(qid, model_a, model_b)

    history = []
    for turn in range(MAX_TURNS):
        current_model = model_a if turn % 2 == 0 else model_b
        role = f"Reviewer {1 if turn % 2 == 0 else 2}"

        full_prompt = build_discussion_prompt(prompt, ans_a, ans_b, history, role)
        response = call_llm(current_model, full_prompt)

        history.append({"turn": turn + 1, "role": role, "model": current_model, "content": response})

        if any(x in response.lower() for x in ["i agree", "we agree", "we both agree"]):
            break

    save_discussion(f"{model_a}_vs_{model_b}", qid, {
        "question_id": qid,
        "models": [model_a, model_b],
        "history": history
    })

# Example run
if __name__ == "__main__":
    run_discussion("q001", "gpt-4", "claude-3")
