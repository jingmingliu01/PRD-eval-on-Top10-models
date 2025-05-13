# generate_answers.py
# This script generates answers from each model for a given set of questions

import os
import json
from pathlib import Path
from utils import call_llm

DATA_DIR = Path("../data")
QUESTIONS_PATH = DATA_DIR / "questions" / "mt_bench.json"
ANSWERS_DIR = DATA_DIR / "answers"

# List of models to generate answers with
MODELS = ["gpt-4", "claude-3", "gemini", "mistral", "llama-3"]

def load_questions():
    with open(QUESTIONS_PATH) as f:
        return json.load(f)

def save_answer(model_name: str, qid: str, answer: str):
    out_dir = ANSWERS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{qid}.txt").write_text(answer)

def generate_for_model(model_name: str, questions):
    for q in questions:
        qid = q["question_id"]
        prompt = q["prompt"]
        print(f"Generating answer for {model_name} on question {qid}...")
        answer = call_llm(model_name, prompt)
        save_answer(model_name, qid, answer)

def run_all():
    questions = load_questions()
    for model in MODELS:
        generate_for_model(model, questions)

if __name__ == "__main__":
    run_all()
