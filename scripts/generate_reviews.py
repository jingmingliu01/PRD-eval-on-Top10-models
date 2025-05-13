# coding: utf-8
import os
import json
from pathlib import Path
from typing import List
from utils import call_llm, build_review_prompt

DATA_DIR = Path("../data")

def load_answers(question_id: str, model_a: str, model_b: str) -> (str, str):
    ans_a = (DATA_DIR / "answers" / model_a / f"{question_id}.txt").read_text()
    ans_b = (DATA_DIR / "answers" / model_b / f"{question_id}.txt").read_text()
    return ans_a, ans_b

def load_question(question_id: str) -> str:
    q_path = DATA_DIR / "questions" / "mt_bench.json"
    questions = json.loads(q_path.read_text())
    return next(q["prompt"] for q in questions if q["question_id"] == question_id)

def save_review_result(reviewer, qid, model_a, model_b, score, rationale):
    out_dir = DATA_DIR / "reviews" / reviewer
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{qid}__{model_a}__vs__{model_b}.json"
    json.dump({
        "question_id": qid,
        "model_a": model_a,
        "model_b": model_b,
        "score": score,           # -1, 0, or 1
        "rationale": rationale
    }, open(out_path, "w"), indent=2)

def review_pair(qid: str, model_a: str, model_b: str, reviewer: str):
    q = load_question(qid)
    ans_a, ans_b = load_answers(qid, model_a, model_b)
    prompt = build_review_prompt(q, ans_a, ans_b)
    score, rationale = call_llm(reviewer, prompt)
    save_review_result(reviewer, qid, model_a, model_b, score, rationale)

def run_all():
    reviewer = "gpt-4"
    models = ["gpt-4", "claude-3", "gemini", "mistral", "llama-3"]
    qids = ["q001", "q002", "q003"]
    for qid in qids:
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                review_pair(qid, models[i], models[j], reviewer)

if __name__ == "__main__":
    run_all()
