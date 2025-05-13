# utils.py
# Helper functions for prompt construction and API calls

from openai import OpenAI
from anthropic import Anthropic

# Prompt to compare two answers using three criteria
def build_review_prompt(question: str, ans1: str, ans2: str) -> str:
    return f"""You are a helpful and fair judge.

Question: {question}

Answer A:
{ans1}

Answer B:
{ans2}

Evaluate based on:
(1) Unsupported information — penalize made-up content.
(2) Core information — whether the answer truly addresses the question.
(3) Coherence — clarity and logical flow.

Respond with exactly one of: 'A is better', 'B is better', or 'They are equal'. Then explain your reasoning."""

# Prompt to conduct one round of peer discussion
def build_discussion_prompt(question, ans_a, ans_b, history, role):
    intro = f"You are {role} participating in a peer review discussion about two answers to the following question:\n\n"
    base = f"Question: {question}\n\nAnswer A:\n{ans_a}\n\nAnswer B:\n{ans_b}\n\n"
    guide = (
        "Please discuss which answer is better using three criteria:\n"
        "(1) Unsupported info, (2) Core information, (3) Coherence.\n"
        "Clearly indicate agreement if consensus is reached."
    )
    dialogue = "\n\nDiscussion history:\n" + "\n".join([f"{h['role']} ({h['model']}): {h['content']}" for h in history])
    return intro + base + guide + dialogue

# Model-agnostic LLM call wrapper (expandable)
def call_llm(model_name: str, prompt: str) -> str:
    if model_name.startswith("gpt"):
        client = OpenAI()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    elif model_name.startswith("claude"):
        client = Anthropic()
        response = client.messages.create(
            model=model_name,
            max_tokens=1024,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
