# peer_rank.py
# This script implements the Peer Rank algorithm described in the PRD paper
# It computes win rates, reviewer weights, and final rankings from collected review data

import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("../data/reviews")

# Load review results
# Each review is a dict with keys: question_id, model_a, model_b, score (-1/0/1), rationale
def load_reviews() -> list:
    reviews = []
    for reviewer_dir in DATA_DIR.iterdir():
        for file in reviewer_dir.glob("*.json"):
            with open(file) as f:
                review = json.load(f)
                review["reviewer"] = reviewer_dir.name
                reviews.append(review)
    return reviews

# Extract all unique models
def extract_models(reviews):
    models = set()
    for r in reviews:
        models.update([r["model_a"], r["model_b"], r["reviewer"]])
    return sorted(models)

# Convert score to [0,1] scale: -1 -> 0, 0 -> 0.5, 1 -> 1
score_transform = lambda x: (x + 1) / 2

def compute_winrate_matrix(reviews, models):
    # Initialize win count matrix: [reviewer][contestant] = total score
    counts = defaultdict(lambda: defaultdict(float))
    totals = defaultdict(lambda: defaultdict(int))

    for r in reviews:
        qid = r["question_id"]
        a, b = r["model_a"], r["model_b"]
        reviewer = r["reviewer"]
        s = r["score"]

        counts[reviewer][a] += score_transform(-s)  # a is in position 1
        counts[reviewer][b] += score_transform(s)   # b is in position 2
        totals[reviewer][a] += 1
        totals[reviewer][b] += 1

    winrate_matrix = {r: {m: counts[r][m] / totals[r][m] if totals[r][m] > 0 else 0.5 
                          for m in models} for r in models}  # reviewers must be in models too
    return winrate_matrix

# Peer Rank algorithm (iterative)
def run_peer_rank(winrates, models, max_iter=20, tol=1e-4):
    alpha = {m: 1.0 for m in models}  # initial uniform weights

    for iteration in range(max_iter):
        # Compute scores: each contestant gets weighted sum of review scores
        scores = {c: sum(alpha[r] * winrates[r][c] for r in models) for c in models}

        # Min-max normalization to [0,1]
        values = np.array(list(scores.values()))
        min_v, max_v = values.min(), values.max()
        norm_scores = {m: (scores[m] - min_v) / (max_v - min_v + 1e-8) for m in models}

        # Normalize so weights sum to 1
        total = sum(norm_scores.values())
        new_alpha = {m: norm_scores[m] / total for m in models}

        # Check convergence
        delta = sum(abs(new_alpha[m] - alpha[m]) for m in models)
        if delta < tol:
            break

        alpha = new_alpha

    # Final ranking: sort by total scores
    final_scores = {m: sum(alpha[r] * winrates[r][m] for r in models) for m in models}
    sorted_models = sorted(final_scores.items(), key=lambda x: -x[1])
    return sorted_models, final_scores, alpha

if __name__ == "__main__":
    reviews = load_reviews()
    models = extract_models(reviews)
    winrates = compute_winrate_matrix(reviews, models)
    ranking, scores, weights = run_peer_rank(winrates, models)

    print("\nPeer Rank Leaderboard:")
    for i, (m, s) in enumerate(ranking):
        print(f"{i+1:2d}. {m:<20s} Score: {s:.4f}  Weight: {weights[m]:.4f}")
