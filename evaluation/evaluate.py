import json
import math
import csv
import requests
import matplotlib.pyplot as plt

API = "http://127.0.0.1:8000/api/v1/search"
TOP_K = 5

GROUND_TRUTH = "ground_truth.json"

with open(GROUND_TRUTH, "r", encoding="utf8") as f:
    gt = json.load(f)

precision_scores = []
recall_scores = []
f1_scores = []
ap_scores = []
rr_scores = []
ndcg_scores = []

for sample in gt:

    query = sample["query"]

    response = requests.post(
        API,
        json={
            "query": query,
            "top_k": TOP_K
        }
    )

    response.raise_for_status()

    retrieved = response.json()["results"]

    retrieved_titles = [x["title"] for x in retrieved]

    relevant_titles = [
        x["title"]
        for x in sample["results"]
        if x["relevant"]
    ]

    hits = 0
    precision_running = []

    rr = 0

    dcg = 0

    for rank, title in enumerate(retrieved_titles, start=1):

        if title in relevant_titles:

            hits += 1

            precision_running.append(hits / rank)

            if rr == 0:
                rr = 1 / rank

            dcg += 1 / math.log2(rank + 1)

    precision = hits / TOP_K

    recall = (
        hits / len(relevant_titles)
        if relevant_titles
        else 0
    )

    if precision + recall:

        f1 = (
            2 * precision * recall /
            (precision + recall)
        )

    else:

        f1 = 0

    ap = (
        sum(precision_running) /
        len(relevant_titles)
        if relevant_titles
        else 0
    )

    ideal = min(
        len(relevant_titles),
        TOP_K
    )

    idcg = sum(
        1 / math.log2(i + 2)
        for i in range(ideal)
    )

    ndcg = dcg / idcg if idcg else 0

    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    ap_scores.append(ap)
    rr_scores.append(rr)
    ndcg_scores.append(ndcg)

results = {
    "Precision@5":
        sum(precision_scores) /
        len(precision_scores),

    "Recall@5":
        sum(recall_scores) /
        len(recall_scores),

    "F1@5":
        sum(f1_scores) /
        len(f1_scores),

    "MAP":
        sum(ap_scores) /
        len(ap_scores),

    "MRR":
        sum(rr_scores) /
        len(rr_scores),

    "NDCG@5":
        sum(ndcg_scores) /
        len(ndcg_scores),
}

print()

print("=" * 40)

for k, v in results.items():

    print(
        f"{k:<15}: {v:.4f}"
    )

print("=" * 40)

with open(
    "evaluation_results.json",
    "w",
    encoding="utf8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )

with open(
    "evaluation_metrics.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow(
        ["Metric", "Value"]
    )

    for k, v in results.items():

        writer.writerow(
            [k, v]
        )

plt.figure(figsize=(8, 5))

plt.bar(
    results.keys(),
    results.values()
)

plt.ylim(0, 1)

plt.ylabel("Score")

plt.title("Search Evaluation Metrics")

plt.tight_layout()

plt.savefig(
    "metrics.png",
    dpi=300
)

plt.show()