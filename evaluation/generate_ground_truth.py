import json
import requests

API = "http://127.0.0.1:8000/api/v1/search"

TOP_K = 5

with open("evaluation_queries.json") as f:
    queries = json.load(f)

ground_truth = []

for sample in queries:

    query = sample["query"]

    response = requests.post(
        API,
        json={
            "query": query,
            "top_k": TOP_K
        }
    )

    response.raise_for_status()

    results = response.json()["results"]

    papers = []

    for r in results:

        papers.append(
            {
                "title": r["title"],
                "relevant": False
            }
        )

    ground_truth.append(
        {
            "query": query,
            "results": papers
        }
    )

with open("ground_truth.json", "w", encoding="utf8") as f:

    json.dump(
        ground_truth,
        f,
        indent=4,
        ensure_ascii=False
    )

print("ground_truth.json created.")