import csv
import json
from pathlib import Path

# ----------------------------------------------------
# CHANGE THIS PATH
# ----------------------------------------------------
ARXIV_JSON = r"C:\Users\utkar\OneDrive\Desktop\build-a-complete-research-grade-project\outputs\arxiv-metadata-oai-snapshot.json"

GROUND_TRUTH = r"C:\Users\utkar\OneDrive\Desktop\build-a-complete-research-grade-project\outputs\hybrid-quantum-search\evaluation\ground_truth.json"
OUTPUT_CSV = r"C:\Users\utkar\OneDrive\Desktop\build-a-complete-research-grade-project\outputs\hybrid-quantum-search\evaluation\ground_truth_with_abstracts.csv"

# ----------------------------------------------------

print("Loading ground truth...")

with open(GROUND_TRUTH, "r", encoding="utf8") as f:
    ground_truth = json.load(f)

titles_needed = {}

for query in ground_truth:
    q = query["query"]

    for paper in query["results"]:
        title = " ".join(
            paper["title"].strip().lower().split()
        )

        titles_needed[title] = {
            "query": q,
            "title": paper["title"],
            "relevant": paper["relevant"],
            "abstract": ""
        }

print(f"Need abstracts for {len(titles_needed)} papers")

found = 0

print("Scanning arXiv snapshot...")

with open(ARXIV_JSON, "r", encoding="utf8") as f:

    for i, line in enumerate(f):

        try:
            paper = json.loads(line)
        except:
            continue

        title = " ".join(
            paper.get("title", "").strip().lower().split()
        )

        if title in titles_needed:

            titles_needed[title]["abstract"] = paper.get(
                "abstract",
                ""
            )

            found += 1

            if found % 10 == 0:
                print(f"Found {found}")

            if found == len(titles_needed):
                break

print()

print(f"Matched {found}/{len(titles_needed)} papers")

print("Writing CSV...")

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf8"
) as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow(
        [
            "Query",
            "Title",
            "Abstract",
            "Relevant"
        ]
    )

    for paper in titles_needed.values():

        writer.writerow(
            [
                paper["query"],
                paper["title"],
                paper["abstract"],
                paper["relevant"]
            ]
        )

print()

print("Done!")
print(f"Saved to {OUTPUT_CSV}")
