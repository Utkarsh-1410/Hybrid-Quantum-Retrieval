import json

with open("ground_truth.json", "r", encoding="utf8") as f:
    ground_truth = json.load(f)

with open("abc.json", "r", encoding="utf8") as f:
    labels = json.load(f)

with open("remaining_labels.json", "r", encoding="utf8") as f:
    labels.extend(json.load(f))

counter = 0

for query in ground_truth:
    for paper in query["results"]:
        paper["relevant"] = labels[counter]["relevant"]
        counter += 1

with open("ground_truth_labeled.json", "w", encoding="utf8") as f:
    json.dump(ground_truth, f, indent=4, ensure_ascii=False)

print("Total labels applied:", counter)
