import json
from pathlib import Path

from app.evaluation.dataset import load_beir_dataset


def test_load_beir_dataset(tmp_path: Path) -> None:
    (tmp_path / "qrels").mkdir()
    (tmp_path / "corpus.jsonl").write_text(
        json.dumps(
            {
                "_id": "d1",
                "title": "Evidence",
                "text": "A scientific result.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "queries.jsonl").write_text(
        json.dumps({"_id": "q1", "text": "scientific result"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\n",
        encoding="utf-8",
    )

    dataset = load_beir_dataset(tmp_path)

    assert dataset.queries == {"q1": "scientific result"}
    assert dataset.qrels == {"q1": {"d1": 1}}
    assert dataset.corpus["d1"].searchable_text == ("Evidence A scientific result.")
