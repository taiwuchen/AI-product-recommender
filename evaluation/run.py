import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.search.indexes import catalog_fingerprint, load_or_build
from src.search.keyword import KeywordSearch
from src.utils.data_loader import ProductDataLoader

ROOT = Path(__file__).resolve().parents[1]


def metrics(retrieved, relevant, k=5):
    hits = [int(product in relevant) for product in retrieved[:k]]
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return {"precision_at_5": sum(hits) / k, "recall_at_5": sum(hits) / len(relevant),
            "ndcg_at_5": dcg / ideal, "reciprocal_rank_at_5": next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0)}


def evaluate(modes, output):
    load_dotenv(ROOT / ".env")
    data_path = ROOT / "ZARA_jackets_men.csv"
    dataset = json.loads((ROOT / "evaluation/queries.json").read_text())
    if dataset["catalog_sha256"] != catalog_fingerprint(data_path):
        raise ValueError("Catalog changed. Review the relevance labels before evaluating.")
    loader = ProductDataLoader(data_path)
    df = loader.preprocess_data()
    links = df["link"].tolist()
    keyword = KeywordSearch(df["text_for_embedding"].tolist())
    for case in dataset["queries"]:
        if not case["relevant_links"] or not set(case["relevant_links"]).issubset(links):
            raise ValueError("Relevance labels contain missing or unknown products.")
    model = db = None
    setup_start = time.perf_counter()
    if any(mode != "keyword" for mode in modes):
        from src.models.text_embedding import MODEL_ID, TextEmbeddingGenerator
        model = TextEmbeddingGenerator()
        db, _ = load_or_build(data_path, df, ROOT / "indexes", "text", MODEL_ID, lambda: model)
        model.generate_text_embedding("jacket")
    setup_ms = (time.perf_counter() - setup_start) * 1000
    rows = []
    for case in dataset["queries"]:
        embedding = None
        embedding_ms = 0
        if model:
            started = time.perf_counter()
            embedding = model.generate_text_embedding(case["query"])
            embedding_ms = (time.perf_counter() - started) * 1000
        for mode in modes:
            started = time.perf_counter()
            if mode == "keyword":
                ids = keyword.search(case["query"])
            else:
                _, indices = db.search_by_text(embedding, query_text=case["query"], keyword_boost=mode == "boosted")
                ids = indices[0].tolist()
            retrieval_ms = (time.perf_counter() - started) * 1000
            retrieved = [links[i] for i in ids]
            rows.append({"query": case["query"], "category": case["category"], "mode": mode,
                         **metrics(retrieved, case["relevant_links"]), "retrieval_ms": retrieval_ms,
                         "search_ms": retrieval_ms + (embedding_ms if mode != "keyword" else 0),
                         "results": [{"id": i, "name": df.iloc[i]["product_name"], "relevant": links[i] in case["relevant_links"]} for i in ids]})
    summary = {}
    for mode in modes:
        values = [row for row in rows if row["mode"] == mode]
        summary[mode] = {metric: statistics.mean(row[metric] for row in values)
                         for metric in ["precision_at_5", "recall_at_5", "ndcg_at_5", "reciprocal_rank_at_5"]}
        times = sorted(row["search_ms"] for row in values)
        summary[mode].update(median_search_ms=statistics.median(times), p95_search_ms=times[math.ceil(len(times) * .95) - 1])
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "catalog_sha256": dataset["catalog_sha256"],
              "queries_sha256": hashlib.sha256((ROOT / "evaluation/queries.json").read_bytes()).hexdigest(),
              "judgments": dataset["judgments"], "python": platform.python_version(), "platform": platform.platform(),
              "products": len(df), "queries": len(dataset["queries"]), "setup_ms": setup_ms,
              "text_model": "text-embedding-005" if model else None, "summary": summary, "per_query": rows}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    lines = ["# Search evaluation", "", f"Run: {report['generated_at']}", "", dataset["judgments"], "",
             f"{len(df)} products; {len(dataset['queries'])} queries. See the JSON report for every ranked result.", "",
             "| Method | P@5 | Recall@5 | nDCG@5 | MRR@5 | Median ms | p95 ms |", "|---|---:|---:|---:|---:|---:|---:|"]
    for mode, values in summary.items():
        lines.append(f"| {mode} | " + " | ".join(f"{value:.3f}" for value in values.values()) + " |")
    lines += ["", "Search time includes query embedding and retrieval, excluding setup, image downloads, and rendering. Semantic and boosted modes share the same measured query embedding cost. One timed pass; latency is illustrative, not a load test.", "",
              "P@5 always divides by five, even when fewer than five relevant products exist. Recall and nDCG help interpret those cases. Unlisted products count as nonrelevant under the provisional labels.", "", "## Lowest-ranked cases", ""]
    for mode in modes:
        lines.append(f"### {mode}")
        lines.append("")
        for row in sorted((row for row in rows if row["mode"] == mode), key=lambda row: row["ndcg_at_5"])[:3]:
            names = "; ".join(f"{result['name']} ({'relevant' if result['relevant'] else 'not labeled relevant'})" for result in row["results"][:3])
            lines.append(f"- **{row['query']}** — nDCG@5 {row['ndcg_at_5']:.3f}. Top three: {names or 'no results'}.")
        lines.append("")
    output.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", choices=["keyword", "semantic", "boosted"], default=["keyword", "semantic", "boosted"])
    parser.add_argument("--output", default="evaluation/results")
    args = parser.parse_args()
    evaluate(args.modes, args.output)
