#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv, json, time, argparse, requests
from pathlib import Path

def run_case(api_base, qid, question, ground_truth, use_rerank, timeout=120):
    url = f"{api_base}/api/agent-chat"
    data = {
        "message": question,
        "enable_ragas": "true",
        "log_eval": "false",
        "fast_ragas": "false",
        "ground_truth": ground_truth or "",
        "use_rerank": "true" if use_rerank else "false",
    }
    t0 = time.perf_counter()
    r = requests.post(url, data=data, timeout=timeout)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    body = r.json()
    return {
        "id": qid,
        "use_rerank": use_rerank,
        "reply": body.get("reply",""),
        "processing_time": body.get("processing_time", dt),
        "retrieval_time": body.get("retrieval_time", None),
        "context_count": body.get("context_count", None),
        "ragas_answer_relevancy": (body.get("ragas") or {}).get("scores",{}).get("answer_relevancy"),
        "ragas_faithfulness": (body.get("ragas") or {}).get("scores",{}).get("faithfulness"),
        "ragas_context_precision": (body.get("ragas") or {}).get("scores",{}).get("context_precision"),
        "ragas_context_recall": (body.get("ragas") or {}).get("scores",{}).get("context_recall"),
    }

def main():
    p = argparse.ArgumentParser(description="A/B benchmark rerank vs no-rerank for PLCnext chatbot")
    p.add_argument("--api", default="http://localhost:5000", help="API base (e.g. http://localhost:5000)")
    p.add_argument("--input", required=True, help="CSV file: id,question,ground_truth")
    p.add_argument("--out", default="ab_results.csv", help="Output CSV")
    p.add_argument("--sleep", type=float, default=0.5, help="Sleep between calls (sec)")
    args = p.parse_args()

    rows = []
    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    out_fields = [
        "id","use_rerank","processing_time","retrieval_time","context_count",
        "ragas_answer_relevancy","ragas_faithfulness","ragas_context_precision","ragas_context_recall","reply"
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as wf:
        w = csv.DictWriter(wf, fieldnames=out_fields)
        w.writeheader()

        for r in rows:
            qid = r.get("id") or ""
            q = r.get("question") or ""
            gt = r.get("ground_truth") or ""
            # A) no-rerank
            a = run_case(args.api, qid, q, gt, use_rerank=False)
            w.writerow(a)
            time.sleep(args.sleep)
            # B) rerank
            b = run_case(args.api, qid, q, gt, use_rerank=True)
            w.writerow(b)
            time.sleep(args.sleep)

    print(f"✅ Done. Saved results to: {args.out}")

if __name__ == "__main__":
    main()
