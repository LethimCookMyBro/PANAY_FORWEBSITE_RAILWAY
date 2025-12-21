#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A/B benchmark + evaluation (rerank vs no-rerank) for PLCnext RAG chatbot.

Input CSV format (UTF-8):
id,question,ground_truth
1,What is the main advantage of PLCnext?,The main advantage is its openness...
2,How to configure VS Code for C++ development on PLCnext Linux?,You can configure...

Outputs:
- ab_results.csv      : raw results (each question has 2 rows: use_rerank=False/True)
- ab_summary.csv      : mean scores grouped by use_rerank
- Console summary     : pretty text summary (mean/median and timing)

Notes:
- Calls /api/agent-chat with: enable_ragas=true, fast_ragas=false, use_rerank flag.
- Retries network failures; keeps going even if some cases fail.
"""

import csv
import json
import time
import argparse
import requests
import statistics
from pathlib import Path

def safe_get(dct, *keys, default=None):
    cur = dct
    try:
        for k in keys:
            cur = cur.get(k, {})
        return cur if cur not in (None, {}) else default
    except Exception:
        return default

def post_agent_chat(api_base, question, ground_truth, use_rerank, timeout=180):
    url = f"{api_base.rstrip('/')}/api/agent-chat"
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
    return body, dt

def run_once(api_base, row_id, question, ground_truth, use_rerank, timeout=180, retries=2, sleep_retry=1.5):
    err = None
    for attempt in range(retries + 1):
        try:
            body, wall = post_agent_chat(api_base, question, ground_truth, use_rerank, timeout=timeout)
            return {
                "id": row_id,
                "question": question,
                "ground_truth": ground_truth,
                "use_rerank": use_rerank,
                "reply": body.get("reply", ""),
                "processing_time": body.get("processing_time", wall),
                "retrieval_time": body.get("retrieval_time", None),
                "context_count": body.get("context_count", None),
                "answer_relevancy": safe_get(body, "ragas", "scores", default={}).get("answer_relevancy"),
                "faithfulness": safe_get(body, "ragas", "scores", default={}).get("faithfulness"),
                "context_precision": safe_get(body, "ragas", "scores", default={}).get("context_precision"),
                "context_recall": safe_get(body, "ragas", "scores", default={}).get("context_recall"),
                "error": "",
            }
        except Exception as e:
            err = str(e)
            time.sleep(sleep_retry)
    # failed after retries
    return {
        "id": row_id,
        "question": question,
        "ground_truth": ground_truth,
        "use_rerank": use_rerank,
        "reply": "",
        "processing_time": None,
        "retrieval_time": None,
        "context_count": None,
        "answer_relevancy": None,
        "faithfulness": None,
        "context_precision": None,
        "context_recall": None,
        "error": err or "unknown error",
    }

def format_float(x):
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "-"

def main():
    ap = argparse.ArgumentParser(description="A/B benchmark (rerank vs no-rerank)")
    ap.add_argument("--api", default="http://localhost:5000", help="API base (default: http://localhost:5000)")
    ap.add_argument("--input", default="questions.csv", help="Input CSV (id,question,ground_truth)")
    ap.add_argument("--out", default="ab_results.csv", help="Output results CSV")
    ap.add_argument("--summary", default="ab_summary.csv", help="Output summary CSV (grouped by use_rerank)")
    ap.add_argument("--runs", type=int, default=1, help="Repeat each condition N times to reduce variance")
    ap.add_argument("--sleep", type=float, default=0.4, help="Sleep seconds between calls")
    ap.add_argument("--timeout", type=int, default=180, help="HTTP timeout seconds per call")
    ap.add_argument("--retries", type=int, default=2, help="Retries for each call on failure")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"❌ Input file not found: {input_path.resolve()}")

    # Load questions
    rows = []
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = (row.get("id") or "").strip()
            q = (row.get("question") or "").strip()
            gt = (row.get("ground_truth") or "").strip()
            if not rid or not q:
                continue
            rows.append({"id": rid, "question": q, "ground_truth": gt})

    if not rows:
        raise SystemExit("❌ No rows in input CSV.")

    # Run benchmark
    out_fields = [
        "id","use_rerank","processing_time","retrieval_time","context_count",
        "answer_relevancy","faithfulness","context_precision","context_recall","reply","error"
    ]
    all_results = []
    with open(args.out, "w", newline="", encoding="utf-8-sig") as wf:
        w = csv.DictWriter(wf, fieldnames=out_fields)
        w.writeheader()

        for row in rows:
            rid, q, gt = row["id"], row["question"], row["ground_truth"]

            # A) use_rerank=False, repeated N runs
            for r_i in range(args.runs):
                res_no = run_once(
                    args.api, rid, q, gt, use_rerank=False,
                    timeout=args.timeout, retries=args.retries
                )
                w.writerow({
                    "id": res_no["id"],
                    "use_rerank": res_no["use_rerank"],
                    "processing_time": res_no["processing_time"],
                    "retrieval_time": res_no["retrieval_time"],
                    "context_count": res_no["context_count"],
                    "answer_relevancy": res_no["answer_relevancy"],
                    "faithfulness": res_no["faithfulness"],
                    "context_precision": res_no["context_precision"],
                    "context_recall": res_no["context_recall"],
                    "reply": res_no["reply"],
                    "error": res_no["error"],
                })
                all_results.append(res_no)
                time.sleep(args.sleep)

            # B) use_rerank=True, repeated N runs
            for r_i in range(args.runs):
                res_yes = run_once(
                    args.api, rid, q, gt, use_rerank=True,
                    timeout=args.timeout, retries=args.retries
                )
                w.writerow({
                    "id": res_yes["id"],
                    "use_rerank": res_yes["use_rerank"],
                    "processing_time": res_yes["processing_time"],
                    "retrieval_time": res_yes["retrieval_time"],
                    "context_count": res_yes["context_count"],
                    "answer_relevancy": res_yes["answer_relevancy"],
                    "faithfulness": res_yes["faithfulness"],
                    "context_precision": res_yes["context_precision"],
                    "context_recall": res_yes["context_recall"],
                    "reply": res_yes["reply"],
                    "error": res_yes["error"],
                })
                all_results.append(res_yes)
                time.sleep(args.sleep)

    print(f"✅ Saved raw results to: {Path(args.out).resolve()}")

    # Build summary by use_rerank
    def collect(col, flt):
        vals = [x[col] for x in all_results if flt(x) and isinstance(x.get(col), (int,float))]
        return vals

    groups = {False: "no_rerank", True: "rerank"}
    summary_rows = []
    metrics = ["answer_relevancy","faithfulness","context_precision","context_recall","processing_time"]
    for key_bool, key_name in groups.items():
        row = {"use_rerank": key_name}
        for m in metrics:
            vals = collect(m, lambda x, kb=key_bool: x.get("use_rerank") == kb and x.get(m) is not None)
            if vals:
                row[m + "_mean"] = round(statistics.fmean(vals), 6)
                row[m + "_median"] = round(statistics.median(vals), 6)
                row[m + "_count"] = len(vals)
            else:
                row[m + "_mean"] = None
                row[m + "_median"] = None
                row[m + "_count"] = 0
        summary_rows.append(row)

    # Save summary CSV
    sum_fields = ["use_rerank"] + [f"{m}_{suf}" for m in metrics for suf in ("mean","median","count")]
    with open(args.summary, "w", newline="", encoding="utf-8-sig") as sf:
        sw = csv.DictWriter(sf, fieldnames=sum_fields)
        sw.writeheader()
        for r in summary_rows:
            sw.writerow(r)

    print(f"✅ Saved summary to: {Path(args.summary).resolve()}")

    # Pretty print to console
    print("\n📊 Summary (grouped by use_rerank)")
    header = ["use_rerank","ans_rel(mean)","faith(mean)","ctx_prec(mean)","ctx_rec(mean)","proc_time(mean)"]
    print(" | ".join(header))
    print("-"*len(" | ".join(header)))
    for r in summary_rows:
        line = [
            r["use_rerank"],
            format_float(r["answer_relevancy_mean"]),
            format_float(r["faithfulness_mean"]),
            format_float(r["context_precision_mean"]),
            format_float(r["context_recall_mean"]),
            format_float(r["processing_time_mean"]),
        ]
        print(" | ".join(line))

    print("\nℹ️ Tip: ดูผลละเอียดในไฟล์ ab_results.csv (มีคำตอบ/contexts/time ต่อเคส) และใช้ ab_summary.csv สำหรับตารางรวมพร้อมค่าเฉลี่ย/มัธยฐาน/จำนวนข้อมูล")

if __name__ == "__main__":
    main()
