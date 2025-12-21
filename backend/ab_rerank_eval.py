import csv
import json
import requests
import time
import pandas as pd

API_URL = "http://localhost:5000/api/agent-chat"

def call_agent_chat(message, ground_truth, use_rerank, enable_ragas=True, fast_ragas=False):
    """ยิง API agent-chat"""
    payload = {
        "message": message,
        "ground_truth": ground_truth,
        "use_rerank": str(use_rerank).lower(),
        "enable_ragas": str(enable_ragas).lower(),
        "fast_ragas": str(fast_ragas).lower(),
        "log_eval": "false",
    }
    resp = requests.post(API_URL, data=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()

def main():
    input_file = "questions.csv"
    output_file = "ab_results.csv"

    results = []

    with open(input_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["id"]
            question = row["question"]
            ground_truth = row.get("ground_truth", "")

            print(f"\n⚡ Running Q{qid}: {question}")

            # --- run without rerank
            start = time.time()
            no_rerank = call_agent_chat(question, ground_truth, use_rerank=False)
            t1 = time.time() - start

            # --- run with rerank
            start = time.time()
            yes_rerank = call_agent_chat(question, ground_truth, use_rerank=True)
            t2 = time.time() - start

            results.append({
                "id": qid,
                "question": question,
                "ground_truth": ground_truth,
                "use_rerank": False,
                "reply": no_rerank.get("reply"),
                "context_count": no_rerank.get("context_count"),
                "processing_time": t1,
                "answer_relevancy": no_rerank.get("ragas", {}).get("scores", {}).get("answer_relevancy"),
                "context_precision": no_rerank.get("ragas", {}).get("scores", {}).get("context_precision"),
                "context_recall": no_rerank.get("ragas", {}).get("scores", {}).get("context_recall"),
                "faithfulness": no_rerank.get("ragas", {}).get("scores", {}).get("faithfulness"),
            })
            results.append({
                "id": qid,
                "question": question,
                "ground_truth": ground_truth,
                "use_rerank": True,
                "reply": yes_rerank.get("reply"),
                "context_count": yes_rerank.get("context_count"),
                "processing_time": t2,
                "answer_relevancy": yes_rerank.get("ragas", {}).get("scores", {}).get("answer_relevancy"),
                "context_precision": yes_rerank.get("ragas", {}).get("scores", {}).get("context_precision"),
                "context_recall": yes_rerank.get("ragas", {}).get("scores", {}).get("context_recall"),
                "faithfulness": yes_rerank.get("ragas", {}).get("scores", {}).get("faithfulness"),
            })

    # --- save results ---
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n✅ Results saved to {output_file}")

    # --- summary stats ---
    summary = df.groupby("use_rerank")[["answer_relevancy","context_precision","context_recall","faithfulness"]].mean()
    print("\n📊 Summary (mean scores):")
    print(summary)

if __name__ == "__main__":
    main()
