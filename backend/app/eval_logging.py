
import json, time, requests

def ollama_generate_with_stats(prompt: str, model="llama3.2", base_url="http://localhost:11434", timeout_s=600):
    """
    เรียก Ollama แบบ non-stream เพื่อดึง timing/token metrics เต็ม ๆ
    คืนค่า: (response_text, timing_dict)
    """
    url = f"{base_url}/api/generate"
    t0 = time.time_ns()
    r = requests.post(url, json={"model": model, "prompt": prompt, "stream": False}, timeout=timeout_s)
    t1 = time.time_ns()
    r.raise_for_status()
    data = r.json()
    timing = {
        "total_duration_ns": int(data.get("total_duration", t1 - t0)),
        "prompt_eval_count": int(data.get("prompt_eval_count", 0)),
        "prompt_eval_duration_ns": int(data.get("prompt_eval_duration", 0)),
        "eval_count": int(data.get("eval_count", 0)),
        "eval_duration_ns": int(data.get("eval_duration", 0)),
    }
    ev_ns = timing["eval_duration_ns"]
    timing["tokens_per_sec"] = (timing["eval_count"] / (ev_ns / 1e9)) if ev_ns else None
    return (data.get("response", "") or "").strip(), timing

def append_eval_run(jsonl_path: str, record: dict):
    """
    บันทึก 1 แถวลงไฟล์ JSONL
    สคีมา: question, contexts(list[str]), answer, ground_truth(optional), timing(dict)
    """
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
