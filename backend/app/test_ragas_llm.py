
import time
import os

# ✅ Force settings
os.environ["ENABLE_RAGAS_LLM"] = "true"
os.environ["RAGAS_LLM_MODEL"] = "llama3.2"
os.environ["RAGAS_TIMEOUT"] = "300"

from ragas_eval import local_ragas_eval

question = "What is PLCnext?"
answer = "PLCnext is an open automation platform by Phoenix Contact."

# ✅ Context แบบสั้นและชัดเจน + มี Ground Truth ที่ตรงกัน 100%
contexts = [
    """Question: What is PLCnext?
Answer: PLCnext is an open automation platform by Phoenix Contact.""",
    
    "PLCnext Technology combines PLC capabilities with flexibility."
]

print("=" * 60)
print("Testing RAGAS LLM Judge (Conservative Settings)")
print("=" * 60)
print(f"Question: {question}")
print(f"Answer: {answer}")
print(f"Contexts: {len(contexts)} items")
print("-" * 60)

t0 = time.time()
result = local_ragas_eval(question, answer, contexts)
elapsed = time.time() - t0

print(f"\nTotal Time: {elapsed:.2f}s")
print(f"Status: {result.get('status')}")
print(f"Judge: {result.get('judge_type')}")
print(f"Has GT: {result.get('has_ground_truth')}")
print(f"RAGAS OK: {result.get('ragas_success')}")

print("\nScores:")
for k, v in result.get('scores', {}).items():
    if v is not None:
        print(f"  {k}: {v:.3f}")
    else:
        print(f"  {k}: None (skipped or failed)")

success = [k for k, v in result.get('scores', {}).items() if v is not None]
print(f"\n{'✓' if len(success) >= 2 else '✗'} Got {len(success)}/4 metrics")