"""
Process Golden QA data for RAGAS evaluation
File: backend/app/process_golden_qa.py
"""

import json
import os
import sys
import time
from typing import List, Dict, Any
from pathlib import Path

# Add app to path
sys.path.append('/app')

# Import dependencies
from app.ragas_eval import local_ragas_eval
from app.chatbot import answer_question
from app.retriever import PostgresVectorRetriever, EnhancedFlashrankRerankRetriever

# Import from main for dependencies
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from psycopg2 import pool

def load_golden_qa(filepath: str) -> List[Dict[str, Any]]:
    """Load golden QA pairs from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_dependencies():
    """Setup database pool, LLM, and embedder"""
    DB_URL = os.getenv("DATABASE_URL")
    if not DB_URL:
        raise RuntimeError("DATABASE_URL environment variable is required.")
    
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
    
    # Setup database pool
    db_pool = pool.SimpleConnectionPool(
        1, 10,
        dsn=DB_URL,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5
    )
    
    # Setup LLM
    llm = OllamaLLM(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        timeout=60
    )
    
    # Setup embedder
    embedder = SentenceTransformer(
        EMBED_MODEL_NAME,
        cache_folder='/app/models'
    )
    
    return db_pool, llm, embedder

def run_batch_evaluation(golden_qa_file: str, output_file: str):
    """
    Run RAGAS evaluation on entire golden dataset
    """
    print(f"Loading Golden QA from: {golden_qa_file}")
    
    # Load golden QA
    if not os.path.exists(golden_qa_file):
        print(f"Error: File not found: {golden_qa_file}")
        return
    
    golden_qa = load_golden_qa(golden_qa_file)
    print(f"Loaded {len(golden_qa)} questions")
    
    # Setup dependencies
    print("Setting up dependencies...")
    db_pool, llm, embedder = setup_dependencies()
    
    results = []
    
    for i, item in enumerate(golden_qa):
        print(f"Evaluating {i+1}/{len(golden_qa)}: {item['question'][:50]}...")
        
        try:
            start_time = time.time()
            
            # Get chatbot answer
            chat_result = answer_question(
                question=item["question"],
                db_pool=db_pool,
                llm=llm,
                embedder=embedder,
                collection="plcnext",
                retriever_class=PostgresVectorRetriever,
                reranker_class=EnhancedFlashrankRerankRetriever
            )
            
            # Run RAGAS evaluation
            ragas_result = local_ragas_eval(
                question=item["question"],
                answer=chat_result["reply"],
                contexts=chat_result.get("contexts", [])
            )
            
            # Combine results
            evaluation_result = {
                "question_id": i,
                "category": item.get("category", "general"),
                "question": item["question"],
                "ground_truth": item["answer"],
                "predicted_answer": chat_result["reply"],
                "retrieval_time": chat_result.get("retrieval_time"),
                "context_count": chat_result.get("context_count"),
                "ragas_scores": ragas_result.get("scores", {}),
                "ragas_status": ragas_result.get("status"),
                "ragas_judge_type": ragas_result.get("judge_type"),
                "total_evaluation_time": time.time() - start_time
            }
            
            results.append(evaluation_result)
            
        except Exception as e:
            print(f"Error evaluating item {i+1}: {e}")
            results.append({
                "question_id": i,
                "question": item["question"],
                "error": str(e)
            })
    
    # Clean up
    if db_pool:
        db_pool.closeall()
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Batch evaluation completed. Results saved to {output_file}")
    return results

def analyze_results(results_file: str):
    """Analyze RAGAS evaluation results"""
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Calculate average scores
    metrics = ["answer_relevancy", "faithfulness", "context_precision", "context_recall"]
    avg_scores = {}
    
    for metric in metrics:
        scores = []
        for r in results:
            if "ragas_scores" in r and r["ragas_scores"]:
                score = r["ragas_scores"].get(metric)
                if score is not None and isinstance(score, (int, float)):
                    scores.append(float(score))
        
        if scores:
            avg_scores[metric] = {
                "average": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "count": len(scores)
            }
    
    print("RAGAS Evaluation Results Summary:")
    print("=" * 50)
    for metric, stats in avg_scores.items():
        print(f"{metric}:")
        print(f"  Average: {stats['average']:.3f}")
        print(f"  Range: {stats['min']:.3f} - {stats['max']:.3f}")
        print(f"  Valid samples: {stats['count']}/{len(results)}")
        print()
    
    # Count by status
    status_counts = {}
    for r in results:
        status = r.get("ragas_status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print("Evaluation Status:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    
    return avg_scores

if __name__ == "__main__":
    # Usage
    golden_qa_path = "/app/data/Knowledge/golden_qa.json"
    output_path = "/app/ragas_evaluation_results.json"
    
    print("Starting batch RAGAS evaluation...")
    results = run_batch_evaluation(golden_qa_path, output_path)
    
    if results:
        print("\nAnalyzing results...")
        analyze_results(output_path)