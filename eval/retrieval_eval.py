"""
Simple retrieval evaluation script.

Input format: JSON file with list of queries:
[
    {
        "id": "q1",
        "query": "What causes ...",
        "relevant_ids": ["text::paper1::3", "image::fig2"]
    },
    ...
]

Usage:
    python retrieval_eval.py --dataset queries.json --k 10
"""
import json
import sys
from pathlib import Path
from typing import List
import argparse
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retriever import hybrid_retrieve


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Calculate Precision@K.
    
    Precision@K = (# of relevant docs in top-K) / K
    """
    if k <= 0:
        return 0.0
    topk = retrieved_ids[:k]
    relevant_in_topk = sum(1 for r in topk if r in relevant_ids)
    return relevant_in_topk / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Calculate Recall@K.
    
    Recall@K = (# of relevant docs in top-K) / (# of total relevant docs)
    """
    if not relevant_ids:
        return 0.0
    topk = retrieved_ids[:k]
    relevant_in_topk = sum(1 for r in topk if r in relevant_ids)
    return relevant_in_topk / len(relevant_ids)


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank.
    
    MRR = 1 / rank of first relevant document
    """
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Calculate NDCG@K (simplified binary relevance).
    """
    import math
    
    if not relevant_ids or k <= 0:
        return 0.0
    
    topk = retrieved_ids[:k]
    
    # DCG
    dcg = 0.0
    for i, rid in enumerate(topk, start=1):
        if rid in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)
    
    # Ideal DCG
    ideal_k = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_k + 1))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def evaluate(dataset_path: str, top_k: int = 10, use_reranker: bool = False):
    """
    Run evaluation on a dataset of queries.
    
    Args:
        dataset_path: Path to JSON file with queries
        top_k: K value for metrics
        use_reranker: Whether to use cross-encoder reranking
    """
    with open(dataset_path) as f:
        data = json.load(f)
    
    if not data:
        print("No queries found in dataset")
        return
    
    pks = []
    rks = []
    mrrs = []
    ndcgs = []
    
    print(f"Evaluating {len(data)} queries with k={top_k}")
    print("-" * 50)
    
    for i, q in enumerate(data):
        query = q.get('query', '')
        relevant = q.get('relevant_ids', [])
        query_id = q.get('id', f'q{i}')
        
        if not query:
            print(f"Skipping query {query_id}: empty query")
            continue
        
        # Perform retrieval
        results = hybrid_retrieve(
            query, 
            top_k_text=top_k // 2, 
            top_k_image=top_k // 2,
            use_reranker=use_reranker
        )
        
        retrieved_ids = [r['id'] for r in results]
        
        # Calculate metrics
        p = precision_at_k(retrieved_ids, relevant, top_k)
        r = recall_at_k(retrieved_ids, relevant, top_k)
        m = mrr(retrieved_ids, relevant)
        n = ndcg_at_k(retrieved_ids, relevant, top_k)
        
        pks.append(p)
        rks.append(r)
        mrrs.append(m)
        ndcgs.append(n)
        
        print(f"  [{query_id}] P@{top_k}={p:.3f}, R@{top_k}={r:.3f}, MRR={m:.3f}, NDCG@{top_k}={n:.3f}")
    
    print("-" * 50)
    print(f"Average Metrics (n={len(pks)}):")
    print(f"  P@{top_k}:    {np.mean(pks):.4f} ± {np.std(pks):.4f}")
    print(f"  R@{top_k}:    {np.mean(rks):.4f} ± {np.std(rks):.4f}")
    print(f"  MRR:       {np.mean(mrrs):.4f} ± {np.std(mrrs):.4f}")
    print(f"  NDCG@{top_k}: {np.mean(ndcgs):.4f} ± {np.std(ndcgs):.4f}")


def create_sample_dataset(output_path: str):
    """Create a sample evaluation dataset for testing."""
    sample = [
        {
            "id": "q1",
            "query": "What are the symptoms of diabetes?",
            "relevant_ids": ["text::diabetes_doc::0", "text::diabetes_doc::1"]
        },
        {
            "id": "q2", 
            "query": "Show me an X-ray of a healthy lung",
            "relevant_ids": ["image::lung_xray_healthy"]
        }
    ]
    
    with open(output_path, 'w') as f:
        json.dump(sample, f, indent=2)
    
    print(f"Sample dataset created at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval performance")
    parser.add_argument(
        "--dataset", 
        required=False, 
        help="Path to JSON queries file"
    )
    parser.add_argument(
        "--k", 
        type=int, 
        default=10,
        help="K value for metrics (default: 10)"
    )
    parser.add_argument(
        "--reranker",
        action="store_true",
        help="Use cross-encoder reranking"
    )
    parser.add_argument(
        "--create-sample",
        type=str,
        help="Create a sample dataset at the given path"
    )
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_dataset(args.create_sample)
    elif args.dataset:
        evaluate(args.dataset, args.k, args.reranker)
    else:
        print("Please provide --dataset or --create-sample")
        parser.print_help()