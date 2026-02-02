"""
Test: End-to-End Validation of Paper-Rule Matching

This test:
1. Takes known papers with known content
2. Verifies that rule matches are CORRECT (not just high similarity)
3. Documents false positives and false negatives
4. Checks if embeddings actually differentiate between topics

CRITICAL FINDING:
- BioBERT embeddings in high-dimensional space have baseline similarity ~0.90
- This is NOT a bug - it's how sentence embeddings work
- The DIFFERENCE between pos_sim and neg_sim matters, not absolute values
- Thresholds need to be adjusted (e.g., 0.92 instead of 0.70)

To run:
    docker exec paperstream python /app/tests/test_content_validation.py
"""

import sqlite3
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass


@dataclass
class GroundTruth:
    """Expected results for a paper"""
    paper_id: str
    expected_matches: List[str]    # Rules that SHOULD match
    expected_non_matches: List[str]  # Rules that should NOT match
    reason: str


# Ground truth for our test papers
GROUND_TRUTH = [
    GroundTruth(
        paper_id="DOI:10.1057/s41599-025-06099-7",
        expected_matches=[],  # This is about AI and tax - NOT a medical study!
        expected_non_matches=[
            "is_rct",           # NOT a randomized controlled trial
            "has_placebo",      # NO placebo
            "is_human_study",   # NOT a human medical study
            "has_blinding",     # NO blinding
            "is_meta_analysis", # NOT a meta-analysis
            "reports_sample_size",  # NOT a clinical study
        ],
        reason="Paper about AI in tax administration - not medical research"
    ),
    GroundTruth(
        paper_id="2601.16040v1",
        expected_matches=[
            "is_human_study",   # Social media study with human participants
            "reports_sample_size",  # Has sample size (8,535 comments, 2,470 posts)
        ],
        expected_non_matches=[
            "is_rct",           # NOT a medical RCT
            "has_placebo",      # NO medical placebo
            "is_blinded",       # Social media users know they're using it
            "is_meta_analysis", # NOT a meta-analysis
        ],
        reason="Social media platform study - human participants but not medical"
    ),
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class ContentValidator:
    """Validates that paper embeddings correctly match rules"""
    
    def __init__(self, db_path: str = "/app/data/paperstream.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        
    def get_paper_embedding(self, paper_id: str) -> np.ndarray:
        """Get paper's abstract embedding"""
        row = self.conn.execute("""
            SELECT embedding FROM paper_sections
            WHERE paper_id = ? AND embedding IS NOT NULL
            ORDER BY CASE section_name 
                WHEN 'abstract' THEN 1 
                WHEN 'introduction' THEN 2 
                ELSE 3 END
            LIMIT 1
        """, (paper_id,)).fetchone()
        
        if not row:
            return None
        return np.frombuffer(row["embedding"], dtype=np.float32)
    
    def get_rules(self) -> Dict[str, Dict]:
        """Get all rules with embeddings"""
        rows = self.conn.execute("""
            SELECT rule_id, question, pos_embedding, neg_embedding, threshold
            FROM rules
        """).fetchall()
        
        rules = {}
        for row in rows:
            rules[row["rule_id"]] = {
                "question": row["question"],
                "pos": np.frombuffer(row["pos_embedding"], dtype=np.float32) if row["pos_embedding"] else None,
                "neg": np.frombuffer(row["neg_embedding"], dtype=np.float32) if row["neg_embedding"] else None,
                "threshold": row["threshold"] or 0.5
            }
        return rules
    
    def evaluate_paper(self, paper_id: str) -> Dict[str, Any]:
        """Evaluate all rules against a paper"""
        paper_emb = self.get_paper_embedding(paper_id)
        if paper_emb is None:
            return {"error": f"No embedding for {paper_id}"}
        
        rules = self.get_rules()
        results = {}
        
        for rule_id, rule in rules.items():
            if rule["pos"] is None:
                continue
                
            pos_sim = cosine_similarity(paper_emb, rule["pos"])
            neg_sim = 0.0
            if rule["neg"] is not None:
                neg_sim = cosine_similarity(paper_emb, rule["neg"])
            
            # Standard matching: pos > threshold AND pos > neg
            standard_match = pos_sim > rule["threshold"] and pos_sim > neg_sim
            
            # Delta-based matching: (pos - neg) > margin
            delta = pos_sim - neg_sim
            delta_match = delta > 0.02  # More than 2% difference
            
            results[rule_id] = {
                "pos_sim": pos_sim,
                "neg_sim": neg_sim,
                "delta": delta,
                "threshold": rule["threshold"],
                "standard_match": standard_match,
                "delta_match": delta_match,
            }
        
        return results
    
    def validate_ground_truth(self, gt: GroundTruth) -> Dict[str, Any]:
        """Validate paper against ground truth"""
        results = self.evaluate_paper(gt.paper_id)
        
        if "error" in results:
            return results
        
        true_positives = []
        false_positives = []
        true_negatives = []
        false_negatives = []
        
        for rule_id, r in results.items():
            matched = r["standard_match"]
            
            if rule_id in gt.expected_matches:
                if matched:
                    true_positives.append(rule_id)
                else:
                    false_negatives.append(rule_id)
            elif rule_id in gt.expected_non_matches:
                if matched:
                    false_positives.append(rule_id)
                else:
                    true_negatives.append(rule_id)
        
        return {
            "paper_id": gt.paper_id,
            "reason": gt.reason,
            "results": results,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "accuracy": {
                "tp": len(true_positives),
                "fp": len(false_positives),
                "tn": len(true_negatives),
                "fn": len(false_negatives),
            }
        }


def main():
    validator = ContentValidator()
    
    print("=" * 80)
    print("CONTENT VALIDATION TEST")
    print("Testing if embeddings correctly identify paper characteristics")
    print("=" * 80)
    
    total_fp = 0
    total_fn = 0
    total_tp = 0
    total_tn = 0
    
    for gt in GROUND_TRUTH:
        print(f"\n--- Paper: {gt.paper_id} ---")
        print(f"Description: {gt.reason}")
        
        validation = validator.validate_ground_truth(gt)
        
        if "error" in validation:
            print(f"ERROR: {validation['error']}")
            continue
        
        acc = validation["accuracy"]
        total_tp += acc["tp"]
        total_fp += acc["fp"]
        total_tn += acc["tn"]
        total_fn += acc["fn"]
        
        print(f"\nResults (threshold-based matching):")
        print(f"  True Positives:  {acc['tp']} - {validation['true_positives']}")
        print(f"  False Positives: {acc['fp']} - {validation['false_positives']}")
        print(f"  True Negatives:  {acc['tn']} - {validation['true_negatives']}")
        print(f"  False Negatives: {acc['fn']} - {validation['false_negatives']}")
        
        if validation["false_positives"]:
            print(f"\n  ⚠️ FALSE POSITIVES - Rules matched that shouldn't:")
            for rule_id in validation["false_positives"]:
                r = validation["results"][rule_id]
                print(f"     {rule_id}: pos={r['pos_sim']:.3f}, neg={r['neg_sim']:.3f}, "
                      f"delta={r['delta']:.3f}, thresh={r['threshold']:.2f}")
        
        if validation["false_negatives"]:
            print(f"\n  ⚠️ FALSE NEGATIVES - Rules didn't match that should:")
            for rule_id in validation["false_negatives"]:
                r = validation["results"][rule_id]
                print(f"     {rule_id}: pos={r['pos_sim']:.3f}, neg={r['neg_sim']:.3f}, "
                      f"delta={r['delta']:.3f}, thresh={r['threshold']:.2f}")
        
        # Show similarity distribution
        print(f"\n  Similarity Distribution (all rules):")
        for rule_id in ["is_rct", "has_placebo", "is_human_study", "reports_sample_size"]:
            if rule_id in validation["results"]:
                r = validation["results"][rule_id]
                status = "✓" if r["standard_match"] else "✗"
                print(f"     {rule_id:25} | pos={r['pos_sim']:.3f} | neg={r['neg_sim']:.3f} | "
                      f"Δ={r['delta']:+.3f} | {status}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total True Positives:  {total_tp}")
    print(f"Total False Positives: {total_fp}")
    print(f"Total True Negatives:  {total_tn}")
    print(f"Total False Negatives: {total_fn}")
    
    if total_fp > 0:
        print(f"\n⚠️ PROBLEM: {total_fp} false positives detected!")
        print("   This means rules are matching papers they shouldn't.")
        print("   Possible fixes:")
        print("   1. Increase thresholds (e.g., from 0.70 to 0.92)")
        print("   2. Use delta-based matching (pos-neg > margin)")
        print("   3. Re-train embeddings with contrastive learning")
    
    # Recommendation
    print("\n" + "-" * 40)
    print("RECOMMENDATION:")
    print("-" * 40)
    print("BioBERT embeddings have high baseline similarity (~0.90).")
    print("The current threshold-based matching is too permissive.")
    print("")
    print("Option 1: Raise thresholds to 0.92+")
    print("Option 2: Use delta matching: match if (pos - neg) > 0.02")
    print("Option 3: Both - high threshold AND positive delta")


if __name__ == "__main__":
    main()
