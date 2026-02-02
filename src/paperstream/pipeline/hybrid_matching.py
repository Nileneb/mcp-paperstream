"""
HYBRID Matching Algorithm: Embedding + Keyword
=============================================

BioBERT embeddings alone don't discriminate well between RCT vs non-RCT.
This hybrid approach combines:
1. Embedding similarity (semantic)
2. Keyword presence (exact matching)

The final match score considers BOTH factors.
"""

import re
from typing import Dict, List, Tuple
import numpy as np


# Keywords that MUST appear for certain rules (case-insensitive)
RULE_KEYWORDS = {
    "is_rct": {
        "positive": [
            r"randomized\s+controlled\s+trial",
            r"randomised\s+controlled\s+trial",
            r"\bRCT\b",
            r"randomly\s+assigned",
            r"random\s+allocation",
            r"randomization",
            r"randomised",
            r"randomized\s+trial",
        ],
        "negative": [
            r"observational\s+study",
            r"retrospective",
            r"non-randomized",
            r"cohort\s+study",
            r"case\s+series",
            r"systematic\s+review",
            r"meta-analysis",
        ],
        "keyword_weight": 0.7,  # Keywords contribute 70% to match decision
    },
    
    "has_placebo": {
        "positive": [
            r"placebo[- ]controlled",
            r"placebo\s+group",
            r"received\s+placebo",
            r"versus\s+placebo",
            r"sham\s+procedure",
            r"sham\s+control",
        ],
        "negative": [
            r"no\s+placebo",
            r"open[- ]label",
            r"active\s+control",
            r"active\s+comparator",
        ],
        "keyword_weight": 0.8,
    },
    
    "is_blinded": {
        "positive": [
            r"double[- ]blind",
            r"single[- ]blind",
            r"triple[- ]blind",
            r"blinded\s+study",
            r"masked\s+study",
            r"participants\s+were\s+blinded",
        ],
        "negative": [
            r"open[- ]label",
            r"unblinded",
            r"not\s+blinded",
            r"no\s+blinding",
        ],
        "keyword_weight": 0.8,
    },
    
    "is_human_study": {
        "positive": [
            r"human\s+subjects",
            r"participants",
            r"patients",
            r"volunteers",
            r"enrolled",
            r"recruited",
            r"human\s+study",
        ],
        "negative": [
            r"animal\s+model",
            r"mouse\s+model",
            r"in\s+vitro",
            r"cell\s+culture",
            r"computational\s+study",
        ],
        "keyword_weight": 0.5,  # Keywords less important for this rule
    },
    
    "has_control_group": {
        "positive": [
            r"control\s+group",
            r"control\s+arm",
            r"controlled\s+study",
            r"compared\s+to\s+control",
            r"placebo\s+group",
        ],
        "negative": [
            r"no\s+control",
            r"uncontrolled",
            r"single[- ]arm",
            r"case\s+series",
        ],
        "keyword_weight": 0.6,
    },
    
    "is_meta_analysis": {
        "positive": [
            r"meta[- ]analysis",
            r"systematic\s+review",
            r"pooled\s+analysis",
            r"PRISMA",
            r"literature\s+review",
        ],
        "negative": [
            r"original\s+research",
            r"clinical\s+trial",
            r"randomized\s+trial",
        ],
        "keyword_weight": 0.9,  # Keywords very important for meta-analysis
    },
}


def count_keyword_matches(text: str, patterns: List[str]) -> int:
    """Count how many keyword patterns match in the text"""
    text_lower = text.lower()
    count = 0
    for pattern in patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            count += 1
    return count


def hybrid_match(
    rule_id: str,
    text: str,
    pos_sim: float,
    neg_sim: float,
    embedding_delta_threshold: float = 0.01
) -> Tuple[bool, float, Dict]:
    """
    Hybrid matching using embedding similarity AND keywords.
    
    Args:
        rule_id: The rule being tested
        text: The paper text
        pos_sim: Cosine similarity with positive embedding
        neg_sim: Cosine similarity with negative embedding
        embedding_delta_threshold: Minimum delta for embedding-based match
    
    Returns:
        (matched, confidence, details)
    """
    if rule_id not in RULE_KEYWORDS:
        # Fall back to pure embedding matching
        delta = pos_sim - neg_sim
        matched = delta > embedding_delta_threshold
        return matched, delta, {"method": "embedding_only"}
    
    rule = RULE_KEYWORDS[rule_id]
    keyword_weight = rule.get("keyword_weight", 0.5)
    embedding_weight = 1.0 - keyword_weight
    
    # Keyword matching
    pos_keywords = count_keyword_matches(text, rule["positive"])
    neg_keywords = count_keyword_matches(text, rule["negative"])
    
    # Keyword score: positive if more positive than negative keywords
    if pos_keywords + neg_keywords > 0:
        keyword_score = (pos_keywords - neg_keywords) / max(pos_keywords + neg_keywords, 1)
    else:
        keyword_score = 0.0
    
    # Embedding score: delta between pos and neg similarity
    embedding_delta = pos_sim - neg_sim
    embedding_score = embedding_delta * 10  # Scale to similar range as keyword score
    
    # Combined score
    combined_score = (
        keyword_weight * keyword_score +
        embedding_weight * embedding_score
    )
    
    # Match if combined score is positive
    matched = combined_score > 0 and (pos_keywords > neg_keywords or embedding_delta > 0)
    
    # Confidence based on how strong the match is
    confidence = min(1.0, max(0.0, combined_score))
    
    return matched, confidence, {
        "method": "hybrid",
        "keyword_weight": keyword_weight,
        "pos_keywords": pos_keywords,
        "neg_keywords": neg_keywords,
        "keyword_score": keyword_score,
        "embedding_delta": embedding_delta,
        "embedding_score": embedding_score,
        "combined_score": combined_score,
    }


def test_hybrid():
    """Test the hybrid matching"""
    
    rct_text = """
    Background: We conducted a randomized controlled trial to evaluate the efficacy.
    Methods: This was a double-blind, placebo-controlled study. Participants were 
    randomly assigned to receive either the intervention (n=150) or placebo (n=148).
    Results: The primary endpoint was met. The intention-to-treat analysis showed
    significant improvement in the treatment group (p<0.001).
    Conclusions: This randomized trial demonstrates efficacy of the treatment.
    """
    
    non_rct_text = """
    This observational cohort study examined the relationship between diet and health.
    We retrospectively analyzed medical records from 500 patients over 5 years.
    No randomization was performed. Results showed correlation but not causation.
    This was a descriptive analysis without a control group.
    """
    
    print("=" * 80)
    print("HYBRID MATCHING TEST")
    print("=" * 80)
    
    # Simulate embeddings (we don't have BioBERT here, so use fake deltas)
    # In reality, you'd compute actual embeddings
    
    print("\n--- RCT Text ---")
    for rule_id in ["is_rct", "has_placebo", "is_blinded", "is_human_study"]:
        # Fake embedding deltas (small positive, as we saw in real tests)
        matched, conf, details = hybrid_match(
            rule_id, rct_text,
            pos_sim=0.88, neg_sim=0.88,  # Nearly equal (realistic)
            embedding_delta_threshold=0.01
        )
        print(f"{rule_id:25} | matched={matched} | conf={conf:.2f} | {details.get('pos_keywords', 0)} pos keywords")
    
    print("\n--- Non-RCT Text ---")
    for rule_id in ["is_rct", "has_placebo", "is_blinded", "is_human_study"]:
        matched, conf, details = hybrid_match(
            rule_id, non_rct_text,
            pos_sim=0.92, neg_sim=0.93,  # Negative delta (realistic)
            embedding_delta_threshold=0.01
        )
        print(f"{rule_id:25} | matched={matched} | conf={conf:.2f} | {details.get('neg_keywords', 0)} neg keywords")


if __name__ == "__main__":
    test_hybrid()
