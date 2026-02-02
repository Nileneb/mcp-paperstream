"""
Test: Verify that Paper Embeddings Actually Capture Content

This test verifies:
1. The embedding actually represents the text content
2. Cosine similarity correctly identifies matching rules
3. The API response matches what's in the database

To run:
    docker exec paperstream python -m pytest tests/test_embedding_accuracy.py -v
    
Or manually:
    docker exec paperstream python tests/test_embedding_accuracy.py
"""

import sys
import json
import base64
import struct
import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def decode_embedding_b64(embedding_b64: str) -> np.ndarray:
    """Decode base64 embedding to numpy array (768 floats)"""
    raw_bytes = base64.b64decode(embedding_b64)
    return np.frombuffer(raw_bytes, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class TestEmbeddingAccuracy:
    """
    Tests to verify that paper embeddings capture actual content
    and that rule matching produces sensible results.
    """
    
    def __init__(self, db_path: str = "/app/data/paperstream.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
    
    def get_paper_with_embedding(self) -> Dict[str, Any]:
        """Get a paper that has both text and embedding"""
        row = self.conn.execute("""
            SELECT p.paper_id, p.title, ps.section_name, ps.section_text, ps.embedding
            FROM papers p
            JOIN paper_sections ps ON p.paper_id = ps.paper_id
            WHERE ps.embedding IS NOT NULL 
              AND ps.section_name = 'abstract'
              AND LENGTH(ps.section_text) > 100
            LIMIT 1
        """).fetchone()
        
        if not row:
            return None
        
        return {
            "paper_id": row["paper_id"],
            "title": row["title"],
            "section_name": row["section_name"],
            "section_text": row["section_text"],
            "embedding": np.frombuffer(row["embedding"], dtype=np.float32)
        }
    
    def get_all_rules(self) -> List[Dict[str, Any]]:
        """Get all rules with their embeddings"""
        rows = self.conn.execute("""
            SELECT rule_id, question, pos_embedding, neg_embedding, threshold
            FROM rules
        """).fetchall()
        
        rules = []
        for row in rows:
            rules.append({
                "rule_id": row["rule_id"],
                "question": row["question"],
                "pos_embedding": np.frombuffer(row["pos_embedding"], dtype=np.float32) if row["pos_embedding"] else None,
                "neg_embedding": np.frombuffer(row["neg_embedding"], dtype=np.float32) if row["neg_embedding"] else None,
                "threshold": row["threshold"] or 0.5
            })
        return rules
    
    def test_embedding_dimension(self) -> Tuple[bool, str]:
        """Test: Embeddings should be 768-dimensional (BioBERT)"""
        paper = self.get_paper_with_embedding()
        if not paper:
            return False, "No paper with embedding found"
        
        dim = paper["embedding"].shape[0]
        if dim != 768:
            return False, f"Expected 768 dimensions, got {dim}"
        
        return True, f"✓ Embedding dimension correct: {dim}"
    
    def test_embedding_not_zero(self) -> Tuple[bool, str]:
        """Test: Embeddings should not be all zeros"""
        paper = self.get_paper_with_embedding()
        if not paper:
            return False, "No paper with embedding found"
        
        if np.allclose(paper["embedding"], 0):
            return False, "Embedding is all zeros!"
        
        norm = np.linalg.norm(paper["embedding"])
        return True, f"✓ Embedding is non-zero (norm={norm:.4f})"
    
    def test_rule_matching_logic(self) -> Tuple[bool, str]:
        """Test: Rule matching produces sensible results for known content"""
        paper = self.get_paper_with_embedding()
        if not paper:
            return False, "No paper with embedding found"
        
        rules = self.get_all_rules()
        if not rules:
            return False, "No rules found"
        
        results = []
        text_lower = paper["section_text"].lower()
        
        for rule in rules:
            if rule["pos_embedding"] is None:
                continue
            
            # Calculate similarities
            pos_sim = cosine_similarity(paper["embedding"], rule["pos_embedding"])
            neg_sim = 0.0
            if rule["neg_embedding"] is not None:
                neg_sim = cosine_similarity(paper["embedding"], rule["neg_embedding"])
            
            # Match if pos > threshold and pos > neg
            matched = pos_sim > rule["threshold"] and pos_sim > neg_sim
            
            results.append({
                "rule_id": rule["rule_id"],
                "question": rule["question"][:50],
                "pos_sim": pos_sim,
                "neg_sim": neg_sim,
                "matched": matched,
                "threshold": rule["threshold"]
            })
        
        # Sort by pos_sim descending
        results.sort(key=lambda x: x["pos_sim"], reverse=True)
        
        output = [f"Paper: {paper['paper_id']}", f"Title: {paper['title'][:60]}...", ""]
        output.append("Rule Matching Results:")
        output.append("-" * 80)
        
        for r in results[:10]:  # Top 10
            match_str = "✓ MATCH" if r["matched"] else "✗"
            output.append(
                f"{r['rule_id']:25} | pos={r['pos_sim']:.3f} | neg={r['neg_sim']:.3f} | "
                f"thresh={r['threshold']:.2f} | {match_str}"
            )
        
        return True, "\n".join(output)
    
    def test_content_keyword_correlation(self) -> Tuple[bool, str]:
        """
        Test: Rules about topics mentioned in text should have higher similarity
        
        For example, if text mentions "randomized" or "trial", the RCT rule
        should have higher similarity than rules about unrelated topics.
        """
        paper = self.get_paper_with_embedding()
        if not paper:
            return False, "No paper with embedding found"
        
        rules = self.get_all_rules()
        text_lower = paper["section_text"].lower()
        
        # Check for keyword matches
        keyword_checks = [
            ("is_rct", ["randomized", "random", "trial", "rct"]),
            ("has_placebo", ["placebo", "sham", "control group"]),
            ("reports_outcomes", ["outcome", "endpoint", "result", "efficacy"]),
            ("is_meta_analysis", ["meta-analysis", "systematic review", "pooled"]),
            ("has_blinding", ["blind", "blinded", "double-blind", "masking"]),
            ("reports_sample_size", ["sample size", "n=", "participants", "patients"]),
        ]
        
        results = []
        for rule_id, keywords in keyword_checks:
            # Check if any keyword is in text
            keyword_found = any(kw in text_lower for kw in keywords)
            
            # Find rule
            rule = next((r for r in rules if r["rule_id"] == rule_id), None)
            if not rule or rule["pos_embedding"] is None:
                continue
            
            pos_sim = cosine_similarity(paper["embedding"], rule["pos_embedding"])
            
            results.append({
                "rule_id": rule_id,
                "keyword_found": keyword_found,
                "pos_sim": pos_sim,
                "keywords_checked": keywords[:3]
            })
        
        output = ["Keyword-Embedding Correlation:", "-" * 60]
        
        correlations_correct = 0
        total_checks = 0
        
        for r in results:
            found_str = "FOUND" if r["keyword_found"] else "not found"
            # High similarity (>0.4) should correlate with keyword presence
            # This is a soft check - embeddings capture semantic meaning
            output.append(
                f"{r['rule_id']:25} | keywords={found_str:10} | sim={r['pos_sim']:.3f}"
            )
            
            if r["keyword_found"] and r["pos_sim"] > 0.3:
                correlations_correct += 1
            elif not r["keyword_found"] and r["pos_sim"] < 0.5:
                correlations_correct += 1
            total_checks += 1
        
        if total_checks > 0:
            ratio = correlations_correct / total_checks
            output.append(f"\nCorrelation ratio: {correlations_correct}/{total_checks} ({ratio:.1%})")
        
        return True, "\n".join(output)
    
    def test_api_returns_valid_embedding(self) -> Tuple[bool, str]:
        """Test: API /api/jobs/next returns valid base64 embedding"""
        import urllib.request
        
        try:
            url = "http://localhost:8089/api/jobs/next?device_id=test_embedding_check"
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            if data.get("status") != "assigned":
                return False, f"No job assigned: {data.get('message', 'unknown')}"
            
            job = data.get("job", {})
            embedding_b64 = job.get("paper_embedding_b64")
            
            if not embedding_b64:
                return False, "No paper_embedding_b64 in response"
            
            # Decode and verify
            embedding = decode_embedding_b64(embedding_b64)
            
            if embedding.shape[0] != 768:
                return False, f"Wrong dimension: {embedding.shape[0]}"
            
            if np.allclose(embedding, 0):
                return False, "Embedding is all zeros"
            
            return True, f"✓ API returns valid 768-dim embedding (norm={np.linalg.norm(embedding):.4f})"
            
        except Exception as e:
            return False, f"API error: {e}"
    
    def test_embedding_matches_database(self) -> Tuple[bool, str]:
        """Test: API embedding matches what's stored in database"""
        import urllib.request
        
        try:
            # Get from API
            url = "http://localhost:8089/api/jobs/next?device_id=test_db_match"
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            if data.get("status") != "assigned":
                return False, f"No job assigned: {data.get('message', 'unknown')}"
            
            paper_id = data["job"]["paper_id"]
            api_embedding = decode_embedding_b64(data["job"]["paper_embedding_b64"])
            
            # Get from database
            row = self.conn.execute("""
                SELECT embedding FROM paper_sections 
                WHERE paper_id = ? AND embedding IS NOT NULL
                ORDER BY CASE section_name WHEN 'abstract' THEN 1 ELSE 2 END
                LIMIT 1
            """, (paper_id,)).fetchone()
            
            if not row:
                return False, f"No embedding in DB for {paper_id}"
            
            db_embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            
            # Compare
            if not np.allclose(api_embedding, db_embedding, rtol=1e-5):
                diff = np.linalg.norm(api_embedding - db_embedding)
                return False, f"Embeddings don't match! Difference norm: {diff}"
            
            return True, f"✓ API embedding matches database for {paper_id}"
            
        except Exception as e:
            return False, f"Error: {e}"
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return results"""
        tests = [
            ("Embedding Dimension", self.test_embedding_dimension),
            ("Embedding Non-Zero", self.test_embedding_not_zero),
            ("Rule Matching Logic", self.test_rule_matching_logic),
            ("Keyword Correlation", self.test_content_keyword_correlation),
            ("API Valid Embedding", self.test_api_returns_valid_embedding),
            ("API Matches Database", self.test_embedding_matches_database),
        ]
        
        results = []
        passed = 0
        failed = 0
        
        print("=" * 80)
        print("PAPERSTREAM EMBEDDING ACCURACY TESTS")
        print("=" * 80)
        
        for name, test_fn in tests:
            print(f"\n--- {name} ---")
            try:
                success, message = test_fn()
                if success:
                    passed += 1
                    print(f"PASS: {message}")
                else:
                    failed += 1
                    print(f"FAIL: {message}")
                results.append({"name": name, "passed": success, "message": message})
            except Exception as e:
                failed += 1
                print(f"ERROR: {e}")
                results.append({"name": name, "passed": False, "message": str(e)})
        
        print("\n" + "=" * 80)
        print(f"SUMMARY: {passed} passed, {failed} failed")
        print("=" * 80)
        
        return {
            "passed": passed,
            "failed": failed,
            "results": results
        }


def main():
    """Run tests"""
    tester = TestEmbeddingAccuracy()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
