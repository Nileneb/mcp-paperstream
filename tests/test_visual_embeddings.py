"""
Test Visual Embeddings - Generate sample voxel data for Unity testing

This script:
1. Loads BioBERT model
2. Generates embeddings for different paper types
3. Converts to voxel grids
4. Exports as JSON for Unity
"""

import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Check for transformers
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("WARNING: transformers not installed. Using mock embeddings.")
    print("Install with: pip install transformers torch")

from paperstream.core.data_model import (
    create_rule_molecule,
    create_paper_molecule,
    embedding_to_voxel_grid,
    enhance_visual_contrast,
    Molecule,
    VoxelGrid,
    VOXEL_TOTAL
)


# Sample texts for different paper types
SAMPLE_TEXTS = {
    "rct_abstract": """
    Background: Randomized controlled trials (RCTs) are the gold standard for evaluating 
    treatment efficacy. This study aims to evaluate the effectiveness of a new intervention.
    Methods: We conducted a double-blind, placebo-controlled randomized trial with 500 
    participants. Patients were randomly assigned to treatment (n=250) or control (n=250) groups.
    Primary outcome was measured at 12 weeks.
    Results: The treatment group showed significant improvement compared to placebo 
    (p<0.001, 95% CI: 1.2-2.4).
    Conclusion: The intervention demonstrates statistically significant efficacy.
    """,
    
    "review_abstract": """
    This systematic review and meta-analysis examines the current evidence on treatment 
    approaches for the condition. We searched PubMed, Cochrane Library, and EMBASE databases.
    A total of 45 studies were included in the qualitative synthesis, with 28 studies 
    included in the meta-analysis. The pooled effect size was moderate (d=0.45, 95% CI: 0.32-0.58).
    Heterogeneity was substantial (I²=67%). Publication bias was assessed using funnel plots.
    Evidence quality was evaluated using GRADE criteria.
    """,
    
    "case_study": """
    We present a rare case of a 45-year-old male presenting with unusual symptoms.
    The patient reported a 3-month history of progressive weakness. Physical examination 
    revealed specific clinical signs. Laboratory findings showed elevated markers.
    Imaging studies demonstrated characteristic features. The patient was treated 
    conservatively with monitoring. At 6-month follow-up, significant improvement was observed.
    This case highlights the importance of early diagnosis and management.
    """,
    
    "methods_section": """
    Study Design: This was a prospective cohort study conducted at three tertiary care centers.
    Participants: Adults aged 18-65 years with confirmed diagnosis were eligible.
    Exclusion criteria included pregnancy, severe comorbidities, and prior treatment.
    Intervention: Participants received standardized treatment protocol for 8 weeks.
    Outcomes: Primary endpoint was clinical improvement at week 8. Secondary endpoints 
    included safety and tolerability measures.
    Statistical Analysis: Sample size was calculated based on expected effect size of 0.5.
    """
}

# Rule definitions for testing
RULE_DEFINITIONS = {
    "is_rct": {
        "question": "Is this a Randomized Controlled Trial?",
        "positive_phrases": [
            "randomized controlled trial", "RCT", "double-blind", "placebo-controlled",
            "randomly assigned", "random allocation", "randomization"
        ],
        "negative_phrases": [
            "systematic review", "meta-analysis", "case report", "case series",
            "observational study", "retrospective", "cohort study"
        ]
    },
    "has_methods": {
        "question": "Does this paper have a clear methods section?",
        "positive_phrases": [
            "study design", "participants", "methods", "statistical analysis",
            "inclusion criteria", "exclusion criteria", "sample size"
        ],
        "negative_phrases": [
            "opinion", "editorial", "commentary", "letter to editor",
            "no methods described", "unclear methodology"
        ]
    }
}


class EmbeddingGenerator:
    """Generates BioBERT embeddings or mock embeddings"""
    
    def __init__(self, use_biobert: bool = True):
        self.model = None
        self.tokenizer = None
        self.use_biobert = use_biobert and HAS_TRANSFORMERS
        
        if self.use_biobert:
            self._load_model()
    
    def _load_model(self):
        """Load BioBERT model"""
        print("Loading BioBERT model...")
        model_name = "dmis-lab/biobert-base-cased-v1.2"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            print("✓ BioBERT loaded successfully")
        except Exception as e:
            print(f"Failed to load BioBERT: {e}")
            print("Falling back to mock embeddings")
            self.use_biobert = False
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if self.use_biobert:
            return self._biobert_embedding(text)
        else:
            return self._mock_embedding(text)
    
    def _biobert_embedding(self, text: str) -> np.ndarray:
        """Generate real BioBERT embedding"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        
        return embedding.astype(np.float32)
    
    def _mock_embedding(self, text: str) -> np.ndarray:
        """Generate deterministic mock embedding based on text"""
        # Use text hash as seed for reproducibility
        seed = hash(text) % (2**32)
        rng = np.random.RandomState(seed)
        
        # Generate embedding with some structure
        embedding = rng.randn(768).astype(np.float32)
        
        # Add text-length dependent bias
        length_factor = len(text) / 1000.0
        embedding[:256] += length_factor * 0.5
        
        return embedding


def generate_test_data(output_dir: Path, use_biobert: bool = True):
    """Generate test data for Unity"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = EmbeddingGenerator(use_biobert=use_biobert)
    
    # Generate paper embeddings
    print("\n--- Generating Paper Embeddings ---")
    papers = {}
    
    for paper_type, text in SAMPLE_TEXTS.items():
        print(f"Processing: {paper_type}")
        embedding = generator.get_embedding(text)
        voxels = embedding_to_voxel_grid(embedding, threshold=0.3)
        
        papers[paper_type] = {
            "text_preview": text[:200] + "...",
            "embedding_stats": {
                "mean": float(embedding.mean()),
                "std": float(embedding.std()),
                "min": float(embedding.min()),
                "max": float(embedding.max())
            },
            "voxel_count": len(voxels),
            "fill_ratio": len(voxels) / VOXEL_TOTAL
        }
        
        # Create a simple molecule for each paper
        mol = create_paper_molecule(
            paper_id=paper_type,
            title=paper_type.replace("_", " ").title(),
            sections=[{
                "name": "abstract" if "abstract" in paper_type else "body",
                "text": text,
                "embedding": embedding
            }]
        )
        
        # Save individual paper
        paper_file = output_dir / f"paper_{paper_type}.json"
        with open(paper_file, "w") as f:
            json.dump(mol.to_dict(), f, indent=2)
        print(f"  → Saved: {paper_file.name} ({len(voxels)} voxels)")
    
    # Generate rule embeddings
    print("\n--- Generating Rule Embeddings ---")
    rules = {}
    
    for rule_id, rule_def in RULE_DEFINITIONS.items():
        print(f"Processing rule: {rule_id}")
        
        # Combine phrases for embedding
        pos_text = " ".join(rule_def["positive_phrases"])
        neg_text = " ".join(rule_def["negative_phrases"])
        
        pos_emb = generator.get_embedding(pos_text)
        neg_emb = generator.get_embedding(neg_text)
        
        mol = create_rule_molecule(
            rule_id=rule_id,
            question=rule_def["question"],
            pos_embedding=pos_emb,
            neg_embedding=neg_emb,
            pos_text=pos_text,
            neg_text=neg_text
        )
        
        rules[rule_id] = {
            "question": rule_def["question"],
            "positive_voxels": len(mol.chunks[0].voxels.voxels),
            "negative_voxels": len(mol.chunks[1].voxels.voxels) if len(mol.chunks) > 1 else 0
        }
        
        # Save individual rule
        rule_file = output_dir / f"rule_{rule_id}.json"
        with open(rule_file, "w") as f:
            json.dump(mol.to_dict(), f, indent=2)
        print(f"  → Saved: {rule_file.name}")
    
    # Save combined test file
    combined = {
        "version": "0.2.0",
        "contract": "DATAMODEL.md",
        "generated_with": "BioBERT" if generator.use_biobert else "Mock",
        "papers": papers,
        "rules": rules
    }
    
    combined_file = output_dir / "sample_voxels.json"
    with open(combined_file, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n✓ Combined file: {combined_file}")
    
    # Generate comparison data (same text, different thresholds)
    print("\n--- Threshold Comparison ---")
    test_text = SAMPLE_TEXTS["rct_abstract"]
    test_emb = generator.get_embedding(test_text)
    
    thresholds = [0.2, 0.3, 0.4, 0.5]
    for t in thresholds:
        voxels = embedding_to_voxel_grid(test_emb, threshold=t)
        print(f"  Threshold {t}: {len(voxels)} voxels ({len(voxels)/768:.1%} fill)")
    
    print("\n✓ Test data generation complete!")
    print(f"  Output directory: {output_dir}")
    
    return combined


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate test voxel data")
    parser.add_argument(
        "--output", "-o",
        default="tests/test_data",
        help="Output directory"
    )
    parser.add_argument(
        "--no-biobert",
        action="store_true",
        help="Use mock embeddings instead of BioBERT"
    )
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent.parent
    output_dir = project_root / args.output
    
    generate_test_data(output_dir, use_biobert=not args.no_biobert)


if __name__ == "__main__":
    main()
