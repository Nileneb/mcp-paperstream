"""
SIMPLIFIED Semantic Rules for Scientific Paper Validation
==========================================================

IMPORTANT: These rules use SIMPLIFIED phrases for better BioBERT matching.

The original rules were too specific (e.g., "computer-generated randomization sequence")
which doesn't match well with general text like "randomized controlled trial".

These simplified rules use CORE TERMS that appear in paper abstracts.
"""

SIMPLIFIED_RULES = {
    "is_rct": {
        "question": "Is this a Randomized Controlled Trial?",
        "positive_phrases": [
            # Core RCT terms - these appear in actual abstracts
            "randomized controlled trial",
            "RCT",
            "randomized trial",
            "randomly assigned",
            "random assignment",
            "randomization",
            "randomised controlled trial",  # British spelling
            "clinical trial",
            "controlled trial",
            "trial participants were randomized",
        ],
        "negative_phrases": [
            "observational study",
            "retrospective study",
            "cohort study",
            "case series",
            "cross-sectional",
            "non-randomized",
            "meta-analysis",
            "systematic review",
            "survey study",
        ],
        "threshold": 0.70
    },
    
    "has_placebo": {
        "question": "Does the study use placebo control?",
        "positive_phrases": [
            "placebo controlled",
            "placebo-controlled",
            "placebo group",
            "placebo control",
            "received placebo",
            "sham procedure",
            "sham control",
            "inactive control",
            "placebo arm",
        ],
        "negative_phrases": [
            "no placebo",
            "active control",
            "active comparator",
            "open-label",
            "no control group",
            "standard of care",
            "usual care",
        ],
        "threshold": 0.65
    },
    
    "is_blinded": {
        "question": "Is the study blinded/masked?",
        "positive_phrases": [
            "double-blind",
            "double blind",
            "single-blind",
            "single blind",
            "triple-blind",
            "blinded study",
            "blinded trial",
            "masked study",
            "participants were blinded",
            "investigators were blinded",
        ],
        "negative_phrases": [
            "open-label",
            "open label",
            "unblinded",
            "not blinded",
            "no blinding",
            "blinding not feasible",
            "participants were aware",
        ],
        "threshold": 0.70
    },
    
    "is_human_study": {
        "question": "Is this a human study?",
        "positive_phrases": [
            "human subjects",
            "human participants",
            "patients",
            "participants",
            "subjects",
            "volunteers",
            "human study",
            "clinical study",
            "enrolled patients",
            "recruited participants",
        ],
        "negative_phrases": [
            "animal study",
            "mouse model",
            "rat model",
            "in vitro",
            "cell culture",
            "simulation study",
            "computational study",
            "no human subjects",
        ],
        "threshold": 0.75
    },
    
    "reports_sample_size": {
        "question": "Does the study report sample size?",
        "positive_phrases": [
            "sample size",
            "n=",
            "participants",
            "enrolled",
            "recruited",
            "total of",
            "study population",
            "patients were enrolled",
            "sample of",
        ],
        "negative_phrases": [
            "sample size not reported",
            "number not specified",
            "theoretical study",
            "review article",
            "no empirical data",
        ],
        "threshold": 0.55
    },
    
    "has_control_group": {
        "question": "Does the study have a control group?",
        "positive_phrases": [
            "control group",
            "control arm",
            "controlled study",
            "compared to control",
            "versus control",
            "placebo group",
            "comparison group",
            "reference group",
        ],
        "negative_phrases": [
            "no control",
            "uncontrolled",
            "single arm",
            "case series",
            "case report",
            "descriptive study",
        ],
        "threshold": 0.60
    },
    
    "is_meta_analysis": {
        "question": "Is this a meta-analysis or systematic review?",
        "positive_phrases": [
            "meta-analysis",
            "systematic review",
            "pooled analysis",
            "literature review",
            "reviewed studies",
            "included studies",
            "database search",
            "PRISMA",
        ],
        "negative_phrases": [
            "original research",
            "primary study",
            "clinical trial",
            "observational study",
            "experimental study",
        ],
        "threshold": 0.65
    },
}


def get_simplified_rules():
    """Return the simplified rules dictionary"""
    return SIMPLIFIED_RULES


if __name__ == "__main__":
    # Print rules for verification
    for rule_id, rule in SIMPLIFIED_RULES.items():
        print(f"\n{rule_id}:")
        print(f"  Question: {rule['question']}")
        print(f"  Positive: {rule['positive_phrases'][:3]}...")
        print(f"  Negative: {rule['negative_phrases'][:3]}...")
