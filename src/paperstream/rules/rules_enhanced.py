"""
Enhanced Semantic Rules for Scientific Paper Validation
========================================================

These rules use semantically rich phrases that BioBERT can match
against paper sections to determine study characteristics.

Each rule contains:
- question: The validation question being answered
- positive_phrases: Phrases indicating positive match (study HAS this characteristic)
- negative_phrases: Phrases indicating negative match (study does NOT have this)
- threshold: Similarity threshold for matching (0.0-1.0)
"""

ENHANCED_RULES = {
    "is_rct": {
        "question": "Is this a Randomized Controlled Trial?",
        "positive_phrases": [
            # Randomization methods
            "participants were randomly assigned to treatment groups",
            "computer-generated randomization sequence",
            "stratified block randomization",
            "central randomization via telephone",
            "random allocation ratio",
            
            # Allocation concealment
            "allocation concealment using sealed opaque envelopes",
            "sequentially numbered drug containers",
            "centralized allocation system",
            
            # Blinding
            "double-blind placebo-controlled design",
            "participants and investigators were masked",
            "triple-blind study design",
            
            # ITT Analysis
            "intention-to-treat analysis",
            "all randomized participants were included",
            "modified intention-to-treat population",
            
            # Reporting
            "CONSORT flow diagram",
            "trial registration number NCT",
            "registered at ClinicalTrials.gov"
        ],
        "negative_phrases": [
            "retrospective chart review",
            "observational cohort study",
            "case series without control group",
            "systematic review and meta-analysis",
            "cross-sectional survey design",
            "patients were selected based on clinical judgment",
            "non-randomized comparison"
        ],
        "threshold": 0.70
    },
    
    "has_placebo": {
        "question": "Does the study use placebo control?",
        "positive_phrases": [
            "matched placebo identical in appearance",
            "placebo tablets manufactured to be indistinguishable",
            "saline injection as placebo control",
            "sham procedure performed in control group",
            "inactive control formulation",
            "placebo capsules containing lactose",
            "matching placebo with identical taste and smell"
        ],
        "negative_phrases": [
            "active comparator study design",
            "head-to-head comparison of treatments",
            "standard of care control arm",
            "no control group was used",
            "open-label design without placebo",
            "usual care comparison group"
        ],
        "threshold": 0.65
    },
    
    "has_control_group": {
        "question": "Does the study include a control group?",
        "positive_phrases": [
            "compared with a control group receiving",
            "patients were randomized to intervention or control",
            "the control arm received standard treatment",
            "parallel group design with active control",
            "waitlist control group",
            "no-treatment control condition"
        ],
        "negative_phrases": [
            "single-arm study design",
            "all patients received the intervention",
            "no comparison group was included",
            "before-after study without controls",
            "case series of consecutive patients"
        ],
        "threshold": 0.65
    },
    
    "is_blinded": {
        "question": "Is the study blinded?",
        "positive_phrases": [
            "double-blind randomized trial",
            "participants were blinded to treatment allocation",
            "investigators masked to group assignment",
            "outcome assessors blinded to intervention",
            "triple-blind study with blinded statistician",
            "single-blind design with blinded participants"
        ],
        "negative_phrases": [
            "open-label study design",
            "unblinded intervention",
            "participants were aware of treatment assignment",
            "no blinding was performed",
            "blinding not feasible due to intervention nature"
        ],
        "threshold": 0.70
    },
    
    "reports_primary_outcome": {
        "question": "Does the study clearly report a primary outcome?",
        "positive_phrases": [
            "the primary endpoint was defined as",
            "primary outcome measure was change in",
            "the primary efficacy endpoint",
            "pre-specified primary outcome",
            "sample size calculated based on primary outcome"
        ],
        "negative_phrases": [
            "exploratory analysis without primary endpoint",
            "multiple outcomes without hierarchy",
            "post-hoc outcome selection",
            "no primary outcome was specified"
        ],
        "threshold": 0.60
    },
    
    "sample_size_adequate": {
        "question": "Is the sample size adequate (>50 participants)?",
        "positive_phrases": [
            "enrolled 100 participants",
            "sample size of 200 patients",
            "a total of 500 subjects completed the study",
            "power calculation indicated 80 patients needed",
            "large-scale multicenter trial"
        ],
        "negative_phrases": [
            "pilot study with 20 participants",
            "small sample of 15 patients",
            "case series of 10 subjects",
            "preliminary data from 25 volunteers",
            "underpowered to detect differences"
        ],
        "threshold": 0.55
    },
    
    "has_statistical_analysis": {
        "question": "Does the study include appropriate statistical analysis?",
        "positive_phrases": [
            "analyzed using ANOVA with post-hoc testing",
            "Cox proportional hazards regression",
            "intention-to-treat analysis performed",
            "mixed-effects model for repeated measures",
            "adjusted for multiple comparisons using Bonferroni",
            "pre-specified statistical analysis plan"
        ],
        "negative_phrases": [
            "descriptive statistics only",
            "no formal statistical testing",
            "qualitative analysis of outcomes",
            "statistical methods not reported"
        ],
        "threshold": 0.60
    },
    
    "reports_p_values": {
        "question": "Does the study report p-values?",
        "positive_phrases": [
            "statistically significant difference p<0.05",
            "p-value for comparison was 0.001",
            "significance level alpha=0.05",
            "two-sided p-value reported",
            "adjusted p-values for multiple testing"
        ],
        "negative_phrases": [
            "no p-values calculated",
            "statistical significance not assessed",
            "effect sizes reported without p-values"
        ],
        "threshold": 0.55
    },
    
    "reports_ci": {
        "question": "Does the study report confidence intervals?",
        "positive_phrases": [
            "95% confidence interval",
            "CI 1.2 to 3.4",
            "mean difference with 95% CI",
            "hazard ratio 0.75 (95% CI 0.60-0.94)",
            "odds ratio with confidence bounds"
        ],
        "negative_phrases": [
            "confidence intervals not reported",
            "only point estimates provided",
            "no interval estimation performed"
        ],
        "threshold": 0.55
    },
    
    "itt_analysis": {
        "question": "Was an intention-to-treat analysis performed?",
        "positive_phrases": [
            "intention-to-treat analysis included all randomized patients",
            "ITT population defined as all randomized subjects",
            "analyzed according to randomized treatment assignment",
            "no participants excluded from ITT analysis",
            "modified ITT excluding patients without post-baseline data"
        ],
        "negative_phrases": [
            "per-protocol analysis only",
            "completers analysis excluding dropouts",
            "patients with protocol deviations excluded",
            "as-treated analysis based on actual treatment"
        ],
        "threshold": 0.65
    },
    
    "has_ethics_approval": {
        "question": "Does the study have ethics committee approval?",
        "positive_phrases": [
            "approved by the institutional review board",
            "ethics committee approval obtained",
            "study protocol approved by IRB",
            "conducted in accordance with Declaration of Helsinki",
            "written informed consent obtained from all participants",
            "ethical approval reference number"
        ],
        "negative_phrases": [
            "ethics approval not required",
            "exempt from IRB review",
            "no ethics statement provided",
            "retrospective data analysis without consent"
        ],
        "threshold": 0.60
    },
    
    "declares_coi": {
        "question": "Does the study declare conflicts of interest?",
        "positive_phrases": [
            "the authors declare no competing interests",
            "conflict of interest statement provided",
            "disclosures: author received consulting fees",
            "funding source had no role in study design",
            "COI: none declared"
        ],
        "negative_phrases": [
            "no conflict statement included",
            "disclosures not reported",
            "funding sources not disclosed"
        ],
        "threshold": 0.50
    },
    
    "pre_registered": {
        "question": "Was the study pre-registered?",
        "positive_phrases": [
            "registered at ClinicalTrials.gov NCT",
            "PROSPERO registration number",
            "protocol published before data collection",
            "pre-registered analysis plan",
            "trial registration: ISRCTN",
            "registered in EU Clinical Trials Register"
        ],
        "negative_phrases": [
            "not registered prospectively",
            "registration after study completion",
            "no trial registration reported",
            "unregistered pilot study"
        ],
        "threshold": 0.65
    },
    
    "is_peer_reviewed": {
        "question": "Is this a peer-reviewed publication?",
        "positive_phrases": [
            "published in peer-reviewed journal",
            "accepted after peer review process",
            "revised manuscript following reviewer comments",
            "published in The Lancet, NEJM, JAMA, BMJ"
        ],
        "negative_phrases": [
            "preprint not yet peer-reviewed",
            "conference abstract only",
            "posted on medRxiv awaiting review",
            "self-published report",
            "working paper draft"
        ],
        "threshold": 0.65
    },
    
    "reports_attrition": {
        "question": "Does the study report attrition/dropout rates?",
        "positive_phrases": [
            "dropout rate was 15% in treatment group",
            "loss to follow-up reported in CONSORT diagram",
            "attrition analysis comparing completers to dropouts",
            "reasons for discontinuation provided",
            "withdrawal due to adverse events documented"
        ],
        "negative_phrases": [
            "attrition not reported",
            "unclear how many completed the study",
            "dropout reasons not specified",
            "no flow diagram provided"
        ],
        "threshold": 0.55
    },
    
    "reports_adverse_events": {
        "question": "Does the study report adverse events?",
        "positive_phrases": [
            "adverse events were monitored throughout",
            "serious adverse events reported in Table",
            "safety population included all treated patients",
            "treatment-emergent adverse events",
            "no serious adverse events observed",
            "adverse event profile was favorable"
        ],
        "negative_phrases": [
            "safety data not collected",
            "adverse events not reported",
            "no safety assessment performed",
            "harms not systematically monitored"
        ],
        "threshold": 0.55
    },
    
    "is_human_study": {
        "question": "Is this a human study?",
        "positive_phrases": [
            "patients were enrolled from clinical sites",
            "human subjects research approved by IRB",
            "participants provided written informed consent",
            "adult volunteers aged 18 and older",
            "clinical trial in hospitalized patients"
        ],
        "negative_phrases": [
            "mouse model of disease",
            "in vitro cell culture experiments",
            "animal study using rats",
            "computational simulation study",
            "analysis of existing databases only"
        ],
        "threshold": 0.75
    }
}


# Helper to get rule count
def get_rule_count() -> int:
    """Returns number of defined rules"""
    return len(ENHANCED_RULES)


# Helper to get all rule IDs
def get_rule_ids() -> list:
    """Returns list of all rule IDs"""
    return list(ENHANCED_RULES.keys())
