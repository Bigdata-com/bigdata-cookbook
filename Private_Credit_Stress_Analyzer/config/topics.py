from __future__ import annotations

from typing import TypeAlias

TopicDict: TypeAlias = dict[str, str | list[str]]

TOPICS: list[TopicDict] = [
    # ── LENDER STRENGTH (positive for lenders) ──────────────────────────────
    {
        "topic_name": "lender_spread_power",
        "topic_text": (
            "{company} spread widening new deal origination pricing power direct lending"
        ),
        "polarity": "positive",
        "applies_to": ["lender"],
    },
    {
        "topic_name": "lender_fundraise_resilience",
        "topic_text": (
            "{company} successful fundraise capital raise despite market volatility private credit"
        ),
        "polarity": "positive",
        "applies_to": ["lender"],
    },
    {
        "topic_name": "lender_nav_stability",
        "topic_text": (
            "{company} net asset value stable maintained portfolio quality BDC"
        ),
        "polarity": "positive",
        "applies_to": ["lender"],
    },
    {
        "topic_name": "lender_covenant_tightening",
        "topic_text": (
            "{company} covenant tightening lender protection credit agreement amendment favorable"
        ),
        "polarity": "positive",
        "applies_to": ["lender"],
    },
    # ── LENDER STRESS (negative for lenders) ────────────────────────────────
    {
        "topic_name": "redemption_pressure",
        "topic_text": (
            "{company} redemption request withdrawal investor liquidity gate private credit fund"
        ),
        "polarity": "negative",
        "applies_to": ["lender"],
    },
    {
        "topic_name": "nav_markdown",
        "topic_text": (
            "{company} NAV markdown write-down loan impairment portfolio devaluation"
        ),
        "polarity": "negative",
        "applies_to": ["lender"],
    },
    {
        "topic_name": "covenant_waiver",
        "topic_text": (
            "{company} covenant waiver amendment forbearance borrower relief private credit"
        ),
        "polarity": "negative",
        "applies_to": ["lender"],
    },
    {
        "topic_name": "pik_stress",
        "topic_text": (
            "{company} PIK toggle payment in kind interest deferral non-cash accrual"
        ),
        "polarity": "negative",
        "applies_to": ["lender"],
    },
    {
        "topic_name": "lender_fee_compression",
        "topic_text": (
            "{company} management fee reduction LP pressure performance fee cut"
        ),
        "polarity": "negative",
        "applies_to": ["lender"],
    },
    # ── BORROWER RESILIENCE (positive for borrowers) ─────────────────────────
    {
        "topic_name": "borrower_revenue_growth",
        "topic_text": (
            "{company} revenue growth maintained AI product launch customer retention strong"
        ),
        "polarity": "positive",
        "applies_to": ["borrower"],
    },
    {
        "topic_name": "borrower_refinancing_success",
        "topic_text": (
            "{company} debt refinanced successfully maturity extended credit facility renewed"
        ),
        "polarity": "positive",
        "applies_to": ["borrower"],
    },
    # ── BORROWER DISTRESS (negative for borrowers) ───────────────────────────
    {
        "topic_name": "ai_disruption_risk",
        "topic_text": (
            "{company} AI disruption software pricing power erosion competitive threat automation SaaS"
        ),
        "polarity": "negative",
        "applies_to": ["borrower"],
    },
    {
        "topic_name": "maturity_wall",
        "topic_text": (
            "{company} debt maturity 2026 2027 refinancing risk leveraged loan BDC private credit wall"
        ),
        "polarity": "negative",
        "applies_to": ["borrower"],
    },
    {
        "topic_name": "default_risk",
        "topic_text": (
            "{company} default restructuring missed payment interest coverage ratio covenant breach"
        ),
        "polarity": "negative",
        "applies_to": ["borrower"],
    },
    {
        "topic_name": "customer_churn",
        "topic_text": (
            "{company} customer churn revenue decline pricing pressure net revenue retention"
        ),
        "polarity": "negative",
        "applies_to": ["borrower"],
    },
    # ── BANK / BACK-LEVERAGE LAYER ───────────────────────────────────────────
    {
        "topic_name": "bank_credit_pullback",
        "topic_text": (
            "{company} back leverage restriction NAV line collateral markdown private credit bank pullback"
        ),
        "polarity": "negative",
        "applies_to": ["bank"],
    },
    {
        "topic_name": "bank_margin_call",
        "topic_text": (
            "{company} margin call collateral posting leverage reduction private credit financing"
        ),
        "polarity": "negative",
        "applies_to": ["bank"],
    },
    {
        "topic_name": "bank_market_share_gain",
        "topic_text": (
            "{company} private credit market share gain conservative exposure winner lending"
        ),
        "polarity": "positive",
        "applies_to": ["bank"],
    },
    {
        "topic_name": "bank_contagion_risk",
        "topic_text": (
            "{company} private credit contagion exposure loss bank risk financial stability"
        ),
        "polarity": "negative",
        "applies_to": ["bank"],
    },
]
