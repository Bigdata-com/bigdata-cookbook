"""Benchmark Bigdata search query phrasing variants across multiple topics.

Compares taxonomy-style vs document-style query text on the same universe and
date range using plan volume, retrieved relevance, and lexical signals in chunks.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from bigdata_smart_batching import deduplicate_documents, execute_search, plan_search
from dotenv import load_dotenv

from src.screener import (
    DEFAULT_END_DATE,
    DEFAULT_SEARCH_CATEGORY,
    DEFAULT_START_DATE,
    UNIVERSE_ID_COLUMN,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIVERSE = PROJECT_ROOT / "XNAS_companies.csv"
DEFAULT_CHUNK_PERCENTAGE = 0.002

EXPOSURE_META_PATTERN = re.compile(
    r"\b(exposed to|exposure|ipo-?driven|capex scaling|beneficiar(?:y|ies)|spillover|"
    r"earning returns|profiting from|gain from)\b",
    re.IGNORECASE,
)

SUPPLIER_LEXICON = re.compile(
    r"\b(suppl(?:y|ies|ier|ying)|vendor|provid(?:e|es|ing)|customer|contract|avionics|"
    r"propulsion|subsystem|component|spacecraft|launch vehicle)\b",
    re.IGNORECASE,
)
COOLING_LEXICON = re.compile(
    r"\b(hvac|chiller|liquid cooling|thermal|cooling system|coolant)\b",
    re.IGNORECASE,
)
NUCLEAR_FUEL_LEXICON = re.compile(
    r"\b(uranium|enrichment|conversion|fuel fabrication|nuclear fuel|u3o8|reactor fuel)\b",
    re.IGNORECASE,
)
CONSTRUCTION_LEXICON = re.compile(
    r"\b(epc|contractor|construction|engineering|procurement|build|project)\b",
    re.IGNORECASE,
)
GOV_PROCUREMENT_LEXICON = re.compile(
    r"\b(contract|tender|procurement|government|ministr|public sector|award)\b",
    re.IGNORECASE,
)
AI_DESIGN_LEXICON = re.compile(
    r"\b(ai|artificial intelligence|machine learning|chip design|eda|semiconductor|"
    r"engineering workflow|tape-?out)\b",
    re.IGNORECASE,
)
DEFENSE_LEXICON = re.compile(
    r"\b(defense|government|military|procurement|contract|national security|dod)\b",
    re.IGNORECASE,
)
FINANCE_LEXICON = re.compile(
    r"\b(financ|lend|insur|guarant|underwrit|loan|credit)\b",
    re.IGNORECASE,
)
GPU_LEXICON = re.compile(
    r"\b(gpu|graphics|accelerator|datacenter|inference|training|cuda|tensor|chip)\b",
    re.IGNORECASE,
)
CLOUD_LEXICON = re.compile(
    r"\b(cloud|hyperscaler|infrastructure|compute|hosting|iaas|paas|workload)\b",
    re.IGNORECASE,
)
SEMICON_EQUIP_LEXICON = re.compile(
    r"\b(wafer|fab|lithography|etch|deposition|semiconductor equipment|foundry|process tool)\b",
    re.IGNORECASE,
)
CYBER_LEXICON = re.compile(
    r"\b(cybersecurity|endpoint|threat|security|firewall|zero trust|ransomware|breach)\b",
    re.IGNORECASE,
)
AD_LEXICON = re.compile(
    r"\b(advertising|ad revenue|digital ads|impressions|cpm|marketing platform|monetization)\b",
    re.IGNORECASE,
)
SAAS_LEXICON = re.compile(
    r"\b(subscription|saas|recurring revenue|arr|software license|cloud software|seat)\b",
    re.IGNORECASE,
)
STREAMING_LEXICON = re.compile(
    r"\b(streaming|subscriber|subscription video|content|ott|viewing|entertainment)\b",
    re.IGNORECASE,
)
BIOTECH_LEXICON = re.compile(
    r"\b(drug|pharmaceutical|therapeutic|clinical|fda|pipeline|biologic|treatment)\b",
    re.IGNORECASE,
)
ECOMMERCE_LEXICON = re.compile(
    r"\b(e-?commerce|online retail|marketplace|gmv|merchant|orders|travel booking)\b",
    re.IGNORECASE,
)
MEMORY_LEXICON = re.compile(
    r"\b(memory|dram|nand|storage|ssd|hdd|flash|data center storage)\b",
    re.IGNORECASE,
)
EV_BATTERY_LEXICON = re.compile(
    r"\b(cathode|anode|battery cell|battery pack|cell manufactur|lithium|nickel|cobalt|"
    r"gigafactory|ev supplier)\b",
    re.IGNORECASE,
)
SOLAR_WIND_LEXICON = re.compile(
    r"\b(solar panel|photovoltaic|inverter|wind turbine|turbine blade|renewable project|"
    r"module manufactur)\b",
    re.IGNORECASE,
)
AUTOMATION_LEXICON = re.compile(
    r"\b(robot|automation system|plc|sensor|machine vision|conveyor|industrial control)\b",
    re.IGNORECASE,
)
INSURANCE_LEXICON = re.compile(
    r"\b(underwrit|premium|policy|claims|reinsur|actuarial|loss ratio)\b",
    re.IGNORECASE,
)
CONSUMER_CREDIT_LEXICON = re.compile(
    r"\b(loan origination|credit card|interest income|lend|borrower|default rate|receivable)\b",
    re.IGNORECASE,
)
PAYMENTS_LEXICON = re.compile(
    r"\b(transaction volume|merchant|payment processing|interchange|acquiring|"
    r"payment network|gateway)\b",
    re.IGNORECASE,
)
APPAREL_LEXICON = re.compile(
    r"\b(apparel|footwear|sneaker|wholesale|direct-to-consumer|retail store|brand)\b",
    re.IGNORECASE,
)
RESTAURANT_LEXICON = re.compile(
    r"\b(franchise|same-store sales|royalty|restaurant unit|drive-thru|menu)\b",
    re.IGNORECASE,
)
MEDICAL_DEVICE_LEXICON = re.compile(
    r"\b(device|implant|surgical|fda clearance|diagnostic|catheter|monitor)\b",
    re.IGNORECASE,
)
HEALTHCARE_SERVICES_LEXICON = re.compile(
    r"\b(staffing|clinician|nurse|home health|patient visit|billable hour|"
    r"healthcare provider)\b",
    re.IGNORECASE,
)

SPACEX_THEME = re.compile(r"\bspace\s*x\b", re.IGNORECASE)
DATA_CENTER_THEME = re.compile(r"\bdata\s*cent(?:er|re)s?\b", re.IGNORECASE)
NUCLEAR_THEME = re.compile(r"\b(nuclear|reactor|uranium|smr)\b", re.IGNORECASE)
SPANISH_GOV_THEME = re.compile(r"\b(spanish|spain|government|public (?:sector|institution))\b", re.IGNORECASE)
AI_THEME = re.compile(r"\b(ai|artificial intelligence|machine learning|chip design)\b", re.IGNORECASE)
DEFENSE_THEME = re.compile(r"\b(defense|military|government contract|national security)\b", re.IGNORECASE)
FINANCE_THEME = re.compile(r"\b(financ|lend|insur|guarant|underwrit)\b", re.IGNORECASE)
GPU_THEME = re.compile(r"\b(gpu|graphics processing|ai accelerator|datacenter chip)\b", re.IGNORECASE)
CLOUD_THEME = re.compile(r"\b(cloud|hyperscaler|aws|azure|google cloud)\b", re.IGNORECASE)
SEMICON_EQUIP_THEME = re.compile(
    r"\b(semiconductor equipment|wafer fab|lithography|foundry capex)\b",
    re.IGNORECASE,
)
CYBER_THEME = re.compile(r"\b(cybersecurity|endpoint security|cloud security)\b", re.IGNORECASE)
AD_THEME = re.compile(r"\b(digital advertising|ad platform|social advertising)\b", re.IGNORECASE)
SAAS_THEME = re.compile(r"\b(enterprise software|saas|subscription software)\b", re.IGNORECASE)
STREAMING_THEME = re.compile(r"\b(streaming|video subscription|ott)\b", re.IGNORECASE)
BIOTECH_THEME = re.compile(r"\b(pharmaceutical|biotech|drug sales|therapeutic)\b", re.IGNORECASE)
ECOMMERCE_THEME = re.compile(r"\b(e-?commerce|online marketplace|digital retail)\b", re.IGNORECASE)
MEMORY_THEME = re.compile(r"\b(memory|dram|nand|ssd|storage demand)\b", re.IGNORECASE)
EV_BATTERY_THEME = re.compile(
    r"\b(electric vehicle|ev battery|battery pack|lithium-ion|gigafactory)\b", re.IGNORECASE
)
SOLAR_WIND_THEME = re.compile(
    r"\b(solar|photovoltaic|wind turbine|renewable energy|clean energy)\b", re.IGNORECASE
)
AUTOMATION_THEME = re.compile(
    r"\b(industrial automation|robotics|factory automation|robotic arm)\b", re.IGNORECASE
)
INSURANCE_THEME = re.compile(r"\b(insurance|underwrit|policyholder|premium)\b", re.IGNORECASE)
CONSUMER_CREDIT_THEME = re.compile(
    r"\b(consumer credit|consumer lending|personal loan|credit card)\b", re.IGNORECASE
)
PAYMENTS_THEME = re.compile(
    r"\b(payment processing|payments network|merchant acquiring|digital payments)\b",
    re.IGNORECASE,
)
APPAREL_THEME = re.compile(
    r"\b(apparel|footwear|athletic wear|fashion brand|sportswear)\b", re.IGNORECASE
)
RESTAURANT_THEME = re.compile(
    r"\b(restaurant|quick service|qsr|franchise|fast food)\b", re.IGNORECASE
)
MEDICAL_DEVICE_THEME = re.compile(
    r"\b(medical device|surgical device|implant|diagnostic equipment)\b", re.IGNORECASE
)
HEALTHCARE_SERVICES_THEME = re.compile(
    r"\b(healthcare services|medical staffing|home health|clinical staffing)\b", re.IGNORECASE
)

DEFAULT_XNAS_COMPANY_LIMIT = 200


@dataclass(frozen=True)
class QueryVariant:
    """A named search query variant to benchmark."""

    variant_id: str
    text: str
    style: str


@dataclass(frozen=True)
class ScenarioConfig:
    """One thematic topic with query variants and scoring patterns."""

    scenario_id: str
    description: str
    theme_pattern: re.Pattern[str]
    role_lexicon: re.Pattern[str]
    variants: tuple[QueryVariant, ...]


@dataclass
class VariantMetrics:
    """Retrieval metrics for one query variant."""

    variant_id: str
    style: str
    query_text: str
    plan_expected_chunks: int
    plan_basket_count: int
    document_count: int
    chunk_count: int
    company_count: int
    mean_relevance: float
    median_relevance: float
    mean_evidence_score: float
    top20_mean_relevance: float
    theme_term_rate_top20: float
    role_lexicon_rate_top20: float
    exposure_meta_rate_top20: float
    top_chunks: list[dict[str, Any]]


def _core_variants(
    taxonomy: str,
    document: str,
    first_person: str,
    disclosure: str,
    hybrid: str,
    keywords: str,
) -> tuple[QueryVariant, ...]:
    """Standard six-variant set used across topics."""
    return (
        QueryVariant("taxonomy_current", taxonomy, "taxonomy"),
        QueryVariant("doc_the_company", document, "document"),
        QueryVariant("doc_we", first_person, "first_person"),
        QueryVariant("doc_disclosure", disclosure, "disclosure"),
        QueryVariant("hybrid_no_meta", hybrid, "hybrid"),
        QueryVariant("keywords", keywords, "keywords"),
    )


SCENARIOS: dict[str, ScenarioConfig] = {
    "spacex_supplier": ScenarioConfig(
        scenario_id="spacex_supplier",
        description="SpaceX launch supply chain",
        theme_pattern=SPACEX_THEME,
        role_lexicon=SUPPLIER_LEXICON,
        variants=_core_variants(
            taxonomy=(
                "Suppliers of engines, avionics, structures, propulsion components, or spacecraft "
                "subsystems exposed to SpaceX IPO-driven production and capex scaling."
            ),
            document=(
                "The company supplies engines, avionics, structures, propulsion components, or "
                "spacecraft subsystems to SpaceX."
            ),
            first_person=(
                "We supply engines, avionics, structures, propulsion components, or spacecraft "
                "subsystems to SpaceX."
            ),
            disclosure=(
                "Revenue from SpaceX represented a significant portion of sales of launch vehicle "
                "components and avionics."
            ),
            hybrid="Companies supplying engines, avionics, structures, or propulsion components to SpaceX.",
            keywords="SpaceX supplier avionics propulsion spacecraft subsystems",
        ),
    ),
    "data_center_cooling": ScenarioConfig(
        scenario_id="data_center_cooling",
        description="Data center cooling vendors",
        theme_pattern=DATA_CENTER_THEME,
        role_lexicon=COOLING_LEXICON,
        variants=_core_variants(
            taxonomy="Cooling vendors gain from HVAC, chillers, and liquid cooling deployments.",
            document="The company provides HVAC, chillers, and liquid cooling systems for data centers.",
            first_person=(
                "We provide liquid cooling, chillers, and thermal management systems for data "
                "center customers."
            ),
            disclosure=(
                "Revenue from data center cooling products, including chillers and liquid cooling, "
                "grew year over year."
            ),
            hybrid="Companies selling HVAC, chillers, and liquid cooling systems to data center operators.",
            keywords="data center liquid cooling chillers HVAC thermal management",
        ),
    ),
    "nuclear_uranium_fuel": ScenarioConfig(
        scenario_id="nuclear_uranium_fuel",
        description="Uranium and nuclear fuel cycle suppliers",
        theme_pattern=NUCLEAR_THEME,
        role_lexicon=NUCLEAR_FUEL_LEXICON,
        variants=_core_variants(
            taxonomy=(
                "Companies profiting from uranium supply, enrichment, conversion, fuel fabrication, "
                "and long-term nuclear fuel contracts."
            ),
            document=(
                "The company supplies uranium, provides enrichment or conversion services, or "
                "fabricates nuclear fuel for utility customers."
            ),
            first_person=(
                "We mine uranium, provide enrichment services, and sell nuclear fuel to power "
                "plant operators."
            ),
            disclosure=(
                "Revenue from uranium sales, enrichment, and nuclear fuel contracts increased "
                "with new reactor demand."
            ),
            hybrid=(
                "Companies supplying uranium, enrichment, conversion, or nuclear fuel fabrication "
                "services."
            ),
            keywords="uranium enrichment conversion nuclear fuel fabrication utility contracts",
        ),
    ),
    "nuclear_epc": ScenarioConfig(
        scenario_id="nuclear_epc",
        description="Nuclear plant EPC and construction contractors",
        theme_pattern=NUCLEAR_THEME,
        role_lexicon=CONSTRUCTION_LEXICON,
        variants=_core_variants(
            taxonomy=(
                "EPC contractors generating revenue from building nuclear islands, balance-of-plant, "
                "and construction services under fixed-price or cost-plus contracts."
            ),
            document=(
                "The company provides EPC, construction, and engineering services for new nuclear "
                "plants and SMR projects."
            ),
            first_person=(
                "We build nuclear islands, balance-of-plant systems, and provide construction "
                "services for nuclear projects."
            ),
            disclosure=(
                "The company was awarded an EPC contract to construct a nuclear power plant or "
                "SMR project."
            ),
            hybrid="Companies building nuclear islands, balance-of-plant, and nuclear construction projects.",
            keywords="nuclear EPC construction reactor SMR engineering procurement",
        ),
    ),
    "spanish_gov_prime_contractor": ScenarioConfig(
        scenario_id="spanish_gov_prime_contractor",
        description="Spanish public-sector prime contractors",
        theme_pattern=SPANISH_GOV_THEME,
        role_lexicon=GOV_PROCUREMENT_LEXICON,
        variants=_core_variants(
            taxonomy=(
                "Prime contractors winning Spanish government procurement contracts for works, "
                "services, or supplies; revenue depends on contract awards and renewals."
            ),
            document=(
                "The company wins contracts to provide works, services, or supplies to Spanish "
                "government ministries, agencies, or state-owned entities."
            ),
            first_person=(
                "We provide works, services, and supplies under contracts with Spanish government "
                "ministries and public agencies."
            ),
            disclosure=(
                "Revenue from Spanish government contracts and public-sector tenders represented "
                "a growing share of sales."
            ),
            hybrid=(
                "Companies serving as prime contractors on Spanish government procurement contracts."
            ),
            keywords="Spain government contract procurement tender public sector prime contractor",
        ),
    ),
    "spanish_gov_it_vendor": ScenarioConfig(
        scenario_id="spanish_gov_it_vendor",
        description="IT vendors selling to Spanish public institutions",
        theme_pattern=SPANISH_GOV_THEME,
        role_lexicon=re.compile(
            r"\b(software|cloud|cybersecurity|systems integration|managed services|it services|"
            r"digital|saas)\b",
            re.IGNORECASE,
        ),
        variants=_core_variants(
            taxonomy=(
                "Vendors selling software, systems integration, cloud, cybersecurity, or managed "
                "services to Spanish public institutions via procurement."
            ),
            document=(
                "The company sells software, cloud, cybersecurity, or IT services to Spanish "
                "public institutions through government procurement."
            ),
            first_person=(
                "We provide cloud, cybersecurity, software, and systems integration services to "
                "Spanish government agencies."
            ),
            disclosure=(
                "The company signed a contract to deliver cloud, cybersecurity, or digital "
                "services to a Spanish public institution."
            ),
            hybrid="Companies selling software, cloud, and IT services to Spanish public-sector customers.",
            keywords="Spain government software cloud cybersecurity IT procurement public sector",
        ),
    ),
    "ai_chip_design": ScenarioConfig(
        scenario_id="ai_chip_design",
        description="AI in semiconductor chip design",
        theme_pattern=AI_THEME,
        role_lexicon=AI_DESIGN_LEXICON,
        variants=_core_variants(
            taxonomy="AI accelerates and optimizes the chip design process.",
            document=(
                "The company uses AI to accelerate chip design, layout, verification, or tape-out "
                "workflows."
            ),
            first_person=(
                "We use AI and machine learning to optimize semiconductor design, verification, and "
                "tape-out."
            ),
            disclosure=(
                "Our AI design tools helped customers reduce chip design cycle time and engineering "
                "cost."
            ),
            hybrid="Companies applying AI to semiconductor design, EDA, and chip verification workflows.",
            keywords="AI chip design semiconductor EDA layout verification tape-out",
        ),
    ),
    "ai_engineering_workflows": ScenarioConfig(
        scenario_id="ai_engineering_workflows",
        description="AI in engineering and product development workflows",
        theme_pattern=AI_THEME,
        role_lexicon=re.compile(
            r"\b(engineering|product development|design|simulation|cad|workflow|copilot|automation)\b",
            re.IGNORECASE,
        ),
        variants=_core_variants(
            taxonomy="AI improves efficiency and accuracy in engineering workflows.",
            document=(
                "The company uses AI to improve engineering workflows, product design, simulation, "
                "and development productivity."
            ),
            first_person=(
                "We deploy AI tools to automate engineering workflows, design reviews, and product "
                "development tasks."
            ),
            disclosure=(
                "Customers adopted our AI features to accelerate engineering design cycles and reduce "
                "development costs."
            ),
            hybrid="Companies using AI to improve engineering, design, and product development workflows.",
            keywords="AI engineering workflow product development design automation simulation",
        ),
    ),
    "defense_government_contractor": ScenarioConfig(
        scenario_id="defense_government_contractor",
        description="Defense contractors benefiting from government procurement",
        theme_pattern=DEFENSE_THEME,
        role_lexicon=DEFENSE_LEXICON,
        variants=_core_variants(
            taxonomy=(
                "Defense contractors and government suppliers exposed to increased SpaceX credibility "
                "and procurement demand following a potential IPO."
            ),
            document=(
                "The company is a defense contractor or government supplier providing systems, "
                "services, or components to military and government customers."
            ),
            first_person=(
                "We provide defense systems, aerospace components, and services under U.S. and "
                "allied government contracts."
            ),
            disclosure=(
                "Revenue from defense contracts and government procurement awards increased during "
                "the period."
            ),
            hybrid="Defense contractors and government suppliers winning military procurement contracts.",
            keywords="defense contractor government procurement military aerospace contract award",
        ),
    ),
    "nuclear_project_finance": ScenarioConfig(
        scenario_id="nuclear_project_finance",
        description="Lenders and insurers on nuclear projects",
        theme_pattern=FINANCE_THEME,
        role_lexicon=FINANCE_LEXICON,
        variants=_core_variants(
            taxonomy=(
                "Lenders, insurers, and guarantors exposed to nuclear project financing, construction "
                "risk, and regulatory or sanctions-driven disruptions."
            ),
            document=(
                "The company provides lending, insurance, guarantees, or project finance for nuclear "
                "power plant development."
            ),
            first_person=(
                "We provide project finance, insurance, and credit guarantees for nuclear plant "
                "construction and operation."
            ),
            disclosure=(
                "The company financed, insured, or guaranteed loans for a nuclear power project or "
                "reactor development."
            ),
            hybrid="Banks, insurers, and guarantors financing nuclear power plant projects.",
            keywords="nuclear project finance lending insurance guarantee reactor construction",
        ),
    ),
}

XNAS_SCENARIOS: dict[str, ScenarioConfig] = {
    "gpu_ai_accelerators": ScenarioConfig(
        scenario_id="gpu_ai_accelerators",
        description="GPU and AI accelerator demand for data centers",
        theme_pattern=GPU_THEME,
        role_lexicon=GPU_LEXICON,
        variants=_core_variants(
            taxonomy=(
                "Semiconductor companies benefiting from surging GPU and AI accelerator demand "
                "driven by hyperscaler capex and generative AI workloads."
            ),
            document=(
                "The company designs, manufactures, or sells GPUs and AI accelerators for "
                "data center training and inference workloads."
            ),
            first_person=(
                "We design and sell GPUs and AI accelerators used in data center training "
                "and inference."
            ),
            disclosure=(
                "Data center revenue from GPU and AI accelerator products grew as cloud "
                "customers expanded AI infrastructure spending."
            ),
            hybrid="Companies designing or selling GPUs and AI accelerators for data center customers.",
            keywords="GPU AI accelerator data center training inference NVIDIA CUDA",
        ),
    ),
    "cloud_infrastructure_capex": ScenarioConfig(
        scenario_id="cloud_infrastructure_capex",
        description="Cloud infrastructure and hyperscaler spending beneficiaries",
        theme_pattern=CLOUD_THEME,
        role_lexicon=CLOUD_LEXICON,
        variants=_core_variants(
            taxonomy=(
                "Technology vendors exposed to hyperscaler cloud infrastructure capex, compute "
                "build-outs, and enterprise cloud migration tailwinds."
            ),
            document=(
                "The company provides cloud infrastructure, compute, networking, or platform "
                "services to enterprise and hyperscaler customers."
            ),
            first_person=(
                "We provide cloud infrastructure, compute capacity, and platform services to "
                "enterprise and hyperscaler customers."
            ),
            disclosure=(
                "Revenue from cloud infrastructure services and hyperscaler customer contracts "
                "increased year over year."
            ),
            hybrid="Companies selling cloud infrastructure, compute, and platform services to hyperscalers.",
            keywords="cloud infrastructure hyperscaler compute networking enterprise migration",
        ),
    ),
    "semiconductor_equipment": ScenarioConfig(
        scenario_id="semiconductor_equipment",
        description="Semiconductor capital equipment vendors",
        theme_pattern=SEMICON_EQUIP_THEME,
        role_lexicon=SEMICON_EQUIP_LEXICON,
        variants=_core_variants(
            taxonomy=(
                "Semiconductor equipment suppliers gaining from foundry and memory fab capex "
                "cycles and advanced node capacity expansion."
            ),
            document=(
                "The company sells wafer fabrication equipment, process tools, or services to "
                "semiconductor foundries and memory manufacturers."
            ),
            first_person=(
                "We sell lithography, etch, deposition, and other wafer fabrication equipment "
                "to chip manufacturers."
            ),
            disclosure=(
                "Orders and revenue from semiconductor capital equipment shipments to foundry "
                "customers increased during the period."
            ),
            hybrid="Companies selling wafer fab equipment and process tools to semiconductor manufacturers.",
            keywords="semiconductor equipment wafer fab lithography etch deposition foundry capex",
        ),
    ),
    "enterprise_cybersecurity": ScenarioConfig(
        scenario_id="enterprise_cybersecurity",
        description="Enterprise cybersecurity software vendors",
        theme_pattern=CYBER_THEME,
        role_lexicon=CYBER_LEXICON,
        variants=_core_variants(
            taxonomy=(
                "Cybersecurity vendors profiting from rising enterprise security budgets, "
                "endpoint protection demand, and cloud workload threats."
            ),
            document=(
                "The company sells cybersecurity software for endpoint protection, cloud security, "
                "or threat detection to enterprise customers."
            ),
            first_person=(
                "We provide endpoint protection, cloud security, and threat detection software "
                "to enterprise customers."
            ),
            disclosure=(
                "Subscription revenue from cybersecurity products and annual recurring contracts "
                "with enterprise customers grew."
            ),
            hybrid="Companies selling endpoint and cloud cybersecurity software to enterprises.",
            keywords="cybersecurity endpoint protection cloud security threat detection enterprise",
        ),
    ),
    "digital_advertising_platform": ScenarioConfig(
        scenario_id="digital_advertising_platform",
        description="Digital advertising and ad platform revenue",
        theme_pattern=AD_THEME,
        role_lexicon=AD_LEXICON,
        variants=_core_variants(
            taxonomy=(
                "Ad-supported platforms and ad-tech vendors exposed to digital advertising spend "
                "cycles and marketer budget shifts."
            ),
            document=(
                "The company generates revenue from digital advertising, ad platforms, or "
                "marketing technology sold to advertisers."
            ),
            first_person=(
                "We monetize our platform through digital advertising and marketing technology "
                "sold to brands and agencies."
            ),
            disclosure=(
                "Advertising revenue increased as impressions and average pricing on our digital "
                "ad platform grew."
            ),
            hybrid="Companies earning revenue from digital advertising platforms and ad technology.",
            keywords="digital advertising ad platform impressions CPM marketing technology revenue",
        ),
    ),
    "enterprise_saas_subscriptions": ScenarioConfig(
        scenario_id="enterprise_saas_subscriptions",
        description="Enterprise SaaS subscription software",
        theme_pattern=SAAS_THEME,
        role_lexicon=SAAS_LEXICON,
        variants=_core_variants(
            taxonomy=(
                "Enterprise software vendors with recurring subscription revenue benefiting from "
                "seat expansion, upsell, and cloud software adoption."
            ),
            document=(
                "The company sells subscription software and cloud applications to enterprise "
                "customers on a recurring revenue basis."
            ),
            first_person=(
                "We sell subscription software and cloud applications to enterprise customers "
                "under annual recurring contracts."
            ),
            disclosure=(
                "Annual recurring revenue and subscription billings from enterprise software "
                "products increased during the quarter."
            ),
            hybrid="Companies selling enterprise subscription software and cloud applications.",
            keywords="enterprise SaaS subscription recurring revenue ARR cloud software",
        ),
    ),
    "data_center_cooling": SCENARIOS["data_center_cooling"],
    "streaming_subscription_revenue": ScenarioConfig(
        scenario_id="streaming_subscription_revenue",
        description="Streaming video subscription businesses",
        theme_pattern=STREAMING_THEME,
        role_lexicon=STREAMING_LEXICON,
        variants=_core_variants(
            taxonomy=(
                "Streaming platforms exposed to subscriber growth, content investment, and "
                "competition in subscription video markets."
            ),
            document=(
                "The company operates a streaming video service and earns subscription revenue "
                "from paid members."
            ),
            first_person=(
                "We operate a streaming video service and earn subscription fees from paid "
                "members worldwide."
            ),
            disclosure=(
                "Paid streaming subscribers and subscription revenue increased as we added "
                "original content."
            ),
            hybrid="Companies operating streaming video services with paid subscription revenue.",
            keywords="streaming video subscription paid members OTT content revenue",
        ),
    ),
    "biotech_pharmaceutical_sales": ScenarioConfig(
        scenario_id="biotech_pharmaceutical_sales",
        description="Biotech and pharmaceutical product revenue",
        theme_pattern=BIOTECH_THEME,
        role_lexicon=BIOTECH_LEXICON,
        variants=_core_variants(
            taxonomy=(
                "Biopharma companies with revenue tied to marketed drug sales, pipeline "
                "approvals, and therapeutic category demand."
            ),
            document=(
                "The company develops and sells pharmaceutical or biologic therapies and "
                "records product revenue from marketed drugs."
            ),
            first_person=(
                "We develop and commercialize pharmaceutical therapies and record product sales "
                "from our marketed drug portfolio."
            ),
            disclosure=(
                "Net product revenue from our marketed therapies increased following label "
                "expansion and patient uptake."
            ),
            hybrid="Biotech and pharmaceutical companies earning revenue from marketed drug sales.",
            keywords="pharmaceutical biotech drug sales therapeutic FDA approved product revenue",
        ),
    ),
    "ecommerce_online_retail": ScenarioConfig(
        scenario_id="ecommerce_online_retail",
        description="E-commerce and online marketplace revenue",
        theme_pattern=ECOMMERCE_THEME,
        role_lexicon=ECOMMERCE_LEXICON,
        variants=_core_variants(
            taxonomy=(
                "Online retail and marketplace operators exposed to consumer e-commerce trends, "
                "take rates, and digital transaction volume."
            ),
            document=(
                "The company operates an e-commerce marketplace or online retail platform and "
                "earns revenue from merchant sales and transactions."
            ),
            first_person=(
                "We operate an online marketplace and earn revenue from merchant sales, "
                "commissions, and retail transactions."
            ),
            disclosure=(
                "Gross merchandise volume and online retail revenue grew as marketplace "
                "transaction activity increased."
            ),
            hybrid="Companies operating e-commerce marketplaces and online retail platforms.",
            keywords="e-commerce marketplace online retail GMV merchant transactions digital commerce",
        ),
    ),
    "memory_storage_demand": ScenarioConfig(
        scenario_id="memory_storage_demand",
        description="Memory and storage semiconductor demand",
        theme_pattern=MEMORY_THEME,
        role_lexicon=MEMORY_LEXICON,
        variants=_core_variants(
            taxonomy=(
                "Memory and storage suppliers benefiting from data center build-outs, AI workload "
                "density, and NAND or DRAM demand recovery."
            ),
            document=(
                "The company manufactures or sells DRAM, NAND flash, SSDs, or hard drives for "
                "data center and enterprise storage customers."
            ),
            first_person=(
                "We manufacture DRAM, NAND flash, and storage products sold to data center "
                "and enterprise customers."
            ),
            disclosure=(
                "Revenue from memory and storage products sold to data center customers "
                "increased with higher bit shipments."
            ),
            hybrid="Companies manufacturing DRAM, NAND, SSD, and storage products for data centers.",
            keywords="DRAM NAND flash SSD storage data center memory semiconductor demand",
        ),
    ),
}

NEW_VERTICALS_SCENARIOS: dict[str, ScenarioConfig] = {
    "ev_battery_supply_chain": ScenarioConfig(
        scenario_id="ev_battery_supply_chain",
        description="EV and battery supply chain",
        theme_pattern=EV_BATTERY_THEME,
        role_lexicon=EV_BATTERY_LEXICON,
        variants=_core_variants(
            taxonomy=(
                "Battery and EV component suppliers benefiting from electric vehicle adoption "
                "and gigafactory capacity expansion."
            ),
            document=(
                "The company manufactures or supplies battery cells, cathodes, anodes, or "
                "battery packs for electric vehicle makers."
            ),
            first_person=(
                "We manufacture battery cells and battery packs supplied to electric vehicle "
                "manufacturers."
            ),
            disclosure=(
                "Revenue from battery cell and EV component shipments increased as gigafactory "
                "capacity ramped."
            ),
            hybrid="Companies supplying battery cells, cathodes, anodes, or packs to electric vehicle manufacturers.",
            keywords="EV battery cell cathode anode lithium gigafactory electric vehicle supplier",
        ),
    ),
    "solar_wind_renewable_energy": ScenarioConfig(
        scenario_id="solar_wind_renewable_energy",
        description="Solar and wind renewable energy equipment",
        theme_pattern=SOLAR_WIND_THEME,
        role_lexicon=SOLAR_WIND_LEXICON,
        variants=_core_variants(
            taxonomy=(
                "Renewable energy equipment makers gaining from solar and wind capacity "
                "buildout and clean energy tax incentives."
            ),
            document=(
                "The company manufactures solar panels, inverters, or wind turbines for "
                "utility and commercial renewable energy projects."
            ),
            first_person=(
                "We manufacture solar panels, inverters, and wind turbine components sold to "
                "renewable energy developers."
            ),
            disclosure=(
                "Shipments and revenue from solar module and wind turbine products increased "
                "with new project installations."
            ),
            hybrid="Companies manufacturing solar panels, inverters, or wind turbines for renewable energy projects.",
            keywords="solar panel photovoltaic inverter wind turbine renewable energy module",
        ),
    ),
    "industrial_automation_robotics": ScenarioConfig(
        scenario_id="industrial_automation_robotics",
        description="Industrial automation and robotics",
        theme_pattern=AUTOMATION_THEME,
        role_lexicon=AUTOMATION_LEXICON,
        variants=_core_variants(
            taxonomy=(
                "Automation vendors benefiting from factory robotics adoption and "
                "manufacturing labor cost pressures."
            ),
            document=(
                "The company manufactures industrial robots, automation systems, or machine "
                "vision equipment for factory customers."
            ),
            first_person=(
                "We manufacture industrial robots and automation systems sold to "
                "manufacturing plant customers."
            ),
            disclosure=(
                "Orders for industrial robots and automation systems increased as "
                "manufacturers invested in factory automation."
            ),
            hybrid="Companies manufacturing industrial robots, automation systems, or machine vision equipment for factories.",
            keywords="industrial robot automation factory machine vision sensor manufacturing",
        ),
    ),
    "insurance_underwriting": ScenarioConfig(
        scenario_id="insurance_underwriting",
        description="Insurance underwriting (P&C and life)",
        theme_pattern=INSURANCE_THEME,
        role_lexicon=INSURANCE_LEXICON,
        variants=_core_variants(
            taxonomy=(
                "Insurers benefiting from premium growth and favorable underwriting cycles "
                "across property, casualty, and life lines."
            ),
            document=(
                "The company underwrites property, casualty, or life insurance policies and "
                "earns premium revenue from policyholders."
            ),
            first_person=(
                "We underwrite insurance policies and earn premium income from policyholders "
                "across our product lines."
            ),
            disclosure=(
                "Net premiums earned and underwriting income increased as policy renewals and "
                "new business grew."
            ),
            hybrid="Companies underwriting property, casualty, or life insurance policies for premium revenue.",
            keywords="insurance underwriting premium policyholder claims reinsurance actuarial",
        ),
    ),
    "consumer_credit_lending": ScenarioConfig(
        scenario_id="consumer_credit_lending",
        description="Consumer credit and lending",
        theme_pattern=CONSUMER_CREDIT_THEME,
        role_lexicon=CONSUMER_CREDIT_LEXICON,
        variants=_core_variants(
            taxonomy=(
                "Consumer lenders benefiting from loan origination growth and credit card "
                "receivable expansion."
            ),
            document=(
                "The company originates consumer loans or credit card receivables and earns "
                "interest income from borrowers."
            ),
            first_person=(
                "We originate personal loans and credit card receivables and earn interest "
                "income from our borrowers."
            ),
            disclosure=(
                "Loan originations and interest income from consumer credit products "
                "increased during the period."
            ),
            hybrid="Companies originating consumer loans or credit card receivables for interest income.",
            keywords="consumer credit lending personal loan credit card interest income receivable",
        ),
    ),
    "payments_processing": ScenarioConfig(
        scenario_id="payments_processing",
        description="Payments processing",
        theme_pattern=PAYMENTS_THEME,
        role_lexicon=PAYMENTS_LEXICON,
        variants=_core_variants(
            taxonomy=(
                "Payments processors benefiting from transaction volume growth and merchant "
                "acquiring expansion."
            ),
            document=(
                "The company processes electronic payments or provides merchant acquiring "
                "services and earns transaction-based fees."
            ),
            first_person=(
                "We process electronic payments and provide merchant acquiring services, "
                "earning fees on transaction volume."
            ),
            disclosure=(
                "Payment transaction volume and processing revenue increased as merchant "
                "acceptance expanded."
            ),
            hybrid="Companies processing electronic payments or providing merchant acquiring services for transaction fees.",
            keywords="payments processing merchant acquiring transaction volume interchange gateway",
        ),
    ),
    "apparel_footwear_brands": ScenarioConfig(
        scenario_id="apparel_footwear_brands",
        description="Apparel and footwear brands",
        theme_pattern=APPAREL_THEME,
        role_lexicon=APPAREL_LEXICON,
        variants=_core_variants(
            taxonomy=(
                "Apparel and footwear brands benefiting from consumer spending trends and "
                "direct-to-consumer channel growth."
            ),
            document=(
                "The company designs and sells apparel or footwear products through wholesale "
                "and direct-to-consumer channels."
            ),
            first_person=(
                "We design and sell apparel and footwear products through wholesale and "
                "direct-to-consumer channels."
            ),
            disclosure=(
                "Net sales of apparel and footwear products increased across wholesale and "
                "direct-to-consumer channels."
            ),
            hybrid="Companies designing and selling apparel or footwear through wholesale and direct-to-consumer channels.",
            keywords="apparel footwear brand sneaker wholesale direct-to-consumer sportswear",
        ),
    ),
    "restaurant_qsr_franchising": ScenarioConfig(
        scenario_id="restaurant_qsr_franchising",
        description="Restaurant and QSR franchising",
        theme_pattern=RESTAURANT_THEME,
        role_lexicon=RESTAURANT_LEXICON,
        variants=_core_variants(
            taxonomy=(
                "Restaurant franchisors benefiting from same-store sales growth and unit "
                "expansion through franchisees."
            ),
            document=(
                "The company franchises or operates quick-service restaurants and earns "
                "royalty or restaurant sales revenue."
            ),
            first_person=(
                "We franchise and operate quick-service restaurants and earn royalty income "
                "from franchisees."
            ),
            disclosure=(
                "Same-store sales and franchise royalty revenue increased as restaurant unit "
                "count grew."
            ),
            hybrid="Companies franchising or operating quick-service restaurants for royalty and sales revenue.",
            keywords="restaurant quick service franchise royalty same-store sales unit growth",
        ),
    ),
    "medical_devices": ScenarioConfig(
        scenario_id="medical_devices",
        description="Medical devices",
        theme_pattern=MEDICAL_DEVICE_THEME,
        role_lexicon=MEDICAL_DEVICE_LEXICON,
        variants=_core_variants(
            taxonomy=(
                "Medical device makers benefiting from procedure volume growth and new "
                "product clearances."
            ),
            document=(
                "The company designs and manufactures medical devices, surgical instruments, "
                "or diagnostic equipment sold to hospitals and clinicians."
            ),
            first_person=(
                "We design and manufacture medical devices and surgical instruments sold to "
                "hospitals and clinicians."
            ),
            disclosure=(
                "Product revenue from medical device sales increased following new FDA "
                "clearances and procedure volume growth."
            ),
            hybrid="Companies manufacturing medical devices, surgical instruments, or diagnostic equipment for hospitals.",
            keywords="medical device surgical implant diagnostic equipment FDA clearance hospital",
        ),
    ),
    "healthcare_services_staffing": ScenarioConfig(
        scenario_id="healthcare_services_staffing",
        description="Healthcare services and staffing",
        theme_pattern=HEALTHCARE_SERVICES_THEME,
        role_lexicon=HEALTHCARE_SERVICES_LEXICON,
        variants=_core_variants(
            taxonomy=(
                "Healthcare services and staffing providers benefiting from clinician demand "
                "and outsourced care delivery."
            ),
            document=(
                "The company provides healthcare staffing, home health, or clinical services "
                "to hospitals and patients."
            ),
            first_person=(
                "We provide healthcare staffing and home health services to hospitals, "
                "clinics, and patients."
            ),
            disclosure=(
                "Revenue from healthcare staffing placements and home health visits increased "
                "during the period."
            ),
            hybrid="Companies providing healthcare staffing, home health, or clinical services to hospitals and patients.",
            keywords="healthcare staffing home health clinician nurse patient services",
        ),
    ),
}

SCENARIO_SUITES: dict[str, dict[str, ScenarioConfig]] = {
    "legacy": SCENARIOS,
    "xnas": XNAS_SCENARIOS,
    "new_verticals": NEW_VERTICALS_SCENARIOS,
    "all": {**SCENARIOS, **XNAS_SCENARIOS, **NEW_VERTICALS_SCENARIOS},
}


def _chunk_rows(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document in documents:
        document_id = document.get("document_id") or document.get("id")
        for chunk in document.get("chunks", []):
            rows.append(
                {
                    "document_id": document_id,
                    "text": str(chunk.get("text", "")),
                    "relevance": float(chunk.get("relevance") or 0.0),
                    "sentiment": float(chunk.get("sentiment") or 0.0),
                    "entity_ids": chunk.get("entity_ids") or [],
                }
            )
    return rows


def _evidence_score(relevance: float, sentiment: float) -> float:
    return relevance * abs(sentiment)


def _rate(pattern: re.Pattern[str], texts: list[str]) -> float:
    if not texts:
        return 0.0
    hits = sum(1 for text in texts if pattern.search(text))
    return hits / len(texts)


def benchmark_variant(
    variant: QueryVariant,
    company_ids: list[str],
    config: ScenarioConfig,
    start_date: str,
    end_date: str,
    chunk_percentage: float,
    requests_per_minute: int,
) -> VariantMetrics:
    """Plan and execute one query variant, returning retrieval metrics."""
    plan = plan_search(
        universe=company_ids,
        start_date=start_date,
        end_date=end_date,
        volume_query_mode="iterative",
        text=variant.text,
        category=DEFAULT_SEARCH_CATEGORY,
    )
    expected_chunks = int(plan.get("chunk_upper_bound_estimate") or 0)
    basket_count = len(plan.get("baskets", []))

    documents = deduplicate_documents(
        execute_search(
            search_plan=plan,
            chunk_percentage=chunk_percentage,
            requests_per_minute=requests_per_minute,
            basket_filtered_entities=True,
        )
    )
    rows = _chunk_rows(documents)
    if rows:
        relevance_values = [row["relevance"] for row in rows]
        evidence_values = [_evidence_score(row["relevance"], row["sentiment"]) for row in rows]
        sorted_rows = sorted(rows, key=lambda row: row["relevance"], reverse=True)
        top20 = sorted_rows[:20]
        top20_texts = [row["text"] for row in top20]
        company_ids_found = {
            entity_id
            for row in rows
            for entity_id in row.get("entity_ids", [])
            if entity_id
        }
        return VariantMetrics(
            variant_id=variant.variant_id,
            style=variant.style,
            query_text=variant.text,
            plan_expected_chunks=expected_chunks,
            plan_basket_count=basket_count,
            document_count=len(documents),
            chunk_count=len(rows),
            company_count=len(company_ids_found),
            mean_relevance=sum(relevance_values) / len(relevance_values),
            median_relevance=float(pd.Series(relevance_values).median()),
            mean_evidence_score=sum(evidence_values) / len(evidence_values),
            top20_mean_relevance=sum(row["relevance"] for row in top20) / len(top20),
            theme_term_rate_top20=_rate(config.theme_pattern, top20_texts),
            role_lexicon_rate_top20=_rate(config.role_lexicon, top20_texts),
            exposure_meta_rate_top20=_rate(EXPOSURE_META_PATTERN, top20_texts),
            top_chunks=[
                {
                    "relevance": round(row["relevance"], 4),
                    "sentiment": round(row["sentiment"], 4),
                    "text": row["text"][:280],
                }
                for row in top20[:3]
            ],
        )

    return VariantMetrics(
        variant_id=variant.variant_id,
        style=variant.style,
        query_text=variant.text,
        plan_expected_chunks=expected_chunks,
        plan_basket_count=basket_count,
        document_count=0,
        chunk_count=0,
        company_count=0,
        mean_relevance=0.0,
        median_relevance=0.0,
        mean_evidence_score=0.0,
        top20_mean_relevance=0.0,
        theme_term_rate_top20=0.0,
        role_lexicon_rate_top20=0.0,
        exposure_meta_rate_top20=0.0,
        top_chunks=[],
    )


def _rank_variants(metrics: list[VariantMetrics]) -> list[dict[str, Any]]:
    """Rank variants by a composite quality score."""
    ranked: list[dict[str, Any]] = []
    for item in metrics:
        composite = (
            item.top20_mean_relevance * 0.45
            + item.role_lexicon_rate_top20 * 0.30
            + item.theme_term_rate_top20 * 0.20
            - item.exposure_meta_rate_top20 * 0.05
        )
        ranked.append({**asdict(item), "composite_score": round(composite, 4)})
    ranked.sort(key=lambda row: row["composite_score"], reverse=True)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def _taxonomy_vs_best(ranked: list[dict[str, Any]]) -> dict[str, Any]:
    taxonomy = next((row for row in ranked if row["variant_id"] == "taxonomy_current"), None)
    winner = ranked[0] if ranked else None
    if taxonomy is None or winner is None:
        return {}
    return {
        "taxonomy_rank": taxonomy["rank"],
        "taxonomy_score": taxonomy["composite_score"],
        "winner_variant_id": winner["variant_id"],
        "winner_style": winner["style"],
        "winner_score": winner["composite_score"],
        "score_delta": round(float(winner["composite_score"]) - float(taxonomy["composite_score"]), 4),
        "relevance_delta": round(
            float(winner["top20_mean_relevance"]) - float(taxonomy["top20_mean_relevance"]),
            4,
        ),
    }


def run_benchmark(
    config: ScenarioConfig,
    universe_path: Path,
    company_limit: int,
    chunk_percentage: float,
    requests_per_minute: int,
    start_date: str,
    end_date: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Run all variants for one scenario and write a JSON report."""
    universe_df = pd.read_csv(universe_path)
    company_ids = universe_df[UNIVERSE_ID_COLUMN].astype(str).head(company_limit).tolist()

    results: list[VariantMetrics] = []
    for variant in config.variants:
        print(f"[{config.scenario_id}] benchmarking {variant.variant_id} ...", flush=True)
        results.append(
            benchmark_variant(
                variant=variant,
                company_ids=company_ids,
                config=config,
                start_date=start_date,
                end_date=end_date,
                chunk_percentage=chunk_percentage,
                requests_per_minute=requests_per_minute,
            )
        )

    ranked = _rank_variants(results)
    comparison = _taxonomy_vs_best(ranked)
    payload = {
        "scenario": config.scenario_id,
        "description": config.description,
        "created_at": datetime.now(UTC).isoformat(),
        "universe_path": str(universe_path),
        "company_limit": company_limit,
        "chunk_percentage": chunk_percentage,
        "start_date": start_date,
        "end_date": end_date,
        "taxonomy_vs_winner": comparison,
        "ranked_variants": ranked,
        "winner": ranked[0] if ranked else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{config.scenario_id}_query_benchmark.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _aggregate_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
  style_wins: dict[str, int] = {}
  rows: list[dict[str, Any]] = []
  for summary in summaries:
    winner = summary.get("winner") or {}
    style = str(winner.get("style", "unknown"))
    style_wins[style] = style_wins.get(style, 0) + 1
    comparison = summary.get("taxonomy_vs_winner") or {}
    rows.append(
        {
            "scenario": summary["scenario"],
            "description": summary.get("description"),
            "winner": winner.get("variant_id"),
            "winner_style": style,
            "taxonomy_rank": comparison.get("taxonomy_rank"),
            "score_delta": comparison.get("score_delta"),
            "winner_query": winner.get("query_text"),
        }
    )
  rows.sort(key=lambda row: float(row.get("score_delta") or 0), reverse=True)
  return {
      "scenario_count": len(summaries),
      "winner_style_counts": style_wins,
      "scenarios": rows,
  }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Benchmark search query phrasing variants.")
    parser.add_argument(
        "--suite",
        choices=sorted(SCENARIO_SUITES),
        default="xnas",
        help="Scenario set: xnas (Nasdaq-100 style topics), legacy (prior mixed topics), or all.",
    )
    parser.add_argument(
        "--scenario",
        default="all",
        help="Topic id within the suite, or 'all' for every topic in the suite.",
    )
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--company-limit", type=int, default=DEFAULT_XNAS_COMPANY_LIMIT)
    parser.add_argument("--chunk-percentage", type=float, default=DEFAULT_CHUNK_PERCENTAGE)
    parser.add_argument("--requests-per-minute", type=int, default=120)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / f"query_benchmark_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
    )
    args = parser.parse_args()

    suite = SCENARIO_SUITES[args.suite]
    scenario_ids = sorted(suite)
    if args.scenario != "all":
        if args.scenario not in suite:
            valid = ", ".join(scenario_ids)
            msg = f"Unknown scenario '{args.scenario}' for suite '{args.suite}'. Valid: {valid}"
            raise SystemExit(msg)
        selected = [args.scenario]
    else:
        selected = scenario_ids

    load_dotenv()
    summaries: list[dict[str, Any]] = []
    for scenario_id in selected:
        summaries.append(
            run_benchmark(
                config=suite[scenario_id],
                universe_path=args.universe,
                company_limit=args.company_limit,
                chunk_percentage=args.chunk_percentage,
                requests_per_minute=args.requests_per_minute,
                start_date=args.start_date,
                end_date=args.end_date,
                output_dir=args.output_dir,
            )
        )

    aggregate = _aggregate_summary(summaries)
    index_path = args.output_dir / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "suite": args.suite,
                "company_limit": args.company_limit,
                "universe": str(args.universe),
                "summaries": summaries,
                "aggregate": aggregate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary_path = args.output_dir / "summary.md"
    summary_path.write_text(_render_summary_markdown(aggregate, args.suite, args.company_limit), encoding="utf-8")

    for summary in summaries:
        winner = summary.get("winner") or {}
        comparison = summary.get("taxonomy_vs_winner") or {}
        print(
            f"\n{summary['scenario']}: {winner.get('variant_id')} ({winner.get('style')}) "
            f"score={winner.get('composite_score')} | taxonomy rank {comparison.get('taxonomy_rank')}/6",
            flush=True,
        )
    print(f"\nSuite: {args.suite} | Companies: {args.company_limit}", flush=True)
    print(f"Winner styles: {aggregate['winner_style_counts']}", flush=True)
    print(f"Wrote reports to {args.output_dir}", flush=True)


def _render_summary_markdown(
    aggregate: dict[str, Any],
    suite: str,
    company_limit: int,
) -> str:
    lines = [
        "# Search Query Benchmark Summary",
        "",
        f"Suite: `{suite}`",
        f"Companies: {company_limit}",
        f"Scenarios: {aggregate['scenario_count']}",
        "",
        "## Winner style counts",
        "",
    ]
    for style, count in sorted(
        aggregate["winner_style_counts"].items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        lines.append(f"- {style}: {count}")
    lines.extend(["", "## Per scenario", "", "| Scenario | Winner | Style | Taxonomy rank | Score delta |", "| --- | --- | --- | ---: | ---: |"])
    for row in aggregate["scenarios"]:
        lines.append(
            f"| {row['scenario']} | {row['winner']} | {row['winner_style']} | "
            f"{row.get('taxonomy_rank', '')} | {row.get('score_delta', '')} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
