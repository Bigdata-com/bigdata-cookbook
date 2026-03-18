from __future__ import annotations

from typing import TypeAlias

EntityDict: TypeAlias = dict[str, str | None]

LENDERS: list[EntityDict] = [
    {"name": "Blue Owl Capital", "ticker": "OWL", "layer": "lender"},
    {"name": "Ares Management", "ticker": "ARES", "layer": "lender"},
    {"name": "Blackstone Credit", "ticker": "BX", "layer": "lender"},
    {"name": "KKR Credit", "ticker": "KKR", "layer": "lender"},
    {"name": "FS KKR Capital", "ticker": "FSK", "layer": "lender"},
    {"name": "Apollo Global", "ticker": "APO", "layer": "lender"},
    {"name": "BlackRock HPS", "ticker": "BLK", "layer": "lender"},
    {"name": "Cliffwater", "ticker": None, "layer": "lender"},
    {"name": "Owl Rock Capital", "ticker": "ORCC", "layer": "lender"},
    {"name": "Prospect Capital", "ticker": "PSEC", "layer": "lender"},
    {"name": "Golub Capital BDC", "ticker": "GBDC", "layer": "lender"},
    {"name": "Blue Owl Technology Income", "ticker": "OTIC", "layer": "lender"},
]

BORROWERS: list[EntityDict] = [
    {"name": "Medallia", "ticker": None, "layer": "borrower"},
    {"name": "Peraton", "ticker": None, "layer": "borrower"},
    {"name": "Zendesk", "ticker": None, "layer": "borrower"},
    {"name": "Informatica", "ticker": "INFA", "layer": "borrower"},
    {"name": "Cotiviti", "ticker": None, "layer": "borrower"},
    {"name": "Dun & Bradstreet", "ticker": "DNB", "layer": "borrower"},
    {"name": "Cloudera", "ticker": None, "layer": "borrower"},
    {"name": "Epicor", "ticker": None, "layer": "borrower"},
    {"name": "Solera", "ticker": None, "layer": "borrower"},
    {"name": "First Brands", "ticker": None, "layer": "borrower"},
]

BANKS: list[EntityDict] = [
    {"name": "JPMorgan Chase", "ticker": "JPM", "layer": "bank"},
    {"name": "Goldman Sachs", "ticker": "GS", "layer": "bank"},
    {"name": "Morgan Stanley", "ticker": "MS", "layer": "bank"},
    {"name": "Barclays", "ticker": "BCS", "layer": "bank"},
    {"name": "Wells Fargo", "ticker": "WFC", "layer": "bank"},
]

ALL_ENTITIES: list[EntityDict] = LENDERS + BORROWERS + BANKS
