"""The demo portfolio: six documents, three contracts, one deliberate chain.

Contoso Systems Ltd. sits in the middle of a contract chain -- it buys
observability from Northwind and sells a platform to Acme. The gap between
what it promised outward and what it secured upstream is real and computed,
not scripted.
"""

from __future__ import annotations

import pathlib
from datetime import date

from api.ingest import ingest_text
from api.pipeline import ContractBundle, analyze_contract
from api.schemas import OurRole

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"

OUR_PARTY = "Contoso Systems Ltd."

PORTFOLIO = [
    {
        "id": "k_northwind",
        "title": "Northwind Observability MSA",
        "counterparty": "Northwind Observability, Inc.",
        "our_role": OurRole.BUYER,
        "files": ["msa_northwind.txt", "order_form_northwind.txt",
                  "amendment_2_northwind.txt"],
    },
    {
        "id": "k_acme",
        "title": "Acme Financial Master Subscription Agreement",
        "counterparty": "Acme Financial Group plc",
        "our_role": OurRole.SELLER,
        "files": ["customer_msa_acme.txt"],
    },
    {
        "id": "k_vertex",
        "title": "Vertex Cloud Systems MSA",
        "counterparty": "Vertex Cloud Systems Inc.",
        "our_role": OurRole.BUYER,
        "files": ["poisoned_msa_vertex.txt"],
    },
    {
        "id": "k_helios",
        "title": "Helios Partners Mutual NDA",
        "counterparty": "Helios Partners LLC",
        "our_role": OurRole.MUTUAL,
        "files": ["nda_helios.txt"],
    },
]


def load(today: date | None = None, use_cache: bool = True) -> list[ContractBundle]:
    today = today or date.today()
    bundles: list[ContractBundle] = []
    for spec in PORTFOLIO:
        docs = [
            ingest_text((CONTRACTS / name).read_text(), name)
            for name in spec["files"]
        ]
        bundles.append(
            analyze_contract(
                docs,
                title=spec["title"],
                counterparty=spec["counterparty"],
                our_role=spec["our_role"],
                our_party=OUR_PARTY,
                today=today,
                contract_id=spec["id"],
                use_cache=use_cache,
            )
        )
    return bundles
