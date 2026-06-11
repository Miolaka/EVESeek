import math
from decimal import Decimal


def ceil_qty(base_qty: int, runs: int, me_level: int, structure_bonus: float = 0.0) -> int:
    """Apply ME + structure reduction per run, then multiply by run count."""
    per_run = max(1, math.ceil(base_qty * (1 - me_level * 0.01) * (1 - structure_bonus)))
    return per_run * runs


def broker_fee_rate(
    broker_relations: int,
    faction_standing: float,
    corp_standing: float,
) -> Decimal:
    rate = (
        Decimal("0.03")
        - Decimal("0.003") * broker_relations
        - Decimal("0.0003") * Decimal(str(faction_standing))
        - Decimal("0.0002") * Decimal(str(corp_standing))
    )
    return max(rate, Decimal("0"))


def job_cost_manufacturing(
    adjusted_price: Decimal,
    system_cost_index: Decimal,
    structure_bonus: float,
    facility_tax: Decimal,
    fw_level: int,
) -> Decimal:
    """Eve manufacturing/reaction job fee formula."""
    SCC_SURCHARGE = Decimal("0.04")
    base = adjusted_price * (
        system_cost_index * Decimal(str(1 - structure_bonus))
        + facility_tax
        + SCC_SURCHARGE
    )
    return base * Decimal(str(1 - fw_level * 0.1))


def job_cost_refining(
    ore_adjusted_price: Decimal,
    system_cost_index: Decimal,
    structure_bonus: float,
    fw_level: int,
) -> Decimal:
    """Eve refining job fee formula (no facility tax)."""
    base = ore_adjusted_price * system_cost_index * Decimal(str(1 - structure_bonus))
    return base * Decimal(str(1 - fw_level * 0.1))
