import math
from decimal import Decimal
from app.data import sde
from app.esi import client as esi
from app.core.utils import (
    ceil_qty,
    broker_fee_rate,
    job_cost_manufacturing,
)
from app.models.schemas import BOMNode, BuildCostRequest, CostBreakdown

ACTIVITY_MANUFACTURING = 1
ACTIVITY_REACTION = 11

_RAW_CATEGORY_IDS = {
    25,   # Asteroid (ore) — no blueprint, stops via blueprint is None too
    18,   # Drone
}
_RAW_GROUP_IDS = {
    711,  # Moon Materials (raw)
    423,  # Ice Products
    426,  # Compressed Ice
    873,  # Gas Cloud
}

# Maps material_source → (station_id, is_buy_order)
_STATION_MAP: dict[str, tuple[int, bool]] = {
    "jita_sell":  (60003760, False),
    "jita_buy":   (60003760, True),
    "amarr_sell": (60008494, False),
    "amarr_buy":  (60008494, True),
}


def _zero_breakdown() -> CostBreakdown:
    z = Decimal("0")
    return CostBreakdown(
        material_costs=z,
        manufacturing_fees=z,
        reaction_fees=z,
        refining_fees=z,
        logistics_costs=z,
    )


def _add_breakdowns(a: CostBreakdown, b: CostBreakdown) -> CostBreakdown:
    return CostBreakdown(
        material_costs=a.material_costs + b.material_costs,
        manufacturing_fees=a.manufacturing_fees + b.manufacturing_fees,
        reaction_fees=a.reaction_fees + b.reaction_fees,
        refining_fees=a.refining_fees + b.refining_fees,
        logistics_costs=a.logistics_costs + b.logistics_costs,
    )


def _market_price(type_id: int, quantity: int, req: BuildCostRequest) -> Decimal:
    station_id, is_buy = _STATION_MAP[req.material_source]
    region_id = esi.STATION_REGION[station_id]
    price = esi.get_station_fill_price(region_id, type_id, station_id, quantity, is_buy)
    if price is None:
        return Decimal("0")
    if is_buy:
        fee = broker_fee_rate(
            req.broker_relations_level,
            req.faction_standing,
            req.corp_standing,
        )
        price = price * (1 + fee)
    return price


def _logistics(volume: Decimal, quantity: int, req: BuildCostRequest) -> Decimal:
    return volume * quantity * req.logistics_cost_isk_per_m3


def _is_raw_category(type_row) -> bool:
    group_row = sde.get_group(type_row["groupID"])
    return group_row is not None and group_row["categoryID"] in _RAW_CATEGORY_IDS


def _build_node(
    type_id: int,
    quantity: int,
    req: BuildCostRequest,
    depth: int = 0,
) -> BOMNode:
    type_row = sde.get_type(type_id)
    name = type_row["typeName"] if type_row else f"Unknown ({type_id})"
    volume = Decimal(str(type_row["volume"])) if type_row and type_row["volume"] else Decimal("0")
    group_id = type_row["groupID"] if type_row else None

    breakdown = _zero_breakdown()
    children: list[BOMNode] = []

    blueprint = sde.get_blueprint_for_product(type_id)
    is_raw = (
        group_id in _RAW_GROUP_IDS
        or (type_row and type_row["groupID"] and _is_raw_category(type_row))
        or blueprint is None
        or depth >= 10
    )

    if is_raw:
        unit_price = _market_price(type_id, quantity, req)
        total_cost = unit_price * quantity
        breakdown.material_costs = total_cost
        breakdown.logistics_costs = _logistics(volume, quantity, req)
        total_cost += breakdown.logistics_costs
        return BOMNode(
            type_id=type_id,
            name=name,
            quantity=quantity,
            cost_per_unit=unit_price,
            total_cost=total_cost,
            cost_breakdown=breakdown,
            children=[],
        )

    bp_type_id = blueprint["blueprint_type_id"]
    activity_id = blueprint["activityID"]
    qty_per_run = blueprint["quantity"] or 1

    max_runs_per_bpc = sde.get_max_production_limit(bp_type_id)
    bpc_copies_needed = math.ceil(quantity / (max_runs_per_bpc * qty_per_run))

    me = req.me_overrides.get(type_id, req.me_level)
    materials = sde.get_activity_materials(bp_type_id, activity_id)

    for mat in materials:
        mat_type_id = mat["materialTypeID"]
        mat_qty = ceil_qty(mat["quantity"], req.runs, me)
        child = _build_node(mat_type_id, mat_qty, req, depth + 1)
        children.append(child)
        breakdown = _add_breakdowns(breakdown, child.cost_breakdown)

    # EIV = sum of adjusted prices of input materials × their base quantities
    eiv = sum(
        (esi.get_adjusted_price(m["materialTypeID"]) or Decimal("0")) * m["quantity"]
        for m in materials
    )
    cost_indices = esi.get_system_cost_index(req.system_id)

    if activity_id == ACTIVITY_MANUFACTURING:
        ci = cost_indices.get("manufacturing", Decimal("0"))
        fee = job_cost_manufacturing(
            eiv, ci,
            req.structure_bonus, req.facility_tax,
            req.fw_level,
        ) * req.runs
        breakdown.manufacturing_fees += fee

    elif activity_id == ACTIVITY_REACTION:
        ci = cost_indices.get("reaction", Decimal("0"))
        fee = job_cost_manufacturing(
            eiv, ci,
            req.structure_bonus, Decimal("0"),
            req.fw_level,
        ) * req.runs
        breakdown.reaction_fees += fee

    breakdown.logistics_costs += _logistics(volume, quantity, req)

    total_cost = (
        breakdown.material_costs
        + breakdown.manufacturing_fees
        + breakdown.reaction_fees
        + breakdown.refining_fees
        + breakdown.logistics_costs
    )

    return BOMNode(
        type_id=type_id,
        name=name,
        quantity=quantity,
        cost_per_unit=total_cost / quantity if quantity else Decimal("0"),
        total_cost=total_cost,
        cost_breakdown=breakdown,
        children=children,
        bpc_copies_needed=bpc_copies_needed,
        max_runs_per_bpc=max_runs_per_bpc,
    )


def build_cost(req: BuildCostRequest):
    from app.models.schemas import BuildCostResponse

    type_row = sde.get_type(req.type_id)
    if not type_row:
        raise ValueError(f"Unknown type_id: {req.type_id}")

    root = _build_node(req.type_id, req.runs, req)

    return BuildCostResponse(
        type_id=req.type_id,
        item_name=type_row["typeName"],
        total_cost=root.total_cost,
        cost_breakdown=root.cost_breakdown,
        bom_tree=root.children,
    )
