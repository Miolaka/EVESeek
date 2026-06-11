import math
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from app.data import sde
from app.esi import client as esi
from app.core.utils import (
    ceil_qty,
    broker_fee_rate,
    job_cost_manufacturing,
)
from app.models.schemas import BOMNode, BPOInfo, BuildCostRequest, CostBreakdown
from app.engine.item_classifier import get_slot

ACTIVITY_MANUFACTURING = 1
ACTIVITY_REACTION = 11

_RAW_CATEGORY_IDS = {
    25,   # Asteroid (ore) — no blueprint, stops via blueprint is None too
    18,   # Drone
}
_RAW_GROUP_IDS = {
    427,  # Moon Materials (Atmospheric Gases, Evaporite Deposits, Tungsten, Platinum, etc.)
    423,  # Ice Products (Heavy Water, Liquid Ozone, isotopes, Strontium Clathrates)
    711,  # Harvestable Cloud (gas cloud harvesting: Fullerites, etc.)
}
_SHIP_CATEGORY_ID = 6

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


def _effective_me_bonus(req: BuildCostRequest, type_id: int, activity_id: int) -> float:
    """Return the structure ME bonus fraction for this node."""
    if not req.activity_me_bonus:
        return req.structure_bonus   # backward compat: no config → use global field
    slot = get_slot(type_id, activity_id)
    return req.activity_me_bonus.get(slot, req.structure_bonus)


def _is_raw_category(type_row) -> bool:
    group_row = sde.get_group(type_row["groupID"])
    return group_row is not None and group_row["categoryID"] in _RAW_CATEGORY_IDS


def _is_ship(type_row) -> bool:
    group_row = sde.get_group(type_row["groupID"])
    return group_row is not None and group_row["categoryID"] == _SHIP_CATEGORY_ID


def _collect_leaf_type_ids(
    type_id: int,
    req: BuildCostRequest,
    depth: int = 0,
    seen: set[int] | None = None,
) -> set[int]:
    """Walk the blueprint tree via SDE only and return every leaf type_id."""
    if seen is None:
        seen = set()
    if type_id in seen:
        return set()
    seen.add(type_id)

    type_row = sde.get_type(type_id)
    group_id = type_row["groupID"] if type_row else None
    blueprint = sde.get_blueprint_for_product(type_id)

    is_raw = (
        group_id in _RAW_GROUP_IDS
        or (type_row and type_row["groupID"] and _is_raw_category(type_row))
        or (type_row and type_row["groupID"] and depth > 0
            and not req.build_t1_hull and _is_ship(type_row))
        or blueprint is None
        or depth >= 10
    )

    if is_raw:
        return {type_id}

    bp_type_id = blueprint["blueprint_type_id"]
    activity_id = blueprint["activityID"]
    materials = sde.get_activity_materials(bp_type_id, activity_id)

    leaf_ids: set[int] = set()
    for mat in materials:
        leaf_ids |= _collect_leaf_type_ids(mat["materialTypeID"], req, depth + 1, seen)
    return leaf_ids


def _prefetch_leaf_prices(leaf_ids: set[int], req: BuildCostRequest) -> None:
    """Warm the ESI order cache for all leaf nodes in parallel before the BOM walk."""
    station_id, _ = _STATION_MAP[req.material_source]
    region_id = esi.STATION_REGION[station_id]

    def _fetch(type_id: int) -> None:
        try:
            esi.get_market_orders(region_id, type_id)
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=20) as pool:
        pool.map(_fetch, leaf_ids)


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
        or (type_row and type_row["groupID"] and depth > 0 and not req.build_t1_hull and _is_ship(type_row))
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

    runs_needed = math.ceil(quantity / qty_per_run)
    max_runs_per_bpc = sde.get_max_production_limit(bp_type_id)
    bpc_copies_needed = math.ceil(runs_needed / max_runs_per_bpc)

    if activity_id == ACTIVITY_REACTION:
        default_me = 0
    elif depth == 0:
        default_me = req.me_level
    else:
        default_me = 10
    me = req.me_overrides.get(type_id, default_me)
    materials = sde.get_activity_materials(bp_type_id, activity_id)
    node_sb = _effective_me_bonus(req, type_id, activity_id)

    for mat in materials:
        mat_type_id = mat["materialTypeID"]
        mat_qty = ceil_qty(mat["quantity"], runs_needed, me, node_sb)
        child = _build_node(mat_type_id, mat_qty, req, depth + 1)
        children.append(child)
        breakdown = _add_breakdowns(breakdown, child.cost_breakdown)

    # EIV = sum of adjusted prices of input materials × their base quantities × runs
    eiv = sum(
        (esi.get_adjusted_price(m["materialTypeID"]) or Decimal("0")) * m["quantity"]
        for m in materials
    ) * runs_needed
    cost_indices = esi.get_system_cost_index(req.system_id)

    if activity_id == ACTIVITY_MANUFACTURING:
        ci = cost_indices.get("manufacturing", Decimal("0"))
        fee = job_cost_manufacturing(
            eiv, ci,
            req.structure_bonus, req.facility_tax,
            req.fw_level,
        )
        breakdown.manufacturing_fees += fee

    elif activity_id == ACTIVITY_REACTION:
        ci = cost_indices.get("reaction", Decimal("0"))
        fee = job_cost_manufacturing(
            eiv, ci,
            req.structure_bonus, Decimal("0"),
            req.fw_level,
        )
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


def _compute_flat_bom(root_type_id: int, root_qty: int, req: BuildCostRequest) -> dict[int, int]:
    """
    Flat BOM explosion using topological sort.

    The recursive _build_node approach fragments demand: when NFB is needed by
    TC, RTA and SA independently, each branch calls ceil(qty/40) separately, so
    19 + 9 + 9 = 37 runs instead of the correct ceil(1200/40) = 30.

    This function collects ALL demand for every intermediate type BEFORE
    computing its production runs, eliminating that over-count.

    Returns {leaf_type_id: total_quantity} for every market-buy item.
    """
    # ── Phase 1: discover the full BOM graph (no quantities, just topology) ──
    node_info: dict[int, dict] = {}
    discover: deque[tuple[int, int]] = deque([(root_type_id, 0)])
    visited: set[int] = set()

    while discover:
        tid, depth = discover.popleft()
        if tid in visited:
            continue
        visited.add(tid)

        type_row = sde.get_type(tid)
        gid = type_row["groupID"] if type_row else None

        is_raw = (
            not type_row
            or gid in _RAW_GROUP_IDS
            or (type_row and _is_raw_category(type_row))
            or (depth > 0 and not req.build_t1_hull and type_row and _is_ship(type_row))
        )

        bp = None if is_raw else sde.get_blueprint_for_product(tid)

        if is_raw or bp is None:
            node_info[tid] = {"leaf": True}
            continue

        mats = list(sde.get_activity_materials(bp["blueprint_type_id"], bp["activityID"]))
        node_info[tid] = {"leaf": False, "bp": bp, "mats": mats}

        for mat in mats:
            mid = mat["materialTypeID"]
            if mid not in visited:
                discover.append((mid, depth + 1))

    # ── Phase 2: build in-degree map for Kahn's topological sort ──
    manufactured: set[int] = {tid for tid, info in node_info.items() if not info["leaf"]}

    in_degree: dict[int, int] = {tid: 0 for tid in manufactured}
    for parent in manufactured:
        seen: set[int] = set()
        for mat in node_info[parent]["mats"]:
            child = mat["materialTypeID"]
            if child in manufactured and child not in seen:
                in_degree[child] += 1
                seen.add(child)

    # ── Phase 3: demand explosion in topological order ──
    demands: dict[int, int] = defaultdict(int)
    demands[root_type_id] = root_qty
    leaf_demands: dict[int, int] = defaultdict(int)

    queue: deque[int] = deque(tid for tid in manufactured if in_degree[tid] == 0)

    while queue:
        type_id = queue.popleft()
        info = node_info[type_id]
        total_qty = demands[type_id]

        bp = info["bp"]
        qty_per_run = bp["quantity"] or 1
        activity_id = bp["activityID"]
        runs_needed = math.ceil(total_qty / qty_per_run)

        if activity_id == ACTIVITY_REACTION:
            default_me = 0
        elif type_id == root_type_id:
            default_me = req.me_level
        else:
            default_me = 10
        me = req.me_overrides.get(type_id, default_me)
        node_sb = _effective_me_bonus(req, type_id, activity_id)

        for mat in info["mats"]:
            mid = mat["materialTypeID"]
            mat_qty = ceil_qty(mat["quantity"], runs_needed, me, node_sb)
            demands[mid] += mat_qty

            if mid in manufactured:
                in_degree[mid] -= 1
                if in_degree[mid] == 0:
                    queue.append(mid)
            else:
                leaf_demands[mid] += mat_qty

    return dict(leaf_demands)


def _discover_manufactured_nodes(root_type_id: int, req: BuildCostRequest) -> list[BPOInfo]:
    """BFS over the BOM graph; returns one BPOInfo per manufactured (non-leaf) node."""
    result: list[BPOInfo] = []
    discover: deque[tuple[int, int]] = deque([(root_type_id, 0)])
    visited: set[int] = set()

    while discover:
        tid, depth = discover.popleft()
        if tid in visited:
            continue
        visited.add(tid)

        type_row = sde.get_type(tid)
        is_raw = (
            not type_row
            or type_row["groupID"] in _RAW_GROUP_IDS
            or _is_raw_category(type_row)
            or (depth > 0 and not req.build_t1_hull and _is_ship(type_row))
        )
        bp = None if is_raw else sde.get_blueprint_for_product(tid)
        if is_raw or bp is None:
            continue

        activity_id = bp["activityID"]
        if activity_id == ACTIVITY_REACTION:
            me = 0
        elif tid == root_type_id:
            me = req.me_level
        else:
            me = req.me_overrides.get(tid, 10)

        result.append(BPOInfo(
            type_id=tid,
            name=type_row["typeName"],
            activity_id=activity_id,
            me_level=me,
            is_root=(tid == root_type_id),
        ))

        for mat in sde.get_activity_materials(bp["blueprint_type_id"], activity_id):
            if mat["materialTypeID"] not in visited:
                discover.append((mat["materialTypeID"], depth + 1))

    return result


def build_cost(req: BuildCostRequest):
    from app.models.schemas import BuildCostResponse

    type_row = sde.get_type(req.type_id)
    if not type_row:
        raise ValueError(f"Unknown type_id: {req.type_id}")

    # Flat BOM gives correct aggregated leaf quantities (no fragmentation)
    flat_leaves = _compute_flat_bom(req.type_id, req.runs, req)

    # Pre-warm ESI cache for all leaves before any pricing calls
    leaf_ids = _collect_leaf_type_ids(req.type_id, req)
    _prefetch_leaf_prices(leaf_ids | set(flat_leaves.keys()), req)

    # Price leaves from flat BOM → correct material + logistics costs
    flat_material_cost = Decimal("0")
    flat_logistics_cost = Decimal("0")
    for tid, qty in flat_leaves.items():
        unit_price = _market_price(tid, qty, req)
        flat_material_cost += unit_price * qty
        t = sde.get_type(tid)
        vol = Decimal(str(t["volume"])) if t and t["volume"] else Decimal("0")
        flat_logistics_cost += _logistics(vol, qty, req)

    # Recursive tree for BOM display and fee computation (fees per manufactured job)
    root = _build_node(req.type_id, req.runs, req)

    corrected = CostBreakdown(
        material_costs=flat_material_cost,
        manufacturing_fees=root.cost_breakdown.manufacturing_fees,
        reaction_fees=root.cost_breakdown.reaction_fees,
        refining_fees=root.cost_breakdown.refining_fees,
        logistics_costs=flat_logistics_cost,
    )

    bpo_list = _discover_manufactured_nodes(req.type_id, req)

    return BuildCostResponse(
        type_id=req.type_id,
        item_name=type_row["typeName"],
        total_cost=(
            flat_material_cost
            + root.cost_breakdown.manufacturing_fees
            + root.cost_breakdown.reaction_fees
            + root.cost_breakdown.refining_fees
            + flat_logistics_cost
        ),
        cost_breakdown=corrected,
        bom_tree=root.children,
        bpo_list=bpo_list,
    )
