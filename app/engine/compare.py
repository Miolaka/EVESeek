import math
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from app.data import sde
from app.esi import client as esi
from app.core.utils import job_cost_refining, broker_fee_rate
from app.engine.bom import build_cost, _STATION_MAP
from app.models.schemas import (
    BuildCostRequest,
    BOMNode,
    CompareMaterialSourceRequest,
    CompareMaterialSourceResponse,
    DirectBuyItem,
    DirectBuyPath,
    CompressedOreItem,
    CompressedOrePath,
    LeftoverItem,
)

# Leftovers are always sold at Jita buy regardless of the user's material_source.
_JITA_STATION_ID = 60003760
_JITA_REGION_ID = 10000002


def _prefetch_ore_orders(leaf_type_ids: set[int], buy_region_id: int) -> None:
    """Fetch market orders for all compressed ores + their mineral byproducts in parallel.

    Fills the disk + in-memory cache before the main compare loop runs, turning all
    subsequent get_station_fill_price calls in that loop into cache hits.
    """
    ore_type_ids: set[int] = set()
    mineral_ids: set[int] = set(leaf_type_ids)

    for mid in leaf_type_ids:
        for src in sde.get_ore_sources_for_mineral(mid):
            oid = src["ore_type_id"]
            ore_type_ids.add(oid)
            for out in sde.get_refining_outputs(oid):
                mineral_ids.add(out["materialTypeID"])

    # (region_id, type_id) pairs to fetch — deduplicated
    pairs: set[tuple[int, int]] = set()
    for oid in ore_type_ids:
        pairs.add((buy_region_id, oid))
    for mid in mineral_ids:
        pairs.add((_JITA_REGION_ID, mid))
        if buy_region_id != _JITA_REGION_ID:
            pairs.add((buy_region_id, mid))

    def _fetch(args: tuple[int, int]) -> None:
        try:
            esi.get_market_orders(args[0], args[1])
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=20) as pool:
        pool.map(_fetch, pairs)


def _collect_leaves(node: BOMNode, acc: dict[int, tuple[str, int]]) -> None:
    """Walk BOM tree, accumulate {type_id: (name, total_quantity)} for all leaf nodes."""
    if not node.children:
        if node.type_id in acc:
            acc[node.type_id] = (node.name, acc[node.type_id][1] + node.quantity)
        else:
            acc[node.type_id] = (node.name, node.quantity)
        return
    for child in node.children:
        _collect_leaves(child, acc)


def _best_ore_for_mineral(
    mineral_type_id: int,
    mineral_name: str,
    mineral_qty: int,
    station_id: int,
    region_id: int,
    req: CompareMaterialSourceRequest,
) -> tuple[CompressedOreItem | None, list[LeftoverItem]]:
    """Find the compressed ore with the lowest effective cost for this mineral quantity.

    Byproduct credit (including excess of target mineral from ceil rounding) is priced
    at Jita buy orders — what you'd actually receive when selling the surplus.
    Returns (best_ore_item, leftover_items) or (None, []) if no ore source found.
    """
    ore_sources = sde.get_ore_sources_for_mineral(mineral_type_id)
    if not ore_sources:
        return None, []

    best: CompressedOreItem | None = None
    best_leftovers: list[LeftoverItem] = []
    best_effective_cost = Decimal("Inf")

    for source in ore_sources:
        ore_type_id = source["ore_type_id"]
        portion_size = source["portionSize"] or 100
        max_minerals_per_batch = source["max_minerals_per_batch"]

        minerals_per_batch = math.floor(max_minerals_per_batch * req.reprocessing_yield)
        if minerals_per_batch == 0:
            continue

        batches = math.ceil(mineral_qty / minerals_per_batch)
        ore_units = batches * portion_size

        ore_unit_price = esi.get_station_fill_price(
            region_id, ore_type_id, station_id, ore_units, False
        )
        if ore_unit_price is None:
            continue

        # Credit, collect leftovers, and sum post-refining volume in one pass.
        all_outputs = sde.get_refining_outputs(ore_type_id)
        byproduct_credit = Decimal("0")
        candidate_leftovers: list[LeftoverItem] = []
        refined_m3 = Decimal("0")

        for output in all_outputs:
            out_type_id = output["materialTypeID"]
            output_qty = math.floor(output["quantity"] * req.reprocessing_yield) * batches
            if output_qty <= 0:
                continue

            # Refined volume: total mineral output regardless of target/byproduct split
            mat_row = sde.get_type(out_type_id)
            if mat_row and mat_row["volume"]:
                refined_m3 += Decimal(str(mat_row["volume"])) * output_qty

            if out_type_id == mineral_type_id:
                leftover_qty = output_qty - mineral_qty  # excess from ceil
            else:
                leftover_qty = output_qty  # full byproduct amount

            if leftover_qty <= 0:
                continue

            buy_price = esi.get_station_fill_price(
                _JITA_REGION_ID, out_type_id, _JITA_STATION_ID, leftover_qty, True
            ) or Decimal("0")

            byproduct_credit += buy_price * leftover_qty

            mat_name = mat_row["typeName"] if mat_row else f"Unknown ({out_type_id})"
            candidate_leftovers.append(LeftoverItem(
                type_id=out_type_id,
                name=mat_name,
                quantity=leftover_qty,
                buy_price=buy_price,
                total_isk=buy_price * leftover_qty,
            ))

        ore_adj = esi.get_adjusted_price(ore_type_id) or Decimal("0")
        refining_fee = (
            job_cost_refining(ore_adj, req.reprocessing_rate, req.refinery_bonus, req.fw_level)
            * ore_units
        )

        ore_type_row = sde.get_type(ore_type_id)
        ore_vol = (
            Decimal(str(ore_type_row["volume"]))
            if ore_type_row and ore_type_row["volume"]
            else Decimal("0")
        )

        total_isk = ore_unit_price * ore_units
        effective_isk = total_isk + refining_fee - byproduct_credit

        if effective_isk < best_effective_cost:
            best_effective_cost = effective_isk
            best_leftovers = candidate_leftovers
            best = CompressedOreItem(
                ore_type_id=ore_type_id,
                ore_name=source["typeName"],
                for_mineral_type_id=mineral_type_id,
                for_mineral_name=mineral_name,
                quantity=ore_units,
                unit_price=ore_unit_price,
                total_isk=total_isk,
                refining_fee=refining_fee,
                byproduct_credit=byproduct_credit,
                effective_isk=effective_isk,
                volume_m3=ore_vol * ore_units,
                refined_m3=refined_m3,
            )

    return best, best_leftovers


def compare_material_source(req: CompareMaterialSourceRequest) -> CompareMaterialSourceResponse:
    bom_req = BuildCostRequest(
        type_id=req.type_id,
        system_id=req.system_id,
        me_level=req.me_level,
        me_overrides=req.me_overrides,
        structure_bonus=req.structure_bonus,
        fw_level=req.fw_level,
        runs=req.runs,
        material_source=req.material_source,
        facility_tax=req.facility_tax,
        logistics_cost_isk_per_m3=Decimal("0"),
        broker_relations_level=req.broker_relations_level,
        faction_standing=req.faction_standing,
        corp_standing=req.corp_standing,
    )
    bom_result = build_cost(bom_req)

    station_id, is_buy = _STATION_MAP[req.material_source]
    region_id = esi.STATION_REGION[station_id]

    leaves: dict[int, tuple[str, int]] = {}
    for node in bom_result.bom_tree:
        _collect_leaves(node, leaves)

    # Pre-warm cache for all compressed ores and their byproducts in parallel
    # before the sequential per-mineral loop runs.
    _prefetch_ore_orders(set(leaves.keys()), region_id)

    direct_items: list[DirectBuyItem] = []
    ore_items: list[CompressedOreItem] = []
    non_mineral_direct: list[DirectBuyItem] = []

    for type_id, (name, qty) in leaves.items():
        type_row = sde.get_type(type_id)
        vol = (
            Decimal(str(type_row["volume"]))
            if type_row and type_row["volume"]
            else Decimal("0")
        )
        unit_price = esi.get_station_fill_price(region_id, type_id, station_id, qty, is_buy) or Decimal("0")
        if is_buy:
            fee = broker_fee_rate(req.broker_relations_level, req.faction_standing, req.corp_standing)
            unit_price = unit_price * (1 + fee)

        direct_items.append(DirectBuyItem(
            type_id=type_id,
            name=name,
            quantity=qty,
            unit_price=unit_price,
            total_isk=unit_price * qty,
            volume_m3=vol * qty,
        ))

        ore_item, _ = _best_ore_for_mineral(type_id, name, qty, station_id, region_id, req)
        if ore_item is not None:
            ore_items.append(ore_item)
        else:
            non_mineral_direct.append(DirectBuyItem(
                type_id=type_id,
                name=name,
                quantity=qty,
                unit_price=unit_price,
                total_isk=unit_price * qty,
                volume_m3=vol * qty,
            ))

    # True leftover: sum all outputs from every chosen ore, then subtract what the
    # build actually needs. A byproduct that is also a needed mineral is only surplus
    # to the extent it exceeds the required quantity.
    total_produced: dict[int, tuple[str, int]] = {}
    for ore_item in ore_items:
        ore_type_row = sde.get_type(ore_item.ore_type_id)
        portion_size = int(ore_type_row["portionSize"]) if ore_type_row and ore_type_row["portionSize"] else 100
        batches = ore_item.quantity // portion_size
        for output in sde.get_refining_outputs(ore_item.ore_type_id):
            out_id = output["materialTypeID"]
            output_qty = math.floor(output["quantity"] * req.reprocessing_yield) * batches
            if output_qty <= 0:
                continue
            mat_row = sde.get_type(out_id)
            mat_name = mat_row["typeName"] if mat_row else f"Unknown ({out_id})"
            _, prev = total_produced.get(out_id, (mat_name, 0))
            total_produced[out_id] = (mat_name, prev + output_qty)

    leftover_items: list[LeftoverItem] = []
    for type_id, (name, qty_produced) in total_produced.items():
        qty_needed = leaves.get(type_id, ("", 0))[1]
        surplus = qty_produced - qty_needed
        if surplus <= 0:
            continue
        buy_price = esi.get_station_fill_price(
            _JITA_REGION_ID, type_id, _JITA_STATION_ID, surplus, True
        ) or Decimal("0")
        leftover_items.append(LeftoverItem(
            type_id=type_id,
            name=name,
            quantity=surplus,
            buy_price=buy_price,
            total_isk=buy_price * surplus,
        ))
    leftover_total_isk = sum((i.total_isk for i in leftover_items), Decimal("0"))

    direct_total_isk = sum((i.total_isk for i in direct_items), Decimal("0"))
    direct_total_m3 = sum((i.volume_m3 for i in direct_items), Decimal("0"))

    non_min_isk = sum((i.total_isk for i in non_mineral_direct), Decimal("0"))
    ore_total_isk = sum((i.total_isk for i in ore_items), Decimal("0")) + non_min_isk
    non_min_m3 = sum((i.volume_m3 for i in non_mineral_direct), Decimal("0"))
    ore_total_m3 = sum((i.volume_m3 for i in ore_items), Decimal("0")) + non_min_m3
    ore_refined_total_m3 = sum((i.refined_m3 for i in ore_items), Decimal("0")) + non_min_m3
    ore_refining_fee = sum((i.refining_fee for i in ore_items), Decimal("0"))
    ore_effective_isk = sum((i.effective_isk for i in ore_items), Decimal("0")) + non_min_isk

    return CompareMaterialSourceResponse(
        direct_buy=DirectBuyPath(
            total_isk=direct_total_isk,
            total_m3=direct_total_m3,
            items=direct_items,
        ),
        compressed_ore=CompressedOrePath(
            total_isk=ore_total_isk,
            effective_isk=ore_effective_isk,
            total_m3=ore_total_m3,
            refined_total_m3=ore_refined_total_m3,
            refining_fee=ore_refining_fee,
            ore_items=ore_items,
            direct_items=non_mineral_direct,
            leftover_items=leftover_items,
            leftover_total_isk=leftover_total_isk,
        ),
    )
