import math
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from app.data import sde
from app.esi import client as esi
from app.core.utils import job_cost_refining, broker_fee_rate
from app.engine.bom import build_cost, _compute_flat_bom, _STATION_MAP
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

_JITA_STATION_ID = 60003760
_JITA_REGION_ID = 10000002


def _prefetch_ore_orders(leaf_type_ids: set[int], buy_region_id: int) -> None:
    """Fetch market orders for all compressed ores + their mineral byproducts in parallel."""
    ore_type_ids: set[int] = set()
    mineral_ids: set[int] = set(leaf_type_ids)

    for mid in leaf_type_ids:
        for src in sde.get_ore_sources_for_mineral(mid):
            oid = src["ore_type_id"]
            ore_type_ids.add(oid)
            for out in sde.get_refining_outputs(oid):
                mineral_ids.add(out["materialTypeID"])

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
    if not node.children:
        if node.type_id in acc:
            acc[node.type_id] = (node.name, acc[node.type_id][1] + node.quantity)
        else:
            acc[node.type_id] = (node.name, node.quantity)
        return
    for child in node.children:
        _collect_leaves(child, acc)


def _ore_candidates_for_mineral(
    mineral_type_id: int,
    mineral_name: str,
    mineral_qty: int,
    station_id: int,
    region_id: int,
    req: CompareMaterialSourceRequest,
) -> list[CompressedOreItem]:
    """Return all valid compressed ore candidates sorted by logistics-adjusted effective_isk.

    Byproduct credit is netted against the cost to haul each surplus unit to Jita, so ores
    with bulky byproducts become less attractive when leftover_logistics_isk_per_m3 > 0.
    """
    ore_sources = sde.get_ore_sources_for_mineral(mineral_type_id)
    if not ore_sources:
        return []

    candidates: list[CompressedOreItem] = []

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

        all_outputs = sde.get_refining_outputs(ore_type_id)
        byproduct_credit = Decimal("0")
        refined_m3 = Decimal("0")

        for output in all_outputs:
            out_type_id = output["materialTypeID"]
            output_qty = math.floor(output["quantity"] * req.reprocessing_yield) * batches
            if output_qty <= 0:
                continue

            mat_row = sde.get_type(out_type_id)
            mat_vol = Decimal(str(mat_row["volume"])) if mat_row and mat_row["volume"] else Decimal("0")
            refined_m3 += mat_vol * output_qty

            if out_type_id == mineral_type_id:
                leftover_qty = output_qty - mineral_qty
            else:
                leftover_qty = output_qty

            if leftover_qty <= 0:
                continue

            buy_price = esi.get_station_fill_price(
                _JITA_REGION_ID, out_type_id, _JITA_STATION_ID, leftover_qty, True
            ) or Decimal("0")

            # Net credit: Jita buy value minus cost to haul the surplus to market
            logistics_per_unit = mat_vol * req.leftover_logistics_isk_per_m3
            net_credit_per_unit = max(Decimal("0"), buy_price - logistics_per_unit)
            byproduct_credit += net_credit_per_unit * leftover_qty

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

        candidates.append(CompressedOreItem(
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
        ))

    candidates.sort(key=lambda c: c.effective_isk)
    return candidates


def _compute_global_leftover(
    ore_items: list[CompressedOreItem],
    leaves: dict[int, tuple[str, int]],
    leftover_logistics_isk_per_m3: Decimal,
    reprocessing_yield: float,
    ore_portion: dict[int, int],
    ore_outputs: dict[int, list],
    item_meta: dict[int, tuple[str, Decimal]],
) -> tuple[list[LeftoverItem], Decimal, Decimal, Decimal]:
    """Compute true global leftover across all chosen ores.

    ore_portion, ore_outputs, item_meta are pre-computed SDE lookups — pass them in so
    this function does no SQLite I/O and can be called thousands of times in the swap loop.

    Returns (leftover_items, leftover_total_isk, leftover_logistics_isk, leftover_net_isk).
    """
    total_produced: dict[int, tuple[str, int]] = {}
    for ore_item in ore_items:
        portion_size = ore_portion.get(ore_item.ore_type_id, 100)
        batches = ore_item.quantity // portion_size
        for output in ore_outputs.get(ore_item.ore_type_id, []):
            out_id = output["materialTypeID"]
            output_qty = math.floor(output["quantity"] * reprocessing_yield) * batches
            if output_qty <= 0:
                continue
            mat_name, _ = item_meta.get(out_id, (f"Unknown ({out_id})", Decimal("0")))
            _, prev = total_produced.get(out_id, (mat_name, 0))
            total_produced[out_id] = (mat_name, prev + output_qty)

    leftover_items: list[LeftoverItem] = []
    leftover_total_isk = Decimal("0")
    leftover_logistics_isk = Decimal("0")

    for type_id, (name, qty_produced) in total_produced.items():
        qty_needed = leaves.get(type_id, ("", 0))[1]
        surplus = qty_produced - qty_needed
        if surplus <= 0:
            continue

        buy_price = esi.get_station_fill_price(
            _JITA_REGION_ID, type_id, _JITA_STATION_ID, surplus, True
        ) or Decimal("0")

        _, vol_per_unit = item_meta.get(type_id, ("", Decimal("0")))
        total_vol = vol_per_unit * surplus
        logistics_isk = total_vol * leftover_logistics_isk_per_m3
        gross_isk = buy_price * surplus
        net_isk = max(Decimal("0"), gross_isk - logistics_isk)

        leftover_total_isk += gross_isk
        leftover_logistics_isk += logistics_isk

        leftover_items.append(LeftoverItem(
            type_id=type_id,
            name=name,
            quantity=surplus,
            buy_price=buy_price,
            total_isk=gross_isk,
            volume_m3=total_vol,
            logistics_isk=logistics_isk,
            net_isk=net_isk,
        ))

    leftover_net_isk = max(Decimal("0"), leftover_total_isk - leftover_logistics_isk)
    return leftover_items, leftover_total_isk, leftover_logistics_isk, leftover_net_isk


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
        activity_me_bonus=req.activity_me_bonus,
    )
    bom_result = build_cost(bom_req)

    station_id, is_buy = _STATION_MAP[req.material_source]
    region_id = esi.STATION_REGION[station_id]

    flat_bom, _ = _compute_flat_bom(bom_req.type_id, bom_req.runs, bom_req)
    leaves: dict[int, tuple[str, int]] = {}
    for tid, qty in flat_bom.items():
        row = sde.get_type(tid)
        leaves[tid] = (row["typeName"] if row else f"Unknown ({tid})", qty)

    _prefetch_ore_orders(set(leaves.keys()), region_id)

    direct_items: list[DirectBuyItem] = []
    non_mineral_direct: list[DirectBuyItem] = []
    all_candidates: dict[int, list[CompressedOreItem]] = {}  # mineral_type_id → sorted candidates

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

        candidates = _ore_candidates_for_mineral(type_id, name, qty, station_id, region_id, req)
        if candidates:
            all_candidates[type_id] = candidates
        else:
            non_mineral_direct.append(DirectBuyItem(
                type_id=type_id,
                name=name,
                quantity=qty,
                unit_price=unit_price,
                total_isk=unit_price * qty,
                volume_m3=vol * qty,
            ))

    # Pre-compute SDE data needed by _compute_global_leftover once, so the swap loop
    # does zero SQLite I/O across potentially thousands of calls.
    all_ore_ids: set[int] = {
        c.ore_type_id for cands in all_candidates.values() for c in cands
    }
    ore_portion: dict[int, int] = {}
    ore_outputs: dict[int, list] = {}
    for oid in all_ore_ids:
        row = sde.get_type(oid)
        ore_portion[oid] = int(row["portionSize"]) if row and row["portionSize"] else 100
        ore_outputs[oid] = list(sde.get_refining_outputs(oid))

    item_meta: dict[int, tuple[str, Decimal]] = {}
    for outputs in ore_outputs.values():
        for out in outputs:
            tid = out["materialTypeID"]
            if tid not in item_meta:
                row = sde.get_type(tid)
                name = row["typeName"] if row else f"Unknown ({tid})"
                vol = Decimal(str(row["volume"])) if row and row["volume"] else Decimal("0")
                item_meta[tid] = (name, vol)

    # Pass 1: cheapest ore per mineral (logistics-adjusted effective_isk)
    ore_by_mineral: dict[int, CompressedOreItem] = {
        tid: cands[0] for tid, cands in all_candidates.items()
    }
    ore_items: list[CompressedOreItem] = list(ore_by_mineral.values())

    leftover_items, leftover_total_isk, leftover_logistics_isk, leftover_net_isk = (
        _compute_global_leftover(
            ore_items, leaves, req.leftover_logistics_isk_per_m3, req.reprocessing_yield,
            ore_portion, ore_outputs, item_meta,
        )
    )

    # Pass 2: if max_leftover_isk is set and exceeded, greedily swap ore choices to reduce
    # leftover until the constraint is satisfied or no further improvement is possible.
    constraint_met = True
    if req.max_leftover_isk is not None and leftover_net_isk > req.max_leftover_isk:
        constraint_met = False
        for _ in range(30):
            if leftover_net_isk <= req.max_leftover_isk:
                constraint_met = True
                break

            best_mineral_tid: int | None = None
            best_new_ore: CompressedOreItem | None = None
            best_reduction = Decimal("0")

            for tid, candidates in all_candidates.items():
                current_ore_id = ore_by_mineral[tid].ore_type_id

                for candidate in candidates:
                    if candidate.ore_type_id == current_ore_id:
                        continue

                    test_items = [
                        candidate if item.for_mineral_type_id == tid else item
                        for item in ore_items
                    ]
                    _, _, _, test_net = _compute_global_leftover(
                        test_items, leaves, req.leftover_logistics_isk_per_m3, req.reprocessing_yield,
                        ore_portion, ore_outputs, item_meta,
                    )
                    reduction = leftover_net_isk - test_net

                    if reduction > best_reduction:
                        best_reduction = reduction
                        best_mineral_tid = tid
                        best_new_ore = candidate

            if best_mineral_tid is None:
                break  # no swap reduces leftover — give up

            ore_by_mineral[best_mineral_tid] = best_new_ore
            ore_items = [ore_by_mineral[tid] for tid in all_candidates]
            leftover_items, leftover_total_isk, leftover_logistics_isk, leftover_net_isk = (
                _compute_global_leftover(
                    ore_items, leaves, req.leftover_logistics_isk_per_m3, req.reprocessing_yield,
                    ore_portion, ore_outputs, item_meta,
                )
            )

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
            leftover_logistics_isk=leftover_logistics_isk,
            leftover_net_isk=leftover_net_isk,
            leftover_constraint_met=constraint_met,
        ),
    )
