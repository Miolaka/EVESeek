import math
from decimal import Decimal
from app.data import sde
from app.esi import client as esi
from app.core.utils import job_cost_refining
from app.models.schemas import RefineCostRequest, RefineCostResponse, RefineOutputItem


def refine_cost(req: RefineCostRequest) -> RefineCostResponse:
    type_row = sde.get_type(req.type_id)
    if not type_row:
        raise ValueError(f"Unknown type_id: {req.type_id}")

    portion_size = type_row["portionSize"] or 1
    outputs_raw = sde.get_refining_outputs(req.type_id)
    if not outputs_raw:
        raise ValueError(f"No refining data for '{type_row['typeName']}' — not an ore or gas")

    batches = req.quantity // portion_size
    if batches == 0:
        raise ValueError(
            f"Quantity {req.quantity} is below the minimum batch size of {portion_size}"
        )

    outputs = []
    for row in outputs_raw:
        qty = math.floor(row["quantity"] * req.reprocessing_yield) * batches
        if qty <= 0:
            continue
        mat_row = sde.get_type(row["materialTypeID"])
        outputs.append(RefineOutputItem(
            type_id=row["materialTypeID"],
            name=mat_row["typeName"] if mat_row else f"Unknown ({row['materialTypeID']})",
            quantity=qty,
        ))

    # Fee is based on adjusted price of the input ore × units processed
    ore_adj = esi.get_adjusted_price(req.type_id) or Decimal("0")
    units_processed = batches * portion_size
    fee_per_unit = job_cost_refining(
        ore_adj,
        req.reprocessing_rate,
        req.structure_bonus,
        req.fw_level,
    )
    refining_fee = fee_per_unit * units_processed

    return RefineCostResponse(
        type_id=req.type_id,
        ore_name=type_row["typeName"],
        quantity_refined=units_processed,
        refining_fee=refining_fee,
        outputs=outputs,
    )
