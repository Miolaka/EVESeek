from decimal import Decimal
from typing import Literal
from pydantic import BaseModel, Field


class BuildCostRequest(BaseModel):
    type_id: int
    region_id: int
    system_id: int
    me_level: int = Field(0, ge=0, le=10)
    structure_bonus: float = Field(0.0, ge=0.0, le=0.10)
    fw_level: int = Field(0, ge=0, le=5)
    runs: int = Field(1, ge=1)
    material_source: Literal["jita_sell", "jita_buy"] = "jita_sell"
    logistics_cost_isk_per_m3: Decimal = Field(Decimal("0"), ge=0)
    broker_relations_level: int = Field(0, ge=0, le=5)
    faction_standing: float = Field(0.0, ge=-10.0, le=10.0)
    corp_standing: float = Field(0.0, ge=-10.0, le=10.0)


class CostBreakdown(BaseModel):
    material_costs: Decimal
    manufacturing_fees: Decimal
    reaction_fees: Decimal
    refining_fees: Decimal
    logistics_costs: Decimal


class BOMNode(BaseModel):
    type_id: int
    name: str
    quantity: int
    cost_per_unit: Decimal
    total_cost: Decimal
    cost_breakdown: CostBreakdown
    children: list["BOMNode"] = []


class BuildCostResponse(BaseModel):
    type_id: int
    item_name: str
    total_cost: Decimal
    cost_breakdown: CostBreakdown
    bom_tree: list[BOMNode]

    model_config = {"json_encoders": {Decimal: float}}


BOMNode.model_rebuild()
