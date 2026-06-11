from decimal import Decimal
from typing import Annotated, Literal
from pydantic import BaseModel, Field, PlainSerializer

# Decimal fields in responses serialize as float for JSON compatibility
ISK = Annotated[Decimal, PlainSerializer(lambda v: float(v), return_type=float)]


class BuildCostRequest(BaseModel):
    type_id: int
    system_id: int
    me_level: int = Field(10, ge=0, le=10)
    me_overrides: dict[int, int] = Field(default_factory=dict)
    structure_bonus: float = Field(0.01, ge=0.0, le=0.10)
    fw_level: int = Field(0, ge=0, le=5)
    runs: int = Field(1, ge=1)
    material_source: Literal["jita_sell", "jita_buy", "amarr_sell", "amarr_buy"] = "jita_sell"
    facility_tax: Decimal = Field(Decimal("0.0025"), ge=0)
    logistics_cost_isk_per_m3: Decimal = Field(Decimal("0"), ge=0)
    broker_relations_level: int = Field(0, ge=0, le=5)
    faction_standing: float = Field(0.0, ge=-10.0, le=10.0)
    corp_standing: float = Field(0.0, ge=-10.0, le=10.0)
    build_t1_hull: bool = True


class CostBreakdown(BaseModel):
    material_costs: ISK
    manufacturing_fees: ISK
    reaction_fees: ISK
    refining_fees: ISK
    logistics_costs: ISK


class BOMNode(BaseModel):
    type_id: int
    name: str
    quantity: int
    cost_per_unit: ISK
    total_cost: ISK
    cost_breakdown: CostBreakdown
    children: list["BOMNode"] = []
    bpc_copies_needed: int = 0
    max_runs_per_bpc: int = 0


class BuildCostResponse(BaseModel):
    type_id: int
    item_name: str
    total_cost: ISK
    cost_breakdown: CostBreakdown
    bom_tree: list[BOMNode]


BOMNode.model_rebuild()


class CompareMaterialSourceRequest(BaseModel):
    type_id: int
    system_id: int
    me_level: int = Field(10, ge=0, le=10)
    me_overrides: dict[int, int] = Field(default_factory=dict)
    structure_bonus: float = Field(0.01, ge=0.0, le=0.10)
    fw_level: int = Field(0, ge=0, le=5)
    runs: int = Field(1, ge=1)
    material_source: Literal["jita_sell", "jita_buy", "amarr_sell", "amarr_buy"] = "jita_sell"
    facility_tax: Decimal = Field(Decimal("0.0025"), ge=0)
    broker_relations_level: int = Field(0, ge=0, le=5)
    faction_standing: float = Field(0.0, ge=-10.0, le=10.0)
    corp_standing: float = Field(0.0, ge=-10.0, le=10.0)
    reprocessing_yield: float = Field(0.876, ge=0.0, le=1.0)
    reprocessing_rate: Decimal = Field(Decimal("0.02"), ge=0)
    refinery_bonus: float = Field(0.0, ge=0.0, le=0.10)
    leftover_logistics_isk_per_m3: Decimal = Field(Decimal("0"), ge=0)
    max_leftover_isk: Decimal | None = None
    build_t1_hull: bool = True


class DirectBuyItem(BaseModel):
    type_id: int
    name: str
    quantity: int
    unit_price: ISK
    total_isk: ISK
    volume_m3: ISK


class DirectBuyPath(BaseModel):
    total_isk: ISK
    total_m3: ISK
    items: list[DirectBuyItem]


class CompressedOreItem(BaseModel):
    ore_type_id: int
    ore_name: str
    for_mineral_type_id: int
    for_mineral_name: str
    quantity: int
    unit_price: ISK
    total_isk: ISK
    refining_fee: ISK
    byproduct_credit: ISK
    effective_isk: ISK
    volume_m3: ISK        # compressed ore volume (haul to refinery)
    refined_m3: ISK       # total mineral volume produced after refining


class LeftoverItem(BaseModel):
    type_id: int
    name: str
    quantity: int
    buy_price: ISK
    total_isk: ISK
    volume_m3: ISK = Decimal("0")    # total m³ of this surplus item
    logistics_isk: ISK = Decimal("0") # haul cost to Jita = volume_m3 × rate
    net_isk: ISK = Decimal("0")       # total_isk − logistics_isk (≥ 0)


class CompressedOrePath(BaseModel):
    total_isk: ISK
    effective_isk: ISK
    total_m3: ISK         # compressed ore volume + non-mineral direct (haul to refinery)
    refined_total_m3: ISK # total mineral volume after refining + non-mineral direct
    refining_fee: ISK
    ore_items: list[CompressedOreItem]
    direct_items: list[DirectBuyItem]
    leftover_items: list[LeftoverItem] = []
    leftover_total_isk: ISK = Decimal("0")
    leftover_logistics_isk: ISK = Decimal("0")  # total haul cost for all leftovers
    leftover_net_isk: ISK = Decimal("0")         # leftover_total_isk − leftover_logistics_isk
    leftover_constraint_met: bool = True          # False if max_leftover_isk could not be satisfied


class CompareMaterialSourceResponse(BaseModel):
    direct_buy: DirectBuyPath
    compressed_ore: CompressedOrePath


class RefineCostRequest(BaseModel):
    type_id: int
    quantity: int = Field(ge=1)
    reprocessing_yield: float = Field(0.876, ge=0.0, le=1.0)
    reprocessing_rate: Decimal = Field(Decimal("0.02"), ge=0)
    structure_bonus: float = Field(0.0, ge=0.0, le=0.10)
    fw_level: int = Field(0, ge=0, le=5)


class RefineOutputItem(BaseModel):
    type_id: int
    name: str
    quantity: int


class RefineCostResponse(BaseModel):
    type_id: int
    ore_name: str
    quantity_refined: int
    refining_fee: ISK
    outputs: list[RefineOutputItem]
