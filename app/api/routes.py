from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    BuildCostRequest, BuildCostResponse,
    RefineCostRequest, RefineCostResponse,
    CompareMaterialSourceRequest, CompareMaterialSourceResponse,
)
from app.data import sde
from app.data.rigs import get_standup_me_rigs
from app.engine.bom import build_cost as _build_cost
from app.engine.refining import refine_cost as _refine_cost
from app.engine.compare import compare_material_source as _compare

router = APIRouter()


@router.post("/build-cost", response_model=BuildCostResponse)
async def build_cost(req: BuildCostRequest):
    try:
        return _build_cost(req)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/refine-cost", response_model=RefineCostResponse)
async def refine_cost(req: RefineCostRequest):
    try:
        return _refine_cost(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/compare-material-source", response_model=CompareMaterialSourceResponse)
async def compare_material_source(req: CompareMaterialSourceRequest):
    try:
        return _compare(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/config/rigs")
def config_rigs():
    return get_standup_me_rigs()


@router.get("/config/structures")
def config_structures():
    return [
        {"id": "npc",     "name": "NPC Station",  "rig_size": 0, "base_me_mfg": 0.0, "base_me_react": 0.0},
        {"id": "raitaru", "name": "Raitaru",       "rig_size": 2, "base_me_mfg": 1.0, "base_me_react": 0.0},
        {"id": "azbel",   "name": "Azbel",         "rig_size": 3, "base_me_mfg": 1.0, "base_me_react": 0.0},
        {"id": "sotiyo",  "name": "Sotiyo",        "rig_size": 4, "base_me_mfg": 1.0, "base_me_react": 0.0},
        {"id": "athanor", "name": "Athanor",       "rig_size": 2, "base_me_mfg": 0.0, "base_me_react": 0.0},
        {"id": "tatara",  "name": "Tatara",        "rig_size": 3, "base_me_mfg": 0.0, "base_me_react": 0.0},
    ]


@router.get("/search")
async def search_types(q: str, limit: int = 20):
    results = sde.search_types(q, limit)
    return [{"type_id": r["typeID"], "name": r["typeName"]} for r in results]


@router.get("/search-systems")
async def search_systems(q: str, limit: int = 20):
    results = sde.search_solar_systems(q, limit)
    return [
        {"system_id": r["solarSystemID"], "name": r["solarSystemName"], "security": round(r["security"], 1)}
        for r in results
    ]
