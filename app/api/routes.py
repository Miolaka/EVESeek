from fastapi import APIRouter, HTTPException
from app.models.schemas import BuildCostRequest, BuildCostResponse
from app.data import sde
from app.engine.bom import build_cost as _build_cost

router = APIRouter()


@router.post("/build-cost", response_model=BuildCostResponse)
async def build_cost(req: BuildCostRequest):
    try:
        return _build_cost(req)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/search")
async def search_types(q: str, limit: int = 20):
    """Search item types by name."""
    results = sde.search_types(q, limit)
    return [{"type_id": r["typeID"], "name": r["typeName"]} for r in results]
