from fastapi import APIRouter, HTTPException
from app.models.schemas import BuildCostRequest, BuildCostResponse
from app.data import sde

router = APIRouter()


@router.post("/build-cost", response_model=BuildCostResponse)
async def build_cost(req: BuildCostRequest):
    # TODO: implement BOM engine
    raise HTTPException(status_code=501, detail="BOM engine not yet implemented")


@router.get("/search")
async def search_types(q: str, limit: int = 20):
    """Search item types by name."""
    results = sde.search_types(q, limit)
    return [{"type_id": r["typeID"], "name": r["typeName"]} for r in results]
