from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.routes import router
from app.esi import disk_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    disk_cache.evict_expired()
    yield


app = FastAPI(title="EVESeek", version="0.1.0", lifespan=lifespan)
app.include_router(router, prefix="/api/v1")
app.mount("/", StaticFiles(directory="static", html=True), name="static")
