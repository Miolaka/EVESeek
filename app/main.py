from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.routes import router

app = FastAPI(title="EVESeek", version="0.1.0")
app.include_router(router, prefix="/api/v1")
app.mount("/", StaticFiles(directory="static", html=True), name="static")
