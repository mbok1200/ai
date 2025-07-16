import asyncio
import logging
from fastapi import FastAPI
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Minimal Test API")

@app.get("/")
async def root():
    return {"message": "Minimal API working", "status": "ok"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    logger.info("Starting minimal FastAPI server...")
    uvicorn.run("app_minimal:app", host="0.0.0.0", port=8000, reload=True)