from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.routers import data, models

app = FastAPI(title="UFRJ InSilico API")

# Servir arquivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Rota para a página de documentação
@app.get("/", response_class=FileResponse)
async def read_docs():
    return "static/index.html"

app.include_router(data.router, prefix="/data", tags=["data"])
app.include_router(models.router, prefix="/models", tags=["models"])
