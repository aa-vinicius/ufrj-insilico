from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from app.routers import data, models

app = FastAPI(
    title="UFRJ InSilico Weather & ML API",
    description="API para dados meteorológicos e modelos de ML do laboratório InSilico da UFRJ",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Montar arquivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Incluir routers
app.include_router(data.router, prefix="/api/data", tags=["Data"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])

@app.get("/", response_class=HTMLResponse)
async def main():
    """Página principal da API com documentação interativa"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/flow", response_class=HTMLResponse)
async def flow_model_page():
    """Página do modelo de previsão de vazão"""
    with open("static/flow.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)
