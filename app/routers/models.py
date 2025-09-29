from fastapi import APIRouter, Query
from app.mocks.models_mock import MOCK_MODEL_RESULTS
from typing import Optional

router = APIRouter()

@router.get("/", summary="Listar resultados de modelos de ML mockados")
def list_models(
    geometry_type: Optional[str] = Query(None, description="Filtrar por tipo de geometria: Point ou Polygon"),
    model_name: Optional[str] = Query(None, description="Filtrar por nome do modelo")
):
    """Lista resultados de modelos de ML com geometrias associadas.
    
    Os resultados incluem previsões com localização espacial em formato GeoJSON.
    """
    filtered_data = MOCK_MODEL_RESULTS
    
    if geometry_type:
        filtered_data = [item for item in filtered_data if item["geometry"]["type"].lower() == geometry_type.lower()]
    
    if model_name:
        filtered_data = [item for item in filtered_data if model_name.lower() in item["model"].lower()]
    
    return filtered_data

@router.get("/points", summary="Listar previsões pontuais")
def list_point_models():
    """Lista apenas resultados de modelos com geometria de ponto."""
    return [item for item in MOCK_MODEL_RESULTS if item["geometry"]["type"] == "Point"]

@router.get("/polygons", summary="Listar previsões de área")
def list_polygon_models():
    """Lista apenas resultados de modelos com geometria de polígono."""
    return [item for item in MOCK_MODEL_RESULTS if item["geometry"]["type"] == "Polygon"]

@router.get("/forecast-areas", summary="Listar áreas de previsão")
def list_forecast_areas():
    """Lista todas as áreas de previsão com seus nomes e modelos associados."""
    areas = []
    for item in MOCK_MODEL_RESULTS:
        areas.append({
            "id": item["id"],
            "location_name": item["location_name"],
            "geometry_type": item["geometry"]["type"],
            "model": item["model"],
            "forecast_time": item["forecast_time"]
        })
    return areas
