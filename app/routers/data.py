from fastapi import APIRouter, Query
from app.mocks.data_mock import MOCK_DATA
from typing import Optional, List

router = APIRouter()

@router.get("/", summary="Listar dados meteorológicos mockados")
def list_data(
    geometry_type: Optional[str] = Query(None, description="Filtrar por tipo de geometria: Point ou Polygon"),
    data_type: Optional[str] = Query(None, description="Filtrar por tipo de dado: temperature, humidity, wind_speed, etc.")
):
    """Lista dados meteorológicos com geometrias associadas.
    
    Os dados incluem localização espacial em formato GeoJSON.
    """
    filtered_data = MOCK_DATA
    
    if geometry_type:
        filtered_data = [item for item in filtered_data if item["geometry"]["type"].lower() == geometry_type.lower()]
    
    if data_type:
        filtered_data = [item for item in filtered_data if item["type"].lower() == data_type.lower()]
    
    return filtered_data

@router.get("/points", summary="Listar apenas dados pontuais")
def list_point_data():
    """Lista apenas dados meteorológicos com geometria de ponto."""
    return [item for item in MOCK_DATA if item["geometry"]["type"] == "Point"]

@router.get("/polygons", summary="Listar apenas dados de área")
def list_polygon_data():
    """Lista apenas dados meteorológicos com geometria de polígono."""
    return [item for item in MOCK_DATA if item["geometry"]["type"] == "Polygon"]

@router.get("/locations", summary="Listar localizações disponíveis")
def list_locations():
    """Lista todas as localizações com seus nomes e tipos de geometria."""
    locations = []
    for item in MOCK_DATA:
        locations.append({
            "id": item["id"],
            "location_name": item["location_name"],
            "geometry_type": item["geometry"]["type"],
            "data_types": [item["type"]]
        })
    return locations
