from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_list_models():
    response = client.get("/models/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    
    # Verificar se os resultados contêm geometrias
    for item in data:
        assert "geometry" in item
        assert "type" in item["geometry"]
        assert "coordinates" in item["geometry"]
        assert item["geometry"]["type"] in ["Point", "Polygon"]
        assert "location_name" in item
        assert "forecast_time" in item

def test_models_point_geometry():
    response = client.get("/models/")
    data = response.json()
    
    # Encontrar um resultado com geometria de ponto
    point_items = [item for item in data if item["geometry"]["type"] == "Point"]
    assert len(point_items) > 0
    
    point_item = point_items[0]
    coordinates = point_item["geometry"]["coordinates"]
    assert len(coordinates) == 2  # [longitude, latitude]
    assert isinstance(coordinates[0], float)  # longitude
    assert isinstance(coordinates[1], float)  # latitude

def test_models_polygon_geometry():
    response = client.get("/models/")
    data = response.json()
    
    # Encontrar um resultado com geometria de polígono
    polygon_items = [item for item in data if item["geometry"]["type"] == "Polygon"]
    assert len(polygon_items) > 0
    
    polygon_item = polygon_items[0]
    coordinates = polygon_item["geometry"]["coordinates"]
    assert len(coordinates) >= 1  # Array de anéis
    assert len(coordinates[0]) >= 4  # Mínimo 4 pontos para fechar o polígono

def test_model_forecast_structure():
    response = client.get("/models/")
    data = response.json()
    
    # Verificar estrutura dos dados de previsão
    for item in data:
        assert "model" in item
        assert "result" in item
        assert "confidence" in item
        assert 0 <= item["confidence"] <= 1  # Confiança entre 0 e 1

def test_filter_models_by_geometry_type():
    # Testar filtro por tipo de geometria Point
    response = client.get("/models/?geometry_type=Point")
    assert response.status_code == 200
    data = response.json()
    for item in data:
        assert item["geometry"]["type"] == "Point"
    
    # Testar filtro por tipo de geometria Polygon
    response = client.get("/models/?geometry_type=Polygon")
    assert response.status_code == 200
    data = response.json()
    for item in data:
        assert item["geometry"]["type"] == "Polygon"

def test_filter_models_by_name():
    response = client.get("/models/?model_name=rain")
    assert response.status_code == 200
    data = response.json()
    for item in data:
        assert "rain" in item["model"].lower()

def test_models_points_endpoint():
    response = client.get("/models/points")
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    for item in data:
        assert item["geometry"]["type"] == "Point"

def test_models_polygons_endpoint():
    response = client.get("/models/polygons")
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    for item in data:
        assert item["geometry"]["type"] == "Polygon"

def test_forecast_areas_endpoint():
    response = client.get("/models/forecast-areas")
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    for item in data:
        assert "location_name" in item
        assert "geometry_type" in item
        assert "model" in item
        assert "forecast_time" in item
