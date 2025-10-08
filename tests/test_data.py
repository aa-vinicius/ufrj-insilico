from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_list_data():
    response = client.get("/api/data/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    
    # Verificar se os dados contêm geometrias
    for item in data:
        assert "geometry" in item
        assert "type" in item["geometry"]
        assert "coordinates" in item["geometry"]
        assert item["geometry"]["type"] in ["Point", "Polygon"]
        assert "location_name" in item

def test_data_point_geometry():
    response = client.get("/api/data/")
    data = response.json()
    
    # Encontrar um item com geometria de ponto
    point_items = [item for item in data if item["geometry"]["type"] == "Point"]
    assert len(point_items) > 0
    
    point_item = point_items[0]
    coordinates = point_item["geometry"]["coordinates"]
    assert len(coordinates) == 2  # [longitude, latitude]
    assert isinstance(coordinates[0], float)  # longitude
    assert isinstance(coordinates[1], float)  # latitude

def test_data_polygon_geometry():
    response = client.get("/api/data/")
    data = response.json()
    
    # Encontrar um item com geometria de polígono
    polygon_items = [item for item in data if item["geometry"]["type"] == "Polygon"]
    assert len(polygon_items) > 0
    
    polygon_item = polygon_items[0]
    coordinates = polygon_item["geometry"]["coordinates"]
    assert len(coordinates) >= 1  # Array de anéis
    assert len(coordinates[0]) >= 4  # Mínimo 4 pontos para fechar o polígono

def test_filter_data_by_geometry_type():
    # Testar filtro por tipo de geometria Point
    response = client.get("/api/data/?geometry_type=Point")
    assert response.status_code == 200
    data = response.json()
    for item in data:
        assert item["geometry"]["type"] == "Point"
    
    # Testar filtro por tipo de geometria Polygon
    response = client.get("/api/data/?geometry_type=Polygon")
    assert response.status_code == 200
    data = response.json()
    for item in data:
        assert item["geometry"]["type"] == "Polygon"

def test_filter_data_by_type():
    response = client.get("/api/data/?data_type=temperature")
    assert response.status_code == 200
    data = response.json()
    for item in data:
        assert item["type"] == "temperature"

def test_data_points_endpoint():
    response = client.get("/api/data/points")
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    for item in data:
        assert item["geometry"]["type"] == "Point"

def test_data_polygons_endpoint():
    response = client.get("/api/data/polygons")
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    for item in data:
        assert item["geometry"]["type"] == "Polygon"

def test_data_locations_endpoint():
    response = client.get("/api/data/locations")
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    for item in data:
        assert "location_name" in item
        assert "geometry_type" in item
        assert "data_types" in item
