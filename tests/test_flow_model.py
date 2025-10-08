"""
Testes específicos para os endpoints do modelo de previsão de vazão
"""

from fastapi.testclient import TestClient
from app.main import app
import json

client = TestClient(app)

def test_flow_model_status():
    """Testa endpoint de status do modelo de vazão"""
    response = client.get("/api/models/flow/status")
    assert response.status_code == 200
    
    data = response.json()
    assert "model_name" in data
    assert "model_type" in data
    assert "version" in data
    assert "is_optimized" in data
    assert "uses_pretrained_models" in data
    assert data["model_name"] == "Flow Prediction Ensemble"
    assert data["is_optimized"] == True

def test_flow_prediction_simple():
    """Testa previsão simples (apenas ano e mês)"""
    request_data = {
        "year": 2024,
        "month": 6
    }
    
    response = client.post("/api/models/flow/predict", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "predicted_flow" in data
    assert "lower_bound" in data
    assert "upper_bound" in data
    assert "uncertainty" in data
    assert "confidence_level" in data
    assert "model_components" in data
    
    assert data["year"] == 2024
    assert data["month"] == 6
    assert isinstance(data["predicted_flow"], (int, float))
    assert data["predicted_flow"] > 0
    assert data["lower_bound"] < data["predicted_flow"]
    assert data["upper_bound"] > data["predicted_flow"]
    assert data["confidence_level"] == 95.0

def test_flow_prediction_with_features():
    """Testa previsão com features meteorológicas"""
    request_data = {
        "year": 2024,
        "month": 3,
        "u2_min": 2.5,
        "u2_max": 4.2,
        "tmin_min": 18.5,
        "tmin_max": 22.1,
        "tmax_min": 28.3,
        "tmax_max": 32.7,
        "rs_min": 16.2,
        "rs_max": 24.8,
        "rh_min": 65.4,
        "rh_max": 88.2,
        "eto_min": 3.2,
        "eto_max": 5.8,
        "pr_min": 1.5,
        "pr_max": 8.7,
        "y_lag1": 95.2,
        "y_lag2": 102.5,
        "y_lag3": 88.9
    }
    
    response = client.post("/api/models/flow/predict", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["year"] == 2024
    assert data["month"] == 3
    assert isinstance(data["predicted_flow"], (int, float))
    assert data["predicted_flow"] > 0

def test_flow_test_data():
    """Testa endpoint de dados de teste"""
    response = client.get("/api/models/flow/test-data")
    assert response.status_code == 200
    
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    
    # Verificar estrutura do primeiro item
    first_item = data[0]
    assert "year" in first_item
    assert "month" in first_item
    assert "observed" in first_item
    assert "predicted" in first_item
    assert "lower_bound" in first_item
    assert "upper_bound" in first_item
    
    assert isinstance(first_item["year"], int)
    assert 1 <= first_item["month"] <= 12
    assert first_item["observed"] > 0
    assert first_item["predicted"] > 0

def test_flow_prediction_edge_cases():
    """Testa casos extremos de previsão"""
    # Teste com ano limite inferior
    response = client.post("/api/models/flow/predict", json={"year": 1998, "month": 1})
    assert response.status_code == 200
    
    # Teste com ano limite superior
    response = client.post("/api/models/flow/predict", json={"year": 2030, "month": 12})
    assert response.status_code == 200
    
    # Teste com mês inválido (deve falhar)
    response = client.post("/api/models/flow/predict", json={"year": 2024, "month": 13})
    assert response.status_code == 422  # Validation error
    
    # Teste com ano inválido (deve falhar)
    response = client.post("/api/models/flow/predict", json={"year": 1990, "month": 6})
    assert response.status_code == 422  # Validation error

def test_flow_seasonal_pattern():
    """Testa se o modelo respeita padrões sazonais"""
    # Previsões para diferentes meses
    winter_response = client.post("/api/models/flow/predict", json={"year": 2024, "month": 6})  # Inverno
    summer_response = client.post("/api/models/flow/predict", json={"year": 2024, "month": 1})  # Verão
    
    assert winter_response.status_code == 200
    assert summer_response.status_code == 200
    
    winter_flow = winter_response.json()["predicted_flow"]
    summer_flow = summer_response.json()["predicted_flow"]
    
    # Verão deveria ter vazão maior que inverno (padrão brasileiro)
    assert summer_flow > winter_flow

def test_flow_model_components():
    """Testa informações dos componentes do modelo ensemble"""
    response = client.post("/api/models/flow/predict", json={"year": 2024, "month": 6})
    assert response.status_code == 200
    
    data = response.json()
    components = data["model_components"]
    
    assert "mlp_weight" in components
    assert "xgboost_weight" in components
    assert "ensemble_method" in components
    
    assert components["mlp_weight"] == 0.6
    assert components["xgboost_weight"] == 0.4
    assert components["ensemble_method"] == "weighted_average"