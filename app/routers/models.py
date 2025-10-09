from fastapi import APIRouter, Query, HTTPException
from app.mocks.models_mock import MOCK_MODEL_RESULTS
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
from datetime import datetime
import pandas as pd
import numpy as np
from app.models.optimized_flow_model import flow_model, optimized_flow_model
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Modelos Pydantic para o modelo de vazão
class FlowPredictionRequest(BaseModel):
    """Modelo para requisição de previsão de vazão - TODAS as variáveis são obrigatórias"""
    year: int = Field(..., ge=1998, le=2030, description="Ano da previsão")
    month: int = Field(..., ge=1, le=12, description="Mês da previsão (1-12)")
    
    # Features meteorológicas (OBRIGATÓRIAS)
    u2_min: float = Field(..., description="Velocidade do vento mínima (m/s)")
    u2_max: float = Field(..., description="Velocidade do vento máxima (m/s)")
    tmin_min: float = Field(..., description="Temperatura mínima (°C)")
    tmin_max: float = Field(..., description="Temperatura máxima mínima (°C)")
    tmax_min: float = Field(..., description="Temperatura máxima mínima (°C)")
    tmax_max: float = Field(..., description="Temperatura máxima (°C)")
    rs_min: float = Field(..., description="Radiação solar mínima (MJ/m²/dia)")
    rs_max: float = Field(..., description="Radiação solar máxima (MJ/m²/dia)")
    rh_min: float = Field(..., description="Umidade relativa mínima (%)")
    rh_max: float = Field(..., description="Umidade relativa máxima (%)")
    eto_min: float = Field(..., description="Evapotranspiração mínima (mm/dia)")
    eto_max: float = Field(..., description="Evapotranspiração máxima (mm/dia)")
    pr_min: float = Field(..., description="Precipitação mínima (mm)")
    pr_max: float = Field(..., description="Precipitação máxima (mm)")
    
    # Features de lag de vazão (OBRIGATÓRIAS)
    y_lag1: float = Field(..., description="Vazão do mês anterior (m³/s)")
    y_lag2: float = Field(..., description="Vazão de 2 meses atrás (m³/s)")
    y_lag3: float = Field(..., description="Vazão de 3 meses atrás (m³/s)")
    y_rm3: float = Field(..., description="Média móvel de 3 meses da vazão (m³/s)")
    
    # Features de lag de precipitação (OBRIGATÓRIAS)
    pr_lag1: float = Field(..., description="Precipitação do mês anterior (mm)")
    pr_lag2: float = Field(..., description="Precipitação de 2 meses atrás (mm)")
    pr_lag3: float = Field(..., description="Precipitação de 3 meses atrás (mm)")
    pr_sum3: float = Field(..., description="Soma de precipitação dos últimos 3 meses (mm)")
    pr_api3: float = Field(..., description="Índice antecedente de precipitação (mm)")

class FlowPredictionResponse(BaseModel):
    """Modelo para resposta de previsão de vazão"""
    year: int
    month: int
    predicted_flow: float = Field(..., description="Vazão prevista (m³/s)")
    lower_bound: float = Field(..., description="Limite inferior do intervalo de confiança (m³/s)")
    upper_bound: float = Field(..., description="Limite superior do intervalo de confiança (m³/s)")
    uncertainty: float = Field(..., description="Incerteza (±σ)")
    confidence_level: float = Field(..., description="Nível de confiança do intervalo")
    model_components: Dict[str, Any] = Field(..., description="Componentes do modelo ensemble")

class TestDataResponse(BaseModel):
    """Modelo para dados de teste"""
    year: int
    month: int
    observed: float
    predicted: float
    lower_bound: float
    upper_bound: float
    obs_min: float
    obs_max: float

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


# === ENDPOINTS DO MODELO DE VAZÃO ===

@router.post("/flow/predict", response_model=FlowPredictionResponse)
async def predict_flow(request: FlowPredictionRequest):
    """
    Fazer previsão de vazão para um mês específico
    Todas as 23 variáveis de entrada são obrigatórias.
    """
    try:
        # Extrair todas as features do request (todas são obrigatórias agora)
        features = {
            'u2_min': request.u2_min,
            'u2_max': request.u2_max,
            'tmin_min': request.tmin_min,
            'tmin_max': request.tmin_max,
            'tmax_min': request.tmax_min,
            'tmax_max': request.tmax_max,
            'rs_min': request.rs_min,
            'rs_max': request.rs_max,
            'rh_min': request.rh_min,
            'rh_max': request.rh_max,
            'eto_min': request.eto_min,
            'eto_max': request.eto_max,
            'pr_min': request.pr_min,
            'pr_max': request.pr_max,
            'y_lag1': request.y_lag1,
            'y_lag2': request.y_lag2,
            'y_lag3': request.y_lag3,
            'y_rm3': request.y_rm3,
            'pr_lag1': request.pr_lag1,
            'pr_lag2': request.pr_lag2,
            'pr_lag3': request.pr_lag3,
            'pr_sum3': request.pr_sum3,
            'pr_api3': request.pr_api3
        }
        
        # Usar o modelo otimizado que já valida todas as variáveis
        result = optimized_flow_model.predict_month(request.year, request.month, features)
        
        return FlowPredictionResponse(
            year=result["year"],
            month=result["month"],
            predicted_flow=result["predicted_flow"],
            lower_bound=result["lower_bound"],
            upper_bound=result["upper_bound"],
            uncertainty=result["uncertainty"],
            confidence_level=result["confidence_level"],
            model_components=result.get("model_components", {})
        )
        
    except Exception as e:
        logger.error(f"Erro na previsão: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro na previsão: {str(e)}")


@router.get("/flow/test-data", response_model=List[TestDataResponse])
async def get_flow_test_data():
    """
    Obter dados de teste do modelo de vazão (comparação observado vs predito)
    """
    try:
        test_data = optimized_flow_model.get_test_data()
        
        return [
            TestDataResponse(
                year=int(row['year']),
                month=int(row['month']),
                observed=round(row['observed'], 2),
                predicted=round(row['predicted'], 2),
                lower_bound=round(row['lower_bound'], 2),
                upper_bound=round(row['upper_bound'], 2),
                obs_min=round(row['obs_min'], 2),  
                obs_max=round(row['obs_max'], 2)
            )
            for _, row in test_data.iterrows()
        ]
    
    except Exception as e:
        logger.error(f"Erro ao obter dados de teste: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter dados de teste: {str(e)}")


@router.get("/flow/status")
async def get_flow_model_status():
    """
    Obter status e informações do modelo de vazão
    """
    model_info = optimized_flow_model.get_model_info()
    
    base_info = {
        "model_name": "Flow Prediction Ensemble",
        "model_type": "MLP + XGBoost",
        "version": "2.0.0",
        "description": "Modelo ensemble otimizado para previsão de vazão com modelos pré-treinados",
        "features": {
            "meteorological": [
                "u2_min", "u2_max", "tmin_min", "tmin_max", "tmax_min", "tmax_max",
                "rs_min", "rs_max", "rh_min", "rh_max", "eto_min", "eto_max", "pr_min", "pr_max"
            ],
            "temporal_lags": ["y_lag1", "y_lag2", "y_lag3"],
            "temporal": ["year", "month"]
        },
        "is_optimized": True,
        "uses_pretrained_models": True
    }
    
    if model_info['status'] == 'loaded':
        training_info = model_info['training_info']
        base_info.update({
            "is_trained": True,
            "ready_for_prediction": True,
            "training_date": training_info.get('training_date'),
            "training_samples": training_info.get('train_samples'),
            "test_samples": training_info.get('test_samples'),
            "performance": {
                "mlp_rmse": training_info.get('mlp_rmse'),
                "mlp_r2": training_info.get('mlp_r2'),
                "xgb_rmse": training_info.get('xgb_rmse'),
                "xgb_r2": training_info.get('xgb_r2'),
                "ensemble_rmse": training_info.get('ensemble_rmse'),
                "ensemble_r2": training_info.get('ensemble_r2')
            },
            "model_files": model_info['model_files']
        })
    else:
        base_info.update({
            "is_trained": False,
            "ready_for_prediction": True,  # Fallback funciona
            "warning": "Modelos pré-treinados não carregados, usando previsão baseada em sazonalidade",
            "recommendation": "Execute scripts/train_flow_model.py para treinar os modelos"
        })
    
    base_info["last_updated"] = datetime.now().isoformat()
    return base_info
