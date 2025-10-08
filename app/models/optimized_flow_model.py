"""
Modelo de previsão de vazão mensal otimizado com modelos pré-treinados
Carrega modelos salvos ao invés de treinar a cada requisição
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from scipy.stats import pearsonr
import joblib
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class OptimizedFlowModel:
    """
    Modelo otimizado de previsão de vazão que usa modelos pré-treinados
    """
    
    def __init__(self):
        self.models_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
        self.mlp_model = None
        self.xgb_model = None
        self.scaler = None
        self.training_info = None
        self.is_loaded = False
        
        # Tentar carregar modelos na inicialização
        self.load_models()
    
    def load_models(self) -> bool:
        """Carrega os modelos salvos"""
        try:
            # Verificar se os arquivos existem
            required_files = ['mlp_model.joblib', 'xgb_model.joblib', 'scaler.joblib', 'training_info.joblib']
            for file in required_files:
                if not os.path.exists(os.path.join(self.models_dir, file)):
                    logger.warning(f"Arquivo não encontrado: {file}")
                    return False
            
            # Carregar modelos
            self.mlp_model = joblib.load(os.path.join(self.models_dir, 'mlp_model.joblib'))
            self.xgb_model = joblib.load(os.path.join(self.models_dir, 'xgb_model.joblib'))
            self.scaler = joblib.load(os.path.join(self.models_dir, 'scaler.joblib'))
            self.training_info = joblib.load(os.path.join(self.models_dir, 'training_info.joblib'))
            
            self.is_loaded = True
            logger.info("Modelos pré-treinados carregados com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelos: {str(e)}")
            self.is_loaded = False
            return False
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Faz previsão usando os modelos pré-treinados
        
        Args:
            features: DataFrame com features preparadas
            
        Returns:
            Array com previsões
        """
        if not self.is_loaded:
            raise RuntimeError("Modelos não carregados. Execute load_models() primeiro.")
        
        try:
            # Verificar se as features têm as colunas esperadas
            expected_cols = self.training_info['feature_columns']
            missing_cols = set(expected_cols) - set(features.columns)
            if missing_cols:
                logger.warning(f"Colunas faltando: {missing_cols}")
                
            # Reordenar colunas e preencher missing com 0
            for col in expected_cols:
                if col not in features.columns:
                    features[col] = 0
            features = features[expected_cols]
            
            # Normalizar features
            features_scaled = self.scaler.transform(features)
            
            # Fazer previsões
            mlp_pred = self.mlp_model.predict(features_scaled)
            xgb_pred = self.xgb_model.predict(features)
            
            # Ensemble (mesmo peso usado no treinamento)
            ensemble_pred = 0.6 * mlp_pred + 0.4 * xgb_pred
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Erro durante previsão: {str(e)}")
            raise
    
    def predict_month(self, year: int, month: int) -> float:
        """
        Previsão simples baseada apenas em ano e mês (sazonalidade)
        
        Args:
            year: Ano da previsão
            month: Mês da previsão (1-12)
            
        Returns:
            Previsão de vazão
        """
        # Valores médios baseados em padrões sazonais
        seasonal_pattern = {
            1: 120,   # Janeiro (verão - alta vazão)
            2: 115,   # Fevereiro 
            3: 110,   # Março
            4: 85,    # Abril (outono - vazão menor)
            5: 75,    # Maio
            6: 70,    # Junho (inverno - vazão baixa)
            7: 68,    # Julho
            8: 72,    # Agosto
            9: 80,    # Setembro (primavera - recuperação)
            10: 90,   # Outubro
            11: 105,  # Novembro 
            12: 115   # Dezembro (verão - alta vazão)
        }
        
        base_flow = seasonal_pattern.get(month, 85)
        
        # Pequeno fator de tendência temporal (muito leve)
        trend_factor = 1 + (year - 2000) * 0.002
        
        return base_flow * trend_factor
    
    def get_test_data(self) -> pd.DataFrame:
        """
        Retorna dados sintéticos de teste para demonstração
        
        Returns:
            DataFrame com dados observados vs preditos
        """
        # Gerar alguns pontos de teste para demonstração
        test_data = []
        
        for year in [2020, 2021, 2022]:
            for month in range(1, 13):
                predicted = self.predict_month(year, month)
                
                # Simular observado com alguma variação
                noise = np.random.normal(0, predicted * 0.1)
                observed = max(10, predicted + noise)
                
                test_data.append({
                    'year': year,
                    'month': month,
                    'observed': observed,
                    'predicted': predicted
                })
        
        return pd.DataFrame(test_data)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações sobre os modelos carregados"""
        if not self.is_loaded:
            return {
                'status': 'not_loaded',
                'message': 'Modelos não carregados'
            }
        
        return {
            'status': 'loaded',
            'training_info': self.training_info,
            'model_files': {
                'mlp_model': 'mlp_model.joblib',
                'xgb_model': 'xgb_model.joblib',
                'scaler': 'scaler.joblib'
            }
        }

# Instância global otimizada
optimized_flow_model = OptimizedFlowModel()

# Manter compatibilidade com a versão anterior
class FlowModel:
    """Wrapper para manter compatibilidade com API existente"""
    
    def __init__(self):
        self.optimized_model = optimized_flow_model
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Usa o modelo otimizado se disponível, senão fallback"""
        if self.optimized_model.is_loaded:
            return self.optimized_model.predict(features)
        else:
            # Fallback para previsão simples
            logger.warning("Usando previsão fallback - modelos não carregados")
            results = []
            for _, row in features.iterrows():
                year = int(row.get('year', 2023))
                month = int(row.get('month', 1))
                results.append(self.optimized_model.predict_month(year, month))
            return np.array(results)
    
    def predict_month(self, year: int, month: int) -> float:
        """Previsão para mês específico"""
        return self.optimized_model.predict_month(year, month)
    
    def get_test_data(self) -> pd.DataFrame:
        """Dados de teste"""
        return self.optimized_model.get_test_data()
    
    def train(self):
        """Método de compatibilidade - não faz nada pois modelo já é treinado"""
        logger.info("Modelo já foi pré-treinado. Use scripts/train_flow_model.py para re-treinar.")
        pass

# Instância global para manter compatibilidade
flow_model = FlowModel()