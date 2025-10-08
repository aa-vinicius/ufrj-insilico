#!/usr/bin/env python3
"""
Script para treinar e salvar o modelo de previsão de vazão.
Este script deve ser executado periodicamente para atualizar os modelos.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import logging

# Adicionar o diretório raiz ao path para importar módulos do app
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.models.flow_model import FlowModel

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlowModelTrainer:
    """Classe para treinar e salvar modelos de vazão"""
    
    def __init__(self):
        self.models_dir = os.path.join(os.path.dirname(__file__), '..', 'app', 'models', 'saved_models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.mlp_model = None
        self.xgb_model = None
        self.scaler = None
        self.training_info = {}
        
    def create_synthetic_data(self):
        """
        Cria dados sintéticos baseados no padrão do notebook VAZÃO_FUNIL.ipynb
        Para produção, substitua por dados reais
        """
        logger.info("Criando dados sintéticos para treinamento...")
        
        # Criar dados temporais (1998-2022)
        years = list(range(1998, 2023))
        months = list(range(1, 13))
        
        data = []
        
        for year in years:
            for month in months:
                # Padrão sazonal baseado no que observamos no notebook
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * (month - 1) / 12)
                trend_factor = 1 + (year - 1998) * 0.01  # Tendência leve
                
                # Features meteorológicas sintéticas
                base_temp = 20 + 10 * np.sin(2 * np.pi * (month - 1) / 12)
                
                record = {
                    'year': year,
                    'month': month,
                    'u2_min': np.random.normal(2.5, 0.5),
                    'u2_max': np.random.normal(4.5, 0.8),
                    'tmin_min': base_temp - 5 + np.random.normal(0, 2),
                    'tmin_max': base_temp - 2 + np.random.normal(0, 2),
                    'tmax_min': base_temp + 2 + np.random.normal(0, 2),
                    'tmax_max': base_temp + 8 + np.random.normal(0, 2),
                    'rs_min': np.random.normal(15, 3),
                    'rs_max': np.random.normal(25, 4),
                    'rh_min': np.random.normal(60, 10),
                    'rh_max': np.random.normal(85, 8),
                    'eto_min': np.random.normal(3, 0.5),
                    'eto_max': np.random.normal(6, 1),
                    'pr_min': np.random.exponential(2),
                    'pr_max': np.random.exponential(8),
                }
                
                # Vazão target baseada em padrões sazonais + noise
                base_flow = 100 * seasonal_factor * trend_factor
                noise = np.random.normal(0, base_flow * 0.15)
                record['flow'] = max(10, base_flow + noise)
                
                data.append(record)
        
        df = pd.DataFrame(data)
        
        # Adicionar features de lag
        df = df.sort_values(['year', 'month']).reset_index(drop=True)
        df['y_lag1'] = df['flow'].shift(1)
        df['y_lag2'] = df['flow'].shift(2) 
        df['y_lag3'] = df['flow'].shift(3)
        
        # Remover primeiras linhas com NaN
        df = df.dropna().reset_index(drop=True)
        
        logger.info(f"Dados criados: {len(df)} registros de {df['year'].min()}-{df['year'].max()}")
        return df
        
    def prepare_features(self, df):
        """Prepara features para treinamento"""
        feature_cols = [
            'year', 'month', 'u2_min', 'u2_max', 'tmin_min', 'tmin_max',
            'tmax_min', 'tmax_max', 'rs_min', 'rs_max', 'rh_min', 'rh_max',
            'eto_min', 'eto_max', 'pr_min', 'pr_max', 'y_lag1', 'y_lag2', 'y_lag3'
        ]
        
        X = df[feature_cols].copy()
        y = df['flow'].copy()
        
        return X, y, feature_cols
        
    def train_models(self, X, y):
        """Treina os modelos MLP e XGBoost"""
        logger.info("Dividindo dados em treino/teste...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Normalizar features
        logger.info("Normalizando features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Treinar MLP
        logger.info("Treinando modelo MLP...")
        self.mlp_model = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=0.001,
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        
        self.mlp_model.fit(X_train_scaled, y_train)
        mlp_pred = self.mlp_model.predict(X_test_scaled)
        
        # Treinar XGBoost
        logger.info("Treinando modelo XGBoost...")
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=20,
            eval_metric='rmse'
        )
        
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        xgb_pred = self.xgb_model.predict(X_test)
        
        # Ensemble prediction
        ensemble_pred = 0.6 * mlp_pred + 0.4 * xgb_pred
        
        # Calcular métricas
        mlp_rmse = np.sqrt(mean_squared_error(y_test, mlp_pred))
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        
        mlp_r2 = r2_score(y_test, mlp_pred)
        xgb_r2 = r2_score(y_test, xgb_pred)
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        
        self.training_info = {
            'training_date': datetime.now().isoformat(),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'mlp_rmse': float(mlp_rmse),
            'mlp_r2': float(mlp_r2),
            'xgb_rmse': float(xgb_rmse),
            'xgb_r2': float(xgb_r2),
            'ensemble_rmse': float(ensemble_rmse),
            'ensemble_r2': float(ensemble_r2),
            'feature_columns': list(X.columns)
        }
        
        logger.info(f"Treinamento concluído:")
        logger.info(f"  MLP RMSE: {mlp_rmse:.2f}, R²: {mlp_r2:.3f}")
        logger.info(f"  XGB RMSE: {xgb_rmse:.2f}, R²: {xgb_r2:.3f}")
        logger.info(f"  Ensemble RMSE: {ensemble_rmse:.2f}, R²: {ensemble_r2:.3f}")
        
        return X_test, y_test
        
    def save_models(self):
        """Salva os modelos treinados"""
        logger.info("Salvando modelos...")
        
        # Salvar MLP
        mlp_path = os.path.join(self.models_dir, 'mlp_model.joblib')
        joblib.dump(self.mlp_model, mlp_path)
        
        # Salvar XGBoost
        xgb_path = os.path.join(self.models_dir, 'xgb_model.joblib')
        joblib.dump(self.xgb_model, xgb_path)
        
        # Salvar scaler
        scaler_path = os.path.join(self.models_dir, 'scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        # Salvar informações de treinamento
        info_path = os.path.join(self.models_dir, 'training_info.joblib')
        joblib.dump(self.training_info, info_path)
        
        logger.info(f"Modelos salvos em: {self.models_dir}")
        
    def run_training(self):
        """Executa o pipeline completo de treinamento"""
        try:
            # Criar/carregar dados
            df = self.create_synthetic_data()
            
            # Preparar features
            X, y, feature_cols = self.prepare_features(df)
            
            # Treinar modelos
            X_test, y_test = self.train_models(X, y)
            
            # Salvar modelos
            self.save_models()
            
            logger.info("Treinamento concluído com sucesso!")
            return True
            
        except Exception as e:
            logger.error(f"Erro durante treinamento: {str(e)}")
            return False

def main():
    """Função principal"""
    trainer = FlowModelTrainer()
    success = trainer.run_training()
    
    if success:
        print("✅ Modelos treinados e salvos com sucesso!")
    else:
        print("❌ Erro durante o treinamento!")
        sys.exit(1)

if __name__ == "__main__":
    main()