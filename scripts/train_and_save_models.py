# Script para treinar e salvar modelos para a API
# Baseado no notebook VAZÃO_FUNIL.ipynb v6.8_balanced_prp_operacional

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from deap import base, creator, tools, algorithms
import random, openpyxl
import joblib
import os

# Caminhos
DATA_PATH = Path("modelos_staging/flow/agragado_meteo_vazao_shifted_station_19091_extended.csv")
MODELS_DIR = Path("app/models/saved_models")

FEATS_BASE = ["u2_min","u2_max","tmin_min","tmin_max","tmax_min","tmax_max",
              "rs_min","rs_max","rh_min","rh_max","eto_min","eto_max","pr_min","pr_max"]
COL_YEAR, COL_MONTH = "year","month"

def nse(y_true,y_pred):
    denom = np.sum((y_true - np.mean(y_true))**2)
    return 1 - np.sum((y_true - y_pred)**2)/denom if denom != 0 else np.nan

def kge(y_true,y_pred):
    r = pearsonr(y_true,y_pred)[0]
    mo, ms = np.mean(y_true), np.mean(y_pred)
    beta = ms/mo if mo!=0 else np.nan
    cvo = np.std(y_true,ddof=1)/mo if mo!=0 else np.nan
    cvs = np.std(y_pred,ddof=1)/ms if ms!=0 else np.nan
    gamma = cvs/cvo if cvo!=0 else np.nan
    return 1 - np.sqrt((r-1)**2 + (beta-1)**2 + (gamma-1)**2)

def pbias(y_true,y_pred):
    return 100*np.sum(y_true - y_pred)/np.sum(y_true)

def metrics(y_true,y_pred,UF):
    return dict(
        RMSE=np.sqrt(mean_squared_error(y_true,y_pred)),
        MAE=mean_absolute_error(y_true,y_pred),
        R2=r2_score(y_true,y_pred),
        NSE=nse(y_true,y_pred),
        KGE=kge(y_true,y_pred),
        PBIAS=pbias(y_true,y_pred),
        PEARSON=pearsonr(y_true,y_pred)[0],
        NSE_picos=nse(y_true[y_true>UF],y_pred[y_true>UF]) if np.any(y_true>UF) else np.nan
    )

def mad_sigma(res):
    med = np.median(res)
    return 1.4826 * np.median(np.abs(res - med))

def train_and_save_models():
    print("===== Treinando e salvando modelos para API =====")
    
    df = pd.read_csv(DATA_PATH).sort_values([COL_YEAR,COL_MONTH]).copy()
    
    # Preparar dados
    df["pr_mean"] = (df["pr_min"] + df["pr_max"]) / 2.0
    df["pr_lag1"] = df["pr_mean"].shift(1)
    df["pr_lag2"] = df["pr_mean"].shift(2)
    df["pr_lag3"] = df["pr_mean"].shift(3)
    df["pr_sum3"] = df["pr_mean"].rolling(3,min_periods=1).sum().shift(1)
    df["pr_api3"] = (0.5*df["pr_mean"].shift(1) + 0.3*df["pr_mean"].shift(2) + 0.2*df["pr_mean"].shift(3))
    df["y_lag1"]=df["flow_next_month"].shift(1)
    df["y_lag2"]=df["flow_next_month"].shift(2)
    df["y_lag3"]=df["flow_next_month"].shift(3)
    df["y_rm3"]=df["flow_next_month"].rolling(3,min_periods=1).mean().shift(1)
    
    needed = FEATS_BASE + ["flow_next_month","flow_next_month_max","flow_next_month_min",
                           COL_YEAR,COL_MONTH,"y_lag1","y_lag2","y_lag3","y_rm3",
                           "pr_mean","pr_lag1","pr_lag2","pr_lag3","pr_sum3","pr_api3"]
    df = df.dropna(subset=needed).copy()
    df[COL_YEAR] = df[COL_YEAR].astype(int)
    df[COL_MONTH] = df[COL_MONTH].astype(int)
    df["month_sin"] = np.sin(2*np.pi*df[COL_MONTH]/12)
    df["month_cos"] = np.cos(2*np.pi*df[COL_MONTH]/12)
    
    FEATS = FEATS_BASE + ["month_sin","month_cos","y_lag1","y_lag2","y_lag3","y_rm3",
                          "pr_lag1","pr_lag2","pr_lag3","pr_sum3","pr_api3"]
    
    # Splits
    train = (df[COL_YEAR]>=1998)&(df[COL_YEAR]<=2017)
    val = (df[COL_YEAR]>=2018)&(df[COL_YEAR]<=2019)
    test = (df[COL_YEAR]>=2020)&(df[COL_YEAR]<=2023)
    
    X_tr = df.loc[train,FEATS].to_numpy()
    X_val= df.loc[val,FEATS].to_numpy()
    X_te = df.loc[test,FEATS].to_numpy()
    y_trA= df.loc[train,"flow_next_month"].to_numpy(float)
    y_trB= df.loc[train,"flow_next_month_max"].to_numpy(float)
    y_teA= df.loc[test,"flow_next_month"].to_numpy(float)
    y_teB= df.loc[test,"flow_next_month_max"].to_numpy(float)
    min_flow_te = df.loc[test,"flow_next_month_min"].to_numpy()
    max_flow_te = df.loc[test,"flow_next_month_max"].to_numpy()
    obs_width = np.maximum(1e-6, max_flow_te - min_flow_te)
    
    dates_te = pd.to_datetime(dict(year=df.loc[test,COL_YEAR],month=df.loc[test,COL_MONTH],day=1))
    Q1,Q3 = np.percentile(y_trA,[25,75]); UF = Q3 + 1.5*(Q3-Q1)
    
    scaler = StandardScaler().fit(np.vstack([X_tr,X_val]))
    X_tr_s,X_te_s = scaler.transform(X_tr),scaler.transform(X_te)
    
    # Modelos A/B
    mlpA = MLPRegressor(hidden_layer_sizes=(64,32),alpha=2e-3,max_iter=3000,random_state=42,early_stopping=True)
    xgbA = XGBRegressor(n_estimators=900,learning_rate=0.02,max_depth=5,subsample=0.9,colsample_bytree=0.9,random_state=42)
    mlpA.fit(X_tr_s,y_trA); xgbA.fit(X_tr,y_trA)
    yhat_trA = 0.5*mlpA.predict(X_tr_s)+0.5*xgbA.predict(X_tr)
    yhat_teA = 0.5*mlpA.predict(X_te_s)+0.5*xgbA.predict(X_te)
    
    mlpB = MLPRegressor(hidden_layer_sizes=(64,32),alpha=2e-3,max_iter=3000,random_state=43,early_stopping=True)
    xgbB = XGBRegressor(n_estimators=900,learning_rate=0.02,max_depth=5,subsample=0.9,colsample_bytree=0.9,random_state=43)
    mlpB.fit(X_tr_s,y_trB); xgbB.fit(X_tr,y_trB)
    yhat_trB = 0.5*mlpB.predict(X_tr_s)+0.5*xgbB.predict(X_tr)
    yhat_teB = 0.5*mlpB.predict(X_te_s)+0.5*xgbB.predict(X_te)
    
    # Ensemble adaptativo
    gate = np.where(df.loc[test,COL_MONTH].isin([11,12,1,2]),0.3,0.7)
    yhat_te = gate*yhat_teA + (1-gate)*yhat_teB
    y_te = gate*y_teA + (1-gate)*y_teB
    
    # Incerteza
    sigma_A = mad_sigma(y_teA - yhat_teA)
    sigma_B = mad_sigma(y_teB - yhat_teB)
    w_desacordo = np.where(df.loc[test,COL_MONTH].isin([11,12,1,2]),0.22,0.08)
    sigma_mix = np.sqrt(gate*sigma_A**2 + (1-gate)*sigma_B**2 + w_desacordo*(yhat_teA-yhat_teB)**2)
    f_sazonal = np.where(df.loc[test,COL_MONTH].isin([11,12,1,2]),1.0,0.75)
    sigma_mix *= f_sazonal
    
    # GA Balanceado (simplificado para API)
    s_opt = 1.258  # Valor obtido do treinamento anterior
    
    σ = sigma_mix*s_opt
    low,up = yhat_te-σ,yhat_te+σ
    p = np.mean((y_te>=low)&(y_te<=up))
    r = np.mean((up-low)/(np.std(y_te)+1e-6))
    width_ratio = np.mean((up-low)/obs_width)
    
    # Métricas
    m_tr = metrics(y_trA,yhat_trA,UF)
    m_te = metrics(y_te,yhat_te,UF)
    
    # Dados de teste para API
    test_data = []
    test_dates = dates_te.tolist()
    test_years = df.loc[test,COL_YEAR].tolist() 
    test_months = df.loc[test,COL_MONTH].tolist()
    
    for i in range(len(test_dates)):
        test_data.append({
            'year': test_years[i],
            'month': test_months[i],
            'observed': float(y_te[i]),
            'predicted': float(yhat_te[i]),
            'lower_bound': float(low[i]),
            'upper_bound': float(up[i]),
            'obs_min': float(min_flow_te[i]),
            'obs_max': float(max_flow_te[i]),
            'uncertainty': float(σ[i])
        })
    
    # Criar diretório se não existir
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Salvar modelos e dados
    joblib.dump(mlpA, MODELS_DIR / 'mlpA_model.joblib')
    joblib.dump(xgbA, MODELS_DIR / 'xgbA_model.joblib') 
    joblib.dump(mlpB, MODELS_DIR / 'mlpB_model.joblib')
    joblib.dump(xgbB, MODELS_DIR / 'xgbB_model.joblib')
    joblib.dump(scaler, MODELS_DIR / 'scaler.joblib')
    
    # Informações de treinamento
    training_info = {
        'feature_columns': FEATS,
        'train_metrics': m_tr,
        'test_metrics': m_te,
        'sigma_A': float(sigma_A),
        'sigma_B': float(sigma_B),
        's_opt': float(s_opt),
        'coverage': float(p),
        'test_data': test_data,
        'model_version': 'v6.8_balanced_prp_operacional',
        'trained_on': str(pd.Timestamp.now())
    }
    
    joblib.dump(training_info, MODELS_DIR / 'training_info.joblib')
    
    print(f"✅ Modelos salvos em: {MODELS_DIR}")
    print(f"✅ Cobertura: {p:.2f}")
    print(f"✅ Dados de teste: {len(test_data)} registros")
    print(f"✅ Features: {len(FEATS)}")
    
    return training_info

if __name__ == "__main__":
    train_and_save_models()