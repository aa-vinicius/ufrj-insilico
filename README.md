# UFRJ InSilico API

Uma API Rest para disponibilizar resultados de modelos meteorológicos e dados, transformando dados brutos em recursos científicos acessíveis via endpoints. O mecanismo central para a automação e reutilização de pesquisas de laboratório.

## 🌊 Modelo de Previsão de Vazão - Estação Funil (19091)

### Funcionalidades Principais

**Modelo de Ensemble ML**: Sistema híbrido usando MLP (Multi-Layer Perceptron) + XGBoost com otimização por algoritmo genético para predição de vazão mensal.

**Interface Web Interativa**: Dashboard completo com visualização temporal, métricas de performance e demonstração da API.

**API Robusta**: Endpoints para predição com validação estrita de todas as variáveis de entrada (23 variáveis obrigatórias).

## Configuração do Ambiente de Desenvolvimento

### Método Rápido (Recomendado)

```bash
# Executar script de configuração automática
source setup_dev.sh
```

### Método Manual

#### 1. Criar e ativar ambiente virtual

```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente virtual (Linux/Mac)
source venv/bin/activate

# Ativar ambiente virtual (Windows)
venv\Scripts\activate
```

#### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

#### 3. Executar a aplicação

```bash
# Desenvolvimento com hot reload (dentro do ambiente virtual)
python3 -m uvicorn app.main:app --reload

# A API estará disponível em http://localhost:8000
```

#### 4. Acessar documentação

- **Página de Documentação Interativa**: http://localhost:8000/ 
- **Swagger UI (FastAPI)**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

#### 5. Executar testes

```bash
# Executar testes (dentro do ambiente virtual)
python3 -m pytest -v
```

## Executar com Docker

```bash
# Construir imagem
docker build -t ufrj-insilico .

# Executar container
docker run -p 8000:8000 ufrj-insilico
```

## Endpoints Disponíveis

### 🌊 Modelo de Vazão - Estação Funil (19091)
- `POST /api/models/flow/predict` - Previsão de vazão mensal com ensemble ML
- `GET /api/models/flow/status` - Status do modelo e informações de treinamento
- `GET /api/models/flow/test-data` - Dados de teste (2020-2023) com métricas
- `GET /static/flow.html` - Interface web interativa com dashboard completo


### Dados Meteorológicos
- `GET /api/data/` - Lista dados meteorológicos com geometrias (filtros opcionais)
- `GET /api/data/points` - Lista apenas dados pontuais
- `GET /api/data/polygons` - Lista apenas dados de área
- `GET /api/data/locations` - Lista localizações disponíveis

### Modelos de ML
- `GET /api/models/` - Lista resultados de modelos com geometrias (filtros opcionais)
- `GET /api/models/points` - Lista previsões pontuais
- `GET /api/models/polygons` - Lista previsões de área
- `GET /api/models/forecast-areas` - Lista áreas de previsão

### Documentação
- `GET /` - Página de documentação interativa personalizada
- `GET /docs` - Documentação Swagger UI (FastAPI)
- `GET /redoc` - Documentação ReDoc (FastAPI)

## 🌊 API de Previsão de Vazão

### Modelo de Ensemble

O modelo utiliza uma abordagem híbrida com dois componentes principais:

- **MLP (Multi-Layer Perceptron)**: Rede neural com arquitetura (64, 32) neurônios
- **XGBoost**: Gradient boosting com `max_depth=5` e `n_estimators=100`
- **Otimização Genética**: Algoritmo DEAP para calibração de parâmetros
- **Mistura Adaptiva**: Gate sazonal para combinar predições dos modelos

### Exemplo de Uso

```bash
curl -X POST "http://localhost:8000/api/models/flow/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "year": 2024,
    "month": 6,
    "u2_min": -1.2,
    "u2_max": 4.5,
    "tmin_min": 16.8,
    "tmin_max": 21.2,
    "tmax_min": 25.3,
    "tmax_max": 30.7,
    "rs_min": 16.4,
    "rs_max": 24.8,
    "rh_min": 62.5,
    "rh_max": 85.2,
    "eto_min": 3.1,
    "eto_max": 5.2,
    "pr_min": 0.0,
    "pr_max": 180.5,
    "y_lag1": 95.6,
    "y_lag2": 87.3,
    "y_lag3": 102.1,
    "y_rm3": 95.0,
    "pr_lag1": 65.2,
    "pr_lag2": 78.9,
    "pr_lag3": 42.1,
    "pr_sum3": 186.2,
    "pr_api3": 89.4
  }'
```

### Resposta da API

```json
{
  "year": 2024,
  "month": 6,
  "predicted_flow": 220.61,
  "lower_bound": 194.14,
  "upper_bound": 247.07,
  "uncertainty": 26.47,
  "confidence_level": 0.83,
  "model_components": {
    "gate": 0.7,
    "sigma_mix": 21.45,
    "s_opt": 1.2
  }
}
```

### Variáveis de Entrada (23 obrigatórias)

#### Meteorológicas (14 variáveis)
- **Vento**: `u2_min`, `u2_max` (m/s)
- **Temperatura**: `tmin_min`, `tmin_max`, `tmax_min`, `tmax_max` (°C)
- **Radiação**: `rs_min`, `rs_max` (MJ/m²/dia)
- **Umidade**: `rh_min`, `rh_max` (%)
- **Evapotranspiração**: `eto_min`, `eto_max` (mm/dia)
- **Precipitação**: `pr_min`, `pr_max` (mm)

#### Lags de Vazão (4 variáveis)
- `y_lag1`, `y_lag2`, `y_lag3`: Vazões dos 3 meses anteriores (m³/s)
- `y_rm3`: Média móvel de 3 meses (m³/s)

#### Lags de Precipitação (5 variáveis)
- `pr_lag1`, `pr_lag2`, `pr_lag3`: Precipitação dos 3 meses anteriores (mm)
- `pr_sum3`: Soma dos 3 meses anteriores (mm)
- `pr_api3`: Índice de precipitação antecedente (mm)

### Performance do Modelo

Métricas calculadas no conjunto de teste (2020-2023):

- **MAE**: ~25.4 m³/s (Erro Absoluto Médio)
- **RMSE**: ~35.8 m³/s (Raiz do Erro Quadrático Médio)
- **R²**: ~0.85 (Coeficiente de Determinação)
- **NSE**: ~0.84 (Nash-Sutcliffe Efficiency)
- **Cobertura**: ~83% (Observações dentro do intervalo de predição)

### Parâmetros de Filtro

#### `/api/data/` e `/api/models/`
- `geometry_type`: Filtrar por tipo de geometria (`Point` ou `Polygon`)
- `data_type`: Filtrar por tipo de dado (`temperature`, `humidity`, etc.)
- `model_name`: Filtrar por nome do modelo (apenas em `/models/`)

## 📊 Interface Web do Modelo de Vazão

### Dashboard Interativo (`/static/flow.html`)

**Visualização Temporal**:
- Gráfico de série temporal (2020-2023) comparando observado vs previsto
- Bandas de incerteza das previsões (amarela) 
- Bandas de variabilidade dos dados observados (cinza)
- Métricas de performance em tempo real

**Demonstração da API**:
- Campo JSON editável com todas as 23 variáveis
- Botão para carregar exemplos aleatórios dos dados de teste
- Teste direto da API com feedback de status HTTP e tempo de resposta
- Validação automática de JSON e tratamento de erros

### Características Técnicas

- **Responsivo**: Funciona em desktop, tablet e mobile
- **Tempo Real**: Indicador de status do modelo
- **Interativo**: Plotly.js para gráficos dinâmicos
- **Validação Estrita**: Todas as 23 variáveis obrigatórias

## Estrutura dos Dados

### Dados Meteorológicos (`/data/`)

Os dados incluem geometrias GeoJSON para localização espacial:

```json
{
  "id": 1,
  "type": "temperature",
  "value": 25.3,
  "unit": "C",
  "timestamp": "2025-09-29T12:00:00Z",
  "geometry": {
    "type": "Point",
    "coordinates": [-43.1729, -22.9068]
  },
  "location_name": "Campus UFRJ - Cidade Universitária"
}
```

### Resultados de Modelos (`/models/`)

Os resultados incluem previsões com geometrias e metadados adicionais:

```json
{
  "id": 1,
  "model": "rain_prediction",
  "result": "rain",
  "confidence": 0.87,
  "timestamp": "2025-09-29T12:00:00Z",
  "forecast_time": "2025-09-29T18:00:00Z",
  "geometry": {
    "type": "Polygon",
    "coordinates": [[[-43.73, -22.82], [-43.10, -22.82], ...]]
  },
  "location_name": "Zona Oeste do Rio de Janeiro"
}
```

### Tipos de Geometria Suportados

- **Point**: Localização específica (estação meteorológica, sensor)
- **Polygon**: Área de cobertura (região metropolitana, zona de previsão)

## 🌐 Interface Web Interativa

A aplicação inclui uma página web moderna e responsiva que serve como:

### 📋 Documentação Técnica Completa
- Visão geral da API e funcionalidades
- Explicação detalhada de cada endpoint
- Exemplos de estruturas de dados
- Guia de parâmetros disponíveis

### 🧪 Ambiente de Testes Interativo
- **Testes rápidos**: Botões para testar cada endpoint individualmente
- **Playground avançado**: Interface para construir requisições personalizadas
- **Filtros dinâmicos**: Seleção de parâmetros por tipo de geometria, modelo, etc.
- **Resultados formatados**: JSON com syntax highlighting e botões de cópia

### 📊 Funcionalidades Avançadas
- **Status da API**: Indicador em tempo real do status do servidor
- **Navegação fluida**: Menu sticky com scroll suave entre seções
- **Design responsivo**: Interface adaptável para desktop, tablet e mobile
- **Tempo de resposta**: Medição e exibição do tempo de cada requisição
- **Cópia fácil**: Botões para copiar resultados JSON para clipboard

### 🎨 Interface Moderna
- Design clean e profissional
- Gradientes e animações sutis
- Ícones intuitivos e cores semânticas
- Layout em grid responsivo

## 🧪 Scripts de Treinamento e Análise

### Treinamento do Modelo
```bash
# Treinar modelos e salvar para produção
python scripts/train_and_save_models.py

# Executar lógica completa do notebook (desenvolvimento)
python scripts/run_funil_model.py
```

### Dados de Teste
- **Período**: 2020-2023 (48 meses)
- **Treinamento**: 1998-2017 (20 anos)
- **Fonte**: Estação 19091 (Funil) - dados meteorológicos + vazão observada
- **Localização**: `modelos_staging/flow/agragado_meteo_vazao_shifted_station_19091_extended.csv`

## Estrutura do Projeto

```
├── app/
│   ├── __init__.py
│   ├── main.py                    # Aplicação principal FastAPI
│   ├── routers/                   # Definição dos endpoints
│   │   ├── data.py               # Endpoints para dados meteorológicos
│   │   └── models.py             # Endpoints para modelos (+ vazão)
│   ├── models/                    # Modelos de Machine Learning
│   │   ├── flow_model.py         # Modelo completo de vazão (referência)
│   │   ├── optimized_flow_model.py # Modelo otimizado para produção
│   │   └── saved_models/         # Modelos treinados (.joblib)
│   │       ├── mlpA_model.joblib
│   │       ├── xgbA_model.joblib
│   │       ├── mlpB_model.joblib
│   │       ├── xgbB_model.joblib
│   │       ├── scaler.joblib
│   │       └── training_info.joblib
│   └── mocks/                    # Dados mockados para desenvolvimento
│       ├── data_mock.py         # Mock de dados meteorológicos
│       └── models_mock.py       # Mock de resultados de modelos
├── scripts/                      # Scripts de treinamento e análise
│   ├── train_and_save_models.py # Treinar e salvar modelos para produção
│   └── run_funil_model.py       # Executar lógica completa do notebook
├── modelos_staging/             # Dados e notebooks de desenvolvimento
│   └── flow/
│       ├── VAZÃO_FUNIL.ipynb   # Notebook original v6.8
│       └── agragado_meteo_vazao_shifted_station_19091_extended.csv
├── static/                      # Interface web
│   ├── flow.html               # Dashboard do modelo de vazão
│   ├── flow-script.js          # JavaScript da interface
│   ├── flow-styles.css         # Estilos CSS da interface
│   ├── index.html              # Página principal da API
│   ├── script.js               # JavaScript geral
│   └── styles.css              # Estilos CSS gerais
├── tests/                       # Testes automatizados
│   ├── test_flow_model.py      # Testes do modelo de vazão
│   ├── test_data.py            # Testes dos endpoints de dados
│   └── test_models.py          # Testes gerais dos modelos
├── requirements.txt             # Dependências Python
├── Dockerfile                   # Configuração Docker
├── setup_dev.sh                # Script de configuração do ambiente
└── README.md                   # Documentação do projeto
```

## 🚀 Próximos Passos

### Melhorias Planejadas
- [ ] Implementação de novos modelos para outras estações
- [ ] API de retreinamento automático
- [ ] Cache Redis para predições frequentes  
- [ ] Monitoramento e logging avançado
- [ ] Exportação de dados em múltiplos formatos
- [ ] Dashboard de administração para gerenciar modelos

### Contribuições
Contribuições são bem-vindas! Por favor, abra uma issue ou pull request para sugestões de melhorias.
