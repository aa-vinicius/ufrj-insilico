# UFRJ InSilico API

Uma API Rest para disponibilizar resultados de modelos meteorológicos e dados, transformando dados brutos em recursos científicos acessíveis via endpoints. O mecanismo central para a automação e reutilização de pesquisas de laboratório.

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

### Dados Meteorológicos
- `GET /data/` - Lista dados meteorológicos com geometrias (filtros opcionais)
- `GET /data/points` - Lista apenas dados pontuais
- `GET /data/polygons` - Lista apenas dados de área
- `GET /data/locations` - Lista localizações disponíveis

### Modelos de ML
- `GET /models/` - Lista resultados de modelos com geometrias (filtros opcionais)
- `GET /models/points` - Lista previsões pontuais
- `GET /models/polygons` - Lista previsões de área
- `GET /models/forecast-areas` - Lista áreas de previsão

### Documentação
- `GET /` - Página de documentação interativa personalizada
- `GET /docs` - Documentação Swagger UI (FastAPI)
- `GET /redoc` - Documentação ReDoc (FastAPI)

### Parâmetros de Filtro

#### `/data/` e `/models/`
- `geometry_type`: Filtrar por tipo de geometria (`Point` ou `Polygon`)
- `data_type`: Filtrar por tipo de dado (`temperature`, `humidity`, etc.)
- `model_name`: Filtrar por nome do modelo (apenas em `/models/`)

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

## Estrutura do Projeto

```
├── app/
│   ├── __init__.py
│   ├── main.py              # Aplicação principal FastAPI
│   ├── routers/             # Definição dos endpoints
│   │   ├── data.py         # Endpoints para dados meteorológicos
│   │   └── models.py       # Endpoints para modelos de ML
│   └── mocks/              # Dados mockados para desenvolvimento
│       ├── data_mock.py    # Mock de dados meteorológicos
│       └── models_mock.py  # Mock de resultados de modelos
├── tests/                  # Testes automatizados
├── requirements.txt        # Dependências Python
├── Dockerfile             # Configuração Docker
└── README.md              # Documentação do projeto
```
