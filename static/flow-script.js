// JavaScript para página do modelo de vazão

const API_BASE = window.location.origin;

// Estado da aplicação
let testData = [];
let modelStatus = { is_trained: false };

// Inicialização
document.addEventListener('DOMContentLoaded', async () => {
    loadExample(); // Pré-preencher imediatamente com exemplo fixo
    await checkModelStatus();
    await loadTestData();
    setupEventListeners();
    renderTimeSeriesChart();
});

// Verificar status do modelo
async function checkModelStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/models/flow/status`);
        modelStatus = await response.json();
        
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        
        if (modelStatus.is_trained) {
            statusDot.className = 'status-dot online';
            statusText.textContent = 'Modelo Treinado';
        } else {
            statusDot.className = 'status-dot offline';
            statusText.textContent = 'Modelo não Treinado';
        }
        
    } catch (error) {
        console.error('Erro ao verificar status:', error);
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        statusDot.className = 'status-dot offline';
        statusText.textContent = 'Erro de Conexão';
    }
}

// Carregar dados de teste
async function loadTestData() {
    try {
        const response = await fetch(`${API_BASE}/api/models/flow/test-data`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        testData = await response.json();
        
        console.log('Dados de teste carregados:', testData.length, 'registros');
        
        // Calcular métricas
        calculateAndDisplayMetrics();
        
    } catch (error) {
        console.error('Erro ao carregar dados de teste:', error);
        
        // Exibir erro na interface
        displayError('Erro ao carregar dados de teste', error.message);
        testData = []; // Array vazio, sem dados mock
    }
}



// Exibir erro na interface
function displayError(title, message) {
    const metricsGrid = document.getElementById('metricsGrid');
    metricsGrid.innerHTML = `
        <div style="grid-column: 1 / -1; padding: 2rem; text-align: center; background: #fee; border: 1px solid #fcc; border-radius: 0.5rem; color: #c33;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">⚠️</div>
            <div style="font-weight: 600; margin-bottom: 0.5rem;">${title}</div>
            <div style="font-size: 0.9rem;">${message}</div>
        </div>
    `;
}

// Calcular e exibir métricas
function calculateAndDisplayMetrics() {
    if (testData.length === 0) {
        displayError('Métricas indisponíveis', 'Os dados de teste não puderam ser carregados da API');
        return;
    }
    
    const observed = testData.map(d => d.observed);
    const predicted = testData.map(d => d.predicted);
    
    // Calcular métricas básicas
    const mae = calculateMAE(observed, predicted);
    const rmse = calculateRMSE(observed, predicted);
    const r2 = calculateR2(observed, predicted);
    const nse = calculateNSE(observed, predicted);
    
    // Calcular cobertura do intervalo
    const coverage = testData.reduce((acc, d) => {
        return acc + (d.observed >= d.lower_bound && d.observed <= d.upper_bound ? 1 : 0);
    }, 0) / testData.length;
    
    // Exibir métricas
    displayMetrics({
        mae: mae.toFixed(2),
        rmse: rmse.toFixed(2),
        r2: r2.toFixed(3),
        nse: nse.toFixed(3),
        coverage: (coverage * 100).toFixed(1) + '%',
        samples: testData.length
    });
}

// Funções de cálculo de métricas
function calculateMAE(observed, predicted) {
    return observed.reduce((sum, obs, i) => sum + Math.abs(obs - predicted[i]), 0) / observed.length;
}

function calculateRMSE(observed, predicted) {
    const mse = observed.reduce((sum, obs, i) => sum + Math.pow(obs - predicted[i], 2), 0) / observed.length;
    return Math.sqrt(mse);
}

function calculateR2(observed, predicted) {
    const meanObs = observed.reduce((sum, val) => sum + val, 0) / observed.length;
    const ssRes = observed.reduce((sum, obs, i) => sum + Math.pow(obs - predicted[i], 2), 0);
    const ssTot = observed.reduce((sum, obs) => sum + Math.pow(obs - meanObs, 2), 0);
    return 1 - (ssRes / ssTot);
}

function calculateNSE(observed, predicted) {
    const meanObs = observed.reduce((sum, val) => sum + val, 0) / observed.length;
    const numerator = observed.reduce((sum, obs, i) => sum + Math.pow(obs - predicted[i], 2), 0);
    const denominator = observed.reduce((sum, obs) => sum + Math.pow(obs - meanObs, 2), 0);
    return 1 - (numerator / denominator);
}

// Exibir métricas na interface
function displayMetrics(metrics) {
    const metricsGrid = document.getElementById('metricsGrid');
    
    metricsGrid.innerHTML = `
        <div class="metric-card">
            <div class="metric-value">${metrics.mae}</div>
            <div class="metric-label">MAE (m³/s)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${metrics.rmse}</div>
            <div class="metric-label">RMSE (m³/s)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${metrics.r2}</div>
            <div class="metric-label">R²</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${metrics.nse}</div>
            <div class="metric-label">NSE</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${metrics.coverage}</div>
            <div class="metric-label">Cobertura</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${metrics.samples}</div>
            <div class="metric-label">Amostras</div>
        </div>
    `;
}

// Renderizar gráfico da série temporal
function renderTimeSeriesChart() {
    if (testData.length === 0) {
        // Exibir mensagem de erro no gráfico
        const chartContainer = document.getElementById('timeSeriesChart');
        chartContainer.innerHTML = `
            <div style="display: flex; align-items: center; justify-content: center; height: 400px; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 0.375rem;">
                <div style="text-align: center; color: #6c757d;">
                    <div style="font-size: 2rem; margin-bottom: 1rem;">⚠️</div>
                    <div style="font-size: 1.1rem; font-weight: 500; margin-bottom: 0.5rem;">Dados não disponíveis</div>
                    <div style="font-size: 0.9rem;">Não foi possível carregar os dados de teste da API</div>
                </div>
            </div>
        `;
        return;
    }
    
    // Preparar dados para o gráfico
    const dates = testData.map(d => `${d.year}-${d.month.toString().padStart(2, '0')}-01`);
    
    const traces = [
        {
            x: dates,
            y: testData.map(d => d.lower_bound),
            type: 'scatter',
            mode: 'lines',
            fill: 'none',
            line: { color: 'transparent' },
            name: 'Intervalo Inferior',
            hoverinfo: 'skip',
            showlegend: false
        },
        {
            x: dates,
            y: testData.map(d => d.upper_bound),
            type: 'scatter',
            mode: 'lines',
            fill: 'tonexty',
            fillcolor: 'rgba(255, 165, 0, 0.3)',
            line: { color: 'transparent' },
            name: 'Incerteza (previsão)',
            hoverinfo: 'skip'
        },
        {
            x: dates,
            y: testData.map(d => d.obs_min),
            type: 'scatter',
            mode: 'lines',
            fill: 'none',
            line: { color: 'transparent' },
            name: 'Obs Min',
            hoverinfo: 'skip',
            showlegend: false
        },
        {
            x: dates,
            y: testData.map(d => d.obs_max),
            type: 'scatter',
            mode: 'lines',
            fill: 'tonexty',
            fillcolor: 'rgba(128, 128, 128, 0.3)',
            line: { color: 'transparent' },
            name: 'Min/max (observado)',
            hoverinfo: 'skip'
        },
        {
            x: dates,
            y: testData.map(d => d.observed),
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#000000', width: 2 },
            marker: { color: '#000000', size: 6 },
            name: 'Observado'
        },
        {
            x: dates,
            y: testData.map(d => d.predicted),
            type: 'scatter',
            mode: 'lines',
            line: { color: '#ef4444', width: 2, dash: 'dash' },
            name: 'Previsto'
        }
    ];
    
    const layout = {
        title: {
            text: 'Station 19091 — Teste (Observado vs Previsto)',
            font: { size: 16 }
        },
        xaxis: {
            title: 'Data',
            type: 'date'
        },
        yaxis: {
            title: 'Vazão (m³/s)'
        },
        legend: {
            x: 1,
            xanchor: 'right',
            y: 1
        },
        margin: { l: 60, r: 60, t: 60, b: 60 },
        showlegend: true,
        hovermode: 'x unified'
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
    };
    
    Plotly.newPlot('timeSeriesChart', traces, layout, config);
}

// Configurar event listeners
function setupEventListeners() {
    // Toggle campos avançados
    const predTypeRadios = document.querySelectorAll('input[name="predType"]');
    const advancedFields = document.getElementById('advancedFields');
    
    predTypeRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.value === 'advanced') {
                advancedFields.style.display = 'block';
            } else {
                advancedFields.style.display = 'none';
            }
        });
    });
    
    // Botão carregar exemplo
    document.getElementById('loadExampleBtn').addEventListener('click', loadExample);
    
    // Botão fazer previsão
    document.getElementById('predictBtn').addEventListener('click', makePrediction);
}

// Carregar exemplo
function loadExample() {
    // Sempre inicializar com um exemplo baseado nos dados reais (Janeiro 2023)
    let examplePayload;
    
    // Escolher um exemplo aleatório dos dados de teste se disponível
    if (testData && testData.length > 0) {
        const randomIndex = Math.floor(Math.random() * testData.length);
        const testRecord = testData[randomIndex];
        
        examplePayload = {
            year: testRecord.year,
            month: testRecord.month,
            u2_min: -1.5 + Math.random() * 3, // Variação realística
            u2_max: 3.0 + Math.random() * 4,
            tmin_min: 14.0 + Math.random() * 6,
            tmin_max: 20.0 + Math.random() * 5,
            tmax_min: 23.0 + Math.random() * 4,
            tmax_max: 29.0 + Math.random() * 6,
            rs_min: 12.0 + Math.random() * 8,
            rs_max: 22.0 + Math.random() * 8,
            rh_min: 55.0 + Math.random() * 20,
            rh_max: 75.0 + Math.random() * 20,
            eto_min: 2.5 + Math.random() * 2,
            eto_max: 4.5 + Math.random() * 2,
            pr_min: 0.0,
            pr_max: 50.0 + Math.random() * 300,
            y_lag1: 50.0 + Math.random() * 200,
            y_lag2: 50.0 + Math.random() * 200,
            y_lag3: 50.0 + Math.random() * 200,
            y_rm3: 50.0 + Math.random() * 200,
            pr_lag1: 10.0 + Math.random() * 100,
            pr_lag2: 10.0 + Math.random() * 100,
            pr_lag3: 10.0 + Math.random() * 100,
            pr_sum3: 30.0 + Math.random() * 300,
            pr_api3: 50.0 + Math.random() * 150
        };
        
        // Arredondar valores para 1 casa decimal
        Object.keys(examplePayload).forEach(key => {
            if (typeof examplePayload[key] === 'number' && key !== 'year' && key !== 'month') {
                examplePayload[key] = Math.round(examplePayload[key] * 10) / 10;
            }
        });
        
    } else {
        // Usar exemplo fixo baseado nos dados reais (Janeiro 2023) quando não há dados disponíveis
        console.warn('Dados de teste não disponíveis, usando exemplo fixo');
        examplePayload = {
            year: 2023,
            month: 1,
            u2_min: -2.9,
            u2_max: 5.0,
            tmin_min: 16.5,
            tmin_max: 22.5,
            tmax_min: 24.8,
            tmax_max: 31.2,
            rs_min: 15.1,
            rs_max: 25.2,
            rh_min: 61.2,
            rh_max: 88.7,
            eto_min: 3.2,
            eto_max: 5.8,
            pr_min: 0.0,
            pr_max: 356.0,
            y_lag1: 172.43,
            y_lag2: 179.47,
            y_lag3: 90.92,
            y_rm3: 147.61,
            pr_lag1: 42.5,
            pr_lag2: 83.5,
            pr_lag3: 45.2,
            pr_sum3: 171.2,
            pr_api3: 104.9
        };
    }
    
    document.getElementById('requestPayload').value = JSON.stringify(examplePayload, null, 2);
}

// Fazer previsão
async function makePrediction() {
    const predictBtn = document.getElementById('predictBtn');
    const responsePayload = document.getElementById('responsePayload');
    const responseStatus = document.getElementById('responseStatus');
    const responseTime = document.getElementById('responseTime');
    
    predictBtn.disabled = true;
    predictBtn.innerHTML = '<span class="loading">Testando...</span>';
    responsePayload.textContent = 'Processando requisição...';
    responseStatus.textContent = '---';
    responseTime.textContent = '---';
    
    try {
        const url = `${API_BASE}/api/models/flow/predict`;
        
        // Pegar payload do textarea editável
        let payload;
        try {
            const payloadText = document.getElementById('requestPayload').value;
            payload = JSON.parse(payloadText);
        } catch (e) {
            throw new Error(`JSON inválido: ${e.message}`);
        }
        
        const startTime = performance.now();
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const endTime = performance.now();
        const responseTimeMs = Math.round(endTime - startTime);
        
        // Atualizar interface
        responseStatus.textContent = response.status;
        responseTime.textContent = `${responseTimeMs}ms`;
        
        if (!response.ok) {
            const errorText = await response.text();
            responsePayload.textContent = errorText;
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        responsePayload.textContent = JSON.stringify(result, null, 2);
        
    } catch (error) {
        console.error('Erro na previsão:', error);
        responseStatus.textContent = 'ERROR';
        responseTime.textContent = '---';
        responsePayload.textContent = `Erro: ${error.message}`;
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerHTML = '🚀 Testar API';
    }
}

// Exibir resultado da previsão
function displayPredictionResult(result) {
    const output = document.getElementById('predictionOutput');
    const monthNames = [
        'Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
        'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro'
    ];
    
    output.innerHTML = `
        <div class="prediction-card">
            <div class="prediction-value">${result.predicted_flow} m³/s</div>
            <div class="prediction-range">
                Intervalo: ${result.lower_bound} - ${result.upper_bound} m³/s
            </div>
            <div style="color: #64748b; margin-bottom: 1rem;">
                Previsão para ${monthNames[result.month - 1]}/${result.year}
            </div>
            <div class="prediction-details">
                <div class="detail-item">
                    <div class="detail-value">±${result.uncertainty}</div>
                    <div class="detail-label">Incerteza</div>
                </div>
                <div class="detail-item">
                    <div class="detail-value">${(result.confidence_level * 100).toFixed(0)}%</div>
                    <div class="detail-label">Confiança</div>
                </div>
                <div class="detail-item">
                    <div class="detail-value">${result.model_components.pred_A}</div>
                    <div class="detail-label">Modelo A</div>
                </div>
                <div class="detail-item">
                    <div class="detail-value">${result.model_components.pred_B}</div>
                    <div class="detail-label">Modelo B</div>
                </div>
            </div>
        </div>
    `;
}

// Atualizar demonstração da API
function updateAPIDemo(method, url, payload, response, responseTime) {
    // Atualizar requisição
    const requestInfo = document.querySelector('.request-info');
    requestInfo.innerHTML = `
        <div class="method" style="background: ${method === 'GET' ? '#10b981' : '#f59e0b'}">${method}</div>
        <div class="url">${url.replace(window.location.origin, '')}</div>
    `;
    
    const requestPayload = document.getElementById('requestPayload');
    if (payload) {
        requestPayload.textContent = JSON.stringify(payload, null, 2);
    } else {
        requestPayload.textContent = 'Sem corpo da requisição (GET)';
    }
    
    // Atualizar resposta
    document.getElementById('responseStatus').textContent = '200';
    document.getElementById('responseTime').textContent = `${responseTime}ms`;
    document.getElementById('responsePayload').textContent = JSON.stringify(response, null, 2);
}