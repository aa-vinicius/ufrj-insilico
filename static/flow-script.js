// JavaScript para página do modelo de vazão

const API_BASE = window.location.origin;

// Estado da aplicação
let testData = [];
let modelStatus = { is_trained: false };

// Inicialização
document.addEventListener('DOMContentLoaded', async () => {
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
        testData = await response.json();
        
        console.log('Dados de teste carregados:', testData.length, 'registros');
        
        // Calcular métricas
        calculateAndDisplayMetrics();
        
    } catch (error) {
        console.error('Erro ao carregar dados de teste:', error);
        
        // Dados mock para demonstração
        testData = generateMockTestData();
        calculateAndDisplayMetrics();
    }
}

// Gerar dados mock para demonstração
function generateMockTestData() {
    const mockData = [];
    const baseYear = 2020;
    
    for (let year = 2020; year <= 2023; year++) {
        for (let month = 1; month <= 12; month++) {
            const observed = 50 + 40 * Math.sin((month - 1) * Math.PI / 6) + Math.random() * 20;
            const predicted = observed + (Math.random() - 0.5) * 10;
            const uncertainty = 15 + Math.random() * 5;
            
            mockData.push({
                year,
                month,
                observed: Math.max(10, observed),
                predicted: Math.max(10, predicted),
                lower_bound: Math.max(5, predicted - uncertainty),
                upper_bound: predicted + uncertainty,
                obs_min: Math.max(5, observed - 15),
                obs_max: observed + 15
            });
        }
    }
    
    return mockData;
}

// Calcular e exibir métricas
function calculateAndDisplayMetrics() {
    if (testData.length === 0) return;
    
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
        setTimeout(renderTimeSeriesChart, 1000);
        return;
    }
    
    // Preparar dados para o gráfico
    const dates = testData.map(d => `${d.year}-${d.month.toString().padStart(2, '0')}-01`);
    
    const traces = [
        {
            x: dates,
            y: testData.map(d => d.upper_bound),
            type: 'scatter',
            mode: 'lines',
            fill: 'tonexty',
            fillcolor: 'rgba(255, 165, 0, 0.3)',
            line: { color: 'transparent' },
            name: 'Intervalo Superior',
            hoverinfo: 'skip'
        },
        {
            x: dates,
            y: testData.map(d => d.lower_bound),
            type: 'scatter',
            mode: 'lines',
            fill: 'none',
            line: { color: 'transparent' },
            name: 'Intervalo Inferior',
            hoverinfo: 'skip'
        },
        {
            x: dates,
            y: testData.map(d => d.obs_max),
            type: 'scatter',
            mode: 'lines',
            fill: 'tonexty',
            fillcolor: 'rgba(128, 128, 128, 0.3)',
            line: { color: 'transparent' },
            name: 'Obs Max',
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
    document.getElementById('predYear').value = '2023';
    document.getElementById('predMonth').value = '1';
    document.querySelector('input[name="predType"][value="advanced"]').checked = true;
    document.getElementById('advancedFields').style.display = 'block';
    
    // Preencher campos com exemplo
    document.getElementById('tmin_min').value = '15.9';
    document.getElementById('tmax_max').value = '30.8';
    document.getElementById('pr_min').value = '172.5';
    document.getElementById('pr_max').value = '269.7';
    document.getElementById('y_lag1').value = '94.8';
}

// Fazer previsão
async function makePrediction() {
    const predictBtn = document.getElementById('predictBtn');
    const predictionOutput = document.getElementById('predictionOutput');
    
    // Desabilitar botão e mostrar loading
    predictBtn.disabled = true;
    predictBtn.innerHTML = '<span class="loading">Prevendo...</span>';
    predictionOutput.innerHTML = '<div class="loading">Processando previsão...</div>';
    
    try {
        const year = parseInt(document.getElementById('predYear').value);
        const month = parseInt(document.getElementById('predMonth').value);
        const predType = document.querySelector('input[name="predType"]:checked').value;
        
        let url, payload;
        
        if (predType === 'simple') {
            // Previsão simples via GET
            url = `${API_BASE}/api/models/flow/predict?year=${year}&month=${month}`;
            
            const startTime = performance.now();
            const response = await fetch(url);
            const endTime = performance.now();
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            
            displayPredictionResult(result);
            updateAPIDemo('GET', url, null, result, Math.round(endTime - startTime));
            
        } else {
            // Previsão avançada via POST
            url = `${API_BASE}/api/models/flow/predict`;
            payload = {
                year,
                month
            };
            
            // Adicionar campos preenchidos
            const fields = ['tmin_min', 'tmax_max', 'pr_min', 'pr_max', 'y_lag1'];
            fields.forEach(field => {
                const input = document.getElementById(field);
                if (input && input.value) {
                    payload[field] = parseFloat(input.value);
                }
            });
            
            const startTime = performance.now();
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            const endTime = performance.now();
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            
            displayPredictionResult(result);
            updateAPIDemo('POST', url, payload, result, Math.round(endTime - startTime));
        }
        
    } catch (error) {
        console.error('Erro na previsão:', error);
        predictionOutput.innerHTML = `
            <div class="prediction-card" style="background: #fef2f2; border-color: #fecaca;">
                <div style="color: #dc2626; font-weight: 600;">Erro na Previsão</div>
                <div style="color: #7f1d1d; margin-top: 0.5rem;">${error.message}</div>
            </div>
        `;
    } finally {
        // Re-habilitar botão
        predictBtn.disabled = false;
        predictBtn.innerHTML = '🚀 Fazer Previsão';
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