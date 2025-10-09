// Configuração da API
const API_BASE_URL = window.location.origin;

// Verificar status da API
async function checkApiStatus() {
    try {
    const response = await fetch(`${API_BASE_URL}/api/data/`);
        const statusDot = document.getElementById('apiStatus');
        const statusText = document.getElementById('statusText');
        
        if (response.ok) {
            statusDot.className = 'status-dot online';
            statusText.textContent = 'API Online';
        } else {
            statusDot.className = 'status-dot offline';
            statusText.textContent = 'API Offline';
        }
    } catch (error) {
        const statusDot = document.getElementById('apiStatus');
        const statusText = document.getElementById('statusText');
        statusDot.className = 'status-dot offline';
        statusText.textContent = 'API Inacessível';
    }
}

// Fazer requisição para endpoint
async function testEndpoint(endpoint, resultId) {
    const resultContainer = document.getElementById(resultId);
    resultContainer.innerHTML = '<div class="loading">🔄 Carregando...</div>';
    
    try {
        const startTime = performance.now();
    const response = await fetch(`${API_BASE_URL}${endpoint}`);
        const endTime = performance.now();
        const responseTime = Math.round(endTime - startTime);
        
        const data = await response.json();
        
        resultContainer.innerHTML = `
            <div class="result-header">
                <span class="status-code ${response.ok ? 'success' : 'error'}">${response.status}</span>
                <span class="response-time">${responseTime}ms</span>
            </div>
            <pre><code class="language-json">${JSON.stringify(data, null, 2)}</code></pre>
        `;
        
        // Aplicar syntax highlighting
        Prism.highlightElement(resultContainer.querySelector('code'));
        
    } catch (error) {
        resultContainer.innerHTML = `
            <div class="result-header">
                <span class="status-code error">ERROR</span>
            </div>
            <div class="error-message">
                <strong>Erro:</strong> ${error.message}
            </div>
        `;
    }
}

// Atualizar parâmetros do playground
function updatePlaygroundParams() {
    const select = document.getElementById('endpoint-select');
    const paramsContainer = document.getElementById('params-container');
    const endpoint = select.value;
    
    let paramsHtml = '';
    
    // Definir parâmetros disponíveis para cada endpoint
    if (endpoint === '/api/data/' || endpoint === '/api/models/') {
        paramsHtml = `
            <div class="form-group">
                <label for="geometry-type">Tipo de Geometria:</label>
                <select id="geometry-type">
                    <option value="">Todos</option>
                    <option value="Point">Point</option>
                    <option value="Polygon">Polygon</option>
                </select>
            </div>
        `;
        
    if (endpoint === '/api/data/') {
            paramsHtml += `
                <div class="form-group">
                    <label for="data-type">Tipo de Dado:</label>
                    <select id="data-type">
                        <option value="">Todos</option>
                        <option value="temperature">Temperature</option>
                        <option value="humidity">Humidity</option>
                        <option value="wind_speed">Wind Speed</option>
                        <option value="precipitation">Precipitation</option>
                        <option value="pressure">Pressure</option>
                    </select>
                </div>
            `;
        }
        
    if (endpoint === '/api/models/') {
            paramsHtml += `
                <div class="form-group">
                    <label for="model-name">Nome do Modelo:</label>
                    <select id="model-name">
                        <option value="">Todos</option>
                        <option value="rain">Rain Prediction</option>
                        <option value="temperature">Temperature Forecast</option>
                        <option value="wind">Wind Pattern</option>
                        <option value="weather">Severe Weather</option>
                    </select>
                </div>
            `;
        }
    } else {
        paramsHtml = '<p class="no-params">Este endpoint não possui parâmetros configuráveis.</p>';
    }
    
    paramsContainer.innerHTML = paramsHtml;
}

// Testar endpoint do playground
async function testPlaygroundEndpoint() {
    const select = document.getElementById('endpoint-select');
    let endpoint = select.value;
    const params = new URLSearchParams();
    
    // Construir query string com parâmetros
    if (endpoint === '/api/data/' || endpoint === '/api/models/') {
        const geometryType = document.getElementById('geometry-type')?.value;
        if (geometryType) params.append('geometry_type', geometryType);
        
    if (endpoint === '/api/data/') {
            const dataType = document.getElementById('data-type')?.value;
            if (dataType) params.append('data_type', dataType);
        }
        
    if (endpoint === '/api/models/') {
            const modelName = document.getElementById('model-name')?.value;
            if (modelName) params.append('model_name', modelName);
        }
    }
    
    const queryString = params.toString();
    if (queryString) {
        endpoint += '?' + queryString;
    }
    
    await testEndpoint(endpoint, 'playground-result');
}

// Navegação smooth scroll
function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Remover classe active de todos os links
            navLinks.forEach(l => l.classList.remove('active'));
            
            // Adicionar classe active ao link clicado
            link.classList.add('active');
            
            // Fazer scroll suave para a seção
            const targetId = link.getAttribute('href').substring(1);
            const targetSection = document.getElementById(targetId);
            
            if (targetSection) {
                targetSection.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Intersection Observer para destacar seção ativa na navegação
function setupSectionObserver() {
    const sections = document.querySelectorAll('.section');
    const navLinks = document.querySelectorAll('.nav-link');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const sectionId = entry.target.id;
                
                // Remover classe active de todos os links
                navLinks.forEach(link => link.classList.remove('active'));
                
                // Adicionar classe active ao link correspondente
                const activeLink = document.querySelector(`[href="#${sectionId}"]`);
                if (activeLink) {
                    activeLink.classList.add('active');
                }
            }
        });
    }, {
        threshold: 0.3,
        rootMargin: '-100px 0px -50% 0px'
    });
    
    sections.forEach(section => {
        observer.observe(section);
    });
}

// Função para copiar JSON para clipboard
function copyToClipboard(text, button) {
    navigator.clipboard.writeText(text).then(() => {
        const originalText = button.textContent;
        button.textContent = '✅ Copiado!';
        button.style.background = '#28a745';
        
        setTimeout(() => {
            button.textContent = originalText;
            button.style.background = '';
        }, 2000);
    });
}

// Adicionar botões de cópia aos resultados
function addCopyButtons() {
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('btn-test')) {
            // Aguardar o resultado ser carregado
            setTimeout(() => {
                const resultContainer = e.target.nextElementSibling;
                const codeElement = resultContainer.querySelector('code');
                
                if (codeElement && !resultContainer.querySelector('.copy-btn')) {
                    const copyBtn = document.createElement('button');
                    copyBtn.className = 'copy-btn';
                    copyBtn.textContent = '📋 Copiar';
                    copyBtn.onclick = () => copyToClipboard(codeElement.textContent, copyBtn);
                    
                    const resultHeader = resultContainer.querySelector('.result-header');
                    if (resultHeader) {
                        resultHeader.appendChild(copyBtn);
                    }
                }
            }, 1000);
        }
    });
}

// Inicialização
document.addEventListener('DOMContentLoaded', () => {
    checkApiStatus();
    updatePlaygroundParams();
    setupNavigation();
    setupSectionObserver();
    addCopyButtons();
    
    // Verificar status da API periodicamente
    setInterval(checkApiStatus, 30000); // A cada 30 segundos
});