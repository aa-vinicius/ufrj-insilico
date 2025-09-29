#!/bin/bash

# Script para ativar o ambiente virtual e executar comandos de desenvolvimento

# Cores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 Ativando ambiente virtual...${NC}"

# Verificar se o ambiente virtual existe
if [ ! -d "venv" ]; then
    echo -e "${BLUE}📦 Criando ambiente virtual...${NC}"
    python3 -m venv venv
fi

# Ativar ambiente virtual
source venv/bin/activate

echo -e "${GREEN}✅ Ambiente virtual ativado!${NC}"

# Verificar se as dependências estão instaladas
if [ ! -f "venv/pyvenv.cfg" ] || [ ! -d "venv/lib/python*/site-packages/fastapi" ]; then
    echo -e "${BLUE}📦 Instalando dependências...${NC}"
    pip install -r requirements.txt
fi

echo -e "${GREEN}🚀 Ambiente pronto para desenvolvimento!${NC}"
echo -e "${BLUE}Comandos disponíveis:${NC}"
echo "  • Executar API: python3 -m uvicorn app.main:app --reload"
echo "  • Executar testes: python3 -m pytest -v"
echo "  • Desativar ambiente: deactivate"
echo ""