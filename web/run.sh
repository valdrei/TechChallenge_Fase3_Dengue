#!/bin/bash
# Script para executar o sistema principal

echo "🚀 INICIANDO SISTEMA MULTI-MODELO DE PREDIÇÃO DE DENGUE"
echo "=" * 60

echo "📍 Verificando localização..."
if [ ! -f "app.py" ]; then
    echo "❌ Execute este script dentro da pasta /web/"
    echo "   cd web && ./run.sh"
    exit 1
fi

echo "🔧 Verificando dependências..."
python -c "import flask, pandas, numpy, scikit_learn, xgboost" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ Instalando dependências..."
    pip install flask pandas numpy scikit-learn xgboost matplotlib seaborn
fi

echo "🌐 Iniciando servidor..."
echo "📋 Sistema Multi-Modelo:"
echo "   🔵 Random Forest (Notebook 4)"
echo "   🟢 XGBoost Otimizado (Notebook 5)"
echo "   🟡 Ensemble (Média)"
echo ""
echo "🔗 Acesse: http://localhost:5000"
echo "🛑 Para parar: Ctrl+C"
echo ""

python app.py