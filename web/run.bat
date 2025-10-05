@echo off
REM Script para executar o sistema principal no Windows

echo 🚀 INICIANDO SISTEMA MULTI-MODELO DE PREDIÇÃO DE DENGUE
echo ============================================================

echo 📍 Verificando localização...
if not exist "app.py" (
    echo ❌ Execute este script dentro da pasta \web\
    echo    cd web ^&^& run.bat
    pause
    exit /b 1
)

echo 🔧 Verificando dependências...
python -c "import flask, pandas, numpy, sklearn, xgboost" >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Instalando dependências...
    pip install flask pandas numpy scikit-learn xgboost matplotlib seaborn
)

echo 🌐 Iniciando servidor...
echo 📋 Sistema Multi-Modelo:
echo    🔵 Random Forest ^(Notebook 4^)
echo    🟢 XGBoost Otimizado ^(Notebook 5^)
echo    🟡 Ensemble ^(Média^)
echo.
echo 🔗 Acesse: http://localhost:5000
echo 🛑 Para parar: Ctrl+C
echo.

python app.py
pause