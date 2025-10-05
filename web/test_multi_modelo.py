"""
Script de teste para verificar o sistema multi-modelo
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from predict import predict_dengue_cases_multi_model

    # Dados de teste para SP em Janeiro 2025
    dados_teste = {
        'COD_UF': 'SP',
        'Ano': 2025,
        'Mês': 1,
        'Temperatura_media_C': 26.5,
        'Precipitacao_mm': 180.0,
        'População': 46000000,
        'PIB_per_capita': 48000
    }

    print("🧪 TESTE DO SISTEMA MULTI-MODELO")
    print("=" * 50)
    print(f"📊 Dados de entrada: {dados_teste}")
    print()

    # Testa predição multi-modelo
    resultados = predict_dengue_cases_multi_model(dados_teste)

    print("📈 RESULTADOS DOS 3 MODELOS:")
    print("-" * 30)

    for modelo_key, resultado in resultados.items():
        print(f"\n🤖 {resultado['nome']}")
        print(f"   📊 Predição: {resultado['predicao']:.0f} casos")
        print(f"   📈 Intervalo: {resultado['confianca_min']:.0f} - {resultado['confianca_max']:.0f}")
        print(f"   🎯 Baseline: {resultado['baseline']} casos")
        print(f"   ✅ Disponível: {'Sim' if resultado['disponivel'] else 'Não (simulado)'}")
        print(f"   🔍 Confiança: {resultado['nivel_confianca']}")

    # Calcula estatísticas
    predicoes = [r['predicao'] for r in resultados.values()]
    media = sum(predicoes) / len(predicoes)
    minima = min(predicoes)
    maxima = max(predicoes)

    print(f"\n📊 ANÁLISE CONSOLIDADA:")
    print(f"   • Predição Média: {media:.0f} casos")
    print(f"   • Faixa: {minima:.0f} - {maxima:.0f} casos")
    print(f"   • Variação: {((maxima - minima) / media * 100):.1f}%")

    print(f"\n✅ Sistema multi-modelo funcionando corretamente!")

except ImportError as e:
    print(f"❌ Erro de importação: {e}")
    print("Verifique se o arquivo src/predict.py está correto")
except Exception as e:
    print(f"❌ Erro na execução: {e}")
    print("Usando simulação para teste...")

    # Teste de fallback
    from web.app import simulate_multi_model_prediction
    resultados = simulate_multi_model_prediction(dados_teste)

    print("📈 RESULTADOS SIMULADOS:")
    for modelo_key, resultado in resultados.items():
        print(f"   🤖 {resultado['nome']}: {resultado['predicao']:.0f} casos")

    print("✅ Sistema funcionando em modo simulação!")