"""
Servidor Web Flask para o Sistema de Predição de Dengue
Tech Challenge - Fase 3 - FIAP

Este arquivo cria uma interface web simples que permite aos usuários
inserir dados e obter predições de casos de dengue através do modelo treinado.
"""

from flask import Flask, render_template, request, jsonify
import sys
import os

# Adiciona o diretório src ao path para importar as funções
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from predict import predict_dengue_cases, predict_dengue_cases_multi_model, validate_input_data
except ImportError:
    print("Erro: Não foi possível importar as funções de predição.")
    print("Certifique-se de que o arquivo src/predict.py existe e está correto.")
    sys.exit(1)

# Inicializa a aplicação Flask
app = Flask(__name__)

# Estados brasileiros com nomes completos
ESTADOS_BRASIL = {
    'AC': 'Acre',
    'AL': 'Alagoas',
    'AP': 'Amapá',
    'AM': 'Amazonas',
    'BA': 'Bahia',
    'CE': 'Ceará',
    'DF': 'Distrito Federal',
    'ES': 'Espírito Santo',
    'GO': 'Goiás',
    'MA': 'Maranhão',
    'MT': 'Mato Grosso',
    'MS': 'Mato Grosso do Sul',
    'MG': 'Minas Gerais',
    'PA': 'Pará',
    'PB': 'Paraíba',
    'PR': 'Paraná',
    'PE': 'Pernambuco',
    'PI': 'Piauí',
    'RJ': 'Rio de Janeiro',
    'RN': 'Rio Grande do Norte',
    'RS': 'Rio Grande do Sul',
    'RO': 'Rondônia',
    'RR': 'Roraima',
    'SC': 'Santa Catarina',
    'SP': 'São Paulo',
    'SE': 'Sergipe',
    'TO': 'Tocantins'
}


@app.route('/')
def index():
    """
    Página principal com o formulário de predição.
    """
    return render_template('index.html', estados=ESTADOS_BRASIL)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para receber dados e retornar predições dos 3 modelos.

    Recebe dados do formulário via POST e retorna predições de todos os modelos em JSON.
    """
    try:
        # Obtém dados do formulário da requisição
        data = {}

        # Coleta todos os dados do formulário
        for key, value in request.form.items():
            # Converte valores numéricos
            if key in ['Ano', 'Mês', 'População', 'PIB_per_capita',
                      'Precipitacao_mm', 'Temperatura_media_C', 'Umidade_relativa_%',
                      'Cobertura_ESF_%', 'Internacoes_diarreia_gastroenterite']:
                try:
                    data[key] = float(value) if '.' in value else int(value)
                except (ValueError, TypeError):
                    data[key] = 0  # Valor padrão se conversão falhar
            else:
                data[key] = value

        if not data:
            return jsonify({'error': 'Nenhum dado fornecido'}), 400

        # Valida os dados de entrada
        validation_errors = validate_input_data(data)
        if validation_errors:
            error_messages = []
            for field, error in validation_errors.items():
                error_messages.append(f"{field}: {error}")
            return jsonify({
                'error': 'Dados inválidos: ' + '; '.join(error_messages)
            }), 400

        # Realiza a predição com múltiplos modelos
        try:
            # Tenta usar os modelos reais primeiro
            resultados_modelos = predict_dengue_cases_multi_model(data)
        except Exception as e:
            # Fallback para simulação se modelos não estão disponíveis
            resultados_modelos = simulate_multi_model_prediction(data)

        # Formata o resultado para a interface
        formatted_result = {
            'modelos': {},
            'resumo': {
                'melhor_modelo': '',
                'menor_predicao': float('inf'),
                'maior_predicao': 0,
                'predicao_media': 0
            },
            'dados_usados': len(data),
            'estado_nome': ESTADOS_BRASIL.get(data.get('COD_UF', ''), 'Desconhecido')
        }

        predicoes = []
        for modelo_key, resultado in resultados_modelos.items():
            formatted_result['modelos'][modelo_key] = {
                'nome': resultado['nome'],
                'predicao': round(resultado['predicao']),
                'confianca_min': round(resultado['confianca_min']),
                'confianca_max': round(resultado['confianca_max']),
                'nivel_confianca': resultado['nivel_confianca'],
                'disponivel': resultado['disponivel'],
                'baseline': resultado['baseline']
            }

            predicoes.append(resultado['predicao'])

            # Atualiza resumo
            if resultado['predicao'] < formatted_result['resumo']['menor_predicao']:
                formatted_result['resumo']['menor_predicao'] = resultado['predicao']
                formatted_result['resumo']['melhor_modelo'] = resultado['nome']

            if resultado['predicao'] > formatted_result['resumo']['maior_predicao']:
                formatted_result['resumo']['maior_predicao'] = resultado['predicao']

        # Calcula média
        if predicoes:
            formatted_result['resumo']['predicao_media'] = sum(predicoes) / len(predicoes)

        return jsonify(formatted_result)

    except Exception as e:
        return jsonify({
            'error': f'Erro interno do servidor: {str(e)}'
        }), 500


def simulate_multi_model_prediction(data):
    """
    Simula predições de múltiplos modelos quando os modelos reais não estão disponíveis.
    """
    import random

    # Simulações baseadas nos baselines conhecidos
    modelos_simulados = {
        'notebook_4': {
            'nome': 'Random Forest Otimizado (Notebook 4) - SIMULADO',
            'baseline': 6047,
            'variacao': 0.3
        },
        'notebook_5_rapido': {
            'nome': 'XGBoost Super Otimizado (Notebook 5 RÁPIDO) - SIMULADO',
            'baseline': 661,
            'variacao': 0.2
        },
        'ensemble': {
            'nome': 'Ensemble (Média dos Modelos) - SIMULADO',
            'baseline': 3354,
            'variacao': 0.25
        }
    }

    resultados = {}

    for modelo_key, info in modelos_simulados.items():
        # Simula predição baseada no baseline e dados
        base_prediction = simulate_prediction_baseline(data, info['baseline'])

        # Adiciona variação específica do modelo
        variation = random.uniform(1 - info['variacao'], 1 + info['variacao'])
        predicao = base_prediction * variation

        # Calcula intervalo de confiança
        std_error = predicao * 0.15

        resultados[modelo_key] = {
            'nome': info['nome'],
            'predicao': max(0, predicao),
            'confianca_min': max(0, predicao - 1.96 * std_error),
            'confianca_max': predicao + 1.96 * std_error,
            'nivel_confianca': 'medium',
            'disponivel': False,  # Simulado
            'baseline': info['baseline']
        }

    return resultados


def simulate_prediction_baseline(data, baseline):
    """
    Simula uma predição baseada no baseline e dados de entrada.
    """
    import random

    # Fatores climáticos (temperatura e precipitação aumentam casos)
    climate_factor = (data.get('Temperatura_media_C', 25) / 25) * 1.2
    rain_factor = (data.get('Precipitacao_mm', 100) / 100) * 1.1

    # Fator populacional (estados maiores têm mais casos)
    pop_factor = (data.get('População', 5000000) / 5000000) * 0.8

    # Fator sazonal (meses de verão têm mais casos)
    mes = data.get('Mês', 6)
    if mes in [12, 1, 2, 3]:  # Verão
        seasonal_factor = 1.5
    elif mes in [9, 10, 11]:  # Primavera
        seasonal_factor = 1.2
    else:  # Outono/Inverno
        seasonal_factor = 0.7

    # Calcula predição baseada no baseline
    prediction = baseline * climate_factor * rain_factor * pop_factor * seasonal_factor * 0.3

    # Adiciona alguma variabilidade
    prediction *= random.uniform(0.8, 1.2)

    # Garante que seja positivo
    return max(0, prediction)


@app.route('/health')
def health_check():
    """
    Endpoint de verificação de saúde do servidor.
    """
    return jsonify({
        'status': 'healthy',
        'service': 'Sistema de Predição de Dengue',
        'version': '1.0.0'
    })


@app.route('/api/info')
def api_info():
    """
    Informações sobre a API com suporte a múltiplos modelos.
    """
    return jsonify({
        'name': 'API de Predição de Dengue - Multi-Modelo',
        'description': 'Sistema ML para predição de casos de dengue no Brasil usando 3 modelos',
        'version': '2.0.0',
        'tech_challenge': 'FIAP - Fase 3',
        'modelos_disponíveis': {
            'notebook_4': {
                'nome': 'Random Forest Otimizado (Notebook 4)',
                'algoritmo': 'Random Forest',
                'baseline_mae': 6047,
                'otimização': 'GridSearchCV'
            },
            'notebook_5_rapido': {
                'nome': 'XGBoost Super Otimizado (Notebook 5 RÁPIDO)',
                'algoritmo': 'XGBoost',
                'baseline_mae': 661,
                'otimização': 'RandomizedSearchCV + Feature Engineering Avançado'
            },
            'ensemble': {
                'nome': 'Ensemble (Média dos Modelos)',
                'algoritmo': 'Ensemble',
                'baseline_mae': 3354,
                'otimização': 'Combinação dos melhores modelos'
            }
        },
        'endpoints': {
            '/': 'Interface web principal',
            '/predict': 'POST - Realizar predição com todos os modelos',
            '/health': 'Verificação de saúde',
            '/api/info': 'Informações da API'
        },
        'performance': {
            'melhor_modelo': 'XGBoost Super Otimizado',
            'mae_minimo': 661,
            'speedup': '10-16x mais rápido que versão original',
            'tempo_execucao': '~30 minutos vs 8 horas'
        },
        'required_fields': [
            'COD_UF', 'Ano', 'Mês', 'Temperatura_media_C', 'Precipitacao_mm'
        ],
        'optional_fields': [
            'População', 'PIB_per_capita', 'Umidade_relativa_%',
            'Cobertura_ESF_%', 'Internacoes_diarreia_gastroenterite'
        ]
    })


if __name__ == '__main__':
    # Configurações para desenvolvimento
    print("🚀 Iniciando Sistema de Predição de Dengue...")
    print("📊 Tech Challenge - FIAP - Fase 3")
    print("🌐 Acesse: http://localhost:5000")
    print("🔄 Para parar: Ctrl+C")
    print("-" * 50)

    # Roda o servidor Flask
    app.run(
        debug=True,          # Modo desenvolvimento
        host='0.0.0.0',      # Aceita conexões de qualquer IP
        port=5000,           # Porta padrão
        threaded=True        # Suporte a múltiplas requisições
    )