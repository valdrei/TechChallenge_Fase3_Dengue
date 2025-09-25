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
    from predict import predict_dengue_cases, validate_input_data
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
    Endpoint para receber dados e retornar predições.

    Recebe dados do formulário via POST e retorna a predição em JSON.
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

        # Realiza a predição
        # Nota: Como não temos o modelo treinado ainda, vamos simular
        # Na implementação real, descomente a linha abaixo:
        # result = predict_dengue_cases(data)

        # SIMULAÇÃO (remover quando o modelo estiver pronto)
        result = simulate_prediction(data)

        # Formata o resultado para a interface
        formatted_result = {
            'predicao': result['prediction'],
            'minimo': result['confidence_lower'],
            'maximo': result['confidence_upper'],
            'confianca': 85,
            'modelo': 'Simulação (Random Forest + XGBoost)',
            'dados_usados': len(data)
        }

        return jsonify(formatted_result)

    except Exception as e:
        return jsonify({
            'error': f'Erro interno do servidor: {str(e)}'
        }), 500


def simulate_prediction(data):
    """
    Simula uma predição enquanto o modelo não está disponível.
    Remove esta função quando o modelo real estiver treinado.
    """
    import random

    # Simula predição baseada em alguns fatores dos dados
    base_cases = 1000

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

    # Calcula predição simulada
    prediction = base_cases * climate_factor * rain_factor * pop_factor * seasonal_factor

    # Adiciona alguma variabilidade
    prediction *= random.uniform(0.8, 1.2)

    # Garante que seja positivo
    prediction = max(0, prediction)

    # Calcula intervalos de confiança
    std_error = prediction * 0.15
    confidence_lower = max(0, prediction - 1.96 * std_error)
    confidence_upper = prediction + 1.96 * std_error

    return {
        'prediction': prediction,
        'confidence_lower': confidence_lower,
        'confidence_upper': confidence_upper,
        'model_confidence': 'medium' if prediction < 2000 else 'high',
        'estado_nome': ESTADOS_BRASIL.get(data.get('COD_UF', ''), 'Desconhecido'),
        'simulado': True  # Indica que é simulação
    }


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
    Informações sobre a API.
    """
    return jsonify({
        'name': 'API de Predição de Dengue',
        'description': 'Sistema ML para predição de casos de dengue no Brasil',
        'version': '1.0.0',
        'tech_challenge': 'FIAP - Fase 3',
        'algorithms': ['Random Forest', 'XGBoost', 'LightGBM'],
        'endpoints': {
            '/': 'Interface web principal',
            '/predict': 'POST - Realizar predição',
            '/health': 'Verificação de saúde',
            '/api/info': 'Informações da API'
        },
        'required_fields': [
            'COD_UF', 'Ano', 'Mês', 'População', 'PIB_per_capita',
            'Precipitacao_mm', 'Temperatura_media_C', 'Umidade_relativa_%',
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