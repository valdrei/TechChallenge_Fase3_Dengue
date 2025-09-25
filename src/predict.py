"""
Funções de predição prontas para deploy
Tech Challenge - Fase 3 - FIAP
"""

import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, Union
import warnings
warnings.filterwarnings('ignore')


def load_model(model_path: str):
    """
    Carrega modelo serializado.

    Args:
        model_path: Caminho para o arquivo .pkl do modelo

    Returns:
        Modelo carregado
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def predict_dengue_cases(input_data: Union[Dict[str, Any], pd.DataFrame],
                        model_path: str = "models/modelo_campeao.pkl") -> Dict[str, float]:
    """
    Função principal de predição para casos de dengue.

    Args:
        input_data: Dicionário ou DataFrame com as features necessárias
        model_path: Caminho para o modelo treinado

    Returns:
        Dicionário com predição e intervalos de confiança
    """

    # Carrega o modelo
    model = load_model(model_path)

    # Converte para DataFrame se necessário
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    # Features necessárias (exemplo - ajustar conforme modelo final)
    required_features = [
        'COD_UF', 'Ano', 'Mês', 'População', 'PIB_per_capita',
        'Precipitacao_mm', 'Temperatura_media_C', 'Umidade_relativa_%',
        'Cobertura_ESF_%', 'Internacoes_diarreia_gastroenterite',
        # Features engineered serão criadas automaticamente
    ]

    # Cria features necessárias se não existirem
    df = create_prediction_features(df)

    # Seleciona apenas as features do modelo
    feature_cols = [col for col in required_features if col in df.columns]
    X = df[feature_cols]

    # Faz a predição
    prediction = model.predict(X)[0]

    # Calcula intervalo de confiança (aproximação)
    # Em produção, usar modelos com incerteza nativa
    std_error = prediction * 0.15  # Aproximação de 15% de erro

    result = {
        'prediction': max(0, prediction),  # Dengue não pode ser negativa
        'confidence_lower': max(0, prediction - 1.96 * std_error),
        'confidence_upper': prediction + 1.96 * std_error,
        'model_confidence': 'medium' if abs(prediction) < 1000 else 'high'
    }

    return result


def create_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features temporais necessárias para predição.

    Args:
        df: DataFrame com dados básicos

    Returns:
        DataFrame com features adicionadas
    """
    df_copy = df.copy()

    # Features sazonais
    if 'Mês' in df_copy.columns:
        df_copy['mes_sin'] = np.sin(2 * np.pi * df_copy['Mês'] / 12)
        df_copy['mes_cos'] = np.cos(2 * np.pi * df_copy['Mês'] / 12)
        df_copy['trimestre'] = ((df_copy['Mês'] - 1) // 3) + 1

    # Features de interação (exemplos)
    if all(col in df_copy.columns for col in ['Temperatura_media_C', 'Umidade_relativa_%']):
        df_copy['temp_x_umidade'] = df_copy['Temperatura_media_C'] * df_copy['Umidade_relativa_%']

    if all(col in df_copy.columns for col in ['Precipitacao_mm', 'Temperatura_media_C']):
        df_copy['precip_x_temp'] = df_copy['Precipitacao_mm'] * df_copy['Temperatura_media_C']

    return df_copy


def validate_input_data(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Valida dados de entrada para predição.

    Args:
        data: Dicionário com os dados de entrada

    Returns:
        Dicionário com erros encontrados (vazio se tudo OK)
    """
    errors = {}

    # Validações básicas
    required_fields = ['COD_UF', 'Ano', 'Mês']
    for field in required_fields:
        if field not in data:
            errors[field] = f"Campo obrigatório '{field}' não fornecido"

    # Validações de range
    if 'Ano' in data and (data['Ano'] < 2014 or data['Ano'] > 2030):
        errors['Ano'] = "Ano deve estar entre 2014 e 2030"

    if 'Mês' in data and (data['Mês'] < 1 or data['Mês'] > 12):
        errors['Mês'] = "Mês deve estar entre 1 e 12"

    if 'Temperatura_media_C' in data and (data['Temperatura_media_C'] < -10 or data['Temperatura_media_C'] > 50):
        errors['Temperatura_media_C'] = "Temperatura deve estar entre -10°C e 50°C"

    return errors


# Exemplo de uso da API
def api_example():
    """
    Exemplo de como usar a função de predição.
    """

    # Dados de exemplo para São Paulo em março de 2025
    input_example = {
        'COD_UF': 'SP',
        'Ano': 2025,
        'Mês': 3,
        'População': 44420459,
        'PIB_per_capita': 47945,
        'Precipitacao_mm': 150.5,
        'Temperatura_media_C': 24.2,
        'Umidade_relativa_%': 78.5,
        'Cobertura_ESF_%': 65.3,
        'Internacoes_diarreia_gastroenterite': 2341
    }

    # Valida os dados
    errors = validate_input_data(input_example)
    if errors:
        print("Erros encontrados:", errors)
        return None

    # Faz a predição
    try:
        result = predict_dengue_cases(input_example)
        print("Predição para dengue em SP (Mar/2025):")
        print(f"Casos estimados: {result['prediction']:.0f}")
        print(f"Intervalo confiança: {result['confidence_lower']:.0f} - {result['confidence_upper']:.0f}")
        return result
    except Exception as e:
        print(f"Erro na predição: {e}")
        return None


if __name__ == "__main__":
    # Executa exemplo
    api_example()