"""
Módulo de utilidades para o projeto de Predição de Dengue
Tech Challenge - Fase 3 - FIAP

Contém funções auxiliares para processamento de dados,
criação de features e avaliação de modelos.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def calculate_metrics(y_true: np.array, y_pred: np.array) -> Dict[str, float]:
    """
    Calcula métricas de avaliação para modelos de regressão.

    Args:
        y_true: Valores reais
        y_pred: Valores preditos

    Returns:
        Dict com métricas: r2, rmse, mae, mape
    """
    metrics = {}

    # R²
    metrics['r2'] = r2_score(y_true, y_pred)

    # RMSE
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAE
    metrics['mae'] = mean_absolute_error(y_true, y_pred)

    # MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    metrics['mape'] = mape

    return metrics


def create_lag_features(df: pd.DataFrame,
                       target_col: str,
                       group_col: str,
                       lags: List[int]) -> pd.DataFrame:
    """
    Cria features de lag (valores passados) agrupadas.

    Args:
        df: DataFrame com os dados
        target_col: Nome da coluna target
        group_col: Nome da coluna para agrupamento (ex: COD_UF)
        lags: Lista com números de períodos para lag

    Returns:
        DataFrame com features de lag adicionadas
    """
    df_copy = df.copy()

    for lag in lags:
        col_name = f'{target_col}_lag_{lag}'
        df_copy[col_name] = df_copy.groupby(group_col)[target_col].shift(lag)

    return df_copy


def create_rolling_features(df: pd.DataFrame,
                          columns: List[str],
                          group_col: str,
                          windows: List[int]) -> pd.DataFrame:
    """
    Cria features de média móvel (rolling) agrupadas.

    Args:
        df: DataFrame com os dados
        columns: Lista de colunas para aplicar rolling
        group_col: Nome da coluna para agrupamento
        windows: Lista com tamanhos das janelas

    Returns:
        DataFrame com features rolling adicionadas
    """
    df_copy = df.copy()

    for col in columns:
        for window in windows:
            col_name = f'{col}_rolling_{window}'
            df_copy[col_name] = (df_copy.groupby(group_col)[col]
                               .rolling(window=window, min_periods=1)
                               .mean()
                               .reset_index(0, drop=True))

    return df_copy


def create_temporal_features(df: pd.DataFrame,
                           date_col: str = None,
                           month_col: str = 'Mês') -> pd.DataFrame:
    """
    Cria features temporais sazonais.

    Args:
        df: DataFrame com os dados
        date_col: Nome da coluna de data (opcional)
        month_col: Nome da coluna de mês

    Returns:
        DataFrame com features temporais adicionadas
    """
    df_copy = df.copy()

    # Features sazonais baseadas no mês
    if month_col in df_copy.columns:
        df_copy['mes_sin'] = np.sin(2 * np.pi * df_copy[month_col] / 12)
        df_copy['mes_cos'] = np.cos(2 * np.pi * df_copy[month_col] / 12)

        # Trimestre
        df_copy['trimestre'] = ((df_copy[month_col] - 1) // 3) + 1

        # Estação do ano (Brasil - Hemisfério Sul)
        df_copy['estacao'] = df_copy[month_col].map({
            12: 'Verão', 1: 'Verão', 2: 'Verão',
            3: 'Outono', 4: 'Outono', 5: 'Outono',
            6: 'Inverno', 7: 'Inverno', 8: 'Inverno',
            9: 'Primavera', 10: 'Primavera', 11: 'Primavera'
        })

    return df_copy


def split_time_series(df: pd.DataFrame,
                     year_col: str = 'Ano',
                     train_end: int = 2021,
                     val_end: int = 2023) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide dados em treino, validação e teste respeitando ordem temporal.

    Args:
        df: DataFrame com os dados
        year_col: Nome da coluna de ano
        train_end: Último ano do conjunto de treino
        val_end: Último ano do conjunto de validação

    Returns:
        Tupla (train_df, val_df, test_df)
    """
    train_df = df[df[year_col] <= train_end].copy()
    val_df = df[(df[year_col] > train_end) & (df[year_col] <= val_end)].copy()
    test_df = df[df[year_col] > val_end].copy()

    return train_df, val_df, test_df


def print_metrics_summary(metrics: Dict[str, float], model_name: str = "Modelo"):
    """
    Imprime resumo formatado das métricas.

    Args:
        metrics: Dicionário com as métricas
        model_name: Nome do modelo
    """
    print(f"\n🏆 PERFORMANCE - {model_name}")
    print("=" * 50)
    print(f"📊 R² Score:     {metrics['r2']:.4f}")
    print(f"📏 RMSE:         {metrics['rmse']:.2f}")
    print(f"📐 MAE:          {metrics['mae']:.2f}")
    print(f"📈 MAPE:         {metrics['mape']:.2f}%")

    # Interpretação do R²
    r2 = metrics['r2']
    if r2 >= 0.8:
        print("✅ Performance: EXCELENTE (>80% da variância explicada)")
    elif r2 >= 0.7:
        print("✅ Performance: BOA (útil para predições)")
    elif r2 >= 0.6:
        print("⚠️  Performance: MODERADA (baseline aceitável)")
    else:
        print("❌ Performance: FRACA (necessita melhorias)")