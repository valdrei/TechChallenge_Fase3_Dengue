# Configurações do Projeto de Predição de Dengue
# Tech Challenge - Fase 3 - FIAP

# === DADOS ===
DATA_PATH = "data/"
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"

# Arquivo principal do dataset
DATASET_FILE = "dados_dengue_clima_saneamento_2014_2025.csv"

# === MODELOS ===
MODELS_PATH = "models/"
OUTPUTS_PATH = "outputs/"

# === CONFIGURAÇÕES DE TREINAMENTO ===
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.15

# Divisão temporal para validação (evitar data leakage)
TRAIN_END_YEAR = 2021
VALIDATION_END_YEAR = 2023
TEST_START_YEAR = 2024

# === FEATURES ===
TARGET_COLUMN = "Quantidade de Casos"
ID_COLUMNS = ["COD_UF", "Ano", "Mês"]

# Features temporais a serem criadas
LAG_FEATURES = [1, 2, 3, 6, 12]  # meses
ROLLING_WINDOWS = [3, 6, 12]      # meses

# === MODELOS E HIPERPARÂMETROS ===
MODELS_CONFIG = {
    "random_forest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "xgboost": {
        "n_estimators": [100, 300, 500, 1000],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0]
    },
    "lightgbm": {
        "num_leaves": [31, 50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "feature_fraction": [0.8, 0.9, 1.0],
        "bagging_fraction": [0.8, 0.9, 1.0],
        "n_estimators": [100, 300, 500, 1000]
    }
}

# === MÉTRICAS DE AVALIAÇÃO ===
METRICS = ["r2", "rmse", "mae", "mape"]

# === OUTRAS CONFIGURAÇÕES ===
FIGSIZE = (12, 8)
DPI = 300