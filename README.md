# 🦟 Sistema de Predição de Dengue
**Tech Challenge - Fase 3 - FIAP**

## 🎯 Sobre o Projeto

Sistema completo de machine learning para **predição de casos de dengue** no Brasil, demonstrando a evolução gradual dos modelos desde baseline até super otimizado, com interface web profissional.

### 🏆 Resultados Alcançados
- **3 Modelos Treinados**: Random Forest + XGBoost Otimizado + XGBoost Super
- **Cobertura Nacional**: Todos os 27 estados brasileiros (2014-2025)
- **Interface Web**: 2 modos (Básico e Avançado) para diferentes perfis de usuário
- **Deploy Ativo**: Sistema funcionando em produção no Render.com

---

## 🚀 Como Usar

### 🌐 **Acesso Online (Recomendado)**
Acesse o sistema já em funcionamento:
```
https://techchallenge-fase3-dengue.onrender.com/
```

### 💻 **Execução Local**
```bash
# 1. Clone o repositório
git clone https://github.com/valdrei/TechChallenge_Fase3_Dengue.git
cd TechChallenge_Fase3_Dengue

# 2. Instale dependências
pip install -r requirements.txt

# 3. Execute o sistema web
cd web
python app.py

# 4. Acesse: http://localhost:5000
```

---

## 📊 Estrutura do Projeto

```
├── notebooks/           # Evolução dos modelos ML
│   ├── 01_exploracao_dados.ipynb
│   ├── 02_engenharia_features.ipynb
│   ├── 03_modelo_random_forest.ipynb
│   ├── 04_modelo_xgboost.ipynb
│   ├── 05_modelo_lightgbm.ipynb
│   └── 06_comparacao_final_ensemble.ipynb
├── notebooks_auxiliares/ # Experimentos séries temporais
│   ├── construção_dataset_dengue_clima_saneamento.ipynb
│   ├── construcao_dataset_serie_temporal.ipynb
│   └── modelos_serie_temporais_SKTime.ipynb
├── web/                 # Sistema web Flask
│   ├── app.py          # Backend com 3 modelos
│   ├── templates/      # Interface HTML
│   └── static/         # CSS e recursos
├── models/             # Modelos treinados (.pkl)
├── data/               # Dados originais
└── docs/               # Documentação
```

---

## 🎮 Interface do Sistema

### 🔹 **Modo Básico**
- **Para**: Gestores públicos, usuários gerais
- **Campos**: Estado, Mês, Ano, Temperatura, Precipitação
- **Auto-completa**: População, PIB e outros dados por estado

### 🔹 **Modo Avançado**
- **Para**: Epidemiologistas, pesquisadores
- **Campos**: Todos os parâmetros disponíveis
- **Controle Total**: PIB, umidade, cobertura de saúde, etc.
- **Predições 50% maiores**: Maior precisão com mais dados

### � **Resultado Multi-Modelo**
Cada predição retorna:
- **3 modelos simultâneos** com intervalos de confiança
- **Melhor modelo** identificado automaticamente
- **Visualização comparativa** clara e intuitiva

---

## 🛠️ Tecnologias Utilizadas

- **Machine Learning**: XGBoost, Random Forest, LightGBM
- **Backend**: Flask, Python 3.9+
- **Frontend**: HTML5, CSS3, JavaScript
- **Deploy**: Render.com, Gunicorn
- **Dados**: Pandas, NumPy, Scikit-learn

---

## 📋 Notebooks - Evolução dos Modelos

| Notebook | Foco Principal | Resultado |
|----------|----------------|-----------|
| **01** | Análise exploratória | Entendimento dos dados |
| **02** | Engenharia de features | Preparação avançada |
| **03** | Random Forest | Modelo baseline |
| **04** | XGBoost Básico | Otimização inicial |
| **05** | LightGBM | Modelo alternativo |
| **06** | Ensemble Final | Comparação e ensemble |

### 📊 **Notebooks Auxiliares - Séries Temporais**

Além dos notebooks principais, o projeto inclui experimentos com abordagens de **séries temporais**:

```
notebooks_auxiliares/
├── construção_dataset_dengue_clima_saneamento.ipynb    # Coleta e integração de dados
├── construcao_dataset_serie_temporal.ipynb             # Estruturação temporal
└── modelos_serie_temporais_SKTime.ipynb                # Modelos ARIMA, Prophet, etc.
```

**🎯 Objetivo**: Explorar abordagens clássicas de séries temporais (ARIMA, Prophet, Exponential Smoothing) como **alternativa aos modelos de ML tradicional**.

**📈 Resultado**: Os modelos de ML (XGBoost/Random Forest) demonstraram **melhor performance** que as abordagens tradicionais de séries temporais para este problema específico, validando a escolha metodológica do projeto.

---

## 🚀 Deploy e Produção

O sistema está configurado para deploy automático:

- **Procfile**: Configuração Render/Heroku
- **requirements.txt**: Dependências Python
- **render.yaml**: Configuração de infraestrutura
- **Gunicorn**: Servidor WSGI para produção

### 🔄 CI/CD Automático
Qualquer push para `main` dispara deploy automático no Render.

---

## 🎓 Contexto Acadêmico

**TechChallenge Fase 3 - FIAP**
- **Objetivo**: Sistema ML completo end-to-end
- **Diferenciais**: Interface web + Multi-modelo + Deploy
- **Aplicação Real**: Ferramenta para gestão de saúde pública

---

## 📞 Suporte

- **Issues**: Use o GitHub Issues para reportar problemas
- **Documentação**: Veja `/docs` para guias detalhados
- **Deploy**: Consulte `DEPLOY_RENDER.md` para instruções

---

**🏆 Resultado**: Sistema profissional de predição de dengue com interface intuitiva, multi-modelo e deploy em produção, demonstrando aplicação prática de ML na saúde pública brasileira!
