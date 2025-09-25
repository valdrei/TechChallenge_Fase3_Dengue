# 🌐 Interface Web - Sistema de Predição de Dengue

## 📋 **Como Executar a Interface Web**

### 1. **Instalar Dependências:**
```bash
# No diretório raiz do projeto
pip install flask

# Ou instalar todas as dependências
pip install -r requirements.txt
```

### 2. **Executar o Servidor:**
```bash
# Navegar para a pasta web
cd web

# Executar o servidor Flask
python app.py
```

### 3. **Acessar a Interface:**
- **URL Local:** http://localhost:5000
- **Interface:** Formulário web interativo
- **API Info:** http://localhost:5000/api/info

---

## 🎯 **Funcionalidades da Interface**

### 📊 **Formulário de Predição:**
- **Dados Geográficos:** Seleção de estado brasileiro e período
- **Dados Climáticos:** Temperatura, precipitação, umidade
- **Dados Demográficos:** População, PIB per capita
- **Dados de Saúde:** Cobertura ESF, internações

### 🤖 **Predição Inteligente:**
- **Algoritmos:** Random Forest, XGBoost, LightGBM (simulado até modelo estar pronto)
- **Resultado:** Estimativa de casos + intervalo de confiança
- **Validação:** Verificação automática de dados de entrada
- **Visualização:** Interface amigável com gráficos

### 🔧 **Features Técnicas:**
- **Responsivo:** Funciona em desktop, tablet e mobile
- **Validação:** Campos obrigatórios e ranges apropriados
- **Feedback:** Loading animations e mensagens de erro claras
- **Exemplo:** Botão para preencher dados de São Paulo automaticamente

---

## 🛠️ **Estrutura dos Arquivos Web**

```
web/
├── app.py                 # Servidor Flask principal
├── templates/
│   └── index.html        # Interface HTML principal
└── static/
    └── style.css         # Estilos CSS customizados
```

### 📄 **Arquivos Criados:**

1. **`app.py`** - Servidor Flask com:
   - Rota principal (`/`) para interface
   - API de predição (`/predict`)
   - Endpoints de info (`/api/info`, `/health`)
   - Validação de dados integrada
   - Simulação temporária até modelo estar pronto

2. **`index.html`** - Interface web com:
   - Formulário responsivo com Bootstrap 5
   - Validação JavaScript no frontend
   - Visualização de resultados dinâmica
   - Botão de exemplo para facilitar testes

3. **`style.css`** - Estilos customizados com:
   - Design moderno e profissional
   - Gradientes e animações
   - Responsividade completa
   - Temas cores relacionados à saúde

---

## 🚀 **Como Usar**

### 🔄 **Fluxo de Uso:**
1. **Abrir** http://localhost:5000
2. **Preencher** dados no formulário (ou usar botão "Preencher Exemplo")
3. **Clicar** "Prever Casos de Dengue"
4. **Visualizar** resultado com estimativa e intervalo de confiança

### 📊 **Exemplo de Dados (São Paulo):**
- **Estado:** SP
- **Período:** Março/2025
- **População:** 44.420.459
- **PIB per capita:** R$ 47.945
- **Precipitação:** 150.5 mm
- **Temperatura:** 24.2°C
- **Umidade:** 78.5%
- **Cobertura ESF:** 65.3%
- **Internações:** 2.341

---

## ⚙️ **Configuração para Produção**

### 🔧 **Usando Gunicorn (Linux/Mac):**
```bash
# Instalar gunicorn
pip install gunicorn

# Executar servidor de produção
cd web
gunicorn --bind 0.0.0.0:8000 app:app
```

### 🔧 **Usando Waitress (Windows):**
```bash
# Instalar waitress
pip install waitress

# Executar servidor de produção
cd web
waitress-serve --host=0.0.0.0 --port=8000 app:app
```

### 🌐 **Deploy na Nuvem:**
- **Heroku:** Incluir `Procfile` com `web: gunicorn app:app`
- **Azure App Service:** Compatível com Flask
- **AWS Elastic Beanstalk:** Deploy direto da pasta web/
- **Google Cloud Run:** Containerizar com Docker

---

## 🔍 **Endpoints da API**

| Endpoint | Método | Descrição |
|----------|---------|-----------|
| `/` | GET | Interface web principal |
| `/predict` | POST | API de predição (JSON) |
| `/health` | GET | Status do servidor |
| `/api/info` | GET | Informações da API |

### 📝 **Exemplo de Requisição API:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "COD_UF": "SP",
    "Ano": 2025,
    "Mês": 3,
    "População": 44420459,
    "PIB_per_capita": 47945,
    "Precipitacao_mm": 150.5,
    "Temperatura_media_C": 24.2,
    "Umidade_relativa_%": 78.5,
    "Cobertura_ESF_%": 65.3,
    "Internacoes_diarreia_gastroenterite": 2341
  }'
```

---

## ⚠️ **Notas Importantes**

### 🤖 **Sobre a Simulação:**
- Atualmente usa **simulação** enquanto modelos não estão treinados
- Quando executar os notebooks e treinar os modelos, a função `predict_dengue_cases()` será usada
- Remove a função `simulate_prediction()` quando modelo real estiver pronto

### 🔄 **Integração com Notebooks:**
- Execute primeiro os notebooks para treinar modelos
- Modelos serão salvos na pasta `models/`
- Interface web automaticamente usará o modelo treinado

### 📈 **Melhorias Futuras:**
- Histórico de predições
- Visualizações gráficas (Plotly)
- Comparação entre modelos
- Export de resultados (PDF/Excel)
- Autenticação de usuários

---

**✅ Interface web pronta para demonstração e uso!** 🌐🦟📊