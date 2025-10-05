# 🌐 Sistema Web - Predição de Dengue

## 🎯 Sobre

Interface web profissional para predição de casos de dengue utilizando 3 modelos de machine learning simultaneamente.

## 🚀 Execução

```bash
python app.py
```
Acesse: http://localhost:5000

## ✨ Funcionalidades

### 🔧 Dois Modos de Operação

#### 📱 Modo Básico
- **Interface simplificada** para usuários iniciantes
- **Auto-preenchimento** de dados históricos por estado
- **Sliders interativos** para temperatura e precipitação
- **Seleção de época** (seca/chuvosa/transição)

#### ⚙️ Modo Avançado
- **Controle total** de todos os parâmetros
- **Campos customizáveis** para especialistas
- **Entrada manual** de dados específicos

### 🤖 Três Modelos Simultâneos

1. **🚀 XGBoost Super Otimizado**
   - Maior precisão com 57 features
   - Confiança: HIGH

2. **⭐ XGBoost Otimizado**
   - Boa performance com 40 features
   - Confiança: HIGH

3. **🌳 Random Forest Baseline**
   - Modelo mais conservador
   - Confiança: MEDIUM

### 📊 Análise Consolidada

- **Comparação visual** dos 3 modelos
- **Intervalos de confiança** para cada predição
- **Estatísticas consolidadas** (média, mínimo, máximo)
- **Nível de risco** baseado na predição média
- **Identificação do modelo** mais conservador

## 🛠️ Tecnologias

- **Backend**: Flask (Python)
- **Frontend**: Bootstrap 5, JavaScript
- **Modelos**: XGBoost, Random Forest (arquivos .pkl)
- **Dados**: Auto-preenchimento com dados reais dos estados

## 📁 Estrutura

```
web/
├── app.py              # Servidor Flask principal
├── templates/
│   └── index.html      # Interface principal
├── static/
│   └── style.css       # Estilos customizados
├── run.bat            # Script Windows
└── run.sh             # Script Linux/Mac
```

## 🎯 Para Desenvolvimento

### Adicionar Novo Estado
Edite `complete_basic_mode_data()` em `app.py`

### Modificar Modelos
Atualize `predict_with_real_models()` em `app.py`

### Customizar Interface
Edite `templates/index.html` e `static/style.css`

---

**Sistema pronto para apresentações e demonstrações!**