# 🚀 Sistema de Predição de Dengue - Multi-Modelo

## 📋 Visão Geral

Sistema web desenvolvido para o **Tech Challenge - Fase 3 - FIAP** que oferece predições de casos de dengue utilizando **3 modelos diferentes de Machine Learning**, permitindo comparação e análise consolidada.

## 🎯 **SISTEMA PRINCIPAL DO PROJETO**

**Este é o sistema OFICIAL e FUNCIONAL do Tech Challenge.**

### 🔄 **Diferença do Notebook 6:**
- **📔 Notebook 6**: Demonstração educativa do processo de deploy
- **🌐 Este Sistema**: Aplicação real e funcional para uso e apresentação

### ✅ **Por que usar este sistema?**
- ✅ **Multi-modelo**: 3 algoritmos simultâneos
- ✅ **Interface profissional**: Design responsivo e intuitivo
- ✅ **Sistema robusto**: Funciona mesmo sem modelos treinados
- ✅ **Performance otimizada**: Resultados em segundos
- ✅ **Pronto para demo**: Ideal para apresentações

## 🤖 Modelos Disponíveis

### 1. 🌲 Random Forest Otimizado (Notebook 4)
- **Algoritmo**: Random Forest com GridSearchCV
- **Performance**: MAE ~6,047 casos
- **Características**: Modelo robusto com otimização exaustiva de hiperparâmetros
- **Tempo de treinamento**: ~45 minutos

### 2. 🚀 XGBoost Super Otimizado (Notebook 5 RÁPIDO)
- **Algoritmo**: XGBoost com RandomizedSearchCV
- **Performance**: MAE ~661 casos (89% melhor que Notebook 4)
- **Características**: Feature engineering avançado, early stopping, execução super otimizada
- **Tempo de treinamento**: ~30 minutos (vs 8 horas da versão original)
- **Speedup**: 10-16x mais rápido

### 3. ⭐ Ensemble (Média dos Modelos)
- **Algoritmo**: Combinação dos modelos anteriores
- **Performance**: MAE ~3,354 casos (média dos modelos)
- **Características**: Reduz variância e melhora estabilidade das predições

## 🎯 Funcionalidades do Sistema Web

### Interface de Usuário
- **Modo Básico**: 5 campos essenciais (Estado, Ano, Mês, Temperatura, Precipitação)
- **Modo Avançado**: Todos os campos disponíveis para especialistas
- **Comparação Visual**: Resultados dos 3 modelos lado a lado
- **Análise Consolidada**: Média e faixa de variação das predições

### Recursos Visuais
- 📊 Cards individuais para cada modelo
- 🎨 Códigos de cores por modelo (Azul, Verde, Amarelo)
- 📈 Indicadores de confiança e disponibilidade
- 🔍 Análise comparativa automática
- 📱 Design responsivo

## 🛠️ Como Executar

### Pré-requisitos
```bash
pip install flask pandas numpy scikit-learn xgboost pickle-mixin
```

### Execução
```bash
cd web
python app.py
```

### Acesso
- **URL**: http://localhost:5000
- **API Info**: http://localhost:5000/api/info
- **Health Check**: http://localhost:5000/health

## 📁 Estrutura de Arquivos

```
web/
├── app.py                     # Servidor Flask principal (MULTI-MODELO)
├── templates/
│   └── index.html            # Interface web atualizada
├── static/
│   └── style.css            # Estilos customizados
└── README_MULTI_MODELO.md   # Esta documentação
```

## 🔄 Fluxo de Predição

1. **Entrada de Dados**: Usuário preenche formulário (modo básico ou avançado)
2. **Validação**: Sistema valida e complementa dados faltantes
3. **Execução dos Modelos**:
   - Tenta carregar modelos reais dos notebooks
   - Se não disponível, usa simulação baseada nos baselines
4. **Consolidação**: Calcula média, faixas e análise comparativa
5. **Apresentação**: Exibe resultados em formato visual comparativo

## 📊 Formato de Resposta da API

```json
{
  "modelos": {
    "notebook_4": {
      "nome": "Random Forest Otimizado (Notebook 4)",
      "predicao": 5234,
      "confianca_min": 4449,
      "confianca_max": 6019,
      "nivel_confianca": "high",
      "disponivel": true,
      "baseline": 6047
    },
    "notebook_5_rapido": {
      "nome": "XGBoost Super Otimizado (Notebook 5 RÁPIDO)",
      "predicao": 845,
      "confianca_min": 718,
      "confianca_max": 972,
      "nivel_confianca": "high",
      "disponivel": true,
      "baseline": 661
    },
    "ensemble": {
      "nome": "Ensemble (Média dos Modelos)",
      "predicao": 3040,
      "confianca_min": 2584,
      "confianca_max": 3496,
      "nivel_confianca": "high",
      "disponivel": true,
      "baseline": 3354
    }
  },
  "resumo": {
    "melhor_modelo": "XGBoost Super Otimizado (Notebook 5 RÁPIDO)",
    "menor_predicao": 845,
    "maior_predicao": 5234,
    "predicao_media": 3040
  },
  "dados_usados": 5,
  "estado_nome": "São Paulo"
}
```

## 🎨 Interface Visual

### Cards dos Modelos
- **Azul (Primary)**: Random Forest (Notebook 4)
- **Verde (Success)**: XGBoost Super Otimizado (Notebook 5)
- **Amarelo (Warning)**: Ensemble

### Indicadores de Status
- ✅ **Modelo Real**: Badge verde quando modelo está disponível
- 🧪 **Simulado**: Badge cinza quando usa simulação baseada em baseline

### Análise de Risco
- 🟢 **Baixo**: < 2,000 casos
- 🟡 **Médio**: 2,000 - 5,000 casos
- 🔴 **Alto**: > 5,000 casos

## 🔧 Configuração dos Modelos

### Caminhos dos Modelos
- **Notebook 4**: `../models/optimized/melhor_modelo_otimizado.pkl`
- **Notebook 5**: `../models/otimizado/modelo_otimizado.pkl`
- **Ensemble**: Calculado dinamicamente

### Fallback
Se os modelos não estiverem disponíveis, o sistema usa simulações inteligentes baseadas nos baselines conhecidos de cada modelo.

## 📈 Performance Comparativa

| Modelo | MAE | Tempo Treino | Speedup | Disponibilidade |
|--------|-----|--------------|---------|-----------------|
| Random Forest (NB4) | 6,047 | 45min | 1x | ✅ |
| XGBoost (NB5 RÁPIDO) | 661 | 30min | 16x | ✅ |
| Ensemble | 3,354 | - | - | ✅ |

## 🚀 Melhorias Implementadas

### vs Versão Original
- ✅ **Multi-Modelo**: Comparação de 3 modelos simultâneos
- ✅ **Interface Visual**: Cards comparativos e análise consolidada
- ✅ **Performance**: Modelos 10-16x mais rápidos
- ✅ **Robustez**: Fallback inteligente quando modelos não disponíveis
- ✅ **UX**: Modo básico e avançado para diferentes tipos de usuário

### Técnicas Aplicadas
- 🔧 **Feature Engineering Vetorizado**: Operações pandas otimizadas
- 🎯 **RandomizedSearchCV**: Busca eficiente de hiperparâmetros
- ⚡ **Early Stopping**: Evita overfitting e acelera treinamento
- 🔄 **Ensemble Learning**: Combina pontos fortes dos modelos
- 🎨 **Design Responsivo**: Interface adaptável a diferentes dispositivos

## 📞 Suporte

Para dúvidas ou problemas:
1. Verifique se todos os notebooks (4 e 5) foram executados
2. Confirme se os arquivos de modelo foram gerados
3. Teste a API via `/api/info` para verificar configuração
4. Use modo simulação se modelos não estiverem disponíveis

---

**Tech Challenge - FIAP - Fase 3**
*Sistema de Predição de Dengue com Machine Learning*