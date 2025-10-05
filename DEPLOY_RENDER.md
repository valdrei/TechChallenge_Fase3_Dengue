# 🚀 Deploy no Render - Sistema de Predição de Dengue

## 📋 Pré-requisitos
- ✅ Conta no [Render](https://render.com)
- ✅ Repositório Git (GitHub, GitLab, etc.)
- ✅ Código já commitado

## 🔧 Arquivos Necessários (JÁ CRIADOS)
- ✅ `Procfile` - Comando de inicialização
- ✅ `render.yaml` - Configuração automática
- ✅ `requirements.txt` - Dependências Python
- ✅ `app.py` - Configurado para produção

## 🚀 Passos para Deploy

### 1️⃣ Conectar ao GitHub
1. Acesse [dashboard.render.com](https://dashboard.render.com)
2. Clique em **"New"** → **"Web Service"**
3. Conecte sua conta do GitHub
4. Selecione o repositório `TechChallenge_Fase3_Dengue`

### 2️⃣ Configurar o Serviço
```yaml
Name: dengue-prediction-system
Environment: Python
Region: Oregon (ou preferência)
Branch: main (ou sua branch)
Build Command: pip install -r requirements.txt
Start Command: gunicorn --chdir web app:app --bind 0.0.0.0:$PORT
```

### 3️⃣ Variáveis de Ambiente
Adicionar no painel do Render:
```
FLASK_ENV=production
FLASK_DEBUG=false
PYTHON_VERSION=3.9.16
```

### 4️⃣ Deploy Automático
- ✅ **Auto-Deploy**: Ativado por padrão
- ✅ **Plan**: Free (suficiente para demo)
- ✅ **Build Time**: ~5-10 minutos

## 🌐 Após o Deploy

### 📱 URL de Acesso
```
https://dengue-prediction-system.onrender.com
```

### 🔍 Verificações
- ✅ Página principal carrega
- ✅ Modelos são carregados corretamente
- ✅ Predições funcionam nos 2 modos
- ✅ Interface responsiva

## ⚡ Configurações de Performance

### 🚀 Otimizações Aplicadas
- ✅ **Gunicorn**: Servidor WSGI otimizado
- ✅ **Threading**: Múltiplas requisições simultâneas
- ✅ **Auto-scale**: Baseado no tráfego
- ✅ **CDN**: Arquivos estáticos otimizados

### 📊 Limites do Plano Free
- **RAM**: 512MB
- **CPU**: Compartilhada
- **Sleep**: Após 15min inativo
- **Uptime**: ~750h/mês

## 🛠️ Troubleshooting

### ❌ Problemas Comuns:

#### 1. **Build Failed**
```bash
# Verificar requirements.txt
pip install -r requirements.txt
```

#### 2. **Modelos não carregam**
```python
# Verificar caminhos relativos no app.py
base_path = os.path.join(os.path.dirname(__file__), '..')
```

#### 3. **Timeout de Build**
- Remover dependências não essenciais
- Usar versões específicas no requirements.txt

### ✅ **Logs de Debug**
```bash
# No dashboard do Render:
Logs → Deploy Logs → Runtime Logs
```

## 🎯 Para Apresentação

### 🌟 Vantagens do Deploy
- ✅ **URL pública** para demonstrações
- ✅ **Interface profissional** acessível de qualquer lugar
- ✅ **Performance estável** para apresentações
- ✅ **HTTPS automático** (segurança)

### 📱 Testagem
Testar nos principais cenários:
- 🔹 **Modo Básico**: SP, Janeiro, temp 25°C
- 🔹 **Modo Avançado**: Todos os campos preenchidos
- 🔹 **Estados diferentes**: RJ, MG, BA
- 🔹 **Responsividade**: Mobile e desktop

---

**🎉 Sistema pronto para demonstrações profissionais!**