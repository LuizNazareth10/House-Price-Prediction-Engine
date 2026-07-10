# House Price Prediction Engine — Guia Completo de Implementação

> **Objetivo:** Construir um pipeline de regressão de ponta a ponta com MLOps profissional, usando o dataset Ames Housing, para aprender na prática como empresas como Zillow, Redfin e OpenDoor estruturam seus sistemas de precificação.

---

## Índice

1. [Visão Geral do Projeto](#1-visão-geral-do-projeto)
2. [Dataset: Qual Usar e Como Obter](#2-dataset-qual-usar-e-como-obter)
3. [Ferramentas do Stack — O Que É Cada Uma](#3-ferramentas-do-stack--o-que-é-cada-uma)
4. [Conceitos de Machine Learning — Explicação Completa](#4-conceitos-de-machine-learning--explicação-completa)
5. [Os 6 Algoritmos que Você Deve Comparar](#5-os-6-algoritmos-que-você-deve-comparar)
6. [Estrutura de Arquivos do Projeto](#6-estrutura-de-arquivos-do-projeto)
7. [Ordem de Implementação (Passo a Passo)](#7-ordem-de-implementação-passo-a-passo)
8. [Fases Detalhadas](#8-fases-detalhadas)
9. [Critérios de "Projeto Completo"](#9-critérios-de-projeto-completo)
10. [Cronograma Sugerido](#10-cronograma-sugerido)
11. [Próximo Passo Imediato](#11-próximo-passo-imediato)

---

## 1. Visão Geral do Projeto

Este não é um projeto de "treinar um modelo e pronto". É um **sistema de ML em produção** com quatro camadas:

```
Dados (DVC) → Experimentos (MLflow) → Modelo (Registry) → API (FastAPI + Docker)
```

| Camada | Responsabilidade |
|--------|------------------|
| **Dados** | Versionar, validar e transformar o dataset de forma reproduzível |
| **Treinamento** | Comparar algoritmos, registrar métricas e hiperparâmetros |
| **Modelo** | Promover o melhor modelo para o registry (produção) |
| **Serving** | Expor predições via REST API containerizada |

### O que você vai aprender

- Por que regressão linear "crua" falha sem feature engineering
- Como gradient descent minimiza a função de perda
- A matemática por trás da regularização L1 (Lasso) e L2 (Ridge)
- Como versionar dados e pipelines como código
- Como rastrear 20+ experimentos e escolher o melhor modelo com testes estatísticos

---

## 2. Dataset: Qual Usar e Como Obter

### Dataset oficial: **Ames Housing Dataset**

| Propriedade | Valor |
|-------------|-------|
| **Nome** | Ames Housing (House Prices — Advanced Regression Techniques) |
| **Origem** | Competição Kaggle / Dean De Cock (Iowa State University) |
| **Amostras** | ~1.460 casas (treino) + ~1.459 (teste, sem target) |
| **Features** | 79 variáveis preditoras + 1 target (`SalePrice`) |
| **Target** | Preço de venda em dólares (regressão contínua) |
| **Tipos de features** | Numéricas, categóricas ordinais e nominais |

### Por que este dataset?

1. **80 features** — força você a dominar feature engineering (missing values, encoding, scaling)
2. **Tamanho ideal** — grande o suficiente para CV, pequeno o suficiente para iterar rápido
3. **Padrão da indústria** — usado em cursos de ML e competições há anos
4. **Realismo** — mistura de variáveis estruturais (área, quartos) e de qualidade (acabamento, garagem)

### Como obter

**Opção A — Kaggle (recomendada):**
```
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
```
Arquivos necessários:
- `train.csv` — dados com `SalePrice` (target)
- `test.csv` — dados sem target (opcional para validação final)
- `data_description.txt` — dicionário de todas as colunas (leia!)

**Opção B — Repositório alternativo:**
```
https://www.openml.org/d/42165
```

### Estrutura dos dados após download

```
data/raw/
├── train.csv              # ~1.460 linhas, 81 colunas (80 features + SalePrice)
├── test.csv               # ~1.459 linhas, 80 colunas (sem SalePrice)
└── data_description.txt   # Documentação de cada feature
```

### Colunas importantes para conhecer desde o início

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `SalePrice` | Target | Preço de venda (dólares) — **sua variável alvo** |
| `GrLivArea` | Numérica | Área habitável acima do solo (sq ft) |
| `OverallQual` | Ordinal (1-10) | Qualidade geral do material e acabamento |
| `YearBuilt` | Numérica | Ano de construção |
| `Neighborhood` | Categórica | Bairro em Ames, Iowa (25 categorias) |
| `TotalBsmtSF` | Numérica | Área total do porão |
| `GarageCars` | Numérica | Capacidade da garagem em carros |

> **Dica:** Leia o `data_description.txt` inteiro antes de qualquer código. Entender o significado de cada feature é metade do feature engineering.

---

## 3. Ferramentas do Stack — O Que É Cada Uma

### Python
Linguagem base. Todo o pipeline (dados, treino, API) roda em Python 3.10+.

### pandas / numpy
| Ferramenta | Papel no projeto |
|------------|------------------|
| **pandas** | Carregar CSVs, explorar dados (EDA), tratar missing values, criar features derivadas |
| **numpy** | Operações numéricas, álgebra linear (útil para entender gradient descent) |

**Exemplo de uso:**
```python
import pandas as pd
df = pd.read_csv("data/raw/train.csv")
df["Age"] = df["YrSold"] - df["YearBuilt"]  # feature derivada
```

### scikit-learn
Biblioteca de ML do ecossistema Python. Você vai usar:

| Módulo | Para quê |
|--------|----------|
| `sklearn.linear_model` | Ridge, Lasso, ElasticNet, LinearRegression |
| `sklearn.ensemble` | RandomForestRegressor, GradientBoostingRegressor |
| `sklearn.preprocessing` | StandardScaler, OneHotEncoder, PolynomialFeatures |
| `sklearn.pipeline` | Pipeline — encadeia preprocessing + modelo em um objeto |
| `sklearn.model_selection` | cross_val_score, GridSearchCV, train_test_split |
| `sklearn.metrics` | mean_absolute_error, r2_score |

**Por que Pipeline?**
```python
# Sem pipeline: risco de data leakage
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # OK

# Com pipeline: tudo encapsulado, seguro e reproduzível
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0))
])
pipe.fit(X_train, y_train)
pipe.predict(X_test)  # scaler aplicado automaticamente
```

### MLflow
**O que é:** Plataforma open-source para o ciclo de vida de ML.

| Funcionalidade | O que faz no seu projeto |
|----------------|--------------------------|
| **Experiment Tracking** | Registra métricas (MAE, R²), parâmetros (alpha, max_depth) e artefatos de cada run |
| **Model Registry** | Armazena versões do modelo com estágios: `Staging` → `Production` |
| **UI Dashboard** | Interface web em `http://localhost:5000` para comparar experimentos |

**Fluxo típico:**
```python
import mlflow

with mlflow.start_run(run_name="ridge_alpha_1.0"):
    mlflow.log_param("alpha", 1.0)
    mlflow.log_param("model", "Ridge")
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2_score(y_test, model.predict(X_test)))
    mlflow.sklearn.log_model(model, "model")
```

**Por que 20+ experimentos?** Cada combinação de algoritmo + hiperparâmetros = 1 experimento. Com 6 algoritmos e variações de hiperparâmetros, você chega facilmente a 20+.

### DVC (Data Version Control)
**O que é:** Git para dados e pipelines de ML.

| Conceito | Analogia Git | No seu projeto |
|----------|--------------|----------------|
| `dvc add` | `git add` | Versiona `train.csv` (gera `.dvc` + envia para remote storage) |
| `dvc.yaml` | `Makefile` | Define stages: `prepare` → `featurize` → `train` → `evaluate` |
| `params.yaml` | Config file | Hiperparâmetros versionados (alpha, n_estimators, etc.) |
| `dvc repro` | `make all` | Reexecuta pipeline inteiro de forma reproduzível |
| `dvc push/pull` | `git push/pull` | Sincroniza dados com remote (local, S3, GDrive) |

**Por que DVC?**
- Outra pessoa clona seu repo e roda `dvc pull && dvc repro` → obtém exatamente os mesmos resultados
- Dados grandes não vão para o Git (só ponteiros `.dvc`)
- Pipeline reproduzível = requisito profissional

### FastAPI
**O que é:** Framework Python moderno para criar APIs REST com validação automática.

**No seu projeto:**
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="House Price Prediction API")

class HouseFeatures(BaseModel):
    GrLivArea: float
    OverallQual: int
    YearBuilt: int
    # ... demais features

@app.post("/predict")
def predict(house: HouseFeatures):
    features = preprocess(house.dict())
    price = model.predict(features)
    return {"predicted_price": float(price)}
```

**Endpoints que você deve implementar:**
| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/predict` | POST | Recebe features, retorna preço predito |
| `/health` | GET | Health check para Docker/Kubernetes |
| `/model-info` | GET | Versão do modelo, métricas, data de treino |

### Docker
**O que é:** Containerização — empacota app + dependências + modelo em uma imagem isolada.

**Por que Docker?**
- "Funciona na minha máquina" deixa de ser problema
- Mesmo ambiente em dev, staging e produção
- Deploy em qualquer cloud (AWS ECS, GCP Cloud Run, etc.)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 4. Conceitos de Machine Learning — Explicação Completa

### 4.1 Regressão Linear — O Baseline

**O que é:** Encontrar uma reta (ou hiperplano) que melhor se ajusta aos dados.

**Fórmula:**
```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

**Função de perda (MSE — Mean Squared Error):**
```
L = (1/n) Σ (yᵢ - ŷᵢ)²
```

**Por que falha sem feature engineering no Ames Housing?**
1. Features categóricas (`Neighborhood`, `HouseStyle`) não são numéricas
2. Relações não-lineares (área × qualidade) não são capturadas
3. Escala diferente entre features (sq ft vs. ano) distorce o gradiente
4. Multicolinearidade (área total vs. área habitável) causa instabilidade

### 4.2 Gradient Descent — Como o Modelo Aprende

**O que é:** Algoritmo iterativo que ajusta os pesos (β) para minimizar a perda.

**Atualização dos pesos:**
```
βⱼ := βⱼ - α · ∂L/∂βⱼ
```

Onde:
- `α` = learning rate (tamanho do passo)
- `∂L/∂βⱼ` = gradiente (direção de maior aumento da perda)

**Para regressão linear, o gradiente da MSE é:**
```
∂L/∂βⱼ = -(2/n) Σ (yᵢ - ŷᵢ) · xᵢⱼ
```

**Intuição:** Se o erro é positivo (previu baixo), aumenta o peso. Se negativo, diminui.

**Variantes:**
| Variante | Descrição |
|----------|-----------|
| Batch GD | Usa todos os dados a cada passo (lento, estável) |
| Stochastic GD | Usa 1 amostra por passo (rápido, ruidoso) |
| Mini-batch GD | Compromisso: usa lotes de N amostras |

> scikit-learn usa soluções analíticas (equação normal) para modelos lineares pequenos, mas entender GD é fundamental para modelos maiores (redes neurais, etc.).

### 4.3 Regularização — Combatendo Overfitting

**O problema:** Modelo complexo demais memoriza o treino e falha no teste (alta variância).

**Bias-Variance Tradeoff:**

```
Erro Total = Bias² + Variância + Irreduzível

Bias alto    → modelo simples demais (underfitting)
Variância alta → modelo complexo demais (overfitting)
```

**Regularização adiciona penalidade aos pesos grandes:**

| Técnica | Penalidade | Efeito |
|---------|------------|--------|
| **Ridge (L2)** | α Σ βⱼ² | Encolhe pesos uniformemente, mantém todas as features |
| **Lasso (L1)** | α Σ \|βⱼ\| | Zera pesos de features irrelevantes (seleção automática) |
| **ElasticNet** | α₁ Σ \|βⱼ\| + α₂ Σ βⱼ² | Combinação de L1 + L2 |

**Função de perda com regularização L2 (Ridge):**
```
L_ridge = (1/n) Σ (yᵢ - ŷᵢ)² + α Σ βⱼ²
```

**Função de perda com regularização L1 (Lasso):**
```
L_lasso = (1/n) Σ (yᵢ - ŷᵢ)² + α Σ |βⱼ|
```

**Como escolher α (hiperparâmetro):**
- α = 0 → sem regularização (regressão linear pura)
- α muito alto → underfitting (pesos zerados)
- Use `GridSearchCV` ou `RandomizedSearchCV` para encontrar o α ideal

### 4.4 Feature Scaling

**O que é:** Normalizar features para mesma escala.

| Método | Fórmula | Quando usar |
|--------|---------|-------------|
| **StandardScaler** | z = (x - μ) / σ | Default para modelos lineares e baseados em distância |
| **MinMaxScaler** | z = (x - min) / (max - min) | Quando precisa de range [0, 1] |
| **RobustScaler** | z = (x - mediana) / IQR | Quando há outliers |

**Por que é necessário?**
- `GrLivArea` (0–5000) vs. `OverallQual` (1–10) → sem scaling, GD converge lentamente
- Ridge/Lasso penalizam pesos — features com escala maior seriam penalizadas injustamente

### 4.5 Feature Engineering

**Técnicas que você deve aplicar no Ames Housing:**

| Técnica | Exemplo | Por quê |
|---------|---------|---------|
| **Tratamento de missing** | `LotFrontage` → mediana por bairro | 80% das colunas têm NAs |
| **Encoding categórico** | OneHotEncoder para `Neighborhood` | Modelos lineares precisam de números |
| **Features derivadas** | `Age = YrSold - YearBuilt` | Captura depreciação |
| **Features de interação** | `TotalSF = GrLivArea + TotalBsmtSF` | Área total é melhor preditor |
| **Transformação log** | `log(SalePrice)` como target | Distribuição right-skewed → log normaliza |
| **Polynomial features** | `GrLivArea²` | Captura relações não-lineares |

### 4.6 Cross-Validation

**O que é:** Dividir dados em K folds, treinar K vezes, avaliar em cada fold.

```
Fold 1: [TEST][train][train][train][train]
Fold 2: [train][TEST][train][train][train]
Fold 3: [train][train][TEST][train][train]
...
```

**Por que usar?**
- Estimativa mais robusta da performance (não depende de uma única divisão)
- Detecta overfitting (alta variância entre folds = instável)
- Padrão: `K=5` ou `K=10`

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
print(f"MAE médio: {-scores.mean():.2f} (+/- {scores.std():.2f})")
```

### 4.7 Métricas de Avaliação

| Métrica | Fórmula | Interpretação |
|---------|---------|---------------|
| **MAE** | (1/n) Σ \|yᵢ - ŷᵢ\| | Erro médio em dólares. **Mais interpretável** para preços |
| **RMSE** | √[(1/n) Σ (yᵢ - ŷᵢ)²] | Penaliza erros grandes (outliers) |
| **R²** | 1 - SS_res/SS_tot | % da variância explicada. 1.0 = perfeito, 0 = baseline |

**Para este projeto, foque em MAE e R²:**
- MAE → "em média, erro de $X no preço"
- R² → "o modelo explica X% da variação dos preços"

### 4.8 Testes de Significância Estatística

**O que é:** Verificar se a diferença entre dois modelos é estatisticamente significativa (não por acaso).

**Teste pareado (Paired t-test):**
```python
from scipy import stats

# MAE de cada fold para modelo A e modelo B
mae_a = [21000, 19500, 22000, 20500, 21500]  # 5-fold CV
mae_b = [23000, 21500, 24000, 22500, 23500]

t_stat, p_value = stats.ttest_rel(mae_a, mae_b)
# p_value < 0.05 → diferença é estatisticamente significativa
```

**Por que importa?** Diferença de $500 no MAE pode ser ruído. O teste estatístico confirma se o modelo vencedor é genuinamente melhor.

---

## 5. Os 6 Algoritmos que Você Deve Comparar

| # | Algoritmo | Tipo | Hiperparâmetros-chave | Por que incluir |
|---|-----------|------|----------------------|-----------------|
| 1 | **Linear Regression** | Linear | Nenhum | Baseline — mostra o problema sem feature engineering |
| 2 | **Ridge** | Linear + L2 | `alpha` | Regularização suave, benchmark linear |
| 3 | **Lasso** | Linear + L1 | `alpha` | Seleção automática de features |
| 4 | **ElasticNet** | Linear + L1+L2 | `alpha`, `l1_ratio` | Compromisso Ridge/Lasso |
| 5 | **Random Forest** | Ensemble (bagging) | `n_estimators`, `max_depth` | Captura não-linearidades, robusto |
| 6 | **Gradient Boosting** | Ensemble (boosting) | `n_estimators`, `learning_rate`, `max_depth` | Estado da arte para tabular |

### Experimento mínimo para 20+ runs no MLflow

```
Linear Regression     × 1 config                    =  1 run
Ridge                 × 5 valores de alpha           =  5 runs
Lasso                 × 5 valores de alpha           =  5 runs
ElasticNet            × 3 configs                    =  3 runs
Random Forest         × 3 configs                    =  3 runs
Gradient Boosting     × 4 configs                    =  4 runs
Feature engineering   × 2 versões (básica vs. full)  =  ×2
                                          Total mínimo ≈ 21+ runs
```

---

## 6. Estrutura de Arquivos do Projeto

```
1-Foundation/
│
├── .dvc/                          # Configuração DVC (gerado por dvc init)
├── .dvcignore                     # Arquivos ignorados pelo DVC
├── .gitignore
├── .env.example                   # Variáveis de ambiente (sem secrets)
│
├── data/
│   ├── raw/                       # Dados originais (versionados com DVC)
│   │   ├── train.csv.dvc
│   │   ├── test.csv.dvc
│   │   └── data_description.txt
│   ├── interim/                   # Dados intermediários (pós-limpeza)
│   │   └── .gitkeep
│   └── processed/                 # Features finais prontas para treino
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── notebooks/
│   ├── 01_eda.ipynb               # Análise exploratória
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_comparison.ipynb  # Comparação visual dos 6 algoritmos
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py           # Carregar CSVs
│   │   ├── validate_data.py       # Validação de schema (Great Expectations opcional)
│   │   └── split_data.py          # Train/test split
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── build_features.py      # Pipeline de feature engineering
│   │   └── transformers.py        # Transformers customizados (sklearn-compatible)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py               # Script principal de treino + MLflow
│   │   ├── evaluate.py            # Avaliação + testes estatísticos
│   │   ├── predict.py             # Carregar modelo e predizer
│   │   └── registry.py            # Promover modelo no MLflow Registry
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI app
│   │   ├── schemas.py             # Pydantic models (request/response)
│   │   └── dependencies.py        # Carregar modelo uma vez no startup
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py              # Leitura de params.yaml
│       └── logging.py             # Configuração de logs
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py               # Testes de carregamento e validação
│   ├── test_features.py           # Testes de feature engineering
│   ├── test_models.py             # Testes de treino e métricas
│   └── test_api.py                # Testes da API (TestClient do FastAPI)
│
├── models/                        # Artefatos do modelo (gerado, no .gitignore)
│   └── .gitkeep
│
├── mlruns/                        # MLflow tracking local (no .gitignore)
│
├── reports/                       # Relatórios gerados
│   ├── model_comparison.md        # Tabela comparativa dos 6 algoritmos
│   └── figures/                   # Gráficos de EDA e resultados
│
├── scripts/
│   ├── download_data.sh           # Script para baixar dados do Kaggle
│   └── run_experiments.sh         # Rodar todos os experimentos
│
├── dvc.yaml                       # Pipeline DVC (stages)
├── params.yaml                    # Hiperparâmetros versionados
├── requirements.txt               # Dependências Python
├── requirements-dev.txt           # Dependências de desenvolvimento (pytest, etc.)
├── Dockerfile                     # Container da API
├── docker-compose.yml             # API + MLflow UI
├── Makefile                       # Atalhos (make train, make api, make test)
├── pyproject.toml                 # Config do projeto (opcional, mas profissional)
└── README.md                      # Documentação principal
```

### Arquivos de configuração-chave

**`dvc.yaml`** — Pipeline reproduzível:
```yaml
stages:
  prepare:
    cmd: python src/data/load_data.py
    deps:
      - src/data/load_data.py
      - data/raw/train.csv
    outs:
      - data/interim/clean.csv

  featurize:
    cmd: python src/features/build_features.py
    deps:
      - src/features/build_features.py
      - data/interim/clean.csv
    outs:
      - data/processed/X_train.csv
      - data/processed/X_test.csv
      - data/processed/y_train.csv
      - data/processed/y_test.csv

  train:
    cmd: python src/models/train.py
    deps:
      - src/models/train.py
      - data/processed/
    params:
      - params.yaml
    outs:
      - models/pipeline.pkl

  evaluate:
    cmd: python src/models/evaluate.py
    deps:
      - src/models/evaluate.py
      - models/pipeline.pkl
      - data/processed/
    metrics:
      - reports/metrics.json
```

**`params.yaml`** — Hiperparâmetros:
```yaml
model:
  name: "ridge"
  alpha: 1.0

training:
  test_size: 0.2
  random_state: 42
  cv_folds: 5

features:
  use_polynomial: false
  log_target: true
```

---

## 7. Ordem de Implementação (Passo a Passo)

Siga esta ordem **estritamente**. Cada fase depende da anterior.

```
FASE 0 ─ Setup do ambiente
   │
FASE 1 ─ Dados + EDA
   │
FASE 2 ─ DVC (versionamento)
   │
FASE 3 ─ Feature Engineering
   │
FASE 4 ─ Baseline (Regressão Linear)
   │
FASE 5 ─ MLflow (experiment tracking)
   │
FASE 6 ─ 6 Algoritmos + 20+ experimentos
   │
FASE 7 ─ Model Registry + seleção final
   │
FASE 8 ─ FastAPI (serving)
   │
FASE 9 ─ Docker (containerização)
   │
FASE 10 ─ Testes + documentação
```

| Fase | O que fazer | Entregável | Tempo estimado |
|------|-------------|------------|----------------|
| **0** | Criar venv, instalar deps, init git | Ambiente funcional | 1–2h |
| **1** | Baixar dados, EDA no notebook | `01_eda.ipynb` + insights documentados | 3–5h |
| **2** | `dvc init`, versionar dados, criar `dvc.yaml` | `dvc repro` funciona | 2–3h |
| **3** | Pipeline de features, tratamento de NAs | `build_features.py` + dados processados | 4–6h |
| **4** | Regressão linear sem e com features | Baseline MAE/R² documentado | 2–3h |
| **5** | Integrar MLflow no script de treino | Tracking funcionando, UI acessível | 2–3h |
| **6** | Treinar 6 algoritmos, grid search, 20+ runs | Dashboard MLflow populado | 6–8h |
| **7** | Testes estatísticos, promover melhor modelo | Modelo em Production no Registry | 2–3h |
| **8** | API FastAPI com `/predict` e `/health` | API local funcionando | 3–4h |
| **9** | Dockerfile + docker-compose | `docker compose up` funciona | 2–3h |
| **10** | Testes pytest, README, relatório final | Projeto completo e documentado | 4–6h |

**Tempo total estimado: 30–45 horas** (distribuídas em 2–4 semanas)

---

## 8. Fases Detalhadas

### FASE 0 — Setup do Ambiente

```bash
# 1. Criar e ativar ambiente virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 2. Instalar dependências
pip install pandas numpy scikit-learn mlflow dvc fastapi uvicorn \
            pydantic pytest httpx scipy matplotlib seaborn python-dotenv

# 3. Inicializar Git
git init
git add .
git commit -m "chore: initial project structure"

# 4. Criar estrutura de pastas
mkdir -p data/raw data/interim data/processed
mkdir -p src/data src/features src/models src/api src/utils
mkdir -p notebooks tests models reports/figures scripts
```

**`requirements.txt` inicial:**
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
mlflow>=2.9
dvc>=3.0
fastapi>=0.104
uvicorn>=0.24
pydantic>=2.0
scipy>=1.11
matplotlib>=3.8
seaborn>=0.13
python-dotenv>=1.0
pytest>=7.4
httpx>=0.25
```

---

### FASE 1 — Dados + EDA

**Objetivo:** Entender profundamente os dados antes de escrever qualquer modelo.

**Checklist do EDA (`01_eda.ipynb`):**

- [ ] Shape do dataset (linhas, colunas)
- [ ] Tipos de cada coluna (numérica, categórica, ordinal)
- [ ] Missing values — quais colunas, quantos, padrão de ausência
- [ ] Distribuição do target (`SalePrice`) — histograma, skewness
- [ ] Correlação das features numéricas com `SalePrice` — heatmap
- [ ] Cardinalidade das features categóricas
- [ ] Outliers visuais (boxplots de `GrLivArea`, `SalePrice`)
- [ ] Relação entre features importantes (scatter plots)

**Perguntas que o EDA deve responder:**
1. O target precisa de transformação log?
2. Quais features têm >50% missing? (candidatas a remoção)
3. Quais features categóricas têm alta cardinalidade?
4. Existem features redundantes (correlação > 0.9)?

---

### FASE 2 — DVC

```bash
# Inicializar DVC
dvc init

# Versionar dados brutos
dvc add data/raw/train.csv
dvc add data/raw/test.csv

# Commitar os ponteiros .dvc (não os CSVs)
git add data/raw/*.dvc data/raw/.gitignore .dvc/
git commit -m "data: add raw datasets with DVC"

# Configurar remote storage (local para começar)
mkdir -p /dvc-storage
dvc remote add -d localstorage /dvc-storage
dvc push
```

---

### FASE 3 — Feature Engineering

**Ordem de transformações:**

```
Raw Data
  → Remover colunas com >80% missing (ou ID)
  → Imputar numéricas (mediana) e categóricas (moda)
  → Criar features derivadas (Age, TotalSF, etc.)
  → Encoding categórico (OneHotEncoder)
  → Log-transform no target (opcional, recomendado)
  → Train/Test split (80/20, random_state=42)
  → Salvar em data/processed/
```

**Regra de ouro:** Toda transformação deve estar dentro de um `sklearn.Pipeline` para evitar data leakage.

---

### FASE 4 — Baseline

```python
# Experimento 1: Linear Regression SEM feature engineering
# → MAE alto, R² baixo → PROVA que feature engineering é necessário

# Experimento 2: Linear Regression COM pipeline completo
# → MAE muito menor → PROVA o valor do preprocessing
```

Documente os dois resultados lado a lado. Essa comparação é um dos insights mais valiosos do projeto.

---

### FASE 5 — MLflow

```bash
# Iniciar UI do MLflow
mlflow ui --port 5000
# Acessar: http://localhost:5000
```

Integre tracking em `src/models/train.py`:
- Log de parâmetros (`mlflow.log_param`)
- Log de métricas (`mlflow.log_metric`)
- Log do modelo (`mlflow.sklearn.log_model`)
- Tags para organizar (`mlflow.set_tag("algorithm", "ridge")`)

---

### FASE 6 — 6 Algoritmos

Para cada algoritmo:
1. Definir grid de hiperparâmetros
2. Rodar `GridSearchCV` com 5-fold CV
3. Registrar cada combinação no MLflow
4. Salvar melhor configuração

```python
param_grid = {
    "ridge": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "lasso": {"model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
    "elasticnet": {"model__alpha": [0.1, 1.0], "model__l1_ratio": [0.3, 0.5, 0.7]},
    "random_forest": {"model__n_estimators": [100, 200], "model__max_depth": [10, 20, None]},
    "gradient_boosting": {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1], "model__max_depth": [3, 5]},
}
```

---

### FASE 7 — Model Registry

```python
# Registrar melhor modelo
model_uri = f"runs:/{best_run_id}/model"
mlflow.register_model(model_uri, "house-price-predictor")

# Promover para Production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="house-price-predictor",
    version=1,
    stage="Production"
)
```

**Teste de significância:**
- Coletar MAE de cada fold dos top 3 modelos
- Rodar `scipy.stats.ttest_rel` entre pares
- Documentar em `reports/model_comparison.md`

---

### FASE 8 — FastAPI

```bash
# Rodar API localmente
uvicorn src.api.main:app --reload --port 8000

# Testar
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"GrLivArea": 1500, "OverallQual": 7, "YearBuilt": 2000, ...}'
```

Documentação automática em: `http://localhost:8000/docs`

---

### FASE 9 — Docker

```bash
# Build e run
docker compose up --build

# Testar API containerizada
curl http://localhost:8000/health
```

**`docker-compose.yml` deve incluir:**
- Serviço `api` (FastAPI)
- Serviço `mlflow` (UI de experimentos) — opcional mas recomendado
- Volume para `models/` e `mlruns/`

---

### FASE 10 — Testes + Documentação

```bash
# Rodar testes
pytest tests/ -v

# Reproduzir pipeline inteiro
dvc repro

# Verificar reprodutibilidade
git clone <repo> && cd <repo> && dvc pull && dvc repro
```

**README.md deve conter:**
- Descrição do projeto
- Como instalar e rodar
- Como treinar modelos
- Como subir a API
- Resultados dos 6 algoritmos (tabela)
- Screenshots do MLflow dashboard

---

## 9. Critérios de "Projeto Completo"

Use este checklist para validar que o projeto está profissional:

### Dados
- [ ] Dados versionados com DVC
- [ ] `dvc repro` roda do zero sem erros
- [ ] Outra pessoa consegue reproduzir com `git clone` + `dvc pull`

### Modelos
- [ ] 6 algoritmos comparados com métricas documentadas
- [ ] 20+ experimentos no MLflow
- [ ] Teste de significância estatística entre top modelos
- [ ] Melhor modelo promovido para Production no Registry

### API
- [ ] Endpoint `/predict` funcional
- [ ] Endpoint `/health` funcional
- [ ] Validação de input com Pydantic
- [ ] Documentação Swagger automática

### Infraestrutura
- [ ] Dockerfile funcional
- [ ] `docker compose up` sobe tudo
- [ ] Testes automatizados passando (`pytest`)

### Documentação
- [ ] README completo
- [ ] Relatório de comparação de modelos
- [ ] Notebooks de EDA e análise

---

## 10. Cronograma Sugerido

### Semana 1 — Fundação
| Dia | Tarefa |
|-----|--------|
| 1 | Fase 0: Setup + Fase 1: Download dados |
| 2–3 | Fase 1: EDA completo no notebook |
| 4 | Fase 2: DVC setup |
| 5–7 | Fase 3: Feature engineering pipeline |

### Semana 2 — Modelagem
| Dia | Tarefa |
|-----|--------|
| 1 | Fase 4: Baseline (com e sem features) |
| 2 | Fase 5: Integração MLflow |
| 3–5 | Fase 6: 6 algoritmos + grid search |
| 6–7 | Fase 7: Testes estatísticos + Registry |

### Semana 3 — Produção
| Dia | Tarefa |
|-----|--------|
| 1–2 | Fase 8: FastAPI |
| 3 | Fase 9: Docker |
| 4–5 | Fase 10: Testes |
| 6–7 | Documentação final + polish |

### Semana 4 (opcional) — Extras
- CI/CD com GitHub Actions
- Monitoramento de drift
- Deploy em cloud (Railway, Render, AWS)
- Dashboard com Streamlit

---

## 11. Próximo Passo Imediato

**Comece agora com estes 3 comandos:**

```bash
# 1. Criar ambiente
cd "C:\Users\Luiz\Documents\LuizNazareth\MLOps and AIOps Projects\1-Foundation"
python -m venv .venv
.venv\Scripts\activate

# 2. Instalar dependências básicas para EDA
pip install pandas numpy matplotlib seaborn jupyter scikit-learn

# 3. Baixar o dataset
# Crie conta no Kaggle, gere API token, e rode:
pip install kaggle
# kaggle competitions download -c house-prices-advanced-regression-techniques -p data/raw/
# unzip data/raw/*.zip -d data/raw/
```

Depois, abra o Jupyter e comece o `01_eda.ipynb`. **Não pule o EDA** — é a fundação de tudo.

---

> **Lembre-se:** O valor deste projeto não está em ter o MAE mais baixo possível. Está em entender *por que* cada decisão foi tomada, *como* cada ferramenta se encaixa no pipeline, e *demonstrar* que você consegue construir um sistema de ML reproduzível e deployável do zero.
