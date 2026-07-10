# House Price Prediction Engine

Pipeline MLOps completo para precificação de imóveis (Ames Housing) — do dado bruto à inferência em produção com dashboard premium.

## Stack

| Camada | Tecnologia |
|--------|------------|
| Dados | DVC, pandas, OpenML/Kaggle |
| Features | scikit-learn Pipeline, 8 features derivadas |
| Experimentos | MLflow (33 runs), 6 algoritmos |
| Registry | MLflow Model Registry → Production |
| API | FastAPI + Pydantic |
| Frontend | Dashboard premium (Chart.js) |
| Infra | Docker, docker-compose |

## Resultados

| Modelo | MAE | R² |
|--------|-----|-----|
| Baseline naive | $22,979 | 0.823 |
| **Linear Regression (Production)** | **$15,367** | **0.924** |
| Ridge (α=0.01) | $15,372 | 0.924 |
| Gradient Boosting (best) | $16,076 | — |

Feature engineering reduziu MAE em **~33%**. Testes t pareados entre top 3 modelos: diferenças **não significativas** (p > 0.05).

## Instalação

```powershell
cd "1-Foundation"
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline DVC

```powershell
# Pipeline completo
.\venv\Scripts\dvc.exe repro

# Stages individuais
.\venv\Scripts\dvc.exe repro prepare
.\venv\Scripts\dvc.exe repro featurize
.\venv\Scripts\dvc.exe repro train
.\venv\Scripts\dvc.exe repro evaluate
```

```
prepare → featurize → baseline
                   └→ train → evaluate → API
```

## API + Dashboard

```powershell
.\venv\Scripts\uvicorn.exe src.api.main:app --reload --port 8000
```

| URL | Descrição |
|-----|-----------|
| http://localhost:8000 | Dashboard premium |
| http://localhost:8000/docs | Swagger UI |
| http://localhost:8000/health | Health check |
| http://localhost:8000/predict | Predição (POST) |

### Exemplo de predição

```powershell
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"GrLivArea": 1710, "OverallQual": 7, "YearBuilt": 2003, "Neighborhood": "CollgCr"}'
```

## MLflow UI

```powershell
.\venv\Scripts\mlflow.exe ui --port 5000 --backend-store-uri sqlite:///mlflow.db
```

Modelo registrado: `house-price-predictor` v1 → **Production**

## Docker

```powershell
# API containerizada
docker compose up --build

# API + MLflow UI
docker compose --profile full up --build
```

```powershell
curl http://localhost:8000/health
```

## Testes

```powershell
.\venv\Scripts\python.exe -m pytest tests/ -v
```

## Estrutura

```
src/
├── data/          # prepare, validate
├── features/      # transformers, build_features
├── models/        # baseline, train, evaluate, registry
├── api/           # FastAPI + dashboard endpoints
└── utils/
frontend/          # Dashboard premium
reports/           # Métricas, comparações
docs/              # FASE1–FASE8_10
```

## Documentação por fase

| Fase | Documento |
|------|-----------|
| 2 — DVC | `docs/FASE2_DVC.md` |
| 3 — Features | `docs/FASE3_FEATURE_ENGINEERING.md` |
| 4 — Baseline | `docs/FASE4_BASELINE.md` |
| 5 — MLflow | `docs/FASE5_MLFLOW.md` |
| 6/7 — Registry | `docs/FASE6_7_REGISTRY.md` |
| 8–10 — API/Docker | `docs/FASE8_10_API.md` |

## Guia completo

Ver `GUIA_PROJETO.md` para o roteiro detalhado das 10 fases.

---

> **House Price Prediction Engine** — projeto de portfólio MLOps demonstrando reprodutibilidade, experimentação rigorosa e deploy de modelos em produção.
