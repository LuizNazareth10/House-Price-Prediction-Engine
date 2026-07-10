# Fases 8–10 — FastAPI, Docker e Documentação

> API de inferência, dashboard premium, containerização e testes finais.

---

## 1. Fase 8 — FastAPI

### Arquivos

| Arquivo | Função |
|---------|--------|
| `src/api/main.py` | App FastAPI, rotas, static files |
| `src/api/schemas.py` | Pydantic — validação de input |
| `src/api/predictor.py` | Carrega preprocessor + modelo, inferência |
| `src/api/dashboard.py` | Agrega métricas dos reports |
| `frontend/` | Dashboard premium (HTML/CSS/JS) |

### Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/health` | Status do modelo |
| POST | `/predict` | Predição de preço |
| GET | `/api/v1/dashboard/overview` | KPIs e pipeline |
| GET | `/api/v1/dashboard/algorithms` | Comparação de algoritmos |
| GET | `/api/v1/dashboard/examples` | Exemplos reais holdout |
| GET | `/` | Dashboard UI |
| GET | `/docs` | Swagger automático |

### Fluxo de predição

```
Input (15 campos principais)
    → merge com defaults do treino (clean.csv)
    → preprocessor.pkl (derived + OHE + scale)
    → best_model.pkl
    → expm1() se log_target
    → preço em dólares + intervalo ±MAE
```

### Executar

```powershell
.\venv\Scripts\uvicorn.exe src.api.main:app --reload --port 8000
```

Abra http://localhost:8000 para o dashboard.

---

## 2. Dashboard Premium

O frontend em `frontend/` inclui:

- **Hero** — KPIs (33 experimentos, 277 features, MAE, R²)
- **Pipeline DVC** — 6 stages visualizados (prepare → serve)
- **Gráficos Chart.js** — baseline vs engineered, 6 algoritmos, scatter actual vs predicted
- **Model Registry** — banner Production
- **Tabela** — métricas por algoritmo
- **Exemplos reais** — 6 casos do holdout com erro %
- **Simulador** — form interativo conectado ao `/predict`

Design: tema escuro luxo (ouro + esmeralda), tipografia Cormorant Garamond + Outfit, glassmorphism.

---

## 3. Fase 9 — Docker

### Dockerfile

Imagem slim Python 3.12 com uvicorn na porta 8000.

### docker-compose.yml

| Serviço | Porta | Profile |
|---------|-------|---------|
| `api` | 8000 | default |
| `mlflow` | 5000 | `full` |

```powershell
docker compose up --build
docker compose --profile full up --build
```

Volumes montados: `models/`, `data/processed/`, `data/interim/`, `reports/`.

---

## 4. Fase 10 — Testes

```powershell
.\venv\Scripts\python.exe -m pytest tests/ -v
```

| Arquivo | Cobertura |
|---------|-----------|
| `tests/test_api.py` | health, predict, dashboard, UI |
| `tests/test_evaluate.py` | significância, registry |
| `tests/test_train.py` | MLflow grid |
| `tests/test_features.py` | pipeline features |
| `tests/test_data.py` | prepare/validate |
| `tests/test_models.py` | baseline, métricas |

---

## 5. Checklist final

- [x] Endpoint `/predict` com Pydantic
- [x] Endpoint `/health`
- [x] Swagger em `/docs`
- [x] Dashboard premium com gráficos
- [x] Exemplos reais do dataset
- [x] Dockerfile funcional
- [x] docker-compose (api + mlflow opcional)
- [x] Testes automatizados
- [x] README completo

---

## 6. Comandos rápidos

```powershell
# Tudo local
.\venv\Scripts\dvc.exe repro evaluate
.\venv\Scripts\uvicorn.exe src.api.main:app --port 8000
.\venv\Scripts\pytest.exe tests/ -v

# Docker
docker compose up --build
```
