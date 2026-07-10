# Fase 4 — Baseline (Regressão Linear)

> Documentação dos experimentos baseline: regressão linear com e sem feature engineering.

---

## 1. Objetivo da fase

Demonstrar **por que feature engineering é indispensável** neste dataset:

| Experimento | Descrição | Resultado esperado |
|-------------|-----------|-------------------|
| **Naive** | `LinearRegression` só em colunas numéricas brutas | MAE alto, R² menor |
| **Engineered** | `LinearRegression` nas 277 features processadas (Fase 3) | MAE menor, R² maior |

Essa comparação é um dos insights mais valiosos do projeto — prova empiricamente o valor das Fases 2 e 3.

---

## 2. O que foi implementado

```
data/interim/clean.csv ──────► Experimento 1 (naive)
data/processed/X_*.csv ─────► Experimento 2 (engineered)
                │
                ▼  stage: baseline
models/
├── baseline_naive.pkl
└── baseline_engineered.pkl

reports/
├── baseline_metrics.json
└── baseline_comparison.md
```

### Arquivos criados

| Arquivo | Função |
|---------|--------|
| `src/models/metrics.py` | MAE, RMSE, R² com suporte a target em log |
| `src/models/baseline.py` | Dois experimentos + relatório comparativo |
| `tests/test_models.py` | 3 testes do baseline |
| `dvc.yaml` | Stage `baseline` adicionado |
| `docs/FASE4_BASELINE.md` | Este documento |

---

## 3. Experimento 1 — Sem feature engineering (naive)

### Estratégia

```python
# Apenas colunas numéricas de clean.csv
# Imputação mediana (mínimo necessário para rodar)
# Sem: encoding categórico, scaling, features derivadas, log target
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", LinearRegression()),
])
```

### Limitações intencionais

- **38 colunas categóricas ignoradas** (`Neighborhood`, `HouseStyle`, etc.)
- Sem `HouseAge`, `TotalSF`, `QualLivArea`
- Target em escala original (dólares), não log
- Apenas **36 features numéricas** brutas

### Resultados (holdout 20%)

| Métrica | Valor |
|---------|-------|
| **MAE** | **$22,979** |
| **RMSE** | $36,840 |
| **R²** | 0.8231 |
| CV MAE (5-fold) | $22,005 |

---

## 4. Experimento 2 — Com feature engineering (engineered)

### Estratégia

```python
# Features já processadas pela Fase 3
# 277 colunas: derived + imputation + OHE + StandardScaler
# Target: log1p(SalePrice)
model = LinearRegression()
model.fit(X_train, y_train)
```

### Resultados (holdout 20%)

| Métrica | Valor |
|---------|-------|
| **MAE** | **$15,367** |
| **RMSE** | $24,139 |
| **R²** | 0.9240 |
| CV MAE (5-fold) | $17,523 |

> MAE e R² reportados em **dólares** (predições revertidas com `expm1` quando o target está em log).

---

## 5. Comparação lado a lado

| Métrica | Sem FE | Com FE | Melhoria |
|---------|--------|--------|----------|
| MAE ($) | $22,979 | $15,367 | **33.1%** |
| RMSE ($) | $36,840 | $24,139 | **34.5%** |
| R² | 0.8231 | 0.9240 | **+12.3%** |
| Features | 36 | 277 | — |

### O que isso prova

1. **Categóricas importam** — ignorar `Neighborhood` e similares custa ~$7,600 de MAE
2. **Scaling e encoding são necessários** — modelos lineares são sensíveis à escala e ao tipo das variáveis
3. **Features derivadas agregam sinal** — `TotalSF`, `QualLivArea` capturam relações que colunas isoladas não capturam
4. **Log no target ajuda** — distribuição de preços é assimétrica (skewness 1.88)

---

## 6. Métricas explicadas

| Métrica | Fórmula | Interpretação neste projeto |
|---------|---------|---------------------------|
| **MAE** | média de \|preço real − preço previsto\| | "Erro médio de $X no preço da casa" |
| **RMSE** | √(média dos erros²) | Penaliza erros grandes (outliers) |
| **R²** | 1 − SS_res/SS_tot | % da variação de preço explicada pelo modelo |
| **CV MAE** | MAE médio em 5-fold cross-validation | Estimativa mais robusta da performance |

---

## 7. Integração com DVC

### Stage `baseline` no `dvc.yaml`

```yaml
baseline:
  cmd: python -m src.models.baseline
  deps:
    - data/interim/clean.csv
    - data/processed/X_train.csv
    - data/processed/X_test.csv
    - data/processed/y_train.csv
    - data/processed/y_test.csv
  params:
    - training.test_size
    - training.random_state
    - training.cv_folds
    - features.log_target
  outs:
    - models/baseline_naive.pkl
    - models/baseline_engineered.pkl
  metrics:
    - reports/baseline_metrics.json
    - reports/baseline_comparison.md
```

### Pipeline completo

```
prepare → featurize → baseline → [train — Fase 5+]
```

---

## 8. Comandos

```powershell
# Rodar só o baseline
dvc repro baseline

# Pipeline completo até aqui
dvc repro

# Manualmente
python -m src.models.baseline

# Testes
pytest tests/test_models.py -v
```

---

## 9. Artefatos salvos

| Artefato | Uso |
|----------|-----|
| `models/baseline_naive.pkl` | Referência do modelo sem FE |
| `models/baseline_engineered.pkl` | Ponto de partida para Fase 5 (MLflow) |
| `reports/baseline_metrics.json` | Métricas estruturadas para dashboards |
| `reports/baseline_comparison.md` | Relatório legível para documentação |

---

## 10. Testes automatizados

| Teste | Valida |
|-------|--------|
| `test_compute_regression_metrics_log_scale` | Conversão log → dólares nas métricas |
| `test_run_baselines_end_to_end` | Pipeline completo gera modelos e relatórios |
| `test_cross_validate_mae_uses_dollar_scale` | CV MAE em escala de dólares |

```powershell
pytest tests/ -v
# 12 passed
```

---

## 11. Próxima fase

**Fase 5 — MLflow:**

1. Integrar `mlflow` em `src/models/train.py`
2. Registrar os baselines como primeiros experimentos
3. Expandir para Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting
4. Meta: **20+ experimentos** no dashboard

O `baseline_engineered.pkl` serve como benchmark — todo modelo futuro deve superá-lo.

---

## 12. Troubleshooting

| Problema | Solução |
|----------|---------|
| `FileNotFoundError: X_train.csv` | Rode `dvc repro featurize` primeiro |
| MAE muito diferente entre runs | Verifique `random_state: 42` em `params.yaml` |
| Engineered pior que naive | Bug no pipeline — não deveria acontecer neste dataset |

---

> **Fase 4 concluída.** Feature engineering reduziu o MAE em 33% e elevou o R² de 0.82 para 0.92. Pronto para experimentação com MLflow na Fase 5.
