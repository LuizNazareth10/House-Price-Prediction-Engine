# Fases 6 e 7 — Model Registry + Testes de Significância

> Documentação da promoção do melhor modelo ao MLflow Model Registry e comparação estatística entre os top modelos.

---

## 1. O que foi implementado

As Fases 6 e 7 fecham o ciclo de experimentação:

```
reports/training_summary.json  →  evaluate.py
        │                              │
        │                              ├── Testes t pareados (top 3)
        │                              ├── reports/model_comparison.md
        │                              └── registry.py → Production
        │
        └── mlflow.db (runs + model registry)
```

| Componente | Função |
|----------|--------|
| `src/models/evaluate.py` | Compara top modelos, roda testes estatísticos, gera relatórios |
| `src/models/registry.py` | Registra melhor run e promove para **Production** |
| `tests/test_evaluate.py` | Testes unitários e de integração |
| Stage `evaluate` no `dvc.yaml` | Reproduz avaliação após `train` |

---

## 2. Model Registry (Fase 7)

O melhor modelo (menor MAE no holdout) é registrado automaticamente:

```python
model_uri = f"runs:/{best_run_id}/model"
mlflow.register_model(model_uri, "house-price-predictor")
client.transition_model_version_stage(..., stage="Production")
```

**Configuração** (`params.yaml`):

```yaml
mlflow:
  registered_model_name: house-price-predictor

evaluation:
  top_n_models: 3
  significance_level: 0.05
```

**Saídas:**

| Arquivo | Conteúdo |
|---------|----------|
| `reports/registry_summary.json` | Nome, versão, stage, run de origem, métricas |
| MLflow Registry | Modelo `house-price-predictor` em **Production** |

### Ver no MLflow UI

```powershell
.\venv\Scripts\mlflow.exe ui --port 5000 --backend-store-uri sqlite:///mlflow.db
```

Abra **Models** → `house-price-predictor` → versão em **Production**.

---

## 3. Testes de significância estatística

Para os **top 3 modelos** (excluindo baselines):

1. Recalcula MAE por fold de cross-validation (5-fold)
2. Aplica **paired t-test** (`scipy.stats.ttest_rel`) entre cada par
3. Compara p-value com α = 0.05

**Interpretação:**

| p-value | Significado |
|---------|-------------|
| p < 0.05 | Diferença de MAE no CV é estatisticamente significativa |
| p ≥ 0.05 | Diferença pode ser aleatória; modelos são equivalentes no CV |

**Saídas:**

| Arquivo | Conteúdo |
|---------|----------|
| `reports/significance_tests.json` | MAE por fold + resultados dos testes |
| `reports/evaluation_metrics.json` | Payload completo da avaliação |
| `reports/model_comparison.md` | Tabela comparativa + conclusão |

---

## 4. Resultados esperados (dataset Ames)

Com o pipeline atual, os top 3 modelos por MAE no holdout são tipicamente:

| Rank | Run | MAE ($) |
|------|-----|---------|
| 1 | `linear_regression_engineered` | ~15,367 |
| 2 | `ridge_alpha_0.01` | ~15,372 |
| 3 | `gbr_est200_lr0.05_depth5` | ~16,076 |

A diferença entre **Linear Regression** e **Ridge (α=0.01)** costuma ser **não significativa** no CV pareado — ambos têm desempenho equivalente. O Linear Regression é promovido por ter o menor MAE no holdout.

---

## 5. Como executar

### Pipeline completo

```powershell
cd "C:\Users\Luiz\Documents\LuizNazareth\MLOps and AIOps Projects\1-Foundation"
.\venv\Scripts\activate
.\venv\Scripts\dvc.exe repro evaluate
```

### Apenas avaliação (após `train`)

```powershell
.\venv\Scripts\python.exe -m src.models.evaluate
```

### Apenas registry

```powershell
.\venv\Scripts\python.exe -m src.models.registry
```

### Testes

```powershell
.\venv\Scripts\python.exe -m pytest tests/test_evaluate.py -v
```

---

## 6. Pipeline DVC atualizado

```
prepare → featurize → baseline
                   └→ train → evaluate
```

O stage `evaluate` depende de:

- `reports/training_summary.json` (métricas do `train`)
- `mlflow.db` (runs e artefatos para registry)
- `data/processed/X_train.csv` e `y_train.csv` (CV para testes)

---

## 7. Carregar modelo de Production

Para usar o modelo promovido em código ou na API (Fase 8):

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.pyfunc.load_model("models:/house-price-predictor/Production")
predictions = model.predict(X_new)
```

---

## 8. Próximos passos (Fase 8+)

- [ ] **Fase 8** — FastAPI com endpoint `/predict` carregando `models:/house-price-predictor/Production`
- [ ] **Fase 9** — Docker + docker-compose
- [ ] **Fase 10** — Testes finais e README com tabela dos 6 algoritmos

---

## 9. Checklist desta fase

- [x] Top N modelos comparados com métricas documentadas
- [x] Teste de significância estatística (paired t-test)
- [x] Relatório `reports/model_comparison.md`
- [x] Melhor modelo promovido para **Production** no Registry
- [x] Stage `evaluate` no DVC
- [x] Testes automatizados
