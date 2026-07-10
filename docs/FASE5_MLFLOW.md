# Fase 5 — MLflow (Experiment Tracking)

> Documentação da integração com MLflow: tracking de 33 experimentos, comparação de algoritmos e seleção do melhor modelo.

---

## 1. O que foi implementado

A Fase 5 adiciona **rastreamento profissional de experimentos** com MLflow:

```
data/processed/  →  train.py  →  mlflow.db (33 runs)
                      │
                      ├── models/best_model.pkl
                      └── reports/training_summary.json
```

Cada experimento registra:
- **Parâmetros** (`algorithm`, `alpha`, `n_estimators`, etc.)
- **Métricas** (`mae`, `rmse`, `r2`, `cv_mae`, `n_features`)
- **Modelo** (artefato sklearn serializado)
- **Tags** (`algorithm`, `feature_set`, `project`, `phase`)

---

## 2. Arquivos criados

| Arquivo | Função |
|---------|--------|
| `src/models/train.py` | Orquestra 33 experimentos e logging MLflow |
| `src/utils/mlflow_utils.py` | Setup de tracking, helpers de log |
| `tests/test_train.py` | Testes de grid e integração MLflow |
| `mlflow.db` | Backend SQLite (local, não vai para Git) |
| `params.yaml` | Seções `mlflow` e `experiments` |
| `docs/FASE5_MLFLOW.md` | Este documento |

---

## 3. Backend de tracking: SQLite

Usamos `sqlite:///mlflow.db` em vez do file store (`mlruns/`), conforme recomendação do MLflow 3.x:

```yaml
mlflow:
  tracking_uri: sqlite:///mlflow.db
  experiment_name: house-price-prediction
```

| Backend | Vantagem |
|---------|----------|
| `mlruns/` (file) | Simples, mas em modo manutenção no MLflow 3 |
| `sqlite:///mlflow.db` | Rápido, local, suportado oficialmente |

---

## 4. Experimentos registrados (33 runs)

### Baselines (Fase 4 revisitados no MLflow)

| Run | MAE | R² |
|-----|-----|-----|
| `baseline_naive` | $22,979 | 0.8231 |
| `baseline_engineered` | $15,367 | 0.9240 |

### 6 algoritmos com grid de hiperparâmetros

| Algoritmo | Runs | Melhor MAE |
|-----------|------|------------|
| Linear Regression | 1 | $15,367 |
| Ridge | 5 | $15,372 |
| Lasso | 5 | $16,393 |
| ElasticNet | 6 | $21,827 |
| Random Forest | 6 | $17,109 |
| Gradient Boosting | 8 | **$16,076** |

### Melhor modelo selecionado

**`linear_regression_engineered`** — MAE $15,367 | R² 0.9240

Salvo em `models/best_model.pkl`.

> O Gradient Boosting (`gbr_est200_lr0.05_depth5`) chegou perto com MAE $16,076 — na Fase 7 faremos testes estatísticos para confirmar se a diferença é significativa.

---

## 5. O que cada run registra no MLflow

```python
with mlflow.start_run(run_name="ridge_alpha_1.0"):
    mlflow.set_tag("algorithm", "ridge")
    mlflow.set_tag("feature_set", "engineered")
    mlflow.log_param("alpha", 1.0)
    mlflow.log_metric("mae", 16056.0)
    mlflow.log_metric("r2", 0.9162)
    mlflow.log_metric("cv_mae", ...)
    mlflow.sklearn.log_model(model, name="model")
```

### Parâmetros versionados (`params.yaml`)

```yaml
experiments:
  ridge:
    alpha: [0.01, 0.1, 1.0, 10.0, 100.0]
  lasso:
    alpha: [0.001, 0.01, 0.1, 1.0, 10.0]
  elasticnet:
    alpha: [0.1, 1.0]
    l1_ratio: [0.3, 0.5, 0.7]
  random_forest:
    n_estimators: [100, 200]
    max_depth: [10, 20, null]
  gradient_boosting:
    n_estimators: [100, 200]
    learning_rate: [0.05, 0.1]
    max_depth: [3, 5]
```

---

## 6. Pipeline DVC

```
prepare → featurize → baseline
                   └→ train (MLflow)
```

### Stage `train`

```yaml
train:
  cmd: venv/Scripts/python -m src.models.train
  outs:
    - models/best_model.pkl
  metrics:
    - reports/training_summary.json
```

> **Windows:** o comando usa `venv/Scripts/python` explicitamente para garantir que o DVC use o ambiente virtual com MLflow instalado.

---

## 7. Comandos

### Treinar e registrar experimentos

```powershell
.\venv\Scripts\activate
python -m src.models.train
# ou
dvc repro train
```

### Abrir dashboard MLflow

```powershell
.\venv\Scripts\activate
mlflow ui --port 5000
```

Acesse: **http://localhost:5000**

No dashboard você pode:
- Comparar runs lado a lado
- Filtrar por `algorithm` (tag)
- Ordenar por `mae` ascendente
- Visualizar artefatos (modelos salvos)

### Consultar runs via CLI

```powershell
mlflow experiments search
mlflow runs list --experiment-id 1
```

---

## 8. Artefatos gerados

| Artefato | Conteúdo |
|----------|----------|
| `mlflow.db` | Banco SQLite com todos os 33 runs |
| `models/best_model.pkl` | Melhor modelo (menor MAE) |
| `reports/training_summary.json` | Resumo JSON com best_run e lista de runs |

---

## 9. Insights dos experimentos

### Regularização (Ridge/Lasso)

- **Ridge** com `alpha=0.01` performa quase igual à regressão linear ($15,372 vs $15,367)
- **Lasso** com `alpha` alto degrada muito (seleção agressiva de features → underfitting)
- Confirma a teoria: L1 zera pesos, L2 apenas encolhe

### Ensemble (RF/GBR)

- Random Forest: MAE ~$17,100 (não superou linear com features engineered)
- Gradient Boosting: melhor GBR com MAE $16,076 — competitivo, mas não venceu linear

### Conclusão parcial

Com **feature engineering completo**, regressão linear já captura bem o sinal. Modelos mais complexos podem ajudar marginalmente — validaremos com testes estatísticos na Fase 7.

---

## 10. Testes automatizados

```powershell
pytest tests/test_train.py -v
```

| Teste | Valida |
|-------|--------|
| `test_build_experiment_grid_has_minimum_runs` | Grid gera múltiplos experimentos |
| `test_run_training_logs_runs_to_mlflow` | Runs aparecem no SQLite backend |

**Total do projeto: 14 testes passando.**

---

## 11. O que commitar no Git

```
src/models/train.py
src/utils/mlflow_utils.py
tests/test_train.py
dvc.yaml
dvc.lock
params.yaml
docs/FASE5_MLFLOW.md
reports/training_summary.json
Makefile
```

**Não commitar:** `mlflow.db`, `mlruns/`, `models/best_model.pkl`

---

## 12. Próxima fase

**Fase 6–7 — Model Registry e testes estatísticos:**

1. Promover `best_model` para MLflow Model Registry
2. Estágios: `Staging` → `Production`
3. Teste pareado (paired t-test) entre top 3 modelos
4. Documentar em `reports/model_comparison.md`

---

## 13. Troubleshooting

| Problema | Solução |
|----------|---------|
| `ModuleNotFoundError: mlflow` no `dvc repro` | Use `venv/Scripts/python` no cmd ou `python -m dvc repro` com venv ativo |
| File store bloqueado | Use `sqlite:///mlflow.db` (já configurado) |
| UI não abre | `mlflow ui --port 5000` com venv ativo |
| Poucos runs no dashboard | Rode `python -m src.models.train` novamente |

---

> **Fase 5 concluída.** 33 experimentos rastreados no MLflow. Dashboard disponível em `http://localhost:5000`.
