# Fase 3 — Feature Engineering

> Documentação da implementação do stage `featurize`: transformação de dados limpos em features prontas para modelagem.

---

## 1. O que foi implementado

A Fase 3 adiciona o stage **`featurize`** ao pipeline DVC, convertendo `data/interim/clean.csv` em matrizes numéricas prontas para treinar modelos na Fase 4.

```
data/interim/clean.csv
        │
        ▼  stage: featurize
data/processed/
├── X_train.csv          # Features de treino (1168 × 277)
├── X_test.csv           # Features de validação holdout (292 × 277)
├── y_train.csv          # Target log(SalePrice)
├── y_test.csv           # Target log(SalePrice) holdout
├── X_inference.csv      # Features do test set Kaggle (292 × 277)
├── preprocessor.pkl     # Pipeline sklearn serializado
└── features_metadata.json
```

**Pipeline completo atual:**

```
raw → prepare → featurize → [train] → [evaluate]
                      ↑
                 você está aqui
```

---

## 2. Arquivos criados

| Arquivo | Função |
|---------|--------|
| `src/features/transformers.py` | `DerivedFeaturesTransformer` — features de domínio |
| `src/features/build_features.py` | Orquestra split, pipeline sklearn e salvamento |
| `src/features/__init__.py` | Exports do pacote |
| `tests/test_features.py` | 3 testes do pipeline de features |
| `dvc.yaml` | Stage `featurize` adicionado |
| `params.yaml` | Parâmetros `features.*` expandidos |
| `docs/FASE3_FEATURE_ENGINEERING.md` | Este documento |

---

## 3. Fluxo de transformações

Todas as etapas estão encapsuladas em um **`sklearn.Pipeline`** para evitar data leakage:

```
1. Features derivadas     (DerivedFeaturesTransformer)
2. Imputação              (mediana → numéricas, moda → categóricas)
3. Encoding categórico    (OneHotEncoder, handle_unknown='ignore')
4. Scaling numérico       (StandardScaler)
5. Split treino/teste     (80/20, random_state=42)
6. Log no target          (log1p(SalePrice))
```

### Por que Pipeline?

O `fit()` roda **apenas no conjunto de treino**. O mesmo transformador é aplicado no holdout e no test set de inferência — exatamente como em produção.

---

## 4. Features derivadas

O `DerivedFeaturesTransformer` cria 8 variáveis com significado de negócio:

| Feature | Fórmula | Intuição |
|---------|---------|----------|
| `HouseAge` | `YrSold - YearBuilt` | Depreciação do imóvel |
| `YearsSinceRemod` | `YrSold - YearRemodAdd` | Tempo desde reforma |
| `GarageAge` | `YrSold - GarageYrBlt` | Idade da garagem (NA se sem garagem) |
| `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` | Área total construída |
| `LivAreaRatio` | `GrLivArea / TotalBsmtSF` | Proporção área habitável vs porão |
| `TotalBath` | `Full + 0.5×Half + BsmtFull + 0.5×BsmtHalf` | Banheiros equivalentes |
| `TotalPorchSF` | Soma de todas as varandas | Área externa coberta/descoberta |
| `QualLivArea` | `OverallQual × GrLivArea` | Interação qualidade × tamanho |

---

## 5. Pré-processamento por tipo de coluna

### Numéricas (45 colunas após derivadas)

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
```

### Categóricas (38 colunas)

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
```

### Polynomial features (opcional)

Controlado por `params.yaml`:

```yaml
features:
  use_polynomial: false   # true ativa PolynomialFeatures(degree=2)
  polynomial_degree: 2
```

Desligado por padrão — com 45 numéricas, grau 2 geraria ~1.000+ colunas.

---

## 6. Target: log(SalePrice)

O EDA (Fase 1) mostrou **skewness = 1.88** no preço. Com `log_target: true`:

```
y_train = log1p(SalePrice)
y_test  = log1p(SalePrice)
```

**Na Fase 4**, ao avaliar MAE em dólares, use `np.expm1(prediction)` para reverter a transformação.

---

## 7. Split treino / validação

| Conjunto | Linhas | Origem |
|----------|--------|--------|
| `X_train` + `y_train` | 1.168 (80%) | `clean.csv` com SalePrice |
| `X_test` + `y_test` | 292 (20%) | Holdout para avaliar modelos |
| `X_inference` | 292 | `clean_test.csv` (sem target, para submissão futura) |

> `X_test`/`y_test` aqui são **holdout interno**, não o test set oficial do Kaggle.

---

## 8. Resultados gerados

Após `dvc repro featurize`:

| Métrica | Valor |
|---------|-------|
| Features finais | **277** (após OneHotEncoder) |
| Linhas treino | 1.168 |
| Linhas validação | 292 |
| Linhas inferência | 292 |
| Log target | Sim |
| Polynomial | Não |

O arquivo `features_metadata.json` registra todos estes metadados para rastreabilidade.

---

## 9. Parâmetros versionados

```yaml
training:
  test_size: 0.2
  random_state: 42

features:
  use_polynomial: false
  polynomial_degree: 2
  log_target: true
```

O DVC rastreia estes parâmetros no `dvc.lock`. Alterar `test_size` ou `log_target` e rodar `dvc repro` reexecuta o stage automaticamente.

---

## 10. Comandos

### Reproduzir só o featurize

```powershell
dvc repro featurize
```

### Reproduzir pipeline completo (prepare + featurize)

```powershell
dvc repro
```

### Rodar manualmente (sem DVC)

```powershell
python -m src.features.build_features
```

### Testes

```powershell
pytest tests/test_features.py -v
```

### Visualizar DAG

```powershell
dvc dag
```

---

## 11. Artefato chave: `preprocessor.pkl`

O pipeline sklearn serializado permite:

- **Fase 4:** Carregar e treinar modelos sobre `X_train.csv`
- **Fase 8 (API):** Transformar input JSON → features antes de predizer
- **Reprodutibilidade:** Mesma transformação em treino e produção

```python
import joblib
preprocessor = joblib.load("data/processed/preprocessor.pkl")
X_new = preprocessor.transform(raw_features_df)
```

---

## 12. Testes automatizados

| Teste | O que valida |
|-------|--------------|
| `test_derived_features_transformer` | Criação de HouseAge, TotalSF, QualLivArea |
| `test_build_preprocessor_shapes` | Output do ColumnTransformer |
| `test_build_feature_datasets_end_to_end` | Pipeline completo com dados mini |

```powershell
pytest tests/test_features.py tests/test_data.py -v
# 9 passed
```

---

## 13. O que commitar no Git

```
src/features/
tests/test_features.py
dvc.yaml
dvc.lock
params.yaml
docs/FASE3_FEATURE_ENGINEERING.md
Makefile
```

**Não commitar:** `data/processed/*.csv`, `data/processed/preprocessor.pkl` (gerenciados pelo DVC via `dvc.lock`)

---

## 14. Próxima fase

**Fase 4 — Baseline:**

1. Stage `train` no `dvc.yaml`
2. Regressão linear **sem** vs **com** feature engineering
3. Métricas MAE e R² documentadas
4. Provar o valor do pipeline construído nas Fases 2 e 3

---

## 15. Troubleshooting

| Problema | Solução |
|----------|---------|
| `FileNotFoundError: clean.csv` | Rode `dvc repro prepare` primeiro |
| Shape mismatch no modelo | Verifique se usou `preprocessor.pkl` para transformar novos dados |
| MAE muito baixo/alto | Lembre que `y_train` está em log — use `expm1` para interpretar em dólares |
| Muitas features (>500) | Desative polynomial ou reduza cardinalidade categórica |

---

> **Fase 3 concluída.** Os dados estão transformados, versionados e prontos para treinar os 6 algoritmos da Fase 4–6.
