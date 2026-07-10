# Fase 1 — Download + EDA

## Pré-requisito: autenticação no Kaggle

1. Acesse https://www.kaggle.com/settings
2. Em **API**, clique em **Create New Token**
3. Salve o arquivo `kaggle.json` em:
   - Windows: `C:\Users\<seu-usuario>\.kaggle\kaggle.json`

Alternativa via variáveis de ambiente:

```powershell
$env:KAGGLE_USERNAME = "LuizNazareth"
$env:KAGGLE_KEY = "KGAT_f97d067dae7282390873182d4375f2f9"
```

## Passo 1 — Instalar dependências da fase

```powershell
cd "C:\Users\Luiz\Documents\LuizNazareth\MLOps and AIOps Projects\1-Foundation"
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Passo 2 — Baixar dataset para `data/raw/`

```powershell
python scripts/download_data.py
```

Arquivos esperados:

- `data/raw/train.csv`
- `data/raw/test.csv`
- `data/raw/data_description.txt`
- `data/raw/sample_submission.csv` (opcional)

## Passo 3 — Rodar EDA

```powershell
jupyter notebook notebooks/01_eda.ipynb
```

Ou no VS Code/Cursor: abra `notebooks/01_eda.ipynb` e execute as células.

## O que o notebook cobre

- [x] Shape do dataset
- [x] Tipos de colunas
- [x] Missing values
- [x] Distribuição do target
- [x] Correlação com `SalePrice`
- [x] Cardinalidade categórica
- [x] Outliers (boxplots)
- [x] Scatter plots
- [x] Features redundantes (>0.9)

## Próxima fase

Depois de responder as 4 perguntas do EDA, siga para **Fase 2 — DVC**.
