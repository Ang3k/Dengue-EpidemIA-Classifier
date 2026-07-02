# Dengue Sense Classifier

Classificador de confirmação versus descarte de dengue com dados públicos do SINAN. O pipeline usa somente informações disponíveis na notificação; exames, encerramento, evolução e sinais posteriores não entram nas features.

## Recorte temporal

| Uso | Anos | Casos rotulados |
|---|---:|---:|
| Treino | 2014–2019 | 7.723.448 |
| Validação, Optuna, early stopping, pesos e limiar | 2020 | 1.331.664 |
| Teste final e simulação histórica | 2021 | 940.304 |
| Total bruto | 2014–2021 | 11.441.770 |
| Total rotulado | 2014–2021 | 9.995.416 |

Anos anteriores a 2014 e 2022 em diante estão fora do projeto. O teste de 2021 não pode alterar modelos, hiperparâmetros, encoders, medianas, pesos ou limiar.

O alvo é harmonizado por período:

- 2014–2016: `{1, 2, 3, 4, 10, 11, 12} → 1` e `{5} → 0`;
- 2017–2021: `{10, 11, 12} → 1` e `{5} → 0`;
- `{0, 8, 9, vazio, inesperado}` → removido e auditado.

## Estrutura principal

```text
data/dengue_manifest.json          snapshot oficial, hashes, esquema e contagens
dengue_pipeline/cleaner.py         limpeza sem estado, orientada a chunks
dengue_pipeline/features.py        esquema único de 105 features
dengue_pipeline/datasets.py        carregamento das partições temporais
dengue_pipeline/models/            MLP, XGBoost e LightGBM
scripts/prepare_dengue_data.py     download, validação e ETL
scripts/train_models.py            treino 2014–2019 e validação 2020
scripts/evaluate_models.py         calibração 2020 e teste final 2021
api.py                             FastAPI e simulação com casos de 2021
```

Os ZIPs oficiais e Parquets derivados ficam fora do Git. O repositório versiona o manifesto, o relatório de auditoria, os modelos finais, suas métricas e o pool reduzido da simulação.

## Ambiente

Requer Python 3.11 e Node.js.

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
npm ci
```

## Preparação dos dados

```powershell
python scripts/prepare_dengue_data.py
```

O script baixa os oito ZIPs oficiais, exige os SHA-256 fixados, lê chunks de 100.000 linhas e grava um Parquet de análise e outro de ML por ano. Ele falha se cabeçalho, tamanho, hash, número de linhas, distribuição de `CLASSI_FIN`, esquema ou presença das duas classes divergir do manifesto.

O relatório fica em `reports/data/dengue_data_audit.csv`. Há uma limitação real na fonte: 2014 não informa os sintomas selecionados, e 2015–2016 têm muitos sintomas ausentes. Ausência permanece `NaN`; `number_of_reported_symptoms` separa “não informado” de “não”.

## Treinamento e avaliação

```powershell
python scripts/train_models.py --n-trials 200 --max-epochs 150 --tuning-sample-size 200000
python scripts/evaluate_models.py --threshold-step 0.01
```

A MLP exige CUDA no treinamento e ajusta `OrdinalEncoder` e medianas somente em 2014–2019. XGBoost e LightGBM treinam em CPU e tratam `NaN` nativamente. Optuna usa amostras estratificadas e determinísticas de até 200 mil registros do treino e de 2020; o ajuste final usa todo o treino.

O treino cria `artifacts/models/model_manifest.json`. A avaliação escolhe pesos e limiar em 2020, avalia 2021 uma única vez e cria `artifacts/models/ensemble_config.json`. A API valida versão do esquema, períodos, lista de features e hashes antes de aceitar os artefatos. Os modelos antigos ficam indisponíveis até o retreinamento completo.

## API e frontend

```powershell
python -m uvicorn api:app --reload
npm run dev
```

O contrato de `POST /predict` mantém os mesmos campos. Sintomas aceitam `0`, `1` ou ausência; campos ausentes viram `NaN`. `GET /health` informa a versão do esquema, os períodos e a compatibilidade dos artefatos. A simulação histórica usa exclusivamente casos rotulados de 2021.

## Validação

```powershell
python -m unittest discover -s tests -v
npm run build
```

Os testes cobrem os dois mapeamentos do alvo, os esquemas de 2014/2019/2020/2021, igualdade de features entre pipeline e API, separação temporal, contagens oficiais e ausência da coluna removida. O ETL registra o pico de RSS por ano e deve permanecer abaixo de 16 GiB; o treino falha se atingir 28 GiB.
