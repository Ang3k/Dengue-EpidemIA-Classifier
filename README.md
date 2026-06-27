# IML-Dengue

Projeto de machine learning sobre dengue usando os dados públicos do SINAN (2017
a 2019). A ideia é tratar os dados, fazer a análise exploratória e treinar um
modelo que separa os casos confirmados dos casos descartados. Esse modelo é a
base de um site de triagem de suspeitas.

## Objetivo

A pergunta que o modelo tenta responder é epidemiológica: com o que se sabe de um
caso na hora da notificação (sintomas, perfil do paciente, local e época do ano),
qual a chance de ele ser dengue confirmada? O alvo é a coluna
`final_classification`, que reduzimos para `0 = descartado` e `1 = confirmado`.

## Organização

```text
dengue_pipeline/       código Python reutilizável
  cleaner.py           classe DengueDataCleaner (limpeza e features)
  sinan_mappings.py    renomeio de colunas e rótulos do SINAN
  cbo_map.py           mapeamento CBO das ocupações
  paths.py             caminhos do projeto
  models/              wrapper de XGBoost/LightGBM com tuning
notebooks/
  cleaning/            execução do tratamento
  analysis/            análise exploratória
  modeling/            treino e avaliação dos modelos
data/
  raw/parquet/         dados originais em Parquet
  raw/csv/             dados originais em CSV
  processed/           bases geradas pelo tratamento
artifacts/models/      modelos treinados em Joblib
docs/references/       documentos do SINAN (dicionário de dados)
reports/figures/       gráficos gerados pela análise
src/                   site em React + TypeScript (Vite)
```

Os notebooks executam e mostram cada etapa. A parte de código que dá para
reaproveitar fica em `dengue_pipeline/`.

## Dados

Usamos os arquivos de dengue de 2017, 2018 e 2019, que somam mais ou menos 2,8
milhões de notificações:

```text
data/raw/parquet/DENGBR17.parquet
data/raw/parquet/DENGBR18.parquet
data/raw/parquet/DENGBR19.parquet
```

As versões em CSV ficam em `data/raw/csv/`. O dicionário de dados do SINAN está em
`docs/references/dicionario_dados_dengue.pdf`.

## Tratamento de dados

A classe `DengueDataCleaner` (em `dengue_pipeline/cleaner.py`) carrega os três
anos, junta as limpezas que cada integrante fez e tem duas saídas:

- `transformar_analise()`: dataframe mais legível, com os rótulos em texto (sexo,
  raça, UF, escolaridade e por aí vai). É o que usamos na análise exploratória.
- `transformar_ml()`: dataframe todo numérico, pronto para o modelo.

No tratamento mais geral fazemos o seguinte:

- renomeamos as colunas do SINAN para nomes mais fáceis de ler;
- criamos os rótulos de sexo, raça, gestação, escolaridade, UF, ocupação (via
  CBO) e classificação final;
- criamos a coluna `days_to_notification` (data da notificação menos a data de
  início dos sintomas);
- removemos as colunas de encerramento, que dariam vazamento;
- transformamos a classificação final em `0 = descartado` e `1 = confirmado`.

Já no `transformar_ml`, as principais mudanças nas colunas são:

- os 12 sintomas e a prova do laço (`tourniquet_test`) saem do código do SINAN
  (1/2/NaN) e viram binário 0/1;
- criamos agregados de sintomas: `number_of_symptoms`,
  `number_of_important_symptoms` e as interações entre pares de sintomas;
- a sazonalidade (mês da notificação, mês de início dos sintomas e semana
  epidemiológica) vira seno e cosseno, e a versão crua é descartada;
- os encodings: one-hot para sexo e raça, ordinal para UF de residência,
  escolaridade e ocupação, e a gravidez vira duas flags binárias;
- o ano (`notification_year` e `symptom_epi_year`) fica de fora das features,
  porque ele não ajuda a generalizar para anos novos.

A base de ML fica salva em:

```text
data/processed/dengue_tratado_ml.parquet
```

## Evitando vazamento de dados

Separamos as responsabilidades de propósito para não vazar informação do teste
para o treino:

- o `cleaner` só faz transformações que não dependem dos dados (one-hot,
  binarização, seno/cosseno, mapeamentos fixos), então pode rodar no conjunto
  inteiro sem problema;
- o que depende de estatística dos dados, como imputar pela mediana, fica no
  `Pipeline` da modelagem e é ajustado só no treino;
- nenhuma coluna que só existe depois que o caso é encerrado entra no modelo.

## Modelagem

O notebook é o `notebooks/modeling/models.ipynb`. Ali comparamos regressão
logística, árvore de decisão, XGBoost e LightGBM. Os dois últimos passam pela
classe `GradientBoostingDiseaseClassifier` (em `dengue_pipeline/models/`), que
ajusta os hiperparâmetros com o Optuna usando o PR-AUC (average precision) como
métrica. Trocamos o recall por essa métrica porque otimizar só o recall acabava
gerando um modelo que classificava quase tudo como positivo.

Algumas decisões importantes:

- Geografia: mantivemos a localização de residência (`residence_municipality`,
  `residence_health_region` e a UF), que prevê bastante por causa da
  endemicidade e não é vazamento. A geografia de notificação e o
  `health_facility` (que tem dezenas de milhares de valores) ficaram de fora,
  porque são mais administrativos e o modelo tende a decorar.
- Validação no tempo: no split padrão treinamos com 2017, 2018 e o começo de
  2019, e testamos no segundo semestre de 2019. Para um teste mais justo de
  generalização dá para treinar em 2017 e 2018 e testar em 2019 inteiro (o que
  chamamos de cross-year).
- Limiar de decisão: o modelo devolve uma probabilidade, e o ponto de corte é uma
  escolha nossa. Para vigilância, em que perder um caso real é pior que um alarme
  falso, vale usar um limiar baixo. O método `evaluate` mostra precisão, recall e
  F1 em vários limiares.
- Desempenho: no recorte temporal o XGBoost fica perto de 0,82 de PR-AUC. Na
  validação cross-year, testando num ano que ele nunca viu, cai para uns 0,68 de
  ROC-AUC. Vale lembrar que o modelo se apoia bastante na geografia e na época do
  ano, e que o próprio alvo depende em parte do critério da vigilância (em
  epidemia, muito caso é confirmado pelo critério clínico-epidemiológico).

Os modelos treinados ficam em `artifacts/models/` e os gráficos de importância em
`reports/figures/modeling/`.

## Como rodar a limpeza

Instalar as dependências:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Gerar a base de ML:

```powershell
py -3.11 -c "from dengue_pipeline import DengueDataCleaner; df = DengueDataCleaner().transformar_ml(); print(df.shape)"
```

Ou é só abrir os notebooks da pasta `notebooks/`.

## Análise exploratória

```text
notebooks/analysis/analise_exploratoria_dengue.ipynb
```

Os gráficos ficam salvos em `reports/figures/`, que é a mesma pasta que a página
de Panorama Epidemiológico do site usa.

## Site

Feito com React, TypeScript e Vite.

Rotas:

```text
/          página inicial (descrição e simulação com os modelos)
/triagem   formulário de triagem
/graphics  Panorama Epidemiológico (gráficos da análise)
```

A triagem e a simulação chamam a API FastAPI em `api.py`, que usa os modelos
treinados de `artifacts/models/`. O arquivo `ml_preprocess.joblib`, gerado pela
primeira célula do notebook de modelagem, é obrigatório para aplicar os mesmos
encoders usados no treino.

### Como iniciar a API

```powershell
.\.venv\Scripts\python -m uvicorn api:app --reload
```

A API fica em `http://localhost:8000`. O endpoint `/health` informa quais
modelos e artefatos de pré-processamento foram carregados.

### Como abrir o site

```powershell
npm install
npm run dev
```

O endereço costuma ser `http://localhost:5173`. Para gerar o build é
`npm run build`. Para usar outra URL de API, defina `VITE_API_URL` no ambiente
antes de iniciar o Vite.

## Próximos passos

- Tratar a geografia de alta cardinalidade com algum encoding por frequência ou
  região (ajustado só no treino) em vez de simplesmente descartar.
- Usar a validação cross-year como métrica principal de generalização.
- Calibrar as probabilidades e definir o limiar de corte da triagem.
