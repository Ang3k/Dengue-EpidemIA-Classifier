# Endpoints necessários para completar o site

## Objetivo

Completar a simulação da página inicial com um caso histórico real e melhorar a
triagem para que o usuário pesquise informações por nome, sem precisar digitar
códigos técnicos.

O Panorama Epidemiológico continuará utilizando gráficos estáticos e não
precisa de novos endpoints.

## Estado atual

| Método | Endpoint | Estado | Finalidade |
|---|---|---|---|
| `GET` | `/health` | Implementado | Verifica os modelos e o pré-processamento |
| `POST` | `/predict` | Implementado | Executa a predição de uma triagem |
| `POST` | `/api/v1/simulations/random` | Pendente | Seleciona um caso real do conjunto de teste e executa a predição |
| `GET` | `/api/v1/triage/options` | Pendente | Retorna as opções gerais do formulário |
| `GET` | `/api/v1/references/occupations` | Pendente | Pesquisa ocupações e códigos CBO |
| `GET` | `/api/v1/references/municipalities` | Pendente | Pesquisa municípios por nome |
| `GET` | `/api/v1/references/health-regions` | Pendente | Retorna a região de saúde de um município |

## Divisão entre duas pessoas

### Pessoa 1 — Simulação da página inicial

Endpoint:

```http
POST /api/v1/simulations/random
```

Responsabilidades:

- selecionar um caso real e anonimizado do conjunto de teste;
- usar somente registros de 2019 com mês de notificação maior ou igual a junho;
- executar MLP, XGBoost e LightGBM;
- retornar dados gerais, classificação observada e probabilidades;
- alterar o simulador da Home;
- criar os testes da simulação.

Branch sugerida:

```text
feature/real-simulation
```

### Pessoa 2 — Formulário inteligente da triagem

Endpoints:

```http
GET /api/v1/triage/options
GET /api/v1/references/occupations
GET /api/v1/references/municipalities
GET /api/v1/references/health-regions
```

Responsabilidades:

- implementar as opções gerais da triagem;
- criar os autocompletes de ocupações e municípios;
- preencher a região de saúde pelo município;
- guardar os códigos internamente;
- calcular semana epidemiológica e dias até a notificação;
- alterar o formulário da Triagem;
- criar os testes dos endpoints e autocompletes.

Branch sugerida:

```text
feature/triage-autocomplete
```

Cada pessoa deve trabalhar em um router e nos componentes da sua própria
funcionalidade. Assim, as duas tarefas ficam independentes e o possível conflito
fica limitado ao registro dos routers na aplicação principal.

## 1. Simulação com um caso histórico real

### Endpoint

```http
POST /api/v1/simulations/random
```

O backend deve:

1. selecionar aleatoriamente um registro em que:

   ```text
   notification_year == 2019
   notification_month >= 6
   ```

2. remover identificadores e campos desnecessários;
3. executar o mesmo pré-processamento usado no treinamento;
4. executar MLP, XGBoost e LightGBM;
5. retornar os dados gerais, a classificação observada e as probabilidades.

Uma `seed` poderá ser enviada para reproduzir uma simulação:

```json
{
  "seed": 42
}
```

Resposta sugerida:

```json
{
  "case": {
    "age": 35,
    "sex": "Feminino",
    "race": "Parda",
    "occupation": "Médico clínico",
    "state": "RJ",
    "municipality": "Rio de Janeiro",
    "symptoms": ["Febre", "Mialgia", "Cefaleia"]
  },
  "observedClassification": "Dengue",
  "prediction": {
    "models": [
      {
        "name": "mlp",
        "probability": 63.2
      },
      {
        "name": "xgboost",
        "probability": 58.7
      },
      {
        "name": "lightgbm",
        "probability": 71.4
      }
    ],
    "average": 64.4,
    "isDengue": true
  }
}
```

O frontend deve parar de gerar pessoas no navegador e passar a consumir esse
endpoint.

## 2. Opções gerais da triagem

### Endpoint

```http
GET /api/v1/triage/options
```

Deve retornar:

- sexos;
- raças;
- escolaridades;
- situações de gestação;
- sintomas;
- UFs e códigos IBGE;
- modelos ativos;
- limiar de classificação.

Esse endpoint elimina listas duplicadas entre backend e frontend.

## 3. Pesquisa de ocupações

### Endpoint

```http
GET /api/v1/references/occupations?query=medico&limit=10
```

A pesquisa deve:

- começar após pelo menos dois caracteres;
- ignorar acentos e letras maiúsculas;
- aguardar aproximadamente 300 ms após a digitação (`debounce`);
- priorizar nomes que começam com o texto;
- mostrar depois os nomes que contêm o texto;
- permitir seleção por mouse e teclado.

Resposta sugerida:

```json
{
  "items": [
    {
      "code": "225125",
      "name": "Médico clínico"
    }
  ]
}
```

O usuário verá o nome da ocupação, enquanto o frontend guardará internamente o
código CBO.

## 4. Pesquisa de municípios

### Endpoint

```http
GET /api/v1/references/municipalities?query=rio&limit=20
GET /api/v1/references/municipalities?query=rio&state=33&limit=20
```

Parâmetros:

| Parâmetro | Obrigatório | Descrição |
|---|---|---|
| `query` | Sim | Parte do nome, com pelo menos dois caracteres |
| `state` | Não | Código da UF para restringir a busca |
| `limit` | Não | Quantidade máxima de resultados |

Ao digitar `rio`, o autocomplete poderá mostrar:

```text
Rio de Janeiro — RJ
Rio Grande — RS
Rio Branco — AC
Rio das Ostras — RJ
```

Resposta sugerida:

```json
{
  "items": [
    {
      "code": 3304557,
      "name": "Rio de Janeiro",
      "stateCode": 33,
      "state": "RJ"
    }
  ]
}
```

A pesquisa deve ter o mesmo debounce, ordenação e suporte para mouse e teclado
do autocomplete de ocupações. O usuário verá o nome e a UF, enquanto o frontend
guardará os códigos.

## 5. Região de saúde

### Endpoint

```http
GET /api/v1/references/health-regions?municipality=3304557
```

Resposta sugerida:

```json
{
  "items": [
    {
      "code": 33005,
      "name": "Metropolitana I",
      "state": "RJ"
    }
  ]
}
```

Após a seleção do município, o frontend deve preencher a região de saúde
automaticamente. Caso exista mais de uma opção válida, o usuário deverá
selecionar pelo nome.

## Campos calculados automaticamente

Os campos abaixo não precisam de endpoints próprios nem devem ser digitados
manualmente:

- semana epidemiológica, calculada pela data dos primeiros sintomas;
- dias até a notificação, calculados pela diferença entre as datas;
- código CBO;
- código IBGE da UF;
- código IBGE do município;
- código da região de saúde.
