# MLP para classificação do estresse estudantil

Rede neural **MLP (perceptron multicamadas) 8–16–3** implementada do zero, em Python puro, que classifica o nível de estresse de estudantes em **baixo, médio ou alto**. O desempenho é medido por validação cruzada com 5 folds.

## Estrutura do projeto

```
.
├── mlp.py                  # núcleo da rede neural: arquitetura, treino e previsão
├── funcoes.py              # dados (CSV), normalização e validação cruzada
├── StressLevelDataset.csv  # dataset
└── README.md
```

## Requisitos

- **Python 3.10 ou superior** (o código usa a instrução `match`).
- Nenhuma biblioteca externa: só a biblioteca padrão (`random`, `math`, `csv`, `pathlib`).

## Como executar

Com os três arquivos (`mlp.py`, `funcoes.py` e `StressLevelDataset.csv`) na mesma pasta:

```bash
python mlp.py
```

Durante o treino, o programa imprime o erro de cada época. No final, mostra a acurácia de cada fold e a média.

## Dataset

`StressLevelDataset.csv` tem **1100 estudantes** e 21 colunas (20 fatores + `stress_level`).

| `stress_level` | Classe | Quantidade |
|---|---|---|
| 0 | Baixo | 373 |
| 1 | Médio | 358 |
| 2 | Alto | 369 |

As classes são equilibradas, então a acurácia é uma métrica adequada.

**Atributos usados como entrada (8):**

| Coluna | Significado |
|---|---|
| `sleep_quality` | qualidade do sono |
| `study_load` | carga de estudo |
| `future_career_concerns` | preocupação com a carreira |
| `peer_pressure` | pressão dos colegas |
| `social_support` | apoio social |
| `bullying` | bullying sofrido |
| `living_conditions` | condições de moradia |
| `teacher_student_relationship` | relação professor-aluno |

Os nomes ficam em inglês porque são os cabeçalhos do CSV. Para mudar os atributos, edite `COLUNAS_ENTRADA` em `funcoes.py`.

## Arquitetura da rede

```
8 entradas  →  16 neurônios ocultos  →  3 saídas (baixo, médio, alto)
```

- Ativação: **sigmoide** (a tangente hiperbólica também está implementada).
- Pesos iniciais aleatórios em [-0,5; 0,5), com um bias por neurônio.
- Total de **195 parâmetros** ajustáveis.
- A classe prevista é a saída de maior valor (argmax).

## Técnicas utilizadas

- **Normalização min-max** (0 a 1), calculada só com os dados de treino de cada fold, para evitar vazamento de dados.
- **Codificação one-hot** do alvo (classe 2 → `[0, 0, 1]`).
- **Backpropagation** com **gradiente descendente estocástico** (os pesos são atualizados a cada linha).
- **Validação cruzada k-fold** (5 folds), com uma rede nova e independente em cada rodada.
- **Semente fixa** (`seed(1)`) para resultados reproduzíveis.

## Fluxo do programa

1. `carregar_csv` lê e valida o arquivo.
2. `coluna_str_para_float` converte os atributos em números.
3. `dividir_validacao_cruzada` separa os dados em 5 folds aleatórios.
4. Para cada fold, `avaliar_algoritmo` usa 1 fold como teste e 4 como treino, e chama `retropropagacao`, que:
   - normaliza treino e teste;
   - cria a rede (`inicializar_rede`);
   - treina por 100 épocas (`treinar_rede`): propagar para frente → retropropagar o erro → atualizar os pesos;
   - prevê o fold de teste (`prever`).
5. `metrica_acuracia` calcula a acurácia de cada fold; o programa imprime a média.

## Funções por arquivo

**`funcoes.py`**
- Dados: `estresse_para_classe` (valida o rótulo 0, 1 ou 2), `carregar_csv` (lê o CSV pelos nomes das colunas e monta as linhas `[atributos..., classe]`), `coluna_str_para_float` (converte uma coluna de texto em números) e `coluna_str_para_int` (genérica, não usada neste projeto).
- Normalização: `dataset_minmax` e `normalizar_dataset`.
- Validação cruzada: `dividir_validacao_cruzada`, `avaliar_algoritmo` e `metrica_acuracia`.

**`mlp.py`**
- Rede: `inicializar_rede`, `ativar`, `transferir` (e variantes), `derivada_transferencia` (e variantes), `propagar_para_frente`.
- Aprendizado: `retropropagar_erro`, `atualizar_pesos`, `treinar_rede`.
- Previsão: `prever`.
- Função principal de cada rodada: `retropropagacao`, que usa a normalização de `funcoes.py`.

## Hiperparâmetros

Ficam no bloco `__main__` de `mlp.py`:

| Parâmetro | Valor |
|---|---|
| `n_folds` | 5 |
| `taxa_aprendizado` | 0,1 |
| `n_epocas` | 100 |
| `n_ocultos` | 16 |

Para usar a tangente hiperbólica, troque `funcao_transferencia = 'sigmoid'` por `'tanh'` no topo de `mlp.py`.

## Resultados

Com a configuração acima e `seed(1)`:

| Fold | Acurácia |
|---|---|
| 1 | 86,82% |
| 2 | 88,64% |
| 3 | 82,27% |
| 4 | 85,91% |
| 5 | 87,73% |
| **Média** | **86,27%** |

Um palpite aleatório acertaria cerca de 33% com 3 classes equilibradas.

## Limitações e melhorias possíveis

- Treino com número fixo de épocas, sem parada antecipada.
- Avaliação apenas por acurácia (faltam matriz de confusão, precisão e recall por classe).
- Hiperparâmetros não foram otimizados.
- Uso de 8 dos 20 atributos do dataset.
- Evoluções possíveis: ReLU/softmax, mini-batch, comparação com bibliotecas como scikit-learn.
