# Classificação de estresse estudantil com MLP

Implementação manual de uma rede neural com **8 entradas, 16 neurônios ocultos e 3 saídas**. Usa o arquivo `StressLevelDataset.csv`, do [Student Stress Factors — Kaggle](https://www.kaggle.com/datasets/rxnach/student-stress-factors-a-comprehensive-analysis), com 1.100 registros e 20 atributos disponíveis. Licença indicada pelo publicador: Apache 2.0.

O alvo `stress_level` já contém as classes **0 = baixo, 1 = médio e 2 = alto**. O programa valida esses valores diretamente. As antigas faixas de 1 a 10 não são utilizadas.

| Entrada | Significado |
| --- | --- |
| `sleep_quality` | Qualidade do sono |
| `study_load` | Carga de estudos |
| `future_career_concerns` | Preocupação com o futuro profissional |
| `peer_pressure` | Pressão dos colegas |
| `social_support` | Apoio social |
| `bullying` | Bullying |
| `living_conditions` | Condições de vida |
| `teacher_student_relationship` | Relação entre professor e estudante |

Para executar na raiz do projeto, usando o ambiente virtual existente:

```bash
.venv/bin/python mlp_produtividade.py
```

Para criar um ambiente novo, use Python 3.10 ou superior:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install matplotlib networkx
.venv/bin/python mlp_produtividade.py
```

A configuração fica no final de `mlp_produtividade.py`: `seed(1)`, `n_folds = 5`, `l_rate = 0.1`, `n_epoch = 100`, `n_hidden = 16` e `show_plots = False`. Use `show_plots = True` para exibir a curva de erro e o desenho da rede em cada fold.

Cada rodada usa 880 registros para treinar uma nova rede e 220 para avaliar. A normalização é ajustada apenas no treino. As linhas `epoch` mostram o índice da época (0 a 99), a taxa de aprendizagem e a soma dos erros quadráticos do treino. `error` não é uma porcentagem. `Scores` mostra a acurácia de cada fold reservado; a última linha mostra a média.

Com essa configuração, a avaliação de referência obteve **86,273% de acurácia média**, com resultados por fold de 86,818%, 88,636%, 82,273%, 85,909% e 87,727%. É uma avaliação interna desta base. Há combinações repetidas das oito entradas; uma checagem adicional mantendo combinações iguais no mesmo conjunto obteve 91,27% em outra divisão. Esses percentuais não devem ser comparados diretamente, pois as divisões diferem.

A documentação pública não esclarece completamente a produção dos rótulos; o resultado não garante o mesmo desempenho em outras populações. O objetivo atual é classificar fatores de estresse estudantil.
