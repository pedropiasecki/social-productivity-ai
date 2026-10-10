"""Funções de apoio da MLP: dados, normalização e validação cruzada.
"""

import math
from csv import DictReader
from random import randrange


# Oito entradas; stress_level já contém a classe e nunca entra como atributo.
COLUNAS_ENTRADA = (
    'sleep_quality', 'study_load', 'future_career_concerns',
    'peer_pressure', 'social_support', 'bullying',
    'living_conditions', 'teacher_student_relationship',
)


# Valida os rótulos já classificados no CSV, sem converter notas de 1 a 10.
def estresse_para_classe(valor):
    """Aceita somente 0 (baixo), 1 (médio) ou 2 (alto) e retorna um inteiro."""
    nivel = float(valor)
    if not nivel.is_integer() or not 0 <= nivel <= 2:
        raise ValueError('stress_level deve ser um inteiro entre 0 e 2')
    return int(nivel)


def carregar_csv(nome_arquivo):
    """Lê o CSV por nomes de coluna, em vez de depender da ordem original."""
    dataset = list()
    with open(nome_arquivo, 'r', newline='', encoding='utf-8-sig') as arquivo:
        leitor_csv = DictReader(arquivo)
        obrigatorias = set(COLUNAS_ENTRADA) | {'stress_level'}
        ausentes = obrigatorias - set(leitor_csv.fieldnames or [])
        if ausentes:
            raise ValueError(f'Colunas ausentes: {sorted(ausentes)}')
        for numero_linha, linha in enumerate(leitor_csv, start=2):
            try:
                entradas = [linha[coluna] for coluna in COLUNAS_ENTRADA]
                if any(valor is None or not valor.strip() for valor in entradas):
                    raise ValueError('Atributo de entrada vazio')
                alvo = estresse_para_classe(linha['stress_level'])
            except (ValueError, TypeError) as erro:
                raise ValueError(f'Linha {numero_linha}: {erro}') from erro
            # Mantém a convenção original: atributos seguidos da classe.
            dataset.append(entradas + [alvo])
    if not dataset:
        raise ValueError('O dataset está vazio')
    return dataset


# Converte uma coluna de texto para float
def coluna_str_para_float(dataset, coluna):
    """Converte uma coluna textual em números decimais, alterando o dataset."""
    for linha in dataset:
        linha[coluna] = float(linha[coluna].strip())
        if not math.isfinite(linha[coluna]):
            raise ValueError('Os atributos de entrada devem ser números finitos')


# Converte uma coluna de texto para inteiro
def coluna_str_para_int(dataset, coluna):
    """Função genérica da base que mapeia rótulos para inteiros consecutivos."""
    valores_classe = [linha[coluna] for linha in dataset]
    unicos = set(valores_classe)
    tabela = dict()
    for i, valor in enumerate(unicos):
        tabela[valor] = i
    for linha in dataset:
        linha[coluna] = tabela[linha[coluna]]
    return tabela


# Encontra os valores mínimo e máximo de cada coluna
def dataset_minmax(dataset):
    """Calcula mínimo e máximo por coluna usando zip para transpor as linhas."""
    return [[min(coluna), max(coluna)] for coluna in zip(*dataset)]


# Reescala as colunas do dataset para o intervalo 0-1
def normalizar_dataset(dataset, minmax):
    """Normaliza cada entrada com (valor - mínimo) / (máximo - mínimo)."""
    for linha in dataset:
        for i in range(len(linha) - 1):
            intervalo = minmax[i][1] - minmax[i][0]
            linha[i] = (linha[i] - minmax[i][0]) / intervalo if intervalo else 0.0


# Divide o dataset em k folds
def dividir_validacao_cruzada(dataset, n_folds):
    """Separa aleatoriamente os exemplos em n_folds grupos, sem reposição."""
    if not 2 <= n_folds <= len(dataset):
        raise ValueError('n_folds deve estar entre 2 e o tamanho do dataset')
    dataset_dividido = list()
    copia_dataset = list(dataset)
    tamanho_fold = int(len(dataset) / n_folds)
    resto = len(dataset) % n_folds
    for i in range(n_folds):
        fold = list()
        # Distribui as linhas que sobraram entre os primeiros folds.
        tamanho_atual = tamanho_fold + (1 if i < resto else 0)
        while len(fold) < tamanho_atual:
            indice = randrange(len(copia_dataset))
            fold.append(copia_dataset.pop(indice))
        dataset_dividido.append(fold)
    return dataset_dividido


# Calcula a porcentagem de acurácia
def metrica_acuracia(reais, previstos):
    """Calcula a acurácia: quantidade de acertos / total de exemplos * 100."""
    acertos = 0
    for i in range(len(reais)):
        if reais[i] == previstos[i]:
            acertos += 1
    return acertos / float(len(reais)) * 100.0


# Avalia um algoritmo usando validação cruzada
def avaliar_algoritmo(dataset, algoritmo, n_folds, *args):
    """Executa validação cruzada: um fold é teste e os demais são treino."""
    folds = dividir_validacao_cruzada(dataset, n_folds)
    pontuacoes = list()
    for fold in folds:
        conjunto_treino = list(folds)
        conjunto_treino.remove(fold)
        conjunto_treino = sum(conjunto_treino, [])
        conjunto_teste = list()
        for linha in fold:
            copia_linha = list(linha)
            conjunto_teste.append(copia_linha)
            copia_linha[-1] = None
        previstos = algoritmo(conjunto_treino, conjunto_teste, *args)
        reais = [linha[-1] for linha in fold]
        acuracia = metrica_acuracia(reais, previstos)
        pontuacoes.append(acuracia)
    return pontuacoes