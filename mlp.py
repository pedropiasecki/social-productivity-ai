"""MLP 8–16–3 para classificar o estresse estudantil em baixo, médio ou alto.
"""

from random import seed
from random import random
import math
from pathlib import Path
from funcoes import (
    carregar_csv, coluna_str_para_float, dataset_minmax,
    normalizar_dataset, avaliar_algoritmo,
)

funcao_transferencia = 'sigmoid'

# Inicializa a rede com pesos no intervalo [-0.5, 0.5).
def inicializar_rede(n_entradas, n_ocultos, n_saidas):
    """Cria as camadas oculta e de saída da MLP (perceptron multicamadas)."""
    rede = list()
    camada_oculta = [{'pesos': [random() - 0.5 for i in range(n_entradas + 1)]} for i in range(n_ocultos)]
    rede.append(camada_oculta)
    camada_saida = [{'pesos': [random() - 0.5 for i in range(n_ocultos + 1)]} for i in range(n_saidas)]
    rede.append(camada_saida)
    return rede


# Calcula a ativação de um neurônio para uma entrada
def ativar(pesos, entradas):
    """Calcula a soma ponderada: bias + soma(peso * entrada)."""
    ativacao = pesos[-1]
    for i in range(len(pesos) - 1):
        ativacao += pesos[i] * entradas[i]
    return ativacao


# Transfere a ativação usando a função sigmoide
def transferir_sigmoide(ativacao):
    """Aplica a sigmoide, que transforma a ativação em um valor entre 0 e 1."""
    return 1.0 / (1.0 + math.exp(-ativacao))


# Transfere a ativação usando a função tanh
def transferir_tanh(ativacao):
    """Aplica a tangente hiperbólica: a saída fica entre -1 e 1."""
    return math.tanh(ativacao)


# Transfere a ativação
def transferir(ativacao):
    """Escolhe a ativação conforme a variável global funcao_transferencia."""
    match funcao_transferencia:
        case 'sigmoid':
            return transferir_sigmoide(ativacao)
        case 'tanh':
            return transferir_tanh(ativacao)
        case _:
            return transferir_sigmoide(ativacao)


# Derivada da função sigmoide
def derivada_sigmoide(saida):
    """Calcula a derivada da sigmoide a partir da saída já calculada."""
    return saida * (1.0 - saida)


# Derivada da função tanh
def derivada_tanh(saida):
    """Calcula a derivada da tangente hiperbólica: 1 - saída ao quadrado."""
    return 1.0 - saida ** 2


# Derivada da função de transferência
def derivada_transferencia(saida):
    """Seleciona a derivada correspondente à função de ativação."""
    match funcao_transferencia:
        case 'sigmoid':
            return derivada_sigmoide(saida)
        case 'tanh':
            return derivada_tanh(saida)
        case _:
            return derivada_sigmoide(saida)


# Propaga a entrada para frente até a saída da rede
def propagar_para_frente(rede, linha):
    """Executa a propagação para frente: entradas -> camada oculta -> saída."""
    entradas = linha

    for camada in rede:
        novas_entradas = []
        for neuronio in camada:
            ativacao = ativar(neuronio['pesos'], entradas)
            neuronio['saida'] = transferir(ativacao)
            novas_entradas.append(neuronio['saida'])
        entradas = novas_entradas

    return entradas


# Retropropaga o erro e o armazena nos neurônios
def retropropagar_erro(rede, esperado):
    """Percorre as camadas de trás para frente e armazena o delta de cada neurônio."""
    for i in reversed(range(len(rede))):
        camada = rede[i]
        erros = list()
        if i != len(rede) - 1:
            for j in range(len(camada)):
                erro = 0.0
                for neuronio in rede[i + 1]:
                    erro += (neuronio['pesos'][j] * neuronio['delta'])
                erros.append(erro)
        else:
            for j in range(len(camada)):
                neuronio = camada[j]
                erros.append(neuronio['saida'] - esperado[j])
        for j in range(len(camada)):
            neuronio = camada[j]
            neuronio['delta'] = erros[j] * derivada_transferencia(neuronio['saida'])


# Atualiza os pesos da rede com o erro
def atualizar_pesos(rede, linha, taxa_aprendizado):
    """Aplica a descida do gradiente: peso -= taxa * delta * entrada."""
    for i in range(len(rede)):
        entradas = linha[:-1]
        if i != 0:
            entradas = [neuronio['saida'] for neuronio in rede[i - 1]]
        for neuronio in rede[i]:
            for j in range(len(entradas)):
                neuronio['pesos'][j] -= taxa_aprendizado * neuronio['delta'] * entradas[j]
            neuronio['pesos'][-1] -= taxa_aprendizado * neuronio['delta']


# Faz uma previsão com a rede
def prever(rede, linha):
    """Retorna o índice da maior saída da rede (argmax)."""
    saidas = propagar_para_frente(rede, linha)
    return saidas.index(max(saidas))


# Treina a rede por um número fixo de épocas
def treinar_rede(rede, treino, taxa_aprendizado, epocas, n_saidas):
    """Treina por um número fixo de épocas, passando por todas as linhas."""
    for epoca in range(epocas):
        soma_erro = 0
        for linha in treino:
            saidas = propagar_para_frente(rede, linha)
            esperado = [0 for i in range(n_saidas)]
            esperado[linha[-1]] = 1
            soma_erro += sum([(esperado[i] - saidas[i]) ** 2 for i in range(len(esperado))])
            retropropagar_erro(rede, esperado)
            atualizar_pesos(rede, linha, taxa_aprendizado)
        print('>época=%d, taxa=%.3f, erro=%.3f' % (epoca, taxa_aprendizado, soma_erro))


# Algoritmo de retropropagação com gradiente descendente estocástico
def retropropagacao(treino, teste, taxa_aprendizado, n_epocas, n_ocultos):
    # Ajusta a normalização só com os dados de treino, preservando as linhas originais.
    """Integra preparação dos dados, criação, treinamento e previsão da MLP."""
    treino = [list(linha) for linha in treino]
    teste = [list(linha) for linha in teste]
    minmax = dataset_minmax(treino)
    normalizar_dataset(treino, minmax)
    normalizar_dataset(teste, minmax)

    n_entradas = len(treino[0]) - 1
    n_saidas = 3  # 0: Baixo, 1: Médio, 2: Alto.
    rede = inicializar_rede(n_entradas, n_ocultos, n_saidas)
    treinar_rede(rede, treino, taxa_aprendizado, n_epocas, n_saidas)

    previsoes = list()
    for linha in teste:
        previsao = prever(rede, linha)
        previsoes.append(previsao)
    return previsoes


if __name__ == '__main__':
    seed(1)
    nome_arquivo = Path(__file__).resolve().parent / 'StressLevelDataset.csv'
    dataset = carregar_csv(nome_arquivo)

    for i in range(len(dataset[0]) - 1):
        coluna_str_para_float(dataset, i)

    # As classes já são inteiros; a normalização ocorre dentro de cada fold.
    n_folds = 5
    taxa_aprendizado = 0.1
    n_epocas = 100
    n_ocultos = 16
    pontuacoes = avaliar_algoritmo(dataset, retropropagacao, n_folds, taxa_aprendizado, n_epocas, n_ocultos)
    print('Pontuações: %s' % pontuacoes)
    print('Acurácia média: %.3f%%' % (sum(pontuacoes) / float(len(pontuacoes))))