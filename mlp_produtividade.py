"""MLP 8–16–3 para classificar o estresse estudantil em baixo, médio ou alto.

Usa oito atributos de StressLevelDataset.csv e avalia cinco redes independentes
por validação cruzada. As funções estão documentadas abaixo.
"""

from random import seed
from random import random
from random import randrange
import matplotlib.pyplot as plt
import networkx as nx
import math
from csv import DictReader
from pathlib import Path


# Oito entradas; stress_level já contém a classe e nunca entra como atributo.
INPUT_COLUMNS = (
    'sleep_quality', 'study_load', 'future_career_concerns',
    'peer_pressure', 'social_support', 'bullying',
    'living_conditions', 'teacher_student_relationship',
)
transfer_neuron_function = 'sigmoid'


# Initialize a network with weights in [-0.5, 0.5).
def initialize_network(n_inputs, n_hidden, n_outputs):
    """Cria as camadas oculta e de saída da MLP (perceptron multicamadas)."""
    network = list()
    hidden_layer = [{'weights': [random() - 0.5 for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() - 0.5 for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Calculate neuron activation for an input
def activate(weights, inputs):
    """Calcula a soma ponderada: bias + soma(peso * entrada)."""
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation using sigmoid function
def transfer_sigmoid(activation):
    """Aplica a sigmoide, que transforma a ativação em um valor entre 0 e 1."""
    return 1.0 / (1.0 + math.exp(-activation))


# Transfer neuron activation using tanh function
def transfer_tanh(activation):
    """Aplica a tangente hiperbólica: a saída fica entre -1 e 1."""
    return math.tanh(activation)


# Transfer neuron activation
def transfer(activation):
    """Escolhe a ativação conforme a variável global transfer_neuron_function."""
    match transfer_neuron_function:
        case 'sigmoid':
            return transfer_sigmoid(activation)
        case 'tanh':
            return transfer_tanh(activation)
        case _:
            return transfer_sigmoid(activation)


# Derivative of the sigmoid function
def transfer_derivative_sigmoid(output):
    """Calcula a derivada da sigmoide a partir da saída já calculada."""
    return output * (1.0 - output)


# Derivative of the tanh function
def transfer_derivative_tanh(output):
    """Calcula a derivada da tangente hiperbólica: 1 - saída ao quadrado."""
    return 1.0 - output ** 2


# Derivative of the transfer function
def transfer_derivative(output):
    """Seleciona a derivada correspondente à função de ativação."""
    match transfer_neuron_function:
        case 'sigmoid':
            return transfer_derivative_sigmoid(output)
        case 'tanh':
            return transfer_derivative_tanh(output)
        case _:
            return transfer_derivative_sigmoid(output)


# Forward propagate input to a network output
def forward_propagate(network, row):
    """Executa a propagação para frente: entradas -> camada oculta -> saída."""
    inputs = row

    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs

    return inputs


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    """Percorre as camadas de trás para frente e armazena o delta de cada neurônio."""
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - expected[j])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error
def update_weights(network, row, l_rate):
    """Aplica a descida do gradiente: peso -= taxa * delta * entrada."""
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= l_rate * neuron['delta']


# Make a prediction with a network
def predict(network, row):
    """Retorna o índice da maior saída da rede (argmax)."""
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


# Train a network for a fixed number of epochs
def train_network(network, train, learning_rate, epochs, n_outputs):
    """Treina por um número fixo de épocas, passando por todas as linhas."""
    errors = []
    for epoch in range(epochs):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, learning_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))
        errors.append(sum_error)
    return errors


def draw_network(network):
    """Desenha a rede com NetworkX e Matplotlib em uma figura própria."""
    plt.figure()
    G = nx.DiGraph()
    for i, layer in enumerate(network):
        for j, neuron in enumerate(layer):
            # The bias is a parameter, not another input neuron.
            for k, weight in enumerate(neuron['weights'][:-1]):
                from_node = (i - 1, k) if i > 0 else k
                to_node = (i, j)
                G.add_node(from_node, layer=i)
                G.add_node(to_node, layer=i + 1)
                G.add_edge(from_node, to_node, weight=weight)

    pos = nx.multipartite_layout(G, subset_key="layer", scale=10)

    nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', font_size=10, font_weight='bold',
            font_color='black', edge_color='gray', width=1.5, alpha=0.9, arrowsize=10)

    labels = nx.get_edge_attributes(G, 'weight')

    labels = {k: round(v, 4) for k, v in labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8, font_weight='bold', alpha=0.9, label_pos=0.2, rotate=False)
    plt.show()
    plt.close()


# Valida os rótulos já classificados no novo CSV, sem converter notas de 1 a 10.
def stress_to_class(value):
    """Aceita somente 0 (baixo), 1 (médio) ou 2 (alto) e retorna um inteiro."""
    level = float(value)
    if not level.is_integer() or not 0 <= level <= 2:
        raise ValueError('stress_level deve ser um inteiro entre 0 e 2')
    return int(level)


def load_csv(filename):
    """Lê o CSV por nomes de coluna, em vez de depender da ordem original."""
    dataset = list()
    with open(filename, 'r', newline='', encoding='utf-8-sig') as file:
        csv_reader = DictReader(file)
        required = set(INPUT_COLUMNS) | {'stress_level'}
        missing = required - set(csv_reader.fieldnames or [])
        if missing:
            raise ValueError(f'Colunas ausentes: {sorted(missing)}')
        for line_number, row in enumerate(csv_reader, start=2):
            try:
                inputs = [row[column] for column in INPUT_COLUMNS]
                if any(value is None or not value.strip() for value in inputs):
                    raise ValueError('Atributo de entrada vazio')
                target = stress_to_class(row['stress_level'])
            except (ValueError, TypeError) as error:
                raise ValueError(f'Linha {line_number}: {error}') from error
            # Keep the original convention: attributes followed by the class.
            dataset.append(inputs + [target])
    if not dataset:
        raise ValueError('O dataset está vazio')
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    """Converte uma coluna textual em números decimais, alterando o dataset."""
    for row in dataset:
        row[column] = float(row[column].strip())
        if not math.isfinite(row[column]):
            raise ValueError('Os atributos de entrada devem ser números finitos')


# Convert string column to integer
def str_column_to_int(dataset, column):
    """Função genérica da base que mapeia rótulos para inteiros consecutivos."""
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Find the min and max values for each column
def dataset_minmax(dataset):
    """Calcula mínimo e máximo por coluna usando zip para transpor as linhas."""
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    """Normaliza cada entrada com (valor - mínimo) / (máximo - mínimo)."""
    for row in dataset:
        for i in range(len(row) - 1):
            interval = minmax[i][1] - minmax[i][0]
            row[i] = (row[i] - minmax[i][0]) / interval if interval else 0.0


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    """Separa aleatoriamente os exemplos em n_folds grupos, sem reposição."""
    if not 2 <= n_folds <= len(dataset):
        raise ValueError('n_folds deve estar entre 2 e o tamanho do dataset')
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    remainder = len(dataset) % n_folds
    for i in range(n_folds):
        fold = list()
        # Distribute leftover rows among the first folds.
        current_size = fold_size + (1 if i < remainder else 0)
        while len(fold) < current_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    """Calcula a acurácia: quantidade de acertos / total de exemplos * 100."""
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    """Executa validação cruzada: um fold é teste e os demais são treino."""
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden, show_plots=True):
    # Fit normalization on training data only, preserving the original rows.
    """Integra preparação dos dados, criação, treinamento e previsão da MLP."""
    train = [list(row) for row in train]
    test = [list(row) for row in test]
    minmax = dataset_minmax(train)
    normalize_dataset(train, minmax)
    normalize_dataset(test, minmax)

    n_inputs = len(train[0]) - 1
    n_outputs = 3  # 0: Baixo, 1: Médio, 2: Alto.
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    errors = train_network(network, train, l_rate, n_epoch, n_outputs)

    if show_plots:
        plt.figure()
        plt.plot(errors)
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.show()
        plt.close()

        draw_network(network)

    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return predictions


if __name__ == '__main__':
    seed(1)
    filename = Path(__file__).resolve().parent / 'StressLevelDataset.csv'
    dataset = load_csv(filename)

    for i in range(len(dataset[0]) - 1):
        str_column_to_float(dataset, i)

    # Classes are already integers; normalization happens inside each fold.
    n_folds = 5
    l_rate = 0.1
    n_epoch = 100
    n_hidden = 16
    show_plots = False  # Use True para exibir o erro e a rede em cada fold.
    scores = evaluate_algorithm(
        dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden, show_plots
    )
    print('Scores: %s' % scores)
    print('Acurácia média: %.3f%%' % (sum(scores) / float(len(scores))))
