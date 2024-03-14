import math

import pandas as pd


def entropy(attribute_data):
    total = len(attribute_data)
    true_count = sum(attribute_data)
    false_count = total - true_count
    if true_count == 0 or false_count == 0:
        return 0

    true_ratio = true_count / total
    false_ratio = false_count / total

    return -(true_ratio * math.log2(true_ratio) + false_ratio * math.log2(false_ratio))


def calculate_information_gain(data: pd.DataFrame, attribute_index: int) -> float:
    total_entropy = entropy(data["JogarTenis"])
    unique_values = data.iloc[:, attribute_index].unique()
    remainder = 0
    for value in unique_values:
        subset = data[data.iloc[:, attribute_index] == value]
        weight = len(subset) / len(data)
        remainder += weight * entropy(subset["JogarTenis"])

    return total_entropy - remainder


def find_best_attribute(data: pd.DataFrame) -> int:
    information_gains = []
    for i in range(1, len(data.columns) - 1):
        information_gains.append(calculate_information_gain(data, i))

    return information_gains.index(max(information_gains)) + 1


def construir_arvore(data: pd.DataFrame, attributes: list[str]) -> dict:
    if len(data["JogarTenis"].unique()) == 1:
        return data["JogarTenis"].iloc[0]

    if len(attributes) == 0:
        return data["JogarTenis"].mode()[0]

    best_attribute = find_best_attribute(data)

    arvore = {attributes[best_attribute - 1]: {}}

    unique_values = data.iloc[:, best_attribute].unique()
    for value in unique_values:
        subset = data[data.iloc[:, best_attribute] == value]
        subtree = construir_arvore(
            subset.drop(columns=[attributes[best_attribute - 1]]),
            attributes[: best_attribute - 1] + attributes[best_attribute:],
        )
        arvore[attributes[best_attribute - 1]][value] = subtree

    return arvore


def pretty(d: dict, indent: int = 0) -> None:
    for key, value in d.items():
        print(" " * indent + "|-" + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 4)
        else:
            print(" " * (indent + 4) + "->" + str(value))


if __name__ == "__main__":
    data = [
        ["Dia", "Perspectiva", "Temperatura", "Umidade", "Vento", "JogarTenis"],
        ["D1", "Ensolarado", "Quente", "Alta", "Fraco", False],
        ["D2", "Ensolarado", "Quente", "Alta", "Forte", False],
        ["D3", "Nublado", "Quente", "Alta", "Fraco", True],
        ["D4", "Chuvoso", "Moderada", "Alta", "Fraco", True],
        ["D5", "Chuvoso", "Fresca", "Normal", "Fraco", True],
        ["D6", "Chuvoso", "Fresca", "Normal", "Forte", False],
        ["D7", "Nublado", "Fresca", "Normal", "Forte", True],
        ["D8", "Ensolarado", "Moderada", "Alta", "Fraco", False],
        ["D9", "Ensolarado", "Fresca", "Normal", "Fraco", True],
        ["D10", "Chuvoso", "Moderada", "Normal", "Fraco", True],
        ["D11", "Ensolarado", "Moderada", "Normal", "Forte", True],
        ["D12", "Nublado", "Moderada", "Alta", "Forte", True],
        ["D13", "Nublado", "Quente", "Normal", "Fraco", True],
        ["D14", "Chuvoso", "Moderada", "Alta", "Forte", False],
    ]

    dataset = pd.DataFrame(data[1:], columns=data[0])

    # Construir árvore de decisão
    atributos = list(dataset.columns[1:-1])
    arvore = construir_arvore(dataset, atributos)

    # Exibir a árvore
    pretty(arvore, indent=2)
