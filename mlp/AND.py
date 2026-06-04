from mlp import Mlp
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

modelo_and = Mlp(0.3, 2, 2, 1)

and_MLP = pd.read_csv(
    "portas logicas/problemAND.csv",
    header=None
)

# Entradas
X = and_MLP.iloc[:, :-1].values.tolist()

# Saídas
Y = and_MLP.iloc[:, -1].values.tolist()

for i in range(len(X)):
    for j in range(len(X[i])):

        if X[i][j] == -1:
            X[i][j] = 0

    if Y[i] == -1:
        Y[i] = 0

for epoca in range(10000):

    indices = list(range(len(X)))
    random.shuffle(indices)

    erro_total = 0

    for i in indices:

        x = X[i]
        y = [Y[i]]  # transforma em lista

        pred = modelo_and.feedforward(x)

        erro_total += (y[0] - pred[0])**2

        modelo_and.backprop(x, y)

    if epoca % 100 == 0:
        print(f"Época {epoca} | Erro: {erro_total:.4f}")

print("\nTESTES:\n")

acertos = 0

for i in range(len(X)):

    pred = modelo_and.feedforward(X[i])

    saida = pred[0]

    previsto = 1 if saida >= 0.5 else 0

    real = Y[i]

    print(f"Entrada: {X[i]}")
    print(f"Esperado: {real}")
    print(f"Previsto: {previsto}")
    print(f"Valor bruto: {saida:.4f}")
    print()

    if previsto == real:
        acertos += 1

print(f"Acurácia: {acertos}/{len(X)}")   