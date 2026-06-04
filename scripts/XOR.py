# Trabalho de IA - Multilayer Perceptron (MLP) - Porta logica XOR
# Integrantes:
#   - Theo Djrdjrjan Brito - No USP: 13688367
#   - Nome Completo - No USP: 00000000

import os
import sys

# Bootstrap de caminhos: localiza src/ e data/ a partir da raiz do projeto.
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))
DADOS = os.path.join(RAIZ, "data", "portas_logicas")

from mlp import Mlp
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# XOR nao e linearmente separavel e tem um longo plato inicial de erro,
# por isso lr maior (0.5) e paciencia generosa na parada antecipada.
modelo_xor = Mlp(0.5, 2, 4, 1)

xor_MLP = pd.read_csv(
    os.path.join(DADOS, "problemXOR.csv"),
    header=None
)

# Entradas
X = [[float(v) for v in row] for row in xor_MLP.iloc[:, :-1].values.tolist()]

# Saídas
Y = [float(val) for val in xor_MLP.iloc[:, -1].values.tolist()]  # type: ignore

for i in range(len(X)):
    for j in range(len(X[i])):

        if X[i][j] == -1:
            X[i][j] = 0

    if Y[i] == -1:
        Y[i] = 0

paciencia = 200   # generosa: evita parar durante o plato inicial do XOR
melhor_erro = float('inf')
epocas_sem_melhora = 0
tolerancia = 1e-4

for epoca in range(10000):

    indices = list(range(len(X)))
    random.shuffle(indices)

    erro_total = 0

    for i in indices:

        x = X[i]
        y = [Y[i]]

        pred = modelo_xor.feedforward(x)

        erro_total += (y[0] - pred[0])**2

        modelo_xor.backprop(x, y)
        
    mse = erro_total / len(X) # Erro quadrático médio

    # Lógica do Early Stopping
    if mse < melhor_erro - tolerancia:
        melhor_erro = mse
        epocas_sem_melhora = 0
    else:
        epocas_sem_melhora += 1

    if epoca % 100 == 0:
        print(f"Época {epoca} | MSE: {mse:.4f} | Sem melhora: {epocas_sem_melhora}")

    if epocas_sem_melhora >= paciencia:
        print(f"\nParada antecipada (Early Stopping) acionada na época {epoca}!")
        print(f"O erro não teve melhoras significativas nas últimas {paciencia} épocas.")
        break

print("\nTESTES:\n")

acertos = 0

for i in range(len(X)):

    pred = modelo_xor.feedforward(X[i])

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