# Trabalho de IA - Multilayer Perceptron (MLP) - Porta logica OR
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
from metricas import avaliar_binario
import random
import pandas as pd

modelo_or = Mlp(0.3, 2, 2, 1)

or_MLP = pd.read_csv(
    os.path.join(DADOS, "problemOR.csv"),
    header=None
)

# Entradas
X = [[float(val) for val in row] for row in or_MLP.iloc[:, :-1].values.tolist()]

# Saídas
Y = [float(val) for val in or_MLP.iloc[:, -1].values.tolist()]  # type: ignore

for i in range(len(X)):
    for j in range(len(X[i])):

        if X[i][j] == -1:
            X[i][j] = 0

    if Y[i] == -1:
        Y[i] = 0

print("=" * 48)
print(" MLP - Porta logica OR")
print("=" * 48)
print("Arquitetura: 2 entradas -> 2 ocultos -> 1 saida | lr=0.3\n")

# Parada antecipada: como o conjunto e minimo (4 exemplos, sem validacao),
# monitora-se o MSE de treino e guarda-se o melhor modelo para restaura-lo.
PACIENCIA = 200
TOLERANCIA = 1e-4
melhor_erro = float('inf')
melhor_pesos = modelo_or.copiar_pesos()
epocas_sem_melhora = 0
epoca = 0

print("--- Treinamento (MSE por epoca) ---")
for epoca in range(10000):

    indices = list(range(len(X)))
    random.shuffle(indices)

    erro_total = 0

    for i in indices:

        x = X[i]
        y = [Y[i]]  # transforma em lista

        pred = modelo_or.feedforward(x)

        erro_total += (y[0] - pred[0])**2

        modelo_or.backprop(x, y)

    mse = erro_total / len(X)

    if mse < melhor_erro - TOLERANCIA:
        melhor_erro = mse
        melhor_pesos = modelo_or.copiar_pesos()
        epocas_sem_melhora = 0
    else:
        epocas_sem_melhora += 1

    if epoca % 1000 == 0:
        print(f"  Epoca {epoca:5d} | MSE: {mse:.5f} | Sem melhora: {epocas_sem_melhora}")

    if epocas_sem_melhora >= PACIENCIA:
        print(f"  Parada antecipada na epoca {epoca} (sem melhora ha {PACIENCIA} epocas).")
        break

modelo_or.restaurar_pesos(melhor_pesos)
print(f"  Treino concluido em {epoca + 1} epocas | MSE final: {melhor_erro:.5f}")

print("\n--- Teste ---")
print(f"  {'Entrada':<10} {'Esperado':>8} {'Previsto':>9} {'Saida':>8}   Resultado")

reais = []
previstos = []

for i in range(len(X)):

    pred = modelo_or.feedforward(X[i])
    saida = pred[0]
    previsto = 1 if saida >= 0.5 else 0
    real = int(Y[i])
    reais.append(real)
    previstos.append(previsto)

    entrada = "[" + ", ".join(str(int(v)) for v in X[i]) + "]"
    print(f"  {entrada:<10} {real:>8} {previsto:>9} {saida:>8.4f}   {'OK' if previsto == real else 'ERRO'}")

print("\n--- Avaliacao (classificacao binaria) ---")
avaliar_binario(reais, previstos)
