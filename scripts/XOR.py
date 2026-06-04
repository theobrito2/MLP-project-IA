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
import pandas as pd


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
tolerancia = 1e-4
melhor_erro = float('inf')
melhor_pesos = modelo_xor.copiar_pesos()
epocas_sem_melhora = 0
epoca = 0

print("=" * 48)
print(" MLP - Porta logica XOR")
print("=" * 48)
print("Arquitetura: 2 entradas -> 4 ocultos -> 1 saida | lr=0.5\n")

print("--- Treinamento (MSE por epoca) ---")
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

    mse = erro_total / len(X)  # Erro quadratico medio

    # Parada antecipada com restauracao dos melhores pesos
    if mse < melhor_erro - tolerancia:
        melhor_erro = mse
        melhor_pesos = modelo_xor.copiar_pesos()
        epocas_sem_melhora = 0
    else:
        epocas_sem_melhora += 1

    if epoca % 500 == 0:
        print(f"  Epoca {epoca:5d} | MSE: {mse:.5f} | Sem melhora: {epocas_sem_melhora}")

    if epocas_sem_melhora >= paciencia:
        print(f"  Parada antecipada na epoca {epoca} "
              f"(sem melhora ha {paciencia} epocas).")
        break

modelo_xor.restaurar_pesos(melhor_pesos)
print(f"  Treino concluido em {epoca + 1} epocas | MSE final: {melhor_erro:.5f}")

print("\n--- Teste ---")
print(f"  {'Entrada':<10} {'Esperado':>8} {'Previsto':>9} {'Saida':>8}   Resultado")

acertos = 0

for i in range(len(X)):

    pred = modelo_xor.feedforward(X[i])
    saida = pred[0]
    previsto = 1 if saida >= 0.5 else 0
    real = int(Y[i])
    ok = previsto == real
    acertos += ok

    entrada = "[" + ", ".join(str(int(v)) for v in X[i]) + "]"
    print(f"  {entrada:<10} {real:>8} {previsto:>9} {saida:>8.4f}   {'OK' if ok else 'ERRO'}")

print(f"\nAcuracia: {acertos}/{len(X)} ({100 * acertos / len(X):.1f}%)")