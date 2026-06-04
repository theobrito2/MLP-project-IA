# Trabalho de IA - Multilayer Perceptron (MLP)
# Integrantes:
#   - Theo Djrdjrjan Brito - No USP: 13688367
#   - Nome Completo - No USP: 00000000
#
# TESTE DE MESA - conferencia da implementacao contra o material de apoio
# "Um exemplo numerico do funcionamento de uma rede MLP usando backpropagation"
# (Andre Paulino de Lima e Sarajane Marques Peres).
#
# O material resolve a mao uma rede 2 -> 3 -> 2 (sigmoide, alpha=0.5) para a
# entrada x=[1,1] e alvo t=[1,0], dando os valores de saida e os pesos apos
# UM passo de backpropagation. Aqui reproduzimos esses calculos com a nossa
# Mlp e comparamos com os valores esperados do material.

import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))

from mlp import Mlp
import numpy as np

# Tolerancia: o material arredonda os valores em 4 casas decimais.
TOL = 1e-3

# ------------------------------------------------------------
# Rede e pesos iniciais do material (Secao 3).
# No material, o bias e o peso v0j/w0k (coluna 0 da matriz). Na nossa Mlp o
# bias fica separado (b1/b2), entao mapeamos:
#   V (oculta): linha j = [v0j(bias), v1j, v2j]  -> b1[j]=v0j ; W1[j]=[v1j,v2j]
#   W (saida):  linha k = [w0k(bias), w1k, w2k, w3k] -> b2[k]=w0k ; W2[k]=[w1k,w2k,w3k]
# ------------------------------------------------------------
modelo = Mlp(0.5, 2, 3, 2)
modelo.b1 = np.array([-0.1, -0.1, 0.1])
modelo.W1 = np.array([[0.1, -0.1],
                      [0.1, 0.1],
                      [-0.1, -0.1]])
modelo.b2 = np.array([-0.1, 0.1])
modelo.W2 = np.array([[0.1, 0.0, 0.1],
                      [-0.1, 0.1, -0.1]])

x = [1, 1]
t = [1, 0]

# Acumula o resultado de todas as conferencias.
tudo_ok = True


def conferir(nome, obtido, esperado):
    global tudo_ok
    obtido = np.asarray(obtido, dtype=float)
    esperado = np.asarray(esperado, dtype=float)
    ok = np.all(np.abs(obtido - esperado) <= TOL)
    tudo_ok = tudo_ok and ok
    marca = "OK " if ok else "FALHOU"
    print(f"  [{marca}] {nome}")
    print(f"           obtido:   {[round(float(v), 4) for v in obtido.ravel()]}")
    print(f"           esperado: {[round(float(v), 4) for v in esperado.ravel()]}")


print("=" * 60)
print(" TESTE DE MESA - MLP (material de apoio)")
print("=" * 60)

# ------------------------------------------------------------
# 1) FORWARD STEP (Secao 3.1 do material)
# ------------------------------------------------------------
print("\n--- Forward step ---")
y = modelo.feedforward(x)
conferir("Saida da camada oculta z", modelo.saidas_ocultas, [0.4750, 0.5250, 0.4750])
conferir("Saida da rede y", y, [0.4988, 0.5144])

# ------------------------------------------------------------
# 2) BACKWARD STEP + AJUSTE DOS PESOS (Secoes 3.2 a 3.4 do material)
# ------------------------------------------------------------
modelo.backprop(x, t)

print("\n--- Pesos da camada de saida apos 1 passo ---")
conferir("Bias da saida (w01, w02)", modelo.b2, [-0.0373, 0.0358])
conferir("Pesos do neuronio de saida 1 (w11, w21, w31)", modelo.W2[0], [0.1298, 0.0329, 0.1298])
conferir("Pesos do neuronio de saida 2 (w12, w22, w32)", modelo.W2[1], [-0.1305, 0.0663, -0.1305])

print("\n--- Pesos da camada oculta apos 1 passo ---")
conferir("Bias da oculta (v01, v02, v03)", modelo.b1, [-0.0968, -0.1016, 0.1032])
conferir("Pesos do neuronio oculto 1 (v11, v21)", modelo.W1[0], [0.1032, -0.0968])
conferir("Pesos do neuronio oculto 2 (v12, v22)", modelo.W1[1], [0.0984, 0.0984])
conferir("Pesos do neuronio oculto 3 (v13, v23)", modelo.W1[2], [-0.0968, -0.0968])

print("\n" + "=" * 60)
if tudo_ok:
    print(" RESULTADO: SUCESSO - todos os valores conferem com o material.")
else:
    print(" RESULTADO: FALHA - ha divergencias acima da tolerancia.")
print("=" * 60)
