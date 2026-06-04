# Trabalho de IA - Multilayer Perceptron (MLP)
# Integrantes:
#   - Theo Djrdjrjan Brito - No USP: 13688367
#   - Nome Completo - No USP: 00000000

import random
import math

# Neuronio sigmoide (referencia conceitual de um unico neuronio).
# A MLP em mlp.py implementa essa mesma logica em forma vetorizada.
class SigmoidNeuron:
    PESO_MIN = -0.5
    PESO_MAX = 0.5
    NOME_ATIVACAO = "Sigmoide (logistica)"

    def __init__(self, num_entradas):
        self.w = [random.uniform(self.PESO_MIN, self.PESO_MAX) for _ in range(num_entradas)]
        self.b = random.uniform(self.PESO_MIN, self.PESO_MAX)

        self.entrada = []
        self.zin = 0
        self.saida = 0

    def ativacao(self, x):
        return 1 / (1 + math.exp(-x))

    # Derivada da sigmoide: sig'(x) = sig(x) * (1 - sig(x))
    def derivada(self):
        return self.saida * (1 - self.saida)

    def forward(self, entradas):
        self.entrada = entradas
        self.zin = sum(w * x for w, x in zip(self.w, entradas)) + self.b
        self.saida = self.ativacao(self.zin)
        return self.saida
