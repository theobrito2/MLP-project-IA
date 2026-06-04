# Trabalho de IA - Multilayer Perceptron (MLP)
# Integrantes:
#   - Theo Djrdjrjan Brito - No USP: 13688367
#   - Nome Completo - No USP: 00000000

import numpy as np

# MLP com uma camada escondida, ativacao sigmoide, treinada por
# Backpropagation (Gradiente Descendente) minimizando o MSE.
# Versao vetorizada: cada linha das matrizes de peso e um neuronio.

PESO_MIN = -0.5
PESO_MAX = 0.5
NOME_ATIVACAO = "Sigmoide (logistica)"


def _sigmoide(x):
    return 1.0 / (1.0 + np.exp(-x))


class Mlp:
    def __init__(self, taxa_aprendizado, num_entradas, num_ocultos, num_saidas):
        self.lr = taxa_aprendizado

        self.num_entradas = num_entradas
        self.num_ocultos = num_ocultos
        self.num_saidas = num_saidas

        # Pesos (W) e bias (b) das camadas oculta (1) e de saida (2),
        # inicializados com distribuicao uniforme em [PESO_MIN, PESO_MAX].
        self.W1 = np.asarray(np.random.uniform(PESO_MIN, PESO_MAX, (num_ocultos, num_entradas)))
        self.b1 = np.asarray(np.random.uniform(PESO_MIN, PESO_MAX, num_ocultos))
        self.W2 = np.asarray(np.random.uniform(PESO_MIN, PESO_MAX, (num_saidas, num_ocultos)))
        self.b2 = np.asarray(np.random.uniform(PESO_MIN, PESO_MAX, num_saidas))

    def feedforward(self, entradas):
        x = np.asarray(entradas, dtype=float)
        self.entrada = x

        # z_in = W.x + b ; saida = sigmoide(z_in), camada a camada
        self.saidas_ocultas = _sigmoide(self.W1 @ x + self.b1)
        self.saida_final = _sigmoide(self.W2 @ self.saidas_ocultas + self.b2)

        return self.saida_final.tolist()

    def backprop(self, entradas, alvo):
        x = np.asarray(entradas, dtype=float)
        alvo = np.asarray(alvo, dtype=float)

        # Delta da saida: (alvo - saida) * sig'(saida), com sig'(s)=s(1-s)
        erro_saida = alvo - self.saida_final
        delta_saida = erro_saida * self.saida_final * (1 - self.saida_final)

        # Delta oculto: erro retropropagado (W2^T . delta_saida) * sig'
        erro_oculto = self.W2.T @ delta_saida
        delta_oculto = erro_oculto * self.saidas_ocultas * (1 - self.saidas_ocultas)

        # Regra delta generalizada: peso += lr * delta * entrada
        self.W2 += self.lr * np.outer(delta_saida, self.saidas_ocultas)
        self.b2 += self.lr * delta_saida
        self.W1 += self.lr * np.outer(delta_oculto, x)
        self.b1 += self.lr * delta_oculto

    def salvar_pesos(self, nome_arquivo):
        with open(nome_arquivo, "w") as f:

            f.write("=== CAMADA OCULTA ===\n\n")
            for i in range(self.num_ocultos):
                f.write(f"Neuronio Oculto {i}\n")
                f.write(f"Pesos: {self.W1[i].tolist()}\n")
                f.write(f"Bias: {self.b1[i]}\n\n")

            f.write("\n=== CAMADA SAIDA ===\n\n")
            for i in range(self.num_saidas):
                f.write(f"Neuronio Saida {i}\n")
                f.write(f"Pesos: {self.W2[i].tolist()}\n")
                f.write(f"Bias: {self.b2[i]}\n\n")

    def salvar_hiperparametros(self, nome_arquivo, seed=None,
                               max_epocas=None, paciencia=None, tolerancia=None):
        with open(nome_arquivo, "w") as f:
            f.write("=== HIPERPARAMETROS DA ARQUITETURA ===\n\n")
            f.write(f"Numero de Entradas: {self.num_entradas}\n")
            f.write(f"Numero de Neuronios Ocultos (1 camada escondida): {self.num_ocultos}\n")
            f.write(f"Numero de Neuronios de Saida: {self.num_saidas}\n")
            f.write(f"Funcao de Ativacao: {NOME_ATIVACAO}\n\n")

            f.write("=== HIPERPARAMETROS DE INICIALIZACAO / TREINAMENTO ===\n\n")
            f.write(f"Taxa de Aprendizado (Learning Rate): {self.lr}\n")
            f.write("Inicializacao dos Pesos: distribuicao uniforme em "
                    f"[{PESO_MIN}, {PESO_MAX}]\n")
            if seed is not None:
                f.write(f"Semente (seed) do gerador aleatorio: {seed}\n")
            if max_epocas is not None:
                f.write(f"Numero maximo de epocas: {max_epocas}\n")
            if paciencia is not None:
                f.write(f"Paciencia (parada antecipada): {paciencia}\n")
            if tolerancia is not None:
                f.write(f"Tolerancia (parada antecipada): {tolerancia}\n")
            f.write("\n")
