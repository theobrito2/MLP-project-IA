from mlp import Mlp
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# SEED
# ==========================================

random.seed(42)
np.random.seed(42)

# ==========================================
# MODELO
# ==========================================

modelo = Mlp(0.05, 120, 64, 26)

# salva pesos iniciais
modelo.salvar_pesos("pesos_iniciais.txt")

# ==========================================
# CARREGA DADOS
# ==========================================

# Carrega os pixels
X = np.genfromtxt(
    "/home/luccas-cruz/Projects/USP/IA/mlp/CARACTERES COMPLETO/X.txt",
    delimiter=","
)

# converte de [-1,1] para [0,1]
X = (X + 1) / 2

# Carrega as letras
with open(
    "/home/luccas-cruz/Projects/USP/IA/mlp/CARACTERES COMPLETO/Y_letra.txt"
) as f:

    Y = [linha.strip() for linha in f]

# ==========================================
# ONE HOT ENCODING
# ==========================================

def letra_para_onehot(letra):

    vetor = [0] * 26

    indice = ord(letra) - ord('A')

    vetor[indice] = 1

    return vetor


Y_onehot = [letra_para_onehot(l) for l in Y]

# ==========================================
# DIVISAO TREINO / TESTE
# ==========================================

X_train = X[:-130]
Y_train = Y_onehot[:-130]

X_test = X[-130:]
Y_test = Y_onehot[-130:]

# ==========================================
# TREINAMENTO
# ==========================================

historico_erros = []

for epoca in range(1000):

    indices = list(range(len(X_train)))

    random.shuffle(indices)

    erro_total = 0

    for i in indices:

        x = X_train[i]
        y = Y_train[i]

        pred = modelo.feedforward(x)

        erro_total += sum(
            (y[j] - pred[j])**2
            for j in range(26)
        )

        modelo.backprop(x, y)

    historico_erros.append(erro_total)

    if epoca % 100 == 0:
        print(f"Época {epoca} | Erro: {erro_total:.4f}")

# ==========================================
# SALVA ERRO POR ÉPOCA
# ==========================================

with open("erro_epocas.txt", "w") as f:

    for epoca, erro in enumerate(historico_erros):

        f.write(
            f"Época {epoca}: {erro}\n"
        )

# ==========================================
# FUNÇÕES AUXILIARES
# ==========================================

def prever_letra(saida):

    idx = saida.index(max(saida))

    return chr(idx + ord('A'))


def onehot_para_letra(vetor):

    idx = vetor.index(1)

    return chr(idx + ord('A'))

# ==========================================
# TESTES
# ==========================================

acertos = 0

with open("saidas_teste.txt", "w") as f:

    for i in range(len(X_test)):

        pred = modelo.feedforward(X_test[i])

        letra_pred = prever_letra(pred)

        letra_real = onehot_para_letra(Y_test[i])

        if letra_pred == letra_real:
            acertos += 1

        # salva saída do teste
        f.write(f"Teste {i}\n")

        f.write(
            f"Esperado: {letra_real}\n"
        )

        f.write(
            f"Previsto: {letra_pred}\n"
        )

        f.write(
            f"Saída bruta: {pred}\n"
        )

        f.write("\n")

# ==========================================
# ACURÁCIA
# ==========================================

print(f"Acurácia: {acertos}/{len(X_test)}")

# salva pesos finais
modelo.salvar_pesos("pesos_finais.txt")

# ==========================================
# VISUALIZAÇÃO
# ==========================================

img = X[0].reshape(10, 12)

plt.imshow(img, cmap='gray')

plt.colorbar()

plt.title(Y[0])

plt.show()