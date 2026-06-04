# Trabalho de IA - Multilayer Perceptron (MLP) - CARACTERES COMPLETO
# Integrantes:
#   - Theo Djrdjrjan Brito - No USP: 13688367
#   - Nome Completo - No USP: 00000000

import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))

DADOS = os.path.join(RAIZ, "data", "caracteres_completo")
RESULTADOS = os.path.join(RAIZ, "results", "caracteres_completo")
os.makedirs(RESULTADOS, exist_ok=True)

from mlp import Mlp
import random
import numpy as np
import matplotlib.pyplot as plt

# Reprodutibilidade e criterio de parada antecipada.
SEED = 42
MAX_EPOCAS = 1000
PACIENCIA = 20
TOLERANCIA = 1e-4

random.seed(SEED)
np.random.seed(SEED)

# ==========================================
# CARREGA DADOS
# ==========================================

# Cada linha = 1 caractere de 10x12 = 120 pixels.
X = np.genfromtxt(os.path.join(DADOS, "X.txt"), delimiter=",")

# Remove a coluna extra de NaN (virgula final de cada linha).
X = X[:, :-1]

# Converte de [-1,1] para [0,1] (faixa da sigmoide).
X = (X + 1) / 2

# Rotulos (letras), ignorando linhas em branco.
with open(os.path.join(DADOS, "Y_letra.txt")) as f:
    Y = [linha.strip() for linha in f if linha.strip()]

# ==========================================
# ONE HOT ENCODING
# ==========================================
# Cada letra (A..Z) vira um vetor de 26 posicoes com 1 na posicao da letra.
def letra_para_onehot(letra):
    vetor = [0] * 26
    vetor[ord(letra) - ord('A')] = 1
    return vetor


Y_onehot = [letra_para_onehot(l) for l in Y]

# ==========================================
# DIVISAO TREINO / VALIDACAO / TESTE (HOLD OUT)
# ==========================================
# Base = alfabetos completos repetidos (A..Z, A..Z, ...).
#   TESTE:     ultimos 130 (5 alfabetos)
#   VALIDACAO: 130 anteriores (usados na parada antecipada)
#   TREINO:    o restante
X_test = X[-130:]
Y_test = Y_onehot[-130:]

X_val = X[-260:-130]
Y_val = Y_onehot[-260:-130]

X_train = X[:-260]
Y_train = Y_onehot[:-260]

# ==========================================
# FUNCOES AUXILIARES
# ==========================================

def prever_letra(saida):
    # Discretizacao: a letra prevista e a do neuronio de maior ativacao.
    idx = saida.index(max(saida))
    return chr(idx + ord('A'))


def onehot_para_letra(vetor):
    idx = vetor.index(1)
    return chr(idx + ord('A'))


def erro_medio(modelo, X_dados, Y_dados):
    # MSE sobre um conjunto (so feedforward, sem atualizar pesos).
    erro_total = 0
    for i in range(len(X_dados)):
        pred = modelo.feedforward(X_dados[i])
        erro_total += sum(
            (Y_dados[i][j] - pred[j]) ** 2
            for j in range(modelo.num_saidas)
        )
    return erro_total / len(X_dados)


def acuracia(modelo, X_dados, Y_dados):
    acertos = 0
    for i in range(len(X_dados)):
        pred = modelo.feedforward(X_dados[i])
        if prever_letra(pred) == onehot_para_letra(Y_dados[i]):
            acertos += 1
    return acertos / len(X_dados)


def treinar(modelo, X_tr, Y_tr, X_vl, Y_vl,
            max_epocas=1000, paciencia=20, tolerancia=1e-4, verbose=True):
    # Treino por Gradiente Descendente com Backpropagation, minimizando o MSE.
    # Parada antecipada: monitora o MSE de validacao; se nao melhora por
    # "paciencia" epocas, interrompe o treino.
    historico_treino = []
    historico_val = []

    melhor_erro = float('inf')
    epocas_sem_melhora = 0

    for epoca in range(max_epocas):

        # embaralha a ordem de apresentacao dos exemplos a cada epoca
        indices = list(range(len(X_tr)))
        random.shuffle(indices)

        erro_total = 0

        for i in indices:
            x = X_tr[i]
            y = Y_tr[i]

            pred = modelo.feedforward(x)
            erro_total += sum((y[j] - pred[j]) ** 2 for j in range(modelo.num_saidas))
            modelo.backprop(x, y)

        mse_treino = erro_total / len(X_tr)
        mse_val = erro_medio(modelo, X_vl, Y_vl)

        historico_treino.append(mse_treino)
        historico_val.append(mse_val)

        # Parada antecipada (sobre o MSE de validacao)
        if mse_val < melhor_erro - tolerancia:
            melhor_erro = mse_val
            epocas_sem_melhora = 0
        else:
            epocas_sem_melhora += 1

        if verbose and epoca % 100 == 0:
            print(f"Epoca {epoca} | MSE treino: {mse_treino:.4f} | "
                  f"MSE val: {mse_val:.4f} | Sem melhora: {epocas_sem_melhora}")

        if epocas_sem_melhora >= paciencia:
            if verbose:
                print(f"\nParada antecipada acionada na epoca {epoca}!")
            break

    return historico_treino, historico_val


def adicionar_ruido(x, taxa_ruido=0.1):
    # Variacao autoral do teste: inverte pixels aleatoriamente (0<->1).
    x_ruidoso = x.copy()
    for i in range(len(x_ruidoso)):
        if random.random() < taxa_ruido:
            x_ruidoso[i] = 1.0 - x_ruidoso[i]
    return x_ruidoso

# ==========================================
# BUSCA DE PARAMETROS (GRID SEARCH)
# ==========================================
# Testa combinacoes de (taxa de aprendizado x neuronios ocultos) com poucas
# epocas e escolhe a melhor pela acuracia na validacao.
# EXECUTAR_BUSCA = False pula a busca e usa os valores padrao.
EXECUTAR_BUSCA = True

def busca_parametros():
    taxas_teste = [0.05, 0.1]
    ocultos_teste = [32, 64]

    melhor = (-1.0, taxas_teste[0], ocultos_teste[0])

    print("\n=== BUSCA DE PARAMETROS (GRID SEARCH) ===")
    for lr in taxas_teste:
        for n_ocultos in ocultos_teste:
            modelo_teste = Mlp(lr, 120, n_ocultos, 26)
            treinar(
                modelo_teste, X_train, Y_train, X_val, Y_val,
                max_epocas=25, paciencia=8, verbose=False
            )
            ac = acuracia(modelo_teste, X_val, Y_val)
            print(f"lr={lr} | ocultos={n_ocultos} | acuracia val={ac:.3f}")

            if ac > melhor[0]:
                melhor = (ac, lr, n_ocultos)

    print(f"Melhor configuracao: lr={melhor[1]} | "
          f"ocultos={melhor[2]} | acuracia val={melhor[0]:.3f}\n")
    return melhor[1], melhor[2]

if EXECUTAR_BUSCA:
    melhor_lr, melhor_ocultos = busca_parametros()
else:
    melhor_lr, melhor_ocultos = 0.05, 64

# ==========================================
# MODELO FINAL
# ==========================================
modelo = Mlp(melhor_lr, 120, melhor_ocultos, 26)

# Salva hiperparametros e pesos iniciais.
modelo.salvar_hiperparametros(
    os.path.join(RESULTADOS, "hiperparametros.txt"),
    seed=SEED,
    max_epocas=MAX_EPOCAS,
    paciencia=PACIENCIA,
    tolerancia=TOLERANCIA,
)
modelo.salvar_pesos(os.path.join(RESULTADOS, "pesos_iniciais.txt"))

# ==========================================
# TREINAMENTO FINAL
# ==========================================
historico_erros, historico_val = treinar(
    modelo, X_train, Y_train, X_val, Y_val,
    max_epocas=MAX_EPOCAS, paciencia=PACIENCIA, tolerancia=TOLERANCIA, verbose=True
)

# Erro (MSE) por epoca.
with open(os.path.join(RESULTADOS, "erro_epocas.txt"), "w") as f:
    for epoca, (e_tr, e_vl) in enumerate(zip(historico_erros, historico_val)):
        f.write(f"Epoca {epoca}: MSE treino={e_tr} | MSE val={e_vl}\n")

# Grafico do erro de treino e validacao.
plt.figure()
plt.plot(historico_erros, label="Treino")
plt.plot(historico_val, label="Validacao")
plt.xlabel("Epoca")
plt.ylabel("Erro Quadratico Medio (MSE)")
plt.title("Comportamento do erro durante o treinamento")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTADOS, "grafico_erro.png"))

# ==========================================
# TESTE (com ruido) + MATRIZ DE CONFUSAO
# ==========================================
acertos = 0
matriz_confusao = np.zeros((26, 26), dtype=int)

with open(os.path.join(RESULTADOS, "saidas_teste.txt"), "w") as f:

    for i in range(len(X_test)):

        x_atual = adicionar_ruido(X_test[i], taxa_ruido=0.05)

        pred = modelo.feedforward(x_atual)

        letra_pred = prever_letra(pred)
        letra_real = onehot_para_letra(Y_test[i])

        # Matriz de confusao: linha = letra real, coluna = letra prevista.
        matriz_confusao[ord(letra_real) - ord('A')][ord(letra_pred) - ord('A')] += 1

        if letra_pred == letra_real:
            acertos += 1

        f.write(f"Teste {i}\n")
        f.write(f"Esperado: {letra_real}\n")
        f.write(f"Previsto: {letra_pred}\n")
        f.write(f"Saida bruta: {pred}\n\n")

print(f"\nAcuracia (teste com ruido): {acertos}/{len(X_test)} "
      f"({100 * acertos / len(X_test):.1f}%)")

# Pesos finais.
modelo.salvar_pesos(os.path.join(RESULTADOS, "pesos_finais.txt"))

# Matriz de confusao (texto e imagem).
letras = [chr(c) for c in range(ord('A'), ord('Z') + 1)]

with open(os.path.join(RESULTADOS, "matriz_confusao.txt"), "w") as f:
    f.write("Matriz de Confusao (linha = real, coluna = previsto)\n\n")
    f.write("    " + " ".join(letras) + "\n")
    for i in range(26):
        linha = " ".join(f"{matriz_confusao[i][j]:>1}" for j in range(26))
        f.write(f"{letras[i]}:  {linha}\n")

plt.figure(figsize=(8, 7))
plt.imshow(matriz_confusao, cmap="Blues")
plt.colorbar(label="Quantidade")
plt.xticks(range(26), letras)
plt.yticks(range(26), letras)
plt.xlabel("Letra prevista")
plt.ylabel("Letra real")
plt.title("Matriz de Confusao - Conjunto de Teste")
plt.savefig(os.path.join(RESULTADOS, "matriz_confusao.png"))

# Visualizacao de um caractere de exemplo.
plt.figure()
img = X[0].reshape(10, 12)
plt.imshow(img, cmap='gray')
plt.colorbar()
plt.title(f"Exemplo de entrada: {Y[0]}")

plt.show()
