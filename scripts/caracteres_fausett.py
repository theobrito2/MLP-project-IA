# Trabalho de IA - Multilayer Perceptron (MLP)
# Integrantes:
#   - Theo Djrdjrjan Brito - No USP: 13688367
#   - Nome Completo - No USP: 00000000
#
# Teste de corretude com os caracteres da Fausett (7 letras x 3 fontes,
# grade 9x7 = 63 pixels, rotulo one-hot de 7 posicoes). Treina nos limpos
# e testa em limpo/ruido/ruido20 para verificar a generalizacao da MLP.

import os
import sys

# Localiza src/ e data/ a partir da raiz do projeto.
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))
DADOS = os.path.join(RAIZ, "data", "caracteres_fausett")
RESULTADOS = os.path.join(RAIZ, "results", "caracteres_fausett")
os.makedirs(RESULTADOS, exist_ok=True)

from mlp import Mlp
import random
import numpy as np
import pandas as pd

# ==========================================
# REPRODUTIBILIDADE E HIPERPARAMETROS
# ==========================================
SEED = 42
MAX_EPOCAS = 5000
PACIENCIA = 50          # parada antecipada: epocas sem melhora toleradas
TOLERANCIA = 1e-5
TAXA_APRENDIZADO = 0.2
NUM_OCULTOS = 15        # uma unica camada escondida

random.seed(SEED)
np.random.seed(SEED)

# Nomes das 7 classes do conjunto da Fausett (uma posicao do one-hot cada).
LETRAS = ['A', 'B', 'C', 'D', 'E', 'J', 'K']
NUM_ENTRADAS = 63       # grade 9x7
NUM_SAIDAS = len(LETRAS)  # 7


# ==========================================
# CARREGAMENTO E PRE-PROCESSAMENTO DOS DADOS
# ==========================================
def carregar(nome_arquivo):
    # 63 primeiras colunas = pixels (entrada), 7 ultimas = rotulo one-hot.
    # Converte de {-1,1} para [0,1] (faixa da sigmoide).
    dados = pd.read_csv(os.path.join(DADOS, nome_arquivo), header=None).values.astype(float)

    X = (dados[:, :NUM_ENTRADAS] + 1) / 2
    Y = (dados[:, NUM_ENTRADAS:] + 1) / 2

    X = [list(linha) for linha in X]
    Y = [list(linha) for linha in Y]
    return X, Y


X_train, Y_train = carregar("caracteres-limpo.csv")
X_ruido, Y_ruido = carregar("caracteres-ruido.csv")
X_ruido20, Y_ruido20 = carregar("caracteres_ruido20.csv")


# ==========================================
# FUNCOES AUXILIARES
# ==========================================
def prever_classe(saida):
    # Discretizacao: classe = neuronio de maior ativacao.
    idx = saida.index(max(saida))
    return LETRAS[idx]


def classe_real(vetor_onehot):
    idx = vetor_onehot.index(max(vetor_onehot))
    return LETRAS[idx]


def erro_medio(modelo, X_dados, Y_dados):
    # MSE sobre um conjunto (so feedforward).
    erro_total = 0
    for i in range(len(X_dados)):
        pred = modelo.feedforward(X_dados[i])
        erro_total += sum((Y_dados[i][j] - pred[j]) ** 2 for j in range(modelo.num_saidas))
    return erro_total / len(X_dados)


def avaliar(modelo, X_dados, Y_dados, nome):
    # Roda a rede no conjunto e imprime acuracia (acertos de classe).
    acertos = 0
    for i in range(len(X_dados)):
        pred = modelo.feedforward(X_dados[i])
        if prever_classe(pred) == classe_real(Y_dados[i]):
            acertos += 1
    print(f"  {nome:<22} acuracia: {acertos}/{len(X_dados)} "
          f"({100 * acertos / len(X_dados):.1f}%)")
    return acertos / len(X_dados)


# ==========================================
# MODELO E TREINAMENTO
# ==========================================
# MLP de 1 camada escondida, treinada por Gradiente Descendente
# (backpropagation), minimizando o MSE.
modelo = Mlp(TAXA_APRENDIZADO, NUM_ENTRADAS, NUM_OCULTOS, NUM_SAIDAS)

print("=" * 56)
print(" MLP - Caracteres da Fausett (teste de corretude)")
print("=" * 56)
print(f"Arquitetura: {NUM_ENTRADAS} entradas -> {NUM_OCULTOS} ocultos -> "
      f"{NUM_SAIDAS} saidas | lr={TAXA_APRENDIZADO}")
print(f"Treino: {len(X_train)} caracteres limpos\n")

print("--- Treinamento (MSE por epoca) ---")

melhor_erro = float('inf')
melhor_pesos = modelo.copiar_pesos()
epocas_sem_melhora = 0
historico = []
epoca = 0

for epoca in range(MAX_EPOCAS):

    # embaralha a ordem de apresentacao dos exemplos a cada epoca
    indices = list(range(len(X_train)))
    random.shuffle(indices)

    erro_total = 0
    for i in indices:
        x = X_train[i]
        y = Y_train[i]

        pred = modelo.feedforward(x)
        erro_total += sum((y[j] - pred[j]) ** 2 for j in range(modelo.num_saidas))
        modelo.backprop(x, y)

    mse = erro_total / len(X_train)
    historico.append(mse)

    # Parada antecipada sobre o MSE de treino (conjunto pequeno, sem validacao),
    # guardando os melhores pesos para restaura-los ao final.
    if mse < melhor_erro - TOLERANCIA:
        melhor_erro = mse
        melhor_pesos = modelo.copiar_pesos()
        epocas_sem_melhora = 0
    else:
        epocas_sem_melhora += 1

    if epoca % 200 == 0:
        print(f"  Epoca {epoca:5d} | MSE: {mse:.5f} | Sem melhora: {epocas_sem_melhora}")

    if epocas_sem_melhora >= PACIENCIA:
        print(f"  Parada antecipada na epoca {epoca} "
              f"(MSE estagnado por {PACIENCIA} epocas).")
        break

modelo.restaurar_pesos(melhor_pesos)
print(f"  Treino concluido em {epoca + 1} epocas | MSE final: {melhor_erro:.5f}")

# ==========================================
# TESTES (corretude + robustez a ruido)
# ==========================================
print("\n--- Teste (acuracia por conjunto) ---")
avaliar(modelo, X_train, Y_train, "Limpo (=treino)")
avaliar(modelo, X_ruido, Y_ruido, "Com ruido")
avaliar(modelo, X_ruido20, Y_ruido20, "Com ruido 20%")

# Detalhamento por exemplo no conjunto com ruido (mostra a discretizacao).
print("\n--- Detalhe (conjunto com ruido) ---")
for i in range(len(X_ruido)):
    pred = modelo.feedforward(X_ruido[i])
    p = prever_classe(pred)
    r = classe_real(Y_ruido[i])
    marca = "OK " if p == r else "ERR"
    print(f"  [{marca}] esperado={r} previsto={p} "
          f"| ativacao max={max(pred):.3f}")

# Salva as saidas do teste com ruido (artefato util).
with open(os.path.join(RESULTADOS, "saidas_teste.txt"), "w") as f:
    for i in range(len(X_ruido)):
        pred = modelo.feedforward(X_ruido[i])
        f.write(f"Teste {i}\n")
        f.write(f"Esperado: {classe_real(Y_ruido[i])}\n")
        f.write(f"Previsto: {prever_classe(pred)}\n")
        f.write(f"Saida bruta: {pred}\n\n")

print(f"\nSaidas salvas em: {os.path.join(RESULTADOS, 'saidas_teste.txt')}")
