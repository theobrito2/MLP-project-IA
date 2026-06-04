# ============================================================
# Trabalho de Inteligencia Artificial - Multilayer Perceptron (MLP)
# CARACTERES COMPLETO
# ------------------------------------------------------------
# Integrantes do grupo (PREENCHER):
#   - Theo Djrdjrjan Brito - No USP: 13688367
#   - Nome Completo 2 - No USP: 0000000
# ============================================================

import os
import sys

# ------------------------------------------------------------
# BOOTSTRAP DE CAMINHOS
# Permite rodar o script de qualquer diretorio: localiza a raiz do
# projeto a partir deste arquivo e define onde estao o codigo-fonte
# (src/), os dados (data/) e onde gravar as saidas (results/).
# ------------------------------------------------------------
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))

DADOS = os.path.join(RAIZ, "data", "caracteres_completo")
RESULTADOS = os.path.join(RAIZ, "results", "caracteres_completo")
os.makedirs(RESULTADOS, exist_ok=True)

from mlp import Mlp
import random
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# SEED (reprodutibilidade) E CRITERIO DE PARADA
# ==========================================
# Hiperparametros de inicializacao/treinamento que serao registrados no
# arquivo de hiperparametros (alem dos que ficam na propria rede).

SEED = 42
MAX_EPOCAS = 1000
PACIENCIA = 20
TOLERANCIA = 1e-4

random.seed(SEED)
np.random.seed(SEED)

# ==========================================
# CARREGA DADOS
# ==========================================

# Carrega os pixels (cada linha = 1 caractere de 10x12 = 120 pixels).
X = np.genfromtxt(
    os.path.join(DADOS, "X.txt"),
    delimiter=","
)

# O arquivo X.txt termina cada linha com virgula, o que faz o genfromtxt
# criar uma coluna extra de NaN. Removemos essa coluna para ficar com 120
# atributos (10x12), compativeis com a camada de entrada da rede.
X = X[:, :-1]

# converte de [-1,1] para [0,1] (faixa adequada para a sigmoide)
X = (X + 1) / 2

# Carrega as letras (rotulos), ignorando eventuais linhas em branco
with open(os.path.join(DADOS, "Y_letra.txt")) as f:
    Y = [linha.strip() for linha in f if linha.strip()]

# ==========================================
# ONE HOT ENCODING
# ==========================================
# Cada letra (A..Z) vira um vetor de 26 posicoes com 1 na posicao da letra.

def letra_para_onehot(letra):

    vetor = [0] * 26

    indice = ord(letra) - ord('A')

    vetor[indice] = 1

    return vetor


Y_onehot = [letra_para_onehot(l) for l in Y]

# ==========================================
# DIVISAO TREINO / VALIDACAO / TESTE (HOLD OUT)
# ==========================================
# A base esta organizada como alfabetos completos repetidos (A..Z, A..Z, ...).
#   - TESTE:      ultimos 130 exemplos (5 alfabetos completos)
#   - VALIDACAO:  130 exemplos anteriores (usados na PARADA ANTECIPADA)
#   - TREINO:     o restante
# A validacao e um conjunto separado do treino, usado para decidir a hora
# de parar (early stopping) sem "olhar" para o conjunto de teste.

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
    # A letra prevista e a do neuronio de saida com maior ativacao.
    idx = saida.index(max(saida))
    return chr(idx + ord('A'))


def onehot_para_letra(vetor):
    idx = vetor.index(1)
    return chr(idx + ord('A'))


def erro_medio(modelo, X_dados, Y_dados):
    # Erro Quadratico Medio (MSE) do modelo sobre um conjunto de dados,
    # SEM atualizar os pesos (apenas feedforward).
    erro_total = 0
    for i in range(len(X_dados)):
        pred = modelo.feedforward(X_dados[i])
        erro_total += sum(
            (Y_dados[i][j] - pred[j]) ** 2
            for j in range(modelo.num_saidas)
        )
    return erro_total / len(X_dados)


def acuracia(modelo, X_dados, Y_dados):
    # Fracao de acertos do modelo (letra prevista == letra esperada).
    acertos = 0
    for i in range(len(X_dados)):
        pred = modelo.feedforward(X_dados[i])
        if prever_letra(pred) == onehot_para_letra(Y_dados[i]):
            acertos += 1
    return acertos / len(X_dados)


def treinar(modelo, X_tr, Y_tr, X_vl, Y_vl,
            max_epocas=1000, paciencia=20, tolerancia=1e-4, verbose=True):
    # ------------------------------------------------------------
    # Laco de treinamento por GRADIENTE DESCENDENTE com BACKPROPAGATION.
    # O erro minimizado e o ERRO QUADRATICO MEDIO (MSE).
    # A PARADA ANTECIPADA (early stopping) monitora o erro de VALIDACAO:
    # se ele nao melhora por "paciencia" epocas seguidas, o treino para.
    # ------------------------------------------------------------
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

            # acumula o erro quadratico do exemplo
            erro_total += sum(
                (y[j] - pred[j]) ** 2
                for j in range(modelo.num_saidas)
            )

            # retropropaga e atualiza os pesos
            modelo.backprop(x, y)

        mse_treino = erro_total / len(X_tr)
        mse_val = erro_medio(modelo, X_vl, Y_vl)

        historico_treino.append(mse_treino)
        historico_val.append(mse_val)

        # ----- Logica do Early Stopping (sobre o erro de VALIDACAO) -----
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
    # Gera uma VARIACAO AUTORAL do dado de teste invertendo pixels
    # aleatoriamente (0 -> 1 e 1 -> 0), simulando ruido na imagem.
    x_ruidoso = x.copy()
    for i in range(len(x_ruidoso)):
        if random.random() < taxa_ruido:
            x_ruidoso[i] = 1.0 - x_ruidoso[i]
    return x_ruidoso

# ==========================================
# BUSCA DE PARAMETROS (GRID SEARCH)
# ==========================================
# Treina varias combinacoes de (taxa de aprendizado x neuronios ocultos)
# usando POUCAS epocas e escolhe a melhor pela ACURACIA na VALIDACAO.
# Como a rede usa loops puros de Python (treino lento), a busca usa um
# grid pequeno e poucas epocas. Coloque EXECUTAR_BUSCA = False para pular.

EXECUTAR_BUSCA = True

def busca_parametros():
    taxas_teste = [0.05, 0.1]
    ocultos_teste = [32, 64]

    # (acuracia_val, lr, n_ocultos) - inicia com acuracia -1 para
    # garantir que a primeira combinacao testada sera registrada.
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

# ==========================================
# DEFINE HIPERPARAMETROS (via busca ou padrao)
# ==========================================

if EXECUTAR_BUSCA:
    melhor_lr, melhor_ocultos = busca_parametros()
else:
    # melhores valores encontrados empiricamente
    melhor_lr, melhor_ocultos = 0.05, 64

# ==========================================
# MODELO FINAL
# ==========================================

modelo = Mlp(melhor_lr, 120, melhor_ocultos, 26)

# salva os hiperparametros (arquitetura + inicializacao) e os pesos INICIAIS
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

# ==========================================
# SALVA ERRO POR EPOCA
# ==========================================

with open(os.path.join(RESULTADOS, "erro_epocas.txt"), "w") as f:
    for epoca, (e_tr, e_vl) in enumerate(zip(historico_erros, historico_val)):
        f.write(f"Epoca {epoca}: MSE treino={e_tr} | MSE val={e_vl}\n")

# ==========================================
# GRAFICO DO COMPORTAMENTO DO ERRO
# ==========================================
# Mostra a curva de erro (MSE) de treino e de validacao ao longo das epocas.

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
# TESTES (com variacao autoral: ruido)
# ==========================================
# Avalia a rede no conjunto de TESTE aplicando ruido aleatorio (variacao
# autoral) e monta a MATRIZ DE CONFUSAO (26x26): linha = letra real,
# coluna = letra prevista.

acertos = 0
matriz_confusao = np.zeros((26, 26), dtype=int)

with open(os.path.join(RESULTADOS, "saidas_teste.txt"), "w") as f:

    for i in range(len(X_test)):

        # Teste com conjunto ruidoso (variacao autoral)
        x_atual = adicionar_ruido(X_test[i], taxa_ruido=0.05)

        pred = modelo.feedforward(x_atual)

        letra_pred = prever_letra(pred)
        letra_real = onehot_para_letra(Y_test[i])

        # atualiza a matriz de confusao
        idx_real = ord(letra_real) - ord('A')
        idx_pred = ord(letra_pred) - ord('A')
        matriz_confusao[idx_real][idx_pred] += 1

        if letra_pred == letra_real:
            acertos += 1

        # salva saida do teste
        f.write(f"Teste {i}\n")
        f.write(f"Esperado: {letra_real}\n")
        f.write(f"Previsto: {letra_pred}\n")
        f.write(f"Saida bruta: {pred}\n")
        f.write("\n")

# ==========================================
# ACURACIA
# ==========================================

print(f"\nAcuracia (teste com ruido): {acertos}/{len(X_test)} "
      f"({100 * acertos / len(X_test):.1f}%)")

# salva pesos finais
modelo.salvar_pesos(os.path.join(RESULTADOS, "pesos_finais.txt"))

# ==========================================
# MATRIZ DE CONFUSAO (salva em texto e em imagem)
# ==========================================

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

# ==========================================
# VISUALIZACAO DE UM CARACTERE (exemplo)
# ==========================================

plt.figure()
img = X[0].reshape(10, 12)
plt.imshow(img, cmap='gray')
plt.colorbar()
plt.title(f"Exemplo de entrada: {Y[0]}")

# Exibe todas as figuras geradas
plt.show()
