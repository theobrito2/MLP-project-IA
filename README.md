# Trabalho de IA — Multilayer Perceptron (MLP)

Implementação de uma rede neural **Multilayer Perceptron** com **uma camada escondida**,
treinada por **Backpropagation (Gradiente Descendente)**, **sem usar bibliotecas de
redes neurais**. O erro minimizado é o **Erro Quadrático Médio (MSE)** e há **parada
antecipada** baseada no erro de validação.

## Integrantes do grupo

- Theo Djrdjrjan Brito — Nº USP: 13688367
- *(preencher os demais integrantes)*

## Estrutura do projeto

```
MLP-project-IA/
├── src/                       # Código-fonte da rede (reutilizável)
│   ├── mlp.py                 # Classe Mlp: feedforward, backprop, persistência
│   └── sigmoid_neuron.py      # Classe SigmoidNeuron: ativação e derivada
│
├── scripts/                   # Experimentos executáveis (um por conjunto de dados)
│   ├── main.py                # Experimento principal (CARACTERES COMPLETO: treino/teste + artefatos)
│   ├── caracteres_fausett.py  # Teste de corretude — caracteres da Fausett (limpo/ruído)
│   ├── AND.py                 # Teste de corretude — porta lógica AND
│   ├── OR.py                  # Teste de corretude — porta lógica OR
│   └── XOR.py                 # Teste de corretude — porta lógica XOR
│
├── data/                      # Conjuntos de dados (entrada)
│   ├── portas_logicas/        # problemAND.csv, problemOR.csv, problemXOR.csv
│   ├── caracteres_completo/   # X.txt, Y_letra.txt (+ X.npy, Y_classe.npy, X_png/)
│   └── caracteres_fausett/    # caracteres da Fausett (limpo e com ruído)
│
├── results/                   # Saídas geradas pelos scripts (ignorado no git)
├── .gitignore
└── README.md
```

> Os scripts localizam `src/` e `data/` automaticamente (independente do diretório
> de onde forem executados) e gravam todas as saídas em `results/`.

## Como executar

Pré-requisitos: Python 3 com `numpy`, `pandas` e `matplotlib`.

```bash
pip install numpy pandas matplotlib

# Experimento principal (CARACTERES COMPLETO):
python scripts/main.py

# Testes de corretude (portas lógicas):
python scripts/AND.py
python scripts/OR.py
python scripts/XOR.py

# Teste de corretude (caracteres da Fausett — treino limpo, teste com ruído):
python scripts/caracteres_fausett.py
```

## Artefatos gerados

O `scripts/main.py` grava em `results/caracteres_completo/`:

| Arquivo | Conteúdo |
| --- | --- |
| `hiperparametros.txt` | Arquitetura + hiperparâmetros de inicialização/treino |
| `pesos_iniciais.txt` | Pesos e bias antes do treinamento |
| `pesos_finais.txt` | Pesos e bias após o treinamento |
| `erro_epocas.txt` | MSE de treino e validação a cada época |
| `saidas_teste.txt` | Saída da rede para cada dado de teste |
| `matriz_confusao.txt` / `.png` | Matriz de confusão (real × previsto) |
| `grafico_erro.png` | Curva de erro (treino e validação) |

## Observações

- **CARACTERES COMPLETO** é o conjunto usado para treino/teste do trabalho.
- **Portas lógicas** e **CARACTERES (Fausett)** servem apenas como teste rápido de
  corretude da implementação. O script `scripts/caracteres_fausett.py` treina nos
  caracteres limpos e testa nas versões com ruído (`data/caracteres_fausett/`).
- A busca de parâmetros (grid search) pode ser desligada em
  `scripts/main.py` via `EXECUTAR_BUSCA = False`.
