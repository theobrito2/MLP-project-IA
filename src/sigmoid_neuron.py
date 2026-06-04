import random
import math

class SigmoidNeuron:
    # Hiperparametros de INICIALIZACAO dos pesos (faixa da distribuicao
    # uniforme usada para sortear pesos e bias). Ficam como atributos de
    # classe para poderem ser registrados no arquivo de hiperparametros.
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

    # Aqui é calculado o valor da DERIVADA DA FUNÇÃO SIGMOID
    # SIG(X)' = SIG(X)(1-SIG(x)) 
    def derivada(self):
        return self.saida * (1 - self.saida)

    def forward(self, entradas):
        self.entrada = entradas
        self.zin = sum(w * x for w, x in zip(self.w, entradas)) + self.b
        self.saida = self.ativacao(self.zin)
        return self.saida