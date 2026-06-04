from SigmoidNeuron import SigmoidNeuron

class Mlp:
    def __init__(self, taxa_aprendizado, num_entradas, num_ocultos, num_saidas):
        self.lr = taxa_aprendizado

        self.num_entradas = num_entradas
        self.num_ocultos = num_ocultos
        self.num_saidas = num_saidas

        self.camada_oculta = [SigmoidNeuron(self.num_entradas) for _ in range(self.num_ocultos)]
        self.neuronios_saida = [SigmoidNeuron(self.num_ocultos) for _ in range(self.num_saidas)]

    def salvar_pesos(self, nome_arquivo):

        with open(nome_arquivo, "w") as f:

            f.write("=== CAMADA OCULTA ===\n\n")

            for i, neuronio in enumerate(self.camada_oculta):

                f.write(f"Neuronio Oculto {i}\n")
                f.write(f"Pesos: {neuronio.w}\n")
                f.write(f"Bias: {neuronio.b}\n\n")

            f.write("\n=== CAMADA SAIDA ===\n\n")

            for i, neuronio in enumerate(self.neuronios_saida):

                f.write(f"Neuronio Saida {i}\n")
                f.write(f"Pesos: {neuronio.w}\n")
                f.write(f"Bias: {neuronio.b}\n\n")

    def salvar_hiperparametros(self, nome_arquivo):

        with open(nome_arquivo, "w") as f:

            f.write("=== Hip ===\n\n")

            for i, neuronio in enumerate(self.camada_oculta):

                f.write(f"Neuronio Oculto {i}\n")
                f.write(f"Pesos: {neuronio.w}\n")
                f.write(f"Bias: {neuronio.b}\n\n")
            

    def feedforward(self, entradas):
        self.saidas_ocultas = [n.forward(entradas) for n in self.camada_oculta]
        self.saida_final = [n.forward(self.saidas_ocultas) for n in self.neuronios_saida]
        return self.saida_final

    def letra_para_onehot(self, letra):
        vetor = [0] * 26
        indice = ord(letra) - ord('A')
        vetor[indice] = 1
        return vetor

    def backprop(self, entradas, alvo):
        # DELTAS DA SAÍDA
        deltas_saida = []

        for i, neuronio in enumerate(self.neuronios_saida):
            output = self.saida_final[i]
            erro = alvo[i] - output
            delta = erro * neuronio.derivada()
            deltas_saida.append(delta)

        # DELTAS OCULTOS
        deltas_ocultos = []

        for j, neuronio in enumerate(self.camada_oculta):
            erro = sum(self.neuronios_saida[i].w[j] * deltas_saida[i] for i in range(self.num_saidas))
            delta = erro * neuronio.derivada()
            deltas_ocultos.append(delta)

        # ATUALIZA SAÍDA
        for i, neuronio in enumerate(self.neuronios_saida):
            for j in range(len(neuronio.w)):
                neuronio.w[j] += self.lr * deltas_saida[i] * self.saidas_ocultas[j]

            neuronio.b += self.lr * deltas_saida[i]

        # ATUALIZA OCULTA
        for i, neuronio in enumerate(self.camada_oculta):
            for j in range(len(neuronio.w)):
                neuronio.w[j] += self.lr * deltas_ocultos[i] * entradas[j]

            neuronio.b += self.lr * deltas_ocultos[i]