from sigmoid_neuron import SigmoidNeuron

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

    def salvar_hiperparametros(self, nome_arquivo, seed=None,
                               max_epocas=None, paciencia=None, tolerancia=None):
        # Salva os hiperparametros exigidos na especificacao, divididos em:
        #   (1) ARQUITETURA da rede (camadas e neuronios);
        #   (2) hiperparametros de INICIALIZACAO (faixa dos pesos, seed,
        #       ativacao e criterio de parada do treinamento).
        # Os parametros de treino (seed, max_epocas, paciencia, tolerancia)
        # sao passados pelo script de treino, pois nao pertencem a rede em si.

        # Le a faixa de inicializacao dos pesos e a ativacao a partir de um
        # neuronio qualquer (todos compartilham os mesmos atributos de classe).
        neuronio_ref = self.camada_oculta[0]

        with open(nome_arquivo, "w") as f:
            f.write("=== HIPERPARAMETROS DA ARQUITETURA ===\n\n")
            f.write(f"Numero de Entradas: {self.num_entradas}\n")
            f.write(f"Numero de Neuronios Ocultos (1 camada escondida): {self.num_ocultos}\n")
            f.write(f"Numero de Neuronios de Saida: {self.num_saidas}\n")
            f.write(f"Funcao de Ativacao: {neuronio_ref.NOME_ATIVACAO}\n\n")

            f.write("=== HIPERPARAMETROS DE INICIALIZACAO / TREINAMENTO ===\n\n")
            f.write(f"Taxa de Aprendizado (Learning Rate): {self.lr}\n")
            f.write("Inicializacao dos Pesos: distribuicao uniforme em "
                    f"[{neuronio_ref.PESO_MIN}, {neuronio_ref.PESO_MAX}]\n")
            if seed is not None:
                f.write(f"Semente (seed) do gerador aleatorio: {seed}\n")
            if max_epocas is not None:
                f.write(f"Numero maximo de epocas: {max_epocas}\n")
            if paciencia is not None:
                f.write(f"Paciencia (parada antecipada): {paciencia}\n")
            if tolerancia is not None:
                f.write(f"Tolerancia (parada antecipada): {tolerancia}\n")
            f.write("\n")

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
        # =========================================================
        # BACKPROPAGATION (Retropropagação do erro)
        # =========================================================
        # Calcula os termos de erro (deltas) de trás para frente.
        # Usa a Regra Delta Generalizada para atualizar pesos.

        # DELTAS DA SAÍDA
        deltas_saida = []

        for i, neuronio in enumerate(self.neuronios_saida):
            output = self.saida_final[i]
            # Erro residual = alvo - saída
            erro = alvo[i] - output
            # Termo de Informação de Erro (Delta) da Saída
            # Regra delta computada sobre o erro minimizando MSE (Mean Squared Error)
            delta = erro * neuronio.derivada()
            deltas_saida.append(delta)

        # DELTAS OCULTOS
        deltas_ocultos = []

        for j, neuronio in enumerate(self.camada_oculta):
            # O erro retropropagado para a camada oculta i é a soma (Delta_k * Peso_jk)
            erro = sum(self.neuronios_saida[i].w[j] * deltas_saida[i] for i in range(self.num_saidas))
            delta = erro * neuronio.derivada()
            deltas_ocultos.append(delta)

        # ATUALIZA SAÍDA
        for i, neuronio in enumerate(self.neuronios_saida):
            for j in range(len(neuronio.w)):
                # Novo Peso = Peso Anterior + Taxa Aprendizado * Delta * Entrada_do_Neuronio
                neuronio.w[j] += self.lr * deltas_saida[i] * self.saidas_ocultas[j]

            # Atualização do Bias 
            neuronio.b += self.lr * deltas_saida[i]

        # ATUALIZA OCULTA
        for i, neuronio in enumerate(self.camada_oculta):
            for j in range(len(neuronio.w)):
                neuronio.w[j] += self.lr * deltas_ocultos[i] * entradas[j]

            neuronio.b += self.lr * deltas_ocultos[i]