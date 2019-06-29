import numpy as np


# Classe representando cada camanda da MLP
class Camada:
    def __init__(self, n_entradas, n_neuronios, tipo_funcao_ativacao=None, pesos=None, bias=None):
        self.n_neuronios = n_neuronios
        self.tipo_funcao_ativacao = tipo_funcao_ativacao
        self.pesos = pesos if pesos is not None else np.random.rand(n_entradas,
                                                                    n_neuronios)  # Se não passar Pesos, inicia randomicamente
        self.tipo_funcao_ativacao = tipo_funcao_ativacao
        self.bias = bias if bias is not None else np.random.rand(
            n_neuronios)  # Se não passar Bias, inicia randomicamente

    def funcao_soma(self, x):
        r = np.dot(x, self.pesos) + self.bias
        self.valor_ultima_funcao_ativacao = self.funcao_ativacao(r)
        return self.valor_ultima_funcao_ativacao

    def funcao_ativacao(self, r):
        # Se nenhuma funca de ativacao for escolhida
        # Linear
        if self.tipo_funcao_ativacao is None:
            return r

        # tanh
        if self.tipo_funcao_ativacao == 'tanh':
            return np.tanh(r)

        # sigmoid
        if self.tipo_funcao_ativacao == 'sigmoid':
            return 1 / (1 + np.exp(-r))

        # softmax
        if self.tipo_funcao_ativacao == 'softmax':
            r -= r.max()  # avoiding overflow
            r_exp = np.exp(r)
            r_row_sum = r_exp.sum(1)
            a = r_exp / r_row_sum.reshape((-1, 1))
            return a

        return r

    def funcao_ativacao_derivada(self, r):
        # Linear
        if self.tipo_funcao_ativacao is None:
            return r

        # tangente hiperbolica
        if self.tipo_funcao_ativacao == 'tanh':
            return 1 - r ** 2

        # sigmoide
        if self.tipo_funcao_ativacao == 'sigmoid':
            return r * (1 - r)

        # softmax
        if self.tipo_funcao_ativacao == 'softmax':
            return r  # Implementar

        return r


class CamadaRecorrente(Camada):

    def __init__(self, n_entradas, n_neuronios, tipo_funcao_ativacao=None, pesos=None, bias=None, atraso=False):
        super().__init__(n_entradas, n_neuronios, tipo_funcao_ativacao, pesos, bias)
        self.camada_recorrente = Camada(1, n_neuronios, tipo_funcao_ativacao, pesos, bias)
        self.valor_recorrente = []
        self.atraso = atraso

    def funcao_soma(self, x, treinamento=False):
        if treinamento:
            if self.valor_recorrente == []:
                self.valor_recorrente = np.zeros([len(x), self.n_neuronios])
            else:
                self.valor_recorrente = np.copy(self.valor_ultima_funcao_ativacao)
                # # Adicionar uma linha de 1`sno inicio
                # first_row = np.zeros(self.valor_ultima_funcao_ativacao.shape[1])
                # self.valor_recorrente = np.vstack([first_row, self.valor_recorrente])
                # # Remove a ultima linha
                # self.valor_recorrente = np.delete(self.valor_recorrente, -1, 0)

            r = np.dot(x, self.pesos) + self.bias
            if self.atraso:  # A recorrencia só ocorre na camada oculta
                # Recorrencia
                saida_ant = np.dot(self.valor_recorrente, self.camada_recorrente.pesos.T)
                # Somatoria (X[t]*W + S[t-1]*V) + bias
                r = r + saida_ant

            # Valor final, com a recorrencia
            self.valor_ultima_funcao_ativacao = self.funcao_ativacao(r)
            self.valor_recorrente = self.valor_ultima_funcao_ativacao
            return self.valor_ultima_funcao_ativacao
        else:
            r = np.dot(x, self.pesos) + self.bias
            self.valor_ultima_funcao_ativacao = self.funcao_ativacao(r)
            return self.funcao_ativacao(r)
