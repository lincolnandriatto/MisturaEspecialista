import numpy as np


class ManipulacaoDados:

    @staticmethod
    def carregar_arquivo(caminho_arquivo):
        content_array = []
        with open(caminho_arquivo) as f:
            # Content_list is the list that contains the read lines.
            for line in f:
                content_array.append(float(line.replace('\n', '')))
            print(content_array)
        return content_array

    @staticmethod
    def preparar_array_X_Y(serie, lag):
        serie_X = np.zeros(((len(serie) - lag), lag))
        serie_Y = np.zeros((len(serie) - lag))

        for index_X in range(lag, len(serie)):
            lag_conta = lag
            for item_X in range(lag):  # percorre de 0 até lag (tamanho do histórico considerado)
                serie_X[index_X - lag][item_X] = serie[index_X - lag_conta]
                lag_conta = lag_conta - 1
            serie_Y[index_X - lag] = serie[index_X]

        return serie_X, serie_
