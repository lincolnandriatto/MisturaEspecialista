import numpy as np

def normalizacaoDados(X):
	media = mediaDados(X)
	desvio = desvioPadraoDados(X)
	sizeDados = X.shape[0]
	dadosNormalizados = np.zeros(sizeDados)
	for i in range(sizeDados):
		dadosNormalizados[i] = (X[i] - media) / desvio

	return dadosNormalizados

def mediaDados(X):
	sizeX = X.shape[0]
	somaTotal = 0
	for i in range(sizeX):
		somaTotal += X[i]

	return somaTotal / sizeX

def desvioPadraoDados(X):
	media = mediaDados(X)
	N = X.shape[0]
	desvioPadrao = 0
	for i in range(N):
		desvioPadrao += ((X[i] - media) ** 2)/N

	return desvioPadrao ** 0.5

def desnormalizacaoDados(Media, Desvio, DadosNormalizados):
	dadosDesnomalizados = np.zeros([DadosNormalizados.shape[0], 1])
	for i in range(DadosNormalizados.shape[0]):
		dadosDesnomalizados[i][0] = (DadosNormalizados[i] * Desvio) + Media;
	return dadosDesnomalizados


#Dados = np.array([[1], [2], [3], [4]])
#DadosNormalizados = normalizacaoDados(Dados);
#Media = mediaDados(Dados);
#DesvioPadrao = desvioPadraoDados(Dados);
#DadosDesnormalizados = desnormalizacaoDados(Media, DesvioPadrao[0], DadosNormalizados)
#print(" Normalizado ",DadosNormalizados)
#print('Dado dernormalizado', DadosDesnormalizados.T)

