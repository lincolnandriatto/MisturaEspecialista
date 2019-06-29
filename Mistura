import numpy as np
import NormalizacaoDados as nd
import SerieTemporal as st
import RedeNeural as rn
#import redeneuralME as rn_mlp_ME
import camadaRN as camada
import matplotlib.pyplot as plt
import auxiliar as manipulacao

# m = quantidade de especialistas
# Ntr = Numero de Dados de Treinamento
# ne = Numero de entradas da rede
# Yg = saída da Rede Gating
# W = pesos dos especialistas
# Wg = pesos da rede gating
def mistura(Xtr, Ytr, lista_config_especialistas):
    # Define o numero de entadas

    m = len(lista_config_especialistas)

    Ntr, ne = Xtr.shape  # numero de entradas
    ns = Ytr.shape[1]
    Xtr = np.append(Xtr, np.ones([Xtr.shape[0], 1]), axis=1)
    ne = ne + 1
    #####
 #   rn_mlp_ME.valid()
    ######
    # Inicializa Rede Gating
    Wg = np.random.rand(m, ne)

    # Inicializa Especialistas
    #W = np.zeros([m, ns, ne])
    W = np.zeros([m, ne, ne])
    variancia = np.ones([m, 1])
    for i in range(m):
        #W[i] = np.random.rand(ns, ne)
        W[i] = np.random.rand(ne, ne)
        variancia[i] = 1

    # Calcula a saida da Rede Gating
    cima = np.exp(np.dot(Xtr, Wg.T))
    baixo = (np.sum(np.exp(np.dot(Xtr, Wg.T)), 1))
    baixo = np.reshape(baixo, (len(baixo), 1))
    ones = np.ones([1, m])
    Yg = cima / baixo * ones
    # Yg = np.exp( np.dot(Xtr, Wg.T) ) / (np.sum( np.exp( np.dot(Xtr, Wg.T)), 1)) * np.ones([1, m])

    # Implementar aqui a MLP
    # Calcula a saida dos Especialistas
    # Modelo Linear
    Ye = np.zeros([m, Ntr, ns])
#    for i in range(m):
#        Ye[i] = np.dot(Xtr, W[i].T)

################################################################################

    especialistaLista = list()

    for i in range(m):
        # Cria a classe de NN

###########################################################

        config_rede_neural = lista_config_especialistas[i]

        if config_rede_neural.tipo_rede_neural == 'recorrente':

            nn_R = rn.RedeNeural_MLP().criar_rede_neural(config_rede_neural.tipo_rede_neural)
            # Definir camadas da Rede
            nn_R.adicionar_camada(camada.CamadaRecorrente(len(Xtr[0]), config_rede_neural.camada[0].num_neuronios, config_rede_neural.camada[0].funcao_ativacao, W[i].T, None, atraso=True))
            nn_R.adicionar_camada(camada.CamadaRecorrente(config_rede_neural.camada[1].num_entradas, config_rede_neural.camada[1].num_neuronios, config_rede_neural.camada[1].funcao_ativacao))

            mse, x = nn_R.treinar(Xtr, Ytr, 0.01, 5000)
            Ye[i] = nn_R.feed_forward(Xtr)

        elif config_rede_neural.tipo_rede_neural == 'batch':

            nn_R = rn.RedeNeural_MLP().criar_rede_neural(config_rede_neural.tipo_rede_neural)
            # Definir camadas da Rede
            nn_R.adicionar_camada(camada.CamadaRecorrente(len(Xtr[0]), ne, 'tanh', W[i].T, None, atraso=True))
            nn_R.adicionar_camada(camada.CamadaRecorrente(ne, 1, 'tanh'))

            mse, x = nn_R.treinar(Xtr, Ytr, 0.01, 5000)
            Ye[i] = nn_R.feed_forward(Xtr)


        #plt.plot(Ye[i])
        #plt.plot(Ytr)
        #plt.show()
        especialistaLista.append(nn_R)

##########################################################################################################

    # Calcula a saida da Mistura
    Ym = np.zeros([Ntr, ns])
    for i in range(m):
        Yge = Yg[:, i] * np.ones([1, ns])
        # Ym = Ym + Ye[i] * Yge.T
        Ym = Ym + Ye[i] * Yge

    # Calculo da função de Verossimilhança
    Py = np.zeros([Ntr, m])
    for i in range(m):
        Yaux = np.copy(Ye[i])

        for j in range(Ntr):  # Precorre os dados
            diff = Ytr[j] - Yaux[j]
            Py[j][i] = np.exp(np.dot(-diff, diff.T) / (2 * variancia[i]))

    Likelihood = np.sum(np.log(np.sum(Yg * Py, 1)))
    Likelihood_ant = 0
    nit = 0
    nitmax = 10

    while abs(Likelihood - Likelihood_ant) > 1e-3 and nit < nitmax:
        nit = nit + 1

        # Passo E
        # haux = Yg.*Py;
        haux = Yg * Py
        h_baixo = np.sum(haux, axis=1)
        h_baixo = np.reshape(h_baixo, (len(h_baixo), 1))
        h_baixo = np.dot(h_baixo, np.ones([1, m]))
        # h = haux./(sum(haux,2)*ones(1,m));
        h = haux / h_baixo

        # Passo M
        Wg = maximiza_gating(Wg, Xtr, m, h)

        for i in range(m):
            W[i], variancia[i] = maximiza_expert(W[i], variancia[i], Xtr, Ytr, h[:, i])

        # Calcula a Likelihood
        Likelihood_ant = Likelihood

        # Calcula a saida
        # Yg = exp(Xtr*Wg')./(sum(exp(Xtr*Wg'),2)*ones(1,m));
        Yg_a = np.exp(np.dot(Xtr, Wg.T))
        Yg_b = np.sum(np.exp(np.dot(Xtr, Wg.T)), 1)
        Yg_b = np.reshape(Yg_b, (len(Yg_b), 1))
        Yg_c = np.ones([1, m])
        Yg_d = np.dot(Yg_b, Yg_c)
        Yg = Yg_a / Yg_d

        # Implementar aqui a MLP
        #for i in range(m):
        #    # Ye{i}=Xtr*W{i}';
        #    Ye[i] = np.dot(Xtr, W[i].T)
        #####
        for i in range(len(especialistaLista)):
            # Cria a classe de NN

##############################################################

            nn_R = especialistaLista[i]
            #nn.camadas[0].pesos = W[i].T
            nn_R._camadas[0].pesos = W[i].T

            mse, x = nn_R.treinar(Xtr, Ytr, 0.01, 5000)
            Ye[i] = nn_R.feed_forward(Xtr)

            #plt.plot(Ye[i])
            #plt.plot(Ytr)
            #plt.show()

###############################################################

        ####

        # Ym=zeros(Ntr,ns);
        Ym = np.zeros([Ntr, ns])

        for i in range(m):
            # Yge=Yg(:,i)*ones(1,ns);
            Yge = Yg[:, i] * np.ones([1, ns])
            # Ym = Ym + Ye{i}.*Yge;
            Ym = Ym + Ye[i] * Yge

        # Cálculo da função de Verossimilhança
        for i in range(m):
            Yaux = Ye[i]
            for j in range(Ntr):
                # diff = Ytr(j,:)-Yaux(j,:);
                diff = Ytr[j, :] - Yaux[j, :]
                # Py(j,i)=exp(-(diff*diff')/(2*var(i)));
                Py[j][i] = np.exp(-np.dot(diff, diff.T) / (2 * variancia[i]))

        # Likelihood = sum(log(sum(Yg.*Py,2)))
        Likelihood = np.sum(np.log(np.sum(Yg * Py, 1)))

        # me.gating=Wg;
    rede_gating = Wg
    # me.expert.W=W;
    especialistas = W

    especialistas_variancia = np.copy(variancia)

    for i in range(m):
        especialistas_variancia[i] = variancia[i]

    return rede_gating, especialistas, especialistas_variancia, especialistaLista


def maximiza_gating(Wg, Xtr, m, h):
    N, ne = Xtr.shape
    Yg_denominador = (np.sum(np.exp(np.dot(Xtr, Wg.T)), axis=1))
    Yg_denominador = np.reshape(Yg_denominador, (len(Yg_denominador), 1))
    Yg_denominador = np.dot(Yg_denominador, np.ones([1, m]))
    Yg_numerador = np.dot(Xtr, Wg.T)
    Yg_numerador = np.exp(Yg_numerador)
    Yg = Yg_numerador / Yg_denominador
    grad_a = (h - Yg)
    grad_b = (Xtr / N)
    grad = np.dot(grad_a.T, grad_b)
    direcao = grad
    nit = 0
    nitmax = 10000
    alfa = 0.1

    # while norm(grad) > 1e-5 and nit < nitmax:
    while np.sqrt(np.sum(grad ** 2)) > 1e-5 and nit < nitmax:
        nit = nit + 1
        Wg = Wg + np.dot(alfa, direcao)
        Yg_numerador = np.dot(Xtr, Wg.T)
        Yg_numerador = np.exp(Yg_numerador)
        Yg_denominador = (np.sum(np.exp(np.dot(Xtr, Wg.T)), axis=1))
        Yg_denominador = np.reshape(Yg_denominador, (len(Yg_denominador), 1))
        Yg_denominador = np.dot(Yg_denominador, np.ones([1, m]))
        Yg = Yg_numerador / Yg_denominador
        grad_a = h - Yg
        grad_b = Xtr / N
        grad = np.dot(grad_a.T, grad_b)
        direcao = grad

        tmp_grad_norm = np.sqrt(np.sum(grad ** 2))

    return Wg


def maximiza_expert(W, variancia, Xtr, Ytr, h):
    Ye = np.dot(Xtr, W.T)
    N, ns = Ye.shape
    # ( (h * ones(1,ns) / variancia ).*(Ytr-Ye) )' * Xtr/N;
    h = np.reshape(h, (len(h), 1))
    grad_a = np.dot(h, np.ones([1, ns]))
    grad_a = grad_a / variancia
    grad_a = (grad_a) * (Ytr - Ye)
    grad_b = Xtr / N
    grad = np.dot(grad_a.T, grad_b)
    direcao = grad
    nit = 0
    nitmax = 10000
    alfa = 0.1

    while np.sqrt(np.sum(grad ** 2)) > 1e-5 and nit < nitmax:
        nit = nit + 1
        W = W + np.dot(alfa, direcao)
        Ye = np.dot(Xtr, W.T)  # Função SOMA
        # grad = ((h*ones(1,ns)/var).*(Ytr-Ye))'*Xtr/N;
        grad_a = np.dot(h, np.ones([1, ns]))
        grad_a = grad_a / variancia
        grad_a = grad_a * (Ytr - Ye)
        grad_b = Xtr / N
        grad = np.dot(grad_a.T, grad_b)
        direcao = grad

    diff = Ytr - Ye
    soma = 0

    for i in range(N):
        # soma = soma + h(i,1)*(diff(i,:)*diff(i,:)');
        val_1 = h[i, 0]
        val_2 = diff[i, :]
        val_2 = np.dot(val_2, val_2.T)
        val_final = np.dot(val_1, val_2)
        soma = soma + val_final

    # variancia = max(0.05, (1/ns) * soma/sum(h));
    max_esq = 0.05
    max_dir_a = (1 / ns)
    max_dir_b = soma / np.sum(h, axis=0)
    max_dir = np.dot(max_dir_a, max_dir_b)
    variancia = np.maximum(max_esq, max_dir)

    return W, variancia


def predizer(rede_gating, especialistas, especialistaLista, X, Y):
    # Adicionando BIAS
    X = np.append(X, np.ones([X.shape[0], 1]), axis=1)

    # Calcula saida GATING
    Yg = np.dot(X, rede_gating.T)

    # Calcula saida Especialistas
    num_espec = len(especialistas)
    Ye = np.zeros([num_espec, len(X), 1])  # considerando que da rede tenha 1 saida
    marge_acerto_especialistas = np.zeros([Ye.shape[0], 2])
    especialista_vencedor = 0
    max_acerto = 0
    for i in range(num_espec):
        nn = especialistaLista[i]
        #nn.camadas[0].pesos = especialistas[i].T
        #Ye[i] = nn.predizer(X)
        Ye[i] = nn.feed_forward(X)

        saida_rede_dn = nd.desnormalizacaoDados(media, desvioPadrao, Ye[i])
        Y_DN = nd.desnormalizacaoDados(media, desvioPadrao, Y)

        plt.plot(saida_rede_dn, label='Predição Especialista '+str(i))
        plt.plot(Y_DN, label='Validação Especialista '+str(i))
        plt.legend(loc='upper rigth')

        #plt.plot(Ye[i])
        #plt.plot(Y)
        #plt.plot(Yg)
        #plt.show()

        #Ye[i] = np.dot(X, especialistas[i].T)
        marge_acerto_especialistas[i][0] = i
        num_acerto = 0
        for j in range(Ye[i].shape[0]):
            print(' compare Ye: ',Ye[i][j][0],' Ye round ',round(Ye[i][j][0]),' Y: ',Y[j][0],' result: ',round(Ye[i][j][0], 1)==round(Y[j][0], 1))
            if round(Ye[i][j][0], 1) == round(Y[j][0], 1):
                num_acerto+=1
        if num_acerto > 0:
            marge_acerto = num_acerto*100/Ye[i].shape[0]
        else:
            marge_acerto = 0
        marge_acerto_especialistas[i][1] = marge_acerto
        if marge_acerto > max_acerto:
            max_acerto = marge_acerto
            especialista_vencedor = i
            marge_acerto_especialistas[i][0] = especialista_vencedor

    # Verifica o ESPECIALISTA vencedor (de acordo com a GATING)
    #especialista_vencedor, total = np.unique(Yg.argmax(1), return_counts=True)
    print("Rede Gating:", Yg)
    print("Todos Especialistas:", Ye)
    print("Especialista Vencedor:", especialista_vencedor)
    print("Y do Vencedor:", Ye[especialista_vencedor])

    for i in range(Ye[especialista_vencedor].shape[0]):
        print(' Ye: ',Ye[especialista_vencedor][i][0],' Ye round ',round(Ye[especialista_vencedor][i][0]),'',Y[i][0])

    print("Marge de acerto ", marge_acerto_especialistas[especialista_vencedor][1])

    #saída da gate
    for i in range(len(Yg)):
        plt.plot(Yg[i], label='Saída Gate Especialista'+str(i))
        plt.legend(loc='upper rigth')
    plt.show()

####
#X = np.array([[1], [2], [3], [4]])
#nd.normalizacaoDados(X)
#lags = 4
#st.maximiza_expert(X, lags)
###

# Dados Treinamento
#X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
#Y = np.array([[0], [0], [0], [1]])

#############################Treinamento################################################
serie_txt = np.array(manipulacao.ManipulacaoDados.carregar_arquivo('serie1_trein.txt'))
serie_normalizada = nd.normalizacaoDados(serie_txt);
media = nd.mediaDados(serie_txt);
desvioPadrao = nd.desvioPadraoDados(serie_txt);

X, Y = manipulacao.ManipulacaoDados.preparar_array_X_Y(serie_normalizada, 10)
Y = np.reshape(Y, (len(Y), 1))

X_t = X[0:45]
Y_t = Y[0:45]

class Camada:
    def __init__(self, _funcao_ativacao, _num_entradas, _num_neuronios):
        self.funcao_ativacao = _funcao_ativacao
        self.num_entradas = _num_entradas
        self.num_neuronios = _num_neuronios

class Configuracao:
    def __init__(self, _tipo_rede_neural, _camada):
        self.tipo_rede_neural = _tipo_rede_neural
        self.camada = _camada

listaConfiguracao = list()

camadasEspecialista1 = list()
camadasEspecialista1.append(Camada('sigmoid', 11, 11))
camadasEspecialista1.append(Camada('tanh', 11, 1))
listaConfiguracao.append(Configuracao('batch', camadasEspecialista1)) #tangente hiperbólica

camadasEspecialista2 = list()
camadasEspecialista2.append(Camada('tanh', 11, 11))
camadasEspecialista2.append(Camada('tanh', 11, 1))
listaConfiguracao.append(Configuracao('recorrente', camadasEspecialista2))  #tangente hiperbólica

rede_gating, especialistas, especialistas_variancia, especialistaLista = mistura(X_t, Y_t, listaConfiguracao)

#X_Test = np.array([[0, 1]])
#Y_Test = np.array([[1]])

#predizer(rede_gating, especialistas, X_Test, Y_Test)
#print("\n")
#X_Test = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
#Y_Test = np.array([[0], [0], [0], [1]])



predizer(rede_gating, especialistas, especialistaLista, X_t, Y_t)
