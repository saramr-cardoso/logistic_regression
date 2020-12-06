# funções utilizadas no meu código

import numpy as np


class regressaologistica():
    
    
    def __init__(self, n_passos = 1000, erro_min = 0.0001, alteracao_min = 0.0001, taxa_aprend = 0.001, limiar_decisao = 0.5):
        """Esta função inicializa a regressão logística"""

        self.n_passos = n_passos
        self.erro_min = erro_min
        self.alteracao_min = alteracao_min
        self.taxa_aprend = taxa_aprend
        self.limiar_decisao = limiar_decisao
        self.betas_n = None
        self.beta0 = None


    def fit(self, X_treino, Y_treino):
        """Esta função treina o modelo através do método de gradiente descendente"""

        deve_parar = False

        # inicializar parâmetros
        n_pop, n_variaveis = np.shape(X_treino)
        
        self.betas_n = np.zeros(n_variaveis)
        self.beta0 = 0
        
        erro_antigo = None

        # gradiente descendente
        for _ in range(self.n_passos):
            
            # ajustar modelo
            modelo_linear = np.dot(X_treino, self.betas_n) + self.beta0
            y_previsto = self.sigmoid(modelo_linear) 

            # calcular erro
            erro = y_previsto - Y_treino
            
            # atualizar parâmetros         
            d_betas_n = (1 / n_pop) * np.dot(X_treino.T, erro)
            d_beta0 = (1 / n_pop) * np.sum(erro)
            
            self.betas_n -= self.taxa_aprend * d_betas_n
            self.beta0 -= self.taxa_aprend * d_beta0
            
            # definir condições de paragem
            erro = np.sum(abs(erro))
            
            if erro < self.erro_min:
                deve_parar = True

            if erro_antigo:
                if self.alteracao_min > (erro_antigo - erro):
                    deve_parar = True
            
            if deve_parar:
                break
            
            erro_antigo = erro         


    def sigmoid(self, z):
        """Esta função é um método auxiliar para a função fit e define a função sigmoide"""

        return 1 / (1 + np.exp(-z))


    def predict(self, X_previsao):
        """Esta função prevê os valores de output para um conjunto de dados sem resultado"""

        modelo_linear = np.dot(X_previsao, self.betas_n) + self.beta0
        y_previsto = self.sigmoid(modelo_linear)
        classe_prevista = [1 if i > self.limiar_decisao else 0 for i in y_previsto]
        
        mensagem = print(f'A(s) classe(s) prevista(s) é/são {classe_prevista}')

        return mensagem
            