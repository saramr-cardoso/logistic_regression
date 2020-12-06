# Regressão Logística

Este projeto implementa, do zero, um algoritmo de regressão logística, recorrendo apenas à biblioteca numpy. 

O algoritmo recebe os seguintes parâmetros e respetivos valores por defeito: 
    Número máximo de passos (1000)
    Erro mínimo (0.0001)
    Alteração mínima entre iterações (0.0001)
    Taxa de aprendizagem (0.001)
    Limiar de decisão (0.5)
    
O modelo de regressão logística é ajustado utilizando gradiente descendente, considerando o erro mínimo e a aleração minima entre iterações como critérios de paragem.
O algoritmo permite prever valores de output (classificação utilizando duas classes) para um conjunto de dados sem resultado.


## Exemplo de aplicação

import numpy as np
import matplotlib.pyplot as plt
from utils import regressaologistica as rl

### Definir dados para treino
X_treino = np.array([0.50,1.00,2.00,4.25,3.25,5.50], ndmin=2).reshape((6,1))
Y_treino = np.array([0,0,0,1,1,1])

### Inicializar o algoritmo
regressor = rl(limiar_decisao = 0.6) # o limiar de decisão foi alterado para melhorar a precisão dos resultados

### Representação gráfica da sigmoide
plt.scatter(X_treino,regressor.sigmoid(X_treino))

### Treinar o modelo
regressor.fit (X_treino, Y_treino)

### Definir inputs para previsão de outputs desconhecidos
X_previsao = np.array([2.7,0.75], ndmin=2).reshape((2,1))

### Previsão de outputs desconhecidos
previsoes = regressor.predict(X_previsao)

### Resultado
A(s) classe(s) prevista(s) é/são [1, 0]

