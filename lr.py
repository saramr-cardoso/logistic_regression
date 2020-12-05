{\rtf1\ansi\ansicpg1252\cocoartf2513
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11180\viewh15520\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import numpy as np\
\
class regressaologistica():\
\
    def __init__(self, num_passos=1000, erro_min=0.001, alteracao=0.001):\
        self.num_passos = num_passos\
        self.erro_min = erro_min\
        self.alteracao = alteracao	\
        self.peso = None\
        self.bias = None\
\
    def train(self, X, y):\
        n_samples, n_features = X.shape\
\
        # Inicializa de par\'e2metros\
        self.peso = np.zeros(n_features)\
        self.bias = 0\
\
        # gradient descent\
        for _ in range(self.num_passos):\
            linear_model = np.dot(X, self.peso) + self.bias\
\
            # executa fun\'e7\'e3o sigmoid\
            y_predicted = self._sigmoid(linear_model)\
\
            # compute gradients\
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))\
            db = (1 / n_samples) * np.sum(y_predicted - y)\
\
            # atualiza par\'e2metros\
            self.peso -= self.erro_min * dw\
            self.bias -= self.erro_min * db\
\
    # predi\'e7\'e3o	\
    def predict(self, X):\
        linear_model = np.dot(X, self.peso) + self.bias\
        y_predicted = self._sigmoid(linear_model)\
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]\
        return np.array(y_predicted_cls)\
\
    # c\'e1lculo da sigmoid\
    def _sigmoid(self, x):\
        return 1 / (1 + np.exp(-x))\
	}