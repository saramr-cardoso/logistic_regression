{\rtf1\ansi\ansicpg1252\cocoartf2513
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;\red183\green111\blue179;\red23\green23\blue23;}
{\*\expandedcolortbl;;\cssrgb\c77255\c52549\c75294;\cssrgb\c11765\c11765\c11765;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11360\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\sl360\partightenfactor0

\f0\fs24 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 i\cf0 \cb1 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 mport numpy as np\
from sklearn.model_selection import train_test_split\
from sklearn import datasets\
import matplotlib.pyplot as pl\
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0
\cf0 import rl\
\
def accuracy(y_true, y_pred):\
    accuracy = np.sum(y_true == y_pred) / len(y_true)\
    return accuracy\
\
bc = datasets.load_breast_cancer()\
X, y = bc.data, bc.target\
\
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)\
\
regressor = rl.regressaologistica(num_passos=1000, erro_min=0.0001, alteracao=0.001)\
regressor.train(X_train, y_train)\
predictions = regressor.predict(X_test)\
\
print(" ")\
print("Acuracidade da Regress\'e3o Log\'edstica", accuracy(y_test, predictions))\
print(" ")}