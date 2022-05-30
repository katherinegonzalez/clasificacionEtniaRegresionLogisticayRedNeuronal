#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 00:14:42 2022

@author: katherinegonzalez
"""

# https://stackoverflow.com/questions/62658215/convergencewarning-lbfgs-failed-to-converge-status-1-stop-total-no-of-iter
# https://stackoverflow.com/questions/38640109/logistic-regression-python-solvers-definitions/52388406#52388406


# cargar datos extraidos

import numpy
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error



datos = numpy.load("/Users/user/Documents/Universidad/Javeriana/AprendizajedeMaquina/Proyecto/Codigo/npz.zip")

Y_edad = datos["edad"]
Y_etnia = datos["etnia"]
Y_genero = datos["genero"]
X_pixeles = datos["pixeles"]
#cv2.startWindowThread()
#image = X_pixeles[48]
#resized = cv2.resize(image, (480,480))
#cv2.imshow("window_name", resized)
#cv2.waitKey(20000)

#cv2.waitKey(1)
#cv2.destroyAllWindows() 
#cv2.waitKey(1)



X_train, X_test, y_train, y_test = train_test_split(X_pixeles, Y_etnia, test_size=0.30, random_state=420)

# Obtener variables dummies para poder etiquetar las estnias en 0 y 1
y_train_reshaped = y_train.reshape(1, 16593)[0]
y_test_reshaped = y_test.reshape(1, 7112)[0]


# escalamos los datos
escalar = StandardScaler()

#dado que X_train es de 3 un array de 3 dimensiones no es posible escalarlo, 
# por eso le hago reshape para que quede de 2 dimensiones:
nsamples, nx, ny = X_train.shape
X_train_reshaped = X_train.reshape((nsamples,nx*ny)) 
X_train_scaled = escalar.fit_transform(X_train_reshaped)

nsamples_test, nx_test, ny_test = X_test.shape
X_test_reshaped = X_test.reshape((nsamples_test,nx_test*ny_test))
X_test_scaled = escalar.fit_transform(X_test_reshaped)

# Red Neuronal
# inicial modelo

modelo_rn = MLPClassifier(hidden_layer_sizes=(20, 20, 20), solver='adam', alpha=0.001, verbose=False)
modelo_rn.out_activation_ = 'softmax'

# entrenamiento
start_time = time()
modelo_rn.fit(X_train_scaled,y_train_reshaped)
elapsed_time = time() - start_time
print("Termino de entrenar modelo Red Neuronal, su tiempo de ejecución en minutos fue de: ", elapsed_time/60)

# predicciones
prediccion_rn = modelo_rn.predict(X_test_scaled)
print('prediccion: ', prediccion_rn)

# RMSE
RMSE_rn = mean_squared_error(y_test_reshaped, prediccion_rn, squared=False)
print('RMSE_rn: ', RMSE_rn)

#%%

def getEtnia(numero):
    if numero == 0:
        return 'White'
    if numero == 1:
        return 'Black'
    if numero == 2:
        return 'Asian'
    if numero == 3:
        return 'Indian'
    if numero == 4:
        return 'Hispanic'
        

# Archivo ejecutable para probar:
def Ejecutable(image):
    if len(image.shape) == 3:
        image_fit = image.reshape(image.shape[0],48*48) 
        return modelo_rn.predict(image_fit)
    else:
        image = cv2.resize(image, (48,48))
        image_fit = image.reshape(1,48*48) 
        return modelo_rn.predict(image_fit)
        
### Cargar imagen desde el computador
image_2 = cv2.imread("/Users/user/Desktop/persona3.png")
image = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

prediccion = Ejecutable(image)
print(prediccion)
print("La etnia de la persona es: ",getEtnia(prediccion[0]))

