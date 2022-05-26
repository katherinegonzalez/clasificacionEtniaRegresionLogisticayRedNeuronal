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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score # combinación entre sensibilidad y precisión
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from time import time



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

y_train_dummy = pd.get_dummies(y_train_reshaped).to_numpy()
y_test_dummy = pd.get_dummies(y_test_reshaped).to_numpy()


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


#con y_dummies tengo 5 modelos, uno para cada etnia, por lo tanto:
mod = ["Modelo_0","Modelo_1","Modelo_2","Modelo_3","Modelo_4"] 
start_time = time()
for modelos in range(5):
    nsamples_y, nx_y = y_train_dummy.shape
    y_data_train = y_train_dummy[:,modelos].reshape(nsamples_y, )
    
    #Defino el algoritmo a utilizar
    mod[modelos] = LogisticRegression(solver='newton-cg', max_iter=100)
    #Entreno el modelo
    mod[modelos].fit(X_train_scaled, y_data_train)
    
    print("Terminó de entrenar el modelo: ", modelos+1 )
elapsed_time = time() - start_time
print("Terminó de entrenar los 5 modelos, su tiempo de ejecución en minutos fue de: ", elapsed_time /60)


def getPrecision_score(tipo, y_pred, modelos):
 if tipo == 'test':
  nsamples_y, nx_y = y_test_dummy.shape
  y_data_test = y_test_dummy[:,modelos].reshape(nsamples_y, )
  precision = precision_score(y_data_test, y_pred)
  #print('precision: ', precision)
 if tipo == 'train':
  nsamples_y, nx_y = y_train_dummy.shape
  y_data_train = y_train_dummy[:,modelos].reshape(nsamples_y, )
  precision = precision_score(y_data_train, y_pred)
 return precision

def getAccuracy(tipo, y_pred, modelos):
 if tipo == 'test':
  nsamples_y, nx_y = y_test_dummy.shape
  y_data_test = y_test_dummy[:,modelos].reshape(nsamples_y, )
  exactitud = accuracy_score(y_data_test, y_pred)
 if tipo == 'train':
  nsamples_y, nx_y = y_train_dummy.shape
  y_data_train = y_train_dummy[:,modelos].reshape(nsamples_y, )
  exactitud = accuracy_score(y_data_train, y_pred)
 return exactitud    
 


def Calificacion(X, tipo): 
    suma_precision = 0
    suma_exactitud = 0
    y_pred = ["Modelo_0","Modelo_1","Modelo_2","Modelo_3","Modelo_4"]
    for modelos in range(5):
        #Realizo una predicción
        y_pred[modelos] = mod[modelos].predict(X)
        if(tipo):
         suma_precision = suma_precision + getPrecision_score(tipo, y_pred[modelos], modelos)
         suma_exactitud = suma_exactitud + getAccuracy(tipo, y_pred[modelos], modelos)
    predictions = pd.concat([pd.DataFrame(y_pred[0]), pd.DataFrame(y_pred[1]), pd.DataFrame(y_pred[2]), pd.DataFrame(y_pred[3]), pd.DataFrame(y_pred[4])], axis=1,)
    predictions.columns =['0', '1', '2', '3', '4']
    Valor = predictions.idxmax(axis = 1) 
    if suma_precision != 0: 
     print('precision total en ', tipo, ': ', suma_precision/5)
    if suma_exactitud != 0:
     print('accuracy total en ', tipo, ': ', suma_exactitud/5)
    return Valor
    

#Validacion del modelo con train:
y_real = y_train.reshape(16593, 1)
y_estimada = Calificacion(X_train_scaled, 'train')
K = numpy.zeros( ( 5, 5 ) )

for i in range(y_real.shape[ 0 ]):
 K[ int( y_real[ i ] ), int( y_estimada[ i ] ) ] += 1
# end for
#print('K: ', K)
Aciertos = numpy.trace(K)/y_real.shape[ 0 ]
print("El porcentaje de imagenes bien clasificadas para train es de: ",Aciertos)


#Validación con los datos de prueba:
y_real_test = y_test.reshape(7112, 1)
y_est_test = Calificacion(X_test_scaled, 'test')
K_Test = numpy.zeros( ( 5, 5 ) )

for i in range(y_est_test.shape[ 0 ]):
 K_Test[ int( y_real_test[ i ] ), int( y_est_test[ i ] ) ] += 1
# end for
Aciertos_Test = numpy.trace(K_Test)/y_real_test.shape[ 0 ]
print("El porcentaje de imagenes bien clasificadas para test es de: ",Aciertos_Test)


#%%


def getEtnia(numero):
 if numero == '0':
  return 'White'
 if numero == '1':
  return 'Black'
 if numero == '2':
  return 'Asian'
 if numero == '3':
  return 'Indian'
 if numero == '4':
  return 'Hispanic'
        

# Archivo ejecutable para probar:
def Ejecutable(image):
    if len(image.shape) == 3:
        image_fit = image.reshape(image.shape[0],48*48) 
        etiqueta_fit = Calificacion(image_fit, '')
        for l in range(image.shape[ 0 ]):
            print("La etnia de la persona es: ",getEtnia(etiqueta_fit[l]))
           
    else:
        image = cv2.resize(image, (48,48))
        image_fit = image.reshape(1,48*48) 
        etiqueta_fit = Calificacion(image_fit, '')
        print("La etnia de la persona es: ",getEtnia(etiqueta_fit[0]))
        
### Cargar imagen desde el computador
image_2 = cv2.imread("/Users/user/Desktop/persona4.png")
image = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

Ejecutable(image)


    
    