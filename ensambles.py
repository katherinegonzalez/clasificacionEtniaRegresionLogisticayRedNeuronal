# cargar datos extraidos

from pickletools import read_long1
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from time import time  

# importar datos

datos = numpy.load("/Users/user/Documents/Universidad/Javeriana/AprendizajedeMaquina/Proyecto/Codigo/npz.zip")


Y_edad = datos["edad"]
Y_etnia = datos["etnia"]
Y_genero = datos["genero"]
X_pixeles = datos["pixeles"]

X_pixeles.shape 
Y_edad.shape  

# separar datos de prueba y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X_pixeles, Y_edad, test_size=0.30, random_state=420)

# reshape
x_train_reshape = X_train.reshape(16593, 48*48)
x_test_reshape = X_test.reshape(7112, 48*48)

# modelo de regresion lineal
modelo_rl = LinearRegression()

# entrenar
inicio_general = time()
start_time = time()
modelo_rl.fit(x_train_reshape,y_train)
elapsed_time = time() - start_time
print("Termino de entrenar modelo regresion lineal, su tiempo de ejecución en minutos fue de: ", elapsed_time/60)


modelo_rl.coef_
modelo_rl.intercept_

# Prediccion con datos de prueba

prediccion_rl = modelo_rl.predict(x_test_reshape)

# RMSE

RMSE = mean_squared_error(y_test, prediccion_rl, squared=False)
print(" ")
print(f"el RMSE del modelo de regresion lineal es: {RMSE}")
# 14.779758481048693

# Red Neuronal
# inicial modelo

modelo_rn = MLPRegressor(hidden_layer_sizes=(200, 50, 20), solver='adam', alpha=0.0001, verbose=False)
# demora 3,38 minutos

# entrenamiento
start_time = time()
modelo_rn.fit(x_train_reshape,y_train)
elapsed_time = time() - start_time
print("Termino de entrenar modelo Red Neuronal, su tiempo de ejecución en minutos fue de: ", elapsed_time/60)

# predicciones
prediccion_rn = modelo_rn.predict(x_test_reshape)

# RMSE
RMSE_rn = mean_squared_error(y_test, prediccion_rn, squared=False)
RMSE_rn
# 13.859954346125443

# parametros
modelo_rn.get_params()

#ENSAMBLE
# ensamble tomado de la pagina de scikit-learn https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization 
estimators = [('reg_lineal', modelo_rl),('red_neuronal', modelo_rn)]
final_estimator = GradientBoostingRegressor(n_estimators=23, subsample = 0.5, min_samples_leaf=25, max_features=1, random_state=42)

reg = StackingRegressor(estimators=estimators, final_estimator=final_estimator)

# entrenar
start_time = time()
reg.fit(x_train_reshape,y_train)
elapsed_time = time() - start_time
print("Termino de entrenar metodo stacking, su tiempo de ejecución en minutos fue de: ", elapsed_time/60)

# predecir

prediccion_ensamble = reg.predict(x_test_reshape)

# RMSE ensamble
RMSE_ensamble = mean_squared_error(y_test, prediccion_ensamble, squared=False)
RMSE_ensamble
# 3.346711614464777

# USANDO  VotingRegressor https://scikit-learn.org/stable/modules/ensemble.html#voting-regressor 

# iniciamos los modelos
ereg = VotingRegressor(estimators=estimators)

# entrenamos el modelo

start_time = time()
ereg.fit(x_train_reshape,y_train)
elapsed_time = time() - start_time
print("Termino de entrenar metodo votacion, su tiempo de ejecución en minutos fue de: ", elapsed_time/60)

prediccion_voting = ereg.predict(x_test_reshape)

# RMSE ensamble
RMSE_voting = mean_squared_error(y_test, prediccion_voting, squared=False)
RMSE_voting
# 12.982673395650098

print("RMSE RL:", RMSE)
print("RMSE RN:", RMSE_rn)
print("RMSE Ensamble: ", RMSE_ensamble)
print("RMSE Voting: ", RMSE_voting) 

tiempo_final = time() - inicio_general
print("FINAL FINAL, su tiempo de ejecución en minutos fue de: ", tiempo_final/60)

