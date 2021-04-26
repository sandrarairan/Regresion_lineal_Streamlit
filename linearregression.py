#import package
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import time
import matplotlib.pyplot as plt

#import the data
data = pd.read_csv("salarios.csv")
st.title("Predicción salario ")
st.markdown("**Guarda las imagenes en la carpeta assets como png**")
#checking the data
st.write("Esta es una aplicación para averiguar qué rango de salario elige usando el aprendizaje automático.")
check_data = st.checkbox("Visualizar el dataset")
if check_data:
    st.write(data.head(2))
    st.write(data.describe())
st.write("Se calcula el salario cambiando la experiencia.")

plot= st.sidebar.checkbox("Mostrar Plot Scatter Train y Test", False)


#input the numbers

experience = st.slider("Experience",int(data.Experiencia.min()),int(data.Experiencia.max()) )
#experience = st.slider("Experience",int(data.Experiencia.min()),int(data.Experiencia.max()),int(data.Experiencia.mean()) )

#splitting your data
X = data.drop('Salario', axis = 1)
y = data['Salario']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, random_state=45)


#modelling
#import your model
model=LinearRegression()
#fitting and predict your model
model.fit(X_train, y_train)
model.predict(X_test)
errors = np.sqrt(mean_squared_error(y_test,model.predict(X_test)))
predictions = model.predict([[experience]])[0]



if plot:
   st.write("scatter Datos de Train")
   fig, ax = plt.subplots()
   ax.scatter(X_train, y_train, color='purple')
   ax.plot(X_train, model.predict(X_train), color='orange')
   plt.title('Salario vs Experiencia')
   plt.xlabel('Experiencia') # creamos nuevos datos adicionales
   plt.ylabel('Salario')
   fig.savefig('/Users/sandrarairan/Documents/desarrollo/streamlit/linearRegression/assets/scatter_train.png')
   st.pyplot(fig)
   

if plot:
   st.write("scatter Datos de Test")
   fig, ax = plt.subplots()
   ax.scatter(X_test, y_test, color='green')
   ax.plot(X_train, model.predict(X_train), color='red')
   plt.title('Salario vs Experiencia')
   plt.xlabel('Experiencia') # creamos nuevos datos adicionales
   plt.ylabel('Salario')
   #fig.write_image('scatter_test.png')
   fig.savefig('/Users/sandrarairan/Documents/desarrollo/streamlit/linearRegression/assets/scatter_test.png')
   st.pyplot(fig) 
   

#checking prediction house price
if st.button("Predecir!"):
    st.header("EL salario predicción es: {}".format(int(predictions)))
    st.subheader("rango predicción: es COL {} - COL {}".format(int(predictions-errors),int(predictions+errors) ))

    st.write("score",model.score(X_test, y_test))

    mse_list=[]
    rmse_list=[]
    r2_list=[]

    mse = mean_squared_error(y_test,model.predict(X_test))
    rmse = sqrt(mse)
    r2=r2_score(y_test,model.predict(X_test))
    mse_list.append(mse)
    rmse_list.append(rmse)
    r2_list.append(r2)

    res=pd.DataFrame(columns=["MSE","RMSE","r2_SCORE"])
    st.write("MSE: el error cuadrático medio de un estimador mide el promedio de los errores al cuadrado, es decir, la diferencia entre el estimador y lo que se estima")

    res["MSE"]=mse_list
    ##RMSE es una medida de la dispersión de estos residuos. En otras palabras, le dice qué tan concentrados están los datos alrededor de la línea de mejor ajuste.
    st.write("RMSE es una medida de la dispersión de estos residuos. En otras palabras, le dice qué tan concentrados están los datos alrededor de la línea de mejor ajuste.")
    
    
    res["RMSE"]=rmse_list
    res["r2_SCORE"]=r2_list

    st.write(res)
    

    