import streamlit as st
import pandas as pd
from PIL import Image
import plotly_express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import seaborn as sns

# Leemos los dataset para los grupos de ingresos alto y bajo

dfbajo = pd.read_csv('tablas/grupo_ingresos_bajos_streamlit.csv')
dfalto = pd.read_csv('tablas/grupo_ingresos_altos_streamlit.csv')
def main():
    st.title("Esperanza de Vida al Nacer")
    st.subheader("Brecha del Indicador entre paises de Grupos distintos")
    st.subheader("Valores Historicos (Promedio)")
    
#Le ofrecemos al usuario la opción de configurar el intervalo de tiempo para la visualización de los datos históricos

    iyear_col, fyear_col = st.columns([5, 5])
    with iyear_col:
        iyear_choice = st.slider(
            "Desde",
            min_value=1990,
            max_value=2020,
            step=1,
            value=1990,
        )
    with fyear_col:
        fyear_choice = st.slider(
            "Hasta",
            min_value=iyear_choice,
            max_value=2020,
            step=1,
            value=2020,
        )
    
    #Ahora construimos las gráficas de la esperanza de vida y el valor de la brecha entre los valores
    dfp = pd.read_csv('tablas/promedios.csv')
    mask1 = (dfp['año']>=iyear_choice)
    mask2 = (dfp['año']<=fyear_choice)

    series1 = dfp[mask1&mask2]['EVN (promedio) Paises de Bajo Ingreso']
    series2 = dfp[mask1&mask2]['EVN (promedio) Paises de Alto Ingreso']
    series3 = dfp[mask1&mask2]['brecha']

    fig = px.line(x=dfp[mask1&mask2]['año'], y=[series1, series2, series3],)
    st.plotly_chart(fig)
    st.subheader("Modelo de Regresión Lineal (Descriptivo)")
    # Cargamos los dataset con las características de cada modelo
    a = pd.read_csv('tablas/dataset_ingresos_altos.csv')
    b = pd.read_csv('tablas/dataset_ingresos_bajos.csv')
    # Definimos las variables
    ya = a[a.columns[4]]
    Xa = a.drop(a.columns[[3,4]], axis='columns')
    yb = b[b.columns[2]]
    Xb = b.drop(b.columns[[1,2]], axis='columns')
    #hacemos el split
    Xa_train, Xa_test, ya_train, ya_test = train_test_split(Xa, ya.values.reshape(-1,1), train_size = 0.8,
                                                            random_state = 1234, shuffle = True)
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, yb.values.reshape(-1,1), train_size = 0.8,
                                                            random_state = 1234, shuffle = True)
    #instanciamos los modelos
    Xa_train = sm.add_constant(Xa_train, prepend=True)
    modelo_a = sm.OLS(endog=ya_train, exog=Xa_train,)
    modelo_a = modelo_a.fit()
    st.write(modelo_a.summary())
    Xb_train = sm.add_constant(Xb_train, prepend=True)
    modelo_b = sm.OLS(endog=yb_train, exog=Xb_train,)
    modelo_b = modelo_b.fit()
    # Hacemos las predicciones sobre el set de prueba
    ya_train = ya_train.flatten()
    prediccion_train = modelo_a.predict(exog = Xa_train)
    residuos_train   = prediccion_train - ya_train
    # graficamos
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 8))

    axes[0, 0].scatter(ya_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.4)
    axes[0, 0].plot([ya_train.min(), ya_train.max()], [ya_train.min(), ya_train.max()],
                    'k--', color = 'black', lw=2)
    axes[0, 0].set_title('Valor predicho vs valor real', fontsize = 10, fontweight = "bold")
    axes[0, 0].set_xlabel('Real')
    axes[0, 0].set_ylabel('Predicción')
    axes[0, 0].tick_params(labelsize = 7)

    axes[0, 1].scatter(list(range(len(ya_train))), residuos_train,
                       edgecolors=(0, 0, 0), alpha = 0.4)
    axes[0, 1].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
    axes[0, 1].set_title('Residuos del modelo', fontsize = 10, fontweight = "bold")
    axes[0, 1].set_xlabel('id')
    axes[0, 1].set_ylabel('Residuo')
    axes[0, 1].tick_params(labelsize = 7)

    sns.histplot(
        data    = residuos_train,
        stat    = "density",
        kde     = True,
        line_kws= {'linewidth': 1},
        color   = "firebrick",
        alpha   = 0.3,
        ax      = axes[1, 0]
    )

    axes[1, 0].set_title('Distribución residuos del modelo', fontsize = 10,
                         fontweight = "bold")
    axes[1, 0].set_xlabel("Residuo")
    axes[1, 0].tick_params(labelsize = 7)


    sm.qqplot(
        residuos_train,
        fit   = True,
        line  = 'q',
        ax    = axes[1, 1], 
        color = 'firebrick',
        alpha = 0.4,
        lw = 2
    )
    axes[1, 1].set_title('Q-Q residuos del modelo', fontsize = 10, fontweight = "bold")
    axes[1, 1].tick_params(labelsize = 7)

    axes[2, 0].scatter(prediccion_train, residuos_train,
                       edgecolors=(0, 0, 0), alpha = 0.4)
    axes[2, 0].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
    axes[2, 0].set_title('Residuos del modelo vs predicción', fontsize = 10, fontweight = "bold")
    axes[2, 0].set_xlabel('Predicción')
    axes[2, 0].set_ylabel('Residuo')
    axes[2, 0].tick_params(labelsize = 7)

    # Se eliminan los axes vacíos
    fig.delaxes(axes[2,1])

    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle('Diagnóstico residuos', fontsize = 12, fontweight = "bold");
    st.pyplot(fig)

if __name__ == "__main__":
  main()
