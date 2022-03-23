import pandas as pd
import streamlit as st
from streamlit import caching
from datetime import datetime
from datetime import timedelta, date
import altair as alt
import base64
import io
from io import StringIO
import streamlit.components.v1 as components
import time
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, auc
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="Squad Scikit-Learn",
    layout="wide",
    page_icon=":shark",
    initial_sidebar_state="expanded",
)


# Menu Lateral
st.sidebar.subheader("Filtros para realizar a predição")

#título
st.title("Prevendo Diabetes")

#dataset
url = "https://raw.githubusercontent.com/allanbraquiel/Stack_Labs_2_Squad_Scikit-Learn/main/datasets/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
df = pd.read_csv(url)


# Pagina Principal
# verificando o dataset
st.subheader("Selecionando apenas um pequeno conjunto de atributos")


dados = df.rename(columns = {'Diabetes_binary':'Diabetes', 
                                         'HighBP':'PressAlta',  
                                         'HighChol':'CholAlto',
                                         'CholCheck':'ColCheck', 
                                         'BMI':'IMC', 
                                         'Smoker':'Fumante', 
                                         'Stroke':'Derrame',
                                         'HeartDiseaseorAttack':'CoracaoEnf', 
                                         'PhysActivity':'AtivFisica', 
                                         'Fruits':'Frutas',
                                         'Veggies':"Vegetais", 
                                         'HvyAlcoholConsump':'ConsAlcool', 
                                         'AnyHealthcare':'PlSaude',
                                         'NoDocbcCost':'DespMedica', 
                                         'GenHlth':'SdGeral',
                                         'MentHlth':'SdMental',
                                         'PhysHlth':'SdFisica',
                                         'DiffWalk':'DifCaminhar', 
                                         'Sex':'Sexo',
                                         'Age':'Idade',
                                         'Education':'Educacao',
                                         'Income':'Renda' })

# atributos para serem exibidos por padrão
defaultcols = ["HighChol", "HighBP", "BMI", "Age", "Sex", "Smoker", "Stroke", "HvyAlcoholConsump", "Diabetes_binary"]

# Exibir o dataframe dos chamados
with st.expander("Descrição do dataset:", expanded=False):
    cols = st.multiselect("", df.columns.tolist(), default=defaultcols)
    # Dataframe
    st.dataframe(df[cols])


#nomedousuário
user_input = st.sidebar.text_input("Digite seu nome")

#escrevendo o nome do usuário
st.write("Paciente:", user_input)



#dados de entrada
# x = df.drop(['Diabetes_binary'],1)
x = df[["HighChol", "HighBP", "BMI", "Age", "Sex", "Smoker", "Stroke", "HvyAlcoholConsump", "Income"]]
y = df['Diabetes_binary']


#separa dados em treinamento e teste
x_train, x_text, y_train, y_test = train_test_split(x, y, test_size=0.2)

#dados dos usuários com a função
def get_user_date():

    HighChol = st.sidebar.slider("Colesterol Alto", 0, 200, 110)

    HighBP  = st.sidebar.slider("Pressão Sanguínea", 0, 122, 72)

    BMI = st.sidebar.slider("Índice de massa corporal", 0.0, 70.0, 15.0)

    Age = st.sidebar.slider("Idade", 15, 100, 25)

    Sex = st.sidebar.selectbox("Sexo", ("Masculino", "Feminino"))
    if Sex == "Masculino":
        Sex = 0
    else:
        Sex = 1

    Smoker = st.sidebar.selectbox("Fumante", ("Sim", "Não"))
    if Smoker == "Sim":
        Smoker = 1
    else:
        Smoker = 0

    Stroke = st.sidebar.selectbox("Derrame", ("Sim", "Não"))
    if Stroke == "Sim":
        Stroke = 1
    else:
        Stroke = 0

    HvyAlcoholConsump  = st.sidebar.selectbox("Consome álcool", ("Sim", "Não"))
    if HvyAlcoholConsump == "Sim":
        HvyAlcoholConsump = 1
    else:
        HvyAlcoholConsump = 0

    Income = st.sidebar.selectbox("Renda Familiar", ("10 Mil", "15 Mil", "20 Mil", "25 Mil", "35 Mil", "50 Mil", 
                                    "75 Mil", "Maior que 75 Mil"))
    if Income == "10 Mil":
        Income = 1
    elif Income == "15 Mil":
        Income = 2
    elif Income == "20 Mil":
        Income = 3
    elif Income == "25 Mil":
        Income = 4
    elif Income == "35 Mil":
        Income = 5
    elif Income == "50 Mil":
        Income = 6
    elif Income == "75 Mil":
        Income = 7
    elif Income == "Maior que 75 Mil":
        Income = 8
    

    #dicionário para receber informações

    user_data = {
                'Colesterol': HighChol,
                'Pressão Sanguínea': HighBP,
                'Índice de massa corporal': BMI,
                'Idade': Age,
                'Sexo': Sex,
                'Fumante': Smoker,
                'Derrame': Stroke,
                'Consome Alcool': HvyAlcoholConsump,
                'Renda': Income
                }

    features = pd.DataFrame(user_data, index=[0])

    return features

user_input_variables = get_user_date()


#cabeçalho
with st.expander("Informações dos dados"):
    #grafico
    graf = st.bar_chart(user_input_variables)
    

dtc = DecisionTreeClassifier(criterion='entropy', max_depth=3)
dtc.fit(x_train, y_train)

#acurácia do modelo
st.subheader('Acurácia do modelo')
st.write(accuracy_score(y_test, dtc.predict(x_text))*100)

#previsão do resultado
prediction = dtc.predict(user_input_variables)

st.subheader('Previsão:')

st.write(prediction)




# fonte: https://medium.com/data-hackers/desenvolvimento-de-um-aplicativo-web-utilizando-python-e-streamlit-b929888456a5