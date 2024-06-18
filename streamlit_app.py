import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

url = 'https://raw.githubusercontent.com/sdrcr74/bank_nov23/main/bank.csv'
bank = pd.read_csv(url)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
bank_cleaned = bank.drop(bank.loc[bank["job"] == "unknown"].index, inplace=True)
bank_cleaned = bank.drop(bank.loc[bank["education"] == "unknown"].index, inplace=True)
bank_cleaned = bank.drop(['contact', 'pdays'], axis = 1)
st.write('Le modèle choisi sera la classification en raison des valeurs discrètes de la variable cible deposit')
st.write(bank_cleaned['deposit'].head())
feats = bank_cleaned.drop(['deposit'], axis = 1)
target = bank_cleaned['deposit']
st.write('Le jeu de données sera donc séparé en 2 dataframes: "feats" et "target"')
if st.checkbox("Afficher code"):
    st.write("feats = bank_cleaned.drop(['deposit'], axis = 1)")
    st.write("target = bank_cleaned['deposit']")
if st.checkbox("Afficher feats"):
    st.dataframe(feats)
if st.checkbox("Afficher target"):
    st.dataframe(target)
st.write("Nous allons procéder à la séparation du jeu de données en jeu d'entrainement X_train et test X_test")
st.write('X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state=42')
st.write("Puis nous allons dans un deuxième temps appliquer la standardisation des variables numériques")
cols = ['age','balance','day','campaign','previous','duration']
scaler = StandardScaler()
cols =['age','balance','day','campaign','previous','duration']
cols1 = bank_cleaned[['age','balance','day','campaign','previous','duration']]
if st.checkbox("Afficher variables numériques"):
   st.dataframe(cols1)
X_train[cols] = scaler.fit_transform(X_train[cols])
X_test[cols] = scaler.transform(X_test[cols])
if st.checkbox("Afficher code"):
  st.write("X_train[cols] = scaler.fit_transform(X_train[cols]")
  st.write("X_test[cols] = scaler.transform(X_test[cols]")
  
def replace_yes_no(x):
  if x == 'no':
    return 0
  if x == 'yes':
    return 1

X_train['default'] = X_train['default'].apply(replace_yes_no)
X_test['default'] = X_test['default'].apply(replace_yes_no)

X_train['housing'] = X_train['housing'].apply(replace_yes_no)
X_test['housing'] = X_test['housing'].apply(replace_yes_no)

X_train['loan'] = X_train['loan'].apply(replace_yes_no)
X_test['loan'] = X_test['loan'].apply(replace_yes_no)

def replace_month(x):
  if x == 'jan':
    return 1
  if x == 'feb':
    return 2
  if x == 'mar':
    return 3
  if x == 'apr':
    return 4
  if x == 'may':
    return 5
  if x == 'jun':
    return 6
  if x == 'jul':
    return 7
  if x == 'aug':
    return 8
  if x == 'sep':
    return 9
  if x == 'oct':
    return 10
  if x == 'nov':
    return 11
  if x == 'dec':
    return 12
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
X_train['month'] = X_train['month'].apply(replace_month)
X_test['month'] = X_test['month'].apply(replace_month)
X_train = pd.get_dummies(X_train, dtype = 'int')
X_test= pd.get_dummies(X_test, dtype = 'int')
le = LabelEncoder()

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
from sklearn.linear_model import LogisticRegression
reglog = LogisticRegression(random_state=42)
reglog.fit(X_train, y_train)
print('Accuracy score du Logistic regression (train) : ',reglog.score(X_train, y_train))

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(random_state=42)
forest.fit(X_train, y_train)
print('Accuracy score du Random Forest (train) : ',forest.score(X_train, y_train))
from sklearn.tree import DecisionTreeClassifier

treecl = DecisionTreeClassifier(random_state=42)
treecl.fit(X_train,y_train)

print('Accuracy score du Decision Tree (train) : ',treecl.score(X_train, y_train))


st.write('Modélisation')
modèle_sélectionné=st.selectbox(label="Modèle", options=['Régression logistique','Decision Tree','Random Forest'])

if modèle_sélectionné=='Régression logistique':
    st.metric(label="accuracy y_train", value=reglog.score(X_train, y_train))
    st.metric(label="accuracy y_train", value=reglog.score(X_test, y_test))
    
if modèle_sélectionné=='Decision Tree':
    st.metric(label="accuracy", value= treecl.score(X_train, y_train))
    st.metric(label="accuracy y_train", value=reglog.score(X_test, y_test))
if modèle_sélectionné=='Random Forest':
    st.metric(label="accuracy", value=forest.score(X_train, y_train))
    st.metric(label="accuracy y_train", value=reglog.score(X_test, y_test))
st.write('Le modèle RandomForest est donc le meilleur modèle au vu des résultats mais nous constatons un problème d'overfitting')
st.write('Afin d’évaluer la précision de notre modèle, nous avons vérifié sa volatilité avec la technique de validation croisée sur le modèle RandomForest. Celle-ci étant peu volatile [0.77762106 0.74424071 0.78232252 0.83921016 0.82267168] , nous pouvons considérer que le modèle est fiable via un train_test_split.')
st.write('Techniques utilisées pour baisser l'overfitting')
techniques=st.selectbox(label='Techniques', options=['Importance_feature','Suppression variable Duration','Bagging','RandomOverSampler','GridSearchCV'])

        


   
