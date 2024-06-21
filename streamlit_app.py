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
st.markdown("<img src='https://raw.githubusercontent.com/sdrcr74/bank_test/main/datascientest_logo.png' width='100' style='display: block; margin: 0 auto;'>" , unsafe_allow_html=True)
st.image("https://raw.githubusercontent.com/sdrcr74/bank_nov23/main/logo2.jpeg")
st.markdown("<h1 style='text-align: justify;'>Prédiction du succès d une campagne de Marketing d’une banque</h1>", unsafe_allow_html=True)
st.subheader("Maxence Malherre, Sophie Dorcier, Stéphane Lascaux, Van-Anh HA")
st.subheader("NOV23_CONTINU_DA - Datascientest", divider='rainbow')
st.divider()
bank_cleaned = bank.drop(bank.loc[bank["job"] == "unknown"].index, inplace=True)
bank_cleaned = bank.drop(bank.loc[bank["education"] == "unknown"].index, inplace=True)
bank_cleaned = bank.drop(['contact', 'pdays'], axis = 1)
st.subheader('Modélisation')
st.write('Le modèle choisi sera la classification en raison des valeurs discrètes de la variable cible Deposit')
if st.button('Variable Deposit'):
    st.write(bank_cleaned['deposit'].head())
feats = bank_cleaned.drop(['deposit'], axis = 1)
target = bank_cleaned['deposit']
st.write('Le jeu de données sera donc séparé en 2 dataframes: "feats" et "target"')
if st.button("Code"):
    st.code("feats = bank_cleaned.drop(['deposit'], axis = 1)")
    st.code("target = bank_cleaned['deposit']")

st.dataframe(feats)

st.dataframe(target)
st.write("1. Nous allons procéder à la séparation du jeu de données en jeu d'entrainement X_train et test X_test avec la répartition 80 et 20%")
X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state=42)

st.code('X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state=42')
st.write("2. Puis nous allons dans un deuxième temps appliquer la standardisation des variables numériques")
cols = ['age','balance','day','campaign','previous','duration']
scaler = StandardScaler()
cols =['age','balance','day','campaign','previous','duration']
cols1 = bank_cleaned[['age','balance','day','campaign','previous','duration']]
if st.button("Code 2"):
  st.code("X_train[cols] = scaler.fit_transform(X_train[cols]")
  st.code("X_test[cols] = scaler.transform(X_test[cols]")
st.write("Variables numériques")
st.dataframe(cols1)
X_train[cols]=scaler.fit_transform(X_train[cols])
X_test[cols] = scaler.transform(X_test[cols])

st.write("3. Ensuite nous encoderons les variables explicatives Housing, Default et Loan de valeur booléenne avec la formule")
st.write('Définition')
st.code('def replace_yes_no(x)')
st.code("if x=='no':")
st.code("  return 0")
st.code("if x=='yes':")
st.code("  return 1")
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
st.write('4. Nous utiliserons également une définition pour la variable month en remplaçant le mois de janvier par 1, février par 2 etc')
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
st.write('5. Nous nous servirons de la fonction get.dummies pour les variables de chaîne de caractères')
st.write('get_dummies')
st.code("X_train = pd.get_dummies(X_train, dtype = 'int')")
st.code("X_test= pd.get_dummies(X_test, dtype = 'int')")

X_train = pd.get_dummies(X_train, dtype = 'int')
X_test= pd.get_dummies(X_test, dtype = 'int')
st.write('X_train')
st.write(X_train.head())
st.write("6. Et pour la dernière étape, nous procéderons à l'encodage de la variable cible avec LabelEncoder")
le = LabelEncoder()
st.write('LabelEncoder'):
st.code('y_train= le.fit_transform(y_train)')
st.code('le.transform(y_test)')
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
from sklearn.linear_model import LogisticRegression
reglog = LogisticRegression(random_state=42)
reglog.fit(X_train, y_train)
print("Accuracy score du Logistic regression (train) : ',reglog.score(X_train, y_train)")

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(random_state=42)
forest.fit(X_train, y_train)
print('Accuracy score du Random Forest (train) : ',forest.score(X_train, y_train))
from sklearn.tree import DecisionTreeClassifier

treecl = DecisionTreeClassifier(random_state=42)
treecl.fit(X_train,y_train)

print('Accuracy score du Decision Tree (train) : ',treecl.score(X_train, y_train))


st.subheader('Résultats du modèle')
st.write("Nous avons sélectionné 3 techniques de classification pour entrainer et tester notre modèle")
modèle_sélectionné=st.selectbox(label="Modèle", options=['Régression logistique','Decision Tree','Random Forest'])

if modèle_sélectionné=='Régression logistique':
    st.metric(label="accuracy y_train", value=round(reglog.score(X_train, y_train),2))
    st.metric(label="accuracy y_test", value=round(reglog.score(X_test, y_test),2))
    
if modèle_sélectionné=='Decision Tree':
    st.metric(label="accuracy y_train", value=round( treecl.score(X_train, y_train),2))
    st.metric(label="accuracy y_test", value=round(treecl.score(X_test, y_test),2))
if modèle_sélectionné=='Random Forest':
    st.metric(label="accuracy y_train", value=round(forest.score(X_train, y_train),2))
    st.metric(label="accuracy y_test", value=round(forest.score(X_test, y_test),2))
st.write("Le modèle RandomForest est donc le meilleur modèle au vu des résultats mais nous constatons un problème d'overfitting")
st.write('Afin d’évaluer la précision de notre modèle, nous avons vérifié sa volatilité avec la technique de validation croisée sur le modèle RandomForest. Celle-ci étant peu volatile [0.77762106 0.74424071 0.78232252 0.83921016 0.82267168] , nous pouvons considérer que le modèle est fiable via un train_test_split.')
st.write("Techniques utilisées pour baisser l'overfitting")
techniques=st.selectbox(label='Techniques', options=['Bagging','RandomOverSampler','GridSearchCV','Importance_feature','Suppression variable Duration'])

feat_importances = pd.DataFrame(forest.feature_importances_, index=X_test.columns, columns =['Importance'] )
feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
feat_importances.plot(kind='bar', figsize=(8,6))
if techniques=='Bagging':
    st.write("La méthode Bagging permet d'améliorer la performance et la stabilité des algorithmes en réduisant la variance et en limitant l'overfitting")
    if st.button('accuracy y_train et y_test Bagging'):
       st.button('1 & 0.84')
       st.write('Nous n’avons pas observé de différence en utilisant le Bagging sur l’overfitting')
if techniques=='GridSearchCV':
    if st.button('Best Hyperparameter'):
       st.write("{'max_depth': 10, 'n_estimators': 1000}")
    if st.button('accuracy y_train et y_test avec max_depth:10'):
       st.button("0.87 & 0.83")
if techniques=='RandomOverSampler':
    if st.button('accuracy y_train et y_test avec SMOTE'):
       st.button('0.83 & 0.83')
       st.write("Le résultat a été concluant avec un résultat de 0,83 sur le modèle d'entraînement.") 
       st.write('Quant au modèle test, nous obtenons également un très bon score avec 0.83.')

if techniques=='Importance_feature':
   st.pyplot(feat_importances.plot(kind='bar').figure)
   st.write('Nous remarquons que la variable Duration prédomine de manière significative sur toutes les autres variables')
   st.write("Suite à l'analyse de l'importance des variables, nous allons réduire notre jeu de données à 5 et 9 variables:")
   if st.button('accuracy y_train et y_test à 5 variables'):                   
      st.button('0.87 & 0.81')   
      st.write("le fait de ne garder que les 5 variables les plus importantes réduit notre score")                                                                                              
   if st.button('accuracy y_train et y_test à 9 variables'):
      st.button('0.89 & 0.84')
      st.write("le fait de ne garder que les 9 variables les plus importantes n’a pas d’impact sur le score")

if techniques=='Suppression variable Duration':
    if st.button('accuracy y_train et y_test sans la variable Duration'): 
       st.button('0.79 & 0.71')
       st.write('L’overfitting sur la random forest a empiré en faisant baisser le score de notre jeu de test.') 
       st.write("Cependant nous avons pu observer une baisse de l'overfitting sur le modèle Logistic regression mais le score est plutôt faible.")

       st.write("En conclusion, l'hyperparamètre max_depth semble être le meilleure solution pour éviter l'overfitting et conserver un modèle prédictif fiable et robuste.")
        
        


   
