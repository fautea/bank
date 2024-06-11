import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
bank=pd.read_csv("bank.csv")
st.title("Bank Marketing Campaign")
st.sidebar.title("Sommaire")
pages=["Contexte du projet","Exploration des données","Analyse des données","DataViz","Modélisation","Conclusion"]
page=st.sidebar.radio("Aller à la page:", pages)
if page==pages[0]:
  st.write("Contexte du projet")
  st.write("L’objectif du projet est d’établir un modèle permettant de prédire le succès d’une campagne marketing d’une banque. Concrètement, il s’agit de prédire si suite à la campagne, un client souscrit ou non au produit Dépôt à terme.")
  st.write("Le jeu de données qui nous a été mis à disposition s’appelle “Bank Marketing Dataset”. Ce jeu de données est disponible librement sur Kaggle, mais à la base il vient de la UC Irvine Machine Learning Repository. Ce sont des données liées aux campagnes de marketing direct d’une banque portugaise. Il date de 2012.Il contient 11 162 lignes de données et 17 colonnes.")
  st.write("Dans un premier temps, nous étudierons les différentes variables puis analyserons le dataset et procéderons à un nettoyage des données: doublons, données manquantes, pertinence des différentes variables.")
  st.image("banking.jpg")
elif page==pages[1]:
  st.write('Exploration des données')
  st.write("Avant d'explorer les données du dataset, il nous a semblé pertinent de comprendre les différentes variables présentes dans le jeu de données.") 
  st.write("Pour la plupart, l'intitulé des variables était clair et compréhensible. Nous allons cependant clarifier certaines variables:") 
  st.write("-balance: montant du compte en banque")
  st.write("-housing: prêt immobilier") 
  st.write("-loan: autre prêt") 
  st.write("-contact: moyen de contact") 
  st.write("-day & month: jour et mois du contact")
  st.write("-duration: durée du contact") 
  st.write("-campaign: nombre de contact durant la campagne marketing") 
  st.write("-pdays: nombre de jours de contact avant la campagne")
  st.write("-previous: nombre de contact avant la campagne")
  st.write("-poutcome: résultat de la dernière campagne")
  st.write("Aperçu de notre dataset")
  st.dataframe(bank.head())
  st.write('Dimensions du Dataframe')
  st.write(bank.shape)
  if st.checkbox("Afficher le nombre de doublons"):
    st.dataframe(bank.duplicated())
  if st.checkbox("Afficher les valeurs manquantes"):
    st.dataframe(bank.isna().sum())
  if st.checkbox("Répartition de la variable deposit"):
    st.dataframe(bank['deposit'].value_counts())
  if st.checkbox("Répartition en % par résultat de la dernière campagne via la variable poutcome"):
    st.dataframe(bank['poutcome'].value_counts()/len(bank))
  if st.checkbox("Pourcentage du nombre de contact lors de la dernière campagne égal à 0"):
    st.dataframe((bank['previous'] == 0).value_counts()/len(bank))
  if st.checkbox("Pourcentage de -1 dans la variable pdays"):
    st.dataframe((len(bank[bank['pdays'] == -1]) / len(bank)))
  if st.checkbox("Nombre de chiffres négatifs dans la variable balance"):
    st.dataframe(len(bank[bank['balance'] < 0]))
elif page==pages[2]:
  st.write("Analyse des données")
  Graphique_sélectionné=st.sidebar.selectbox(label="Graphique", options=['Répartition par âge','Répartition par métier','Répartition par statut marital','Répartition par éducation','Répartition par mois','Répartition par défauts de paiement', 'Répartition par prêt immobilier','Répartition des prêts à la conso','Répartition par type de contact','Résultat sur la dernière campagne marketing','Répartition du nombre de dépôts à terme','Répartition du nombre de contact de la dernière campagne','Répartition des types de métier en fonction des dépôts à terme'])
  if Graphique_sélectionné =='Répartition par âge':   
    fig=sns.displot(x='age', data=bank)
    plt.title('Répartition par âge')
    st.pyplot(fig)
  if Graphique_sélectionné =='Répartition par métier':    
    fig1=sns.displot(x='job', data=bank)
    plt.xticks(rotation=90)
    plt.title('Répartition par métier')
    st.pyplot(fig1)
  if Graphique_sélectionné =='Répartition par statut marital': 
    fig2=sns.displot(x='marital', data=bank)
    plt.title('Répartition par statut marital')
    st.pyplot(fig2)
  if Graphique_sélectionné =='Répartition par éducation': 
    fig3=sns.displot(x='education', data=bank)
    plt.title('Répartition par éducation')
    st.pyplot(fig3)
  if Graphique_sélectionné =='Répartition par mois': 
    fig4=sns.displot(x='month', data=bank)
    plt.title('Répartition par mois')
    st.pyplot(fig4)
  if Graphique_sélectionné =='Répartition par défauts de paiement': 
    fig5=sns.displot(x='default', data=bank)
    plt.title('Répartition par défauts de paiement')
    st.pyplot(fig5)
  if Graphique_sélectionné =='Répartition par prêt immobilier': 
    fig6=sns.displot(x='housing',data=bank)
    plt.title('Répartition par prêt immobilier')
    st.pyplot(fig6)
  if Graphique_sélectionné =='Répartition des prêts à la conso': 
    fig7=sns.displot(x='loan', data=bank)  
    plt.title('Répartition des prêts à la conso')
    st.pyplot(fig7)
  if Graphique_sélectionné =='Répartition par type de contact': 
    fig8=sns.displot(x='contact', data=bank, stat = 'percent')
    plt.title('Répartition par type de contact')
    st.pyplot(fig8)
  if Graphique_sélectionné =='Résultat sur la dernière campagne marketing': 
    fig9=sns.displot(x='poutcome', data=bank, stat = 'percent')
    plt.title('Résultat sur la dernière campagne marketing')
    st.pyplot(fig9)
  if Graphique_sélectionné =='Répartition du nombre de dépôts à terme': 
    fig10=sns.displot(x='deposit', data=bank, stat = 'percent')
    plt.title('Répartition du nombre de dépôts à terme')
    st.pyplot(fig10)
  if Graphique_sélectionné =='Répartition du nombre de contact de la dernière campagne': 
    fig11=sns.displot(x='previous', data=bank, stat = 'percent')
    plt.title('Répartition du nombre de contact de la dernière campagne')
    st.pyplot(fig11)
  
  if Graphique_sélectionné=='Répartition des types de métier en fonction des dépôts à terme':
    b_df = pd.DataFrame()
    b_df['yes'] = bank[bank['deposit'] == 'yes']['job'].value_counts()
    b_df['no'] = bank[bank['deposit'] == 'no']['job'].value_counts()
    st.pyplot(b_df.plot.bar(title = 'Job & Deposit ', color=['b','r']).figure)
elif page==pages[3]:
  st.write("DataViz")
  st.write("Notre variable cible est ‘deposit’, elle renseigne si le client a fait un dépôt à terme. Les classes pour cette variable sont bien équilibrées, ce qui est positif pour la suite de nos prédictions.")
  st.write("Selon nos analyses du jeu de données, nous avons remarqué que la variable la plus pertinente est la durée du contact (duration).")
  st.write("Les autres variables qui nous semblent les plus pertinentes sont :")
  st.write("-Age")
  st.write("-Job")
  st.write("-Month")
  st.write("-Balance")
  st.write("Notre variable cible est ‘deposit’, elle renseigne si le client a fait un dépôt à terme. Les classes pour cette variable sont bien équilibrées, ce qui est positif pour la suite de nos prédictions.")         
  st.write("Nous avons pu identifier des relations entre certaines variables explicatives et notre variable cible :")
  
  Graphique=st.selectbox(label="",options=['Heatmap',"Dépôt à terme en fonction de la durée du contact","Répartition par mois en fonction du dépôt à terme",'Relation Age, balance et Deposit',"Dépôt à terme en fonction de l'age"])
  if Graphique=='Heatmap':
    st.write("La durée de contact (duration) aura un impact important dans la modélisation. En effet, nous remarquons que cette variable est la plus corrélée avec notre variable cible.")
   
    deposit = lambda x:1 if x=='yes' else 0
    bank['deposit_num'] = bank['deposit'].apply(deposit)

    default = lambda x:1 if x=='yes' else 0
    bank['default_num'] = bank['default'].apply(default)

    housing = lambda x:1 if x=='yes' else 0
    bank['housing_num'] = bank['housing'].apply(housing)

    loan = lambda x:1 if x=='yes' else 0
    bank['loan_num'] = bank['loan'].apply(loan)

    matrice = bank.corr(numeric_only = callable )
    fig, ax = plt.subplots(figsize = (10,10))
    sns.heatmap(matrice, annot=True, cmap = 'rainbow', ax = ax);
    st.write(fig)
   
  if Graphique=="Dépôt à terme en fonction de la durée du contact": 
    st.write("Aussi nous observons que la durée de contact médian est plus élevée pour les personnes ayant souscrit un dépôt à terme.")
    fig13=sns.catplot(x='deposit', y='duration', data=bank, kind = 'box')
    plt.title("Dépôt à terme en fonction de la durée du contact")
    st.pyplot(fig13.fig)
  

  if Graphique=='Relation Age, balance et Deposit': 
    st.write("L’âge et la balance influent également sur le deposit. Plus ils sont élevés, plus il y a de dépôt.")
    fig12=px.scatter(bank,x="balance",y="age", color='deposit', title='Relation Age, balance et Deposit')
    st.plotly_chart(fig12) 
  if Graphique=="Répartition par mois en fonction du dépôt à terme":
    st.write("Entre Deposit et Month où l’on remarque des pics de refus ou de contrat accepté selon les mois.")
    st.write('Le mois va avoir un impact, nous remarquons notamment qu’au mois de mai le volume de client contacté est le plus élevé. Le nombre de personnes n’ayant pas souscrit à un dépôt à terme est plus important sur ce mois.')
    fig18=sns.catplot(x='month', kind='count', data=bank, hue = 'deposit')
    plt.title("Répartition par mois en fonction du dépôt à terme");
    st.pyplot(fig18.fig) 
  if Graphique=="Dépôt à terme en fonction de l'age":
    st.write("Cependant l’âge aura certainement un impact un peu moins important étant donné que comme nous pouvons le voir l’âge médian est similaire entre les personnes qui ont souscrit ou non.")
    fig17=sns.catplot(x='deposit', y='age', data=bank, kind = 'box')
    plt.title("Dépôt à terme en fonction de l'age");
    st.pyplot(fig17.fig)

bank_cleaned = bank.drop(bank.loc[bank["job"] == "unknown"].index, inplace=True)
bank_cleaned = bank.drop(bank.loc[bank["education"] == "unknown"].index, inplace=True)
bank_cleaned = bank.drop(['contact', 'pdays'], axis = 1)
  
  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
feats = bank_cleaned.drop(['deposit'], axis = 1)
target = bank_cleaned['deposit']
X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state=42)
scaler = StandardScaler()
cols = ['age','balance','day','campaign','previous','duration']

X_train[cols] = scaler.fit_transform(X_train[cols])
X_test[cols] = scaler.transform(X_test[cols])
  
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

elif page==pages[4]:
  st.write('Modélisation')
  modèle_sélectionné=st.selectbox(label="Modèle", options=['Régression logistique','Decision Tree','Random Forest'])

  if modèle_sélectionné=='Régression logistique':
    st.metric(label="accuracy", value=reglog.score(X_train, y_train))
  
  if modèle_sélectionné=='Decision Tree':
    st.metric(label="accuracy", value= treecl.score(X_train, y_train))

  if modèle_sélectionné=='Random Forest':
    st.metric(label="accuracy", value=forest.score(X_train, y_train))
